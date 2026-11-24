//! LDS-backed geometry stage I/O. One GS invocation receives its input
//! vertices' LDS offsets packed in v0/v1. It writes a private slice of the
//! GS-to-VS ring and sends stream messages. Its copy VS selects components
//! of each vertex from that slice and exports them.

use std::collections::{BTreeMap, HashMap};

use crate::gfx10::{Fields, Operand, Sdwa};
use crate::ir::{Inst, Program, Reg, Value};
use crate::native::{GeometryIo, StageIo};
use crate::{ShaderError, ir};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Component {
    Value(Value),
    Constant(u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VertexOutput {
    /// Native export target (12 for position, 32+ for parameters).
    pub target: u8,
    pub channels: [Option<Component>; 4],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamOp {
    Emit(Vec<VertexOutput>),
    Cut,
}

/// The SSA values supplied by each input load, and the outputs live at
/// each stream instruction. Offsets are relative to the GS machine code.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Streams {
    /// (destination SSA value, input vertex, signature dword).
    pub reads: BTreeMap<usize, Vec<(Value, u8, u16)>>,
    pub messages: BTreeMap<usize, StreamOp>,
    pub(crate) address_steps: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Word {
    Unknown,
    Imm(u32),
    Packed(u8),
    Shifted(u8),
    Input(u8),
    TableLo,
    TableHi,
    MaskedHi,
    RingPage,
    RingBase,
    WaveOffset,
    WaveBase,
    LaneLo,
    Lane,
    Output,
    VertexOffset,
    VertexBytes,
    RingElement(u32),
}

fn fail(inst: &Inst, what: &'static str) -> ShaderError {
    ShaderError::Unsupported {
        offset: inst.machine.offset,
        what,
    }
}

#[derive(Default)]
struct Origins(HashMap<Value, Word>);

impl Origins {
    fn get(&self, i: &Inst, op: &Operand) -> Word {
        let r = match *op {
            Operand::Vector { reg, .. } => Reg::V(reg),
            Operand::Scalar { code, .. } => Reg::S(code),
            Operand::Literal(n) => return Word::Imm(n),
            Operand::Const(n) => {
                return crate::resource::inline(n).map_or(Word::Unknown, Word::Imm);
            }
            _ => return Word::Unknown,
        };
        i.read(r)
            .and_then(|v| self.0.get(&v).copied())
            .unwrap_or(Word::Unknown)
    }

    fn input(&mut self, p: &Program, reg: Reg, word: Word) {
        if let Some((_, v)) = p.inputs.iter().find(|(r, _)| *r == reg) {
            self.0.insert(*v, word);
        }
    }

    fn step(&mut self, i: &Inst, stride: u32) -> Option<Word> {
        use Word::*;
        let (dst, value) = match &i.fields {
            Fields::Sop {
                dst: Some(dst),
                src,
                ..
            } => {
                let a = src.iter().map(|o| self.get(i, o)).collect::<Vec<_>>();
                (
                    dst,
                    match (i.name(), a.as_slice()) {
                        ("s_mov_b32", [x]) => *x,
                        ("s_and_b32", [TableHi, Imm(255)]) => MaskedHi,
                        ("s_lshl_b32", [RingPage, Imm(8)]) => RingBase,
                        ("s_add_i32", [RingBase, WaveOffset] | [WaveOffset, RingBase]) => WaveBase,
                        _ => return None,
                    },
                )
            }
            Fields::Valu(v)
                if v.neg == 0
                    && v.abs == 0
                    && v.opsel == 0
                    && !v.clamp
                    && v.omod == 0
                    && v.dpp.is_none() =>
            {
                let a = v.src.iter().map(|o| self.get(i, o)).collect::<Vec<_>>();
                let high_word = Some(Sdwa {
                    dst_sel: 6,
                    dst_unused: 0,
                    src_sel: [6, 5],
                    sext: 0,
                });
                (
                    v.dst.as_ref()?,
                    match (i.name(), a.as_slice(), v.sdwa) {
                        ("v_mov_b32", [x], None) => *x,
                        ("v_mbcnt_lo_u32_b32", [Imm(u32::MAX), Imm(0)], None) => LaneLo,
                        ("v_mbcnt_hi_u32_b32", [Imm(u32::MAX), LaneLo], None) => Lane,
                        ("v_lshlrev_b32", [Imm(2), Packed(n)], None) => Shifted(*n),
                        ("v_lshlrev_b32", [Imm(2), Packed(n)], s) if s == high_word => Input(n + 1),
                        ("v_and_b32", [Imm(0x3fffc), Shifted(n)], None) => Input(*n),
                        ("v_mad_u32_u24", [Imm(n), Lane, WaveBase], None) if *n == stride => Output,
                        ("v_lshlrev_b32", [Imm(2), VertexOffset], None) => VertexBytes,
                        ("v_lshl_add_u32", [RingPage, Imm(8), VertexBytes], None) => Output,
                        _ => return None,
                    },
                )
            }
            Fields::Smem {
                data: Some(dst),
                base: Operand::Scalar { code, dwords: 2 },
                offset: 0xb8,
                soffset: Operand::Scalar { code: 125, .. },
                ..
            } if i.name() == "s_load_dword"
                && i.read(Reg::S(*code)).and_then(|v| self.0.get(&v)) == Some(&TableLo)
                && i.read(Reg::S(code + 1)).and_then(|v| self.0.get(&v)) == Some(&MaskedHi) =>
            {
                (dst, RingPage)
            }
            _ => return None,
        };
        let reg = match *dst {
            Operand::Vector { reg, .. } => Reg::V(reg),
            Operand::Scalar { code, .. } => Reg::S(code),
            _ => return None,
        };
        self.0.insert(i.write(reg)?, value);
        Some(value)
    }
}

#[derive(Clone, Copy)]
enum CopySource {
    Word(u32),
    Constant(u32),
}

struct CopyExport {
    target: u8,
    channels: [Option<CopySource>; 4],
}

fn copy_exports(gs: &GeometryIo) -> Result<Vec<CopyExport>, ShaderError> {
    let copy = &gs.copy;
    let p = ir::build(&copy.code, copy.wave64, copy.vgprs)?;
    // Each translated EMIT runs the same copy exports for every active invocation.
    // An EXEC write in the copy can suppress those exports for individual lanes.
    if let Some(i) = p
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .find(|i| i.writes.iter().any(|(r, _)| matches!(r, Reg::S(126 | 127))))
    {
        return Err(fail(i, "a geometry copy execution mask"));
    }
    // Stream-zero copies either execute straight through or branch directly
    // to endpgm when the stream number in s3[31:24] is nonzero.
    let body = match p.blocks.as_slice() {
        [b] if b.exit == ir::Exit::End => &b.insts[..],
        [entry, body, end]
            if entry.exit
                == (ir::Exit::Branch {
                    taken: 2,
                    not_taken: 1,
                })
                && body.exit == ir::Exit::Jump(2)
                && end.insts.len() == 1
                && end.insts[0].name() == "s_endpgm" =>
        {
            let mut ops = entry
                .insts
                .iter()
                .filter(|i| !matches!(i.name(), "v_nop" | "s_inst_prefetch" | "s_setprio"));
            let predicate = (ops.next(), ops.next(), ops.next(), ops.next());
            let (Some(shift), Some(cmp), Some(branch), None) = predicate else {
                return Err(fail(&entry.insts[0], "the geometry copy stream predicate"));
            };
            let r = (shift.machine.words[0] >> 16) & 127;
            if shift.machine.words != [0x90009803 | (r << 16)]
                || cmp.machine.words != [0xbf078000 | r]
                || branch.name() != "s_cbranch_scc1"
            {
                return Err(fail(branch, "the geometry copy stream predicate"));
            }
            &body.insts[..]
        }
        _ => {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "geometry copy control flow",
            });
        }
    };
    let mut origins = Origins::default();
    origins.input(&p, Reg::S(0), Word::TableLo);
    origins.input(&p, Reg::S(1), Word::TableHi);
    origins.input(&p, Reg::V(0), Word::VertexOffset);
    let mut exports = Vec::new();
    for i in body {
        let origin = origins.step(i, 0);
        match &i.fields {
            Fields::Ds {
                addr: Some(a),
                dst: Some(Operand::Vector { reg, .. }),
                offset0,
                offset1,
                gds: false,
                ..
            } if i.name() == "ds_read2_b32" && origins.get(i, a) == Word::Output => {
                for (k, off) in [*offset0, *offset1].into_iter().enumerate() {
                    let v = i
                        .write(Reg::V(reg + k as u8))
                        .ok_or(fail(i, "a geometry copy load destination"))?;
                    if u32::from(off) >= gs.vertex_dwords {
                        return Err(fail(i, "a geometry copy read outside the vertex"));
                    }
                    origins.0.insert(v, Word::RingElement(u32::from(off)));
                }
            }
            Fields::Exp {
                target,
                src,
                compr: false,
                ..
            } => {
                let mut channels = [None; 4];
                for (ch, src) in src.iter().enumerate() {
                    if let Some(src) = src {
                        channels[ch] = Some(match origins.get(i, src) {
                            Word::RingElement(k) => CopySource::Word(k),
                            Word::Imm(n) => CopySource::Constant(n),
                            _ => {
                                return Err(fail(
                                    i,
                                    "a geometry copy export with an untraced source",
                                ));
                            }
                        });
                    }
                }
                exports.push(CopyExport {
                    target: *target,
                    channels,
                });
            }
            Fields::Exp { .. } => return Err(fail(i, "a compressed geometry copy export")),
            _ if origin.is_some_and(|x| x != Word::Unknown)
                || matches!(
                    i.name(),
                    "s_endpgm" | "s_waitcnt" | "s_nop" | "v_nop" | "s_inst_prefetch"
                ) => {}
            _ => return Err(fail(i, "an unmodelled geometry copy instruction")),
        }
    }
    if exports.is_empty() {
        return Err(ShaderError::Unsupported {
            offset: 0,
            what: "an empty geometry copy program",
        });
    }
    Ok(exports)
}

pub(crate) fn raise(p: &Program, gs: &GeometryIo) -> Result<Streams, ShaderError> {
    let unsupported = |what| ShaderError::Unsupported { offset: 0, what };
    let stride_bytes = gs
        .invocation_dwords
        .checked_mul(4)
        .ok_or(unsupported("the geometry ring size"))?;
    if gs.stream_config != [1, 0]
        || gs.vertex_dwords == 0
        || gs.vertex_dwords.checked_mul(gs.max_vertices) != Some(gs.invocation_dwords)
        || !matches!(gs.copy.io, StageIo::Vs { .. })
    {
        return Err(unsupported("the geometry output ring layout"));
    }
    if p.blocks.len() != 1
        || p.blocks[0].exit != ir::Exit::End
        || p.blocks[0]
            .insts
            .iter()
            .any(|i| i.writes.iter().any(|(r, _)| matches!(r, Reg::S(126 | 127))))
    {
        return Err(unsupported("geometry stream I/O across control flow"));
    }
    let exports = copy_exports(gs)?;
    let mut origins = Origins::default();
    for (reg, word) in [
        (Reg::V(0), Word::Packed(0)),
        (Reg::V(1), Word::Packed(2)),
        (Reg::S(8), Word::TableLo),
        (Reg::S(9), Word::TableHi),
        (Reg::S(2), Word::WaveOffset),
    ] {
        origins.input(p, reg, word);
    }
    let mut streams = Streams::default();
    let mut ring = HashMap::new();
    let mut vertex: u32 = 0;
    for i in &p.blocks[0].insts {
        if matches!(i.fields, Fields::Exp { .. })
            || i.name().starts_with("s_sendmsg") && i.name() != "s_sendmsg"
        {
            return Err(fail(i, "an additional geometry export or message"));
        }
        if origins
            .step(i, stride_bytes)
            .is_some_and(|w| !matches!(w, Word::Imm(_) | Word::Unknown))
        {
            streams.address_steps.push(i.machine.offset);
        }
        match &i.fields {
            Fields::Ds {
                addr: Some(a),
                dst: Some(Operand::Vector { reg, .. }),
                offset0,
                offset1,
                gds: false,
                ..
            } => {
                let Word::Input(index) = origins.get(i, a) else {
                    return Err(fail(i, "a geometry input vertex address"));
                };
                let mut reads = Vec::new();
                let mut read = |k, dword| {
                    let v = i
                        .write(Reg::V(reg + k))
                        .ok_or(fail(i, "a geometry input destination"))?;
                    reads.push((v, index, dword));
                    Ok(())
                };
                match i.name() {
                    "ds_read_b32" => {
                        let bytes = u16::from_le_bytes([*offset0, *offset1]);
                        if bytes % 4 != 0 {
                            return Err(fail(i, "an unaligned geometry input"));
                        }
                        read(0, bytes / 4)?;
                    }
                    "ds_read2_b32" => {
                        read(0, u16::from(*offset0))?;
                        read(1, u16::from(*offset1))?;
                    }
                    _ => return Err(fail(i, "a geometry input load")),
                }
                streams.reads.insert(i.machine.offset, reads);
            }
            Fields::Ds {
                addr: Some(a),
                data,
                offset0,
                offset1,
                gds: false,
                ..
            } if i.name() == "ds_write2_b32" && origins.get(i, a) == Word::Output => {
                for (src, off) in data.iter().zip([*offset0, *offset1]) {
                    let Operand::Vector { reg, .. } = src else {
                        return Err(fail(i, "a geometry ring store operand"));
                    };
                    let v = i
                        .read(Reg::V(*reg))
                        .ok_or(fail(i, "a geometry ring store source"))?;
                    if u32::from(off) >= gs.invocation_dwords {
                        return Err(fail(i, "a geometry ring store outside the invocation"));
                    }
                    // The console's copy VS reads the ring after the GS ends;
                    // the translation takes each slot's value at its EMIT. A
                    // slot written after its vertex was emitted would differ.
                    if u32::from(off) < vertex * gs.vertex_dwords {
                        return Err(fail(i, "a geometry ring store to a vertex already emitted"));
                    }
                    ring.insert(u32::from(off), v);
                }
            }
            Fields::Ds { .. } => return Err(fail(i, "a geometry LDS operation")),
            Fields::Sopp { imm } if i.name() == "s_sendmsg" => {
                let op = match *imm {
                    0x22 => {
                        if vertex >= gs.max_vertices {
                            return Err(fail(i, "too many geometry output vertices"));
                        }
                        let mut output = Vec::new();
                        for e in &exports {
                            let mut channels = [None; 4];
                            for (ch, src) in e.channels.iter().enumerate() {
                                channels[ch] = match src {
                                    Some(CopySource::Word(off)) => {
                                        let v = *ring
                                            .get(&(vertex * gs.vertex_dwords + off))
                                            .ok_or(fail(
                                                i,
                                                "a geometry output the invocation did not write",
                                            ))?;
                                        Some(Component::Value(v))
                                    }
                                    Some(CopySource::Constant(n)) => Some(Component::Constant(*n)),
                                    None => None,
                                };
                            }
                            output.push(VertexOutput {
                                target: e.target,
                                channels,
                            });
                        }
                        vertex += 1;
                        StreamOp::Emit(output)
                    }
                    0x12 => StreamOp::Cut,
                    3 => continue,
                    _ => return Err(fail(i, "a geometry stream message")),
                };
                streams.messages.insert(i.machine.offset, op);
            }
            _ => {}
        }
    }
    Ok(streams)
}

pub(crate) fn invocation_ir(native: &crate::NativeProgram) -> Result<Program, ShaderError> {
    let mut code = native.code.clone();
    let list = crate::gfx10::decode(&code)?;
    // The GS body runs on the low GSVertCount lanes. The host invokes
    // the shader only for an active primitive. Preserve the count and
    // SCC calculations, removing only the two writes that establish EXEC.
    let mut established = false;
    for (n, i) in list.iter().enumerate() {
        if i.name() != Some("s_bfm_b64") {
            continue;
        }
        let Some(prefix) = n.checked_sub(1).and_then(|k| list.get(k..k + 4)) else {
            continue;
        };
        // s_bfe_i32 r, s3, (8:8); s_bfm_b64 exec, r, 0;
        // s_cmp_eq_u32 r, 64; s_cselect_b64 exec, -1, exec.
        let r = (prefix[0].words[0] >> 16) & 127;
        if established
            || list[..n - 1]
                .iter()
                .any(|op| !matches!(op.name(), Some("v_nop" | "s_nop" | "s_inst_prefetch")))
            || !native.wave64
            || prefix[0].words != [0x9400ff03 | r << 16, 0x80008]
            || i.words != [0x92fe8000 | r]
            || prefix[2].words != [0xbf06c000 | r]
            || prefix[3].words != [0x85fe7ec1]
        {
            continue;
        }
        for inst in [i, &prefix[3]] {
            code[inst.offset..inst.offset + 4].copy_from_slice(&0xbf800000u32.to_le_bytes());
        }
        established = true;
    }
    let p = ir::build(&code, native.wave64, native.vgprs)?;
    if !established
        || p.blocks.len() != 1
        || p.blocks[0]
            .insts
            .iter()
            .any(|i| i.writes.iter().any(|(r, _)| matches!(r, Reg::S(126 | 127))))
    {
        return Err(ShaderError::Unsupported {
            offset: 0,
            what: "a geometry invocation mask",
        });
    }
    Ok(p)
}
