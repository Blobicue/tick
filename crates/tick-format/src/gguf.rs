//! GGUF v3 container: metadata, tensor table, and read-at tensor access.
//!
//! Tensors are addressed by NAME (files store them in arbitrary — the
//! LoRA adapter proves lexicographic — order), weights are read on
//! demand via `read_at` so the kernel page cache owns residency.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::Path;

pub const T_F32: u32 = 0;
pub const T_F16: u32 = 1;
pub const T_Q8_0: u32 = 8;
pub const T_MXFP4: u32 = 39;
pub const T_TQ1_0: u32 = 34; // 256-elem blocks, 54 bytes: qs[48]+qh[4]+f16 d

/// (elements per block, bytes per block); None = layout not needed yet.
fn block_layout(ty: u32) -> Option<(u64, u64)> {
    match ty {
        T_F32 => Some((1, 4)),
        T_F16 => Some((1, 2)),
        T_Q8_0 => Some((32, 34)),
        T_MXFP4 => Some((32, 17)),
        T_TQ1_0 => Some((256, 54)),
        _ => None,
    }
}

pub fn type_name(ty: u32) -> String {
    match ty {
        T_F32 => "F32".into(),
        T_F16 => "F16".into(),
        T_Q8_0 => "Q8_0".into(),
        T_MXFP4 => "MXFP4".into(),
        T_TQ1_0 => "TQ1_0".into(),
        _ => format!("type{ty}"),
    }
}

/// A metadata value; arrays flatten to one variant.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Arr(Vec<Value>),
}

impl Value {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U8(v) => Some(*v as u64),
            Value::I8(v) => Some(*v as u64),
            Value::U16(v) => Some(*v as u64),
            Value::I16(v) => Some(*v as u64),
            Value::U32(v) => Some(*v as u64),
            Value::I32(v) => Some(*v as u64),
            Value::U64(v) => Some(*v),
            Value::I64(v) => Some(u64::try_from(*v).ok()?),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Value::F32(v) => Some(*v),
            Value::F64(v) => Some(*v as f32),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    /// ggml `ne` order: dims\[0\] is the contiguous dimension.
    pub dims: Vec<u64>,
    pub ty: u32,
    /// byte offset relative to `Gguf::data_start`.
    pub offset: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
    pub fn n_bytes(&self) -> Result<u64, String> {
        let (block, bytes) = block_layout(self.ty).ok_or_else(|| {
            format!(
                "unsupported tensor type {} ({})",
                self.ty,
                type_name(self.ty)
            )
        })?;
        let blocks_along_dim0 = self.dims.first().copied().unwrap_or(0).div_ceil(block);
        Ok(blocks_along_dim0 * self.dims.iter().skip(1).product::<u64>() * bytes)
    }
}

pub struct Gguf {
    pub version: u32,
    /// sorted by key so serialized facts are deterministic
    pub meta: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    pub by_name: HashMap<String, usize>,
    pub alignment: u64,
    pub data_start: u64,
    /// data_start + max(offset + n_bytes) over all tensors; <= file_len
    pub data_end: u64,
    pub file_len: u64,
}

/// Growable header buffer: reads more of the file as the parser advances.
struct Hdr {
    file: File,
    buf: Vec<u8>,
    pos: usize,
}

impl Hdr {
    const MAX_HEADER: usize = 64 << 20;

    fn new(mut file: File) -> Result<Self, String> {
        let mut buf = vec![0u8; 1 << 20];
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("read failed: {e}"))?;
        buf.truncate(n);
        Ok(Hdr { file, buf, pos: 0 })
    }

    fn ensure(&mut self, n: usize) -> Result<(), String> {
        while self.buf.len() - self.pos < n {
            if self.buf.len() >= Self::MAX_HEADER {
                return Err("gguf header exceeds 64MB - refusing".into());
            }
            let mut chunk = vec![0u8; 1 << 20];
            let got = self
                .file
                .read(&mut chunk)
                .map_err(|e| format!("read failed: {e}"))?;
            if got == 0 {
                return Err(format!("gguf header truncated at byte {}", self.buf.len()));
            }
            chunk.truncate(got);
            self.buf.extend_from_slice(&chunk);
        }
        Ok(())
    }

    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        self.ensure(n)?;
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32, String> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64, String> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
    fn str(&mut self) -> Result<String, String> {
        let n = self.u64()? as usize;
        if n > (8 << 20) {
            return Err(format!("string field of {n} bytes - refusing"));
        }
        let b = self.take(n)?;
        String::from_utf8(b.to_vec()).map_err(|e| format!("string not utf8: {e}"))
    }
}

fn read_value(h: &mut Hdr, ty: u32) -> Result<Value, String> {
    Ok(match ty {
        0 => Value::U8(h.take(1)?[0]),
        1 => Value::I8(h.take(1)?[0] as i8),
        2 => {
            let b = h.take(2)?;
            Value::U16(u16::from_le_bytes([b[0], b[1]]))
        }
        3 => {
            let b = h.take(2)?;
            Value::I16(i16::from_le_bytes([b[0], b[1]]))
        }
        4 => Value::U32(h.u32()?),
        5 => Value::I32(h.u32()? as i32),
        6 => {
            let b = h.take(4)?;
            Value::F32(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        }
        7 => Value::Bool(h.take(1)?[0] != 0),
        8 => Value::Str(h.str()?),
        10 => Value::U64(h.u64()?),
        11 => Value::I64(h.u64()? as i64),
        12 => {
            let b = h.take(8)?;
            Value::F64(f64::from_le_bytes([
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
            ]))
        }
        9 => {
            let et = h.u32()?;
            if et == 9 {
                return Err("nested arrays are not representable in gguf".into());
            }
            let n = h.u64()? as usize;
            if n > 5_000_000 {
                return Err(format!("array of {n} elements - refusing"));
            }
            // a hostile count must not pre-allocate before the truncated
            // header error fires; grow incrementally instead
            let mut out = Vec::with_capacity(n.min(65_536));
            for _ in 0..n {
                out.push(read_value(h, et)?);
            }
            Value::Arr(out)
        }
        other => return Err(format!("unknown gguf metadata type {other}")),
    })
}

impl Gguf {
    /// Parse the header of a gguf file. The returned handle keeps no
    /// file open; open the path again and pass `&mut File` to the
    /// read_raw helpers for tensor access.
    pub fn open(path: &Path) -> Result<Gguf, String> {
        let name = path.display().to_string();
        let file = File::open(path).map_err(|e| format!("{name}: {e}"))?;
        let mut h = Hdr::new(file)?;
        let magic = h.take(4)?;
        if magic != b"GGUF" {
            return Err(format!("{name}: not a gguf file (magic {magic:?})"));
        }
        let version = h.u32()?;
        if !(2..=3).contains(&version) {
            return Err(format!(
                "{name}: gguf version {version} unsupported (want 2..=3)"
            ));
        }
        let n_tensors = h.u64()? as usize;
        let n_kv = h.u64()? as usize;
        if n_tensors > 100_000 || n_kv > 10_000 {
            return Err(format!(
                "{name}: implausible counts ({n_tensors} tensors, {n_kv} kvs)"
            ));
        }
        let mut meta = BTreeMap::new();
        for _ in 0..n_kv {
            let key = h.str()?;
            let ty = h.u32()?;
            let val = read_value(&mut h, ty)?;
            meta.insert(key, val);
        }
        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let tname = h.str()?;
            let nd = h.u32()? as usize;
            if nd == 0 || nd > 4 {
                return Err(format!("{name}: tensor {tname} has {nd} dims"));
            }
            let mut dims = Vec::with_capacity(nd);
            for _ in 0..nd {
                dims.push(h.u64()?);
            }
            let ty = h.u32()?;
            let offset = h.u64()?;
            tensors.push(TensorInfo {
                name: tname,
                dims,
                ty,
                offset,
            });
        }
        let header_end = h.pos as u64;
        let alignment = meta
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32);
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(format!("{name}: bad alignment {alignment}"));
        }
        let data_start = header_end.div_ceil(alignment) * alignment;
        let mut by_name = HashMap::with_capacity(tensors.len());
        for (i, t) in tensors.iter().enumerate() {
            if by_name.insert(t.name.clone(), i).is_some() {
                return Err(format!("{name}: duplicate tensor name {}", t.name));
            }
        }
        // layout guard: every declared tensor must fit the real file
        let file_len = path
            .metadata()
            .map_err(|e| format!("{name}: stat failed: {e}"))?
            .len();
        let mut data_end = data_start;
        for t in &tensors {
            let end = t
                .offset
                .checked_add(t.n_bytes()?)
                .and_then(|e| data_start.checked_add(e))
                .ok_or_else(|| format!("{name}: tensor {} overflows u64", t.name))?;
            if end > file_len {
                return Err(format!(
                    "{name}: tensor {} claims bytes through {end} but file is {file_len}",
                    t.name
                ));
            }
            data_end = data_end.max(end);
        }
        Ok(Gguf {
            version,
            meta,
            tensors,
            by_name,
            alignment,
            data_start,
            data_end,
            file_len,
        })
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo, String> {
        let i = *self
            .by_name
            .get(name)
            .ok_or_else(|| format!("tensor {name} not present"))?;
        Ok(&self.tensors[i])
    }

    pub fn meta_str_vec(&self, key: &str) -> Option<Vec<String>> {
        match self.meta.get(key) {
            Some(Value::Arr(items)) => Some(
                items
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect(),
            ),
            _ => None,
        }
    }

    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.meta.get(key).and_then(|v| v.as_str())
    }
    pub fn meta_u64(&self, key: &str) -> Option<u64> {
        self.meta.get(key).and_then(|v| v.as_u64())
    }
    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        self.meta.get(key).and_then(|v| v.as_f32())
    }

    /// Raw bytes of one tensor. Sizes derive from type+dims and are
    /// bounds-checked against the file length the caller observed.
    pub fn read_raw(&self, name: &str, file: &mut File, file_len: u64) -> Result<Vec<u8>, String> {
        let t = self.tensor(name)?;
        let n = t.n_bytes()? as usize;
        let start = self.data_start + t.offset;
        if start + n as u64 > file_len {
            return Err(format!(
                "tensor {name} claims bytes {start}..{} but file is {file_len}",
                start + n as u64
            ));
        }
        let mut buf = vec![0u8; n];
        file.read_exact_at(&mut buf, start)
            .map_err(|e| format!("tensor {name} read failed: {e}"))?;
        Ok(buf)
    }

    /// First `max_bytes` of a tensor (tests sample block prefixes; full
    /// expert tensors run to hundreds of MB).
    pub fn read_raw_max(
        &self,
        name: &str,
        file: &mut File,
        file_len: u64,
        max_bytes: u64,
    ) -> Result<Vec<u8>, String> {
        let t = self.tensor(name)?;
        let n = t.n_bytes()?.min(max_bytes) as usize;
        let start = self.data_start + t.offset;
        if start + n as u64 > file_len {
            return Err(format!(
                "tensor {name} claims bytes {start}..{} but file is {file_len}",
                start + n as u64
            ));
        }
        let mut buf = vec![0u8; n];
        file.read_exact_at(&mut buf, start)
            .map_err(|e| format!("tensor {name} read failed: {e}"))?;
        Ok(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_layouts_match_file_math() {
        // 3D expert mat [2880,2880,32] MXFP4: 90 blocks along dim0
        let t = TensorInfo {
            name: "x".into(),
            dims: vec![2880, 2880, 32],
            ty: T_MXFP4,
            offset: 0,
        };
        assert_eq!(t.n_bytes().unwrap(), 90 * 2880 * 32 * 17);
        let q = TensorInfo {
            name: "q".into(),
            dims: vec![2880, 4096],
            ty: T_Q8_0,
            offset: 0,
        };
        assert_eq!(q.n_bytes().unwrap(), 90 * 4096 * 34);
    }
}
