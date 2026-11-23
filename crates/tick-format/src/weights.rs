//! tickfile v0 — the engine's native weight format (SPEC_WEIGHTS.md).
//!
//! Writer (TickWriter) + reader (TickFile). Conversion from tq1_0 GGUF
//! copies tensor bytes verbatim (ternary blocks are layout-identical),
//! so a converted model decodes bit-identically to the GGUF load.

use crate::gguf::{Gguf, Value};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Seek, Write};
use std::os::unix::io::AsRawFd;
use std::path::Path;

pub const MAGIC: &[u8; 4] = b"TICK";
/// v2: layer-streamed layout — tensors written in load order
/// (token_embd, blk.0..blk.N with each layer's tensors contiguous and
/// 4096-byte aligned, then the output norms), so a layer window can be
/// mmap'd/decoded independently. Readers accept 0 and 2.
pub const VERSION: u32 = 2;

/// The layer-streamed write order: token_embd first, blk.N in numeric
/// order (a layer's tensors contiguous), output.* last. Tensors whose
/// names don't match keep their incoming relative order at the end.
fn layer_sort_key(name: &str) -> (u32, String) {
    if name == "token_embd.weight" {
        return (0, String::new());
    }
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some((n, _)) = rest.split_once('.') {
            if let Ok(il) = n.parse::<u32>() {
                return (1 + il, name.to_string());
            }
        }
    }
    (u32::MAX, name.to_string())
}

/// 4096-byte data alignment for the v2 layer windows.
pub const ALIGN: u64 = 4096;

pub const DT_F32: u32 = 0;
pub const DT_F16: u32 = 1;
pub const DT_TERNARY: u32 = 2;

fn w_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn w_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn w_str(buf: &mut Vec<u8>, s: &str) {
    w_u32(buf, s.len() as u32);
    buf.extend_from_slice(s.as_bytes());
}

fn kv_value_bytes(buf: &mut Vec<u8>, v: &Value) {
    match v {
        Value::Str(s) => {
            buf.push(0);
            w_str(buf, s);
        }
        Value::Bool(b) => {
            buf.push(3);
            buf.push(*b as u8);
        }
        Value::U8(x) => {
            buf.push(1);
            w_u64(buf, *x as u64);
        }
        Value::I8(x) => {
            buf.push(1);
            w_u64(buf, *x as i64 as u64);
        }
        Value::U16(x) => {
            buf.push(1);
            w_u64(buf, *x as u64);
        }
        Value::I16(x) => {
            buf.push(1);
            w_u64(buf, *x as i64 as u64);
        }
        Value::U32(x) => {
            buf.push(1);
            w_u64(buf, *x as u64);
        }
        Value::I32(x) => {
            buf.push(1);
            w_u64(buf, *x as i64 as u64);
        }
        Value::U64(x) => {
            buf.push(1);
            w_u64(buf, *x);
        }
        Value::I64(x) => {
            buf.push(1);
            w_u64(buf, *x as u64);
        }
        Value::F32(x) => {
            // store the VALUE widened to f64 (lossless), not the f32
            // bit pattern — the reader does f64::from_bits
            buf.push(2);
            w_u64(buf, (*x as f64).to_bits());
        }
        Value::F64(x) => {
            buf.push(2);
            w_u64(buf, x.to_bits());
        }
        Value::Arr(items) => {
            if items.iter().all(|i| matches!(i, Value::Str(_))) {
                buf.push(4);
                w_u32(buf, items.len() as u32);
                for i in items {
                    if let Value::Str(s) = i {
                        w_str(buf, s);
                    }
                }
            } else {
                buf.push(5);
                w_u32(buf, items.len() as u32);
                for i in items {
                    let x = match i {
                        Value::U8(v) => *v as u64,
                        Value::I8(v) => *v as i64 as u64,
                        Value::U16(v) => *v as u64,
                        Value::I16(v) => *v as i64 as u64,
                        Value::U32(v) => *v as u64,
                        Value::I32(v) => *v as i64 as u64,
                        Value::U64(v) => *v,
                        Value::I64(v) => *v as u64,
                        Value::F32(v) => v.to_bits() as u64,
                        Value::F64(v) => v.to_bits(),
                        _ => 0,
                    };
                    w_u64(buf, x);
                }
            }
        }
    }
}

struct PendingTensor {
    name: String,
    dtype: u32,
    dims: Vec<u64>,
    bytes: Vec<u8>,
}

/// Incremental tickfile builder. KV and tensors buffer in memory (the
/// largest model fits comfortably); `finish` writes the single file.
pub struct TickWriter {
    kv: Vec<(String, Value)>,
    tensors: Vec<PendingTensor>,
}

impl TickWriter {
    pub fn new() -> TickWriter {
        TickWriter {
            kv: Vec::new(),
            tensors: Vec::new(),
        }
    }

    pub fn add_kv(&mut self, key: &str, value: Value) {
        self.kv.push((key.to_string(), value));
    }

    pub fn add_tensor(&mut self, name: &str, dtype: u32, dims: &[u64], bytes: Vec<u8>) {
        self.tensors.push(PendingTensor {
            name: name.to_string(),
            dtype,
            dims: dims.to_vec(),
            bytes,
        });
    }

    pub fn finish(&self, path: &Path) -> Result<(), String> {
        self.finish_ordered(path, true)
    }

    /// Write the tickfile. `layer_order` sorts tensors into the v2
    /// layer-streamed order and aligns each data block to 4096 bytes
    /// (a layer's tensors then occupy one padded window that can be
    /// mmap'd and decoded independently); false preserves insertion
    /// order with no padding (v0 layout, version field still 2 — the
    /// reader never requires the ordering).
    pub fn finish_ordered(&self, path: &Path, layer_order: bool) -> Result<(), String> {
        let mut tensors: Vec<&PendingTensor> = self.tensors.iter().collect();
        if layer_order {
            tensors.sort_by_key(|t| layer_sort_key(&t.name));
        }
        let mut kv_buf = Vec::new();
        w_u32(&mut kv_buf, self.kv.len() as u32);
        for (k, v) in &self.kv {
            w_str(&mut kv_buf, k);
            kv_value_bytes(&mut kv_buf, v);
        }
        let header_len = 40u64;
        let dir_offset = header_len + kv_buf.len() as u64;
        let data_offset = dir_offset + dir_placeholder_len(&self.tensors) as u64;
        let mut f = File::create(path).map_err(|e| e.to_string())?;
        let mut head = Vec::new();
        head.extend_from_slice(MAGIC);
        w_u32(&mut head, VERSION);
        w_u64(&mut head, header_len);
        w_u64(&mut head, dir_offset);
        w_u64(&mut head, data_offset);
        head.extend_from_slice(&[0u8; 8]); // file_len patched below
        f.write_all(&head).map_err(|e| e.to_string())?;
        f.write_all(&kv_buf).map_err(|e| e.to_string())?;
        // stream the data section (with v2 alignment padding), then
        // write the directory once the offsets are final
        f.seek(std::io::SeekFrom::Start(data_offset))
            .map_err(|e| e.to_string())?;
        let mut off = data_offset;
        let mut dir_entries: Vec<(String, u32, Vec<u64>, u64, u64)> = Vec::new();
        for t in &tensors {
            if layer_order {
                let pad = (ALIGN - (off % ALIGN)) % ALIGN;
                if pad > 0 {
                    f.write_all(&vec![0u8; pad as usize])
                        .map_err(|e| e.to_string())?;
                    off += pad;
                }
            }
            dir_entries.push((
                t.name.clone(),
                t.dtype,
                t.dims.clone(),
                off,
                t.bytes.len() as u64,
            ));
            f.write_all(&t.bytes).map_err(|e| e.to_string())?;
            off += t.bytes.len() as u64;
        }
        let mut dir_buf = Vec::new();
        w_u32(&mut dir_buf, dir_entries.len() as u32);
        for (name, dtype, dims, eoff, eln) in &dir_entries {
            w_str(&mut dir_buf, name);
            w_u32(&mut dir_buf, *dtype);
            w_u32(&mut dir_buf, dims.len() as u32);
            for d in dims {
                w_u64(&mut dir_buf, *d);
            }
            w_u64(&mut dir_buf, *eoff);
            w_u64(&mut dir_buf, *eln);
        }
        f.seek(std::io::SeekFrom::Start(dir_offset))
            .map_err(|e| e.to_string())?;
        f.write_all(&dir_buf).map_err(|e| e.to_string())?;
        use std::os::unix::fs::FileExt;
        f.write_all_at(&off.to_le_bytes(), 32)
            .map_err(|e| e.to_string())?;
        Ok(())
    }
}

impl Default for TickWriter {
    fn default() -> Self {
        Self::new()
    }
}

fn dir_placeholder_len(tensors: &[PendingTensor]) -> usize {
    let mut n = 4usize;
    for t in tensors {
        n += 4 + t.name.len() + 4 + 4 + 8 * t.dims.len() + 8 + 8;
    }
    n
}

/// A parsed tickfile: KV metadata + tensor directory, data read on
/// demand (same access shape as the GGUF source).
pub struct TickFile {
    pub meta: BTreeMap<String, Value>,
    /// the whole file, memory-mapped (v0.2's alignment makes layer
    /// windows mmap-friendly; read_raw is now zero-copy up to the
    /// decode step)
    pub map: Mmap,
    pub dir: BTreeMap<String, (u32, Vec<u64>, u64, u64)>, // dtype, dims, offset, len
}

/// A minimal read-only mmap (no libc dep beyond what's already in —
/// actually uses mmap(2) via libc, which the crate already depends on).
pub struct Mmap {
    ptr: *mut u8,
    len: usize,
}

impl Mmap {
    fn open(file: &File, len: usize) -> Result<Mmap, String> {
        if len == 0 {
            return Ok(Mmap {
                ptr: std::ptr::null_mut(),
                len: 0,
            });
        }
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err("mmap failed".into());
        }
        Ok(Mmap {
            ptr: ptr as *mut u8,
            len,
        })
    }
    fn slice(&self) -> &[u8] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        if self.len > 0 {
            unsafe {
                libc::munmap(self.ptr as *mut _, self.len);
            }
        }
    }
}

// SAFETY: the mapping is read-only and outlives any borrow of it
unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Rd<'a> {
    fn u32(&mut self) -> Result<u32, String> {
        if self.p + 4 > self.b.len() {
            return Err("tickfile truncated".into());
        }
        let v = u32::from_le_bytes(self.b[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        Ok(v)
    }
    fn u64(&mut self) -> Result<u64, String> {
        if self.p + 8 > self.b.len() {
            return Err("tickfile truncated".into());
        }
        let v = u64::from_le_bytes(self.b[self.p..self.p + 8].try_into().unwrap());
        self.p += 8;
        Ok(v)
    }
    fn bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.p + n > self.b.len() {
            return Err("tickfile truncated".into());
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn str(&mut self) -> Result<String, String> {
        let n = self.u32()? as usize;
        let s = self.bytes(n)?;
        String::from_utf8(s.to_vec()).map_err(|_| "tickfile: bad utf8".into())
    }
}

fn parse_kv_value(r: &mut Rd) -> Result<Value, String> {
    let ty = *r.bytes(1)?.first().ok_or("empty")?;
    Ok(match ty {
        0 => Value::Str(r.str()?),
        1 => {
            let x = r.u64()?;
            if x <= i64::MAX as u64 {
                Value::U64(x)
            } else {
                Value::I64(x as i64)
            }
        }
        2 => Value::F64(f64::from_bits(r.u64()?)),
        3 => Value::Bool(*r.bytes(1)?.first().ok_or("empty")? != 0),
        4 => {
            let n = r.u32()? as usize;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(Value::Str(r.str()?));
            }
            Value::Arr(v)
        }
        5 => {
            let n = r.u32()? as usize;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(Value::U64(r.u64()?));
            }
            Value::Arr(v)
        }
        other => return Err(format!("tickfile: bad kv type {other}")),
    })
}

impl TickFile {
    pub fn open(path: &Path) -> Result<TickFile, String> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let len = path.metadata().map_err(|e| e.to_string())?.len();
        let map = Mmap::open(&file, len as usize)?;
        let buf = map.slice();
        let head = &buf[0..40];
        if &head[0..4] != MAGIC {
            return Err("not a tickfile".into());
        }
        let kv_off = u64::from_le_bytes(head[8..16].try_into().unwrap());
        let dir_off = u64::from_le_bytes(head[16..24].try_into().unwrap());
        // kv section: read up to dir (bounded)
        let kv_raw = &buf[kv_off as usize..dir_off as usize];
        let mut r = Rd { b: kv_raw, p: 0 };
        let nkv = r.u32()?;
        let mut meta = BTreeMap::new();
        for _ in 0..nkv {
            let k = r.str()?;
            let v = parse_kv_value(&mut r)?;
            meta.insert(k, v);
        }
        // dir: read to data_offset (from header)
        let data_off = u64::from_le_bytes(head[24..32].try_into().unwrap());
        let dir_raw = &buf[dir_off as usize..data_off as usize];
        let mut r = Rd { b: dir_raw, p: 0 };
        let nt = r.u32()?;
        let mut dir = BTreeMap::new();
        for _ in 0..nt {
            let name = r.str()?;
            let dtype = r.u32()?;
            let nd = r.u32()? as usize;
            let mut dims = Vec::with_capacity(nd);
            for _ in 0..nd {
                dims.push(r.u64()?);
            }
            let off = r.u64()?;
            let ln = r.u64()?;
            if off + ln > len {
                return Err(format!("tickfile: tensor {name} out of range"));
            }
            dir.insert(name, (dtype, dims, off, ln));
        }
        Ok(TickFile { meta, map, dir })
    }

    pub fn read_raw(&self, name: &str) -> Result<Vec<u8>, String> {
        let (_, _, off, ln) = self
            .dir
            .get(name)
            .ok_or_else(|| format!("tickfile: no tensor {name}"))?;
        Ok(self.map.slice()[*off as usize..(*off + *ln) as usize].to_vec())
    }
}

impl TickFile {
    pub fn meta_u64(&self, key: &str) -> Option<u64> {
        self.meta.get(key).and_then(|v| v.as_u64())
    }
    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        self.meta.get(key).and_then(|v| v.as_f32())
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
}

/// Convert a tq1_0 GGUF to a tickfile. Tensor bytes copy verbatim —
/// ternary blocks are layout-identical (54-byte blocks, (row, block)
/// row-major); dtype 34 -> ternary, 1 -> f16, 0 -> f32.
pub fn convert_gguf(gguf_path: &Path, out_path: &Path) -> Result<u64, String> {
    let g = Gguf::open(gguf_path)?;
    let mut file = File::open(gguf_path).map_err(|e| e.to_string())?;
    let file_len = gguf_path.metadata().map_err(|e| e.to_string())?.len();
    let mut w = TickWriter::new();
    for (k, v) in &g.meta {
        w.add_kv(k, v.clone());
    }
    w.add_kv(
        "tick.source_path",
        Value::Str(gguf_path.display().to_string()),
    );
    w.add_kv("tick.source_bytes", Value::U64(file_len));
    w.add_kv(
        "tick.converted_at",
        Value::F64(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0),
        ),
    );
    let mut total = 0u64;
    for t in &g.tensors {
        let dtype = match t.ty {
            0 => DT_F32,
            1 => DT_F16,
            34 => DT_TERNARY,
            other => {
                return Err(format!(
                    "{}: gguf type {other} has no tickfile dtype",
                    t.name
                ))
            }
        };
        let bytes = g.read_raw(&t.name, &mut file, file_len)?;
        total += bytes.len() as u64;
        let dims: Vec<u64> = t.dims.to_vec();
        w.add_tensor(&t.name, dtype, &dims, bytes);
    }
    w.finish(out_path)?;
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writer_reader_roundtrip() {
        let mut w = TickWriter::new();
        w.add_kv("k.str", Value::Str("hello".into()));
        w.add_kv("k.int", Value::U64(42));
        w.add_kv("k.f", Value::F64(1.5));
        w.add_kv("k.arr", Value::Arr(vec![Value::U64(1), Value::U64(2)]));
        w.add_tensor("t.f32", DT_F32, &[3, 1], vec![0u8; 12]);
        w.add_tensor("t.tern", DT_TERNARY, &[256, 2], vec![7u8; 108]);
        let p = std::env::temp_dir().join("tickfile_unit.tick");
        w.finish(&p).unwrap();
        let t = TickFile::open(&p).unwrap();
        assert_eq!(t.meta.get("k.str").and_then(|v| v.as_str()), Some("hello"));
        assert_eq!(t.meta.get("k.int").and_then(|v| v.as_u64()), Some(42));
        let arr = t.meta.get("k.arr").unwrap();
        assert!(matches!(arr, Value::Arr(ref a) if a.len() == 2));
        assert_eq!(t.read_raw("t.f32").unwrap().len(), 12);
        assert_eq!(t.read_raw("t.tern").unwrap(), vec![7u8; 108]);
        let _ = std::fs::remove_file(&p);
    }
}
