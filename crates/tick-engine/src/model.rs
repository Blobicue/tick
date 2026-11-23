//! BitNet b1.58-2B-4T forward pass, mirroring the frozen reference
//! graph (llama-patched src/models/bitnet.cpp + ggml reference ops):
//!
//! embed → per layer [ RMSNorm(attn_norm) → BitLinear qkv → RoPE(NEOX,
//! base 500000) → causal GQA softmax (scale 1/√128) → RMSNorm(attn_sub_norm)
//! → BitLinear o → +residual → RMSNorm(ffn_norm) → BitLinear gate,up →
//! relu²(gate)⊙up → RMSNorm(ffn_sub_norm) → BitLinear down → +residual ]
//! → RMSNorm(output_norm) → tied f16 embedding head.
//!
//! ggml-mirroring details that are parity-load-bearing:
//! - RMSNorm: sum of squares in f64; scale = 1/sqrt(mean+eps); the
//!   elementwise order is (x*scale)*w (ggml rms_norm then ggml_mul).
//! - softmax: max in f32, expf(x-max), sum in f64, divide by the f32
//!   sum (ggml soft_max non-fa path).
//! - K/V cache stores f16 (llama.cpp default), read back as f32.

use std::fs::File;
use std::path::Path;
use tick_format::gguf::Gguf;
use tick_kernels::bitlinear::{bitlinear, decode_tq1_0, half_to_f32, TqTensor};
use tick_kernels::TickTelemetry;

pub struct Config {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize, // 128
    pub hidden: usize,   // 2560
    pub ffn: usize,      // 6912
    pub rms_eps: f32,
    pub rope_base: f32,
    pub vocab: usize,
}

pub struct Layer {
    pub attn_norm: Vec<f32>,
    pub attn_sub_norm: Vec<f32>,
    pub wq: TqTensor,
    pub wk: TqTensor,
    pub wv: TqTensor,
    pub wo: TqTensor,
    pub ffn_norm: Vec<f32>,
    pub ffn_sub_norm: Vec<f32>,
    pub gate: TqTensor,
    pub up: TqTensor,
    pub down: TqTensor,
}

/// Which compiled fattn path the reference used for this token:
/// prefill ubatches of 64+ query rows tile; decode switches to split-kv
/// when the PADDED kv length reaches 512 — llama pads n_kv to 256
/// blocks (llama_kv_cache::get_n_kv), so the split starts at REAL
/// n_kv = 257, not 512. Everything else runs one_chunk.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum AttnRegime {
    OneChunk,
    Tiled,
    SplitKV,
}

/// The reference's decode-regime decision at position pos (n_kv=pos+1):
/// split-kv once the 256-padded kv length hits 512.
pub fn decode_regime(pos: usize) -> AttnRegime {
    let n_kv = pos + 1;
    let padded = n_kv.div_ceil(256) * 256;
    if padded >= 512 {
        AttnRegime::SplitKV
    } else {
        AttnRegime::OneChunk
    }
}

/// TICK_PROFILE accumulators (microseconds) for the forward breakdown.
pub mod prof {
    use std::cell::RefCell;
    #[derive(Default)]
    pub struct Acc {
        pub matvec: u128,
        pub rope: u128,
        pub attn: u128,
        pub norm: u128,
        pub quant: u128,
        pub other: u128,
        pub tokens: u64,
    }
    thread_local! {
        pub static ACC: RefCell<Acc> = RefCell::new(Acc::default());
    }
    pub fn add(field: &str, us: u128) {
        ACC.with(|a| {
            let mut a = a.borrow_mut();
            match field {
                "matvec" => a.matvec += us,
                "rope" => a.rope += us,
                "attn" => a.attn += us,
                "norm" => a.norm += us,
                "quant" => a.quant += us,
                _ => a.other += us,
            }
        });
    }
    pub fn token() {
        ACC.with(|a| a.borrow_mut().tokens += 1);
    }
    pub fn dump() {
        ACC.with(|a| {
            let a = a.borrow();
            if a.tokens == 0 { return; }
            let t = a.tokens as f64;
            println!("prof tok={t} matvec={:.1}ms rope={:.1}ms attn={:.1}ms norm={:.1}ms quant={:.1}ms other={:.1}ms",
                a.matvec as f64 / t / 1000.0, a.rope as f64 / t / 1000.0,
                a.attn as f64 / t / 1000.0, a.norm as f64 / t / 1000.0,
                a.quant as f64 / t / 1000.0, a.other as f64 / t / 1000.0);
        });
    }
    pub fn enabled() -> bool {
        std::env::var("TICK_PROFILE").is_ok()
    }
}

pub struct Model {
    pub cfg: Config,
    /// tied token embedding / lm head, f16 rows [vocab, hidden]
    pub embd: Vec<u16>,
    /// the same rows pre-widened to f32 (exact, incl. subnormals) —
    /// the head is DRAM-bound on f32 rows (1.3GB/token at ~76GB/s);
    /// f16-rows+widen measured SLOWER (compute-bound widen)
    pub embd32: Vec<f32>,
    pub embd_norm: Vec<f32>,
    pub layers: Vec<Layer>,
    /// KV cache per layer, f16 bit patterns, \[cap\]\[n_head_kv*head_dim\]
    pub kv_cap: usize,
    pub k_cache: Vec<Vec<u16>>,
    pub v_cache: Vec<Vec<u16>>,
    pub telemetry: TickTelemetry,
    /// parity debugging: write named intermediates here (tensorprobe format)
    pub trace: Option<std::path::PathBuf>,
    /// rope trig cache for the current position (shared across layers)
    pub rope_cache: Vec<(f32, f32)>,
    pub rope_cache_pos: Option<usize>,
}

/// ggml rms_norm + mul(weight): f64 sum, (x*scale)*w.
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut sum = 0.0f64;
    for &v in x {
        sum += (v * v) as f64;
    }
    let mean = (sum / n as f64) as f32;
    let scale = 1.0f32 / (mean + eps).sqrt();
    x.iter().zip(w).map(|(&v, &g)| (v * scale) * g).collect()
}

/// ggml_v_expf — the ARM optimized-routines expf polynomial used by
/// softmax (max error 1.45 ulp; > 88.38 -> inf, < -103.97 -> 0).
/// Every op is fused multiply-add/substraction exactly as vectorized.
pub fn ggml_v_expf(x: f32) -> f32 {
    const LOG2E: f32 = 1.442_695;
    let r = 12582912.0;
    let z = x.mul_add(LOG2E, r);
    let n = z - r;
    let b = (-n).mul_add(1.428_606_8e-6, (-n).mul_add(0.693_145_75, x));
    let e = z.to_bits().wrapping_shl(23);
    let k = f32::from_bits(e.wrapping_add(1.0f32.to_bits()));
    let c = n.abs() > 126.0;
    let u = b * b;
    let j = u.mul_add(
        u.mul_add(
            b.mul_add(0.008_247_39, 0.041_899_767),
            b.mul_add(0.166_683_96, 0.499_991_27),
        ),
        b * 0.999_999_4,
    );
    if !c {
        return j.mul_add(k, k);
    }
    let d = if n <= 0.0 { 0x8200_0000u32 } else { 0 };
    let s1 = f32::from_bits(d.wrapping_add(0x7f00_0000));
    let s2 = f32::from_bits(e.wrapping_sub(d));
    if n.abs() > 192.0 {
        return s1 * s1;
    }
    j.mul_add(s2, s2) * s1
}

fn t_cfg(
    u: impl Fn(&str, u64) -> u64,
    f: impl Fn(&str) -> Option<f32>,
    vocab: usize,
) -> Result<Config, String> {
    Ok(Config {
        n_layer: u("bitnet.block_count", 30) as usize,
        n_head: u("bitnet.attention.head_count", 20) as usize,
        n_head_kv: u("bitnet.attention.head_count_kv", 5) as usize,
        head_dim: {
            let hidden = u("bitnet.embedding_length", 2560) as usize;
            let heads = u("bitnet.attention.head_count", 20) as usize;
            hidden / heads
        },
        hidden: u("bitnet.embedding_length", 2560) as usize,
        ffn: u("bitnet.feed_forward_length", 6912) as usize,
        rms_eps: f("bitnet.attention.layer_norm_rms_epsilon").unwrap_or(1e-5),
        rope_base: f("bitnet.rope.freq_base").unwrap_or(10000.0),
        vocab,
    })
}

impl Model {
    /// Open a model from either format: our native tickfile (magic
    /// "TICK") or a GGUF (import compat). Both paths share the tensor
    /// loaders via a read-closure; only the container parse differs.
    pub fn open(path: &Path, kv_cap: usize) -> Result<Model, String> {
        let mut magic = [0u8; 4];
        {
            use std::os::unix::fs::FileExt;
            let f = File::open(path).map_err(|e| e.to_string())?;
            f.read_exact_at(&mut magic, 0).map_err(|e| e.to_string())?;
        }
        if &magic == tick_format::weights::MAGIC {
            let t = tick_format::weights::TickFile::open(path)?;
            let cfg = t_cfg(
                |k, d| t.meta_u64(k).unwrap_or(d),
                |k| t.meta_f32(k),
                t.dir
                    .get("token_embd.weight")
                    .map(|(_, d, ..)| d[1] as usize)
                    .ok_or("tickfile: no token_embd")?,
            )?;
            let read = |name: &str| t.read_raw(name);
            return Model::build(cfg, &read, kv_cap);
        }
        let g = Gguf::open(path)?;
        let cfg = t_cfg(
            |k, d| g.meta_u64(k).unwrap_or(d),
            |k| g.meta_f32(k),
            g.tensor("token_embd.weight")?.dims[1] as usize,
        )?;
        let file = File::open(path).map_err(|e| e.to_string())?;
        let file_len = path.metadata().map_err(|e| e.to_string())?.len();
        let read = std::sync::Mutex::new((g, file, file_len));
        let read = |name: &str| {
            let (g, file, len) = &mut *read.lock().unwrap();
            g.read_raw(name, file, *len)
        };
        Model::build(cfg, &read, kv_cap)
    }
    fn build(
        cfg: Config,
        read: &(dyn Fn(&str) -> Result<Vec<u8>, String> + Sync),
        kv_cap: usize,
    ) -> Result<Model, String> {
        fn tqr(
            read: &dyn Fn(&str) -> Result<Vec<u8>, String>,
            name: &str,
            rows: usize,
            cols: usize,
        ) -> Result<TqTensor, String> {
            decode_tq1_0(&read(name)?, rows, cols)
        }
        fn f32r(
            read: &dyn Fn(&str) -> Result<Vec<u8>, String>,
            name: &str,
        ) -> Result<Vec<f32>, String> {
            Ok(read(name)?
                .chunks_exact(4)
                .map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                .collect())
        }
        // token_embd: f16 [hidden, vocab] gguf layout (ne0=hidden) —
        // we store column-major as-is and dot per vocab row on access
        let embd_raw = read("token_embd.weight")?;
        let embd: Vec<u16> = embd_raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        let embd_norm = f32r(&read, "output_norm.weight")?;

        // layer decode is pure CPU (tq1_0 unpack) and layers are
        // independent — decode them across the par pool and keep any
        // error for the caller
        let nl = cfg.n_layer;
        let layers_res: std::sync::Mutex<Vec<Option<Result<Layer, String>>>> =
            std::sync::Mutex::new((0..nl).map(|_| None).collect());
        tick_kernels::par::for_chunks(nl, 1, |s, e| {
            let h = cfg.hidden;
            let kv = cfg.n_head_kv * cfg.head_dim;
            let ff = cfg.ffn;
            for il in s..e {
                let b = format!("blk.{il}");
                let layer = (|| {
                    Ok(Layer {
                        attn_norm: f32r(&read, &format!("{b}.attn_norm.weight"))?,
                        attn_sub_norm: f32r(&read, &format!("{b}.attn_sub_norm.weight"))?,
                        wq: tqr(&read, &format!("{b}.attn_q.weight"), h, h)?,
                        wk: tqr(&read, &format!("{b}.attn_k.weight"), kv, h)?,
                        wv: tqr(&read, &format!("{b}.attn_v.weight"), kv, h)?,
                        wo: tqr(&read, &format!("{b}.attn_output.weight"), h, h)?,
                        ffn_norm: f32r(&read, &format!("{b}.ffn_norm.weight"))?,
                        ffn_sub_norm: f32r(&read, &format!("{b}.ffn_sub_norm.weight"))?,
                        gate: tqr(&read, &format!("{b}.ffn_gate.weight"), ff, h)?,
                        up: tqr(&read, &format!("{b}.ffn_up.weight"), ff, h)?,
                        down: tqr(&read, &format!("{b}.ffn_down.weight"), h, ff)?,
                    })
                })();
                layers_res.lock().unwrap()[il] = Some(layer);
            }
        });
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for r in layers_res.into_inner().unwrap() {
            layers.push(r.ok_or("layer decode missing")??);
        }
        let kv_elems = kv_cap * cfg.n_head_kv * cfg.head_dim;
        Ok(Model {
            kv_cap,
            k_cache: vec![vec![0u16; kv_elems]; cfg.n_layer],
            v_cache: vec![vec![0u16; kv_elems]; cfg.n_layer],
            telemetry: TickTelemetry::new(),
            trace: None,
            rope_cache: Vec::new(),
            rope_cache_pos: None,
            cfg,
            // the widen of 328M rows is the largest serial cost left
            // on load — parallel across the pool (exact per element)
            embd32: {
                let n = embd.len();
                let mut buf = vec![0.0f32; n];
                let p = tick_kernels::par::SendPtr(buf.as_mut_ptr());
                let p = &p;
                tick_kernels::par::for_chunks(n, 1 << 20, |s, e| {
                    let p = p.0;
                    for i in s..e {
                        unsafe {
                            *p.add(i) = tick_kernels::bitlinear::half_to_f32(embd[i]);
                        }
                    }
                });
                buf
            },
            embd,
            embd_norm,
            layers,
        })
    }

    /// f32→f16 bits (round-to-nearest-even, ggml FP32_TO_FP16).
    pub fn f32_to_f16_bits(v: f32) -> u16 {
        // round-trip via half arithmetic, implemented bit-exactly
        let x = v;
        let sign = ((x.to_bits() >> 16) & 0x8000) as u16;
        let mut mant = x.to_bits() & 0x007f_ffff;
        let exp = ((x.to_bits() >> 23) & 0xff) as i32;
        if exp == 0xff {
            return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
        }
        let e = exp - 127 + 15;
        if e >= 0x1f {
            return sign | 0x7c00;
        }
        if e <= 0 {
            if e < -10 {
                return sign;
            }
            mant |= 0x0080_0000;
            let shift = 14 - e;
            let m = mant >> shift;
            let rem = mant & ((1 << shift) - 1);
            let half = 1 << (shift - 1);
            let mut m = m;
            if rem > half || (rem == half && (m & 1) == 1) {
                m += 1;
            }
            return sign | m as u16;
        }
        let mut m = mant >> 13;
        let rem = mant & 0x1fff;
        if rem > 0x1000 || (rem == 0x1000 && (m & 1) == 1) {
            m += 1;
            if m == 0x400 {
                m = 0;
                return sign | (((e + 1) as u16) << 10);
            }
        }
        sign | ((e as u16) << 10) | m as u16
    }

    pub fn f32_to_f16_bits_pub(v: f32) -> u16 {
        Self::f32_to_f16_bits(v)
    }

    /// One token forward at position `pos`; returns the final hidden.
    /// Attention regime defaults to OneChunk (the reference's fattn
    /// one_chunk path) — use forward_token_regime for the tiled
    /// (prefill ubatch >= 64) and split-kv (decode n_kv >= 512) paths.
    pub fn forward_token(&mut self, token: usize, pos: usize) -> Result<Vec<f32>, String> {
        self.forward_token_regime(token, pos, AttnRegime::OneChunk)
    }

    pub fn forward_token_regime(
        &mut self,
        token: usize,
        pos: usize,
        regime: AttnRegime,
    ) -> Result<Vec<f32>, String> {
        if prof::enabled() {
            prof::token();
        }
        if token >= self.cfg.vocab {
            return Err(format!("token {token} out of vocab"));
        }
        let h = self.cfg.hidden;
        // embedding row: gguf ne layout [hidden, vocab] → row-major
        // over hidden for each vocab column: element (d, v) at v*h + d
        let mut x: Vec<f32> = self.embd[token * h..(token + 1) * h]
            .iter()
            .map(|&b| half_to_f32(b))
            .collect();
        let scale = 1.0f32 / (self.cfg.head_dim as f32).sqrt();
        let (n_head, n_kv, hd) = (self.cfg.n_head, self.cfg.n_head_kv, self.cfg.head_dim);
        let gqa = n_head / n_kv;

        for il in 0..self.cfg.n_layer {
            let layer = &self.layers[il];
            let x_prev6 = x.clone();
            let normed = rms_norm(&x, &layer.attn_norm, self.cfg.rms_eps);
            if il == 0 && pos == 1 {
                self.dump_trace("x-0-p1", &x);
                self.dump_trace("attn_norm-0-p1", &normed);
            }
            if il == 6 {
                self.dump_trace("attn_norm-6", &normed);
            }

            let (q, k, v) = if prof::enabled() {
                let t0 = std::time::Instant::now();
                let r =
                    tick_kernels::bitlinear::bitlinear3(&layer.wq, &layer.wk, &layer.wv, &normed);
                prof::add("matvec", t0.elapsed().as_micros());
                r
            } else {
                tick_kernels::bitlinear::bitlinear3(&layer.wq, &layer.wk, &layer.wv, &normed)
            };

            // NEOX rope on q and k: per head, pairs (i, i+hd/2);
            // theta by successive multiplication (ggml loop rounding)
            let mut qr = Vec::with_capacity(q.len());
            let mut kr = Vec::with_capacity(k.len());
            let _rope_t = if prof::enabled() {
                Some(std::time::Instant::now())
            } else {
                None
            };
            {
                let half = hd / 2;
                // the theta sequence depends only on (pos, dim) — cache
                // the trig once per token instead of per layer per head.
                // Identical recurrence and libm calls: bit-identical.
                if self.rope_cache_pos != Some(pos) || self.rope_cache.len() != half {
                    let theta_scale = self.cfg.rope_base.powf(-2.0 / hd as f32);
                    let mut theta = pos as f32;
                    self.rope_cache = Vec::with_capacity(half);
                    for _ in 0..half {
                        self.rope_cache.push((theta.cos(), theta.sin()));
                        theta *= theta_scale;
                    }
                    self.rope_cache_pos = Some(pos);
                }
                let rc = &self.rope_cache;
                for (vin, out) in [(&q, &mut qr), (&k, &mut kr)] {
                    for head in 0..(vin.len() / hd) {
                        let b = head * hd;
                        let mut hi = Vec::with_capacity(half);
                        for i in 0..half {
                            let (cos, sin) = rc[i];
                            let x0 = vin[b + i];
                            let x1 = vin[b + i + half];
                            // the reference's scalar rope (ops.cpp
                            // rotate_pairs) compiles with fp-contract:
                            // x0*cos - x1*sin => fl(x0·cos - fl(x1·sin))
                            // — the SECOND product fuses (fmsub), the
                            // first pre-rounds. Strict Rust differs by
                            // ulps at pos>0 (pos 0: sin=0 hides it).
                            // scalar C rotate_pairs compiled with
                            // fp-contract: GCC fuses the FIRST product
                            // (fmsub/fmadd on x0), pre-rounds the second
                            out.push(x0.mul_add(cos, -(x1 * sin)));
                            hi.push(x0.mul_add(sin, x1 * cos));
                        }
                        out.extend(hi);
                    }
                }
            }

            if il == 0 {
                // pos-tagged: ref tensorprobe layout is [dim, head, token]
                self.dump_trace(&format!("Kcur-0-p{pos}"), &kr);
                self.dump_trace(&format!("Vcur-0-p{pos}"), &v);
                self.dump_trace(&format!("Qcur-0-p{pos}"), &qr);
            }
            if let Some(t0) = _rope_t {
                prof::add("rope", t0.elapsed().as_micros());
            }
            // append k/v to the f16 cache
            let kvh = n_kv * hd;
            for (i, &kv) in kr.iter().enumerate() {
                self.k_cache[il][pos * kvh + i] = Self::f32_to_f16_bits(kv);
            }
            for (i, &vv) in v.iter().enumerate() {
                self.v_cache[il][pos * kvh + i] = Self::f32_to_f16_bits(vv);
            }

            // causal attention — the reference runs ggml FLASH ATTENTION
            // (ggml_flash_attn_ext, "__fattn__" tensors; cparams.flash_attn
            // is on in the frozen lane), NOT the decomposed kq/softmax/
            // V-mix path. The CPU one_chunk reference implementation
            // (ops.cpp ggml_compute_forward_flash_attn_ext_f16_one_chunk —
            // used for neq1 < Q_TILE_SZ=64 and split-kv needs n_kv >= 512,
            // so it covers every battery prefill and decode):
            //   Q -> f16 row conversion; per KV position (masked -inf
            //   positions are SKIPPED entirely): s = ggml_vec_dot_f16(K,Q)
            //   * scale; ONLINE softmax with running max M — on a new max
            //   the f16 accumulator VKQ16 is rescaled by expf(Mold-M)
            //   (vec_scale_f16), else V is accumulated with expf(s-M)
            //   (vec_mad_f16, FUSED f16 FMA with the scalar f16-rounded);
            //   S = S*ms + vs in f32; finally widen VKQ16 to f32 and
            //   scale by 1/S. expf is the SCALAR libm (glibc), not the
            //   vectorized polynomial.
            let mut attn = vec![0.0f32; n_head * hd];
            let _attn_t = if prof::enabled() {
                Some(std::time::Instant::now())
            } else {
                None
            };
            for hq in 0..n_head {
                let kv_head = hq / gqa;
                match regime {
                    AttnRegime::Tiled => {
                        let out = self.attn_tiled_row(
                            &qr[hq * hd..(hq + 1) * hd],
                            il,
                            kv_head,
                            hd,
                            pos,
                            scale,
                        );
                        attn[hq * hd..(hq + 1) * hd].copy_from_slice(&out);
                        continue;
                    }
                    AttnRegime::SplitKV => {
                        let out = self.attn_split_row(
                            &qr[hq * hd..(hq + 1) * hd],
                            il,
                            kv_head,
                            hd,
                            pos,
                            scale,
                        );
                        attn[hq * hd..(hq + 1) * hd].copy_from_slice(&out);
                        continue;
                    }
                    AttnRegime::OneChunk => {}
                }
                let q16: Vec<u16> = qr[hq * hd..(hq + 1) * hd]
                    .iter()
                    .map(|&f| tick_kernels::bitlinear::f32_to_f16(f))
                    .collect();
                let mut vkq16 = vec![0u16; hd];
                let mut m = f32::NEG_INFINITY;
                let mut s_sum = 0.0f32;
                for ic in 0..=pos {
                    let kb =
                        &self.k_cache[il][ic * kvh + kv_head * hd..ic * kvh + (kv_head + 1) * hd];
                    let s = tick_kernels::bitlinear::dot_f16_sve(kb, &q16) * scale;
                    let ms;
                    let vs;
                    if s > m {
                        let mold = m;
                        m = s;
                        ms = (mold - m).exp();
                        tick_kernels::bitlinear::vec_scale_f16(&mut vkq16, ms);
                        vs = 1.0;
                    } else {
                        ms = 1.0;
                        vs = (s - m).exp();
                    }
                    let vb =
                        &self.v_cache[il][ic * kvh + kv_head * hd..ic * kvh + (kv_head + 1) * hd];
                    tick_kernels::bitlinear::vec_mad_f16(&mut vkq16, vb, vs);
                    // fp-contracted in the reference build (fma)
                    s_sum = s_sum.mul_add(ms, vs);
                }
                let inv = if s_sum == 0.0 { 0.0 } else { 1.0 / s_sum };
                let oh = &mut attn[hq * hd..(hq + 1) * hd];
                for d in 0..hd {
                    oh[d] = tick_kernels::bitlinear::half_to_f32(vkq16[d]) * inv;
                }
            }

            if pos == 0 && il == 6 {
                self.dump_trace("attnmix-6", &attn);
            }
            if pos == 5 && il < 2 {
                self.dump_trace(&format!("kqv_out-{il}"), &attn);
            }
            if pos == 1 && il == 0 {
                self.dump_trace("attnmix_p1", &attn);
            }
            if let Some(t0) = _attn_t {
                prof::add("attn", t0.elapsed().as_micros());
            }
            let sub = rms_norm(&attn, &layer.attn_sub_norm, self.cfg.rms_eps);
            if pos == 0 && il == 6 {
                self.dump_trace("attn_sub_norm-6", &sub);
            }
            let ao6 = bitlinear(&layer.wo, &sub);
            if pos == 0 && il == 6 {
                self.dump_trace("attn_out-6", &ao6);
            }
            let fi6: Vec<f32> = ao6.iter().zip(&x_prev6).map(|(a, b)| a + b).collect();
            if pos == 0 && il == 6 {
                self.dump_trace("ffn_inp-6", &fi6);
            }
            let attn_out = if prof::enabled() {
                let t0 = std::time::Instant::now();
                let r = bitlinear(&layer.wo, &sub);
                prof::add("matvec", t0.elapsed().as_micros());
                r
            } else {
                bitlinear(&layer.wo, &sub)
            };
            let ffn_inp: Vec<f32> = x.iter().zip(&attn_out).map(|(a, b)| a + b).collect();
            if il == 0 && pos <= 15 {
                self.dump_trace(&format!("sub-0-p{pos}"), &sub);
            }
            if pos == 1 && il == 0 {
                self.dump_trace("attn_out-0-p1", &attn_out);
                self.dump_trace("ffn_inp-0-p1", &ffn_inp);
            }

            let normed2 = rms_norm(&ffn_inp, &layer.ffn_norm, self.cfg.rms_eps);
            if il == 6 {
                self.dump_trace("ffn_norm-6", &normed2);
            }
            let (g, u) = if prof::enabled() {
                let t0 = std::time::Instant::now();
                let r = tick_kernels::bitlinear::bitlinear2(&layer.gate, &layer.up, &normed2);
                prof::add("matvec", t0.elapsed().as_micros());
                r
            } else {
                tick_kernels::bitlinear::bitlinear2(&layer.gate, &layer.up, &normed2)
            };
            if pos == 1 && il == 0 {
                self.dump_trace("normed2-0-p1", &normed2);
                self.dump_trace("g-0-p1", &g);
                self.dump_trace("u-0-p1", &u);
            }
            if il == 6 {
                self.dump_trace("ffn_gate-6", &g);
                self.dump_trace("ffn_up-6", &u);
            }
            // relu² (gate) ⊙ up
            let mut act = vec![0.0f32; self.cfg.ffn];
            for i in 0..self.cfg.ffn {
                let r = if g[i] > 0.0 { g[i] } else { 0.0 };
                act[i] = r * r * u[i];
            }

            let sub2 = rms_norm(&act, &layer.ffn_sub_norm, self.cfg.rms_eps);
            if il == 6 {
                self.dump_trace("ffn_sub_norm-6", &sub2);
            }
            let ffn_out_v = bitlinear(&layer.down, &sub2);
            if il == 6 {
                self.dump_trace("ffn_down-6", &ffn_out_v);
            }
            if pos == 1 && il == 0 {
                self.dump_trace("act-0-p1", &act);
                self.dump_trace("sub2-0-p1", &sub2);
                self.dump_trace("down-0-p1", &ffn_out_v);
            }
            let ffn_out = if prof::enabled() {
                let t0 = std::time::Instant::now();
                let r = bitlinear(&layer.down, &sub2);
                prof::add("matvec", t0.elapsed().as_micros());
                r
            } else {
                bitlinear(&layer.down, &sub2)
            };
            x = ffn_inp.iter().zip(&ffn_out).map(|(a, b)| a + b).collect();
            self.dump_trace(&format!("l_out-{il}"), &x);
            self.telemetry.tick(il as u32);
        }
        self.telemetry.tick(u32::MAX);
        Ok(x)
    }

    /// The tiled fattn row (ops.cpp ggml_compute_forward_flash_attn_ext_tiled
    /// as compiled on this build): ALL-f32 — Q stays raw f32, scores are a
    /// scalar fma chain against f32-widened K, softmax per KV tile of 64
    /// via ggml_v_expf with f64 4-chunk sums, V accumulated with a scalar
    /// fma chain into an f32 accumulator. Fitted bit-exact vs ggml's own
    /// flash_attn_ext (fattnprobe rows:70, 0/179200 differ).
    fn attn_tiled_row(
        &self,
        q: &[f32],
        il: usize,
        kv_head: usize,
        hd: usize,
        pos: usize,
        scale: f32,
    ) -> Vec<f32> {
        let n_kv = pos + 1;
        let kvh_stride = self.cfg.n_head_kv * hd;
        let mut s_sum = 0.0f32;
        let mut m = f32::NEG_INFINITY;
        let mut vkq32 = vec![0.0f32; hd];
        let mut ic0 = 0usize;
        while ic0 < n_kv {
            let kv_tile = (n_kv - ic0).min(64);
            if ic0 > pos {
                ic0 += 64;
                continue;
            }
            let mut kq = [0.0f32; 64];
            for tk in 0..kv_tile {
                let base = (ic0 + tk) * kvh_stride + kv_head * hd;
                let kb = &self.k_cache[il][base..base + hd];
                let mut acc = 0.0f32;
                for dk in 0..hd {
                    acc = q[dk].mul_add(tick_kernels::bitlinear::half_to_f32(kb[dk]), acc);
                }
                kq[tk] = acc;
            }
            for tk in 0..64 {
                kq[tk] *= scale;
            }
            for tk in kv_tile..64 {
                kq[tk] = f32::NEG_INFINITY;
            }
            for tk in 0..kv_tile {
                if ic0 + tk > pos {
                    kq[tk] = f32::NEG_INFINITY;
                }
            }
            let mut tile_max = f32::NEG_INFINITY;
            for tk in 0..64 {
                if kq[tk] > tile_max {
                    tile_max = kq[tk];
                }
            }
            let mold = m;
            let mnew = if tile_max > mold { tile_max } else { mold };
            if mnew > mold {
                let ms = (mold - mnew).exp();
                for v in vkq32.iter_mut() {
                    *v *= ms;
                }
                s_sum *= ms;
            }
            m = mnew;
            let mut y = [0.0f32; 64];
            let mut sum = 0.0f64;
            for c in 0..16 {
                let lane = [
                    ggml_v_expf(kq[c * 4] - mnew),
                    ggml_v_expf(kq[c * 4 + 1] - mnew),
                    ggml_v_expf(kq[c * 4 + 2] - mnew),
                    ggml_v_expf(kq[c * 4 + 3] - mnew),
                ];
                sum += ((lane[0] + lane[1]) + (lane[2] + lane[3])) as f64;
                y[c * 4..c * 4 + 4].copy_from_slice(&lane);
            }
            s_sum = (s_sum as f64 + sum) as f32;
            for tk in 0..64 {
                if y[tk] == 0.0 {
                    continue;
                }
                let base = (ic0 + tk) * kvh_stride + kv_head * hd;
                let vb = &self.v_cache[il][base..base + hd];
                let yv = y[tk];
                for d in 0..hd {
                    let vd = tick_kernels::bitlinear::half_to_f32(vb[d]);
                    vkq32[d] = yv.mul_add(vd, vkq32[d]);
                }
            }
            ic0 += 64;
        }
        let inv = if s_sum == 0.0 { 0.0 } else { 1.0 / s_sum };
        vkq32.iter().map(|v| v * inv).collect()
    }

    /// The split-kv decode path (n_kv >= 512): KV chunks of
    /// ceil(n_kv/nth) with nth = 10 (the lane runs -t 10), each chunk
    /// through the one_chunk f16-accumulator arithmetic, then the
    /// partials reduce — fma on the OLD partial, the new partial
    /// pre-multiplied (fitted bit-exact vs fattnprobe nth=10,
    /// n_kv=600, 0/2560 differ).
    fn attn_split_row(
        &self,
        q: &[f32],
        il: usize,
        kv_head: usize,
        hd: usize,
        pos: usize,
        scale: f32,
    ) -> Vec<f32> {
        const LANE_THREADS: usize = 10;
        let n_kv = pos + 1;
        // the reference graphs n_kv PADDED to 256-blocks; chunk_size is
        // ceil(nek1_padded / nth) and chunks span the padded range —
        // fully-masked (unwritten) chunks yield S=0 partials which the
        // reduce skips, so clipping chunk ends to the real n_kv is
        // arithmetically identical.
        let padded = n_kv.div_ceil(256) * 256;
        let kvh_stride = self.cfg.n_head_kv * hd;
        let q16: Vec<u16> = q
            .iter()
            .map(|&f| tick_kernels::bitlinear::f32_to_f16(f))
            .collect();
        let chunk_size = padded.div_ceil(LANE_THREADS);
        let mut m_f = f32::NEG_INFINITY;
        let mut s_f = 0.0f32;
        let mut vkq_f = vec![0.0f32; hd];
        for c in 0..LANE_THREADS {
            let start = c * chunk_size;
            if start >= n_kv {
                continue;
            }
            let end = (start + chunk_size).min(n_kv);
            // one_chunk over [start, end)
            let mut vkq16 = vec![0u16; hd];
            let mut m = f32::NEG_INFINITY;
            let mut s_sum = 0.0f32;
            for ic in start..end {
                let base = ic * kvh_stride + kv_head * hd;
                let kb = &self.k_cache[il][base..base + hd];
                let s = tick_kernels::bitlinear::dot_f16_sve(kb, &q16) * scale;
                let (ms, vs);
                if s > m {
                    let mold = m;
                    m = s;
                    ms = (mold - m).exp();
                    tick_kernels::bitlinear::vec_scale_f16(&mut vkq16, ms);
                    vs = 1.0;
                } else {
                    ms = 1.0;
                    vs = (s - m).exp();
                }
                let vb = &self.v_cache[il][base..base + hd];
                tick_kernels::bitlinear::vec_mad_f16(&mut vkq16, vb, vs);
                s_sum = s_sum.mul_add(ms, vs);
            }
            let s_c = s_sum;
            if s_c == 0.0 {
                continue;
            }
            let m_new = if m > m_f { m } else { m_f };
            let so = (m_f - m_new).exp();
            let sn = (m - m_new).exp();
            for d in 0..hd {
                let vc = tick_kernels::bitlinear::half_to_f32(vkq16[d]);
                vkq_f[d] = vkq_f[d].mul_add(so, vc * sn);
            }
            s_f = s_f.mul_add(so, s_c * sn);
            m_f = m_new;
        }
        let inv = if s_f == 0.0 { 0.0 } else { 1.0 / s_f };
        vkq_f.iter().map(|v| v * inv).collect()
    }

    /// Export the model as a tickfile (v0.2 layer-streamed layout) —
    /// the trainer's emission seam: any in-memory Model (trained,
    /// mutated, or loaded) writes an artifact the engine runs back.
    /// KV carries the config keys the loader reads plus provenance.
    pub fn save_tickfile(&self, path: &Path, source: &str) -> Result<(), String> {
        use tick_format::gguf::Value;
        use tick_format::weights::{TickWriter, DT_F16, DT_F32, DT_TERNARY};
        let c = &self.cfg;
        let mut w = TickWriter::new();
        let f32kv = |v: f32| Value::F64(v as f64);
        w.add_kv("bitnet.block_count", Value::U64(c.n_layer as u64));
        w.add_kv("bitnet.attention.head_count", Value::U64(c.n_head as u64));
        w.add_kv(
            "bitnet.attention.head_count_kv",
            Value::U64(c.n_head_kv as u64),
        );
        w.add_kv("bitnet.embedding_length", Value::U64(c.hidden as u64));
        w.add_kv("bitnet.feed_forward_length", Value::U64(c.ffn as u64));
        w.add_kv("bitnet.attention.layer_norm_rms_epsilon", f32kv(c.rms_eps));
        w.add_kv("bitnet.rope.freq_base", f32kv(c.rope_base));
        w.add_kv(
            "tick.saved_at",
            Value::F64(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0),
            ),
        );
        w.add_kv("tick.source_path", Value::Str(source.to_string()));
        // tensors in the loader's naming contract
        let f32t = |w: &mut TickWriter, name: &str, v: &[f32]| {
            let mut b = Vec::with_capacity(v.len() * 4);
            for x in v {
                b.extend_from_slice(&x.to_bits().to_le_bytes());
            }
            w.add_tensor(name, DT_F32, &[v.len() as u64, 1], b);
        };
        // embd: f16 column-major [hidden, vocab] as loaded
        let mut eb = Vec::with_capacity(self.embd.len() * 2);
        for x in &self.embd {
            eb.extend_from_slice(&x.to_le_bytes());
        }
        w.add_tensor(
            "token_embd.weight",
            DT_F16,
            &[c.hidden as u64, c.vocab as u64],
            eb,
        );
        f32t(&mut w, "output_norm.weight", &self.embd_norm);
        for (il, l) in self.layers.iter().enumerate() {
            let b = format!("blk.{il}");
            f32t(&mut w, &format!("{b}.attn_norm.weight"), &l.attn_norm);
            f32t(
                &mut w,
                &format!("{b}.attn_sub_norm.weight"),
                &l.attn_sub_norm,
            );
            let tq = |w: &mut TickWriter, name: &str, t: &tick_kernels::bitlinear::TqTensor| {
                w.add_tensor(
                    name,
                    DT_TERNARY,
                    &[t.cols as u64, t.rows as u64],
                    tick_kernels::bitlinear::encode_tq1_0(t),
                );
            };
            tq(&mut w, &format!("{b}.attn_q.weight"), &l.wq);
            tq(&mut w, &format!("{b}.attn_k.weight"), &l.wk);
            tq(&mut w, &format!("{b}.attn_v.weight"), &l.wv);
            tq(&mut w, &format!("{b}.attn_output.weight"), &l.wo);
            f32t(&mut w, &format!("{b}.ffn_norm.weight"), &l.ffn_norm);
            f32t(&mut w, &format!("{b}.ffn_sub_norm.weight"), &l.ffn_sub_norm);
            tq(&mut w, &format!("{b}.ffn_gate.weight"), &l.gate);
            tq(&mut w, &format!("{b}.ffn_up.weight"), &l.up);
            tq(&mut w, &format!("{b}.ffn_down.weight"), &l.down);
        }
        w.finish_ordered(path, true)?;
        Ok(())
    }

    /// bench seam: widen f16 bits to f32 (subset of the old embd32)
    pub fn widen_embd_subset(bits: &[u16]) -> Vec<f32> {
        bits.iter()
            .map(|&b| tick_kernels::bitlinear::half_to_f32(b))
            .collect()
    }

    pub fn dump_trace(&self, name: &str, v: &[f32]) {
        use std::io::Write;
        let Some(path) = &self.trace else { return };
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(f, "TENSOR {} n={}", name, v.len());
            for x in v {
                let _ = writeln!(f, "{:.9e}", x);
            }
        }
    }

    /// Tied head: logits = embd_column · normed_x (f16 dots, f32).
    /// gguf layout [hidden, vocab]: element (d, v) at v*hidden + d —
    /// each vocab token's row is contiguous, dot in f32 after f16→f32.
    pub fn logits(&self, x: &[f32]) -> Vec<f32> {
        // the head gemm's llamafile path is compiled out with the rest:
        // x is converted to f16 (RNE) first, then ggml_vec_dot_f16 per
        // vocab row — BIT-EXACT vs reference pos-0 logits
        // (128256/128256) once half_to_f32 stopped doubling subnormals
        let h = self.cfg.hidden;
        let x16: Vec<u16> = x.iter().map(|&f| Self::f32_to_f16_bits(f)).collect();
        let xw: Vec<f32> = x16
            .iter()
            .map(|&b| tick_kernels::bitlinear::half_to_f32(b))
            .collect();
        let mut out = vec![0.0f32; self.cfg.vocab];
        #[cfg(target_arch = "aarch64")]
        tick_kernels::par::par_rows(&mut out, |v| {
            let row = &self.embd32[v * h..(v + 1) * h];
            tick_kernels::bitlinear::dot_f32x16_neon(row, &xw)
        });
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = &xw;
            tick_kernels::par::par_rows(&mut out, |v| {
                let row = &self.embd[v * h..(v + 1) * h];
                tick_kernels::bitlinear::dot_f16_sve(row, &x16)
            });
        }
        let _ = &x16;
        out
    }

    /// Greedy generate: prefill tokens, then `n_gen` argmax steps.
    pub fn greedy(&mut self, tokens: &[u32], n_gen: usize) -> Result<(Vec<u32>, Vec<f32>), String> {
        let (out, logits) = self.greedy_until(tokens, n_gen, &[], &mut |_id| {})?;
        Ok((out, logits))
    }

    /// Streaming greedy: generates up to n_gen tokens, stopping early
    /// on any stop id (the stop token itself is yielded to the
    /// callback but NOT appended to the returned ids). on_token fires
    /// per generated token from the inference thread.
    pub fn greedy_until(
        &mut self,
        tokens: &[u32],
        n_gen: usize,
        stop: &[u32],
        on_token: &mut dyn FnMut(u32),
    ) -> Result<(Vec<u32>, Vec<f32>), String> {
        if tokens.is_empty() {
            return Err("empty prompt".into());
        }
        if tokens.len() + n_gen >= self.kv_cap {
            return Err("sequence exceeds kv cap".into());
        }
        let mut hidden = None;
        // reference fattn geometry per position: prompts up to 512
        // tokens are ONE ubatch (n_ubatch=512); 64+ query rows tile,
        // and a prompt-tail ubatch of <64 rows runs one_chunk. Decode
        // steps split KV across the lane's 10 threads at n_kv >= 512.
        let plen = tokens.len();
        for (i, &t) in tokens.iter().enumerate() {
            let ubatch_len = if plen <= 512 {
                plen
            } else if i < (plen / 512) * 512 {
                512
            } else {
                plen % 512
            };
            let regime = if ubatch_len >= 64 {
                AttnRegime::Tiled
            } else {
                AttnRegime::OneChunk
            };
            hidden = Some(self.forward_token_regime(t as usize, i, regime)?);
        }
        let mut hidden = hidden.ok_or("no hidden")?;
        let mut out = Vec::new();
        let mut logits = Vec::new();
        for pos in tokens.len()..tokens.len() + n_gen {
            let normed = rms_norm(&hidden, &self.embd_norm, self.cfg.rms_eps);
            logits = self.logits(&normed);
            let argmax = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            on_token(argmax);
            if stop.contains(&argmax) {
                logits.clear();
                break;
            }
            out.push(argmax);
            hidden = self.forward_token_regime(argmax as usize, pos, decode_regime(pos))?;
        }
        Ok((out, logits))
    }
}
