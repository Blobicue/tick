//! M4 trainer — from-scratch ternary QAT for the tick engine.
//!
//! The design law: the trainer never invents its own arithmetic —
//! inference-side kernels are the ENGINE's (`bitlinear::` dot, the
//! same packed-trit layout), and the trained artifact is a tickfile
//! the stock `Model::open` runs. Training state keeps master FLOAT
//! weights; the ternary view (pack + f16 scales) is derived by the
//! same quantizer inference uses, and gradients flow through it via
//! the straight-through estimator (STE).

use tick_format::gguf::Value;
use tick_kernels::bitlinear::{f32_to_f16, half_to_f32, pack_trits, TqTensor};

/// The tiny L3-resident target (SPEC.md M4): ~25M ternary params.
#[derive(Clone, Debug)]
pub struct TinyConfig {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub hidden: usize,
    pub ffn: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub rope_base: f32,
}

impl Default for TinyConfig {
    fn default() -> Self {
        // 8 layers × ~2.8M ternary + tied embd 4096×512 f16:
        // ≈ 26.7M ternary (~9.4MB artifact incl. f16 scales+embd).
        // ffn 1536 (not 1408): ternary blocks are 256-wide — every
        // matrix dim must be a multiple of QK_K or decode refuses
        TinyConfig {
            n_layer: 8,
            n_head: 8,
            n_head_kv: 2,
            head_dim: 64,
            hidden: 512,
            ffn: 1536,
            vocab: 4096,
            rms_eps: 1e-5,
            rope_base: 10000.0,
        }
    }
}

/// A trainable BitLinear weight: float master + derived ternary view.
pub struct TrainableTq {
    pub master: Vec<f32>, // rows × cols, row-major
    pub rows: usize,
    pub cols: usize,
}

impl TrainableTq {
    pub fn new(rows: usize, cols: usize, mut init: impl FnMut() -> f32) -> Self {
        TrainableTq {
            master: (0..rows * cols).map(|_| init()).collect(),
            rows,
            cols,
        }
    }

    /// BitNet b1.58 weight quantization: per-256-block scale =
    /// mean|w| (the abs-mean estimator), w_q = round(w / scale) clipped
    /// to {-1, 0, +1} — the exact scheme the engine's TqTensor
    /// represents (f16 scale + ternary).
    pub fn ternary(&self) -> TqTensor {
        let cols = self.cols;
        assert!(
            cols.is_multiple_of(tick_kernels::bitlinear::QK_K),
            "ternary cols must be a multiple of {} (got {cols}) — the              tq block format has no tail; pick dims accordingly",
            tick_kernels::bitlinear::QK_K
        );
        let n_blocks = cols.div_ceil(tick_kernels::bitlinear::QK_K);
        let mut trits = vec![0i8; self.rows * n_blocks * tick_kernels::bitlinear::QK_K];
        let mut d = vec![0f32; self.rows * n_blocks];
        for r in 0..self.rows {
            for b in 0..n_blocks {
                let s = b * tick_kernels::bitlinear::QK_K;
                let e = ((b + 1) * tick_kernels::bitlinear::QK_K).min(cols);
                // abs-mean over the block (clamp min so all-zero blocks
                // give scale 0 → all-zero trits)
                let mut sum = 0.0f32;
                for c in s..e {
                    sum += self.master[r * cols + c].abs();
                }
                let scale = sum / ((e - s) as f32);
                let scale16 = half_to_f32(f32_to_f16(scale));
                d[r * n_blocks + b] = scale16;
                for c in s..e {
                    let w = self.master[r * cols + c];
                    let t = if scale16 == 0.0 {
                        0.0
                    } else {
                        (w / scale16).round().clamp(-1.0, 1.0)
                    };
                    trits[r * n_blocks * tick_kernels::bitlinear::QK_K
                        + b * tick_kernels::bitlinear::QK_K
                        + (c - s)] = t as i8;
                }
            }
        }
        TqTensor {
            rows: self.rows,
            cols,
            n_blocks,
            packed: pack_trits(&trits),
            d,
        }
    }

    /// A copy of the master weights (checkpoint shim).
    pub fn clone_view(&self) -> TrainableTq {
        TrainableTq {
            master: self.master.clone(),
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// STE backward through the ternary view: grad flows to master
    /// where the quantized weight is inside the clip region (|w/scale|
    /// ≤ 1, i.e., not saturated); saturated weights get zero grad —
    /// the standard QAT estimator. `grads` is dL/d(w_quantized·scale)
    /// in row-major rows×cols (the dense-shaped gradient).
    pub fn ste_update(&mut self, grads: &[f32], lr: f32) {
        let cols = self.cols;
        let n_blocks = cols.div_ceil(tick_kernels::bitlinear::QK_K);
        for r in 0..self.rows {
            for b in 0..n_blocks {
                let s = b * tick_kernels::bitlinear::QK_K;
                let e = ((b + 1) * tick_kernels::bitlinear::QK_K).min(cols);
                let scale = {
                    let mut sum = 0.0f32;
                    for c in s..e {
                        sum += self.master[r * cols + c].abs();
                    }
                    half_to_f32(f32_to_f16(sum / ((e - s) as f32)))
                };
                for c in s..e {
                    let i = r * cols + c;
                    let g = grads[i];
                    let in_region = scale == 0.0 || (self.master[i] / scale).abs() <= 1.5;
                    if in_region {
                        self.master[i] -= lr * g;
                    }
                }
            }
        }
        let _ = n_blocks;
    }
}

/// Emit a runnable tickfile from trained pieces — the same writer path
/// `Model::save_tickfile` uses, but for a from-scratch tiny model (no
/// pretrained Model object). The stock engine loads the result.
pub fn write_tiny_tickfile(
    cfg: &TinyConfig,
    embd: &[u16], // [vocab × hidden] f16 bits, tied head
    norms: &TinyNorms,
    mats: &TinyMats,
    path: &std::path::Path,
) -> Result<(), String> {
    use tick_format::weights::{TickWriter, DT_F16, DT_F32, DT_TERNARY};
    let mut w = TickWriter::new();
    w.add_kv("bitnet.block_count", Value::U64(cfg.n_layer as u64));
    w.add_kv("bitnet.attention.head_count", Value::U64(cfg.n_head as u64));
    w.add_kv(
        "bitnet.attention.head_count_kv",
        Value::U64(cfg.n_head_kv as u64),
    );
    w.add_kv("bitnet.embedding_length", Value::U64(cfg.hidden as u64));
    w.add_kv("bitnet.feed_forward_length", Value::U64(cfg.ffn as u64));
    w.add_kv(
        "bitnet.attention.layer_norm_rms_epsilon",
        Value::F64(cfg.rms_eps as f64),
    );
    w.add_kv("bitnet.rope.freq_base", Value::F64(cfg.rope_base as f64));
    w.add_kv(
        "tick.saved_at",
        Value::F64(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0),
        ),
    );
    w.add_kv("tick.source_path", Value::Str("tick-train M4.0".into()));
    let f32t = |w: &mut TickWriter, name: &str, v: &[f32]| {
        let mut b = Vec::with_capacity(v.len() * 4);
        for x in v {
            b.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        w.add_tensor(name, DT_F32, &[v.len() as u64, 1], b);
    };
    let mut eb = Vec::with_capacity(embd.len() * 2);
    for x in embd {
        eb.extend_from_slice(&x.to_le_bytes());
    }
    w.add_tensor(
        "token_embd.weight",
        DT_F16,
        &[cfg.hidden as u64, cfg.vocab as u64],
        eb,
    );
    f32t(&mut w, "output_norm.weight", &norms.output_norm);
    for il in 0..cfg.n_layer {
        let b = format!("blk.{il}");
        let l = &mats.layers[il];
        f32t(&mut w, &format!("{b}.attn_norm.weight"), &l.attn_norm);
        f32t(
            &mut w,
            &format!("{b}.attn_sub_norm.weight"),
            &l.attn_sub_norm,
        );
        let tq = |w: &mut TickWriter, name: &str, t: &TrainableTq| {
            w.add_tensor(
                name,
                DT_TERNARY,
                &[t.cols as u64, t.rows as u64],
                tick_kernels::bitlinear::encode_tq1_0(&t.ternary()),
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

/// Norm vectors for the tiny model (RMSNorm weights; init to 1.0).
pub struct TinyNorms {
    pub output_norm: Vec<f32>,
}

pub struct TinyLayerMats {
    pub attn_norm: Vec<f32>,
    pub attn_sub_norm: Vec<f32>,
    pub wq: TrainableTq,
    pub wk: TrainableTq,
    pub wv: TrainableTq,
    pub wo: TrainableTq,
    pub ffn_norm: Vec<f32>,
    pub ffn_sub_norm: Vec<f32>,
    pub gate: TrainableTq,
    pub up: TrainableTq,
    pub down: TrainableTq,
}

pub struct TinyMats {
    pub layers: Vec<TinyLayerMats>,
}

/// Random-init the full tiny model (normal(0, fan-in^-0.5) like torch
/// default for the masters; norms at 1.0).
pub fn init_tiny(cfg: &TinyConfig, seed: u64) -> (Vec<u16>, TinyNorms, TinyMats) {
    let mut rng = Rng { s: seed };
    let h = cfg.hidden;
    let kv = cfg.n_head_kv * cfg.head_dim;
    let ff = cfg.ffn;
    let embd: Vec<u16> = (0..cfg.vocab * h)
        .map(|_| f32_to_f16(rng.normal(0.0, (h as f32).powf(-0.5))))
        .collect();
    let norms = TinyNorms {
        output_norm: vec![1.0; h],
    };
    let mut layers = Vec::with_capacity(cfg.n_layer);
    for _ in 0..cfg.n_layer {
        let mut mk =
            |r: usize, c: usize| TrainableTq::new(r, c, || rng.normal(0.0, (c as f32).powf(-0.5)));
        layers.push(TinyLayerMats {
            attn_norm: vec![1.0; h],
            attn_sub_norm: vec![1.0; h],
            wq: mk(h, h),
            wk: mk(kv, h),
            wv: mk(kv, h),
            wo: mk(h, h),
            ffn_norm: vec![1.0; h],
            ffn_sub_norm: vec![1.0; ff],
            gate: mk(ff, h),
            up: mk(ff, h),
            down: mk(h, ff),
        });
    }
    (embd, norms, TinyMats { layers })
}

/// xorshift64* — deterministic, no deps.
pub struct Rng {
    pub s: u64,
}

impl Rng {
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> u64 {
        let mut x = self.s;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.s = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    pub fn normal(&mut self, mean: f32, std: f32) -> f32 {
        // Box–Muller from two uniforms
        let u1 = (self.next() >> 11) as f64 / (1u64 << 53) as f64 + 1e-12;
        let u2 = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std * z as f32
    }
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next() >> 11) as f32 / (1u64 << 53) as f32)
    }
}

// ---------------------------------------------------------------------------
// M4.1 — hand-autograd for the tiny transformer (single sequence, batch 1).
// Forward uses the DENSE-VALUED quantized weights (ternary × scale from
// the masters) so training flows through weight quantization (STE on the
// backward); activations stay f32 (activation-QA lands with M4.2).
// ---------------------------------------------------------------------------

/// Dense-valued quantized weights (rows × cols row-major): t·d.
pub fn dense_of(t: &TqTensor) -> Vec<f32> {
    let mut out = vec![0.0f32; t.rows * t.cols];
    for r in 0..t.rows {
        for blk in 0..t.n_blocks {
            let d = t.d[r * t.n_blocks + blk];
            let base = r * t.n_blocks * 64 + blk * 64;
            for j in 0..256usize {
                let c = blk * 256 + j;
                if c < t.cols {
                    out[r * t.cols + c] =
                        tick_kernels::bitlinear::trit_at(&t.packed[base..base + 64], j) as f32 * d;
                }
            }
        }
    }
    out
}

/// RMSNorm forward (the engine's exact semantics: f64 sum, (x·s)·w).
pub fn rms_fwd(x: &[f32], w: &[f32], eps: f32) -> (Vec<f32>, f32) {
    let n = x.len();
    let mut sum = 0.0f64;
    for &v in x {
        sum += (v * v) as f64;
    }
    let scale = 1.0f32 / (sum as f32 / n as f32 + eps).sqrt();
    let y = x.iter().zip(w).map(|(&v, &g)| (v * scale) * g).collect();
    (y, scale)
}

/// RMSNorm backward: gx, gw.
pub fn rms_bwd(x: &[f32], w: &[f32], scale: f32, gy: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = x.len();
    let mut gx = vec![0.0f32; n];
    let mut gw = vec![0.0f32; n];
    let mut dot = 0.0f64;
    for i in 0..n {
        dot += (gy[i] * w[i] * x[i]) as f64;
    }
    let dot = dot as f32;
    for i in 0..n {
        gx[i] = scale * w[i] * gy[i] - (x[i] * scale * scale * scale / n as f32) * dot;
        gw[i] = gy[i] * x[i] * scale;
    }
    (gx, gw)
}

/// The values inference actually multiplies: quantize to q8_K (the
/// engine's exact activation quantizer) and dequantize. QAT training
/// forward feeds THESE into the dense matmul, so the loss surface is
/// the one inference realizes.
pub fn q8_ste_dequant(x: &[f32]) -> Vec<f32> {
    let blocks = tick_kernels::bitlinear::quantize_q8_k(x);
    let mut out = Vec::with_capacity(x.len());
    for b in &blocks {
        for j in 0..256 {
            out.push(b.q[j] as f32 * b.d);
        }
    }
    out.truncate(x.len());
    out
}

/// QAT linear forward: y = W · dequant(q8(x)). Returns (y, x_deq) —
/// keep x_deq for the backward (gw = gy ⊗ x_deq: inference's weight
/// gradient sees the dequantized activations).
pub fn qat_lin_fwd(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let xq = q8_ste_dequant(x);
    (lin_fwd(w, rows, cols, &xq), xq)
}

/// y = W x (dense W rows×cols row-major) — forward + input grad.
pub fn lin_fwd(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; rows];
    for r in 0..rows {
        let mut s = 0.0f32;
        for c in 0..cols {
            s += w[r * cols + c] * x[c];
        }
        y[r] = s;
    }
    y
}

/// backward: gx = Wᵀ gy; gw = gy ⊗ x (dense; STE moves it to masters).
pub fn lin_bwd(w: &[f32], rows: usize, cols: usize, x: &[f32], gy: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut gx = vec![0.0f32; cols];
    let mut gw = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let g = gy[r];
        for c in 0..cols {
            gx[c] += w[r * cols + c] * g;
            gw[r * cols + c] = g * x[c];
        }
    }
    (gx, gw)
}

/// NEOX rope forward on one head's 2·half values: returns (y, cos, sin).
pub fn rope_fwd(x: &[f32], pos: usize, base: f32) -> (Vec<f32>, Vec<(f32, f32)>) {
    let hd = x.len();
    let half = hd / 2;
    let theta_scale = base.powf(-2.0 / hd as f32);
    let mut cs = Vec::with_capacity(half);
    let mut theta = pos as f32;
    let mut y = x.to_vec();
    for i in 0..half {
        let (c, s) = (theta.cos(), theta.sin());
        let (x0, x1) = (x[i], x[i + half]);
        y[i] = x0 * c - x1 * s;
        y[i + half] = x0 * s + x1 * c;
        cs.push((c, s));
        theta *= theta_scale;
    }
    (y, cs)
}

/// rope backward (rotation inverse: negate sin).
pub fn rope_bwd(gy: &[f32], cs: &[(f32, f32)]) -> Vec<f32> {
    let hd = gy.len();
    let half = hd / 2;
    let mut gx = vec![0.0f32; hd];
    for i in 0..half {
        let (c, s) = cs[i];
        gx[i] = c * gy[i] + s * gy[i + half];
        gx[i + half] = -s * gy[i] + c * gy[i + half];
    }
    gx
}

/// causal self-attention forward for one layer (all heads), storing the
/// softmax probs for backward. x: [T, hidden] with per-head slices
/// already extracted per head. Returns (out [T, hidden], probs).
pub fn attn_fwd(
    q: &[Vec<f32>], // [T][hd] per head
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    scale: f32,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let t = q.len();
    let hd = q[0].len();
    let mut out = vec![vec![0.0f32; hd]; t];
    let mut probs = vec![Vec::new(); t];
    for i in 0..t {
        let mut s = Vec::with_capacity(i + 1);
        let mut mx = f32::NEG_INFINITY;
        for j in 0..=i {
            let mut d = 0.0f32;
            for c in 0..hd {
                d += q[i][c] * k[j][c];
            }
            s.push(d * scale);
            if s[j] > mx {
                mx = s[j];
            }
        }
        let mut p = vec![0.0f32; i + 1];
        let mut sum = 0.0f64;
        for j in 0..=i {
            p[j] = (s[j] - mx).exp();
            sum += p[j] as f64;
        }
        let inv = 1.0 / sum;
        for j in 0..=i {
            p[j] *= inv as f32;
            for c in 0..hd {
                out[i][c] += p[j] * v[j][c];
            }
        }
        probs[i] = p;
    }
    (out, probs)
}

/// attention backward: gq/gk/gv per position.
#[allow(clippy::type_complexity)]
pub fn attn_bwd(
    q: &[Vec<f32>],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    probs: &[Vec<f32>],
    scale: f32,
    gout: &[Vec<f32>],
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let t = q.len();
    let hd = q[0].len();
    let mut gq = vec![vec![0.0f32; hd]; t];
    let mut gk = vec![vec![0.0f32; hd]; t];
    let mut gv = vec![vec![0.0f32; hd]; t];
    for i in 0..t {
        let n = i + 1;
        // dP = gout · Vᵀ ; dV = Pᵀ gout
        let mut dp = vec![0.0f32; n];
        for j in 0..n {
            let mut d = 0.0f32;
            for c in 0..hd {
                d += gout[i][c] * v[j][c];
            }
            dp[j] = d;
            let p = probs[i][j];
            for c in 0..hd {
                gv[j][c] += p * gout[i][c];
            }
        }
        // softmax bwd: ds_j = p_j (dp_j - Σ dp·p)
        let mut dot = 0.0f32;
        for j in 0..n {
            dot += dp[j] * probs[i][j];
        }
        for j in 0..n {
            let ds = probs[i][j] * (dp[j] - dot) * scale;
            for c in 0..hd {
                gq[i][c] += ds * k[j][c];
                gk[j][c] += ds * q[i][c];
            }
        }
    }
    (gq, gk, gv)
}

/// ReLU² FFN: act = relu(g)² · u. Returns grads (dg, du).
pub fn relu2_bwd(g: &[f32], u: &[f32], gact: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = g.len();
    let mut dg = vec![0.0f32; n];
    let mut du = vec![0.0f32; n];
    for i in 0..n {
        let r = if g[i] > 0.0 { g[i] } else { 0.0 };
        dg[i] = 2.0 * r * u[i] * gact[i];
        du[i] = r * r * gact[i];
    }
    (dg, du)
}

/// cross-entropy forward/backward for one logits row: returns
/// (loss, dlogits) given the target index (softmax + NLL).
pub fn xent_fwd_bwd(logits: &[f32], target: usize) -> (f32, Vec<f32>) {
    let n = logits.len();
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for &l in logits {
        sum += ((l - mx) as f64).exp();
    }
    let lse = sum.ln() as f32 + mx;
    let mut d = vec![0.0f32; n];
    let inv = (1.0 / sum) as f32;
    for (i, &l) in logits.iter().enumerate() {
        let p = ((l - mx) as f64).exp() as f32 * inv;
        d[i] = p;
    }
    d[target] -= 1.0;
    (lse - logits[target], d)
}
