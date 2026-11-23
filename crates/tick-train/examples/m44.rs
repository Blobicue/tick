//! m44 — the 28M rung: the batched QAT trainer at the default
//! TinyConfig (8 layers, 512 hidden, 1536 ffn) on the wikitext corpus
//! the frozen reference lane itself scores on. Byte-level, evaluated
//! on fixed sampled windows of the held-out split, checkpoint-on-
//! improve, artifact destined for tick-pkg publication.

// Parity/training harness: index loops mirror the reference math.
#![allow(clippy::needless_range_loop)]
#![allow(unused_assignments)]
#![allow(dead_code)]
// `!(x < t)` NaN-safe gates are deliberate (law: NaN comparisons are false).
#![allow(clippy::neg_cmp_op_on_partial_ord)]
#![allow(clippy::manual_memcpy)]
use tick_train::train::*;
struct L {
    attn_norm: Vec<f32>,
    attn_sub_norm: Vec<f32>,
    wq: TrainableTq,
    wk: TrainableTq,
    wv: TrainableTq,
    wo: TrainableTq,
    ffn_norm: Vec<f32>,
    ffn_sub_norm: Vec<f32>,
    gate: TrainableTq,
    up: TrainableTq,
    down: TrainableTq,
}

fn main() -> Result<(), String> {
    use tick_train::train::*;

    // ---- config: 2 layers, hidden 256, vocab 256 (bytes) ----
    let cfg = TinyConfig {
        n_layer: 8,
        n_head: 8,
        n_head_kv: 1,
        head_dim: 64,
        hidden: 512,
        ffn: 1536,
        vocab: 256,
        rms_eps: 1e-5,
        rope_base: 10000.0,
    };
    let corpus_bytes = std::fs::read("/home/juzz/models/bitnet/wiki.test.raw")
        .map_err(|e| format!("corpus: {e}"))?;
    let corpus: &[u8] = &corpus_bytes;
    let split = corpus.len() * 9 / 10;
    let (train_b, val_b) = (&corpus[..split], &corpus[split..]);
    println!(
        "corpus: {} bytes (train {}, val {}), {} layers hidden {}",
        corpus.len(),
        train_b.len(),
        val_b.len(),
        cfg.n_layer,
        cfg.hidden
    );
    // windowed training: windows of 160 bytes, stride 40 over the train split
    const W: usize = 160;

    // ---- training state: float masters + float norms + float embd ----
    let mut rng = Rng { s: 1234 };
    let h = cfg.hidden;
    let kv = cfg.n_head_kv * cfg.head_dim;
    let ff = cfg.ffn;
    let mut mk =
        |r: usize, c: usize| TrainableTq::new(r, c, || rng.normal(0.0, (c as f32).powf(-0.5)));
    let mut layers: Vec<L> = (0..cfg.n_layer)
        .map(|_| L {
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
        })
        .collect();
    // tied embd master (float; quantized to f16 at checkpoint time)
    let mut embd_m: Vec<f32> = (0..cfg.vocab * h)
        .map(|_| rng.normal(0.0, (h as f32).powf(-0.5)))
        .collect();
    let mut out_norm: Vec<f32> = vec![1.0; h];
    EMBD_GRAD.with(|eg| *eg.borrow_mut() = vec![0.0f32; cfg.vocab * h]);
    OUT_NORM_GRAD.with(|og| *og.borrow_mut() = vec![0.0f32; h]);

    // dense views recomputed each step (weight QAT: forward sees t·d)
    let dense = |w: &TrainableTq| dense_of(&w.ternary());

    const B: usize = 8; // windows accumulated per optimizer step
    let lr = 0.0015f32; // curvature-limited, not noise-limited (m43 law)
    let steps: usize = std::env::var("TICK_M44_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let warmup = 80usize;
    let mut best_val = f32::INFINITY;
    let mut best_step = 0usize;
    let mut val_trace: Vec<f32> = Vec::new();
    let _clip = 1.0f32; // global grad-norm clip
    let mut loss0 = 0.0f32;
    let mut mean_loss = 0.0f32;
    let n_windows = (train_b.len().saturating_sub(W)) / 40 + 1;
    // gradient accumulators: filled across the batch, applied once
    struct LAcc {
        wq: Vec<f32>,
        wk: Vec<f32>,
        wv: Vec<f32>,
        wo: Vec<f32>,
        gate: Vec<f32>,
        up: Vec<f32>,
        down: Vec<f32>,
        attn_norm: Vec<f32>,
        attn_sub_norm: Vec<f32>,
        ffn_norm: Vec<f32>,
    }
    let z = |n: usize| vec![0.0f32; n];
    let mut acc: Vec<LAcc> = (0..cfg.n_layer)
        .map(|_| LAcc {
            wq: z(h * h),
            wk: z(kv * h),
            wv: z(kv * h),
            wo: z(h * h),
            gate: z(ff * h),
            up: z(ff * h),
            down: z(h * ff),
            attn_norm: z(h),
            attn_sub_norm: z(h),
            ffn_norm: z(h),
        })
        .collect();
    let mut embd_acc: Vec<f32> = vec![0.0f32; cfg.vocab * h];
    struct Snapshot {
        layers: Vec<L>,
        embd: Vec<f32>,
        out_norm: Vec<f32>,
    }
    let mut snap: Option<Snapshot> = None;
    let mut ema_gn = 0.0f64; // running avg of the full-batch grad norm
    let zero_accs = |acc: &mut Vec<LAcc>, embd_acc: &mut Vec<f32>| {
        for a in acc.iter_mut() {
            for v in a.wq.iter_mut() {
                *v = 0.0;
            }
            for v in a.wk.iter_mut() {
                *v = 0.0;
            }
            for v in a.wv.iter_mut() {
                *v = 0.0;
            }
            for v in a.wo.iter_mut() {
                *v = 0.0;
            }
            for v in a.gate.iter_mut() {
                *v = 0.0;
            }
            for v in a.up.iter_mut() {
                *v = 0.0;
            }
            for v in a.down.iter_mut() {
                *v = 0.0;
            }
            for v in a.attn_norm.iter_mut() {
                *v = 0.0;
            }
            for v in a.attn_sub_norm.iter_mut() {
                *v = 0.0;
            }
            for v in a.ffn_norm.iter_mut() {
                *v = 0.0;
            }
        }
        embd_acc.fill(0.0);
    };
    for step in 0..steps {
        let lr = lr * (step.min(warmup) as f32 + 1.0) / (warmup as f32 + 1.0);
        // NIGHT-1 LAW: fixed 0.0015 past the fast phase is hot — the
        // run only descended once its oscillation damped on its own
        // (2.94 terminal). Halve after step 300 and the descent phase
        // starts immediately instead of ~400 steps later.
        let lr = if step >= 300 { lr * 0.5 } else { lr };
        zero_accs(&mut acc, &mut embd_acc);
        let mut batch_loss = 0.0f64;
        for _bi in 0..B {
            // window selection (round-robin over train windows)
            let w0 = ((step * B + _bi) % n_windows) * 40;
            let wend = (w0 + W).min(train_b.len());
            let seq: Vec<usize> = train_b[w0..wend].iter().map(|&b| b as usize).collect();
            let t_len = seq.len();
            // ================= FORWARD =================
            // x[t] = embd row of seq[t]
            let mut x: Vec<Vec<f32>> = Vec::with_capacity(t_len);
            for &tok in &seq {
                x.push(embd_m[tok * h..(tok + 1) * h].to_vec());
            }
            // per-layer caches
            #[derive(Default)]
            struct Cache {
                x_in: Vec<Vec<f32>>, // the layer input (pre-norm)
                nx: Vec<Vec<f32>>,   // normed input
                nscale: Vec<f32>,
                qw: Vec<Vec<f32>>,
                kw: Vec<Vec<f32>>,
                vw: Vec<Vec<f32>>,         // roped
                qcs: Vec<Vec<(f32, f32)>>, // rope cos/sin per pos per head? store per pos concatenated
                probs: Vec<Vec<f32>>,      // per head flattened? — keep per head below
                attn_out: Vec<Vec<f32>>,   // concat heads
                sub_in: Vec<Vec<f32>>,     // attn output (pre sub_norm)
                sub_scale: Vec<f32>,
                ao: Vec<Vec<f32>>, // wo(attn_out)
                fi: Vec<Vec<f32>>, // ffn_inp = x + ao
                n2: Vec<Vec<f32>>,
                n2s: Vec<f32>,
                gv: Vec<Vec<f32>>,
                uv: Vec<Vec<f32>>, // gate/up outputs
            }
            let mut caches: Vec<Cache> = Vec::new();
            let scale_attn = 1.0f32 / (cfg.head_dim as f32).sqrt();
            for l in 0..cfg.n_layer {
                let L {
                    attn_norm,
                    attn_sub_norm,
                    wq,
                    wk,
                    wv,
                    ..
                } = &layers[l];
                let wq_d = dense(wq);
                let wk_d = dense(wk);
                let wv_d = dense(wv);
                let mut nx = Vec::with_capacity(t_len);
                let mut nscale = Vec::with_capacity(t_len);
                for t in 0..t_len {
                    let (y, s) = rms_fwd(&x[t], attn_norm, cfg.rms_eps);
                    // QAT: the engine quantizes each bitlinear input to q8;
                    // train on the dequantized values (STE backward)
                    nx.push(q8_ste_dequant(&y));
                    nscale.push(s);
                }
                // qkv projections
                let mut qraw = Vec::with_capacity(t_len);
                let mut kraw = Vec::with_capacity(t_len);
                let mut vraw = Vec::with_capacity(t_len);
                for t in 0..t_len {
                    qraw.push(lin_fwd(&wq_d, h, h, &nx[t]));
                    kraw.push(lin_fwd(&wk_d, kv, h, &nx[t]));
                    vraw.push(lin_fwd(&wv_d, kv, h, &nx[t]));
                }
                // rope on q and k (full vector; layout matches engine: per
                // head, pairs (i, i+hd/2) within each head slice)
                let mut qw = Vec::with_capacity(t_len);
                let mut kw = Vec::with_capacity(t_len);
                let mut qcs_all: Vec<Vec<(f32, f32)>> = Vec::with_capacity(t_len);
                for t in 0..t_len {
                    let (qr, cs) = rope_full(&qraw[t], t, cfg.rope_base, cfg.n_head, cfg.head_dim);
                    let (kr, _) =
                        rope_full(&kraw[t], t, cfg.rope_base, cfg.n_head_kv, cfg.head_dim);
                    qw.push(qr);
                    kw.push(kr);
                    qcs_all.push(cs);
                }
                // attention per q head (gqa)
                let n_head = cfg.n_head;
                let gqa = n_head / cfg.n_head_kv;
                let mut attn_out = vec![vec![0.0f32; h]; t_len];
                let mut all_probs: Vec<Vec<f32>> = vec![Vec::new(); t_len * n_head];
                for hq in 0..n_head {
                    let kvh = hq / gqa;
                    let qs: Vec<Vec<f32>> = (0..t_len)
                        .map(|t| qw[t][hq * cfg.head_dim..(hq + 1) * cfg.head_dim].to_vec())
                        .collect();
                    let ks: Vec<Vec<f32>> = (0..t_len)
                        .map(|t| kw[t][kvh * cfg.head_dim..(kvh + 1) * cfg.head_dim].to_vec())
                        .collect();
                    let vs: Vec<Vec<f32>> = (0..t_len)
                        .map(|t| vraw[t][kvh * cfg.head_dim..(kvh + 1) * cfg.head_dim].to_vec())
                        .collect();
                    let (o, p) = attn_fwd(&qs, &ks, &vs, scale_attn);
                    for t in 0..t_len {
                        attn_out[t][hq * cfg.head_dim..(hq + 1) * cfg.head_dim]
                            .copy_from_slice(&o[t]);
                        all_probs[t * n_head + hq] = p[t].clone();
                    }
                }
                // sub_norm -> wo -> residual
                let wo_d = dense(&layers[l].wo);
                let mut sub_in = Vec::with_capacity(t_len);
                let mut sub_scale = Vec::with_capacity(t_len);
                let mut ao = Vec::with_capacity(t_len);
                let mut fi = Vec::with_capacity(t_len);
                for t in 0..t_len {
                    let (y, s) = rms_fwd(&attn_out[t], attn_sub_norm, cfg.rms_eps);
                    sub_in.push(q8_ste_dequant(&y));
                    sub_scale.push(s);
                    let o = lin_fwd(&wo_d, h, h, &sub_in[t]);
                    ao.push(o.clone());
                    fi.push(
                        x[t].iter()
                            .zip(&o)
                            .map(|(a, b)| a + b)
                            .collect::<Vec<f32>>(),
                    );
                }
                // ffn
                let ffn_norm = &layers[l].ffn_norm;
                let ffn_sub_norm = &layers[l].ffn_sub_norm;
                let mut n2 = Vec::with_capacity(t_len);
                let mut n2s = Vec::with_capacity(t_len);
                for t in 0..t_len {
                    let (y, s) = rms_fwd(&fi[t], ffn_norm, cfg.rms_eps);
                    n2.push(q8_ste_dequant(&y));
                    n2s.push(s);
                }
                let gate_d = dense(&layers[l].gate);
                let up_d = dense(&layers[l].up);
                let mut gv = Vec::with_capacity(t_len);
                let mut uv = Vec::with_capacity(t_len);
                for t in 0..t_len {
                    gv.push(lin_fwd(&gate_d, ff, h, &n2[t]));
                    uv.push(lin_fwd(&up_d, ff, h, &n2[t]));
                }
                // relu2 + subnorm + down applied lazily in the loss pass below
                let x_in = x.clone();
                caches.push(Cache {
                    x_in,
                    nx,
                    nscale,
                    qw,
                    kw,
                    vw: vraw,
                    qcs: qcs_all,
                    probs: all_probs,
                    attn_out,
                    sub_in,
                    sub_scale,
                    ao,
                    fi: fi.clone(),
                    n2,
                    n2s,
                    gv,
                    uv,
                });
                // update x for the next layer: relu2 -> subnorm -> down -> +fi
                let down_d = dense(&layers[l].down);
                let mut nx2 = Vec::with_capacity(t_len);
                for t in 0..t_len {
                    let mut act = vec![0.0f32; ff];
                    for i in 0..ff {
                        let r = if caches[l].gv[t][i] > 0.0 {
                            caches[l].gv[t][i]
                        } else {
                            0.0
                        };
                        act[i] = r * r * caches[l].uv[t][i];
                    }
                    let (sn, _) = rms_fwd(&act, ffn_sub_norm, cfg.rms_eps);
                    let snq = q8_ste_dequant(&sn);
                    let d = lin_fwd(&down_d, h, ff, &snq);
                    nx2.push(
                        fi[t]
                            .iter()
                            .zip(&d)
                            .map(|(a, b)| a + b)
                            .collect::<Vec<f32>>(),
                    );
                }
                x = nx2;
            }
            // head: logits = embd · rms_norm(x_T) for every position
            let mut losses = Vec::with_capacity(t_len - 1);
            let mut gnormed: Vec<Vec<f32>> = Vec::with_capacity(t_len - 1);
            let mut normed_rows: Vec<Vec<f32>> = Vec::with_capacity(t_len - 1);
            let mut nscale_head = Vec::with_capacity(t_len - 1);
            let mut total = 0.0f64;
            for t in 0..t_len - 1 {
                let (ny, s) = rms_fwd(&x[t], &out_norm, cfg.rms_eps);
                let logits = lin_fwd(&embd_m, cfg.vocab, h, &ny);
                let (loss, dl) = xent_fwd_bwd(&logits, seq[t + 1]);
                losses.push(loss);
                total += loss as f64;
                // gnormed = embdᵀ · dl
                let mut gn = vec![0.0f32; h];
                for v in 0..cfg.vocab {
                    let g = dl[v];
                    if g != 0.0 {
                        for c in 0..h {
                            gn[c] += embd_m[v * h + c] * g;
                        }
                    }
                }
                gnormed.push(gn);
                normed_rows.push(ny.clone());
                nscale_head.push(s);
                // grad on embd rows (dense; f16 STE at apply)
                for v in 0..cfg.vocab {
                    let g = dl[v];
                    if g != 0.0 {
                        for c in 0..h {
                            EMBD_GRAD.with(|eg| eg.borrow_mut()[v * h + c] += g * ny[c]);
                        }
                    }
                }
            }
            mean_loss = (total / (t_len - 1) as f64) as f32;
            if step == 0 && _bi == 0 {
                loss0 = mean_loss;
            }
            batch_loss += mean_loss as f64;

            // ================= BACKWARD =================
            // (head-grad clipping moved to apply time: the FULL gradient —
            // matrices, norms, and both embd paths — is clipped as one)
            // gx at the head
            let mut gx: Vec<Vec<f32>> = vec![vec![0.0f32; h]; t_len];
            for t in 0..t_len - 1 {
                let (gxi, _) = rms_bwd(&x[t], &out_norm, nscale_head[t], &gnormed[t]);
                gx[t] = gxi;
                OUT_NORM_GRAD.with(|og| {
                    let mut og = og.borrow_mut();
                    for c in 0..h {
                        og[c] += gnormed[t][c] * x[t][c] * nscale_head[t];
                    }
                });
            }
            // walk layers in reverse
            for l in (0..cfg.n_layer).rev() {
                let Cache {
                    nx,
                    nscale,
                    qw,
                    kw,
                    vw,
                    qcs,
                    probs,
                    attn_out,
                    sub_in,
                    sub_scale,
                    ao: _,
                    fi,
                    n2,
                    n2s,
                    gv,
                    uv,
                    ..
                } = &caches[l];
                // --- ffn backward ---
                let down_d = dense(&layers[l].down);
                let mut gact_snorm_in = vec![vec![0.0f32; ff]; t_len];
                let mut gsn_scale = Vec::new();
                let mut gsn_x = vec![vec![0.0f32; ff]; t_len];
                for t in 0..t_len {
                    // recompute sn (activation sub-norm out) to have its cache
                    let mut act = vec![0.0f32; ff];
                    for i in 0..ff {
                        let r = if gv[t][i] > 0.0 { gv[t][i] } else { 0.0 };
                        act[i] = r * r * uv[t][i];
                    }
                    let (sn, ss) = rms_fwd(&act, &layers[l].ffn_sub_norm, cfg.rms_eps);
                    let snq = q8_ste_dequant(&sn);
                    let gsn = lin_bwd(&down_d, h, ff, &snq, &gx[t]).0; // input grad only (STE: identity through q8)
                    let (ga, _) = rms_bwd(&act, &layers[l].ffn_sub_norm, ss, &gsn);
                    // down grads (weight grad sees the dequantized input)
                    let gwd = lin_bwd(&down_d, h, ff, &snq, &gx[t]).1;
                    for i in 0..gwd.len() {
                        acc[l].down[i] += gwd[i];
                    }
                    // relu2
                    let (gg, gu) = relu2_bwd(&gv[t], &uv[t], &ga);
                    gact_snorm_in[t] = gg;
                    gsn_x[t] = gu;
                    gsn_scale.push(ss);
                }
                let gate_d = dense(&layers[l].gate);
                let up_d = dense(&layers[l].up);
                let mut gn2 = vec![vec![0.0f32; h]; t_len];
                for t in 0..t_len {
                    let (gi, gw) = lin_bwd(&gate_d, ff, h, &n2[t], &gact_snorm_in[t]);
                    for i in 0..gw.len() {
                        acc[l].gate[i] += gw[i];
                    }
                    gn2[t] = gi;
                    let (ui, uw) = lin_bwd(&up_d, ff, h, &n2[t], &gsn_x[t]);
                    for i in 0..uw.len() {
                        acc[l].up[i] += uw[i];
                    }
                    for c in 0..h {
                        gn2[t][c] += ui[c];
                    }
                }
                // ffn_norm + residual split
                let mut gfi = vec![vec![0.0f32; h]; t_len];
                for t in 0..t_len {
                    let (gxi, gwn) = rms_bwd(&fi[t], &layers[l].ffn_norm, n2s[t], &gn2[t]);
                    gfi[t] = gxi;
                    for c in 0..h {
                        acc[l].ffn_norm[c] += gwn[c];
                    }
                }
                // gx flowing through residual: gx_prev = gfi (from x) + gao
                let mut gx_prev = vec![vec![0.0f32; h]; t_len];
                let mut gao = vec![vec![0.0f32; h]; t_len];
                for t in 0..t_len {
                    gx_prev[t] = gfi[t].clone();
                    gao[t] = gfi[t].clone(); // fi = x + ao  => both get gfi
                }
                // wo backward
                let wo_d = dense(&layers[l].wo);
                let mut gsub_in = vec![vec![0.0f32; h]; t_len];
                for t in 0..t_len {
                    let (gi, gw) = lin_bwd(&wo_d, h, h, &sub_in[t], &gao[t]);
                    gsub_in[t] = gi;
                    for i in 0..gw.len() {
                        acc[l].wo[i] += gw[i];
                    }
                }
                // sub_norm backward
                let mut gattn = vec![vec![0.0f32; h]; t_len];
                for t in 0..t_len {
                    let (ga, gwn) = rms_bwd(
                        &attn_out[t],
                        &layers[l].attn_sub_norm,
                        sub_scale[t],
                        &gsub_in[t],
                    );
                    gattn[t] = ga;
                    for c in 0..h {
                        acc[l].attn_sub_norm[c] += gwn[c];
                    }
                }
                // attention backward per head
                let n_head = cfg.n_head;
                let gqa = n_head / cfg.n_head_kv;
                let mut gq_full = vec![vec![0.0f32; h]; t_len];
                let mut gk_full = vec![vec![0.0f32; kv]; t_len];
                let mut gv_full = vec![vec![0.0f32; kv]; t_len];
                for hq in 0..n_head {
                    let kvh = hq / gqa;
                    let qs: Vec<Vec<f32>> = (0..t_len)
                        .map(|t| qw[t][hq * cfg.head_dim..(hq + 1) * cfg.head_dim].to_vec())
                        .collect();
                    let ks: Vec<Vec<f32>> = (0..t_len)
                        .map(|t| kw[t][kvh * cfg.head_dim..(kvh + 1) * cfg.head_dim].to_vec())
                        .collect();
                    let vs: Vec<Vec<f32>> = (0..t_len)
                        .map(|t| vw[t][kvh * cfg.head_dim..(kvh + 1) * cfg.head_dim].to_vec())
                        .collect();
                    let ps: Vec<Vec<f32>> =
                        (0..t_len).map(|t| probs[t * n_head + hq].clone()).collect();
                    let go: Vec<Vec<f32>> = (0..t_len)
                        .map(|t| gattn[t][hq * cfg.head_dim..(hq + 1) * cfg.head_dim].to_vec())
                        .collect();
                    let (gq, gk, gvv) = attn_bwd(&qs, &ks, &vs, &ps, scale_attn, &go);
                    for t in 0..t_len {
                        for c in 0..cfg.head_dim {
                            gq_full[t][hq * cfg.head_dim + c] = gq[t][c];
                            gk_full[t][kvh * cfg.head_dim + c] += gk[t][c];
                            gv_full[t][kvh * cfg.head_dim + c] += gvv[t][c];
                        }
                    }
                }
                // rope backward
                let mut gqraw = vec![vec![0.0f32; h]; t_len];
                let mut gkraw = vec![vec![0.0f32; kv]; t_len];
                for t in 0..t_len {
                    let gqr = rope_full_bwd(&gq_full[t], &qcs[t], cfg.n_head, cfg.head_dim);
                    let gkr = rope_full_bwd(&gk_full[t], &qcs[t], cfg.n_head_kv, cfg.head_dim);
                    gqraw[t] = gqr;
                    gkraw[t] = gkr;
                }
                // qkv projections backward
                let wq_d = dense(&layers[l].wq);
                let wk_d = dense(&layers[l].wk);
                let wv_d = dense(&layers[l].wv);
                for t in 0..t_len {
                    let (gi, gw) = lin_bwd(&wq_d, h, h, &nx[t], &gqraw[t]);
                    for i in 0..gw.len() {
                        acc[l].wq[i] += gw[i];
                    }
                    for c in 0..h {
                        gx_prev[t][c] += gi[c];
                    }
                    let (gi, gw) = lin_bwd(&wk_d, kv, h, &nx[t], &gkraw[t]);
                    for i in 0..gw.len() {
                        acc[l].wk[i] += gw[i];
                    }
                    for c in 0..h {
                        gx_prev[t][c] += gi[c];
                    }
                    let (gi, gw) = lin_bwd(&wv_d, kv, h, &nx[t], &gv_full[t]);
                    for i in 0..gw.len() {
                        acc[l].wv[i] += gw[i];
                    }
                    for c in 0..h {
                        gx_prev[t][c] += gi[c];
                    }
                }
                // attn_norm backward + gx for the layer below
                let mut gnext = vec![vec![0.0f32; h]; t_len];
                for t in 0..t_len {
                    let (gxi, gwn) = rms_bwd(
                        &caches[l].x_in[t],
                        &layers[l].attn_norm,
                        nscale[t],
                        &gx_prev[t],
                    );
                    for c in 0..h {
                        acc[l].attn_norm[c] += gwn[c];
                    }
                    gnext[t] = gxi;
                }
                gx = gnext;
            }
            // embedding input grads: scatter into the accumulator
            for t in 0..t_len {
                let tok = seq[t];
                for c in 0..h {
                    embd_acc[tok * h + c] += gx[t][c];
                }
            }
        } // end batch (B windows accumulated)

        mean_loss = (batch_loss / B as f64) as f32;
        // ---- apply the averaged batch gradient under a FULL global-norm
        // clip (m41/m42 clipped only the backbone grad; the head-side
        // EMBD_GRAD was never clipped and spiked the tied embedding) ----
        let mut g2 = 0.0f64;
        for a in acc.iter() {
            for buf in [
                &a.wq,
                &a.wk,
                &a.wv,
                &a.wo,
                &a.gate,
                &a.up,
                &a.down,
                &a.attn_norm,
                &a.attn_sub_norm,
                &a.ffn_norm,
            ] {
                for v in buf.iter() {
                    g2 += (*v * *v) as f64;
                }
            }
        }
        EMBD_GRAD.with(|eg| {
            for v in eg.borrow().iter() {
                g2 += (*v * *v) as f64;
            }
        });
        OUT_NORM_GRAD.with(|og| {
            for v in og.borrow().iter() {
                g2 += (*v * *v) as f64;
            }
        });
        for v in embd_acc.iter() {
            g2 += (*v * *v) as f64;
        }
        let gn = g2.sqrt();
        // spike clip: bound only steps whose global norm exceeds 2x the
        // running average — steady progress runs unscaled, catastrophic
        // updates (the 4-nat val jumps) get tamed
        let thresh = 2.0 * ema_gn;
        let gscale = if thresh > 0.0 && gn > thresh {
            thresh / gn
        } else {
            1.0
        };
        ema_gn = if ema_gn == 0.0 {
            gn
        } else {
            0.9 * ema_gn + 0.1 * gn
        };
        let inv_b = (1.0f32 / B as f32) * gscale as f32;
        for l in 0..cfg.n_layer {
            {
                let a = &mut acc[l];
                for buf in [
                    &mut a.wq,
                    &mut a.wk,
                    &mut a.wv,
                    &mut a.wo,
                    &mut a.gate,
                    &mut a.up,
                    &mut a.down,
                ] {
                    for v in buf.iter_mut() {
                        *v *= inv_b;
                    }
                }
            }
            layers[l].wq.ste_update(&acc[l].wq, lr);
            layers[l].wk.ste_update(&acc[l].wk, lr);
            layers[l].wv.ste_update(&acc[l].wv, lr);
            layers[l].wo.ste_update(&acc[l].wo, lr);
            layers[l].gate.ste_update(&acc[l].gate, lr);
            layers[l].up.ste_update(&acc[l].up, lr);
            layers[l].down.ste_update(&acc[l].down, lr);
            for c in 0..h {
                layers[l].attn_norm[c] -= lr * acc[l].attn_norm[c] * inv_b;
                layers[l].attn_sub_norm[c] -= lr * acc[l].attn_sub_norm[c] * inv_b;
                layers[l].ffn_norm[c] -= lr * acc[l].ffn_norm[c] * inv_b;
            }
        }
        // head-embd + input-embd + out_norm grads (averaged)
        EMBD_GRAD.with(|eg| {
            let mut eg = eg.borrow_mut();
            for i in 0..cfg.vocab * h {
                let g = (eg[i] + embd_acc[i]) * inv_b;
                embd_m[i] -= lr * g;
            }
            eg.fill(0.0);
        });
        OUT_NORM_GRAD.with(|og| {
            let mut og = og.borrow_mut();
            for c in 0..h {
                out_norm[c] -= lr * og[c] * inv_b;
            }
            og.fill(0.0);
        });

        // ---- eval + checkpoint-on-improve ----
        if step % 40 == 0 || step == steps - 1 {
            let val = eval_sampled(&layers, &embd_m, &out_norm, &cfg, val_b, 8, 256);
            println!(
                "step {step}: train {mean_loss:.4}  val {val:.4}  gn {gn:.2} (clip x{gscale:.2}) (uniform 5.545)"
            );
            val_trace.push(val);
            if val < best_val {
                best_val = val;
                best_step = step;
                snap = Some(Snapshot {
                    layers: layers
                        .iter()
                        .map(|l| L {
                            attn_norm: l.attn_norm.clone(),
                            attn_sub_norm: l.attn_sub_norm.clone(),
                            wq: l.wq.clone_view(),
                            wk: l.wk.clone_view(),
                            wv: l.wv.clone_view(),
                            wo: l.wo.clone_view(),
                            ffn_norm: l.ffn_norm.clone(),
                            ffn_sub_norm: l.ffn_sub_norm.clone(),
                            gate: l.gate.clone_view(),
                            up: l.up.clone_view(),
                            down: l.down.clone_view(),
                        })
                        .collect(),
                    embd: embd_m.clone(),
                    out_norm: out_norm.clone(),
                });
            }
        }
    }
    println!("train loss: {loss0:.4} -> {mean_loss:.4}; best val {best_val:.4} @ step {best_step}");
    // NIGHT-1 LAW (the hard way): gates REPORT, they never abort before
    // the artifact work — night 1's best checkpoint died in memory when
    // the val gate returned early and the tickfile was never written.
    let val_ok = best_val < 2.5;
    // post-warmup oscillation gate. The first ceil(warmup/25)+1 = 2
    // evals sit on the warmup hill (a one-way passage both m42 and m43
    // show); oscillation proper is DEGRADATION after descent begins.
    // m42's single-window run moved +1.46 nats uphill mid-run; a healthy
    // descent never degrades by >= 0.5.
    let warmup_evals = warmup.div_ceil(40) + 1;
    let max_up = val_trace
        .windows(2)
        .skip(warmup_evals.saturating_sub(1))
        .map(|w| w[1] - w[0])
        .filter(|d| *d > 0.0)
        .fold(0.0f32, f32::max);
    println!(
        "val trace: [{}]  max post-warmup degradation {max_up:.3} (gate < 0.5; m42 single-window degraded +1.46 mid-run)",
        val_trace.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>().join(", ")
    );
    let osc_ok = max_up < 0.5;
    let snap = snap.ok_or("no checkpoint snapshot was taken")?;
    let snap_val = eval_sampled(
        &snap.layers,
        &snap.embd,
        &snap.out_norm,
        &cfg,
        val_b,
        8,
        256,
    );
    let final_val = eval_sampled(&layers, &embd_m, &out_norm, &cfg, val_b, 8, 256);
    println!(
        "snapshot val {snap_val:.4} (best {best_val:.4}); final-state val {final_val:.4}{}",
        if final_val > best_val + 1e-4 {
            " — snapshot was load-bearing"
        } else {
            ""
        }
    );
    let snap_ok = (snap_val - best_val).abs() <= 1e-4;

    // ---- checkpoint as tickfile FROM THE BEST-VAL SNAPSHOT; engine NLL ----
    let embd16: Vec<u16> = snap
        .embd
        .iter()
        .map(|&f| tick_kernels::bitlinear::f32_to_f16(f))
        .collect();
    let out_path = "/tmp/m44_night2.tick";
    {
        // reuse write_tiny_tickfile via a shim TinyMats-like structure
        let mats = TinyMats {
            layers: snap
                .layers
                .iter()
                .map(|l| TinyLayerMats {
                    attn_norm: l.attn_norm.clone(),
                    attn_sub_norm: l.attn_sub_norm.clone(),
                    wq: l.wq.clone_view(),
                    wk: l.wk.clone_view(),
                    wv: l.wv.clone_view(),
                    wo: l.wo.clone_view(),
                    ffn_norm: l.ffn_norm.clone(),
                    ffn_sub_norm: l.ffn_sub_norm.clone(),
                    gate: l.gate.clone_view(),
                    up: l.up.clone_view(),
                    down: l.down.clone_view(),
                })
                .collect(),
        };
        let norms = TinyNorms {
            output_norm: snap.out_norm.clone(),
        };
        write_tiny_tickfile(&cfg, &embd16, &norms, &mats, std::path::Path::new(out_path))?;
    }
    // engine NLL over the corpus (teacher-forced bytes)
    let probe = &val_b[..256];
    let mut m = tick_engine::model::Model::open(std::path::Path::new(out_path), probe.len() + 8)?;
    let ids: Vec<u32> = probe.iter().map(|&b| b as u32).collect();
    // greedy_through is not public; use forward_token per position and
    // logits on the fly
    let mut hidden = None;
    let mut nll = 0.0f64;
    let mut n = 0usize;
    for i in 0..ids.len() - 1 {
        hidden = Some(m.forward_token(ids[i] as usize, i)?);
        let hn = hidden.unwrap();
        let ny = tick_engine::model::rms_norm(&hn, &m.embd_norm, m.cfg.rms_eps);
        let logits = m.logits(&ny);
        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f64;
        for &l in &logits {
            sum += ((l - mx) as f64).exp();
        }
        let lse = sum.ln() + mx as f64;
        nll += lse - logits[ids[i + 1] as usize] as f64;
        n += 1;
    }
    let engine_nll = nll / n as f64;
    println!("engine NLL on the checkpoint: {engine_nll:.4} (trainer {mean_loss:.4})");
    let engine_ok = !(engine_nll < best_val as f64 + 0.35) || engine_nll.is_nan();
    println!(
        "gate: val<2.5             {} (best {best_val:.4})",
        verdict(val_ok)
    );
    println!(
        "gate: post-warmup deg<0.5 {} ({max_up:.3})",
        verdict(osc_ok)
    );
    println!(
        "gate: snapshot integrity  {} ({snap_val:.4} vs {best_val:.4})",
        verdict(snap_ok)
    );
    println!(
        "gate: engine consistency  {} ({engine_nll:.4} vs {best_val:.4}+0.35)",
        verdict(!engine_ok)
    );
    println!("artifact: {out_path}");
    let engine_pass = !engine_ok;
    if val_ok && osc_ok && snap_ok && engine_pass {
        println!("M4.4 GATE: PASS — the 28M model trains on the reference corpus and the artifact is the best-val snapshot");
        Ok(())
    } else {
        Err(format!(
            "M4.4 VERDICT: FAIL — artifact still emitted at {out_path}; gates: val={val_ok} osc={osc_ok} snap={snap_ok} engine={engine_pass}"
        ))
    }
}

fn verdict(ok: bool) -> &'static str {
    if ok {
        "PASS"
    } else {
        "FAIL"
    }
}

/// teacher-forced NLL of the val split through the SAME q8-quantized
/// forward the trainer uses (no greedy, no state).
fn eval_val(layers: &[L], embd_m: &[f32], out_norm: &[f32], cfg: &TinyConfig, val: &[u8]) -> f32 {
    let seq: Vec<usize> = val.iter().map(|&b| b as usize).collect();
    let t_len = seq.len();
    let h = cfg.hidden;
    let mut x: Vec<Vec<f32>> = Vec::with_capacity(t_len);
    for &tok in &seq {
        x.push(embd_m[tok * h..(tok + 1) * h].to_vec());
    }
    for l in layers {
        let wq_d = dense_of(&l.wq.ternary());
        let wk_d = dense_of(&l.wk.ternary());
        let wv_d = dense_of(&l.wv.ternary());
        let wo_d = dense_of(&l.wo.ternary());
        let gate_d = dense_of(&l.gate.ternary());
        let up_d = dense_of(&l.up.ternary());
        let down_d = dense_of(&l.down.ternary());
        let mut nx = Vec::with_capacity(t_len);
        for t in 0..t_len {
            let (y, _) = rms_fwd(&x[t], &l.attn_norm, cfg.rms_eps);
            nx.push(q8_ste_dequant(&y));
        }
        let kv = cfg.n_head_kv * cfg.head_dim;
        let mut qw = Vec::with_capacity(t_len);
        let mut kw = Vec::with_capacity(t_len);
        let mut vraw = Vec::with_capacity(t_len);
        for t in 0..t_len {
            let qr = lin_fwd(&wq_d, h, h, &nx[t]);
            let kr = lin_fwd(&wk_d, kv, h, &nx[t]);
            let vr = lin_fwd(&wv_d, kv, h, &nx[t]);
            let (qr_r, _) = rope_full(&qr, t, cfg.rope_base, cfg.n_head, cfg.head_dim);
            let (kr_r, _) = rope_full(&kr, t, cfg.rope_base, cfg.n_head_kv, cfg.head_dim);
            qw.push(qr_r);
            kw.push(kr_r);
            vraw.push(vr);
        }
        let n_head = cfg.n_head;
        let gqa = n_head / cfg.n_head_kv;
        let scale_attn = 1.0f32 / (cfg.head_dim as f32).sqrt();
        let mut attn_out = vec![vec![0.0f32; h]; t_len];
        for hq in 0..n_head {
            let kvh = hq / gqa;
            let qs: Vec<Vec<f32>> = (0..t_len)
                .map(|t| qw[t][hq * cfg.head_dim..(hq + 1) * cfg.head_dim].to_vec())
                .collect();
            let ks: Vec<Vec<f32>> = (0..t_len)
                .map(|t| kw[t][kvh * cfg.head_dim..(kvh + 1) * cfg.head_dim].to_vec())
                .collect();
            let vs: Vec<Vec<f32>> = (0..t_len)
                .map(|t| vraw[t][kvh * cfg.head_dim..(kvh + 1) * cfg.head_dim].to_vec())
                .collect();
            let (o, _) = attn_fwd(&qs, &ks, &vs, scale_attn);
            for t in 0..t_len {
                attn_out[t][hq * cfg.head_dim..(hq + 1) * cfg.head_dim].copy_from_slice(&o[t]);
            }
        }
        let mut fi = Vec::with_capacity(t_len);
        for t in 0..t_len {
            let (y, _) = rms_fwd(&attn_out[t], &l.attn_sub_norm, cfg.rms_eps);
            let yq = q8_ste_dequant(&y);
            let o = lin_fwd(&wo_d, h, h, &yq);
            fi.push(
                x[t].iter()
                    .zip(&o)
                    .map(|(a, b)| a + b)
                    .collect::<Vec<f32>>(),
            );
        }
        let mut nx2 = Vec::with_capacity(t_len);
        for t in 0..t_len {
            let (y, _) = rms_fwd(&fi[t], &l.ffn_norm, cfg.rms_eps);
            let yq = q8_ste_dequant(&y);
            let gv = lin_fwd(&gate_d, cfg.ffn, h, &yq);
            let uv = lin_fwd(&up_d, cfg.ffn, h, &yq);
            let mut act = vec![0.0f32; cfg.ffn];
            for i in 0..cfg.ffn {
                let r = if gv[i] > 0.0 { gv[i] } else { 0.0 };
                act[i] = r * r * uv[i];
            }
            let (sn, _) = rms_fwd(&act, &l.ffn_sub_norm, cfg.rms_eps);
            let snq = q8_ste_dequant(&sn);
            let d = lin_fwd(&down_d, h, cfg.ffn, &snq);
            nx2.push(
                fi[t]
                    .iter()
                    .zip(&d)
                    .map(|(a, b)| a + b)
                    .collect::<Vec<f32>>(),
            );
        }
        x = nx2;
    }
    let mut total = 0.0f64;
    for t in 0..t_len - 1 {
        let (ny, _) = rms_fwd(&x[t], out_norm, cfg.rms_eps);
        let logits = lin_fwd(embd_m, cfg.vocab, h, &ny);
        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f64;
        for &li in &logits {
            sum += ((li - mx) as f64).exp();
        }
        let lse = sum.ln() + mx as f64;
        total += lse - logits[seq[t + 1]] as f64;
    }
    (total / (t_len - 1) as f64) as f32
}

// --- shims to keep the example readable ---

thread_local! {
    static EMBD_GRAD: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    static OUT_NORM_GRAD: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// rope over the full multi-head vector (per-head slices).
fn rope_full(
    x: &[f32],
    pos: usize,
    base: f32,
    n_head: usize,
    hd: usize,
) -> (Vec<f32>, Vec<(f32, f32)>) {
    let mut y = x.to_vec();
    let half = hd / 2;
    let theta_scale = base.powf(-2.0 / hd as f32);
    let mut cs = Vec::with_capacity(half);
    let mut theta = pos as f32;
    for _i in 0..half {
        let (c, s) = (theta.cos(), theta.sin());
        cs.push((c, s));
        theta *= theta_scale;
    }
    for h in 0..n_head {
        let b = h * hd;
        for i in 0..half {
            let (c, s) = cs[i];
            let (x0, x1) = (x[b + i], x[b + i + half]);
            y[b + i] = x0 * c - x1 * s;
            y[b + i + half] = x0 * s + x1 * c;
        }
    }
    (y, cs)
}

fn rope_full_bwd(gy: &[f32], cs: &[(f32, f32)], n_head: usize, hd: usize) -> Vec<f32> {
    let mut gx = gy.to_vec();
    let half = hd / 2;
    for h in 0..n_head {
        let b = h * hd;
        for i in 0..half {
            let (c, s) = cs[i];
            gx[b + i] = c * gy[b + i] + s * gy[b + i + half];
            gx[b + i + half] = -s * gy[b + i] + c * gy[b + i + half];
        }
    }
    gx
}

/// Teacher-forced NLL over n FIXED, evenly-spaced windows of the val
/// split (deterministic across evals; the full 129KB tail is too slow
/// to score every 25 steps at this size).
fn eval_sampled(
    layers: &[L],
    embd_m: &[f32],
    out_norm: &[f32],
    cfg: &TinyConfig,
    val: &[u8],
    n: usize,
    w: usize,
) -> f32 {
    let mut total = 0.0f64;
    for i in 0..n {
        let start = (i * (val.len().saturating_sub(w))) / n.max(1);
        let win = &val[start..start + w];
        total += eval_val(layers, embd_m, out_norm, cfg, win) as f64;
    }
    (total / n as f64) as f32
}
