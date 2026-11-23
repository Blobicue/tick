//! m42 — activation-quantized QAT on a real corpus: the m41 trainer
//! with q8 STE at every bitlinear input, held-out eval, and
//! best-checkpoint selection. Byte-level, checkpoints as tickfiles.

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
        n_layer: 3,
        n_head: 4,
        n_head_kv: 1,
        head_dim: 64,
        hidden: 256,
        ffn: 512,
        vocab: 256,
        rms_eps: 1e-5,
        rope_base: 10000.0,
    };
    let corpus: &[u8] = CORPUS.as_bytes();
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

    let lr = 0.001f32;
    let steps = 900;
    let warmup = 150usize;
    let _best_val = f32::INFINITY;
    let _best_step = 0usize;
    let clip = 1.0f32; // global grad-norm clip
    let mut loss0 = 0.0f32;
    let mut mean_loss = 0.0f32;
    let n_windows = (train_b.len().saturating_sub(W)) / 40 + 1;
    let mut best_val = f32::INFINITY;
    let mut best_step = 0usize;
    for step in 0..steps {
        let lr = lr * (step.min(warmup) as f32 + 1.0) / (warmup as f32 + 1.0);
        // window selection (round-robin over train windows)
        let w0 = (step % n_windows) * 40;
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
                let (kr, _) = rope_full(&kraw[t], t, cfg.rope_base, cfg.n_head_kv, cfg.head_dim);
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
                    attn_out[t][hq * cfg.head_dim..(hq + 1) * cfg.head_dim].copy_from_slice(&o[t]);
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
        if step == 0 {
            loss0 = mean_loss;
        }
        if step % 100 == 0 || step == steps - 1 {
            let val = eval_val(&layers, &embd_m, &out_norm, &cfg, val_b);
            println!("step {step}: train {mean_loss:.4}  val {val:.4} (uniform 5.545)");
            if val < best_val {
                best_val = val;
                best_step = step;
            }
        }

        // ================= BACKWARD =================
        // global-norm clip on the head gradient keeps the hand-autograd
        // stable (a shared-out embedding + tied head amplify fast)
        {
            let mut g2 = 0.0f64;
            for g in &gnormed {
                for v in g {
                    g2 += (*v * *v) as f64;
                }
            }
            let n = g2.sqrt() as f32;
            if n > clip {
                let s = clip / n;
                for g in gnormed.iter_mut() {
                    for v in g.iter_mut() {
                        *v *= s;
                    }
                }
            }
        }
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
                layers[l].down.ste_update(&gwd, lr);
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
                layers[l].gate.ste_update(&gw, lr);
                gn2[t] = gi;
                let (ui, uw) = lin_bwd(&up_d, ff, h, &n2[t], &gsn_x[t]);
                layers[l].up.ste_update(&uw, lr);
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
                    layers[l].ffn_norm[c] -= lr * gwn[c];
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
                layers[l].wo.ste_update(&gw, lr);
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
                    layers[l].attn_sub_norm[c] -= lr * gwn[c];
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
                layers[l].wq.ste_update(&gw, lr);
                for c in 0..h {
                    gx_prev[t][c] += gi[c];
                }
                let (gi, gw) = lin_bwd(&wk_d, kv, h, &nx[t], &gkraw[t]);
                layers[l].wk.ste_update(&gw, lr);
                for c in 0..h {
                    gx_prev[t][c] += gi[c];
                }
                let (gi, gw) = lin_bwd(&wv_d, kv, h, &nx[t], &gv_full[t]);
                layers[l].wv.ste_update(&gw, lr);
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
                    layers[l].attn_norm[c] -= lr * gwn[c];
                }
                gnext[t] = gxi;
            }
            gx = gnext;
        }
        // embedding grads (scatter into master)
        // NOTE: input embeddings for position t come from x_in[t] = embd[seq[t]];
        // the gradient flowing to gx[0..t_len] maps to embd rows seq[t]
        for t in 0..t_len {
            let tok = seq[t];
            for c in 0..h {
                embd_m[tok * h + c] -= lr * gx[t][c];
            }
        }
        // apply head-embd grads + out_norm grads
        EMBD_GRAD.with(|eg| {
            let mut eg = eg.borrow_mut();
            for i in 0..cfg.vocab * h {
                embd_m[i] -= lr * eg[i];
            }
            eg.fill(0.0);
        });
        OUT_NORM_GRAD.with(|og| {
            let mut og = og.borrow_mut();
            for c in 0..h {
                out_norm[c] -= lr * og[c];
            }
            og.fill(0.0);
        });
    }
    println!("train loss: {loss0:.4} -> {mean_loss:.4}; best val {best_val:.4} @ step {best_step}");
    if !(best_val < 4.0) {
        return Err(format!(
            "M4.2 val gate failed: best val {best_val:.4} >= 4.0"
        ));
    }

    // ---- checkpoint as tickfile; engine-side NLL consistency ----
    let embd16: Vec<u16> = embd_m
        .iter()
        .map(|&f| tick_kernels::bitlinear::f32_to_f16(f))
        .collect();
    let out_path = "/tmp/tiny_m41.tick";
    {
        // reuse write_tiny_tickfile via a shim TinyMats-like structure
        let mats = TinyMats {
            layers: layers
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
            output_norm: out_norm.clone(),
        };
        write_tiny_tickfile(&cfg, &embd16, &norms, &mats, std::path::Path::new(out_path))?;
    }
    // engine NLL over the corpus (teacher-forced bytes)
    let mut m = tick_engine::model::Model::open(std::path::Path::new(out_path), val_b.len() + 8)?;
    let ids: Vec<u32> = val_b.iter().map(|&b| b as u32).collect();
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
    if !(engine_nll < best_val as f64 + 0.35) || engine_nll.is_nan() {
        return Err(format!(
            "checkpoint consistency failed: engine {engine_nll:.4} vs trainer val {best_val:.4}"
        ));
    }
    println!("M4.2 GATE: PASS — activation-QAT training generalizes and the checkpoint runs in the stock engine");
    Ok(())
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

const CORPUS: &str = "the lighthouse keeper named Elias climbed the spiral stair and lit the lamp that turned all night, and the ships came home safely through the dark water, every single one of them, every night. in the morning he wrote the weather in the log book: wind from the north west, clouds low over the water, three sails to the south. the lamp burned four hundred hours on one wick, and the glass was clean because he cleaned it every day at noon. when the fog came the bell rang twice every minute, and the sound went out over the water and came back from the cliffs weaker than it left. elias knew every rock below the cliff by the shape of the water over it, a tall wave where the reef was shallow, flat water where the sand was deep. the supply boat came once a month with flour, lamp oil, tea, and letters from his daughter in the city. she wrote about school and the park in autumn and the horses in the field next to the house. he read every letter three times and answered with short careful lines about the weather, the birds, and the light. in winter the ice came down from the north and the water turned grey and heavy, and the lamp had to burn through the snow that stuck to the glass. he built a small wooden shelter around the lamp room and painted it white in spring. the paint lasted two years. the light itself never failed, not once in thirty years, because he tended it like a living thing, and it answered by burning steady while the storm took the roof off the boat house and the pier and the fish shed. the village on the mainland thought the light was a promise, and it was. the engine room was below the lamp room, and in it a weight of stone on a chain turned the gears that worked the light. elias wound the weight every six hours, in the dark, by feel, because the candle in the engine room guttered in the draft. the gears were older than his father. he oiled them on sunday and listened to them turn, a low even note like a hum, and when the note changed he knew which gear spoke and what it wanted. the chain was the newest thing in the tower, and he watched it for rust the way other men watch the sky. twice a year he took the whole works apart, laid the pieces on a cloth in the order they came off, and put them back before dark. the light burned while he worked, because he moved the burner to the window ledge first. this was against the rules and no one ever came to check. the sea gave up new things after every storm: wood with words carved in it, a blue glass float unbroken, a boot, a door with the paint still on one side. he kept the floats on the sill and let the sea have the rest back. one autumn a goose came down the stair with a hurt wing and lived in the engine room until the wing mended, and after that it came back every year on the same week, and he left grain for it in the corner by the gears. his daughter asked him once why he never came to live in the city, and he wrote back that the light needed him and that the city had lights of its own. she understood this in the way children understand, which is completely and without questions, and she sent him a drawing of the tower with the beam going out in a wide fan over boats she had invented. he pinned it above the log book where the damp could not reach it, and on the long nights between winds he looked at it and thought about the field next to the house and the horses in it, and the light turned and turned and the ships came home. every night he wrote the same first line in the log, weather and wind, and every night the sea answered with a different sea. this was the whole of his learning in thirty years: that the light does not argue with the dark, it only burns; and that the dark, which is endless, moves.";

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
