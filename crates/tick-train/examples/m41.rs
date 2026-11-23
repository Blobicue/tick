//! m41 — the real training loop: a tiny transformer, hand-autograd
//! backward, byte-level corpus, SGD on the masters (STE through the
//! ternary view), checkpoints as tickfiles the stock engine reads.

// Parity/training harness: index loops mirror the reference math.
#![allow(clippy::needless_range_loop)]
#![allow(unused_assignments)]
#![allow(dead_code)]
// `!(x < t)` NaN-safe gates are deliberate (law: NaN comparisons are false).
#![allow(clippy::neg_cmp_op_on_partial_ord)]
#![allow(clippy::manual_memcpy)]
fn main() -> Result<(), String> {
    use tick_train::train::*;

    // ---- config: 2 layers, hidden 256, vocab 256 (bytes) ----
    let cfg = TinyConfig {
        n_layer: 2,
        n_head: 4,
        n_head_kv: 1,
        head_dim: 64,
        hidden: 256,
        ffn: 512,
        vocab: 256,
        rms_eps: 1e-5,
        rope_base: 10000.0,
    };
    let corpus = b"the lighthouse keeper named Elias climbed the spiral stair \
and lit the lamp that turned all night, and the ships came home safely \
through the dark water, every single one of them, every night.";
    let seq: Vec<usize> = corpus.iter().map(|&b| b as usize).collect();
    let t_len = seq.len();
    println!(
        "corpus: {t_len} bytes, {} layers hidden {}",
        cfg.n_layer, cfg.hidden
    );

    // ---- training state: float masters + float norms + float embd ----
    let mut rng = Rng { s: 1234 };
    let h = cfg.hidden;
    let kv = cfg.n_head_kv * cfg.head_dim;
    let ff = cfg.ffn;
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

    let lr = 0.002f32;
    let steps = 600;
    let clip = 1.0f32; // global grad-norm clip
    let mut loss0 = 0.0f32;
    let mut mean_loss = 0.0f32;
    for step in 0..steps {
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
                nx.push(y);
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
                sub_in.push(y.clone());
                sub_scale.push(s);
                let o = lin_fwd(&wo_d, h, h, &y);
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
                n2.push(y);
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
                let d = lin_fwd(&down_d, h, ff, &sn);
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
        if step % 50 == 0 || step == steps - 1 {
            println!(
                "step {step}: loss {mean_loss:.4} (byte-ppl {:.2})",
                mean_loss.exp()
            );
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
                let gsn = lin_bwd(&down_d, h, ff, &sn, &gx[t]).0; // input grad only
                let (ga, _) = rms_bwd(&act, &layers[l].ffn_sub_norm, ss, &gsn);
                // down grads
                let gwd = lin_bwd(&down_d, h, ff, &sn, &gx[t]).1;
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
    println!(
        "loss: {loss0:.4} -> {mean_loss:.4} ({:.1}x)",
        loss0 / mean_loss
    );
    if !(mean_loss < loss0 * 0.4) {
        return Err(format!(
            "M4.1 memorize gate failed: loss {loss0:.4} -> {mean_loss:.4}"
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
    let mut m = tick_engine::model::Model::open(std::path::Path::new(out_path), t_len + 8)?;
    let ids: Vec<u32> = seq.iter().map(|&b| b as u32).collect();
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
    if !(engine_nll < mean_loss as f64 + 1.2) || engine_nll.is_nan() {
        return Err(format!(
            "checkpoint consistency failed: engine {engine_nll:.4} vs trainer {mean_loss:.4}"
        ));
    }
    println!("M4.1 GATE: PASS — full backward trains; checkpoint runs in the stock engine");
    Ok(())
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
