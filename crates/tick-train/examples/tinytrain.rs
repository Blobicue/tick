//! tinytrain — the M4.0 gate: from-scratch tiny model -> tickfile ->
//! stock engine loads and generates; plus the STE QAT sanity (a
//! BitLinear layer fit to a random dense target converges).

// Parity/training harness: index loops mirror the reference math.
#![allow(clippy::needless_range_loop)]
#![allow(unused_assignments)]
#![allow(dead_code)]
// `!(x < t)` NaN-safe gates are deliberate (law: NaN comparisons are false).
#![allow(clippy::neg_cmp_op_on_partial_ord)]
#![allow(clippy::manual_memcpy)]
fn main() -> Result<(), String> {
    use tick_train::train::*;
    let cfg = TinyConfig::default();
    println!(
        "tiny: {} layers, hidden {}, ffn {}, vocab {} — {:.1}M ternary",
        cfg.n_layer,
        cfg.hidden,
        cfg.ffn,
        cfg.vocab,
        (cfg.n_layer as f64
            * (4.0 * (cfg.hidden * cfg.hidden) as f64
                + 2.0 * (cfg.n_head_kv * cfg.head_dim * cfg.hidden) as f64
                + 2.0 * (cfg.ffn * cfg.hidden) as f64
                + (cfg.hidden * cfg.ffn) as f64))
            / 1e6
    );
    // 1) init -> tickfile -> engine runs it
    let (embd, norms, mats) = init_tiny(&cfg, 42);
    let out = std::path::Path::new("/tmp/tiny_m4.tick");
    write_tiny_tickfile(&cfg, &embd, &norms, &mats, out)?;
    let meta = std::fs::metadata(out).unwrap().len();
    println!("tiny tickfile: {meta} bytes ({:.2} MB)", meta as f64 / 1e6);
    let mut m = tick_engine::model::Model::open(out, 128)?;
    let ids: Vec<u32> = [0u32, 1, 2, 3, 4].to_vec();
    let (gen, logits) = m.greedy(&ids, 8)?;
    println!(
        "engine ran the from-scratch artifact: {} gen tokens, {} logits, argmax {}",
        gen.len(),
        logits.len(),
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0)
    );
    assert_eq!(logits.len(), cfg.vocab);

    // 2) STE QAT sanity: fit one BitLinear (512x512) to a QUANTIZED
    //    target (ternary×scale — exactly representable on the grid),
    //    from a different init. Loss must fall far: gradient flows
    //    through the ternary view + f16 scales via the estimator.
    let mut rng = Rng { s: 7 };
    let rows = 512usize;
    let cols = 512usize;
    let mut trng = Rng { s: 99 };
    let target_w = TrainableTq::new(rows, cols, || trng.normal(0.0, (cols as f32).powf(-0.5)));
    let target = quantize_dense(&target_w);
    let mut w = TrainableTq::new(rows, cols, || rng.normal(0.0, (cols as f32).powf(-0.5)));
    let loss0 = fit_loss(&w, &target);
    let mut grads = vec![0.0f32; rows * cols];
    for step in 0..400 {
        // L = 0.5 * ||q(w) - target||^2 elementwise; each master weight
        // is independent, so dL/dw = (q - t) directly (no /N — that
        // normalization is for batch means, and here it makes the
        // per-element step ~1e-6 while the masters are ~4e-2)
        let q = quantize_dense(&w);
        for i in 0..rows * cols {
            grads[i] = q[i] - target[i];
        }
        w.ste_update(&grads, 0.5);
        if step % 100 == 99 {
            println!("step {}: fit loss {:.6e}", step + 1, fit_loss(&w, &target));
        }
    }
    let loss1 = fit_loss(&w, &target);
    println!(
        "STE QAT sanity: loss {loss0:.6e} -> {loss1:.6e} ({:.1}x reduction)",
        loss0 / loss1
    );
    if loss1 >= loss0 * 0.3 {
        return Err("QAT sanity failed: loss did not converge".into());
    }
    println!("M4.0 GATE: PASS — from-scratch tickfile runs in the engine + STE converges");
    Ok(())
}

/// MSE between the quantized (dense-valued) weights and the target.
fn fit_loss(w: &tick_train::train::TrainableTq, target: &[f32]) -> f32 {
    let q = quantize_dense(w);
    let mut s = 0.0f64;
    for (a, b) in q.iter().zip(target) {
        s += ((a - b) * (a - b)) as f64;
    }
    (s / q.len() as f64) as f32
}

/// The quantized weights as dense values (ternary × block scale).
fn quantize_dense(w: &tick_train::train::TrainableTq) -> Vec<f32> {
    let t = w.ternary();
    let cols = t.cols;
    let mut out = vec![0.0f32; t.rows * cols];
    for r in 0..t.rows {
        for c in 0..cols {
            let blk = c / 256;
            let tri = crate_trit(&t, r, blk, c % 256);
            out[r * cols + c] = tri as f32 * t.d[r * t.n_blocks + blk];
        }
    }
    out
}

fn crate_trit(t: &tick_kernels::bitlinear::TqTensor, r: usize, blk: usize, j: usize) -> i32 {
    tick_kernels::bitlinear::trit_at(
        &t.packed[r * t.n_blocks * 64 + blk * 64..r * t.n_blocks * 64 + (blk + 1) * 64],
        j,
    )
}
