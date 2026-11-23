//! bench — kernel/forward microbenchmarks + NEON-vs-scalar spot check.

// NEON kernel probes: only meaningful (and only compilable) where the
// vector kernels exist. Builds empty on non-aarch64 hosts.
#![cfg(target_arch = "aarch64")]
// Parity/training harness: index loops mirror the reference math.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(unused_assignments)]
#![allow(dead_code)]
fn main() -> Result<(), String> {
    #[cfg(not(target_arch = "aarch64"))]
    {
        eprintln!("NEON probe: aarch64-only (vector kernels do not exist on this arch)");
        return Ok(());
    }
    #[cfg(target_arch = "aarch64")]
    run()
}

#[cfg(target_arch = "aarch64")]
fn run() -> Result<(), String> {
    let g = tick_format::gguf::Gguf::open(std::path::Path::new(
        "/home/juzz/models/bitnet/bitnet-b1.58-2B-4T-tq1_0.gguf",
    ))?;
    let mut m = tick_engine::model::Model::open(
        std::path::Path::new("/home/juzz/models/bitnet/bitnet-b1.58-2B-4T-tq1_0.gguf"),
        64,
    )?;
    let vocab = tick_token::token::Vocab::from_gguf(&g)?;
    let ids = tick_token::token::encode_chat("What is 17 plus 25?", &vocab);
    let t0 = std::time::Instant::now();
    let h = m.forward_token(ids[0] as usize, 0)?;
    let t_fwd = t0.elapsed();
    let normed = tick_engine::model::rms_norm(&h, &m.embd_norm, m.cfg.rms_eps);
    let t1 = std::time::Instant::now();
    let logits = m.logits(&normed);
    let t_head = t1.elapsed();
    println!(
        "forward(1 tok): {t_fwd:?}   head: {t_head:?}   argmax: {}",
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0)
    );
    // layer matvec bench: gate (6912x2560)
    let layer = &m.layers[0];
    let x = normed.clone();
    let t2 = std::time::Instant::now();
    let _ = tick_kernels::bitlinear::bitlinear(&layer.gate, &x);
    let t_gate = t2.elapsed();
    println!("gate 6912x2560: {t_gate:?}");
    let t3 = std::time::Instant::now();
    let _ = tick_kernels::bitlinear::bitlinear(&layer.wo, &x);
    let t_wo = t3.elapsed();
    println!("wo   2560x2560: {t_wo:?}");
    // dispatch overhead: empty chunks
    let sink = std::sync::atomic::AtomicUsize::new(0);
    let t = std::time::Instant::now();
    for _ in 0..1000 {
        tick_kernels::par::for_chunks(6912, 1, |s, e| {
            sink.fetch_xor(e.wrapping_add(s), std::sync::atomic::Ordering::Relaxed);
        });
    }
    let dt_empty = t.elapsed() / 1000;
    println!(
        "for_chunks empty dispatch: {dt_empty:?} (threads: {})",
        tick_kernels::par::threads()
    );
    // real rows through par_rows vs serial loop
    let xq2 = tick_kernels::bitlinear::quantize_q8_k_pub(&x);
    let mut out = vec![0.0f32; layer.gate.rows];
    let t = std::time::Instant::now();
    tick_kernels::par::par_rows(&mut out, |r| {
        tick_kernels::bitlinear::dot_tq1_0_q8_k(&layer.gate, r, &xq2)
    });
    let dt_par = t.elapsed();
    let t = std::time::Instant::now();
    for r in 0..layer.gate.rows {
        out[r] = tick_kernels::bitlinear::dot_tq1_0_q8_k(&layer.gate, r, &xq2);
    }
    let dt_ser = t.elapsed();
    println!("gate par: {dt_par:?}  serial: {dt_ser:?}");
    // A/B head kernel microbench: f32-rows vs f16-rows (+widen), 100 passes
    let hid2 = m.cfg.hidden;
    let n = 4096usize; // rows subset
    let xw2: Vec<f32> = (0..hid2).map(|i| (i as f32 * 0.001).sin()).collect();
    let _x16b: Vec<u16> = xw2
        .iter()
        .map(|&f| tick_kernels::bitlinear::f32_to_f16(f))
        .collect();
    let rows32: Vec<f32> = tick_engine::model::Model::widen_embd_subset(&m.embd[..n * hid2]);
    let t = std::time::Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..100 {
        for v in 0..n {
            sink +=
                tick_kernels::bitlinear::dot_f32x16_neon(&rows32[v * hid2..(v + 1) * hid2], &xw2);
        }
    }
    let dt32 = t.elapsed() / 100;
    let t = std::time::Instant::now();
    for _ in 0..100 {
        for v in 0..n {
            sink += tick_kernels::bitlinear::dot_f16row_f32x_neon(
                &m.embd[v * hid2..(v + 1) * hid2],
                &xw2,
            );
        }
    }
    let dt16 = t.elapsed() / 100;
    println!(
        "head kernel A/B (4096 rows): f32-rows {dt32:?}  f16-rows+widen {dt16:?}  (sink {sink:.3})"
    );
    tick_engine::model::prof::dump();
    Ok(())
}
