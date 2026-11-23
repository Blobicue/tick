//! neonverify — NEON kernels vs their scalar forms on real data.

// NEON kernel probes: only meaningful (and only compilable) where the
// vector kernels exist. Builds empty on non-aarch64 hosts.
#![cfg(target_arch = "aarch64")]
// Parity/training harness: index loops mirror the reference math.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(unused_assignments)]
#![allow(dead_code)]
fn main() -> Result<(), String> {
    let path = "/home/juzz/models/bitnet/bitnet-b1.58-2B-4T-tq1_0.gguf";
    let mut m = tick_engine::model::Model::open(std::path::Path::new(path), 64)?;

    // 1) NEON head vs dot_f16_sve on the f16 rows: all vocab rows
    let g = tick_format::gguf::Gguf::open(std::path::Path::new(path))?;
    let vocab = tick_token::token::Vocab::from_gguf(&g)?;
    let ids = tick_token::token::encode_chat("What is 17 plus 25?", &vocab);
    let h = m.forward_token(ids[0] as usize, 0)?;
    let normed = tick_engine::model::rms_norm(&h, &m.embd_norm, m.cfg.rms_eps);
    let hid = m.cfg.hidden;
    let x16: Vec<u16> = normed
        .iter()
        .map(|&f| tick_kernels::bitlinear::f32_to_f16(f))
        .collect();
    let xw: Vec<f32> = x16
        .iter()
        .map(|&b| tick_kernels::bitlinear::half_to_f32(b))
        .collect();
    let mut bad = 0usize;
    let mut worst = 0.0f32;
    for v in 0..m.cfg.vocab {
        let row16 = &m.embd[v * hid..(v + 1) * hid];
        let a = tick_kernels::bitlinear::dot_f16_sve(row16, &x16);
        let b = tick_kernels::bitlinear::dot_f16row_f32x_neon(row16, &xw);
        if a != b {
            bad += 1;
            worst = worst.max((a - b).abs());
        }
    }
    println!(
        "head NEON vs scalar: {bad}/{} differ, worst {worst:.3e}",
        m.cfg.vocab
    );

    // 2) NEON ternary vs scalar on real rows (gate of layer 0)
    let layer = &m.layers[0];
    let xq = tick_kernels::bitlinear::quantize_q8_k_pub(&normed);
    let mut bad2 = 0usize;
    for r in 0..layer.gate.rows {
        let a = tick_kernels::bitlinear::dot_tq1_0_q8_k(&layer.gate, r, &xq);
        // force scalar: NEON path is inside dot_tq1_0_q8_k; compare via
        // an independent scalar reimplementation
        let mut sumf = 0.0f32;
        const QK: usize = 256;
        let nb = layer.gate.n_blocks;
        for b in 0..nb {
            let pbase = r * nb * 64 + b * 64;
            let packed = &layer.gate.packed[pbase..pbase + 64];
            let mut sum = 0i32;
            for j in 0..QK {
                sum += tick_kernels::bitlinear::trit_at(packed, j) * xq[b].q[j] as i32;
            }
            let d = layer.gate.d[r * nb + b] * xq[b].d;
            sumf = d.mul_add(sum as f32, sumf);
        }
        if a != sumf {
            bad2 += 1;
        }
    }
    println!("ternary NEON vs scalar: {bad2}/{} differ", layer.gate.rows);
    Ok(())
}
