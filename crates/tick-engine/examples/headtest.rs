//! headtest — try head-dot variants against ref pos-0 logits.

// Parity/training harness: index loops mirror the reference math.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(unused_assignments)]
#![allow(dead_code)]
fn main() -> Result<(), String> {
    let path = "/home/juzz/models/bitnet/bitnet-b1.58-2B-4T-tq1_0.gguf";
    let mut m = tick_engine::model::Model::open(std::path::Path::new(path), 64)?;
    let hidden = m.forward_token(128000, 0)?;
    let x = tick_engine::model::rms_norm(&hidden, &m.embd_norm, m.cfg.rms_eps);
    let h = m.cfg.hidden;
    let ref_l: Vec<f32> = std::fs::read_to_string("/tmp/ref_pos0_logits.txt")
        .unwrap()
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    // real loops (avoid placeholder mess)
    // variants: fma vs muladd x hsum order
    let mut maxes = [0.0f64; 8];
    let mut exacts = [0usize; 8];
    for (v, rl) in ref_l.iter().enumerate() {
        let row = &m.embd[v * h..(v + 1) * h];
        // the true candidate: ggml fallback — x converted to f16 (RNE),
        // then ggml_vec_dot_f16 (plain NEON order) on the f16 pair
        let x16: Vec<u16> = x
            .iter()
            .map(|&f| tick_engine::model::Model::f32_to_f16_bits_pub(f))
            .collect();
        let variants: [(&str, f32); 7] = [
            (
                "fallback16",
                tick_kernels::bitlinear::dot_f16_sve(row, &x16),
            ),
            (
                "svefit_widened",
                tick_kernels::bitlinear::vec_dot_f16(
                    &row.iter()
                        .map(|&b| tick_kernels::bitlinear::half_to_f32(b))
                        .collect::<Vec<_>>(),
                    &x16.iter()
                        .map(|&b| tick_kernels::bitlinear::half_to_f32(b))
                        .collect::<Vec<_>>(),
                ),
            ),
            (
                "svefit_rawx",
                tick_kernels::bitlinear::vec_dot_f16(
                    &row.iter()
                        .map(|&b| tick_kernels::bitlinear::half_to_f32(b))
                        .collect::<Vec<_>>(),
                    &x,
                ),
            ),
            (
                "dotf32_widened",
                tick_kernels::bitlinear::dot_f32(
                    &row.iter()
                        .map(|&b| tick_kernels::bitlinear::half_to_f32(b))
                        .collect::<Vec<_>>(),
                    &x,
                ),
            ),
            (
                "dotf32_widened_x16",
                tick_kernels::bitlinear::dot_f32(
                    &row.iter()
                        .map(|&b| tick_kernels::bitlinear::half_to_f32(b))
                        .collect::<Vec<_>>(),
                    &x16.iter()
                        .map(|&b| tick_kernels::bitlinear::half_to_f32(b))
                        .collect::<Vec<_>>(),
                ),
            ),
            ("f64_seq", {
                let mut t = 0.0f64;
                for i in 0..h {
                    t += (tick_kernels::bitlinear::half_to_f32(row[i])
                        * tick_kernels::bitlinear::half_to_f32(x16[i]))
                        as f64;
                }
                t as f32
            }),
            ("f64_seq_rawx", {
                let mut t = 0.0f64;
                for i in 0..h {
                    t += (tick_kernels::bitlinear::half_to_f32(row[i]) * x[i]) as f64;
                }
                t as f32
            }),
        ];
        for (vi, (_, val)) in variants.iter().enumerate() {
            let e = (val - *rl) as f64;
            if e.abs() > maxes[vi] {
                maxes[vi] = e.abs();
            }
            if e == 0.0 {
                exacts[vi] += 1;
            }
        }
    }
    const NAMES: [&str; 7] = [
        "fallback16",
        "svefit_widened",
        "svefit_rawx",
        "dotf32_widened",
        "dotf32_widened_x16",
        "f64_seq",
        "f64_seq_rawx",
    ];
    for (vi, name) in NAMES.iter().enumerate() {
        println!(
            "{name:>16}: max |err| {:#.7} exact {}/{}",
            maxes[vi],
            exacts[vi],
            ref_l.len()
        );
    }
    Ok(())
}
