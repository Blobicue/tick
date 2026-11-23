//! splitcheck — the split-kv decode mirror vs the ggml probe.
//! Chunks of ceil(n_kv/nth) with nth=10 (the lane's -t 10); per chunk
//! the one_chunk f16-accumulator arithmetic; libm-expf partials reduce.

// Parity/training harness: index loops mirror the reference math.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(unused_assignments)]
#![allow(dead_code)]

fn load_bits(path: &str, n_kv: usize, n_head_kv: usize, hd: usize) -> Vec<Vec<u16>> {
    let all: Vec<u16> = std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter_map(|l| u16::from_str_radix(l.trim(), 16).ok())
        .collect();
    // file is head-major: [kvh][ic][d]
    let mut out = vec![vec![0u16; n_head_kv * hd]; n_kv];
    for kvh in 0..n_head_kv {
        for ic in 0..n_kv {
            let base = (kvh * n_kv + ic) * hd;
            out[ic][kvh * hd..(kvh + 1) * hd].copy_from_slice(&all[base..base + hd]);
        }
    }
    out
}

/// one_chunk over [start, end): all-visible decode positions — returns
/// (M, S, VKQ widened to f32)
fn one_chunk(
    q16: &[u16],
    kc: &[Vec<u16>],
    vc: &[Vec<u16>],
    kv_head: usize,
    hd: usize,
    start: usize,
    end: usize,
    scale: f32,
) -> (f32, f32, Vec<f32>) {
    let mut vkq16 = vec![0u16; hd];
    let mut m = f32::NEG_INFINITY;
    let mut s_sum = 0.0f32;
    for ic in start..end {
        let kb = &kc[ic][kv_head * hd..(kv_head + 1) * hd];
        let s = tick_kernels::bitlinear::dot_f16_sve(kb, q16) * scale;
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
        let vb = &vc[ic][kv_head * hd..(kv_head + 1) * hd];
        tick_kernels::bitlinear::vec_mad_f16(&mut vkq16, vb, vs);
        s_sum = s_sum.mul_add(ms, vs);
    }
    let wide: Vec<f32> = vkq16
        .iter()
        .map(|&b| tick_kernels::bitlinear::half_to_f32(b))
        .collect();
    (m, s_sum, wide)
}

fn split_row(
    q: &[f32],
    kc: &[Vec<u16>],
    vc: &[Vec<u16>],
    kv_head: usize,
    hd: usize,
    n_kv: usize,
    scale: f32,
    nth: usize,
    mode: u8, // 0=none, 1=fma-right(c*sn fused), 2=fma-left(f*so fused)
) -> Vec<f32> {
    let q16: Vec<u16> = q
        .iter()
        .map(|&f| tick_kernels::bitlinear::f32_to_f16(f))
        .collect();
    let chunk_size = n_kv.div_ceil(nth);
    let mut m_f = f32::NEG_INFINITY;
    let mut s_f = 0.0f32;
    let mut vkq_f = vec![0.0f32; hd];
    for c in 0..nth {
        let start = c * chunk_size;
        if start >= n_kv {
            continue;
        }
        let end = (start + chunk_size).min(n_kv);
        let (m_c, s_c, vkq_c) = one_chunk(&q16, kc, vc, kv_head, hd, start, end, scale);
        if s_c == 0.0 {
            continue;
        }
        let m_new = if m_c > m_f { m_c } else { m_f };
        let so = (m_f - m_new).exp();
        let sn = (m_c - m_new).exp();
        for d in 0..hd {
            vkq_f[d] = match mode {
                1 => vkq_c[d].mul_add(sn, vkq_f[d] * so),
                2 => vkq_f[d].mul_add(so, vkq_c[d] * sn),
                _ => vkq_f[d] * so + vkq_c[d] * sn,
            };
        }
        s_f = match mode {
            1 => s_c.mul_add(sn, s_f * so),
            2 => s_f.mul_add(so, s_c * sn),
            _ => s_f * so + s_c * sn,
        };
        m_f = m_new;
    }
    let inv = if s_f == 0.0 { 0.0 } else { 1.0 / s_f };
    vkq_f.iter().map(|v| v * inv).collect()
}

fn main() -> Result<(), String> {
    let hd = 128usize;
    let n_kv = 600usize;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let kc = load_bits("/tmp/split_k.txt", n_kv, 5, hd);
    let vc = load_bits("/tmp/split_v.txt", n_kv, 5, hd);
    let q: Vec<f32> = std::fs::read_to_string("/tmp/split_q.txt")
        .unwrap()
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    let probe: Vec<f32> = std::fs::read_to_string("/tmp/split_out.txt")
        .unwrap()
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    for mode in 0u8..3 {
        let mut bad = 0usize;
        for h in 0..20 {
            let out = split_row(
                &q[h * hd..(h + 1) * hd],
                &kc,
                &vc,
                h / 4,
                hd,
                n_kv,
                scale,
                10,
                mode,
            );
            for d in 0..hd {
                if probe[h * hd + d] != out[d] {
                    bad += 1;
                }
            }
        }
        println!("split mirror (mode={mode}): {bad}/2560 differ");
    }
    Ok(())
}
