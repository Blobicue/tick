//! tiledcheck — the tiled fattn mirror vs the ggml probe output.
//! Reads /tmp/fattnfix_trace.txt (engine fixture) and /tmp/tiled_out.txt.

// Parity/training harness: index loops mirror the reference math.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(unused_assignments)]
#![allow(dead_code)]
use std::collections::HashMap;

fn load(path: &str) -> HashMap<String, Vec<Vec<f32>>> {
    let mut out: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    let mut cur_name = String::new();
    let mut cur: Vec<f32> = Vec::new();
    for line in std::fs::read_to_string(path).unwrap().lines() {
        if line.starts_with("TENSOR ") {
            if !cur_name.is_empty() {
                out.entry(cur_name.clone())
                    .or_default()
                    .push(std::mem::take(&mut cur));
            }
            cur_name = line.split_whitespace().nth(1).unwrap().to_string();
        } else if let Ok(v) = line.trim().parse::<f32>() {
            cur.push(v);
        }
    }
    if !cur_name.is_empty() {
        out.entry(cur_name).or_default().push(cur);
    }
    out
}

/// The tiled fattn row arithmetic as compiled on this build:
/// all-f32; scores = scalar fma chain over DK; softmax per KV tile of
/// 64 via ggml_v_expf with f64 4-chunk sums; V accumulated with a
/// scalar fma chain over the tile; online M/S rescale.
fn tiled_row(
    q: &[f32],            // RAW f32 Q row (DK)
    k_cache: &[Vec<u16>], // per position: 640 f16 bits
    v_cache: &[Vec<u16>],
    kv_head: usize,
    hd: usize,
    n_kv: usize,
    pos: usize,
    scale: f32,
) -> Vec<f32> {
    let mut s_sum = 0.0f32;
    let mut m = f32::NEG_INFINITY;
    let mut vkq32 = vec![0.0f32; hd];
    let mut ic0 = 0usize;
    while ic0 < n_kv {
        let kv_tile = (n_kv - ic0).min(64);
        // fully masked tile for this row?
        if ic0 > pos {
            ic0 += 64;
            continue;
        }
        // scores: scalar fma chain; pads -> -inf
        let mut kq = [0.0f32; 64];
        for tk in 0..kv_tile {
            let kb = &k_cache[ic0 + tk][kv_head * hd..(kv_head + 1) * hd];
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
        // causal mask add: ic0+tk > pos -> -inf
        for tk in 0..kv_tile {
            if ic0 + tk > pos {
                kq[tk] = f32::NEG_INFINITY;
            }
        }
        // tile max over all 64
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
        // softmax via ggml_v_expf, 4-chunk f64 sum
        let mut y = [0.0f32; 64];
        let mut sum = 0.0f64;
        for c in 0..16 {
            let mut vsum;
            let lane = [
                tick_engine::model::ggml_v_expf(kq[c * 4] - mnew),
                tick_engine::model::ggml_v_expf(kq[c * 4 + 1] - mnew),
                tick_engine::model::ggml_v_expf(kq[c * 4 + 2] - mnew),
                tick_engine::model::ggml_v_expf(kq[c * 4 + 3] - mnew),
            ];
            vsum = (lane[0] + lane[1]) + (lane[2] + lane[3]);
            y[c * 4..c * 4 + 4].copy_from_slice(&lane);
            sum += vsum as f64;
            let _ = &mut vsum;
        }
        s_sum = (s_sum as f64 + sum) as f32;
        // V accumulation: scalar fma chain over the tile (pads 0)
        for tk in 0..64 {
            if y[tk] == 0.0 {
                continue; // no-op guard: fma(x, 0, acc) = acc — but keeps EXACT semantics even without skipping; skip is safe
            }
            let vb = &v_cache[ic0 + tk][kv_head * hd..(kv_head + 1) * hd];
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

fn main() -> Result<(), String> {
    let mine = load("/tmp/fattnfix_trace.txt");
    let probe: Vec<f32> = std::fs::read_to_string("/tmp/tiled_out.txt")
        .unwrap()
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    let hd = 128usize;
    let n_head = 20usize;
    let n_kv = 70usize;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let kc: Vec<Vec<u16>> = (0..n_kv)
        .map(|p| {
            mine[&format!("Kcur-0-p{p}")][0]
                .iter()
                .map(|&f| tick_kernels::bitlinear::f32_to_f16(f))
                .collect()
        })
        .collect();
    let vc: Vec<Vec<u16>> = (0..n_kv)
        .map(|p| {
            mine[&format!("Vcur-0-p{p}")][0]
                .iter()
                .map(|&f| tick_kernels::bitlinear::f32_to_f16(f))
                .collect()
        })
        .collect();
    let mut bad = 0usize;
    let mut worst = 0.0f32;
    for t in 0..n_kv {
        for h in 0..n_head {
            let q = mine[&format!("Qcur-0-p{t}")][0][h * hd..(h + 1) * hd].to_vec();
            let out = tiled_row(&q, &kc, &vc, h / 4, hd, n_kv, t, scale);
            // probe rows: token-major, head-fastest: row = t*20 + h
            let base = (t * n_head + h) * hd;
            for d in 0..hd {
                let p = probe[base + d];
                if p != out[d] {
                    bad += 1;
                    worst = worst.max((p - out[d]).abs());
                }
            }
        }
    }
    println!(
        "tiled mirror vs probe: {bad}/{} differ, worst {worst:.3e}",
        n_kv * n_head * hd
    );
    Ok(())
}
