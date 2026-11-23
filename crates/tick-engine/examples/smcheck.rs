//! smcheck — engine softmax vs ggml soft_max_ext (softmaxprobe output)

// Parity/training harness: index loops mirror the reference math.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(unused_assignments)]
#![allow(dead_code)]
fn main() -> Result<(), String> {
    let scale: f32 = std::env::args()
        .nth(1)
        .unwrap()
        .parse()
        .map_err(|_| "scale")?;
    let wp: Vec<f32> = std::fs::read_to_string(std::env::args().nth(2).unwrap())
        .map_err(|e| e.to_string())?
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    let ref_p: Vec<f32> = std::fs::read_to_string(std::env::args().nth(3).unwrap())
        .map_err(|e| e.to_string())?
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    let _n_valid = ref_p.iter().filter(|v| **v > 0.0).count();
    // engine path: scores scaled, softmax over VALID entries (pos+1)
    let scores: Vec<f32> = wp.iter().map(|s| s * scale).collect();
    let valid = ref_p.iter().filter(|v| **v != 0.0).count().max(1);
    let sc: Vec<f32> = scores[..valid].to_vec();
    let max = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let scr: Vec<f32> = sc.iter().map(|s| s - max).collect();
    let mut sum = 0.0f64;
    let mut probs = Vec::new();
    let mut i4 = 0;
    while i4 + 4 <= scr.len() {
        let v: Vec<f32> = scr[i4..i4 + 4]
            .iter()
            .map(|&x| tick_engine::model::ggml_v_expf(x))
            .collect();
        sum += ((v[0] + v[1]) + (v[2] + v[3])) as f64;
        probs.extend(v);
        i4 += 4;
    }
    if i4 < scr.len() {
        let mut lane = [0.0f32; 4];
        for (k, item) in scr[i4..].iter().enumerate() {
            lane[k] = tick_engine::model::ggml_v_expf(*item);
            probs.push(lane[k]);
        }
        let validn = scr.len() - i4;
        let mut vsum = if validn >= 2 {
            lane[0] + lane[1]
        } else {
            lane[0]
        };
        if validn >= 3 {
            vsum += lane[2];
        }
        if validn >= 4 {
            vsum += lane[3];
        }
        sum += vsum as f64;
    }
    let inv = (1.0 / sum) as f32;
    println!("sum={sum:.9e} inv={inv:.9e}");
    for (i, p) in probs.iter().enumerate() {
        let ours = p * inv;
        println!(
            "p{i}: ours {ours:.9e}  ref {:+.9e}  {}",
            ref_p[i],
            if ours == ref_p[i] { "EXACT" } else { "DIFF" }
        );
    }
    Ok(())
}
