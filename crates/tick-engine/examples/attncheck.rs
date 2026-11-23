//! attncheck — recompute layer-0 token-1 attention from REFERENCE
//! Kcur/Qcur/Vcur dumps (tensorprobe) with the engine's primitives,
//! and diff against our attnmix_p1 trace. Isolates wiring vs kernels.

// Parity/training harness: index loops mirror the reference math.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(unused_assignments)]
#![allow(dead_code)]
fn main() -> Result<(), String> {
    let mut kcur: Vec<Vec<f32>> = Vec::new();
    let mut qcur: Vec<Vec<f32>> = Vec::new();
    let mut vcur: Vec<Vec<f32>> = Vec::new();
    {
        let mut tensors: Vec<(String, Vec<f32>)> = Vec::new();
        for line in std::fs::read_to_string("/tmp/ref_tensors.txt")
            .unwrap()
            .lines()
        {
            if line.starts_with("TENSOR ") {
                tensors.push((
                    line.split_whitespace().nth(1).unwrap().to_string(),
                    Vec::new(),
                ));
            } else if let Some(last) = tensors.last_mut() {
                if let Ok(v) = line.trim().parse::<f32>() {
                    last.1.push(v);
                }
            }
        }
        for (n, v) in &tensors {
            if n == "Kcur-0" {
                kcur.push(v.clone());
            }
            if n == "Qcur-0" {
                qcur.push(v.clone());
            }
            if n == "Vcur-0" {
                vcur.push(v.clone());
            }
        }
    }
    let k = kcur.last().unwrap();
    let q = qcur.last().unwrap();
    let v = vcur.last().unwrap();
    let hkv = 640usize;
    let h = 2560usize;
    let hd = 128usize;
    let n_head = 20usize;
    let n_head_kv = 5usize;
    let gqa = n_head / n_head_kv;
    let scale = 1.0f32 / (hd as f32).sqrt();

    // our attnmix_p1 dump from the trace
    let mut mine: Option<Vec<f32>> = None;
    {
        let mut cur_name = String::new();
        let mut cur: Vec<f32> = Vec::new();
        for line in std::fs::read_to_string("/tmp/mine_tensors.txt")
            .unwrap()
            .lines()
        {
            if line.starts_with("TENSOR ") {
                if cur_name == "attnmix_p1" {
                    mine = Some(cur.clone());
                }
                cur_name = line.split_whitespace().nth(1).unwrap().to_string();
                cur.clear();
            } else if let Ok(x) = line.trim().parse::<f32>() {
                cur.push(x);
            }
        }
        if cur_name == "attnmix_p1" {
            mine = Some(cur.clone());
        }
    }
    let mine = mine.expect("no attnmix_p1 in trace");

    let to16 = tick_engine::model::Model::f32_to_f16_bits_pub;
    let mut expected = vec![0.0f32; n_head * hd];
    let pos = 1usize;
    for hq in 0..n_head {
        let kv_head = hq / gqa;
        let qrow: Vec<u16> = q[pos * h + hq * hd..pos * h + (hq + 1) * hd]
            .iter()
            .map(|&f| to16(f))
            .collect();
        let mut scores = Vec::new();
        for j in 0..=pos {
            let krow: Vec<u16> = k[j * hkv + kv_head * hd..j * hkv + (kv_head + 1) * hd]
                .iter()
                .map(|&f| to16(f))
                .collect();
            scores.push(tick_kernels::bitlinear::dot_f16_sve(&qrow, &krow) * scale);
        }
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sc: Vec<f32> = scores.iter().map(|s| s - max).collect();
        let mut sum = 0.0f64;
        let mut probs = Vec::new();
        for chunk in sc.chunks(4) {
            let lane: Vec<f32> = chunk
                .iter()
                .map(|&x| tick_engine::model::ggml_v_expf(x))
                .collect();
            let mut vsum = if lane.len() >= 2 {
                lane[0] + lane[1]
            } else {
                lane[0]
            };
            if lane.len() >= 3 {
                vsum += lane[2];
            }
            if lane.len() >= 4 {
                vsum += lane[3];
            }
            sum += vsum as f64;
            probs.extend(lane);
        }
        let inv = (1.0 / sum) as f32;
        let p16: Vec<u16> = probs.iter().map(|&p| to16(p * inv)).collect();
        for d in 0..hd {
            let col: Vec<u16> = (0..=pos)
                .map(|j| to16(v[j * hkv + kv_head * hd + d]))
                .collect();
            expected[hq * hd + d] = tick_kernels::bitlinear::dot_f16_sve(&p16, &col);
        }
    }
    let mut worst = 0.0f32;
    let mut worst_i = 0;
    for (i, (a, b)) in expected.iter().zip(&mine).enumerate() {
        if (a - b).abs() > worst {
            worst = (a - b).abs();
            worst_i = i;
        }
    }
    println!(
        "attnmix L0 p1: worst |ref-derived - ours| = {worst:.3e} at elem {worst_i} (head {}, dim {})",
        worst_i / hd,
        worst_i % hd
    );
    println!("expected[..4] = {:?}", &expected[..4.min(expected.len())]);
    println!("ours[..4]     = {:?}", &mine[..4.min(mine.len())]);
    Ok(())
}
