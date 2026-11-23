//! nll — teacher-forced mean NLL over a token sequence (stdin ids).
//! The ppl probe: reference llama-perplexity on wiki chunk 1 reports
//! PPL 9.1435 (ln ≈ 2.2135); a broken stack lands near ln(128256)≈11.8.

// Parity/training harness: index loops mirror the reference math.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(unused_assignments)]
#![allow(dead_code)]

use std::io::Read;

fn tokens_str(_req: &serde_json::Value, id: usize) -> String {
    // lazy: no vocab here; id only
    format!("#{id}")
}

fn main() -> Result<(), String> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/juzz/models/bitnet/bitnet-b1.58-2B-4T-tq1_0.gguf".into());
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| e.to_string())?;
    let req: serde_json::Value = serde_json::from_str(&input).map_err(|e| e.to_string())?;
    let ids: Vec<u32> = req["ids"]
        .as_array()
        .ok_or("ids")?
        .iter()
        .filter_map(|v| v.as_u64().map(|x| x as u32))
        .collect();

    let mut model = tick_engine::model::Model::open(std::path::Path::new(&model_path), 1024)?;
    // prefill all but the last token; score tokens[1..] against logits
    let mut total = 0.0f64;
    let mut n = 0u64;
    for i in 0..ids.len() - 1 {
        let hidden = model.forward_token(ids[i] as usize, i)?;
        // logits for predicting ids[i+1]
        let normed = tick_engine::model::rms_norm(&hidden, &model.embd_norm, model.cfg.rms_eps);
        let logits = model.logits(&normed);
        // logsoftmax nll of the target
        let target = ids[i + 1] as usize;
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut lse = 0.0f64;
        for &l in &logits {
            lse += ((l - max) as f64).exp();
        }
        let lse = lse.ln() + max as f64;
        if i == 0 {
            use std::io::Write;
            let mut f = std::fs::File::create("/tmp/mine_pos0_logits.txt").unwrap();
            for l in &logits {
                writeln!(f, "{:.6e}", l).unwrap();
            }
        }
        if true {
            let mut top: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
            top.sort_by(|a, b| b.1.total_cmp(&a.1));
            top.truncate(5);
            let s: Vec<String> = top.iter().map(|(i, v)| format!("{}:{:.4}", i, v)).collect();
            println!("mine top5: {}", s.join(" "));
        }
        total += lse - logits[target] as f64;
        if std::env::var("TICK_NLL_TRACE").is_ok() && n < 48 {
            let t2 = tokens_str(&req, target);
            println!(
                "pos {n:3} target {target:6} {t2:?} nll {:.4}",
                lse - logits[target] as f64
            );
        }
        n += 1;
    }
    let mean = total / n as f64;
    println!("tokens scored: {n}");
    println!(
        "mean NLL: {mean:.4}  (PPL {:.4})  | reference chunk1: 2.2135 (9.1435)",
        mean.exp()
    );
    Ok(())
}
