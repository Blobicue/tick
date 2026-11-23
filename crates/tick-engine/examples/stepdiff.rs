//! stepdiff — teacher-forced engine steps vs logitprobe2's reference
//! top-5: finds the FIRST position where the logits diverge.

// Parity/training harness: index loops mirror the reference math.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(unused_assignments)]
#![allow(dead_code)]
fn main() -> Result<(), String> {
    let ids: Vec<u32> = std::fs::read_to_string("/tmp/teacher_ids.txt")
        .unwrap()
        .trim()[2..]
        .trim_end_matches(']')
        .split(", ")
        .filter_map(|s| s.parse().ok())
        .collect();
    let plen = 35usize;
    let mut m = tick_engine::model::Model::open(
        std::path::Path::new("/home/juzz/models/bitnet/bitnet-b1.58-2B-4T-tq1_0.gguf"),
        1024,
    )?;
    let _ = std::fs::remove_file("/tmp/mine_steps_tensors.txt");
    m.trace = Some("/tmp/mine_steps_tensors.txt".into());
    let ref_raw = std::fs::read_to_string("/tmp/ref_steps.log").unwrap();
    let ref_lines: Vec<&str> = ref_raw.lines().filter(|l| l.starts_with("pos ")).collect();
    let parse_top = |l: &str| -> Vec<(u64, f64)> {
        l.split("top5:")
            .nth(1)
            .unwrap_or("")
            .split_whitespace()
            .filter_map(|t| {
                let (i, v) = t.split_once(':')?;
                Some((i.parse().ok()?, v.parse().ok()?))
            })
            .collect()
    };
    // prefill (tiled? 35 < 64 -> one_chunk)
    let mut hidden = None;
    for (i, &t) in ids[..plen].iter().enumerate() {
        hidden = Some(m.forward_token(t as usize, i)?);
    }
    let mut hidden = hidden.ok_or("no hidden".to_string())?;
    let mut first_bad: Option<(usize, String)> = None;
    for (step, &t) in ids[plen..].iter().enumerate() {
        let pos = plen + step;
        let normed = tick_engine::model::rms_norm(&hidden, &m.embd_norm, m.cfg.rms_eps);
        let logits = m.logits(&normed);
        // compare vs reference pos (ref_lines[step] == this step's top5)
        let rl = ref_lines.get(step).copied().unwrap_or("");
        let top = parse_top(rl);
        let mut mine_top: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
        mine_top.sort_by(|a, b| b.1.total_cmp(&a.1));
        mine_top.truncate(5);
        let ok = !top.is_empty()
            && top
                .iter()
                .zip(&mine_top)
                .all(|(r, mne)| r.0 == mne.0 as u64 && (r.1 - mne.1 as f64).abs() < 2e-4);
        // per-step top-1 drift (ref vs mine)
        if let Some((ri, rv)) = top.first() {
            let mv = logits[*ri as usize];
            let d = (rv - mv as f64).abs();
            if d > 1e-5 {
                println!("step {step} pos {pos} n_kv {} top1 {ri} drift {d:.3e} (ref {rv:.6} mine {:.6})", pos + 1, mv);
            }
        }
        if !ok && first_bad.is_none() {
            let fmt = |v: &[(usize, f32)]| -> String {
                v.iter()
                    .map(|(i, x)| format!("{i}:{x:.4}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            first_bad = Some((
                pos,
                format!(
                    "step {step} pos {pos} n_kv {}:\n  ref : {}\n  mine: {}",
                    pos + 1,
                    rl.split("top5:").nth(1).unwrap_or("?").trim(),
                    fmt(&mine_top)
                ),
            ));
        }
        // teacher-forced next
        let _n_kv = pos + 1;
        hidden = m.forward_token_regime(t as usize, pos, tick_engine::model::decode_regime(pos))?;
    }
    match first_bad {
        Some((_pos, msg)) => {
            println!("FIRST DIVERGENCE:\n{msg}");
            Ok(())
        }
        None => {
            println!("all {} steps match within 2e-4", ids.len() - plen);
            Ok(())
        }
    }
}
