//! prefilltrace — our per-layer l_out for a whole prompt treated as
//! ONE prefill batch (batch_len = len), mirroring tensorprobe's
//! single llama_decode. Stdin: {"ids": [...]}.

// Parity/training harness: index loops mirror the reference math.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(unused_assignments)]
#![allow(dead_code)]
fn main() -> Result<(), String> {
    let path = "/home/juzz/models/bitnet/bitnet-b1.58-2B-4T-tq1_0.gguf";
    let mut input = String::new();
    use std::io::Read;
    std::io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| e.to_string())?;
    let req: serde_json::Value = serde_json::from_str(&input).map_err(|e| e.to_string())?;
    let ids: Vec<u32> = req["ids"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|v| v.as_u64().map(|x| x as u32))
        .collect();
    let _ = std::fs::remove_file("/tmp/mine_tensors.txt");
    let mut m = tick_engine::model::Model::open(std::path::Path::new(path), 512)?;
    m.trace = Some("/tmp/mine_tensors.txt".into());

    let mut hidden = None;
    for (i, &t) in ids.iter().enumerate() {
        hidden = Some(m.forward_token(t as usize, i)?);
    }
    let hidden = hidden.unwrap();
    let rn = tick_engine::model::rms_norm(&hidden, &m.embd_norm, m.cfg.rms_eps);
    m.dump_trace("result_norm", &rn);
    Ok(())
}
