//! fattnfix — dump per-position Qcur (f32), K/V cache bits (f16 hex)
//! for layer 0 over a whole prompt: fixture for tiled/split-kv probes.
//! stdin: {"ids": [...]}

// Parity/training harness: index loops mirror the reference math.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(unused_assignments)]
#![allow(dead_code)]
fn main() -> Result<(), String> {
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
    let mut m = tick_engine::model::Model::open(
        std::path::Path::new("/home/juzz/models/bitnet/bitnet-b1.58-2B-4T-tq1_0.gguf"),
        1024,
    )?;
    // raw forward hooks: use trace dumps via dump_trace on layer 0
    m.trace = Some("/tmp/fattnfix_trace.txt".into());
    let _ = std::fs::remove_file("/tmp/fattnfix_trace.txt");
    for (i, &t) in ids.iter().enumerate() {
        let _ = m.forward_token(t as usize, i)?;
    }
    Ok(())
}
