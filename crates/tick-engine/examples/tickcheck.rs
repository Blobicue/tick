//! tickcheck — the format's parity gate: same prompt through the
//! GGUF-loaded engine and the tickfile-loaded engine; token-exact.

// Parity/training harness: index loops mirror the reference math.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(unused_assignments)]
#![allow(dead_code)]
fn main() -> Result<(), String> {
    let gguf = "/home/juzz/models/bitnet/bitnet-b1.58-2B-4T-tq1_0.gguf";
    let tick = "/home/juzz/models/bitnet/bitnet-b1.58-2B-4T.tick";
    // battery p03, chat-templated (from the frozen goldens workflow)
    let ids: Vec<u32> = serde_json::from_str(include_str!("../fixtures/p03_ids.json"))
        .map_err(|e| e.to_string())?;
    let n = 48usize;
    let t0 = std::time::Instant::now();
    let mut a = tick_engine::model::Model::open(std::path::Path::new(gguf), 256)?;
    let (ga, _) = a.greedy(&ids, n)?;
    println!("gguf : {} tokens in {:?}", ga.len(), t0.elapsed());
    let t1 = std::time::Instant::now();
    let mut b = tick_engine::model::Model::open(std::path::Path::new(tick), 256)?;
    let (gb, _) = b.greedy(&ids, n)?;
    println!("tick : {} tokens in {:?}", gb.len(), t1.elapsed());
    if ga == gb {
        println!("PARITY: token-exact across formats ({} tokens)", ga.len());
        Ok(())
    } else {
        let first = ga.iter().zip(&gb).position(|(x, y)| x != y);
        Err(format!("MISMATCH at token {:?}", first))
    }
}
