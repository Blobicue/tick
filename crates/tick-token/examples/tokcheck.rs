//! tokcheck — the in-crate tokenizer vs the frozen id fixtures:
//! every battery prompt, templated + encoded, must equal pNN_ids.json
//! byte-for-byte. Also diffs the tickfile-loaded vocab.

// Parity/training harness: index loops mirror the reference math.
#![allow(clippy::needless_range_loop)]
fn main() -> Result<(), String> {
    let gguf = "/home/juzz/models/bitnet/bitnet-b1.58-2B-4T-tq1_0.gguf";
    let tick = "/home/juzz/models/bitnet/bitnet-b1.58-2B-4T.tick";
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../tick-engine/fixtures");
    let battery: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{dir}/battery.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;

    let g = tick_format::gguf::Gguf::open(std::path::Path::new(gguf))?;
    let t0 = std::time::Instant::now();
    let vocab = tick_token::token::Vocab::from_gguf(&g)?;
    println!(
        "gguf vocab: {} tokens, {} merges, {} specials in {:?}",
        vocab.texts.len(),
        vocab.bpe_ranks.len(),
        vocab.specials.len(),
        t0.elapsed()
    );

    let mut pass = 0;
    let total = battery["cases"].as_array().map(Vec::len).unwrap_or(0);
    for case in battery["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let prompt = case["prompt"].as_str().unwrap();
        let fixture: Vec<u32> = serde_json::from_str(
            &std::fs::read_to_string(format!("{dir}/{id}_ids.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let t1 = std::time::Instant::now();
        let ids = tick_token::token::encode_chat(prompt, &vocab);
        let dt = t1.elapsed();
        if ids == fixture {
            pass += 1;
            println!("  {id}: MATCH ({}, {:.1?})", ids.len(), dt);
        } else {
            let first = ids.iter().zip(&fixture).position(|(a, b)| a != b);
            println!(
                "  {id}: MISMATCH len {} vs {} first-diff at {:?}",
                ids.len(),
                fixture.len(),
                first
            );
        }
    }
    println!("gguf tokenizer: {pass}/{total}");

    // tickfile vocab must agree on every prompt too
    if std::path::Path::new(tick).exists() {
        let tf = tick_format::weights::TickFile::open(std::path::Path::new(tick))?;
        let v2 = tick_token::token::Vocab::from_tickfile(&tf)?;
        let mut pass2 = 0;
        for case in battery["cases"].as_array().unwrap() {
            let id = case["id"].as_str().unwrap();
            let prompt = case["prompt"].as_str().unwrap();
            let fixture: Vec<u32> = serde_json::from_str(
                &std::fs::read_to_string(format!("{dir}/{id}_ids.json"))
                    .map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            if tick_token::token::encode_chat(prompt, &v2) == fixture {
                pass2 += 1;
            }
        }
        println!("tick tokenizer: {pass2}/{total}");
        if pass + pass2 != 2 * total {
            return Err("tokenizer gate failed".into());
        }
    } else if pass != total {
        return Err("tokenizer gate failed".into());
    }
    Ok(())
}
