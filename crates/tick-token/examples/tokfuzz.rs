//! tokfuzz — differential: split_llama3 vs JSONL cases
//! ({"t": text, "p": [pieces]}) produced by the real python regex.

// Parity/training harness: index loops mirror the reference math.
#![allow(clippy::needless_range_loop)]
fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/tokfuzz_cases.jsonl".into());
    let content = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let mut checked = 0usize;
    let mut bad = 0usize;
    for line in content.lines() {
        let case: serde_json::Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
        let text = case["t"].as_str().unwrap_or("");
        let expected: Vec<String> = case["p"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let got: Vec<String> = tick_token::token::split_llama3(text)
            .iter()
            .map(|&r| {
                let (a, b) = tick_token::token::range_to_bytes(text, r);
                text[a..b].to_string()
            })
            .collect();
        checked += 1;
        if got != expected {
            bad += 1;
            if bad <= 5 {
                println!("MISMATCH text={text:?}\n  regex: {expected:?}\n  ours : {got:?}");
            }
        }
    }
    println!("checked {checked}, mismatches {bad}");
    if bad > 0 {
        return Err("scanner diverges from the regex".into());
    }
    Ok(())
}
