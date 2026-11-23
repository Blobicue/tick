//! greedy — parity runner: token ids in, greedy text out.
//!
//! stdin: JSON {"ids": [128000, ...], "n": 64}
//! stdout: the generated text (byte-level vocab decode)
//!
//! Tokenization is a deliberate M2 seam: ids come from
//! scripts/tokenize.py (gguf-py BPE against the same model file), so
//! the forward pass is validated independently of tokenizer bugs.
//! The in-crate tokenizer lands in M2.5.

// Parity/training harness: index loops mirror the reference math.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(unused_assignments)]
#![allow(dead_code)]

use std::io::Read;

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
        .ok_or("ids array required")?
        .iter()
        .filter_map(|v| v.as_u64().map(|x| x as u32))
        .collect();
    let n: usize = req["n"].as_u64().unwrap_or(64) as usize;

    let t0 = std::time::Instant::now();
    let g = tick_format::gguf::Gguf::open(std::path::Path::new(&model_path))?;
    // vocab strings for decode (byte-level: control chars map raw)
    let tokens = g.meta_str_vec("tokenizer.ggml.tokens").ok_or("tokens")?;
    println!(
        "loaded header in {:?} ({} vocab)",
        t0.elapsed(),
        tokens.len(),
    );

    let t1 = std::time::Instant::now();
    let mut model = tick_engine::model::Model::open(std::path::Path::new(&model_path), 1024)?;
    println!("weights decoded in {:?}", t1.elapsed());

    let t2 = std::time::Instant::now();
    let (out, _) = model.greedy(&ids, n)?;
    let n_gen: usize = out.iter().position(|&id| id == 128009).unwrap_or(out.len());
    let dt = t2.elapsed().as_secs_f32();
    println!(
        "generated {} tokens (stopped at eot) in {:.1}s ({:.2} tok/s)",
        n_gen,
        dt,
        n_gen as f32 / dt
    );
    println!("ids: {:?}", out);
    let out: Vec<u32> = out.iter().cloned().take_while(|&id| id != 128009).collect();
    let mut bytes: Vec<u8> = Vec::new();
    let unmap = gpt2_unmap();
    for &id in &out {
        let piece = tokens.get(id as usize).map(String::as_str).unwrap_or("");
        for c in piece.chars() {
            match unmap.get(&c) {
                Some(b) => bytes.push(*b),
                None => {
                    let mut buf = [0u8; 4];
                    bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
    }
    let text = String::from_utf8_lossy(&bytes);
    println!("text: {}", json_escape(&text));
    Ok(())
}

/// GPT-2 bytes_to_unicode inverse: char -> raw byte.
fn gpt2_unmap() -> std::collections::HashMap<char, u8> {
    let bs: Vec<u8> = (33..=126u8).chain(161..=172u8).chain(174..=255u8).collect();
    let mut forward = std::collections::HashMap::new();
    let mut n = 0u8;
    for b in 0..=255u8 {
        if bs.contains(&b) {
            forward.insert(b as u32, b as u32);
        } else {
            forward.insert(b as u32, 256 + n as u32);
            n += 1;
        }
    }
    forward
        .into_values()
        .map(|c| (char::from_u32(c).unwrap(), c as u8))
        .collect()
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out
}
