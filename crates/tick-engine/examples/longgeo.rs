//! longgeo — end-to-end geometry gates: raw prompt text -> in-crate
//! tokenize -> greedy -> text. args: <ngens> (prompt on stdin).

// Parity/training harness: index loops mirror the reference math.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(unused_assignments)]
#![allow(dead_code)]
fn main() -> Result<(), String> {
    let n_gen: usize = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "64".into())
        .parse()
        .map_err(|_| "n")?;
    let mut prompt = String::new();
    use std::io::Read;
    std::io::stdin()
        .read_to_string(&mut prompt)
        .map_err(|e| e.to_string())?;
    let prompt = prompt.trim().to_string();
    let g = tick_format::gguf::Gguf::open(std::path::Path::new(
        "/home/juzz/models/bitnet/bitnet-b1.58-2B-4T-tq1_0.gguf",
    ))?;
    let vocab = tick_token::token::Vocab::from_gguf(&g)?;
    let ids = tick_token::token::encode_chat(&prompt, &vocab);
    eprintln!("prompt ids: {}", ids.len());
    let t0 = std::time::Instant::now();
    let mut m = tick_engine::model::Model::open(
        std::path::Path::new("/home/juzz/models/bitnet/bitnet-b1.58-2B-4T-tq1_0.gguf"),
        1024,
    )?;
    let (out, _) = m.greedy(&ids, n_gen)?;
    let dt = t0.elapsed().as_secs_f32();
    let gen: Vec<u32> = out.iter().cloned().take_while(|&t| t != 128009).collect();
    eprintln!(
        "generated {} tokens in {dt:.1}s ({:.2} tok/s)",
        gen.len(),
        gen.len() as f32 / dt
    );
    println!("{}", tick_token::token::decode(&gen, &vocab));
    Ok(())
}
