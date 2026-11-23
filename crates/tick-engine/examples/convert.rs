//! convert — GGUF (tq1_0) -> tickfile. Usage: convert [in.gguf] [out.tick]

// Parity/training harness: index loops mirror the reference math.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(unused_assignments)]
#![allow(dead_code)]
fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let in_path = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("/home/juzz/models/bitnet/bitnet-b1.58-2B-4T-tq1_0.gguf");
    let out_path = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("/home/juzz/models/bitnet/bitnet-b1.58-2B-4T.tick");
    let t0 = std::time::Instant::now();
    let n = tick_format::weights::convert_gguf(
        std::path::Path::new(in_path),
        std::path::Path::new(out_path),
    )?;
    println!(
        "converted {n} tensor bytes in {:?} -> {out_path}",
        t0.elapsed()
    );
    Ok(())
}
