//! probemirror — read vmixprobe-style input files (A f16 hex, B f32)
//! and print this engine's tinyblas_dot for comparison against
//! ggml's own mul_mat. args: k Afile Bfile

// Parity/training harness: index loops mirror the reference math.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(unused_assignments)]
#![allow(dead_code)]
fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let k: usize = args[1].parse().map_err(|_| "k")?;
    let a: Vec<u16> = std::fs::read_to_string(&args[2])
        .map_err(|e| e.to_string())?
        .lines()
        .filter_map(|l| u16::from_str_radix(l.trim(), 16).ok())
        .collect();
    let b: Vec<f32> = std::fs::read_to_string(&args[3])
        .map_err(|e| e.to_string())?
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();
    let rows = a.len() / k;
    let b16: Vec<u16> = b
        .iter()
        .map(|&f| tick_engine::model::Model::f32_to_f16_bits_pub(f))
        .collect();
    if std::env::var("DUMP_B16").is_ok() {
        use std::io::Write;
        let mut f = std::fs::File::create("/tmp/b16_rust.txt").unwrap();
        for &h in &b16 {
            writeln!(f, "{h:04x}").unwrap();
        }
    }
    for r in 0..rows {
        let row = &a[r * k..(r + 1) * k];
        let fb = tick_kernels::bitlinear::dot_f16_sve(row, &b16);
        let sv = tick_kernels::bitlinear::vec_dot_f16(
            &row.iter()
                .map(|&h| tick_kernels::bitlinear::half_to_f32(h))
                .collect::<Vec<f32>>(),
            &b16.iter()
                .map(|&h| tick_kernels::bitlinear::half_to_f32(h))
                .collect::<Vec<f32>>(),
        );
        println!("{fb:.9e}\t{sv:.9e}");
    }
    Ok(())
}
