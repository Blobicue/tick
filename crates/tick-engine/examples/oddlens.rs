//! oddlens — dot_f16_sve vs ggml's ggml_vec_dot_f16 at every length
//! 1..300 on real V/K bits (leftover paths n>=17, n%16!=0 included).

// Parity/training harness: index loops mirror the reference math.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(unused_assignments)]
#![allow(dead_code)]
fn main() -> Result<(), String> {
    let content = std::fs::read_to_string("/tmp/odd_AB_bits.txt").map_err(|e| e.to_string())?;
    let gg: std::collections::HashMap<String, f64> = serde_json::from_str(
        &std::fs::read_to_string("/tmp/odd_lens.json").map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let mut bad = vec![];
    for line in content.lines() {
        let (n_s, a_s, b_s) = {
            let mut it = line.split('\t');
            let n = it.next().unwrap_or("");
            let a = it.next().unwrap_or("");
            let b = it.next().unwrap_or("");
            (n.to_string(), a.to_string(), b.to_string())
        };
        let parse = |s: &str| -> Vec<u16> {
            s.split_whitespace()
                .filter_map(|x| u16::from_str_radix(x, 16).ok())
                .collect()
        };
        let a = parse(&a_s);
        let b = parse(&b_s);
        let n: usize = n_s.parse().map_err(|_| "n")?;
        let mine = tick_kernels::bitlinear::dot_f16_sve(&a, &b);
        let ggml = *gg.get(&n_s).ok_or("missing ggml")? as f32;
        if mine != ggml {
            bad.push((n, mine, ggml));
        }
    }
    println!("odd-length dot check: {}/{} differ", bad.len(), gg.len());
    for (n, m, g) in bad.iter().take(8) {
        println!("  n={n}: mine {m:.9e} ggml {g:.9e}");
    }
    Ok(())
}
