//! trace1 — my intermediates for token 128000 @ pos 0 (tensorprobe diff).

// Parity/training harness: index loops mirror the reference math.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(unused_assignments)]
#![allow(dead_code)]
fn main() -> Result<(), String> {
    let path = "/home/juzz/models/bitnet/bitnet-b1.58-2B-4T-tq1_0.gguf";
    let _ = std::fs::remove_file("/tmp/mine_tensors.txt");
    let mut m = tick_engine::model::Model::open(std::path::Path::new(path), 64)?;
    m.trace = Some("/tmp/mine_tensors.txt".into());
    let mut hh = None;

    let _kf = std::fs::File::create("/tmp/kq_p1_head0.txt").unwrap();
    for i in 0..2 {
        hh = Some(m.forward_token([128000u32, 791][i] as usize, i)?);
    }
    let hidden = hh.unwrap();
    let rn = tick_engine::model::rms_norm(&hidden, &m.embd_norm, m.cfg.rms_eps);
    m.dump_trace("result_norm", &rn);
    Ok(())
}
