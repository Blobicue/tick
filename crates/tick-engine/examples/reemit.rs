//! reemit — the v0.3 trainer-path gate: load the GGUF, save_tickfile,
//! reload the re-emitted artifact, and run the same battery prompts on
//! both loads. Text must match the frozen goldens AND the two loads
//! must agree token-for-token (what the M4 trainer's output will face).

// Parity/training harness: index loops mirror the reference math.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(unused_assignments)]
#![allow(dead_code)]
fn main() -> Result<(), String> {
    let gguf = "/home/juzz/models/bitnet/bitnet-b1.58-2B-4T-tq1_0.gguf";
    let out = "/tmp/reemitted.tick";
    let t0 = std::time::Instant::now();
    let mut m = tick_engine::model::Model::open(std::path::Path::new(gguf), 512)?;
    println!("gguf load: {:?}", t0.elapsed());
    let t1 = std::time::Instant::now();
    m.save_tickfile(std::path::Path::new(out), gguf)?;
    println!(
        "save_tickfile: {:?} ({})",
        t1.elapsed(),
        std::fs::metadata(out).unwrap().len()
    );
    // reload the re-emitted artifact
    let t2 = std::time::Instant::now();
    let mut m2 = tick_engine::model::Model::open(std::path::Path::new(out), 512)?;
    println!("reemit load: {:?}", t2.elapsed());
    // sanity: vocab + embd bits identical
    assert_eq!(m2.cfg.vocab, m.cfg.vocab);
    assert_eq!(m2.embd, m.embd, "embd bits differ");
    for il in [0, 15, 29] {
        assert_eq!(
            m2.layers[il].wq.packed, m.layers[il].wq.packed,
            "layer {il} wq packed differ"
        );
        assert_eq!(
            m2.layers[il].down.d, m.layers[il].down.d,
            "layer {il} down scales differ"
        );
    }
    println!("tensor bits identical");

    // battery prompts on both loads
    let g = tick_format::gguf::Gguf::open(std::path::Path::new(gguf))?;
    let vocab = tick_token::token::Vocab::from_gguf(&g)?;
    let battery: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/tick-engine/battery.json"
        ))
        .unwrap(),
    )
    .map_err(|e| e.to_string())?;
    let mut pass_orig = 0;
    let mut pass_re = 0;
    let mut both = 0;
    let total = battery["cases"].as_array().unwrap().len();
    for case in battery["cases"].as_array().unwrap() {
        let prompt = case["prompt"].as_str().unwrap();
        let golden = case["output"].as_str().unwrap();
        let ids = tick_token::token::encode_chat(prompt, &vocab);
        let (o1, _) = m.greedy(&ids, 64)?;
        let (o2, _) = m2.greedy(&ids, 64)?;
        let detok = |out: &[u32]| {
            let gen: Vec<u32> = out.iter().cloned().take_while(|&t| t != 128009).collect();
            tick_token::token::decode(&gen, &vocab).trim().to_string()
        };
        let t1 = detok(&o1);
        let t2 = detok(&o2);
        if t1 == golden.trim() {
            pass_orig += 1;
        }
        if t2 == golden.trim() {
            pass_re += 1;
        }
        if t1 == t2 && o1 == o2 {
            both += 1;
        }
    }
    println!("original: {pass_orig}/{total}  re-emitted: {pass_re}/{total}  token-identical: {both}/{total}");
    if pass_orig != total || pass_re != total || both != total {
        return Err("reemit gate failed".into());
    }
    println!(
        "REEMIT GATE: PASS — the trainer path (in-memory Model -> tickfile -> engine) is closed"
    );
    Ok(())
}
