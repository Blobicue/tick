//! The parity battery: the frozen golden outputs from the reference
//! lane (patched bitnet.cpp, BitNet b1.58-2B-4T, greedy temp 0).
//!
//! M1 gate: the fixture exists, parses, and carries the reference
//! metadata (binary/model hashes) — the anchor our engine will be held
//! to. The actual text/token parity assertions activate in M2 when the
//! forward pass exists; until then this test FAILS if someone moves,
//! edits, or regenerates the battery without the frozen lane.

// Parity/training harness: index loops mirror the reference math.
#![allow(clippy::needless_range_loop)]
// `!(x < t)` NaN-safe gates are deliberate (law: NaN comparisons are false).
#![allow(clippy::neg_cmp_op_on_partial_ord)]
#![allow(clippy::manual_memcpy)]
#![allow(dead_code)]

use serde_json::Value;

const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/battery.json");

#[test]
fn golden_battery_is_present_and_well_formed() {
    let raw = std::fs::read_to_string(FIXTURE).expect(
        "battery.json missing — regenerate ONLY with scripts/gen_battery.sh on the frozen lane",
    );
    let d: Value = serde_json::from_str(&raw).expect("battery.json unparseable");

    let meta = d["meta"].as_object().expect("meta object");
    for key in ["llama_cli_sha256", "model_sha256", "flags", "temp"] {
        assert!(meta.contains_key(key), "meta.{key} missing");
    }
    // the lane is frozen: these hashes must never drift silently
    // (llama-patched tree, ReLU^2 activation — see SPEC.md §1)
    assert_eq!(
        meta["llama_cli_sha256"].as_str().unwrap(),
        "058faca2008fd07c6fad7ba57b95f4584db9bd147602eb21900d51542c9d715b",
        "reference binary hash changed — the lane is no longer the frozen one"
    );

    let cases = d["cases"].as_array().expect("cases array");
    assert!(cases.len() >= 8, "battery suspiciously small");
    for case in cases {
        assert!(case["prompt"].as_str().is_some_and(|p| !p.is_empty()));
        let out = case["output"].as_str().expect("output text");
        assert!(!out.is_empty(), "empty golden output");
        assert!(case["tokens"].as_u64().unwrap_or(0) > 0, "token count");
    }
}

#[test]
fn converted_tickfile_exists_and_matches() {
    // The format contract: the converted artifact sits next to its
    // source and the tickcheck example gates token parity. Light
    // check here (full gate is `cargo run --example tickcheck`).
    let p = "/home/juzz/models/bitnet/bitnet-b1.58-2B-4T.tick";
    if !std::path::Path::new(p).exists() {
        return; // not converted on this machine — skip
    }
    let t = tick_format::weights::TickFile::open(std::path::Path::new(p)).expect("parse");
    assert_eq!(t.meta_u64("bitnet.block_count"), Some(30));
    assert!(t.dir.contains_key("blk.0.attn_q.weight"));
    assert!(t.dir.contains_key("token_embd.weight"));
    assert_eq!(t.dir.len(), 332);
}

#[test]
fn in_crate_tokenizer_matches_frozen_ids() {
    // Phase 2 gate: token.rs (llama-bpe scanner + BPE + the lane's
    // custom chat template) must reproduce every frozen id fixture
    // byte-for-byte. Skipped off-box (needs the model file).
    let gguf = "/home/juzz/models/bitnet/bitnet-b1.58-2B-4T-tq1_0.gguf";
    if !std::path::Path::new(gguf).exists() {
        return;
    }
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures");
    let raw = std::fs::read_to_string(format!("{dir}/battery.json")).expect("battery.json");
    let d: Value = serde_json::from_str(&raw).expect("parse");
    let g = tick_format::gguf::Gguf::open(std::path::Path::new(gguf)).expect("gguf");
    let vocab = tick_token::token::Vocab::from_gguf(&g).expect("vocab");
    let tick = "/home/juzz/models/bitnet/bitnet-b1.58-2B-4T.tick";
    let v2 = if std::path::Path::new(tick).exists() {
        let tf = tick_format::weights::TickFile::open(std::path::Path::new(tick)).expect("tick");
        Some(tick_token::token::Vocab::from_tickfile(&tf).expect("tick vocab"))
    } else {
        None
    };
    for case in d["cases"].as_array().expect("cases") {
        let id = case["id"].as_str().unwrap();
        let prompt = case["prompt"].as_str().unwrap();
        let fixture: Vec<u32> = serde_json::from_str(
            &std::fs::read_to_string(format!("{dir}/{id}_ids.json")).expect("ids fixture"),
        )
        .expect("ids parse");
        let got = tick_token::token::encode_chat(prompt, &vocab);
        assert_eq!(
            got, fixture,
            "{id}: in-crate encode diverges from frozen ids"
        );
        if let Some(v2) = &v2 {
            let got2 = tick_token::token::encode_chat(prompt, v2);
            assert_eq!(got2, fixture, "{id}: tickfile vocab encode diverges");
        }
    }
}

#[test]
fn reemit_trainer_path_roundtrip() {
    // v0.3 seam: in-memory Model -> save_tickfile -> reload must be
    // token-identical and battery-exact. Skipped off-box.
    let gguf = "/home/juzz/models/bitnet/bitnet-b1.58-2B-4T-tq1_0.gguf";
    if !std::path::Path::new(gguf).exists() {
        return;
    }
    let out = "/tmp/tick_reemit_test.tick";
    let mut m = tick_engine::model::Model::open(std::path::Path::new(gguf), 256).expect("load");
    m.save_tickfile(std::path::Path::new(out), gguf)
        .expect("save");
    let mut m2 = tick_engine::model::Model::open(std::path::Path::new(out), 256).expect("reload");
    assert_eq!(m2.embd, m.embd, "embd bits differ across reemit");
    for il in [0, 15, 29] {
        assert_eq!(
            m2.layers[il].wq.packed, m.layers[il].wq.packed,
            "layer {il} trits differ"
        );
        assert_eq!(
            m2.layers[il].down.d, m.layers[il].down.d,
            "layer {il} scales differ"
        );
    }
    // one prompt through both loads, token-identical
    let g = tick_format::gguf::Gguf::open(std::path::Path::new(gguf)).expect("gguf");
    let vocab = tick_token::token::Vocab::from_gguf(&g).expect("vocab");
    let ids = tick_token::token::encode_chat("What is 9 times 6?", &vocab);
    let (o1, _) = m.greedy(&ids, 16).expect("gen1");
    let (o2, _) = m2.greedy(&ids, 16).expect("gen2");
    assert_eq!(o1, o2, "re-emitted model diverges");
    let _ = std::fs::remove_file(out);
}
