# tick

A packaged ternary LLM stack: native weight format, bit-exact kernels,
in-crate tokenizer, inference engine, QAT trainer, plugin API, and an
OpenAI-compatible server — one crate per dependency, each with its own
CI pipeline.

Built for BitNet b1.58-class models on the formats that actually ship
(tq1_0 ternary weights, q8_K activations), with inference arithmetic
mirrored op-for-op from a frozen reference implementation and gated by
token-exact parity tests.

## Crates

| crate | what it owns | depends on |
|---|---|---|
| [`tick-kernels`](crates/tick-kernels) | `Tri`/`TernaryMat` 2-bit vocabulary, tq1_0 packing + encode/decode, q8_K quantization, NEON/SVE dot kernels, f16 helpers, lock-free pool | — |
| [`tick-format`](crates/tick-format) | tickfile v0 native format (layer-streamed, 4096-aligned, mmap reads) + GGUF v3 import reader ([spec](crates/tick-format/SPEC_WEIGHTS.md)) | — |
| [`tick-token`](crates/tick-token) | llama-bpe tokenizer: regex-free scanner over codepoint flags, GPT-2 merges, chat templates | tick-format |
| [`tick-engine`](crates/tick-engine) | BitNet b1.58 inference: flash-attention regimes (one_chunk / tiled / split-kv), greedy generation, tickfile emission, profiling | tick-kernels, tick-format |
| [`tick-train`](crates/tick-train) | from-scratch ternary QAT: abs-mean ternary view + straight-through estimator, q8 activation STE, best-val checkpointing | tick-kernels, tick-format |
| [`tick-server`](crates/tick-server) | wedge-free OpenAI-compatible server (`/v1/chat/completions` stream + non-stream, `/tokenize`, `/telemetry`) | tick-engine, tick-token |

Each crate has its own CI workflow (`.github/workflows/ci-<crate>.yml`)
that runs on changes to its path only.

## Quickstart: serve a model

```bash
cargo install --git https://github.com/Blobicue/tick tick-server
TICK_MODEL=/path/to/bitnet-b1.58-2B-4T.tick tick-server --port 8096
```

The server accepts a tickfile (preferred) or a GGUF, resolves the
vocab from a sibling `<name>-tq1_0.gguf` when loading a tickfile, and
authenticates with a bearer key from `~/.config/omarchy-fleet/api-key`
(override with `TICK_FLEET_KEY`).

## Quickstart: inference in Rust

```rust
use tick_engine::Model;

let mut m = Model::open("/path/to/model.tick".as_ref(), 4096)?;
let (ids, _) = m.greedy(&prompt_ids, 64)?;
```

## Quickstart: train

`tick-train` trains ternary models whose checkpoints the inference
engine runs directly — the trainer's quantized forward matches engine
inference to within thousandths of a nat. See
[`crates/tick-train/examples`](crates/tick-train/examples) for the
memorization, held-out-eval, and batched-QAT milestones (m41–m43).

## Parity discipline

The inference arithmetic mirrors a frozen, sha256-pinned reference
build. Token-exact gates (`tickcheck`, the battery in
[`tick-engine/tests`](crates/tick-engine/tests/battery.rs)) prove the
engine's output unchanged across formats and refactors. Kernels carry
crate-level lint allowances where idiom rewrites would churn
verified-parity code — see the comments in each `lib.rs`.

## License

Dual-licensed under MIT or Apache-2.0, at your option.
