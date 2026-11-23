# tick-engine

BitNet b1.58 inference: the forward pass mirrors the frozen reference
graph op-for-op — ggml flash-attention one_chunk/tiled/split-kv
regimes, RMSNorm, NEOX RoPE, GQA, relu² FFN, tied f16 head — plus
greedy generation, tickfile emission, and tick-level profiling.

The battery in `tests/` proves output tokens unchanged; run
`cargo run --example tickcheck` (needs local model files) for
cross-format parity.
