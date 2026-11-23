//! tick-engine — BitNet b1.58 inference: the forward pass mirroring
//! the frozen reference graph (ggml flash-attention one_chunk/tiled/
//! split-kv regimes, RMSNorm, NEOX RoPE, GQA, relu² FFN, tied f16
//! head), greedy generation, tickfile emission, and tick profiling.

// The forward pass mirrors the frozen reference graph op-for-op
// (including its loop shapes and constant spellings); keep lints from
// pushing mechanical rewrites of verified-parity code.
// unknown-lints: bridges older toolchains that predate newer lint names
#![allow(unknown_lints)]
#![allow(unused_assignments)]
#![allow(
    clippy::needless_range_loop,
    clippy::approx_constant,
    clippy::type_complexity,
    // chunks_exact (not as_chunks): keeps the older-toolchain portability
    // the reference-shaped rope/pair loops were written against.
    clippy::chunks_exact_to_as_chunks
)]

pub mod model;

pub use model::Model;
