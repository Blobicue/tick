//! tick-engine — BitNet b1.58 inference: the forward pass mirroring
//! the frozen reference graph (ggml flash-attention one_chunk/tiled/
//! split-kv regimes, RMSNorm, NEOX RoPE, GQA, relu² FFN, tied f16
//! head), greedy generation, tickfile emission, and tick profiling.

// The forward pass mirrors the frozen reference graph op-for-op
// (including its loop shapes and constant spellings); keep lints from
// pushing mechanical rewrites of verified-parity code.
#![allow(unused_assignments)]
#![allow(
    clippy::needless_range_loop,
    clippy::approx_constant,
    clippy::type_complexity
)]

pub mod model;

pub use model::Model;
