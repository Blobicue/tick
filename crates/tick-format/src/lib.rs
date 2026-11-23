//! tick-format — weight containers: the tickfile v0 native format
//! (layer-streamed, 4096-aligned; spec in SPEC_WEIGHTS.md) and the
//! GGUF v3 reader used to import from it. Writing requires
//! `tick-kernels` only via the data primitives re-exported here.
pub mod gguf;
pub mod weights;

pub use gguf::Gguf;
pub use weights::{TickFile, TickWriter};
