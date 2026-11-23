//! tick-token — the in-crate llama-bpe tokenizer: a regex-free
//! scanner over codepoint flags (fuzz-verified against the frozen
//! reference ids), GPT-2 BPE merges, vocab loading from GGUF or
//! tickfile metadata, chat templating, and byte-level decoding.
pub mod token;
pub mod unicode;
mod unicode_data;

pub use token::{decode, encode, encode_chat, Vocab};
