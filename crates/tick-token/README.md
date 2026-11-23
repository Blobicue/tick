# tick-token

The in-crate llama-bpe tokenizer: a regex-free scanner over codepoint
flag tables, GPT-2 BPE merges, vocab loading from GGUF or tickfile
metadata, chat templating, and byte-level decoding. Fuzz-verified
byte-identical to the frozen reference ids on both vocab sources.
