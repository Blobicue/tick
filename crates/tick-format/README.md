# tick-format

Weight containers: the tickfile v0 native format (layer-streamed,
4096-byte aligned windows, mmap reads — [spec](SPEC_WEIGHTS.md)) and
the GGUF v3 reader used for import. Zero Rust dependencies beyond
libc; tickfile writing consumes packed ternary blocks as raw bytes.
