# tickfile — the tick-engine weight format (v0)

The engine's native artifact. GGUF is an import path (via the frozen
BitNet lane); tickfile is what the engine, and eventually everything
else (trainer, cone/genome mutators, dashboards), reads and writes.

Design law: **the file stores what the engine loads** — no generic
container generality, no format features without a consumer.

## Layout (little-endian)

```
0   .. 4    magic "TICK"
4   .. 8    u32 version (0)
8   .. 16   u64 kv_offset
16  .. 24   u64 tensor_dir_offset
24  .. 32   u64 data_offset
32  .. 40   u64 file_len

KV section @kv_offset:
    u32 count
    entry: u32 key_len, key utf8, u8 vtype, value
    vtype: 0 str (u32 len + bytes) | 1 u64 | 2 f64 | 3 bool
           4 arr_str (u32 n, then n str) | 5 arr_u64 (u32 n, then n u64)

Tensor directory @tensor_dir_offset:
    u32 count
    entry: u32 name_len, name utf8, u32 dtype, u32 ndims,
           ndims x u64 dims (ne[0] = contiguous dim),
           u64 data_offset, u64 data_len
    dtype: 0 f32 | 1 f16 | 2 ternary (tq blocks)

Data @data_offset: raw tensor bytes in directory order.
```

## Tensor encodings

- **f32** — raw; norms and other vectors.
- **f16** — raw; the tied embedding/head matrix `[hidden, vocab]`.
- **ternary (dtype 2)** — proven tq blocks, verbatim: per 256-weight
  block, 48 bytes of 5-trits-per-byte (ceil-div base-3), 4 bytes of
  4-trits-per-byte, then an f16 block scale — 54 bytes/block, blocks
  in (row, block) row-major order = **exactly the tq1_0 layout**.
  Conversion from tq1_0 GGUF is a byte-copy; decode is the existing
  `decode_tq1_0` (1.69 bits/weight, cache-friendly per SPEC.md R3).

## KV content (v0)

- `tick.source_path`, `tick.source_bytes`, `tick.converted_at` —
  provenance (sha256 lands with the manifest tool).
- The model config keys the engine needs (`bitnet.*`), copied from
  the source GGUF.
- The tokenizer keys (`tokenizer.ggml.*`: tokens, merges, bos/eos,
  chat template) so the artifact is self-contained; the in-crate BPE
  encoder that consumes them is the v0.1 milestone (the python seam
  in `scripts/tok.py` retires then).

## v0.2 — layer-streamed layout (shipped 2026-11-23)

Version field 2. The tensor data section is written in LOAD order —
`token_embd.weight`, then blk.0..blk.N with every layer's tensors
contiguous and each data block 4096-byte aligned, then the output
norms — so one layer's weights occupy a single padded window that can
be mmap'd and decoded independently (the R3 per-layer streaming
regime). The directory is unchanged (offsets carry the alignment);
readers accept version 0 and 2 alike — the writer's
`finish_ordered(path, layer_order=false)` still emits the v0 packing
for tests.

With v0.2 the loader decodes layers in parallel across the tick pool
(the tq1_0 unpack is pure CPU and layers are independent) and widens
the head's 328M f16 elements in parallel too: cold load drops from
~1.9s to ~0.9s on the GB10.

## v0.3 seams (shipped 2026-11-23)

The trainer's emission path is closed end-to-end:

- **`encode_tq1_0`** — the exact inverse of `decode_tq1_0`: a
  243-entry byte table built by running the decode over all 256
  candidate bytes (the closed form is ggml's ceil-division chain;
  the table sidesteps it and is exact by construction). Fuzz-tested
  round-trip over random trits + f16-pre-rounded scales.
- **`Model::save_tickfile`** — any in-memory Model (loaded, trained,
  mutated) writes a v0.2 layer-streamed tickfile with the config KV.
  The reemit gate proves the loop: GGUF → Model → tickfile → reload
  gives token-identical generation and 10/10 battery text on both
  loads (`examples/reemit`; permanent test in `tests/battery.rs`).
- **mmap `read_raw`** — TickFile maps the whole artifact read-only;
  raw reads are now slices of the mapping (no pread syscalls; the
  decode copy remains). M4's "emit tickfile, no gguf in the training
  loop" has no remaining format dependency.

## Roadmap

- ~~v0.1~~ — in-crate BPE (shipped); sha256 provenance.
- ~~v0.2~~ — layer-streamed layout (shipped, above).
- ~~v0.3 seams~~ — encoder + save_tickfile + mmap reader (shipped,
  above); remaining: cone/genome sidecar section (truth-gated fact
  registry hooks).
- **M4** — the 25M QAT ternary training burn now has its whole
  emission path ready; it writes tickfile directly.

## Contract

The engine loads GGUF (compat) and tickfile; parity gates run against
the frozen lane regardless of source. Everything new builds on
tickfile only.
