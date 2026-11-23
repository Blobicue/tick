# Contributing to tick

Thanks for helping. This repo has one unusual property you should know
before your first PR: **the inference arithmetic is parity-gated**, and
that shapes what kinds of change are welcome where.

## The gate culture

- The engine mirrors a frozen, sha256-pinned reference implementation
  op-for-op (flash-attention regimes, softmax order, f16 rounding,
  quantization estimators). The battery in
  `crates/tick-engine/tests/battery.rs` and the `tickcheck` example
  prove output tokens are unchanged.
- **Refactors of verified-parity code must keep tokens identical.**
  Where clippy would push mechanical rewrites of mirror code, we keep
  the reference shape and carry documented lint allowances instead
  (see the `#![allow]` blocks and their comments in each crate).
- Kernel/index tests (`tick-kernels`) and format round-trips
  (`tick-format`) run everywhere. Full-model parity tests need local
  model files and are skip-guarded: they run where models exist, and
  skip cleanly on CI runners.

## Layout

One crate per dependency, each with its own CI workflow
(`.github/workflows/ci-<crate>.yml`) triggered by changes under its
path. If your PR touches one crate, only that crate's pipeline runs —
plus the workspace aggregate when shared files change
(`Cargo.toml`, `Cargo.lock`, workflow definitions).

## Before you push

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

New public API needs a doc comment and at least one test that would
fail without it. New kernels need a bit-exactness test against a
scalar reference in the same PR.

## Commit style

`<crate>: <what changed>` — e.g. `tick-kernels: widen f16 subnormals
in dot_f16_sve`. Keep the parity implications in the body when they
exist ("token-identical on the battery" or "changes tokens because
…").

## Ports of the trainer

`tick-train` milestones (m41 memorization, m42 held-out eval, m43
batched QAT) are runnable examples with printed PASS gates. Changes to
training math should reproduce the printed gate, not just compile.

## License

Contributions are dual-licensed under MIT or Apache-2.0, at the
project's option.
