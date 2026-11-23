# tick-registry

Our distribution backend — the cargo sparse registry protocol on plain
filesystem storage, plus the tickfile package family.

## Wire compatibility

A stock cargo speaks this server natively:

```bash
# clients: ~/.cargo/config.toml
#   [registries.tick]
#   index = "sparse+https://crates.gameladder.io/idx/"
cargo login --registry tick <token>
cargo publish --registry tick
cargo add tick-engine --registry tick
```

Implements: `config.json` (`dl`/`api`), the sparse index layout with
ETag/304 conditional GETs, the length-prefixed binary publish stream
(not HTTP multipart), duplicate-version rejection, yank/unyank (the
protocol's only post-insert mutation), owners, search, and the
`{crate}/{version}/download` route with sha256 verification.

`tests/cargo_e2e.rs` is the conformance gate — it drives a real cargo
against a temp-hosted instance: publish, depend, build, duplicate,
yank, 304.

## Auth

Tokens are operator-minted (`REGISTRY_ROOT=/srv/tick-registry tick-registry
token add <login>`) and stored hashed; cargo sends the token verbatim
in the `Authorization` header (no Bearer scheme — accepted both ways).

## Tickfile family

The `/pkg-index/` + `/artifacts/` tree and the `tick-pkg` CLI
distribute tickfile models/adapters with the same index discipline
(names, versions, sha256 cksums, yank) so weights ship like code.
