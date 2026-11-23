# tick-pkg

The client for the tickfile package family on a tick-registry server:
weights ship like code, with names, versions, sha256 verification,
and yank semantics.

```bash
export TICK_PKG_REGISTRY=https://crates.gameladder.io
export TICK_PKG_TOKEN=<operator-issued token>

tick-pkg publish model.tick --name lighthouse-tiny --vers 0.2.0 \
    --meta arch=3x256,layers=3
tick-pkg list lighthouse-tiny
tick-pkg add lighthouse-tiny@0.2.0 --out ./models   # sha256 verified
tick-pkg info lighthouse-tiny@0.2.0                 # full index record
```
