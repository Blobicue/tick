# tick-plugin

The community contract for tick-server plugins: contribute axum
routes, drive inference through a `Handle` that can never park a
handler on a generation, and read instant engine snapshots.

- `Handle::generate` / `generate_text` — submit work over the same
  unbounded command channel the core routes use (oneshot reply).
- `Handle::status` / `last_generation` — lock-snapshot telemetry.
- `Handle::authed` — the server's bearer-key check, so plugin routes
  can honor the same credential.

Hosts other than tick-server can speak `EngineCommand` directly.
