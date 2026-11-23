# tick-server

The wedge-free OpenAI-compatible server: a single dedicated inference
thread owns the Model, handler tasks command it over an unbounded
channel and read instant lock snapshots — `/health` and `/telemetry`
never wait behind generation, proven live during active streams.

Endpoints: `/v1/chat/completions` (stream + non-stream), `/v1/models`,
`/tokenize`, `/health`, `/telemetry`. Serves tickfile or GGUF.
