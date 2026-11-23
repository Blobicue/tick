//! tick-server — the wedge-free OpenAI-compatible server: a single
//! dedicated inference thread owns the Model, handlers command it via
//! an unbounded mpsc channel and read instant RwLock snapshots, so
//! /health and /telemetry never wait behind generation.
pub mod server;
