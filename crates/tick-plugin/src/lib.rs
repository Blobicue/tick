//! tick-plugin — the community contract: write server plugins for
//! tick-server without touching its internals.
//!
//! A plugin contributes axum routes that are merged into the server's
//! router. Plugins never get direct access to the [`Model`](https://docs.rs/tick-engine)
//! — all inference goes through [`Handle`], which sends commands over
//! the same unbounded channel the core handlers use and reads instant
//! lock snapshots. That preserves the server's wedge-free law: a
//! plugin route can never park a handler thread on a generation.
//!
//! # Example
//!
//! ```
//! use tick_plugin::{Handle, TickPlugin};
//!
//! struct Hello;
//!
//! impl TickPlugin for Hello {
//!     fn name(&self) -> &'static str { "hello" }
//!     fn router(&self, handle: Handle) -> axum::Router {
//!         axum::Router::new().route(
//!             "/plugin/hello",
//!             axum::routing::get(move || async move {
//!                 let s = handle.status();
//!                 format!("hello: engine busy={}", s.busy)
//!             }),
//!         )
//!     }
//! }
//! ```
//!
//! Routes are merged after the core routes; using a reserved core
//! path (`/health`, `/v1/*`, `/tokenize`, `/telemetry`) panics at
//! startup. Protect your routes with [`Handle::authed`] to honor the
//! server's bearer key.
#![allow(unknown_lints)]

use std::sync::{Arc, RwLock};
use tick_token::Vocab;
use tokio::sync::{mpsc, oneshot};

/// A generation request as the engine executes it: token ids, a token
/// budget, and stop ids (generation halts before emitting them).
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub stop: Vec<u32>,
}

/// What a completed generation reports.
#[derive(Debug, Clone)]
pub struct GenerateReply {
    pub ids: Vec<u32>,
    pub text: String,
    pub prompt_len: usize,
    pub secs: f32,
    pub tok_s: f32,
}

/// The command protocol between handlers and the single inference
/// thread. Public so hosts other than tick-server can speak it.
pub enum EngineCommand {
    Generate {
        prompt: Vec<u32>,
        max_tokens: usize,
        stop: Vec<u32>,
        stream: Option<mpsc::UnboundedSender<String>>,
        done: oneshot::Sender<Result<GenerateReply, String>>,
    },
}

/// Live engine state, republished by the inference thread. Reads are
/// instant lock snapshots — never block on a generation.
#[derive(Default)]
pub struct EngineState {
    pub loaded: bool,
    pub busy: bool,
    pub inflight: usize,
    /// generation stats republished after each completion
    pub last_gen: Option<serde_json::Value>,
}

/// A point-in-time copy of [`EngineState`] for plugin telemetry.
#[derive(Debug, Clone, Default)]
pub struct EngineStatus {
    pub loaded: bool,
    pub busy: bool,
    pub inflight: usize,
}

/// The plugin's view of the running server. Clone-able and cheap to
/// hold; every method either sends a command or reads a snapshot.
#[derive(Clone)]
pub struct Handle {
    cmd: mpsc::UnboundedSender<EngineCommand>,
    state: Arc<RwLock<EngineState>>,
    vocab: Arc<Vocab>,
    key: Arc<String>,
    /// server-wide cap; `generate` clamps to it
    pub max_tokens_cap: usize,
    /// the stop id the chat template uses (u32::MAX when the vocab has
    /// no eot, e.g. byte-level test models)
    pub eot: u32,
}

impl Handle {
    /// Host-side constructor. Plugins receive a Handle; only the
    /// hosting server (or a test harness speaking the same command
    /// protocol) builds one.
    pub fn new(
        cmd: mpsc::UnboundedSender<EngineCommand>,
        state: Arc<RwLock<EngineState>>,
        vocab: Arc<Vocab>,
        key: Arc<String>,
        max_tokens_cap: usize,
        eot: u32,
    ) -> Handle {
        Handle {
            cmd,
            state,
            vocab,
            key,
            max_tokens_cap,
            eot,
        }
    }

    /// Full control: submit token ids, await the reply. The handler
    /// task waits on a oneshot — the inference thread is never
    /// blocked by this call.
    pub async fn generate(&self, req: GenerateRequest) -> Result<GenerateReply, String> {
        let (tx, rx) = oneshot::channel();
        self.cmd
            .send(EngineCommand::Generate {
                prompt: req.prompt,
                max_tokens: req.max_tokens.min(self.max_tokens_cap),
                stop: req.stop,
                stream: None,
                done: tx,
            })
            .map_err(|_| "engine stopped".to_string())?;
        rx.await
            .map_err(|_| "engine dropped the request".to_string())?
    }

    /// Text convenience: encode with the server's vocab (plain encode,
    /// no chat template — compose your own prompt formatting), stop at
    /// the server's eot id.
    pub async fn generate_text(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<GenerateReply, String> {
        let ids = tick_token::token::encode(prompt, &self.vocab);
        self.generate(GenerateRequest {
            prompt: ids,
            max_tokens,
            stop: vec![self.eot],
        })
        .await
    }

    /// Decode ids back to text with the server's vocab.
    pub fn decode(&self, ids: &[u32]) -> String {
        tick_token::token::decode(ids, &self.vocab)
    }

    /// Instant snapshot — safe to poll from any handler.
    pub fn status(&self) -> EngineStatus {
        let s = self.state.read().unwrap();
        EngineStatus {
            loaded: s.loaded,
            busy: s.busy,
            inflight: s.inflight,
        }
    }

    /// The last generation's published stats, if any.
    pub fn last_generation(&self) -> Option<serde_json::Value> {
        self.state.read().unwrap().last_gen.clone()
    }

    /// Bearer-key check identical to the core routes. Guard your
    /// routes with it unless they are intentionally public.
    pub fn authed(&self, headers: &axum::http::HeaderMap) -> bool {
        headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v == format!("Bearer {}", self.key))
    }
}

/// What a plugin provides: a name and a router. The router is merged
/// into the server's; reserved core paths panic at startup.
pub trait TickPlugin: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn router(&self, handle: Handle) -> axum::Router;
}
