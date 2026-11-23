//! The tick server on :8096 — OpenAI-compatible subset over the
//! tick engine: /v1/chat/completions (non-stream + SSE), /v1/models,
//! /tokenize, /telemetry. The fleet key authorizes (/health stays
//! open for no-auth probes).
//!
//! Concurrency law (async-handlers-never-park, proven on the
//! spark-engine :8095 postmortem): the Model lives on ONE dedicated
//! inference thread; async handlers never touch it. Handlers send
//! commands over a channel (awaited, not parked) and read state from
//! an instant shared snapshot; nothing on the runtime ever blocks on
//! model math.
//!
//! Chat rendering follows the model's own CUSTOM gguf chat_template
//! (role|capitalize + ': ' + content|trim + '<|eot_id|>' per message,
//! then 'Assistant: ') — the same template encode_chat applies, so
//! served generations are byte-identical to the battery goldens.

use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json, Router,
};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tick_engine::model::Model;
use tick_format::gguf::Gguf;
use tick_token::token::{decode, encode, Vocab};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::StreamExt as _;

use tick_plugin::{EngineCommand, EngineState, GenerateReply, Handle, TickPlugin};

pub struct App {
    cmd_tx: mpsc::UnboundedSender<EngineCommand>,
    shared: Arc<RwLock<EngineState>>,
    pub vocab: Arc<Vocab>,
    pub eot: u32,
    pub key: String,
    pub model_path: PathBuf,
    pub kv_cap: usize,
    pub max_tokens_cap: usize,
}

impl App {
    /// The plugin-facing view: same command channel, same snapshots.
    pub fn handle(self: &Arc<Self>) -> Handle {
        Handle::new(
            self.cmd_tx.clone(),
            Arc::clone(&self.shared),
            Arc::clone(&self.vocab),
            Arc::new(self.key.clone()),
            self.max_tokens_cap,
            self.eot,
        )
    }

    pub fn open(
        model_path: PathBuf,
        kv_cap: usize,
        max_tokens_cap: usize,
    ) -> Result<Arc<App>, String> {
        let key = std::fs::read_to_string(
            std::env::var("HOME").unwrap_or_else(|_| "/home/juzz".into())
                + "/.config/omarchy-fleet/api-key",
        )
        .map_err(|e| format!("fleet key: {e}"))?
        .trim()
        .to_string();
        // the vocab always comes from the GGUF (identical in the
        // tickfile; the GGUF is always present on this machine)
        // the vocab source: the gguf next to a tickfile, or the file
        // itself when it already is one
        let gguf = if model_path.extension().and_then(|e| e.to_str()) == Some("tick") {
            // bitnet-b1.58-2B-4T.tick -> bitnet-b1.58-2B-4T-tq1_0.gguf
            let stem = model_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            model_path.with_file_name(format!("{stem}-tq1_0.gguf"))
        } else {
            model_path.clone()
        };
        // vocab source: the sibling gguf; when absent (e.g. a
        // standalone tiny tickfile from tick-train), fall back to a
        // byte-level vocab where id == byte value
        let (vocab, eot) = if gguf.exists() {
            let g = Gguf::open(&gguf)?;
            let v = Arc::new(Vocab::from_gguf(&g)?);
            let eot = v.token_to_id.get("<|eot_id|>").copied().unwrap_or(128009);
            (v, eot)
        } else {
            (Arc::new(Vocab::byte_level()), u32::MAX)
        };
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        let shared = Arc::new(RwLock::new(EngineState {
            loaded: false,
            busy: false,
            inflight: 0,
            last_gen: None,
        }));
        let thread_shared = Arc::clone(&shared);
        let thread_model = model_path.clone();
        let thread_vocab = Arc::clone(&vocab);
        std::thread::Builder::new()
            .name("tick-inference".into())
            .spawn(move || {
                inference_loop(thread_model, kv_cap, cmd_rx, thread_shared, thread_vocab);
            })
            .map_err(|e| format!("inference thread: {e}"))?;
        Ok(Arc::new(App {
            cmd_tx,
            shared,
            vocab,
            eot,
            key,
            model_path,
            kv_cap,
            max_tokens_cap,
        }))
    }
}

/// The single thread that owns the Model. Loads weights lazily on the
/// first Generate (so /health and /tokenize answer immediately at
/// boot), publishes snapshots into `shared`, answers commands.
fn inference_loop(
    model_path: PathBuf,
    kv_cap: usize,
    mut cmd_rx: mpsc::UnboundedReceiver<EngineCommand>,
    shared: Arc<RwLock<EngineState>>,
    vocab: Arc<Vocab>,
) {
    let mut model: Option<Model> = None;
    while let Some(cmd) = cmd_rx.blocking_recv() {
        let EngineCommand::Generate {
            prompt,
            max_tokens,
            stop,
            stream,
            done,
        } = cmd;
        if model.is_none() {
            match Model::open(&model_path, kv_cap) {
                Ok(m) => model = Some(m),
                Err(e) => {
                    let _ = done.send(Err(e));
                    continue;
                }
            }
        }
        let prompt_len = prompt.len();
        {
            let mut s = shared.write().unwrap();
            s.loaded = true;
            s.busy = true;
            s.inflight = 0;
        }
        let t0 = Instant::now();
        let shared_ref = &shared;
        let stream_ref = &stream;
        let stop_ref = &stop;
        let mut inflight = 0usize;
        let res = model
            .as_mut()
            .unwrap()
            .greedy_until(&prompt, max_tokens, &stop, &mut |id| {
                inflight += 1;
                if let Ok(mut s) = shared_ref.write() {
                    s.inflight = inflight;
                }
                if let Some(tx) = stream_ref {
                    if !stop_ref.contains(&id) {
                        // per-token delta (detok of one id)
                        let _ = tx.send(decode(&[id], &vocab));
                    }
                }
            });
        let secs = t0.elapsed().as_secs_f32();
        let n = res.as_ref().map(|(ids, _)| ids.len()).unwrap_or(0);
        let reply = res.map(|(ids, _)| GenerateReply {
            text: decode(&ids, &vocab),
            ids,
            prompt_len,
            secs,
            tok_s: if secs > 0.0 { n as f32 / secs } else { 0.0 },
        });
        let _ = done.send(reply);
        {
            let mut s = shared.write().unwrap();
            s.busy = false;
            s.inflight = 0;
            if let Some(m) = &model {
                s.last_gen = Some(json!({
                    "ticks": m.telemetry.tick,
                    "layers": m.cfg.n_layer,
                    "kv_cap": kv_cap,
                    "pool_threads": tick_kernels::par::threads(),
                }));
            }
        }
    }
}

/// The core routes (stateful) merged with every plugin's router.
/// Route conflicts with reserved core paths panic here, at startup,
/// where they belong.
pub fn build_router(app: Arc<App>, plugins: Vec<Box<dyn TickPlugin>>) -> Router {
    let mut router = Router::new()
        .route("/health", axum::routing::get(health))
        .route("/v1/models", axum::routing::get(models))
        .route("/tokenize", axum::routing::post(tokenize))
        .route("/v1/chat/completions", axum::routing::post(chat))
        .route("/telemetry", axum::routing::get(telemetry_snapshot))
        .with_state(app.clone());
    for p in plugins {
        router = router.merge(p.router(app.handle()));
    }
    router
}

/// Serve with plugins mounted. The canonical entry for hosts that
/// compose their own binary.
pub async fn serve_with(
    bind: &str,
    app: Arc<App>,
    plugins: Vec<Box<dyn TickPlugin>>,
) -> Result<(), String> {
    let router = build_router(app, plugins);
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .map_err(|e| format!("bind {bind}: {e}"))?;
    axum::serve(listener, router)
        .await
        .map_err(|e| e.to_string())
}

pub async fn serve(bind: &str, app: Arc<App>) -> Result<(), String> {
    serve_with(bind, app, Vec::new()).await
}

fn authed(headers: &HeaderMap, key: &str) -> bool {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v == format!("Bearer {key}"))
}

fn err(status: StatusCode, msg: &str) -> (StatusCode, Json<Value>) {
    (status, Json(json!({"error": {"message": msg}})))
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn models() -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{"id": "tick-engine-bitnet-2b4t", "object": "model", "owned_by": "tick"}]
    }))
}

async fn tokenize(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    if !authed(&headers, &app.key) {
        return Err(err(StatusCode::UNAUTHORIZED, "unauthorized"));
    }
    let content = body["content"].as_str().unwrap_or_default();
    Ok(Json(json!({"tokens": encode(content, &app.vocab)})))
}

struct ChatRequest {
    messages: Vec<(String, String)>,
    max_tokens: usize,
    stream: bool,
}

fn parse_chat(body: &Value) -> Result<ChatRequest, String> {
    let msgs = body["messages"]
        .as_array()
        .ok_or("messages array required")?;
    let mut out = Vec::with_capacity(msgs.len());
    for m in msgs {
        let role = m["role"].as_str().unwrap_or("user").to_string();
        let content = m["content"].as_str().unwrap_or_default().to_string();
        out.push((role, content));
    }
    Ok(ChatRequest {
        messages: out,
        max_tokens: body["max_tokens"].as_u64().unwrap_or(64) as usize,
        stream: body["stream"].as_bool().unwrap_or(false),
    })
}

/// The model's own chat template (gguf chat_template): per message
/// `{Role}: {content|trim}<|eot_id|>`, then `Assistant: ` — plus BOS.
/// encode_chat handles the single-turn form; this generalizes to N
/// messages with the same byte contract.
fn render_chat(messages: &[(String, String)]) -> String {
    let mut s = String::new();
    for (role, content) in messages {
        // jinja `capitalize`: first char upper, rest lower
        let mut cs = role.chars();
        let r = match (cs.next(), cs.as_str()) {
            (Some(f), rest) => format!("{}{}", f.to_uppercase(), rest.to_lowercase()),
            (None, _) => String::new(),
        };
        let c = content.trim();
        s.push_str(&format!("{r}: {c}<|eot_id|>"));
    }
    s.push_str("Assistant: ");
    s
}

async fn chat(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, (StatusCode, Json<Value>)> {
    if !authed(&headers, &app.key) {
        return Err(err(StatusCode::UNAUTHORIZED, "unauthorized"));
    }
    let req = parse_chat(&body).map_err(|e| err(StatusCode::BAD_REQUEST, &e))?;
    let max_tokens = req.max_tokens.min(app.max_tokens_cap);
    let prompt = render_chat(&req.messages);
    let prompt_tokens: Vec<u32> = {
        // BOS + template — identical to encode_chat on single-turn
        let bos = app
            .vocab
            .token_to_id
            .get("<|begin_of_text|>")
            .copied()
            .unwrap_or(128000);
        let mut ids = vec![bos];
        ids.extend(encode(&prompt, &app.vocab));
        ids
    };
    if prompt_tokens.len() + max_tokens >= app.kv_cap {
        return Err(err(StatusCode::BAD_REQUEST, "sequence exceeds kv cap"));
    }

    let (stream_tx, stream_rx) = if req.stream {
        let (tx, rx) = mpsc::unbounded_channel::<String>();
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };
    let (done_tx, done_rx) = oneshot::channel();
    app.cmd_tx
        .send(EngineCommand::Generate {
            prompt: prompt_tokens,
            max_tokens,
            stop: vec![app.eot],
            stream: stream_tx,
            done: done_tx,
        })
        .map_err(|_| err(StatusCode::INTERNAL_SERVER_ERROR, "engine stopped"))?;

    if let Some(rx) = stream_rx {
        let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx).map(|delta| {
            Ok::<_, std::convert::Infallible>(Event::default().data(chunk_json(&delta)))
        });
        return Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response());
    }
    let out = done_rx
        .await
        .map_err(|_| err(StatusCode::INTERNAL_SERVER_ERROR, "engine stopped"))?
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e.as_str()))?;
    let stop_hit = out.ids.len() < max_tokens;
    Ok(Json(json!({
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": out.text},
            "finish_reason": if stop_hit { "stop" } else { "length" }
        }],
        "usage": {
            "prompt_tokens": out.prompt_len,
            "completion_tokens": out.ids.len(),
            "completion_seconds": out.secs,
            "tok_s": out.tok_s
        },
        "model": "tick-engine-bitnet-2b4t"
    }))
    .into_response())
}

/// Instant snapshot: busy/inflight/tick stats — never blocks on a
/// generation (reads what the inference thread publishes).
async fn telemetry_snapshot(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    if !authed(&headers, &app.key) {
        return Err(err(StatusCode::UNAUTHORIZED, "unauthorized"));
    }
    let s = app
        .shared
        .read()
        .map_err(|_| err(StatusCode::INTERNAL_SERVER_ERROR, "state lock poisoned"))?;
    Ok(Json(json!({
        "loaded": s.loaded,
        "busy": s.busy,
        "inflight": s.inflight,
        "last_gen": s.last_gen.clone(),
        "engine": {
            "model": app.model_path.display().to_string(),
            "kv_cap": app.kv_cap,
            "pool_threads": tick_kernels::par::threads(),
        },
    })))
}

fn chunk_json(delta: &str) -> String {
    let mut out = String::with_capacity(delta.len() + 40);
    out.push_str("{\"delta\": \"");
    for c in delta.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out.push_str("\"}");
    out
}
