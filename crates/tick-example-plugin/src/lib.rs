//! tick-example-plugin — the reference community plugin: a status
//! board and a minimal ask endpoint over the tick-plugin contract.
//! Copy this crate as a template for your own.

use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Json;
use serde_json::json;
use tick_plugin::{Handle, TickPlugin};

/// The reference plugin: `GET /plugin/status`, `POST /plugin/ask`.
pub struct StatusBoard;

impl TickPlugin for StatusBoard {
    fn name(&self) -> &'static str {
        "status-board"
    }

    fn router(&self, handle: Handle) -> axum::Router {
        let status_handle = handle.clone();
        let ask_handle = handle;
        axum::Router::new()
            .route(
                "/plugin/status",
                get(move || async move {
                    let s = status_handle.status();
                    let last = status_handle.last_generation();
                    Json(json!({
                        "plugin": "status-board",
                        "engine": {
                            "loaded": s.loaded,
                            "busy": s.busy,
                            "inflight": s.inflight,
                        },
                        "last_generation": last,
                    }))
                }),
            )
            .route(
                "/plugin/ask",
                post(move |headers: HeaderMap, body: String| async move {
                    if !ask_handle.authed(&headers) {
                        return (
                            StatusCode::UNAUTHORIZED,
                            Json(json!({"error": {"message": "bad key"}})),
                        )
                            .into_response();
                    }
                    let budget = body.trim().parse::<usize>().unwrap_or(64);
                    match ask_handle.generate_text("Hello", budget).await {
                        Ok(reply) => Json(json!({
                            "text": reply.text,
                            "tok_s": reply.tok_s,
                        }))
                        .into_response(),
                        Err(e) => (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(json!({"error": {"message": e}})),
                        )
                            .into_response(),
                    }
                }),
            )
    }
}
