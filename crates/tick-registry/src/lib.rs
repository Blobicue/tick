//! tick-registry — our own distribution backend: the cargo sparse
//! registry protocol (publish / yank / owners / search / downloads)
//! plus the tickfile package family, on plain filesystem storage.
//!
//! Wire compatibility is the contract: a stock `cargo` speaking
//! `sparse+https://<host>/` can publish, depend, and build against
//! this server with no client-side tooling. The conformance proof is
//! `tests/cargo_e2e.rs`, which drives real cargo end-to-end.
//!
//! The client's registry index URL is `<base>/idx/` — the index
//! family lives under its own prefix exactly like crates.io splits
//! index.crates.io from static.crates.io, which keeps the root free
//! for the download route and the API.
//!
//! Layout under the storage root:
//! `index/` sparse index · `crates/` .crate tarballs ·
//! `pkg-index/` + `artifacts/` tickfile family · `state.json`.
#![allow(unknown_lints)]
#![allow(clippy::needless_range_loop)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{Path as AxPath, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use tokio::sync::RwLock;

pub mod publish_stream;
pub mod sparse;

use publish_stream::{parse_publish_body, sha256_hex, validate_tarball};
use sparse::{index_line, index_rel_path, valid_crate_name};

/// Persisted server state: tokens (sha256-hex), crate owners, and the
/// search summary. One writer (the state lock), atomic file replace.
#[derive(Default, serde::Deserialize)]
#[serde(default)]
pub struct Persisted {
    pub tokens: BTreeMap<String, String>,
    pub owners: BTreeMap<String, Vec<String>>,
    pub summaries: BTreeMap<String, Value>,
    pub audit: Vec<String>,
}

pub struct Reg {
    pub root: PathBuf,
    /// public base URL, e.g. `https://crates.gameladder.io` (no slash)
    pub public_url: String,
    pub state: RwLock<Persisted>,
}

pub type RegArc = Arc<Reg>;

impl Reg {
    pub fn new(root: &Path, public_url: &str) -> std::io::Result<RegArc> {
        for d in ["index", "crates", "pkg-index", "artifacts"] {
            std::fs::create_dir_all(root.join(d))?;
        }
        let mut p = Persisted::default();
        let sp = root.join("state.json");
        if sp.exists() {
            p = serde_json::from_str(&std::fs::read_to_string(sp)?)
                .map_err(std::io::Error::other)?;
        }
        Ok(Arc::new(Reg {
            root: root.to_path_buf(),
            public_url: public_url.trim_end_matches('/').to_string(),
            state: RwLock::new(p),
        }))
    }

    pub async fn save_state(&self) -> Result<(), String> {
        let s = self.state.read().await;
        let tmp = self.root.join("state.json.tmp");
        let json = serde_json::to_string_pretty(&json!({
            "tokens": s.tokens, "owners": s.owners,
            "summaries": s.summaries, "audit": s.audit,
        }))
        .map_err(|e| e.to_string())?;
        std::fs::write(&tmp, json).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, self.root.join("state.json")).map_err(|e| e.to_string())
    }

    pub fn index_path(&self, name: &str) -> PathBuf {
        self.root.join("index").join(index_rel_path(name))
    }

    pub fn crate_file(&self, name: &str, vers: &str) -> PathBuf {
        self.root
            .join("crates")
            .join(name)
            .join(vers)
            .join(format!("{name}-{vers}.crate"))
    }

    fn authed(&self, headers: &HeaderMap, st: &Persisted) -> Option<String> {
        let v = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
        // cargo's built-in token provider sends the token VERBATIM in
        // the Authorization header (no Bearer scheme); accept both
        // spellings so curl-style Bearer clients work too
        let tok = v.strip_prefix("Bearer ").unwrap_or(v).trim();
        let h = sha256_hex(tok.as_bytes());
        st.tokens
            .iter()
            .find(|(_, hash)| **hash == h)
            .map(|(l, _)| l.clone())
    }
}

fn err_json(status: StatusCode, detail: &str) -> Response {
    (status, Json(json!({"errors": [{"detail": detail}]}))).into_response()
}

fn ok_json(v: Value) -> Response {
    Json(v).into_response()
}

pub fn build_router(reg: RegArc) -> axum::Router {
    use axum::routing::{delete, get, put};
    axum::Router::new()
        .route("/idx/config.json", get(config_json))
        .route("/idx/{*path}", get(index_file))
        .route("/api/v1/crates/new", put(publish))
        .route("/api/v1/crates", get(search))
        .route(
            "/api/v1/crates/{name}/owners",
            get(list_owners).put(add_owners).delete(remove_owners),
        )
        .route(
            "/api/v1/crates/{name}/{version}/yank",
            delete(yank).put(unyank),
        )
        .route("/api/v1/crates/{name}/{version}/unyank", put(unyank))
        .route("/{crate}/{version}/download", get(download))
        .with_state(reg)
        .layer(axum::extract::DefaultBodyLimit::max(64 * 1024 * 1024))
}

async fn config_json(State(reg): State<RegArc>) -> Response {
    ok_json(json!({
        "dl": format!("{}/{{crate}}/{{version}}/download", reg.public_url),
        "api": reg.public_url.clone(),
        "auth-required": false,
    }))
}

async fn index_file(
    State(reg): State<RegArc>,
    AxPath(path): AxPath<String>,
    headers: HeaderMap,
) -> Response {
    // path traversal guard: reject .. and absolute segments
    if path
        .split('/')
        .any(|seg| seg == ".." || seg.is_empty() || seg.starts_with('.'))
    {
        return err_json(StatusCode::NOT_FOUND, "no such index entry");
    }
    let full = reg
        .root
        .join("index")
        .join(path.strip_prefix("idx/").unwrap_or(&path));
    match tokio::fs::read(&full).await {
        Err(_) => err_json(StatusCode::NOT_FOUND, "no such index entry"),
        Ok(bytes) => {
            let etag = format!("\"{}\"", sha256_hex(&bytes));
            if let Some(inm) = headers
                .get(header::IF_NONE_MATCH)
                .and_then(|v| v.to_str().ok())
            {
                if inm == etag || inm == "W/".to_string() + &etag {
                    return StatusCode::NOT_MODIFIED.into_response();
                }
            }
            Response::builder()
                .header(header::ETAG, etag)
                .header(header::CONTENT_TYPE, "text/plain")
                .body(axum::body::Body::from(bytes))
                .unwrap()
        }
    }
}

async fn download(
    State(reg): State<RegArc>,
    AxPath((crate_name, version)): AxPath<(String, String)>,
) -> Response {
    let file = reg.crate_file(&crate_name, &version);
    match tokio::fs::read(&file).await {
        Err(_) => err_json(StatusCode::NOT_FOUND, "no such crate version"),
        Ok(bytes) => Response::builder()
            .header(header::CONTENT_TYPE, "application/octet-stream")
            .body(axum::body::Body::from(bytes))
            .unwrap(),
    }
}

async fn publish(State(reg): State<RegArc>, headers: HeaderMap, body: Bytes) -> Response {
    let login = {
        let st = reg.state.read().await;
        match reg.authed(&headers, &st) {
            Some(l) => l,
            None => return err_json(StatusCode::FORBIDDEN, "invalid registry token"),
        }
    };
    let parsed = match parse_publish_body(&body) {
        Ok(p) => p,
        Err(e) => return err_json(StatusCode::BAD_REQUEST, &e),
    };
    let name = match parsed.meta["name"].as_str() {
        Some(n) if valid_crate_name(n) => n.to_string(),
        _ => return err_json(StatusCode::BAD_REQUEST, "invalid crate name"),
    };
    let vers = match parsed.meta["vers"].as_str() {
        Some(v) => v.to_string(),
        None => return err_json(StatusCode::BAD_REQUEST, "missing version"),
    };

    // duplicate-version check + ownership in one critical section
    {
        let st = reg.state.write().await;
        if let Some(owners) = st.owners.get(&name) {
            if !owners.contains(&login) {
                return err_json(
                    StatusCode::FORBIDDEN,
                    &format!("token owner {login} does not own crate {name}"),
                );
            }
        }
        drop(st);
    }
    let idx_path = reg.index_path(&name);
    if let Ok(existing) = std::fs::read_to_string(&idx_path) {
        for line in existing.lines() {
            if let Ok(v) = serde_json::from_str::<Value>(line) {
                if v["vers"] == vers.as_str() {
                    return err_json(
                        StatusCode::FORBIDDEN,
                        &format!("crate version {vers} is already uploaded"),
                    );
                }
            }
        }
    }
    if let Err(e) = validate_tarball(&parsed.crate_bytes, &name, &vers) {
        return err_json(StatusCode::BAD_REQUEST, &e);
    }
    let cksum = sha256_hex(&parsed.crate_bytes);
    let file = reg.crate_file(&name, &vers);
    if let Some(dir) = file.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Err(e) = std::fs::write(&file, &parsed.crate_bytes) {
        return err_json(StatusCode::INTERNAL_SERVER_ERROR, &format!("store: {e}"));
    }
    let now = chrono_like_now();
    let line = index_line(&parsed.meta, &cksum, &now);
    if let Some(dir) = idx_path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    use std::io::Write;
    let mut f = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&idx_path)
    {
        Ok(f) => f,
        Err(e) => return err_json(StatusCode::INTERNAL_SERVER_ERROR, &format!("index: {e}")),
    };
    if let Err(e) = writeln!(f, "{line}") {
        return err_json(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("index append: {e}"),
        );
    }

    {
        let mut st = reg.state.write().await;
        st.owners
            .entry(name.clone())
            .or_default()
            .push(login.clone());
        st.owners.get_mut(&name).unwrap().dedup();
        st.summaries.insert(
            name.clone(),
            json!({
                "name": name, "max_version": vers,
                "description": parsed.meta["description"].clone(),
                "updated_at": now,
            }),
        );
        st.audit
            .push(format!("{now} {login} published {name}-{vers}"));
    }
    if let Err(e) = reg.save_state().await {
        return err_json(StatusCode::INTERNAL_SERVER_ERROR, &format!("state: {e}"));
    }
    ok_json(json!({
        "warnings": {"invalid_categories": [], "invalid_badges": [], "other": []}
    }))
}

async fn yank_flip(
    reg: &RegArc,
    headers: &HeaderMap,
    name: &str,
    vers: &str,
    to: bool,
) -> Response {
    let login = {
        let st = reg.state.read().await;
        match reg.authed(headers, &st) {
            Some(l) => l,
            None => return err_json(StatusCode::FORBIDDEN, "invalid registry token"),
        }
    };
    {
        let st = reg.state.read().await;
        match st.owners.get(name) {
            Some(o) if o.contains(&login) => {}
            _ => {
                return err_json(
                    StatusCode::FORBIDDEN,
                    &format!("token owner {login} does not own crate {name}"),
                )
            }
        }
    }
    let idx_path = reg.index_path(name);
    let content = match std::fs::read_to_string(&idx_path) {
        Ok(c) => c,
        Err(_) => return err_json(StatusCode::NOT_FOUND, "no such crate"),
    };
    let mut out = String::new();
    let mut flipped = false;
    for line in content.lines() {
        if let Ok(mut v) = serde_json::from_str::<Value>(line) {
            if v["vers"] == vers {
                v["yanked"] = json!(to);
                flipped = true;
                out.push_str(&v.to_string());
                out.push('\n');
                continue;
            }
        }
        out.push_str(line);
        out.push('\n');
    }
    if !flipped {
        return err_json(StatusCode::NOT_FOUND, "no such crate version");
    }
    let tmp = idx_path.with_extension("tmp");
    if std::fs::write(&tmp, out)
        .and_then(|_| std::fs::rename(&tmp, &idx_path))
        .is_err()
    {
        return err_json(StatusCode::INTERNAL_SERVER_ERROR, "index rewrite failed");
    }
    {
        let mut st = reg.state.write().await;
        st.audit.push(format!(
            "{} {login} {} {name}-{vers}",
            chrono_like_now(),
            if to { "yanked" } else { "unyanked" }
        ));
    }
    let _ = reg.save_state().await;
    ok_json(json!({"ok": true}))
}

async fn yank(
    State(reg): State<RegArc>,
    headers: HeaderMap,
    AxPath((name, version)): AxPath<(String, String)>,
) -> Response {
    yank_flip(&reg, &headers, &name, &version, true).await
}

async fn unyank(
    State(reg): State<RegArc>,
    headers: HeaderMap,
    AxPath((name, version)): AxPath<(String, String)>,
) -> Response {
    yank_flip(&reg, &headers, &name, &version, false).await
}

async fn list_owners(State(reg): State<RegArc>, AxPath(name): AxPath<String>) -> Response {
    let st = reg.state.read().await;
    match st.owners.get(&name) {
        Some(owners) => ok_json(json!({
            "users": owners.iter().map(|l| json!({
                "id": 0, "login": l, "name": l
            })).collect::<Vec<_>>()
        })),
        None => err_json(StatusCode::NOT_FOUND, "no such crate"),
    }
}

async fn add_owners(
    State(reg): State<RegArc>,
    headers: HeaderMap,
    AxPath(name): AxPath<String>,
    Json(body): Json<Value>,
) -> Response {
    let login = {
        let st = reg.state.read().await;
        match reg.authed(&headers, &st) {
            Some(l) => l,
            None => return err_json(StatusCode::FORBIDDEN, "invalid registry token"),
        }
    };
    let new_users: Vec<String> = body["users"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    if new_users.is_empty() {
        return err_json(StatusCode::BAD_REQUEST, "users array required");
    }
    let mut st = reg.state.write().await;
    match st.owners.get_mut(&name) {
        None => return err_json(StatusCode::NOT_FOUND, "no such crate"),
        Some(owners) => {
            if !owners.contains(&login) {
                return err_json(
                    StatusCode::FORBIDDEN,
                    &format!("token owner {login} does not own crate {name}"),
                );
            }
            for u in new_users {
                if !owners.contains(&u) {
                    owners.push(u);
                }
            }
        }
    }
    let _ = reg.save_state().await;
    ok_json(json!({"ok": true, "msg": "owners successfully added"}))
}

async fn remove_owners(
    State(reg): State<RegArc>,
    headers: HeaderMap,
    AxPath(name): AxPath<String>,
    Json(body): Json<Value>,
) -> Response {
    let login = {
        let st = reg.state.read().await;
        match reg.authed(&headers, &st) {
            Some(l) => l,
            None => return err_json(StatusCode::FORBIDDEN, "invalid registry token"),
        }
    };
    let users: Vec<String> = body["users"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let mut st = reg.state.write().await;
    let owners_len = match st.owners.get(&name) {
        None => return err_json(StatusCode::NOT_FOUND, "no such crate"),
        Some(o) => {
            if !o.contains(&login) {
                return err_json(
                    StatusCode::FORBIDDEN,
                    &format!("token owner {login} does not own crate {name}"),
                );
            }
            o.len()
        }
    };
    if users.len() >= owners_len {
        return err_json(StatusCode::BAD_REQUEST, "cannot remove the last owner");
    }
    if let Some(o) = st.owners.get_mut(&name) {
        o.retain(|o| !users.contains(o));
    }
    let _ = reg.save_state().await;
    ok_json(json!({"ok": true, "msg": "owners successfully removed"}))
}

async fn search(
    State(reg): State<RegArc>,
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let query = q.get("q").cloned().unwrap_or_default();
    let st = reg.state.read().await;
    let hits: Vec<Value> = st
        .summaries
        .values()
        .filter(|s| {
            let n = s["name"].as_str().unwrap_or_default();
            n.contains(&query)
        })
        .cloned()
        .collect();
    let total = hits.len();
    ok_json(json!({"crates": hits, "meta": {"total": total}}))
}

/// Minimal UTC timestamp in the index's allowed format. No chrono
/// dependency: days-from-civil inversion over SystemTime.
fn chrono_like_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = secs / 86400;
    let rem = secs % 86400;
    let (y, m, d) = civil_from_days(days as i64);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Mint a token for `login`: prints it once (only time the raw token
/// exists), stores the sha256. Returns the raw token.
pub async fn mint_token(reg: &RegArc, login: &str) -> Result<String, String> {
    use std::io::Read;
    let mut raw = [0u8; 32];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut raw))
        .map_err(|e| e.to_string())?;
    let token = hex::encode(raw);
    {
        let mut st = reg.state.write().await;
        st.tokens
            .insert(login.to_string(), sha256_hex(token.as_bytes()));
        st.audit
            .push(format!("{} token minted for {login}", chrono_like_now()));
    }
    reg.save_state().await?;
    Ok(token)
}
