//! tick-registry server + admin CLI.

use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), String> {
    let root = std::env::var("REGISTRY_ROOT").unwrap_or_else(|_| "/srv/tick-registry".into());
    let public = std::env::var("REGISTRY_PUBLIC_URL")
        .unwrap_or_else(|_| "https://crates.gameladder.io".into());
    let bind = std::env::var("REGISTRY_BIND").unwrap_or_else(|_| "127.0.0.1:8795".into());

    let reg = tick_registry::Reg::new(&PathBuf::from(&root), &public).map_err(|e| e.to_string())?;

    match std::env::args().nth(1).as_deref() {
        None | Some("serve") => {
            let router = tick_registry::build_router(reg);
            let listener = tokio::net::TcpListener::bind(&bind)
                .await
                .map_err(|e| format!("bind {bind}: {e}"))?;
            eprintln!("tick-registry listening on {bind} (root {root}, public {public})");
            axum::serve(listener, router)
                .await
                .map_err(|e| e.to_string())
        }
        Some("token") => {
            // tick-registry token add <login> — prints the token once
            let login = std::env::args().nth(3).ok_or("usage: token add <login>")?;
            if std::env::args().nth(2).as_deref() != Some("add") {
                return Err("usage: token add <login>".into());
            }
            let tok = tick_registry::mint_token(&reg, &login).await?;
            println!("{tok}");
            Ok(())
        }
        Some(other) => Err(format!(
            "unknown command {other:?} (serve | token add <login>)"
        )),
    }
}
