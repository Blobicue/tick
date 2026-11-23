//! tick-server — the tick engine's inference server on :8096.
//!
//! Usage: tick-server [--port 8096] [--kv-cap 1024] [--max-tokens 256]
//!             [--model /path/to/model.(tick|gguf)]
//! Env: TICK_SERVER_MODEL overrides the default (tickfile when
//! present — everything builds on tickfile).

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str, default: &str| -> String {
        let mut it = args.iter();
        while let Some(a) = it.next() {
            if a == name {
                return it.next().cloned().unwrap_or_default();
            }
        }
        default.to_string()
    };
    let port = flag("--port", "8096");
    let kv_cap: usize = flag("--kv-cap", "1024").parse().unwrap_or(1024);
    let max_tokens: usize = flag("--max-tokens", "256").parse().unwrap_or(256);
    let tick = "/home/juzz/models/bitnet/bitnet-b1.58-2B-4T.tick";
    let gguf = "/home/juzz/models/bitnet/bitnet-b1.58-2B-4T-tq1_0.gguf";
    let default_model = if std::path::Path::new(tick).exists() {
        tick
    } else {
        gguf
    };
    let model =
        std::env::var("TICK_SERVER_MODEL").unwrap_or_else(|_| flag("--model", default_model));
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("tokio runtime");
    rt.block_on(async move {
        let app =
            tick_server::server::App::open(model.into(), kv_cap, max_tokens).expect("app open");
        eprintln!(
            "tick-server listening on 0.0.0.0:{port} (model: {}, weights lazy)",
            app.model_path.display()
        );
        tick_server::server::serve(&format!("0.0.0.0:{port}"), app)
            .await
            .expect("serve");
    });
}
