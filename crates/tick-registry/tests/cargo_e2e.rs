//! The conformance gate: a STOCK cargo publishes, depends, builds,
//! yanks, and 304s against this registry over plain loopback HTTP
//! (permitted for localhost per the sparse-registry RFC).

use std::path::{Path, PathBuf};
use std::process::Command;

struct Server {
    #[allow(dead_code)]
    reg: std::sync::Arc<tick_registry::Reg>,
    port: u16,
    _root: tempfile::TempDir,
}

async fn start_server() -> Server {
    let root = tempfile::tempdir().unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let reg = tick_registry::Reg::new(root.path(), &format!("http://127.0.0.1:{port}")).unwrap();
    let router = tick_registry::build_router(std::sync::Arc::clone(&reg));
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    Server {
        reg,
        port,
        _root: root,
    }
}

fn cargo(home: &Path, cwd: &Path, args: &[&str], token: &str, port: u16) -> Command {
    let mut c = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()));
    c.args(args)
        .current_dir(cwd)
        .env("CARGO_HOME", home)
        .env("CARGO_NET_OFFLINE", "false")
        .env("CARGO_REGISTRIES_LOCALTICK_TOKEN", token)
        .env("CARGO_TARGET_DIR", home.join("target"));
    c.env_remove("CARGO_UNSTABLE_SPARSE_REGISTRY");
    let _ = port;
    c
}

fn write_client_config(home: &Path, port: u16) {
    std::fs::write(
        home.join("config.toml"),
        format!("[registries.localtick]\nindex = \"sparse+http://127.0.0.1:{port}/idx/\"\n"),
    )
    .unwrap();
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[tokio::test(flavor = "multi_thread")]
async fn stock_cargo_publishes_depends_builds_and_yanks() {
    let srv = start_server().await;
    let token = tick_registry::mint_token(&srv.reg, "test-owner")
        .await
        .unwrap();

    let home = tempfile::tempdir().unwrap();
    write_client_config(home.path(), srv.port);

    // 1) publish the fixture crate with a stock cargo
    let out = cargo(
        home.path(),
        &fixture("publish-pkg"),
        &[
            "publish",
            "--registry",
            "localtick",
            "--allow-dirty",
            "--no-verify",
        ],
        &token,
        srv.port,
    )
    .output()
    .expect("cargo publish");
    assert!(
        out.status.success(),
        "publish failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    // index file exists at the sparse layout with the right fields
    let idx = srv.reg.root.join("index").join("lo/ca/localtick-fixture");
    let content = std::fs::read_to_string(idx).unwrap();
    let line: serde_json::Value = serde_json::from_str(content.lines().next().unwrap()).unwrap();
    assert_eq!(line["name"], "localtick-fixture");
    assert_eq!(line["vers"], "0.1.0");
    assert!(line["cksum"].as_str().unwrap().len() == 64);
    assert_eq!(line["yanked"], false);

    // .crate stored where the dl route serves it
    let crate_file = srv
        .reg
        .root
        .join("crates/localtick-fixture/0.1.0/localtick-fixture-0.1.0.crate");
    assert!(crate_file.exists());

    // 2) a consumer project depends on it through the registry
    let consumer = tempfile::tempdir().unwrap();
    std::fs::write(
        consumer.path().join("Cargo.toml"),
        r#"
[package]
name = "consumer"
version = "0.1.0"
edition = "2021"

[workspace]

[dependencies]
localtick-fixture = { version = "0.1.0", registry = "localtick" }
"#,
    )
    .unwrap();
    std::fs::create_dir_all(consumer.path().join("src")).unwrap();
    std::fs::write(
        consumer.path().join("src/main.rs"),
        "fn main() { println!(\"{}\", localtick_fixture::answer()); }\n",
    )
    .unwrap();
    let out = cargo(
        home.path(),
        consumer.path(),
        &["run", "--quiet"],
        &token,
        srv.port,
    )
    .output()
    .expect("cargo run");
    assert!(
        out.status.success(),
        "consumer build failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "42");

    // 3) duplicate publish is rejected with the spec error shape
    let out = cargo(
        home.path(),
        &fixture("publish-pkg"),
        &[
            "publish",
            "--registry",
            "localtick",
            "--allow-dirty",
            "--no-verify",
        ],
        &token,
        srv.port,
    )
    .output()
    .unwrap();
    assert!(!out.status.success(), "duplicate publish must fail");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("already exists"),
        "stderr should carry the detail: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // 4) yank: a fresh resolution must refuse the yanked version
    let client = reqwest::Client::new();
    let r = client
        .delete(format!(
            "http://127.0.0.1:{}/api/v1/crates/localtick-fixture/0.1.0/yank",
            srv.port
        ))
        .header("authorization", format!("Bearer {token}"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(r.json::<serde_json::Value>().await.unwrap()["ok"], true);

    let fresh = tempfile::tempdir().unwrap();
    std::fs::write(
        fresh.path().join("Cargo.toml"),
        r#"
[package]
name = "fresh"
version = "0.1.0"
edition = "2021"

[workspace]

[dependencies]
localtick-fixture = { version = "0.1.0", registry = "localtick" }
"#,
    )
    .unwrap();
    std::fs::create_dir_all(fresh.path().join("src")).unwrap();
    std::fs::write(fresh.path().join("src/main.rs"), "fn main() {}\n").unwrap();
    let out = cargo(
        home.path(),
        fresh.path(),
        &["generate-lockfile"],
        &token,
        srv.port,
    )
    .output()
    .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("yanked"),
        "fresh resolution should refuse the yanked version: {stderr}"
    );

    // 5) ETag conditional GET on the index
    let idx_url = format!("http://127.0.0.1:{}/idx/lo/ca/localtick-fixture", srv.port);
    let r1 = client.get(&idx_url).send().await.unwrap();
    let etag = r1
        .headers()
        .get("etag")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    let r2 = client
        .get(&idx_url)
        .header("if-none-match", &etag)
        .send()
        .await
        .unwrap();
    assert_eq!(r2.status(), 304, "conditional GET must 304");
}
