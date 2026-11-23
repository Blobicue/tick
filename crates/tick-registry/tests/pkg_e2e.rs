//! The tickfile-family gate: publish through the same framing, list,
//! sha256-verified download, corruption rejection, yank.

#[tokio::test(flavor = "multi_thread")]
async fn pkg_family_publish_list_add_yank_roundtrip() {
    let root = tempfile::tempdir().unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let base = format!("http://127.0.0.1:{port}");
    let reg = tick_registry::Reg::new(root.path(), &base).unwrap();
    let router = tick_registry::build_router(std::sync::Arc::clone(&reg));
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    let token = tick_registry::mint_token(&reg, "pkg-owner").await.unwrap();

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap();
    let auth = format!("Bearer {token}");

    // a real tickfile artifact: reuse the tiny model the wedge test builds
    let artifact_bytes = make_tiny_tickfile();

    // 1) publish via the framed body
    let meta = serde_json::json!({
        "name": "lighthouse-tiny", "vers": "0.2.0", "family": "tickfile",
        "meta": {"arch": "3x256", "layers": 3, "provenance": "tick-train init seed 42"},
    });
    let body =
        tick_registry::publish_stream::frame_publish_body(&meta.to_string(), &artifact_bytes);
    let r = client
        .put(format!("{base}/pkg/v1/publish"))
        .header("authorization", &auth)
        .body(body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "publish: {}", r.text().await.unwrap());
    let reply: serde_json::Value = r.json().await.unwrap();
    let cksum = reply["cksum"].as_str().unwrap().to_string();

    // duplicate rejected
    let r = client
        .put(format!("{base}/pkg/v1/publish"))
        .header("authorization", &auth)
        .body(tick_registry::publish_stream::frame_publish_body(
            &meta.to_string(),
            &artifact_bytes,
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403, "duplicate must be rejected");

    // 2) index line visible at the sparse layout
    let idx = client
        .get(format!("{base}/pkg-index/li/gh/lighthouse-tiny"))
        .send()
        .await
        .unwrap();
    assert_eq!(idx.status(), 200);
    let line: serde_json::Value =
        serde_json::from_str(idx.text().await.unwrap().lines().next().unwrap()).unwrap();
    assert_eq!(line["vers"], "0.2.0");
    assert_eq!(line["family"], "tickfile");
    assert_eq!(line["cksum"], cksum.as_str());
    assert_eq!(line["meta"]["arch"], "3x256");

    // 3) download verifies byte-for-byte against the cksum
    let dl = client
        .get(format!("{base}/pkg/lighthouse-tiny/0.2.0/download"))
        .send()
        .await
        .unwrap();
    assert_eq!(dl.status(), 200);
    let got = dl.bytes().await.unwrap();
    assert_eq!(tick_registry::publish_stream::sha256_hex(&got), cksum);

    // 4) yank hides the artifact from downloads
    let r = client
        .delete(format!("{base}/pkg/v1/lighthouse-tiny/0.2.0/yank"))
        .header("authorization", &auth)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let dl = client
        .get(format!("{base}/pkg/lighthouse-tiny/0.2.0/download"))
        .send()
        .await
        .unwrap();
    assert_eq!(dl.status(), 404, "yanked versions must 404");
}

fn make_tiny_tickfile() -> Vec<u8> {
    use tick_train::train::*;
    let cfg = TinyConfig {
        n_layer: 3,
        n_head: 4,
        n_head_kv: 1,
        head_dim: 64,
        hidden: 256,
        ffn: 512,
        vocab: 256,
        rms_eps: 1e-5,
        rope_base: 10000.0,
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.tick");
    let (embd, norms, mats) = init_tiny(&cfg, 42);
    write_tiny_tickfile(&cfg, &embd, &norms, &mats, &path).unwrap();
    std::fs::read(&path).unwrap()
}
