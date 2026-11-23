//! The wedge-free proof with a plugin mounted: while a generation
//! runs, `/health` AND a plugin route must answer fast — a parked
//! handler would block for the whole generation. Runs everywhere
//! (builds a tiny random tickfile via tick-train; no model files).

// `!(x < t)` NaN-safe gates are deliberate (law: NaN comparisons are false).
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use std::sync::Arc;
use std::time::{Duration, Instant};
use tick_server::server::{build_router, App};

fn tiny_tickfile(path: &str) -> Result<(), String> {
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
    let (embd, norms, mats) = init_tiny(&cfg, 42);
    write_tiny_tickfile(&cfg, &embd, &norms, &mats, std::path::Path::new(path))
}

#[tokio::test]
async fn plugin_routes_never_park_behind_generation() {
    let path = "/tmp/tick_plugin_wedge_test.tick";
    tiny_tickfile(path).expect("tiny tickfile");
    let app = App::open(path.into(), 8192, 4096).expect("app open");

    let key = app.key.clone();
    let router = build_router(
        Arc::clone(&app),
        vec![Box::new(tick_example_plugin::StatusBoard)],
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    let client = reqwest_for_test();
    let auth = format!("Bearer {key}");

    // fire a long generation through the plugin's own ask route.
    // tiny-model decode is overhead-dominated (~140ms/token), so a
    // 4000-token budget keeps the engine busy for minutes — far past
    // the poll window. The task is ABANDONED (never awaited): the
    // test cares only that generation was RUNNING while we polled.
    let ask_task = {
        let client = client.clone();
        let auth = auth.clone();
        let url = format!("http://{addr}/plugin/ask");
        tokio::spawn(async move {
            let ask = client
                .post(url)
                .header("authorization", &auth)
                .body("4000")
                .send()
                .await
                .expect("ask sent");
            assert_eq!(ask.status(), 200);
        })
    };
    // let the request reach the inference thread before first poll
    tokio::time::sleep(Duration::from_millis(150)).await;

    // poll the instant routes WHILE that generation must be running
    let mut worst_health = Duration::ZERO;
    let mut worst_plugin = Duration::ZERO;
    let mut saw_busy = false;
    let poll_start = Instant::now();
    while poll_start.elapsed() < Duration::from_millis(500) {
        let t = Instant::now();
        let h = client
            .get(format!("http://{addr}/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(h.status(), 200);
        worst_health = worst_health.max(t.elapsed());

        let t = Instant::now();
        let p = client
            .get(format!("http://{addr}/plugin/status"))
            .send()
            .await
            .unwrap();
        assert_eq!(p.status(), 200);
        let body: serde_json::Value = p.json().await.unwrap();
        if body["engine"]["busy"].as_bool() == Some(true) {
            saw_busy = true;
        }
        worst_plugin = worst_plugin.max(t.elapsed());
    }

    assert!(
        saw_busy,
        "never observed busy=true — generation not overlapping"
    );
    assert!(
        !(worst_health >= Duration::from_millis(100)),
        "health parked: worst {worst_health:?}"
    );
    assert!(
        !(worst_plugin >= Duration::from_millis(100)),
        "plugin route parked: worst {worst_plugin:?}"
    );
    // and the wedge-free property, precisely: both routes answered
    // hundreds of times faster than the multi-second generation
    println!(
        "wedge-free: worst health {worst_health:?}, worst plugin {worst_plugin:?} during an abandoned long generation"
    );
    ask_task.abort(); // the reply is irrelevant; keep the test <1s
}

fn reqwest_for_test() -> reqwest::Client {
    reqwest::Client::new()
}
