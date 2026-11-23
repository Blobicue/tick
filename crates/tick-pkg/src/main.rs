//! tick-pkg — the client for the tickfile package family: weights
//! ship like code, with the same versioning, sha256 verification, and
//! yank semantics the cargo family uses.

use std::path::PathBuf;

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1);
}

fn usage() -> ! {
    eprintln!(
        "usage:
  tick-pkg publish <file> --name <name> --vers <v> [--family tickfile|adapter]
                     [--registry <base>] [--token <tok>] [--meta k=v ...]
  tick-pkg add <name>@<vers> [--registry <base>] --out <dir> [--token <tok>]
  tick-pkg list <name> [--registry <base>]
  tick-pkg info <name>@<vers> [--registry <base>]"
    );
    std::process::exit(2);
}

async fn fetch_index(base: &str, name: &str) -> Vec<serde_json::Value> {
    let lower = name.to_lowercase();
    let chars: Vec<char> = lower.chars().collect();
    let layout = match chars.len() {
        1 => format!("1/{lower}"),
        2 => format!("2/{lower}"),
        3 => format!("3/{}/{}", chars[0], lower),
        _ => {
            let a: String = chars[..2].iter().collect();
            let b: String = chars[2..4].iter().collect();
            format!("{a}/{b}/{lower}")
        }
    };
    let url = format!("{base}/pkg-index/{layout}");
    let r = reqwest::get(&url)
        .await
        .unwrap_or_else(|e| die(&format!("{url}: {e}")));
    if r.status() == reqwest::StatusCode::NOT_FOUND {
        die(&format!("no package named {name} at {base}"));
    }
    let text = r
        .text()
        .await
        .unwrap_or_else(|e| die(&format!("index read: {e}")));
    text.lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }
    let mut registry = std::env::var("TICK_PKG_REGISTRY").unwrap_or_else(|_| {
        std::env::var("REGISTRY_PUBLIC_URL")
            .unwrap_or_else(|_| "https://crates.gameladder.io".into())
    });
    let mut token = std::env::var("TICK_PKG_TOKEN").unwrap_or_default();
    let mut named: std::collections::BTreeMap<String, String> = Default::default();
    let mut positional: Vec<String> = Vec::new();
    let mut it = args.into_iter();
    let cmd = it.next().unwrap();
    let mut expect_value = false;
    let mut last_flag = String::new();
    for a in it {
        if expect_value {
            match last_flag.as_str() {
                "--registry" => registry = a,
                "--token" => token = a,
                _ => {
                    named.insert(last_flag.trim_start_matches("--").to_string(), a);
                }
            }
            expect_value = false;
            continue;
        }
        if let Some(f) = a.strip_prefix("--") {
            if f == "registry" || f == "token" {
                last_flag = a.clone();
                expect_value = true;
            } else {
                named.insert(f.to_string(), String::new());
                last_flag = a.clone();
                expect_value = true; // all --flags take values (or mark presence)
            }
        } else {
            positional.push(a);
        }
    }
    let get = |k: &str| named.get(k).cloned().filter(|v| !v.is_empty());

    match cmd.as_str() {
        "publish" => {
            let file = positional.first().unwrap_or_else(|| die("file required"));
            let name = get("name").unwrap_or_else(|| die("--name required"));
            let vers = get("vers").unwrap_or_else(|| die("--vers required"));
            let family = get("family").unwrap_or_else(|| "tickfile".into());
            if family != "tickfile" && family != "adapter" {
                die("family must be tickfile or adapter");
            }
            let bytes = std::fs::read(file).unwrap_or_else(|e| die(&format!("{file}: {e}")));
            let mut meta_obj = serde_json::json!({
                "name": name, "vers": vers, "family": family, "meta": {},
            });
            for kv in get("meta").unwrap_or_default().split(',') {
                if let Some((k, v)) = kv.split_once('=') {
                    meta_obj["meta"][k.trim()] = serde_json::json!(v.trim());
                }
            }
            let body =
                tick_registry::publish_stream::frame_publish_body(&meta_obj.to_string(), &bytes);
            let client = reqwest::Client::new();
            let r = client
                .put(format!("{registry}/pkg/v1/publish"))
                .header("authorization", &token)
                .body(body)
                .send()
                .await
                .unwrap_or_else(|e| die(&format!("upload: {e}")));
            if !r.status().is_success() {
                let detail = r.text().await.unwrap_or_default();
                die(&format!("publish failed: {}", detail.trim()));
            }
            let out: serde_json::Value = r.json().await.unwrap();
            println!("published {family}/{name}-{vers} cksum {}", out["cksum"]);
        }
        "add" => {
            let spec = positional
                .first()
                .unwrap_or_else(|| die("<name>@<vers> required"));
            let (name, vers) = spec
                .split_once('@')
                .unwrap_or_else(|| die("use <name>@<vers>"));
            let out_dir = get("out").unwrap_or_else(|| die("--out required"));
            let versions = fetch_index(&registry, name).await;
            let line = versions
                .iter()
                .find(|v| v["vers"] == vers)
                .unwrap_or_else(|| die(&format!("no version {vers} of {name}")));
            if line["yanked"].as_bool() == Some(true) {
                die(&format!("{name}@{vers} is yanked"));
            }
            let url = format!("{registry}/pkg/{name}/{vers}/download");
            let client = reqwest::Client::new();
            let r = client
                .get(&url)
                .send()
                .await
                .unwrap_or_else(|e| die(&format!("{url}: {e}")));
            if !r.status().is_success() {
                die(&format!("download failed: {}", r.status()));
            }
            let bytes = r
                .bytes()
                .await
                .unwrap_or_else(|e| die(&format!("read: {e}")));
            let got = tick_registry::publish_stream::sha256_hex(&bytes);
            if got != line["cksum"].as_str().unwrap_or_default() {
                die(&format!(
                    "sha256 mismatch: got {got}, index says {}",
                    line["cksum"].as_str().unwrap_or_default()
                ));
            }
            let dest: PathBuf = PathBuf::from(&out_dir).join(format!("{name}-{vers}.tickfile"));
            std::fs::write(&dest, &bytes).unwrap_or_else(|e| die(&format!("write: {e}")));
            println!(
                "{} ({}, {} bytes, sha256 verified)",
                dest.display(),
                line["family"].as_str().unwrap_or("tickfile"),
                bytes.len()
            );
            if !line["meta"].is_null() {
                println!("meta: {}", line["meta"]);
            }
        }
        "list" => {
            let name = positional.first().unwrap_or_else(|| die("name required"));
            for v in fetch_index(&registry, name).await {
                println!(
                    "{}  {}  cksum:{}{}",
                    v["vers"],
                    v["pubtime"],
                    v["cksum"],
                    if v["yanked"].as_bool() == Some(true) {
                        "  [yanked]"
                    } else {
                        ""
                    }
                );
            }
        }
        "info" => {
            let spec = positional
                .first()
                .unwrap_or_else(|| die("<name>@<vers> required"));
            let (name, vers) = spec
                .split_once('@')
                .unwrap_or_else(|| die("use <name>@<vers>"));
            let line = fetch_index(&registry, name)
                .await
                .into_iter()
                .find(|v| v["vers"] == vers)
                .unwrap_or_else(|| die(&format!("no version {vers}")));
            println!("{}", serde_json::to_string_pretty(&line).unwrap());
        }
        _ => usage(),
    }
}
