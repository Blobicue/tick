//! The cargo sparse-index protocol pieces: the index path layout for
//! crate names, and the index-record mapping from publish metadata.

use serde_json::{json, Value};

/// Where a crate's index lines live under the index root, per the
/// sparse protocol: 1- and 2-char names under `1/`, `2/`; 3-char under
/// `3/{first}/`; 4+ under `{first-two}/{second-two}/`. Lowercased.
pub fn index_rel_path(name: &str) -> String {
    let lower = name.to_lowercase();
    let chars: Vec<char> = lower.chars().collect();
    match chars.len() {
        1 => format!("1/{}", lower),
        2 => format!("2/{}", lower),
        3 => format!("3/{}/{}", chars[0], lower),
        _ => {
            let a: String = chars[..2].iter().collect();
            let b: String = chars[2..4].iter().collect();
            format!("{a}/{b}/{lower}")
        }
    }
}

/// One index line for a freshly published version, mapped from the
/// publish-metadata JSON per the registry reference:
/// - dep `name` in the index is the Cargo.toml dependency key (the
///   `explicit_name_in_toml`), with `package` carrying the real crate
///   name only when renamed
/// - `cksum` is the sha256 of the `.crate` tarball
/// - `yanked` starts false; it is the only field the protocol lets a
///   registry mutate after insertion
pub fn index_line(meta: &Value, cksum: &str, pubtime: &str) -> String {
    let deps: Vec<Value> = meta["deps"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .map(|d| {
                    let real_name = d["name"].as_str().unwrap_or_default();
                    let rename = d["explicit_name_in_toml"].as_str();
                    let mut dep = json!({
                        "name": rename.unwrap_or(real_name),
                        "req": d["version_req"].clone(),
                        "features": d["features"].clone(),
                        "optional": d["optional"].as_bool().unwrap_or(false),
                        "default_features": d["default_features"].as_bool().unwrap_or(true),
                        "target": d["target"].clone(),
                        "kind": d["kind"].clone(),
                        "registry": d["registry"].clone(),
                    });
                    if let Some(r) = rename {
                        dep["package"] = json!(real_name);
                        dep["name"] = json!(r);
                    }
                    dep
                })
                .collect()
        })
        .unwrap_or_default();
    let mut line = json!({
        "name": meta["name"].clone(),
        "vers": meta["vers"].clone(),
        "deps": deps,
        "cksum": cksum,
        "features": meta["features"].clone(),
        "yanked": false,
        "links": meta["links"].clone(),
        "v": 1,
        "pubtime": pubtime,
    });
    if let Some(rv) = meta["rust_version"].as_str() {
        line["rust_version"] = json!(rv);
    }
    line.to_string()
}

/// Validate a crate name per the protocol: ASCII alphanumeric, `-`, `_`.
pub fn valid_crate_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_follows_the_sparse_spec() {
        assert_eq!(index_rel_path("a"), "1/a");
        assert_eq!(index_rel_path("ab"), "2/ab");
        assert_eq!(index_rel_path("abc"), "3/a/abc");
        assert_eq!(index_rel_path("tick-registry"), "ti/ck/tick-registry");
        assert_eq!(index_rel_path("Tick-Server"), "ti/ck/tick-server");
    }

    #[test]
    fn dep_rename_maps_to_package_field() {
        let meta = serde_json::json!({
            "name": "fixture", "vers": "0.1.0",
            "deps": [
                {"name": "serde_core", "version_req": "^1",
                 "explicit_name_in_toml": "serde",
                 "features": [], "optional": false,
                 "default_features": true, "target": null,
                 "kind": "normal", "registry": null},
            ],
            "features": {},
        });
        let line: Value =
            serde_json::from_str(&index_line(&meta, "cafebabe", "2026-11-23T00:00:00Z")).unwrap();
        assert_eq!(line["deps"][0]["name"], "serde");
        assert_eq!(line["deps"][0]["package"], "serde_core");
        assert_eq!(line["cksum"], "cafebabe");
        assert_eq!(line["yanked"], false);
    }
}
