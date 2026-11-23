//! The publish wire format: cargo PUTs a length-prefixed binary
//! stream (NOT HTTP multipart) — u32-LE metadata length, the metadata
//! JSON, u32-LE tarball length, the `.crate` tarball.

use flate2::read::GzDecoder;
use serde_json::Value;
use sha2::{Digest, Sha256};

pub struct PublishBody {
    pub meta: Value,
    pub crate_bytes: Vec<u8>,
}

/// Parse the framed publish body. Rejects trailing garbage and
/// length fields that overrun the body.
pub fn parse_publish_body(body: &[u8]) -> Result<PublishBody, String> {
    let rd_u32 = |b: &[u8]| -> u32 { u32::from_le_bytes([b[0], b[1], b[2], b[3]]) };
    if body.len() < 8 {
        return Err("body too short for the publish framing".into());
    }
    let meta_len = rd_u32(&body[..4]) as usize;
    if 4 + meta_len + 4 > body.len() {
        return Err("metadata length overruns body".into());
    }
    let meta_bytes = &body[4..4 + meta_len];
    let file_len = rd_u32(&body[4 + meta_len..4 + meta_len + 4]) as usize;
    let file_start = 4 + meta_len + 4;
    if file_start + file_len != body.len() {
        return Err(format!(
            "tarball length {file_len} does not complete the body ({} trailing bytes)",
            body.len() - file_start
        ));
    }
    let meta: Value =
        serde_json::from_slice(meta_bytes).map_err(|e| format!("metadata JSON: {e}"))?;
    let crate_bytes = body[file_start..].to_vec();
    Ok(PublishBody { meta, crate_bytes })
}

/// Build the framed publish body (the inverse of `parse_publish_body`).
pub fn frame_publish_body(meta_json: &str, file: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(8 + meta_json.len() + file.len());
    b.extend_from_slice(&(meta_json.len() as u32).to_le_bytes());
    b.extend_from_slice(meta_json.as_bytes());
    b.extend_from_slice(&(file.len() as u32).to_le_bytes());
    b.extend_from_slice(file);
    b
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}

/// Sanity-check the tarball: it must be a valid gzip'd tar containing
/// `{name}-{vers}/Cargo.toml` whose package name and version match
/// the publish metadata.
pub fn validate_tarball(bytes: &[u8], name: &str, vers: &str) -> Result<(), String> {
    let gz = GzDecoder::new(bytes);
    let mut archive = tar::Archive::new(gz);
    let want_path = format!("{name}-{vers}/Cargo.toml");
    let mut found: Option<String> = None;
    for entry in archive.entries().map_err(|e| format!("tar: {e}"))? {
        let mut entry = entry.map_err(|e| format!("tar entry: {e}"))?;
        let path = entry
            .path()
            .map_err(|e| format!("tar path: {e}"))?
            .to_string_lossy()
            .to_string();
        if path == want_path {
            let mut toml = String::new();
            use std::io::Read;
            entry
                .read_to_string(&mut toml)
                .map_err(|e| format!("read Cargo.toml: {e}"))?;
            found = Some(toml);
            break;
        }
    }
    let toml = found.ok_or_else(|| format!("tarball missing {want_path}"))?;
    let get_str = |key: &str| -> Option<String> {
        // naive scan of the [package] section; the metadata is the
        // authoritative source, this is a consistency check only
        let mut in_package = false;
        for line in toml.lines() {
            let t = line.trim();
            if t.starts_with('[') {
                in_package = t == "[package]";
                continue;
            }
            if in_package {
                if let Some((k, v)) = t.split_once('=') {
                    if k.trim() == key {
                        return Some(v.trim().trim_matches('"').to_string());
                    }
                }
            }
        }
        None
    };
    match get_str("name") {
        Some(n) if n == name => {}
        other => return Err(format!("Cargo.toml name {other:?} != metadata {name:?}")),
    }
    match get_str("version") {
        Some(v) if v == vers => {}
        other => return Err(format!("Cargo.toml version {other:?} != metadata {vers:?}")),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(meta: &str, file: &[u8]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&(meta.len() as u32).to_le_bytes());
        b.extend_from_slice(meta.as_bytes());
        b.extend_from_slice(&(file.len() as u32).to_le_bytes());
        b.extend_from_slice(file);
        b
    }

    #[test]
    fn framing_roundtrip_and_rejects_overrun() {
        let body = frame(r#"{"name":"a","vers":"0.1.0"}"#, b"pretend tarball");
        let p = parse_publish_body(&body).unwrap();
        assert_eq!(p.meta["name"], "a");
        assert_eq!(p.crate_bytes, b"pretend tarball");

        let mut bad = body.clone();
        bad.extend_from_slice(b"trailing");
        assert!(parse_publish_body(&bad).is_err());

        let mut overrun = body.clone();
        overrun[0] = 0xFF; // metadata length = huge
        assert!(parse_publish_body(&overrun).is_err());
    }

    #[test]
    fn sha256_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
