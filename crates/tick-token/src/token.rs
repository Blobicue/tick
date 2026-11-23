//! In-crate tokenizer: the llama-bpe (llama3) pre-tokenizer as a
//! regex-free scanner over codepoint flags + GPT-2 style BPE over
//! byte-encoded words — faithful to llama.cpp's tokenize path for
//! tokenizer.ggml.model=gpt2 with pre=llama-bpe, which is what the
//! frozen lane (and scripts/tok.py, verified byte-identical vs
//! llama-tokenize) applies. The Vocab core and unicode tables are
//! ported from spark-engine; the pre-tokenizer differs from
//! spark-engine's o200k scanner (no up/low splitting, no '/' in the
//! punctuation tail, contraction is a separate alternative).

use crate::unicode::flags_of;
use std::collections::HashMap;
use std::sync::OnceLock;
use tick_format::gguf::{Gguf, Value};

pub struct Vocab {
    pub token_to_id: HashMap<String, u32>,
    pub bpe_ranks: HashMap<(String, String), u32>,
    /// control/user-defined tokens, longest-first for leftmost-longest scan
    pub specials: Vec<(String, u32)>,
    /// id -> raw vocab text (byte-encoded form for normal tokens)
    pub texts: Vec<String>,
}

impl Vocab {
    /// tokens + merges + token types from gguf metadata (duplicate token
    /// texts resolve last-wins, duplicate merge pairs first-wins —
    /// llama.cpp semantics, and tok.py agrees on this vocabulary).
    /// A byte-level vocab: id == byte value for 0..255. The fallback
    /// for standalone tickfiles (e.g. tick-train tiny artifacts) that
    /// ship without a sibling GGUF.
    pub fn byte_level() -> Vocab {
        let mut token_to_id = std::collections::HashMap::new();
        let mut texts = Vec::with_capacity(256);
        for b in 0u32..256 {
            let c = char::from_u32(b).unwrap();
            token_to_id.insert(c.to_string(), b);
            texts.push(c.to_string());
        }
        Vocab {
            token_to_id,
            bpe_ranks: Default::default(),
            specials: Vec::new(),
            texts,
        }
    }

    pub fn from_meta(meta: &std::collections::BTreeMap<String, Value>) -> Result<Vocab, String> {
        let tokens = match meta.get("tokenizer.ggml.tokens") {
            Some(Value::Arr(a)) => a,
            _ => return Err("tokenizer.ggml.tokens missing".into()),
        };
        let types = match meta.get("tokenizer.ggml.token_type") {
            Some(Value::Arr(a)) => a,
            _ => return Err("tokenizer.ggml.token_type missing".into()),
        };
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        let mut specials = Vec::new();
        for (i, t) in tokens.iter().enumerate() {
            if let Value::Str(s) = t {
                token_to_id.insert(s.clone(), i as u32);
                // 3 = CONTROL, 4 = USER_DEFINED (llama splits on
                // these); accept any int width — the tickfile KV
                // round-trip widens I32 arrays to U64
                let is_special = matches!(types.get(i).and_then(|v| v.as_u64()), Some(3) | Some(4));
                if is_special {
                    specials.push((s.clone(), i as u32));
                }
            }
        }
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.0.cmp(&b.0)));
        let mut bpe_ranks = HashMap::new();
        if let Some(Value::Arr(merges)) = meta.get("tokenizer.ggml.merges") {
            for (rank, m) in merges.iter().enumerate() {
                if let Value::Str(s) = m {
                    if let Some((a, b)) = s.split_once(' ') {
                        bpe_ranks
                            .entry((a.to_string(), b.to_string()))
                            .or_insert(rank as u32);
                    }
                }
            }
        }
        let texts = tokens
            .iter()
            .map(|t| {
                if let Value::Str(s) = t {
                    s.clone()
                } else {
                    String::new()
                }
            })
            .collect();
        Ok(Vocab {
            token_to_id,
            bpe_ranks,
            specials,
            texts,
        })
    }

    pub fn from_gguf(g: &Gguf) -> Result<Vocab, String> {
        Self::from_meta(&g.meta)
    }

    pub fn from_tickfile(t: &tick_format::weights::TickFile) -> Result<Vocab, String> {
        Self::from_meta(&t.meta)
    }
}

/// GPT-2 byte-to-visible-character map (llama's unicode_byte_to_utf8):
/// printable latin-1 ranges map to themselves, everything else to
/// U+0100 + n in byte order.
fn byte_to_utf8(b: u8) -> &'static str {
    static MAP: OnceLock<Box<[String; 256]>> = OnceLock::new();
    let map = MAP.get_or_init(|| {
        let mut m: Box<[String; 256]> = Box::new(std::array::from_fn(|_| String::new()));
        for ch in [0x21u32..=0x7E, 0xA1..=0xAC, 0xAE..=0xFF]
            .into_iter()
            .flatten()
        {
            m[ch as usize] = char::from_u32(ch).map_or_else(String::new, |c| c.to_string());
        }
        let mut n = 0u32;
        for ch in 0..256u32 {
            if m[ch as usize].is_empty() {
                m[ch as usize] =
                    char::from_u32(256 + n).map_or_else(String::new, |c| c.to_string());
                n += 1;
            }
        }
        m
    });
    &map[b as usize]
}

/// Inverse of the GPT-2 byte map: visible char -> original byte.
fn utf8_to_byte(c: char) -> Option<u8> {
    static REV: OnceLock<HashMap<char, u8>> = OnceLock::new();
    let rev = REV.get_or_init(|| {
        let mut m = HashMap::new();
        for b in 0..=255u8 {
            m.insert(byte_to_utf8(b).chars().next().unwrap_or('\0'), b);
        }
        m
    });
    rev.get(&c).copied()
}

/// Contraction suffix: (?i:'s|'t|'re|'ve|'m|'ll|'d) — returns length.
fn contraction_len(chars: &[char], at: usize) -> usize {
    if chars.get(at) != Some(&'\'') {
        return 0;
    }
    let either = |i: usize, a: char, b: char| chars.get(i) == Some(&a) || chars.get(i) == Some(&b);
    if either(at + 1, 's', 'S')
        || either(at + 1, 't', 'T')
        || either(at + 1, 'm', 'M')
        || either(at + 1, 'd', 'D')
    {
        return 2;
    }
    if either(at + 1, 'r', 'R') && either(at + 2, 'e', 'E') {
        return 3;
    }
    if either(at + 1, 'v', 'V') && either(at + 2, 'e', 'E') {
        return 3;
    }
    if either(at + 1, 'l', 'L') && either(at + 2, 'l', 'L') {
        return 3;
    }
    0
}

/// The llama3 pre-tokenizer, regex-free. Implements
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
///  ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
/// as an ordered-alternative machine over codepoints (regex
/// alternation is leftmost-first, greedy with backtracking on the
/// optional classes). Returns char-index ranges.
pub fn split_llama3(text: &str) -> Vec<(usize, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let fl: Vec<_> = chars.iter().map(|&c| flags_of(c as u32)).collect();
    let letter = |i: usize| i < n && fl[i].letter();
    let number = |i: usize| i < n && fl[i].number();
    let ws = |i: usize| i < n && fl[i].whitespace();
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut pos = 0usize;
    while pos < n {
        // branch 1: (?i: contraction)
        let cl = contraction_len(&chars, pos);
        if cl > 0 {
            out.push((pos, pos + cl));
            pos += cl;
            continue;
        }
        // branch 2: [^\r\n\p{L}\p{N}]? \p{L}+ — the optional prefix
        // may be whitespace/punct/mark (anything but \r \n letters
        // numbers); try prefix first, backtrack to empty on failure
        let c = chars[pos];
        let f = fl[pos];
        let prefixed = c != '\r' && c != '\n' && !f.letter() && !f.number();
        let mut matched = None;
        if prefixed && letter(pos + 1) {
            let mut e = pos + 1;
            while letter(e) {
                e += 1;
            }
            matched = Some(e);
        } else if letter(pos) {
            let mut e = pos;
            while letter(e) {
                e += 1;
            }
            matched = Some(e);
        }
        if let Some(end) = matched {
            out.push((pos, end));
            pos = end;
            continue;
        }
        // branch 3: \p{N}{1,3}, grouped in threes from the left
        if number(pos) {
            let mut ini = pos;
            let mut p = pos;
            while number(p) {
                p += 1;
                if p - ini >= 3 {
                    out.push((ini, p));
                    ini = p;
                }
            }
            if p > ini {
                out.push((ini, p));
            }
            pos = p;
            continue;
        }
        // branch 4: ` ?[^\s\p{L}\p{N}]+[\r\n]*` — a LITERAL space (not
        // \s), then the punctuation/symbol class, then \r\n tail
        let after_space = if c == ' ' && pos + 1 < n {
            pos + 1
        } else {
            pos
        };
        let f2 = fl[after_space];
        if (after_space > pos || (!f2.whitespace() && !f2.letter() && !f2.number()))
            && !f2.whitespace()
            && !f2.letter()
            && !f2.number()
        {
            let mut p = after_space;
            while p < n && !fl[p].whitespace() && !fl[p].letter() && !fl[p].number() {
                p += 1;
            }
            while p < n && matches!(chars[p], '\r' | '\n') {
                p += 1;
            }
            out.push((pos, p));
            pos = p;
            continue;
        }
        // branches 5-7: whitespace runs
        let mut num_ws = 0usize;
        let mut last_rn_end = 0usize;
        while pos + num_ws < n && ws(pos + num_ws) {
            if matches!(chars[pos + num_ws], '\r' | '\n') {
                last_rn_end = pos + num_ws + 1;
            }
            num_ws += 1;
        }
        if num_ws > 0 {
            if last_rn_end > 0 {
                // \s*[\r\n]+: through the last \r or \n of the run
                out.push((pos, last_rn_end));
                pos = last_rn_end;
            } else if num_ws > 1 && pos + num_ws < n {
                // \s+(?!\S): all but one; the last prefixes the next word
                out.push((pos, pos + num_ws - 1));
                pos += num_ws - 1;
            } else {
                out.push((pos, pos + num_ws));
                pos += num_ws;
            }
            continue;
        }
        // no match: single codepoint
        out.push((pos, pos + 1));
        pos += 1;
    }
    out
}

/// char-index range -> byte range in the source text.
pub fn range_to_bytes(text: &str, r: (usize, usize)) -> (usize, usize) {
    let start = text
        .char_indices()
        .nth(r.0)
        .map(|(b, _)| b)
        .unwrap_or(text.len());
    let end = text
        .char_indices()
        .nth(r.1)
        .map(|(b, _)| b)
        .unwrap_or(text.len());
    (start, end)
}

/// Decode token ids to text by reversing the GPT-2 byte map; ids that
/// are not byte-encoded (specials) pass through literally.
pub fn decode(ids: &[u32], vocab: &Vocab) -> String {
    let mut out = Vec::new();
    for &id in ids {
        let Some(text) = vocab.texts.get(id as usize) else {
            continue;
        };
        let raw = text.chars().all(|c| utf8_to_byte(c).is_some());
        if raw {
            for c in text.chars() {
                out.push(utf8_to_byte(c).unwrap_or(b'?'));
            }
        } else {
            out.extend_from_slice(text.as_bytes());
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// BPE over one byte-encoded word: repeatedly merge the adjacent pair
/// with the lowest rank (ties by leftmost) — llama's greedy rank queue
/// for words short enough that a scan is equivalent and simpler.
fn bpe_word(word: &str, vocab: &Vocab) -> Vec<u32> {
    let mut syms: Vec<String> = word.chars().map(|c| c.to_string()).collect();
    loop {
        let mut best: Option<(u32, usize)> = None;
        for i in 0..syms.len().saturating_sub(1) {
            if let Some(&rank) = vocab.bpe_ranks.get(&(syms[i].clone(), syms[i + 1].clone())) {
                let take = match best {
                    None => true,
                    Some((br, bi)) => (rank, i) < (br, bi),
                };
                if take {
                    best = Some((rank, i));
                }
            }
        }
        let Some((_, i)) = best else { break };
        let right = syms.remove(i + 1);
        syms[i].push_str(&right);
    }
    let mut ids = Vec::with_capacity(syms.len());
    for s in &syms {
        if let Some(&id) = vocab.token_to_id.get(s) {
            ids.push(id);
        } else {
            // llama's byte fallback: per raw byte, silently skipping misses
            for b in s.bytes() {
                if let Some(&id) = vocab.token_to_id.get(byte_to_utf8(b)) {
                    ids.push(id);
                }
            }
        }
    }
    ids
}

/// Full encode: split on special tokens first (llama's tokenize does
/// this before pre-tokenization), then pretokenize + byte-encode + BPE
/// each plain segment.
pub fn encode(text: &str, vocab: &Vocab) -> Vec<u32> {
    let mut out = Vec::new();
    let mut rest = text;
    'outer: loop {
        if vocab.specials.is_empty() {
            out.extend(encode_plain(rest, vocab));
            break;
        }
        // leftmost special occurrence, longest spelling wins
        let mut best: Option<usize> = None;
        let mut best_tok = 0u32;
        for (spec, id) in &vocab.specials {
            if let Some(at) = rest.find(spec.as_str()) {
                if best.is_none() || at < best.unwrap() {
                    best = Some(at);
                    best_tok = *id;
                }
            }
        }
        let Some(at) = best else {
            out.extend(encode_plain(rest, vocab));
            break 'outer;
        };
        out.extend(encode_plain(&rest[..at], vocab));
        let spec_len = vocab
            .specials
            .iter()
            .filter(|(_, id)| *id == best_tok)
            .map(|(s, _)| s.len())
            .max()
            .unwrap_or(0);
        out.push(best_tok);
        rest = &rest[at + spec_len..];
    }
    out
}

fn encode_plain(text: &str, vocab: &Vocab) -> Vec<u32> {
    let mut out = Vec::new();
    for r in split_llama3(text) {
        let (s, e) = range_to_bytes(text, r);
        let word = &text[s..e];
        let encoded: String = word.bytes().map(byte_to_utf8).collect();
        out.extend(bpe_word(&encoded, vocab));
    }
    out
}

/// The frozen lane's chat template (the model's own CUSTOM
/// gguf chat_template with add_generation_prompt + BOS): everything
/// the battery prompts need, byte-for-byte as make_ids.py built it.
pub fn encode_chat(prompt: &str, vocab: &Vocab) -> Vec<u32> {
    let bos = vocab
        .token_to_id
        .get("<|begin_of_text|>")
        .copied()
        .unwrap_or(128000);
    let text = format!("User: {prompt}<|eot_id|>Assistant: ");
    let mut ids = vec![bos];
    ids.extend(encode(&text, vocab));
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(t: &str) -> Vec<String> {
        split_llama3(t)
            .iter()
            .map(|&r| {
                let (a, b) = range_to_bytes(t, r);
                t[a..b].to_string()
            })
            .collect()
    }

    #[test]
    fn llama3_splits() {
        // contraction is its own piece; no case splitting; digits in 3s
        assert_eq!(pieces("it's"), vec!["it", "'s"]);
        assert_eq!(pieces("I'M"), vec!["I", "'M"]);
        assert_eq!(pieces("aBc"), vec!["aBc"]);
        assert_eq!(pieces("1234567"), vec!["123", "456", "7"]);
        // leading space joins the word (branch-2 prefix)
        assert_eq!(pieces("a  b"), vec!["a", " ", " b"]);
        assert_eq!(pieces("caf\u{e9}"), vec!["caf\u{e9}"]);
        // '/' is a branch-2 prefix: it joins the following word
        assert_eq!(pieces("a/b/c"), vec!["a", "/b", "/c"]);
        // space + punctuation fuses; \r\n tail belongs to the piece
        assert_eq!(pieces("x !\ny"), vec!["x", " !\n", "y"]);
        // non-contraction apostrophe prefixes the word
        assert_eq!(pieces("l'h\u{f4}tel"), vec!["l", "'h\u{f4}tel"]);
        assert_eq!(pieces("3.14"), vec!["3", ".", "14"]);
        // trailing whitespace keeps all-but-one before a word
        assert_eq!(pieces("a   b"), vec!["a", "  ", " b"]);
        // newline runs go through the last \r or \n
        assert_eq!(pieces("a\n\n\nb"), vec!["a", "\n\n\n", "b"]);
    }

    #[test]
    fn byte_map_identity_and_shift() {
        assert_eq!(byte_to_utf8(b'a'), "a");
        assert_eq!(byte_to_utf8(b' '), "\u{120}"); // Ġ
        assert_eq!(byte_to_utf8(0x00), "\u{100}");
        assert_eq!(byte_to_utf8(0xFF), "\u{ff}");
    }
}
