#!/usr/bin/env python3
"""tokenize.py — Llama3-style BPE against the BitNet GGUF vocab.

M2 seam: the engine consumes token ids; this script produces them from
the same model file the reference lane loads (gguf-py reader, standard
gpt2/llama3 byte-level BPE with the llama3 pretokenizer regex).
"""
import json, sys
sys.path.insert(0, "/home/juzz/models/bitnet/llama-patched/gguf-py")
from gguf import GGUFReader

MODEL = sys.argv[2] if len(sys.argv) > 2 else \
    "/home/juzz/models/bitnet/bitnet-b1.58-2B-4T-tq1_0.gguf"

def load():
    r = GGUFReader(MODEL)
    meta = {f.name: f for f in r.fields.values()}
    def parts(f):
        return f.parts.tolist() if hasattr(f.parts, "tolist") else list(f.parts)
    tokens = []
    for f in r.fields.values():
        if f.name == "tokenizer.ggml.tokens":
            # string array: parts slices; decode via reader data
            tokens = [t.decode("utf-8", "replace") if isinstance(t, bytes) else str(t)
                      for t in read_str_array(r, f)]
    merges = []
    for f in r.fields.values():
        if f.name == "tokenizer.ggml.merges":
            merges = [t.decode("utf-8", "replace") if isinstance(t, bytes) else str(t)
                      for t in read_str_array(r, f)]
    pre = None
    for f in r.fields.values():
        if f.name == "tokenizer.ggml.pre":
            pre = bytes(f.parts[f.data[0]]).decode()
    bos = 128000
    for f in r.fields.values():
        if f.name == "tokenizer.ggml.bos_token_id":
            bos = int(f.parts[f.data[0]][0])
    return tokens, merges, pre, bos

def read_str_array(reader, field):
    # field.data = [idx into parts for count, idx for first item...]
    # GGUFReader exposes str arrays via field.parts; simplest: re-scan
    # using reader._data and offsets is fiddly — use field.t? gguf-py
    # sets field.t = kvtARR with items in field.data; strings live in
    # field.parts slices. The reader builds a list on demand:
    out = []
    for idx in field.data:
        part = field.parts[idx]
        out.append(bytes(part))
    return out

def b2u():
    # gpt2 bytes_to_unicode: identity bytes stay, the rest shift to 256+n
    bs = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    out = {}
    n = 0
    for b in range(256):
        if b in bs:
            out[b] = chr(b)
        else:
            out[b] = chr(256 + n)
            n += 1
    return out

B2U = b2u()

def encode(text, tokens, merges):
    tok_to_id = {t: i for i, t in enumerate(tokens)}
    ranks = {}
    for i, m in enumerate(merges):
        a, b = m.split(" ", 1) if " " in m else (m, "")
        ranks[(a, b)] = i
    import regex as re
    pat = re.compile(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
    ids = []
    for piece in pat.findall(text):
        chunk = "".join(B2U[b] for b in piece.encode("utf-8"))
        parts = [c for c in chunk]
        while len(parts) > 1:
            best = None
            for i in range(len(parts) - 1):
                r = ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best is None or r < best[0]):
                    best = (r, i)
            if best is None:
                break
            _, i = best
            parts = parts[:i] + [parts[i] + parts[i + 1]] + parts[i + 2:]
        for p in parts:
            ids.append(tok_to_id[p])
    return ids

if __name__ == "__main__":
    text = sys.argv[1]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    tokens, merges, pre, bos = load()
    ids = [bos] + encode(text, tokens, merges)
    print(json.dumps({"ids": ids, "n": n, "pre": pre, "vocab": len(tokens)}))
