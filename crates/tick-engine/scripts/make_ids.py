#!/usr/bin/env python3
"""make_ids.py — battery prompt → token ids under the lane's ACTUAL
chat template.

DISCOVERY (2026-11-22): llama-cli -st (chat mode) applies the model's
GGUF chat_template, which for 2B4T is the CUSTOM one —
    "User: {prompt}<|eot_id|>Assistant: "
NOT the llama3 header scaffold. Verified against the lane itself:
verbose log reports task.n_tokens (p03: 12) and these ids match it.
The old p03_ids.json (17-token llama3 scaffold) did NOT match what
the reference was fed — it only ever anchored gguf-vs-tick format
parity, where any ids work.

Output: tests/fixtures/tick-engine/pNN_ids.json for every battery
case, using tok.py's BPE (verified byte-identical vs llama-tokenize)
with special-token splitting.
"""
import json, re, sys, importlib.util, pathlib

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tok", HERE / "tok.py")
tok = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tok)

BATTERY = HERE / "../../../tests/fixtures/tick-engine/battery.json"

def main():
    tokens, merges, pre, bos = tok.load()
    special = [t for t in tokens if t.startswith("<|")]
    pat = "(" + "|".join(map(re.escape, sorted(special, key=len, reverse=True))) + ")"

    def enc_special(s):
        out = []
        for seg in re.split(pat, s):
            if not seg:
                continue
            if seg in special:
                out.append(tokens.index(seg))
            else:
                out.extend(tok.encode(seg, tokens, merges))
        return out

    cases = json.loads(BATTERY.read_text())["cases"]
    for case in cases:
        prompt = case["prompt"]
        # the GGUF chat_template with add_generation_prompt, + BOS
        # (tokenizer.ggml.add_bos_token = true)
        text = f"User: {prompt}<|eot_id|>Assistant: "
        ids = [bos] + enc_special(text)
        out = HERE / f"../../../tests/fixtures/tick-engine/{case['id']}_ids.json"
        out.write_text(json.dumps(ids))
        print(f"{case['id']}: {len(ids)} tokens")

if __name__ == "__main__":
    main()
