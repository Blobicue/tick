#!/bin/bash
# gen_battery.sh — regenerate the golden parity battery from the FROZEN
# reference lane (patched bitnet.cpp, ReLU^2 activation, BitNet
# b1.58-2B-4T tq1_0, greedy temp 0, CPU-only 10 threads).
#
# Regenerating with ANY other binary/model/flags invalidates parity.
set -euo pipefail

BIN=/home/juzz/models/bitnet/llama-patched/build/bin/llama-cli
MODEL=/home/juzz/models/bitnet/bitnet-b1.58-2B-4T-tq1_0.gguf
OUT="$(dirname "$0")/../../../tests/fixtures/tick-engine/battery.json"
export CUDA_VISIBLE_DEVICES=""
export LD_LIBRARY_PATH=/home/juzz/models/bitnet/llama-patched/build/bin

PROMPTS=(
  "What is 17 plus 25?"
  "What is 9 times 6?"
  "The capital of France is"
  "Name three colors."
  "Write a bash loop that prints the numbers 1 to 5:"
  "Once upon a time in a lighthouse,"
  "def fibonacci(n):"
  "Water boils at"
  "List the first five prime numbers."
  "Translate 'good morning' to French:"
)

BIN_SHA=$(sha256sum "$BIN" | cut -d' ' -f1)
MODEL_SHA=$(sha256sum "$MODEL" | cut -d' ' -f1)
echo "lane: llama-cli $BIN_SHA"
echo "model: $MODEL_SHA"

python3 - "$BIN" "$MODEL" "$BIN_SHA" "$MODEL_SHA" "$OUT" "${PROMPTS[@]}" <<'EOF'
import json, subprocess, sys, time

bin_path, model, bin_sha, model_sha, out = sys.argv[1:6]
prompts = sys.argv[6:]

def gen(prompt, n=64):
    r = subprocess.run(
        [bin_path, "-m", model, "-ngl", "0", "-t", "10", "--temp", "0",
         "-n", str(n), "-st", "--no-display-prompt", "-p", prompt],
        capture_output=True, timeout=600, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        sys.exit(f"lane failed on {prompt!r}: {r.stderr[-400:]}")
    out = r.stdout.decode("utf-8", errors="replace")
    lines = out.splitlines()
    # anchor: the LAST "> <prompt>" echo line; generation follows it
    anchor = max((i for i, l in enumerate(lines) if l == f"> {prompt}"), default=-1)
    if anchor < 0:
        sys.exit(f"no prompt echo found for {prompt!r} — lane output format changed")
    body = []
    for l in lines[anchor + 1:]:
        if l.startswith("[ ") or l.startswith("Exiting"):
            break
        body.append(l)
    # drop leading decoration leftovers (spinner residue / blank lines)
    while body and (not body[0].strip() or "\x08" in body[0][:24] and not body[0].split("\x08")[-1].strip()):
        body.pop(0)
    if body and "\x08" in body[0]:
        body[0] = body[0].split("\x08")[-1]  # text after the spinner residue
    return "\n".join(body).strip()

cases = []
for i, p in enumerate(prompts):
    t0 = time.time()
    text = gen(p)
    cases.append({"id": f"p{i+1:02d}", "prompt": p, "output": text,
                  "tokens": 64, "wall_s": round(time.time() - t0, 2)})
    print(f"  p{i+1:02d} ({cases[-1]['wall_s']}s): {text[:60]!r}")

doc = {"meta": {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "llama_cli": bin_path, "llama_cli_sha256": bin_sha,
        "model": model, "model_sha256": model_sha,
        "flags": "-ngl 0 -t 10 --temp 0 -n 64 -st --no-display-prompt",
        "temp": 0.0,
        "activation": "ReLU^2 (patched lane; upstream SiLU is WRONG for 2B4T)",
       },
       "cases": cases}
with open(out, "w") as f:
    json.dump(doc, f, indent=1)
print(f"wrote {out} ({len(cases)} cases)")
EOF
