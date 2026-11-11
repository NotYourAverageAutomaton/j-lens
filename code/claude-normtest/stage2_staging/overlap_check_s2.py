"""Stage-2 disjointness (Claude 2026-11-09; plan post P140): overlap_check.py (fc221155) extended. The 32 Stage-2 docs vs ALL 34
existing texts (16 Stage-1, 6 held, 6 record, 6 fresh step-2 passages) AND pairwise within Stage 2, same normalised-sha +
12-word-shingle functions; optional 2nd arg = the 4 spares, checked the same way (against old, the 32 and each other). Token count with the model tokenizer (each must reach 256 with BOS). Must print ALL CLEAR."""
import ast, hashlib, json, re, sys
from transformers import AutoTokenizer
B = "$JLENS_HOME/claude-normtest"; P = json.load(open(sys.argv[1]))
def lst(path, name):
    for n in ast.parse(open(path).read()).body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == name for t in n.targets): return ast.literal_eval(n.value)
old = [d["text"] for d in json.load(open(f"{B}/stage1_corpus.json"))] + lst(f"{B}/pilot_hutch.py", "HELD_OUT") + lst("$JLENS_HOME/whole_model_fit.py", "PROMPTS") \
    + [p["text"] for p in json.load(open(f"{B}/step2/passages.json"))]
assert len(old) == 34, len(old)
assert len(P) == 32, len(P)
if len(sys.argv) > 2: SP = json.load(open(sys.argv[2])); assert len(SP) == 4, len(SP); P = P + SP   # post P141 #3: spares validated the same way
norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
def sh(s, k=12):
    w = norm(s).split(); return {" ".join(w[i:i + k]) for i in range(max(0, len(w) - k + 1))}
OS = set().union(*[sh(t) for t in old]); OH = {hashlib.sha256(norm(t).encode()).hexdigest() for t in old}
tok = AutoTokenizer.from_pretrained("$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b")
bad = 0
for i, p in enumerate(P):
    if hashlib.sha256(p["text"].encode()).hexdigest() != p["sha256"]: print(p["source"], "sha256 MISMATCH"); bad += 1; continue
    others = set().union(*[sh(q["text"]) for j, q in enumerate(P) if j != i])
    ov = len(sh(p["text"]) & OS) + (hashlib.sha256(norm(p["text"]).encode()).hexdigest() in OH); ovs = len(sh(p["text"]) & others)
    n = len(tok(p["text"], add_special_tokens=False).input_ids); fail = ov > 0 or ovs > 0 or n < 255; bad += fail
    print(f'{p["genre"]:24s} {p["source"][:34]:34s} overlap-old {ov} overlap-s2 {ovs} tokens {n} {"FAIL" if fail else "OK"}')
print("ALL CLEAR" if bad == 0 else f"{bad} FAILED")
