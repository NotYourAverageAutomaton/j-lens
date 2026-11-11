"""Step-2 disjointness (Claude 2026-11-07; Deron P118): normalised-sha + 12-word-shingle overlap of the 6 passages vs ALL 28
existing texts (16 Stage-1, 6 held, 6 record), the same functions as score_stage1. Also token count with the model tokenizer
(each must reach 256 with BOS so positions 16..254 exist). Must print OVERLAP 0 for all six."""
import ast, hashlib, json, re, sys
from transformers import AutoTokenizer
B = "$JLENS_HOME/claude-normtest"; P = json.load(open(sys.argv[1]))
def lst(path, name):
    for n in ast.parse(open(path).read()).body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == name for t in n.targets): return ast.literal_eval(n.value)
old = [d["text"] for d in json.load(open(f"{B}/stage1_corpus.json"))] + lst(f"{B}/pilot_hutch.py", "HELD_OUT") + lst("$JLENS_HOME/whole_model_fit.py", "PROMPTS")
assert len(old) == 28, len(old)
norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
def sh(s, k=12):
    w = norm(s).split(); return {" ".join(w[i:i + k]) for i in range(max(0, len(w) - k + 1))}
OS = set().union(*[sh(t) for t in old]); OH = {hashlib.sha256(norm(t).encode()).hexdigest() for t in old}
tok = AutoTokenizer.from_pretrained("$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b")
bad = 0
for p in P:
    ov = len(sh(p["text"]) & OS) + (hashlib.sha256(norm(p["text"]).encode()).hexdigest() in OH)
    n = len(tok(p["text"], add_special_tokens=False).input_ids); bad += ov > 0 or n < 255
    print(f'{p["genre"]:20s} sha {p["sha256"][:12]} shingle/sha overlap {ov}  tokens {n} {"OK" if ov == 0 and n >= 255 else "FAIL"}')
print("ALL CLEAR" if bad == 0 else f"{bad} FAILED")
