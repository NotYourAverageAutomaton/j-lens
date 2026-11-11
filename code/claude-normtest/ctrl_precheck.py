"""Control validity pre-check, CPU only (Claude 2026-11-04; Deron P046 items 3+4, card C01). No GPU, no window.
SELF-CHECK first: recompute eval.py's fut20 for jlens / logit / old ctrl from held_h.pt on CPU and require it matches
eval_record.json (e02c34da) within 0.002 at every layer. Otherwise the CPU path isn't eval's path: STOP AND LOOK (exit 2).
Then the signed controls, each must FAIL to be usable: fut20 < .01 AND below logit at every scored layer.
  PRIMARY   = row-permuted J_l (P J_l, fixed seed): same singular values, wrong map
  SECONDARY = J_{l'} with |l'-l| >= 16 and cos(J_{l'}, I) < .5 (the farthest such layer)"""
import json, os, sys, time, torch
from safetensors import safe_open
from transformers import AutoTokenizer
torch.set_num_threads(max(1, os.cpu_count() - 2))
OUT = "$JLENS_HOME/claude-normtest"; JD = "$JLENS_HOME/results-whole-swift-record"; MD = "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b"
LAYERS = [16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 62]; OLD_CTRL = {l: (62 if l < 40 else 16) for l in LAYERS}; EPS = 1e-6; SKIP = 16
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)
idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
def W(n):
    with safe_open(os.path.join(MD, idx[n]), "pt") as f: return f.get_tensor(n).float()
WU = W("lm_head.weight"); g1 = 1.0 + W("model.language_model.norm.weight")
held = torch.load(os.path.join(OUT, "held_h.pt")); ref = json.load(open(os.path.join(OUT, "eval_record.json")))["layers"]
src = open(os.path.join(OUT, "pilot_hutch.py")).read()
HELD_OUT = eval(src[src.index("HELD_OUT = ["):].split("\n]\n")[0].split("=", 1)[1] + "\n]")
tok = AutoTokenizer.from_pretrained(MD); bos = tok.bos_token_id
nexts = []   # per held position, the next <=10 actual token ids (eval.py: ids[0, q+1:q+11])
for text in HELD_OUT:
    ids = tok(text, truncation=True, max_length=256 - (1 if bos is not None else 0), add_special_tokens=False).input_ids
    ids = ([bos] if bos is not None else []) + ids
    for q in range(SKIP, len(ids) - 1): nexts.append(torch.tensor(ids[q + 1:q + 11]))
assert len(nexts) == held["h_final"].shape[0], f"position count {len(nexts)} != held {held['h_final'].shape[0]}"
log("positions", len(nexts), "(matches held_h.pt)")
def fut20(x):
    xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * g1
    t20 = (xn @ WU.T).topk(20, -1).indices
    return sum(float(torch.isin(n, t20[i]).float().mean()) for i, n in enumerate(nexts)) / len(nexts)
Jc = {}
def J(l):
    if l not in Jc: Jc[l] = torch.load(os.path.join(JD, f"J_l_{l}.pt")).float()
    return Jc[l]
eye = torch.eye(5120)
cosI = {l: float((J(l) * eye).sum() / J(l).norm() / eye.norm()) for l in range(16, 63)}
res = {"selfcheck": {}, "primary": {}, "secondary": {}, "secondary_layer": {}, "logit": {}}
gen = torch.Generator().manual_seed(20261104)
ok_self = True
for l in LAYERS:
    h = held["layers"][l]
    fj, fl, fc = fut20(h @ J(l).T), fut20(h), fut20(h @ J(OLD_CTRL[l]).T)
    r = ref[str(l)]; d = max(abs(fj - r["jlens"]["fut20"]), abs(fl - r["logit"]["fut20"]), abs(fc - r["ctrl"]["fut20"]))
    res["selfcheck"][l] = {"jlens": round(fj, 4), "logit": round(fl, 4), "ctrl": round(fc, 4), "max_abs_diff_vs_eval": round(d, 5)}; res["logit"][l] = fl
    if d > 0.002: ok_self = False
    perm = torch.randperm(5120, generator=gen); res["primary"][l] = fut20(h @ J(l)[perm].T)
    cands = [m for m in range(16, 63) if abs(m - l) >= 16 and cosI[m] < .5]
    m = max(cands, key=lambda m: abs(m - l)) if cands else None; res["secondary_layer"][l] = m
    res["secondary"][l] = fut20(h @ J(m).T) if m is not None else None
    log(f"L{l}: self d={d:.5f} | logit {fl:.4f} primary {res['primary'][l]:.4f} secondary(L{m}) {res['secondary'][l] if m is None else round(res['secondary'][l], 4)}")
res["selfcheck_pass"] = ok_self
if not ok_self:
    json.dump(res, open(os.path.join(OUT, "ctrl_precheck.json"), "w"), indent=1); print("SELF-CHECK FAILED: CPU path != eval path. STOP AND LOOK."); sys.exit(2)
for name in ("primary", "secondary"):
    bad = [l for l in LAYERS if res[name][l] is None or not (res[name][l] < .01 and res[name][l] < res["logit"][l])]
    res[f"{name}_valid"] = not bad; res[f"{name}_fails_at"] = bad
json.dump(res, open(os.path.join(OUT, "ctrl_precheck.json"), "w"), indent=1)
print("SELF-CHECK PASS | primary valid:", res["primary_valid"], res["primary_fails_at"], "| secondary valid:", res["secondary_valid"], res["secondary_fails_at"])
