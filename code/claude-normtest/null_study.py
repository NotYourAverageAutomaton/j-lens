"""Null study for the shrink check (Claude 2026-11-07, report-only, NOT the gated check). How often does a matrix with NO
structure pass the S-family cross-fit + bootstrap bar (lo at 1.25% > 0)? Two nulls, 8 draws each, per mid-band layer:
  noise: Gaussian with the Frobenius norm of D6 = J6 - I (record J, already published and scored)
  perm:  D6 row-permuted (as the must-fail control does to D22)
Reads only the record J6, held_h, the head. Never reads Stage 1 J_p or J22. fut20 path as shrink_check.py."""
import json, os, sys, time, torch
from safetensors import safe_open
from transformers import AutoTokenizer
import ast
torch.set_num_threads(int(os.environ.get("SCORE_THREADS", 12)))
BASE = "$JLENS_HOME/claude-normtest"; JD = "$JLENS_HOME/results-whole-swift-record"; MD = "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b"
GATE_L = [20, 24, 28, 32, 36, 40, 44]; ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]; EPS = 1e-6; SKIP = 16; B = 10000; DRAWS = 8
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
def W(n):
    with safe_open(os.path.join(MD, idx[n]), "pt") as f: return f.get_tensor(n).float()
WU = W("lm_head.weight"); g1 = 1.0 + W("model.language_model.norm.weight"); held = torch.load(os.path.join(BASE, "held_h.pt"))
tree = ast.parse(open(os.path.join(BASE, "pilot_hutch.py")).read())
HELD_OUT = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "HELD_OUT" for t in n.targets)][0]
tok = AutoTokenizer.from_pretrained(MD); bos = tok.bos_token_id; nexts = []
for text in HELD_OUT:
    ids = tok(text, truncation=True, max_length=256 - (1 if bos is not None else 0), add_special_tokens=False).input_ids
    ids = ([bos] if bos is not None else []) + ids
    for q in range(SKIP, len(ids) - 1): nexts.append(torch.tensor(ids[q + 1:q + 11]))
NPOS = len(nexts); passage = torch.tensor(held["passage"]); inA = torch.tensor([int(p) in (0, 1, 2) for p in passage.tolist()]); inB = ~inA
def fut20_vec(x):
    xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * g1; t20 = (xn @ WU.T).topk(20, -1).indices
    return torch.tensor([float(torch.isin(n, t20[i]).float().mean()) for i, n in enumerate(nexts)])
FL = {l: fut20_vec(held["layers"][l]) for l in GATE_L}
D6 = {l: torch.load(os.path.join(JD, f"J_l_{l}.pt")).float() - torch.eye(5120) for l in GATE_L}
BIDX = torch.randint(0, NPOS, (B, NPOS), generator=torch.Generator().manual_seed(20261107))
def run(Dmap):
    per_l = []
    for l in GATE_L:
        h = held["layers"][l]; hD = h @ Dmap(l).T; M = torch.stack([fut20_vec(h + a * hD) for a in ALPHAS])
        iA, iB = int(torch.argmax(M[:, inA].mean(1))), int(torch.argmax(M[:, inB].mean(1)))
        per_l.append(torch.where(inB, M[iA], M[iB]) - FL[l])
    g = torch.stack(per_l).mean(0); bs = g[BIDX].mean(1)
    return float(g.mean()), float(bs.quantile(0.0125)), float(bs.quantile(0.9875))
out = {"noise": [], "perm": []}
for d in range(DRAWS):
    gn = torch.Generator().manual_seed(1000 + d)
    nz = {l: (lambda R: R * (D6[l].norm() / R.norm()))(torch.randn(5120, 5120, generator=gn)) for l in GATE_L}
    m, lo, hi = run(lambda l: nz[l]); out["noise"].append({"seed": 1000 + d, "mean_g": m, "ci": [lo, hi], "pass": lo > 0}); log("noise", d, m, lo, hi)
    gp = torch.Generator().manual_seed(2000 + d)
    pm = {l: D6[l][torch.randperm(5120, generator=gp)] for l in GATE_L}
    m, lo, hi = run(lambda l: pm[l]); out["perm"].append({"seed": 2000 + d, "mean_g": m, "ci": [lo, hi], "pass": lo > 0}); log("perm", d, m, lo, hi)
    del nz, pm
for k in out: out[k + "_pass_rate"] = sum(r["pass"] for r in out[k]) / DRAWS
json.dump(out, open(os.path.join(BASE, "shrink_null_study.json"), "w"), indent=1); log("pass rates", out["noise_pass_rate"], out["perm_pass_rate"])
