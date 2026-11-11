"""Low-rank gate, CPU only (Claude 2026-11-04; design Deron P042 / vault bfd780a, card C01). No GPU, no window.
Question: if the re-fit only measures r chosen directions U (in FINAL-residual space) instead of all 5120, does the lens still
give the exact map's readout? Projected map = U U^T J_l (exactly what an r-cotangent fit would give, minus data noise).
Scored like the pilot: on the 306 held-out positions (held_h.pt), top-1 agreement and top-10 overlap vs the UNPROJECTED exact
J_l readout, at L24/L40/L52. Floor (Deron 1a70281): top-1 >= .90 AND top-10 >= .95 at all three. r* = smallest r that clears.
U is NEVER taken from the 6-doc J (circular, post P042). Two sources:
  (a) "unembed": top-r right singular vectors of W_U diag(1+g) (the post-norm readout map)
  (b) "actpca":  top-r eigenvectors of the uncentered second moment of h_final on 50 non-fit documents (nonfit_hfinal.pt)
Plus Deron's stop-and-look: exact vs exact with tiny relative noise on J (1e-3, 1e-2) -> is the scorer near a tie ceiling?
Readout (Qwen3_5RMSNorm, confirmed in transformers source): x/rms(x) * (1 + w) then lm_head. Usage: python gate_lowrank.py"""
import json, os, time, torch
from safetensors import safe_open
torch.set_num_threads(max(1, os.cpu_count() - 2)); torch.manual_seed(20261104)
OUT = "$JLENS_HOME/claude-normtest"; JD = "$JLENS_HOME/results-whole-swift-record"
MD = "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b"; LAYERS = [24, 40, 52]; RS = [256, 512, 1024, 2048]; EPS = 1e-6
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)
idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
def W(name):
    with safe_open(os.path.join(MD, idx[name]), "pt") as f: return f.get_tensor(name).float()
WU = W("lm_head.weight"); g1 = 1.0 + W("model.language_model.norm.weight")
log("lm_head", tuple(WU.shape), "norm", tuple(g1.shape))
held = torch.load(os.path.join(OUT, "held_h.pt")); nonfit = torch.load(os.path.join(OUT, "nonfit_hfinal.pt"))["h_final"]
def logits(x):
    xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * g1
    return xn @ WU.T
def agree(lg, ref1, ref10):
    t1 = float((lg.argmax(-1) == ref1).float().mean())
    t10 = lg.topk(10, -1).indices
    ov = float(torch.stack([torch.isin(t10[i], ref10[i]).float().mean() for i in range(t10.shape[0])]).mean())
    return round(t1, 4), round(ov, 4)
# U sources (built once, largest r; smaller r = leading columns)
M = (WU.T @ WU) * torch.outer(g1, g1)                     # = (W_U diag(1+g))^T (W_U diag(1+g)), no 5 GB copy
ev, vec = torch.linalg.eigh(M); U_unembed = vec[:, ev.argsort(descending=True)]
C = nonfit.T @ nonfit; ev2, vec2 = torch.linalg.eigh(C); U_actpca = vec2[:, ev2.argsort(descending=True)]
log("U built: unembed + actpca;", "nonfit positions", nonfit.shape[0])
res = {"design": "Deron P042 / bfd780a", "floor": {"top1": .90, "top10": .95}, "n_held": int(held["h_final"].shape[0]),
       "rs": RS, "layers": LAYERS, "results": {}, "ceiling": {}}
for l in LAYERS:
    J = torch.load(os.path.join(JD, f"J_l_{l}.pt")).float(); h = held["layers"][l]
    x = h @ J.T; lg = logits(x); ref1 = lg.argmax(-1); ref10 = lg.topk(10, -1).indices
    # Deron P044 cond.2 SELF-CHECK: CPU readout must reproduce the GPU (pilot code path) ids at every position, else stop.
    gr = held["gpu_ref"][l]; m1 = float((ref1 == gr["top1"]).float().mean()); m10 = float((ref10 == gr["top10"]).all(-1).float().mean())
    res.setdefault("selfcheck", {})[f"L{l}"] = {"top1_match": m1, "top10_exact_match": m10}
    if m1 < 1.0: log(f"SELF-CHECK FAIL L{l}: top1 match {m1:.4f} (top10 {m10:.4f}). STOP AND LOOK."); res["verdict"] = "SELF-CHECK FAILED"; json.dump(res, open(os.path.join(OUT, "gate_lowrank.json"), "w"), indent=1); raise SystemExit(2)
    # Deron P044 cond.3: top-1 vs top-2 logit gap; ties within fp16 resolution (~1e-3 relative at logit scale)
    top2 = lg.topk(2, -1).values; gap = top2[:, 0] - top2[:, 1]; tol = top2[:, 0].abs() * 2 ** -10
    res["ceiling"][f"L{l}_gap"] = {"median": round(float(gap.median()), 4), "min": round(float(gap.min()), 5), "n_ties_fp16": int((gap <= tol).sum()), "n": int(gap.numel())}
    for eps in (1e-3, 1e-2):     # scorer stop-and-look: how stable is the exact readout itself?
        Jn = J * (1 + eps * torch.randn_like(J)); res["ceiling"][f"L{l}_eps{eps}"] = agree(logits(h @ Jn.T), ref1, ref10)
    for name, U in (("unembed", U_unembed), ("actpca", U_actpca)):
        for r in RS:
            Ur = U[:, :r]; xr = (x @ Ur) @ Ur.T
            res["results"][f"{name}_L{l}_r{r}"] = agree(logits(xr), ref1, ref10)
    log(f"L{l} done")
for name in ("unembed", "actpca"):
    rstar = None
    for r in RS:
        if all(res["results"][f"{name}_L{l}_r{r}"][0] >= .90 and res["results"][f"{name}_L{l}_r{r}"][1] >= .95 for l in LAYERS): rstar = r; break
    res[f"rstar_{name}"] = rstar
res["verdict"] = ("r* <= 2048 -> PROCEED (Deron adjudicates)" if any(res[f"rstar_{n}"] for n in ("unembed", "actpca"))
                  else "r* > 2048 for both U -> path 4 (exact-at-convergence) or STOP, per post P042")
json.dump(res, open(os.path.join(OUT, "gate_lowrank.json"), "w"), indent=1)
print("\nU        layer  " + "  ".join(f"r={r:<5}" for r in RS))
for name in ("unembed", "actpca"):
    for l in LAYERS: print(f"{name:8s} L{l:<4} " + "  ".join(f"{res['results'][f'{name}_L{l}_r{r}'][0]:.2f}/{res['results'][f'{name}_L{l}_r{r}'][1]:.2f}" for r in RS))
print("ceiling (top1/top10, exact vs exact+noise):", res["ceiling"]); print("r*:", res["rstar_unembed"], res["rstar_actpca"], "|", res["verdict"])
