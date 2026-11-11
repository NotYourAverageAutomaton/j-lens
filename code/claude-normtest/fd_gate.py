"""FD gate on the RECORD build (Claude 2026-11-04; bars Deron P049 / vault ddc15c8, SUPERSEDES post P048; card C01).
Question: does the fit's OWN backward path (whole_model_fit loader + L42 fp32 pre-hook + our_jlens capture, grad root 16,
retain_grad at L) give the true derivative of h_final w.r.t. h_L? Estimand, per seeded pair (U on h_63 at valid t', V on h_L at
valid t; same valid mask as fit_prompt):
  AD  = < d<U, h_63>/d h_L , V >          (backward through the fit's capture code; grad read at L)
  FD  = ( <U, h_63(h_L + e V)> - <U, h_63(h_L - e V)> ) / 2e   (forward hook adds +-e V to layer L's output)
V is scaled to ||h_L[valid]||, so e is RELATIVE. rel_err = |AD - FD| / |AD|; median over 8 pairs (2 fit prompts x 4).
PRIMARY L40 (crosses the L42 hook), L44, L52: PASS iff median rel_err <= 1e-3 at the better eps AND within 10x at the other
  eps (plateau). eps {1e-2, 1e-3}.
CONTROL (must FAIL, else the gate is VOID): AD read at L-1 vs FD at L, median rel_err >= 1e-2 at every layer.
L24 (bf16 band): <= 1e-2 with plateau, eps {1e-1, 3e-2, 1e-2}; no plateau = UNMEASURABLE (Stage 1 may run, but its gate and
  the J1 re-score aren't read until the bf16 band is checked against an fp32 reference).
Any PRIMARY fail -> Stage 1 does not start. Writes fd_gate.json (with md5 of this script, our_jlens.py, whole_model_fit.py)."""
import hashlib, json, os, sys, time, torch
sys.path.insert(0, "$JLENS_HOME")
import whole_model_fit as wmf
import importlib.util
spec = importlib.util.spec_from_file_location("our_jlens", "$JLENS_HOME/our_jlens.py")
oj = importlib.util.module_from_spec(spec); sys.modules[spec.name] = oj; spec.loader.exec_module(oj)
from transformers import AutoTokenizer
OUT = "$JLENS_HOME/claude-normtest"; MD = "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b"; SMOKE = "--smoke" in sys.argv
MAX_SEQ = 150; ROOT = 16; SKIP = oj.SKIP_FIRST_N_POSITIONS
PRIMARY = {40: (1e-2, 1e-3), 44: (1e-2, 1e-3), 52: (1e-2, 1e-3)}; BF16 = {24: (1e-1, 3e-2, 1e-2)}
PROMPTS = wmf.PROMPTS[:2]; PAIRS_PER_PROMPT = 4
if SMOKE: PRIMARY, BF16, PROMPTS, PAIRS_PER_PROMPT = {52: (1e-2, 1e-3)}, {}, wmf.PROMPTS[:1], 1
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
T0 = time.time(); log = lambda *a: print(f"[{(time.time()-T0)/60:5.1f}m]", *a, flush=True)
dev = torch.device("cuda:0"); model, tc = wmf.load_whole_model(MD, dev, log)
lang = model.model.language_model if hasattr(model.model, "language_model") else model.model
lang.layers[42].register_forward_pre_hook(   # IDENTICAL to whole_model_fit.py main()
    lambda m, a, k: ((a[0].float(),) + a[1:], k) if a else (a, {**k, "hidden_states": k["hidden_states"].float()}), with_kwargs=True)
tok = AutoTokenizer.from_pretrained(MD); lens = oj.OurJacobianLens(model, tok, add_bos=True); TGT = lens.n_layers - 1
gen = torch.Generator(device="cpu").manual_seed(20261104)

def ad_and_h(ids, valid, L, U):
    """Fit's own path: capture sources {L, L-1}, root 16, forward, backward(h_tgt, U). Returns grad at L, grad at L-1, h_L."""
    handles, store = lens._capture_residuals([L, L - 1], TGT, grad_root_layer=ROOT)
    try:
        with torch.enable_grad():
            model(input_ids=ids, use_cache=False)
            torch.autograd.backward(store[TGT], U.to(store[TGT].dtype))
            return store[L].grad.detach().float(), store[L - 1].grad.detach().float(), store[L].detach().float()
    finally:
        for h in handles: h.remove()

def fwd_final(ids, L, delta):
    """Forward with +delta added to layer L's output (no grad). Returns h_63 fp32."""
    cap = {}
    def add(m, a, out):
        if isinstance(out, tuple): return (out[0] + delta.to(out[0].dtype),) + tuple(out[1:])
        return out + delta.to(out.dtype)
    def grab(m, a, out): cap["h"] = (out[0] if isinstance(out, tuple) else out).detach().float()
    hs = [lens.layers[L].register_forward_hook(add), lens.layers[TGT].register_forward_hook(grab)]
    try:
        with torch.no_grad(): model(input_ids=ids, use_cache=False)
    finally:
        for h in hs: h.remove()
    return cap["h"]

res = {"bars": "Deron P049 / ddc15c8", "md5": {"fd_gate.py": md5(__file__), "our_jlens.py": md5("$JLENS_HOME/our_jlens.py"),
       "whole_model_fit.py": md5("$JLENS_HOME/whole_model_fit.py")}, "layers": {}}
for L, epss in {**PRIMARY, **BF16}.items():
    rows = []
    for pr in PROMPTS:
        ids = lens._encode(pr, MAX_SEQ); seq = ids.shape[1]
        valid = torch.zeros(seq, dtype=torch.bool); valid[SKIP:seq - 1] = True
        for _ in range(PAIRS_PER_PROMPT):
            d = lens.d_model
            U = torch.zeros(1, seq, d); U[0, valid] = torch.randn(int(valid.sum()), d, generator=gen)
            V = torch.zeros(1, seq, d); V[0, valid] = torch.randn(int(valid.sum()), d, generator=gen)
            gL, gL1, hL = ad_and_h(ids, valid, L, U.to(dev))
            V = (V / V.norm() * hL[0, valid].norm().cpu()).to(dev)          # relative scale: ||V|| = ||h_L[valid]||
            Ud = U.to(dev)
            ad = float((gL * V).sum()); ad_ctrl = float((gL1 * V).sum())
            fd = {}
            for e in epss:
                hp = fwd_final(ids, L, e * V); hm = fwd_final(ids, L, -e * V)
                fd[e] = float((Ud * (hp - hm)).sum()) / (2 * e)
            rows.append({"ad": ad, "ad_ctrl_Lm1": ad_ctrl, "fd": fd})
    med = lambda xs: float(torch.tensor(xs).median())
    rel = {e: med([abs(r["ad"] - r["fd"][e]) / max(abs(r["ad"]), 1e-30) for r in rows]) for e in epss}
    ctrl = {e: med([abs(r["ad_ctrl_Lm1"] - r["fd"][e]) / max(abs(r["ad_ctrl_Lm1"]), 1e-30) for r in rows]) for e in epss}
    best = min(rel, key=rel.get); others = [rel[e] for e in epss if e != best]
    plateau = all(o <= 10 * max(rel[best], 1e-12) for o in others)
    bar = 1e-3 if L in PRIMARY else 1e-2
    verdict = "PASS" if (rel[best] <= bar and plateau) else ("UNMEASURABLE" if (L in BF16 and not plateau) else "FAIL")
    ctrl_fails = min(ctrl.values()) >= 1e-2
    res["layers"][L] = {"rel_err_median": rel, "best_eps": best, "plateau": plateau, "bar": bar, "verdict": verdict,
                        "control_rel_err_median": ctrl, "control_fails_as_required": ctrl_fails, "pairs": rows}
    log(f"L{L}: rel {({k: f'{v:.2e}' for k, v in rel.items()})} best {best} plateau {plateau} -> {verdict} | control {({k: f'{v:.2e}' for k, v in ctrl.items()})} fails={ctrl_fails}")
prim = [res["layers"][L] for L in PRIMARY]
if not all(x["control_fails_as_required"] for x in res["layers"].values()): res["gate"] = "VOID (a control did not fail)"
elif all(x["verdict"] == "PASS" for x in prim): res["gate"] = "PASS (primary)" + ("" if not BF16 else f"; L24 {res['layers'][24]['verdict']}")
else: res["gate"] = "FAIL (a primary layer failed): Stage 1 does not start; J1 = INSTRUMENT-INVALID per post P048"
res["minutes"] = round((time.time() - T0) / 60, 1); res["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
json.dump(res, open(os.path.join(OUT, "fd_gate" + ("_smoke" if SMOKE else "") + ".json"), "w"), indent=1); log("GATE:", res["gate"])
