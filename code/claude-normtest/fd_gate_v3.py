"""FD RE-GATE v3 on the RECORD build (Claude 2026-11-04; bars Deron P052 / vault ca3827d; card C01). The v1 FAIL stays a fail.
Estimand per seeded pair (as v1): AD = <d<U,h_63>/dh_L, V> via the fit's own capture/backward (root 16, retain_grad at L);
FD = (<U,h_63(h_L+eV)> - <U,h_63(h_L-eV)>)/2e via a forward hook on layer L's output. V scaled to ||h_L[valid]||; 8 pairs.
(a) L44, L52 on the record build: eps {1e-1, 3e-2, 1e-2}. PASS iff the flattest adjacent pair has BOTH median rel errs <= 1e-3
    AND their ratio <= 3. Control (AD at L-1 vs FD at L) >= 1e-2. The error-vs-eps slope is logged.
(b) L40 (PRIMARY) then L24: record AD vs an fp32-REFERENCE AD. Reference = same session, layers L..41 upcast to fp32 plus a
    pre-hook casting layer L's input to fp32, so fp32 runs from the injection to h_63.
    Step 1: the reference must pass its own FD under (a)'s rule at L. Else (b) is VOID at L.
    Step 2: rel(record AD_L, ref AD_L) <= 5e-2 at L40 (<= 1e-1 at L24). Control: rel(record AD_{L-1}, ref AD_L) >= .2 AND
            control/err >= 4.
    L24's reference upcasts 18 layers; if it doesn't fit (OOM), L24 = UNMEASURABLE.
Stage 1 starts only if (a) passes at L44+L52 AND (b) passes at L40. Writes fd_gate_v3.json with md5s + peak GiB."""
import hashlib, json, os, sys, time, torch
sys.path.insert(0, "$JLENS_HOME")
import whole_model_fit as wmf
import importlib.util
spec = importlib.util.spec_from_file_location("our_jlens", "$JLENS_HOME/our_jlens.py")
oj = importlib.util.module_from_spec(spec); sys.modules[spec.name] = oj; spec.loader.exec_module(oj)
from transformers import AutoTokenizer
OUT = "$JLENS_HOME/claude-normtest"; MD = "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b"
MAX_SEQ = 150; ROOT = 16; SKIP = oj.SKIP_FIRST_N_POSITIONS; EPS = (1e-1, 3e-2, 1e-2)
PROMPTS = wmf.PROMPTS[:2]; PAIRS = 4
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
T0 = time.time(); log = lambda *a: print(f"[{(time.time()-T0)/60:5.1f}m]", *a, flush=True)
dev = torch.device("cuda:0"); model, tc = wmf.load_whole_model(MD, dev, log)
lang = model.model.language_model if hasattr(model.model, "language_model") else model.model
cast = lambda m, a, k: ((a[0].float(),) + a[1:], k) if a else (a, {**k, "hidden_states": k["hidden_states"].float()})
lang.layers[42].register_forward_pre_hook(cast, with_kwargs=True)        # IDENTICAL to whole_model_fit.py main()
tok = AutoTokenizer.from_pretrained(MD); lens = oj.OurJacobianLens(model, tok, add_bos=True); TGT = lens.n_layers - 1
res = {"bars": "Deron P052 / ca3827d", "md5": {"fd_gate_v3.py": md5(__file__), "our_jlens.py": md5("$JLENS_HOME/our_jlens.py"),
       "whole_model_fit.py": md5("$JLENS_HOME/whole_model_fit.py")}, "a": {}, "b": {}}
# one fixed set of (prompt, U, V-direction) per layer, reused across record and reference builds
def make_pairs(L):
    g = torch.Generator().manual_seed(20261104 + L); out = []
    for pr in PROMPTS:
        ids = lens._encode(pr, MAX_SEQ); seq = ids.shape[1]; valid = torch.zeros(seq, dtype=torch.bool); valid[SKIP:seq - 1] = True
        for _ in range(PAIRS):
            U = torch.zeros(1, seq, lens.d_model); U[0, valid] = torch.randn(int(valid.sum()), lens.d_model, generator=g)
            V = torch.zeros(1, seq, lens.d_model); V[0, valid] = torch.randn(int(valid.sum()), lens.d_model, generator=g)
            out.append((ids, valid, U, V))
    return out
def ad(ids, L, U):
    handles, store = lens._capture_residuals([L, L - 1], TGT, grad_root_layer=ROOT)
    try:
        with torch.enable_grad():
            model(input_ids=ids, use_cache=False); torch.autograd.backward(store[TGT], U.to(dev, store[TGT].dtype))
            return store[L].grad.detach().float(), store[L - 1].grad.detach().float(), store[L].detach().float()
    finally:
        for h in handles: h.remove()
def fwd(ids, L, delta):
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
med = lambda xs: float(torch.tensor(xs).median())
def measure(L, pairs, with_fd=True):
    rows = []
    for ids, valid, U, V0 in pairs:
        gL, gL1, hL = ad(ids, L, U)
        V = (V0 / V0.norm() * hL[0, valid].norm().cpu()).to(dev)
        r = {"ad": float((gL * V).sum()), "ad_Lm1": float((gL1 * V).sum()), "fd": {}}
        if with_fd:
            for e in EPS: r["fd"][e] = float((U.to(dev) * (fwd(ids, L, e * V) - fwd(ids, L, -e * V))).sum()) / (2 * e)
        rows.append(r)
    return rows
def rule_a(rows, key="ad"):
    rel = {e: med([abs(r[key] - r["fd"][e]) / max(abs(r[key]), 1e-30) for r in rows]) for e in EPS}
    pairs = [(EPS[i], EPS[i + 1]) for i in range(len(EPS) - 1)]
    flat = min(pairs, key=lambda p: max(rel[p[0]], rel[p[1]]) / max(min(rel[p[0]], rel[p[1]]), 1e-30))
    a, b = rel[flat[0]], rel[flat[1]]; ratio = max(a, b) / max(min(a, b), 1e-30)
    import math; slope = (math.log10(rel[EPS[-1]]) - math.log10(rel[EPS[0]])) / (math.log10(EPS[-1]) - math.log10(EPS[0]))
    return {"rel": rel, "flattest_pair": flat, "ratio": ratio, "slope_log_err_vs_log_eps": slope, "pass": a <= 1e-3 and b <= 1e-3 and ratio <= 3}
# (a) L44, L52 on the record build
for L in (44, 52):
    rows = measure(L, make_pairs(L)); ra = rule_a(rows)
    ctrl = {e: med([abs(r["ad_Lm1"] - r["fd"][e]) / max(abs(r["ad_Lm1"]), 1e-30) for r in rows]) for e in EPS}
    ra["control"] = ctrl; ra["control_ok"] = min(ctrl.values()) >= 1e-2; ra["pass"] = ra["pass"] and ra["control_ok"]
    res["a"][L] = ra; log(f"(a) L{L}: rel {({k: f'{v:.2e}' for k, v in ra['rel'].items()})} flat {ra['flattest_pair']} ratio {ra['ratio']:.2f} slope {ra['slope_log_err_vs_log_eps']:.2f} ctrl_ok {ra['control_ok']} -> {'PASS' if ra['pass'] else 'FAIL'}")
# (b) record AD first (no FD needed: bf16 injection is unmeasurable), for L40 and L24
pairs = {L: make_pairs(L) for L in (40, 24)}
rec = {L: measure(L, pairs[L], with_fd=False) for L in (40, 24)}
log("(b) record AD captured for L40, L24")
upcast_done = set()
def upcast(lo, hi):
    for i in range(lo, hi + 1):
        if i in upcast_done: continue
        for p in lang.layers[i].parameters(): p.data = p.data.float()
        upcast_done.add(i)
    lang.layers[lo].register_forward_pre_hook(cast, with_kwargs=True)
for L, bar in ((40, 5e-2), (24, 1e-1)):
    try:
        upcast(L, 41); torch.cuda.empty_cache()
        ref = measure(L, pairs[L]); s1 = rule_a(ref)
        err = med([abs(a["ad"] - b["ad"]) / max(abs(b["ad"]), 1e-30) for a, b in zip(rec[L], ref)])
        ctl = med([abs(a["ad_Lm1"] - b["ad"]) / max(abs(b["ad"]), 1e-30) for a, b in zip(rec[L], ref)])
        s2 = {"err_record_vs_ref": err, "bar": bar, "control": ctl, "control_over_err": ctl / max(err, 1e-30)}
        ok1 = s1["pass"]; ok2 = err <= bar and ctl >= .2 and ctl / max(err, 1e-30) >= 4
        verdict = "VOID (reference failed its own FD)" if not ok1 else ("PASS" if ok2 else "FAIL")
        res["b"][L] = {"step1_reference_fd": s1, "step2": s2, "verdict": verdict, "record": rec[L], "reference": ref}
        log(f"(b) L{L}: ref FD {({k: f'{v:.2e}' for k, v in s1['rel'].items()})} pass={ok1} | record-vs-ref {err:.3e} (bar {bar}) control {ctl:.3e} ratio {ctl/max(err,1e-30):.1f} -> {verdict}")
    except torch.cuda.OutOfMemoryError:
        res["b"][L] = {"verdict": "UNMEASURABLE (reference upcast does not fit)"}; log(f"(b) L{L}: OOM -> UNMEASURABLE"); torch.cuda.empty_cache()
res["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
okA = all(res["a"][L]["pass"] for L in (44, 52)); okB = res["b"].get(40, {}).get("verdict") == "PASS"
res["gate"] = ("PASS: Stage 1 may start" if okA and okB else "FAIL: Stage 1 does not start") + f" | L24 {res['b'].get(24, {}).get('verdict')}"
res["minutes"] = round((time.time() - T0) / 60, 1); res["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
json.dump(res, open(os.path.join(OUT, "fd_gate_v3.json"), "w"), indent=1, default=str); log("GATE v3:", res["gate"], "| peak", res["peak_gib"], "GiB")
