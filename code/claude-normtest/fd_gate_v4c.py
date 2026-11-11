"""FD GATE v4c = v4 part (c) ONLY, re-run after v4 crashed in segment [28->36] (Claude bug: layers 21..28 upcast to fp32 but layer 21 got
bf16 input from layer 20; fixed by a persistent fp32 pre-hook at layer 21). Same rule, same seeds, same pairs as v4. v4 (a)(b) + [20->28] results
stand from fd_gate_v4_run.log. This file was v4 (Claude 2026-11-04; rule signed BEFORE the run: Deron P053 / vault ac83046 + post P054 edge case, SUPERSEDES post P052; card C01).
ONE session, the same 8 (U,V) per test at every eps; seeds and ||V|| logged; per-pair AD/FD logged (explains v1 vs v3: different
pair draws -- v1 one shared generator across layers, v3 seeded per layer -- and rel err = |AD-FD|/|AD| amplifies small-|AD| pairs).
RULE at every tested point: eps {1e-1, 3e-2, 1e-2, 5e-3, 3e-3, 1e-3}. PASS iff (i) min over eps of the median rel err <= bar;
(ii) the large-eps-side log-log slope (fit over >= 2 steps, from eps=1e-1 down to the argmin) is in [1.5, 2.5]; (iii) control holds.
(a) record build, L44 + L52: bar 1e-3; control = AD at L-1 vs FD at L, min-over-eps median >= 1e-2.
(b) L40: fp32 reference (layers 40-41 upcast) validated by the rule (bar 1e-3); record AD vs ref AD <= 5e-2;
    control = record AD_39 vs ref AD_40 >= 5e-2 AND control/err >= 4.
(c) bf16 band by SEGMENTS [20->28], [28->36], [36->40]: random U at the segment END e; AD = <d<U,h_e>/dh_s, V>.
    record AD (record path: root at s, bf16) vs a reference where layers s+1..e run fp32 from an fp32 root (validated by the rule,
    bar 1e-3, FD injected in fp32 at the root); <= 5e-2; start-off-by-one control (record AD wrt h_{s-1}) ratio >= 4.
Stage 1 STARTS iff (a) + (b) pass. Its gate and the J1 re-score are READ only if all 3 segments pass.
MEMORY: never above v3's 85.1 GiB. Order = (a) -> all RECORD ADs for (b)/(c) on the untouched build -> (b) upcast 40-41 ->
downcast 42-63 to bf16 (unused by (c), frees ~18 GiB) -> upcast 21..39 -> segment references (forward stopped at e).
Abort cleanly (exit 4) if any allocation would pass 85.1 GiB."""
import hashlib, json, math, os, sys, time, torch
sys.path.insert(0, "$JLENS_HOME")
import whole_model_fit as wmf
import importlib.util
spec = importlib.util.spec_from_file_location("our_jlens", "$JLENS_HOME/our_jlens.py")
oj = importlib.util.module_from_spec(spec); sys.modules[spec.name] = oj; spec.loader.exec_module(oj)
from transformers import AutoTokenizer
OUT = "$JLENS_HOME/claude-normtest"; MD = "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b"
MAX_SEQ = 150; ROOT = 16; SKIP = oj.SKIP_FIRST_N_POSITIONS; EPS = (1e-1, 3e-2, 1e-2, 5e-3, 3e-3, 1e-3); PEAK_CAP = 85.1
PROMPTS = wmf.PROMPTS[:2]; PAIRS = 4; SEGS = ((28, 36), (36, 40))
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
T0 = time.time(); log = lambda *a: print(f"[{(time.time()-T0)/60:5.1f}m]", *a, flush=True)
dev = torch.device("cuda:0"); model, tc = wmf.load_whole_model(MD, dev, log)
lang = model.model.language_model if hasattr(model.model, "language_model") else model.model
cast = lambda m, a, k: ((a[0].float(),) + a[1:], k) if a else (a, {**k, "hidden_states": k["hidden_states"].float()})
lang.layers[42].register_forward_pre_hook(cast, with_kwargs=True)        # IDENTICAL to whole_model_fit.py main()
tok = AutoTokenizer.from_pretrained(MD); lens = oj.OurJacobianLens(model, tok, add_bos=True); TGT = lens.n_layers - 1; D = lens.d_model
res = {"rule": "Deron P053 / ac83046", "eps": EPS, "md5": {"fd_gate_v4c.py": md5(__file__), "our_jlens.py": md5("$JLENS_HOME/our_jlens.py"),
       "whole_model_fit.py": md5("$JLENS_HOME/whole_model_fit.py")}, "a": {}, "b": {}, "c": {}}
def peak_ok():
    p = torch.cuda.max_memory_allocated() / 2**30
    if p > PEAK_CAP: res["abort"] = f"peak {p:.1f} GiB > {PEAK_CAP}"; json.dump(res, open(os.path.join(OUT, "fd_gate_v4c.json"), "w"), indent=1, default=str); log("ABORT", res["abort"]); sys.exit(4)
def make_pairs(tag, n_seed):
    g = torch.Generator().manual_seed(n_seed); out = []
    for pr in PROMPTS:
        ids = lens._encode(pr, MAX_SEQ); seq = ids.shape[1]; valid = torch.zeros(seq, dtype=torch.bool); valid[SKIP:seq - 1] = True
        for _ in range(PAIRS):
            U = torch.zeros(1, seq, D); U[0, valid] = torch.randn(int(valid.sum()), D, generator=g)
            V = torch.zeros(1, seq, D); V[0, valid] = torch.randn(int(valid.sum()), D, generator=g)
            out.append((ids, valid, U, V))
    res.setdefault("seeds", {})[tag] = n_seed; return out
class Stop(Exception): pass
def run(ids, L_src, L_end, U, mode, delta=None, stop=False):
    """mode 'record': root at L_src via the fit's own _capture_residuals (bf16 root below 42), target L_end, also h_{L_src-1}.
       mode 'ref': fp32 root at L_src (output cast to fp32, requires_grad), target L_end; delta (fp32) added at the root.
       Returns (grad at L_src, grad at L_src-1 or None, h_src fp32, <U,h_end>)."""
    if mode == "record":
        handles, store = lens._capture_residuals([L_src, L_src - 1] if L_src - 1 > ROOT else [L_src], L_end,
                                                 grad_root_layer=ROOT if L_src - 1 > ROOT else L_src)
        try:
            with torch.enable_grad():
                try: model(input_ids=ids, use_cache=False)
                except Stop: pass
                h_end = store[L_end]; torch.autograd.backward(h_end, U.to(dev, h_end.dtype))
                g1 = store[L_src - 1].grad.detach().float() if (L_src - 1) in store and store[L_src - 1].grad is not None else None
                return store[L_src].grad.detach().float(), g1, store[L_src].detach().float(), float((U.to(dev) * h_end.float()).sum())
        finally:
            for h in handles: h.remove()
    st = {}
    def root(m, a, out):
        h = (out[0] if isinstance(out, tuple) else out).float().detach()
        if delta is not None: h = h + delta
        h.requires_grad_(delta is None); st["root"] = h
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    def end(m, a, out):
        st["end"] = out[0] if isinstance(out, tuple) else out
        if stop: raise Stop
    hs = [lens.layers[L_src].register_forward_hook(root), lens.layers[L_end].register_forward_hook(end)]
    try:
        with torch.enable_grad() if delta is None else torch.no_grad():
            try: model(input_ids=ids, use_cache=False)
            except Stop: pass
            val = float((U.to(dev) * st["end"].float()).sum())
            if delta is None:
                torch.autograd.backward(st["end"], U.to(dev, st["end"].dtype)); return st["root"].grad.detach().float(), None, st["root"].detach().float(), val
            return None, None, None, val
    finally:
        for h in hs: h.remove()
def fd_record(ids, L, V, U, e):   # record-build FD (used in (a), fp32 layers): forward hook adds +-eV at L's output
    vals = []
    for s in (1, -1):
        cap = {}
        def add(m, a, out):
            return (out[0] + (s * e * V).to(out[0].dtype),) + tuple(out[1:]) if isinstance(out, tuple) else out + (s * e * V).to(out.dtype)
        def grab(m, a, out): cap["h"] = (out[0] if isinstance(out, tuple) else out).detach().float()
        hs = [lens.layers[L].register_forward_hook(add), lens.layers[TGT].register_forward_hook(grab)]
        try:
            with torch.no_grad(): model(input_ids=ids, use_cache=False)
        finally:
            for h in hs: h.remove()
        vals.append(float((U.to(dev) * cap["h"]).sum()))
    return (vals[0] - vals[1]) / (2 * e)
med = lambda xs: float(torch.tensor(xs).median())
def rule(ad_list, fd_by_eps, bar):
    rel = {e: med([abs(a - f) / max(abs(a), 1e-30) for a, f in zip(ad_list, fd_by_eps[e])]) for e in EPS}
    emin = min(rel, key=rel.get); i_min = EPS.index(emin)
    xs = [math.log10(e) for e in EPS[:i_min + 1]]; ys = [math.log10(max(rel[e], 1e-30)) for e in EPS[:i_min + 1]]
    if len(xs) >= 3:
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys); slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    else: slope = None   # fewer than 2 steps on the large-eps side: (ii) can't be shown
    ok = rel[emin] <= bar and slope is not None and 1.5 <= slope <= 2.5
    # Deron P054 (read before the gate ran): argmin at 1e-1 or 3e-2 -> slope not computable = UNMEASURABLE-SLOPE, not FAIL/PASS
    status = "UNMEASURABLE-SLOPE" if slope is None else ("PASS" if ok else "FAIL")
    # REPORT-ONLY (post P054 item 1): error normalized by the mean |AD| over the pairs (less sensitive to small-|AD| pairs)
    mabs = sum(abs(a) for a in ad_list) / max(len(ad_list), 1)
    reln = {e: med([abs(a - f) / max(mabs, 1e-30) for a, f in zip(ad_list, fd_by_eps[e])]) for e in EPS}
    return {"rel": rel, "min": rel[emin], "argmin_eps": emin, "slope_large_eps_side": slope, "bar": bar, "pass_i_ii": ok, "status_i_ii": status, "rel_norm_mean_abs_ad_REPORT_ONLY": reln}
def scaled(V0, h, valid): return (V0 / V0.norm() * h[0, valid].norm().cpu()).to(dev)
# ---------------- record ADs for (b) and (c), untouched build
Pc = {}; recc = {}
for (s, e_) in SEGS:
    Pc[(s, e_)] = make_pairs(f"c_{s}_{e_}", 20261104 + 1000 + s); recc[(s, e_)] = []
    for ids, valid, U, V0 in Pc[(s, e_)]:
        Ue = torch.zeros_like(U); Ue[0, valid] = U[0, valid]
        g, g1, h, _ = run(ids, s, e_, Ue, "record"); V = scaled(V0, h, valid)
        recc[(s, e_)].append({"V": V.cpu(), "h": h.cpu(), "ad": float((g * V).sum()), "ad_Sm1": float((g1 * V).sum()) if g1 is not None else None})
    peak_ok()
log("record ADs captured for (b) and (c)")
def upcast(lo, hi):
    for i in range(lo, hi + 1):
        for p in lang.layers[i].parameters(): p.data = p.data.float()
def downcast(lo, hi):
    for i in range(lo, hi + 1):
        for p in lang.layers[i].parameters(): p.data = p.data.to(torch.bfloat16)
# ---------------- (c) segments: free the upper stack (unused below 40), then upcast 21..39
downcast(42, TGT); upcast(21, 40); lang.layers[21].register_forward_pre_hook(cast, with_kwargs=True)   # THE FIX: fp32 input from layer 21 on
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
for (s, e_) in SEGS:
    ads, fds, hdiff = [], {e: [] for e in EPS}, []
    for (ids, valid, U, V0), rc in zip(Pc[(s, e_)], recc[(s, e_)]):
        Ue = torch.zeros_like(U); Ue[0, valid] = U[0, valid]; V = rc["V"].to(dev)
        g, _, hr, _ = run(ids, s, e_, Ue, "ref", stop=True); ads.append(float((g * V).sum()))
        hdiff.append(float((hr.cpu() - rc["h"]).norm() / rc["h"].norm()))   # Deron P056 note 1: linearization-point shift, REPORT-ONLY
        for e in EPS:
            vp = run(ids, s, e_, Ue, "ref", delta=e * V, stop=True)[3]; vm = run(ids, s, e_, Ue, "ref", delta=-e * V, stop=True)[3]
            fds[e].append((vp - vm) / (2 * e))
        peak_ok()
    r = rule(ads, fds, 1e-3)
    err = med([abs(rc["ad"] - a) / max(abs(a), 1e-30) for rc, a in zip(recc[(s, e_)], ads)])
    cs = [abs(rc["ad_Sm1"] - a) / max(abs(a), 1e-30) for rc, a in zip(recc[(s, e_)], ads) if rc["ad_Sm1"] is not None]
    ctl = med(cs) if cs else None
    r["h_s_ref_vs_record_rel_REPORT_ONLY"] = {"median": med(hdiff), "max": max(hdiff)}
    r.update({"record_vs_ref": err, "control": ctl, "control_over_err": (ctl / max(err, 1e-30)) if ctl is not None else None})
    r["pass"] = r["pass_i_ii"] and err <= 5e-2 and ctl is not None and ctl / max(err, 1e-30) >= 4; res["c"][f"{s}->{e_}"] = r
    json.dump(res, open(os.path.join(OUT, "fd_gate_v4c.json"), "w"), indent=1, default=str)
    log(f"(c) [{s}->{e_}]: ref min {r['min']:.2e}@{r['argmin_eps']} slope {r['slope_large_eps_side']} | rec-vs-ref {err:.2e} ctl {ctl} -> {'PASS' if r['pass'] else 'FAIL'}")
res["peak_gib_after_reset"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
res["segments_pass"] = all(v["pass"] for v in res["c"].values())
res["minutes"] = round((time.time() - T0) / 60, 1); res["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
json.dump(res, open(os.path.join(OUT, "fd_gate_v4c.json"), "w"), indent=1, default=str)
log("GATE v4c: remaining segments pass =", res["segments_pass"])
