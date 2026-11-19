"""J-Lens STAGE 2 scorer, CPU only, file-only (Claude 2026-11-11; bar Deron P137, signed before any J54 data; card C01).
BAR (post P137, verbatim in substance):
  Data: the fresh 1,434 positions (step2/fresh_h.pt, the 6 fresh passages; only J22 has been seen on them).
  D(q) = mean over L in {20,24,...,44} of [KL(P_final || P_J22)(q,L) - KL(P_final || P_J54)(q,L)]   (positive = J54 better).
  PASS iff mean(D) > 0 AND the paired bootstrap 95% CI lower bound (B 10000, seed 20261110, over positions) > 0
       AND mean(D) > 0.0782 (raw doc-sampling SD at n=21, step 0). Anything else is FAIL. No re-runs, no metric swaps.
  Registered prediction (report-only): gain in [0.25, 1.30], point 0.46 (1/n); J54 - logit stays > 0; shrunk J54 report-only.
  J54 builds ONLY by the same n_valid-weighted pool as J22 (no optimizer): J54 = (N_rec*J6 + S1 + S2) / (N_rec + n1 + n2),
  where S1/n1 = stage1 acc (16 docs) and S2/n2 = stage2 acc (32 docs).
ORDER (every step before the gate can only STOP, verdict NO-SCORE):
  0 memory guard; pins; md5 of every file read.
  1 stage 2 complete: progress.json done == 0..31 == every acc_<l>.pt's doc list; no commit.json left (a doc mid-commit).
    Unless --controls-only, an incomplete run is NO-SCORE (the bar is J54, not J_k).
  2 disjointness: no stage2 doc (and no stage1 doc) shares a normalised hash or a 12-word shingle with the 6 fresh passages.
  3 resume: stage1 acc == sum n_p*J_p (as dose_kl b278063c); stage2 acc tot == sum meta n_valid of its docs and
    sum == sum n_p*J_p (rel <= 1e-5), each J_p md5 == its meta.
  4 controls (STOP): (c1) logit and J22 fresh mean KL per layer reproduce step2/dose_kl.night.json table.fresh (the graded 43dcd4a9; dose_kl.json was rewritten by the day p@k run, fresh table identical), |diff| <= 1e-4;
    (c2) oracle h_final gives KL <= 1e-4 everywhere; (c3) J54 must DIFFER from J22 (max abs > 0: stage 2 entered the pool);
    (c4) pooled-state route (h @ J^T by linearity) equals the direct J54 map route, |mean KL diff| <= 1e-4 per layer.
  5 GATE as above. A non-finite KL vector or D is a STOP (NO-SCORE), never a FAIL (Deron P147 fix 1).  6 REPORT-ONLY: cluster-by-passage CI; per-layer KL table (logit, J22, J54, shrunk J54 at the frozen
    step-2 a); J54 - logit mean + CI; shrunk J54 - logit; prediction check.
--selftest: engine-free, data-free: the gate function on planted vectors must FAIL on D = 0, on a mean under 0.0782 with a
  tight CI, on a positive mean whose CI crosses 0, and must PASS on a planted 0.30 gain. Nothing on disk is read.
--controls-only: steps 0-4 on whatever stage 2 has so far, NO gate and no J54 KL reported (so nobody sees results early).
Usage: SCORE_THREADS=12 nice -n 10 python score_stage2.py [--selftest | --controls-only]   Output: stage2_score.json"""
import hashlib, json, os, re, resource, sys, time, platform, torch
B = 10000; BSEED = 20261110; FLOOR = 0.0782; GATE_L = [20, 24, 28, 32, 36, 40, 44]; EPS = 1e-6; DIM = 5120; N_REC_PARTS = [131, 107, 96, 96, 93, 119]

def gate(D):   # D: per-position gain vector. Returns the decision dict. The ONLY place the bar lives.
    g = torch.Generator().manual_seed(BSEED); n = D.shape[0]
    bs = D[torch.randint(0, n, (B, n), generator=g)].mean(1); lo, hi = float(bs.quantile(.025)), float(bs.quantile(.975)); m = float(D.mean())
    ok = m > 0 and lo > 0 and m > FLOOR
    return {"mean_D": m, "ci95": [lo, hi], "B": B, "seed": BSEED, "floor": FLOOR, "mean_gt_0": m > 0, "ci_lo_gt_0": lo > 0, "mean_gt_floor": m > FLOOR,
            "decision": "PASS" if ok else "FAIL"}

if "--selftest" in sys.argv:
    g = torch.Generator().manual_seed(1); noise = lambda s: torch.randn(1434, generator=g) * s
    cases = {"zero gain (must FAIL)": (torch.zeros(1434), "FAIL"), "mean 0.05 tight CI (under floor, must FAIL)": (0.05 + noise(0.01), "FAIL"),
             "mean 0.10 CI crosses 0 (must FAIL)": (0.10 + noise(2.5), "FAIL"), "planted 0.30 gain (must PASS)": (0.30 + noise(0.2), "PASS")}
    ok = True
    for name, (v, want) in cases.items():
        r = gate(v); good = r["decision"] == want; ok &= good
        print(("  ok  " if good else "  FAIL"), name, "->", r["decision"], f"mean {r['mean_D']:.4f} ci [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]")
    print("SELFTEST OK" if ok else "SELFTEST FAILED"); sys.exit(0 if ok else 1)

from safetensors import safe_open
torch.set_num_threads(int(os.environ.get("SCORE_THREADS", max(1, os.cpu_count() - 4))))
BASE = "$JLENS_HOME/claude-normtest"; JD = "$JLENS_HOME/results-whole-swift-record"; MD = "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b"
S1, S2, ST2 = (os.path.join(BASE, d) for d in ("stage1", "stage2", "step2")); OUTP = os.path.join(BASE, "stage2_score.json")
CONTROLS_ONLY = "--controls-only" in sys.argv; NEED_GIB = 30.0
PIN = {"stage2_fit.py": "19bc36df", "stage2_corpus.json": "84f9f759", "stage1_corpus.json": "19c71211", "step2/fresh_h.pt": "bcb22760",
       "step2/passages.json": "2d302976", "step2/dose_kl.night.json": "43dcd4a9", "score_shrink.json": "0a925a72",
       "$JLENS_HOME/whole_model_fit_record.log": "d0cf6a41"}
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
peak_gib = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
def mem_avail_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"): return int(line.split()[1]) / 2**20
res = {"script_md5": md5(__file__), "bar": "Deron P137", "mode": "controls-only" if CONTROLS_ONLY else "score", "verdict": "INCOMPLETE",
       "stops": [], "pins": {}, "files": {}, "env": {"torch": torch.__version__, "python": platform.python_version(), "threads": torch.get_num_threads(), "host": platform.node()}}
def dump(): json.dump(res, open(OUTP if not CONTROLS_ONLY else OUTP.replace(".json", ".controls.json"), "w"), indent=1)
def stop(why): res["stops"].append(why); res["verdict"] = "NO-SCORE: " + why; dump(); log("STOP:", why); sys.exit(3)
# ---- 0 memory guard, pins ----------------------------------------------------------------------------------------------
ma0 = mem_avail_gib(); res["memory"] = {"mem_available_at_start_gib": round(ma0, 1), "required_gib": NEED_GIB}
if ma0 < NEED_GIB: stop(f"memory guard: MemAvailable {ma0:.1f} GiB < {NEED_GIB} (live engine protected)")
for name, want in PIN.items():
    got = md5(name if name.startswith("/") else os.path.join(BASE, name)); res["pins"][name] = got
    if want and not got.startswith(want): stop(f"pin {name} {got[:8]} != {want}")
log("pins OK")
# ---- 1 stage 2 complete --------------------------------------------------------------------------------------------------
corpus2 = json.load(open(os.path.join(BASE, "stage2_corpus.json"))); NDOC2 = len(corpus2)
prog = json.load(open(os.path.join(S2, "progress.json")))["done"]
if os.path.exists(os.path.join(S2, "commit.json")): stop("stage2/commit.json present: a doc is mid-commit; let stage2_fit.py recover() first")
complete = sorted(prog) == list(range(NDOC2))
res["stage2_progress"] = {"done": sorted(prog), "n_docs": NDOC2, "complete": complete}
if not complete and not CONTROLS_ONLY: stop(f"stage 2 incomplete: {len(prog)} of {NDOC2} docs; the bar is J54 (no partial scoring)")
DONE2 = sorted(prog)
# ---- 2 disjointness vs the 6 FRESH passages --------------------------------------------------------------------------------
norm_t = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
def shingles(s, k=12):
    w = norm_t(s).split(); return {" ".join(w[i:i + k]) for i in range(max(0, len(w) - k + 1))}
fresh_txt = [p["text"] for p in json.load(open(os.path.join(ST2, "passages.json")))]
f_sh = set().union(*[shingles(t) for t in fresh_txt]); f_hash = {hashlib.sha256(norm_t(t).encode()).hexdigest() for t in fresh_txt}
corpus1 = json.load(open(os.path.join(BASE, "stage1_corpus.json")))
ov = lambda corp: [i + 1 for i, d in enumerate(corp) if hashlib.sha256(norm_t(d["text"]).encode()).hexdigest() in f_hash or shingles(d["text"]) & f_sh]
res["disjoint_vs_fresh"] = {"stage2_overlap": ov(corpus2), "stage1_overlap": ov(corpus1), "fresh_passages": len(fresh_txt)}
if res["disjoint_vs_fresh"]["stage2_overlap"] or res["disjoint_vs_fresh"]["stage1_overlap"]: stop(f"training docs overlap the fresh passages: {res['disjoint_vs_fresh']}")
log("disjoint OK")
# ---- 3 resume checks ------------------------------------------------------------------------------------------------------
parts = [int(x) for x in re.findall(r"n_valid=(\d+)", open("$JLENS_HOME/whole_model_fit_record.log").read())]; N_REC = sum(parts)
if parts != N_REC_PARTS or N_REC != 642: stop(f"N_rec parts {parts} != {N_REC_PARTS}")
def docs_meta(d, idx):
    m = [json.load(open(os.path.join(d, f"doc_{i:02d}", "meta.json"))) for i in idx]
    if [x["doc"] for x in m] != list(idx): stop(f"meta doc numbers in {d} != {list(idx)}")
    return m
m1 = docs_meta(S1, range(1, 17)); n1 = [int(x["n_valid"]) for x in m1]
m2 = docs_meta(S2, [i + 1 for i in DONE2]); n2 = [int(x["n_valid"]) for x in m2]
if min(n1 + n2) <= 0: stop("a doc has n_valid <= 0")
res["N"] = {"N_rec": N_REC, "n1": sum(n1), "n2": sum(n2), "docs_in_J54": 6 + 16 + len(DONE2)}
fresh = torch.load(os.path.join(ST2, "fresh_h.pt")); NF = fresh["h_final"].shape[0]
if NF != 1434: stop(f"fresh positions {NF} != 1434")
fpass = torch.tensor(fresh["passage"])
idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
def W(n):
    with safe_open(os.path.join(MD, idx[n]), "pt") as f: return f.get_tensor(n).float()
WU = W("lm_head.weight"); g1 = 1.0 + W("model.language_model.norm.weight")
def logits_of(x): return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * g1) @ WU.T
LPF = torch.log_softmax(logits_of(fresh["h_final"]), -1); PF = LPF.exp()
def klv(x): return (PF * (LPF - torch.log_softmax(logits_of(x), -1))).sum(-1)
def check_acc(d, l, meta, n, label):
    Jl = []
    for mt in meta:
        p = os.path.join(d, f"doc_{mt['doc']:02d}", f"J_p_{l}.pt"); h = md5(p); res["files"][f"{label}/doc_{mt['doc']:02d}/J_p_{l}.pt"] = h
        if h != mt["J_p_md5"][str(l)]: stop(f"{label} doc {mt['doc']} J_p_{l} md5 != meta")
        Jl.append(torch.load(p).float())
    ap = os.path.join(d, f"acc_{l}.pt"); res["files"][f"{label}/acc_{l}.pt"] = md5(ap); a = torch.load(ap)
    s, tot = a[0].float(), int(a[1])
    if label == "stage2" and sorted(a[2]) != DONE2: stop(f"stage2 acc_{l} docs {sorted(a[2])} != progress {DONE2}")
    want = sum(nn * j for nn, j in zip(n, Jl)); rel = float((s - want).norm() / want.norm())
    if tot != sum(n) or rel > 1e-5: stop(f"{label} acc_{l}: tot {tot} vs {sum(n)}, rel {rel:.2e}")
    del Jl; return s, tot, rel
DK = json.load(open(os.path.join(ST2, "dose_kl.night.json")))["table"]["fresh"]
SH = json.load(open(os.path.join(BASE, "score_shrink.json"))); A = {l: float(SH["report"]["frozen_step2_parameters"]["real_per_layer"][str(l)]) for l in GATE_L}
KL = {"logit": [], "J22": [], "J54": [], "shrunk54": []}; ctl, tab, accrep = {}, {}, {}
for l in GATE_L:
    J6 = torch.load(os.path.join(JD, f"J_l_{l}.pt")).float(); res["files"][f"record/J_l_{l}.pt"] = md5(os.path.join(JD, f"J_l_{l}.pt"))
    s1, t1, r1 = check_acc(S1, l, m1, n1, "stage1"); s2, t2, r2 = check_acc(S2, l, m2, n2, "stage2")
    accrep[l] = {"stage1_rel": r1, "stage2_rel": r2, "stage2_tot": t2}
    J22 = (N_REC * J6 + s1) / (N_REC + t1); J54 = (N_REC * J6 + s1 + s2) / (N_REC + t1 + t2)
    hf = fresh["layers"][l]; k_lo, k22 = klv(hf), klv(hf @ J22.T)
    c1 = max(abs(float(k_lo.mean()) - DK[str(l)]["logit"]["kl"]), abs(float(k22.mean()) - DK[str(l)]["J22"]["kl"]))
    c2 = float(klv(fresh["h_final"]).abs().max())
    c3 = float((J54 - J22).abs().max())   # must DIFFER: stage 2 actually moved the map (0 = S2 never entered the pool)
    k54 = klv(hf @ J54.T)
    zpool = (N_REC * (hf @ J6.T) + hf @ s1.T + hf @ s2.T) / (N_REC + t1 + t2)   # linearity route
    kz = klv(zpool)
    c4 = abs(float(kz.mean()) - float(k54.mean()))
    c4max = float((kz - k54).abs().max())   # Deron P148: report-only per-position max |dKL|; fp32 reorder should give ~1e-6, exactly 0.0 everywhere = shared path -> STOP
    ctl[l] = {"c1_logit_J22_vs_dose_kl": c1, "c2_code_smoke_test_cannot_fail_on_data": c2, "c3_J54_differs_from_J22_maxabs": c3, "c4_pool_vs_direct": c4, "c4_max_per_position_absdKL": c4max}
    if c4max == 0.0: res["controls"] = ctl; stop(f"c4 per-position |dKL| is exactly 0.0 at every position at L{l}: the pool route shares the direct route's path (Deron P148)")
    if not (c1 <= 1e-4 and c2 <= 1e-4 and c3 > 0.0 and c4 <= 1e-4): res["controls"] = ctl; stop(f"control failed at L{l}: {ctl[l]}")   # NaN compares False, so a NaN control STOPS
    if not CONTROLS_ONLY:
        ks = klv(hf + A[l] * (hf @ (J54 - torch.eye(DIM)).T))
        for nm, v in (("logit", k_lo), ("J22", k22), ("J54", k54), ("shrunk54", ks)):
            if not bool(torch.isfinite(v).all()): stop(f"non-finite KL in {nm} at L{l} ({int((~torch.isfinite(v)).sum())} positions): instrument fault, not a FAIL")
            KL[nm].append(v)
        tab[l] = {k_: float(v[-1].mean()) for k_, v in KL.items()}
    del J6, J22, J54, s1, s2, zpool; res["memory"][f"rss_after_L{l}"] = peak_gib(); log(f"L{l} controls OK")
res["resume"] = accrep; res["controls"] = ctl; dump(); log("controls c1-c4 OK")
if CONTROLS_ONLY: res["verdict"] = "CONTROLS-OK (no gate run; no J54 KL reported)"; dump(); log(res["verdict"]); sys.exit(0)
# ---- 5 GATE ---------------------------------------------------------------------------------------------------------------
M = {k_: torch.stack(v).mean(0) for k_, v in KL.items()}   # mid-band mean per fresh position
D = M["J22"] - M["J54"]
if not bool(torch.isfinite(D).all()): stop(f"non-finite D ({int((~torch.isfinite(D)).sum())} positions): instrument fault, not a FAIL")
res["gate"] = gate(D); res["verdict"] = res["gate"]["decision"]; dump()
log("GATE:", res["verdict"], f"mean D {res['gate']['mean_D']:.4f} CI {res['gate']['ci95']}")
# ---- 6 REPORT-ONLY -------------------------------------------------------------------------------------------------------
def cluster_ci(v, seed):
    g = torch.Generator().manual_seed(seed); cl = torch.stack([v[fpass == c].mean() for c in range(6)])
    bs = torch.stack([cl[torch.randint(0, 6, (6,), generator=g)].mean() for _ in range(B)]); return cl.tolist(), [float(bs.quantile(.025)), float(bs.quantile(.975))]
pp, cc = cluster_ci(D, BSEED + 1)
rep = {"heading": "REPORT-ONLY (post P137). Nothing below changes the verdict.", "D_cluster_by_passage": {"per_passage": pp, "ci95_crude": cc}, "per_layer_kl": tab}
for k_ in ("J54", "shrunk54", "J22"):
    d = M[k_] - M["logit"]; g = gate(d); rep[f"{k_}_minus_logit"] = {"mean": g["mean_D"], "ci95": g["ci95"]}
m = res["gate"]["mean_D"]
rep["prediction_check"] = {"registered": "gain in [0.25, 1.30], point 0.46 (1/n); J54 - logit stays > 0", "gain": m,
                           "gain_in_band": 0.25 <= m <= 1.30, "J54_minus_logit_gt_0": rep["J54_minus_logit"]["mean"] > 0}
res["report"] = rep; res["memory"]["rss_peak_gib"] = peak_gib(); res["minutes"] = round((time.time() - T0) / 60, 1); res["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S"); dump()
log("done:", res["verdict"])
