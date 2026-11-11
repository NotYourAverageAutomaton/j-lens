"""J-Lens STAGE 2 STEP 0: KL dose curve + fresh-position KL, CPU only, file-only, REPORT-ONLY (Claude 2026-11-08; spec Deron P131 s2 +
post P132 p@k row + post P133 prediction line; card C01). Nothing here gates anything; its one sentence goes to Niz, who decides Stage 2.
METRIC (post P131): KL(P_final || P_lens) per position, mean over mid-band L20..44 by 4, lower is better; top-1 agreement secondary.
PREDICTION, written before this runs (post P133; from Jett round 2 on Pythia, where shrinkage was a KL fix): on the 1,434 FRESH step-2
positions the shrunk lens at the frozen step-2 a lands near the logit lens on KL (|mid-band mean KL_shrunk - KL_logit| <= 0.5 nats) and
below raw J22 at every mid-band layer. Already known, in-sample (score_shrink.json readout_table, the 306 positions that picked a):
shrunk - logit = -0.47..-0.95 nats per layer, mean -0.65; raw J22 is worse than logit at every mid-band layer (post P131).
ORDER (every step before 6 can only STOP; the outputs are report rows):
  0 memory guard (MemAvailable >= 30 GiB at start; mid-run >= 15 GiB before every subset); pins; md5 of every file read.
  1 disjointness of the 16 Stage-1 docs vs the 6 held passages (as f2409a09). The fresh passages' overlap check is step 2's (fc221155).
  2 resume (b)+(c) exactly as f2409a09; J22 pooled n_valid-weighted (post P089 Q1). Per layer, z-precompute h @ J6^T and h @ J_p^T for the
    306 held AND the 1,434 fresh positions, so every subset's lens state is a weighted sum (linearity of h @ J^T), not a new 5120^2 map.
  3 controls (STOP): (c1) a = 0 shrunk state is bit-equal to h, so its KL equals logit's exactly (FAIL-way: gain 0); (c2) oracle state
    h_final gives KL <= 1e-4 at every position (PASS-way); (c3) J6 and logit KL on held reproduce score_stage1.json per_layer_table kl,
    |diff| <= 1e-4; (c4) the z-pooled all-16 state reproduces score_stage1 J22 kl, |diff| <= 1e-4; (c5) shrunk at the frozen a reproduces
    score_shrink.json readout_table n22 kl AND fut20 for that a, |diff| <= 1e-4 / 1e-6; (c6) TL-big and TL-shuf KL reproduce score_tl.json.
  4 table: per layer, KL and top-1 for logit, J6, J22, shrunk(frozen a), TL-big, TL-shuf on held (306), and logit, J6, J22, shrunk on
    fresh (1,434). On fresh, report the paired bootstrap 95% CI (B 10000, seed 20261110, over positions; cluster-by-passage reported)
    of mid-band KL vs logit for J22 and shrunk ("better than logit" needs CI < 0; report-only here, post P131).
  5 DOSE on held: the SAME 750 subsets as f2409a09 (generator seed 20261106+3, same draw order); n = 6 + k (J6 = 6 docs as one unit).
    Raw J_n: all 50 subsets per k. Shrunk at frozen a: the FIRST 10 subsets per k (DEVIATION from post P131's 50, declared: one readout is
    3.84 s measured on the GPU host, so the full spec is ~11 h of CPU beside Hans; this trim is ~7 h). Checkpointed; --stop-at HH:MM pauses
    before the 06:30 render; a re-run resumes and must reproduce the same J6/J22 means.
  6 REPORT: M_k = mean over subsets and positions of mid-band KL; slope b of M_k vs ln(6+k), k = 0..16, CI by resampling POSITIONS
    (B 2000, seed 20261110); X = b * (ln 54 - ln 22), the projected mid-band KL change from 22 to 54 docs (negative = better).
    Y = half-width of the paired bootstrap 95% CI (B 10000, seed 20261110) on the 1,434 fresh positions of mean[KL(J22) - KL(J_14)]
    for the first 5 k=8 subsets (a measured dose difference on Stage 2's own scoring set; PROXY for the not-yet-existing J22-J54
    difference: declared). Doc-sampling spread: SD over the 50 subsets of their mean KL, per k (with 16 docs, no random 22-of-many draw
    exists; the k=15 spread is the nearest, declared). Same slope/X for the shrunk dose. The sentence for Niz:
    "projected KL gain at 54 = -X vs CI half-width Y".
  7 p@k row (post P132, report-only): on held, N = 0..3, k = 1, 5, 10: is the probe's top-1 in the MODEL's own top-k at T+N (P_final at
    T+N from held_h; positions without T+N in their passage dropped and counted); surprisal -log P_final,T+N(probe top-1). Lenses:
    logit, J22, shrunk, TL-big; baselines: unigram top-1 (unigram_top20[0]), skip-bigram (most frequent x_{T+1+N} given x_T, counted
    on wikitext-103 train from zip 91c00ae2 after the held/record/Stage-1 overlap drop), TL-shuf. All positions + PAPER-MATCHED subset
    (final top-1 == true next token); an x_T unseen in the corpus falls back to the unigram token (counted). Deviations (post P132): real held text, not greedy continuations. Claim flag, mid-band mean per
    position: paired 95% CI (B 10000, seed 20261111) of p@k(lens) - p@k(baseline) > 0 vs ALL THREE baselines, N >= 1; else report row.
Output step2/dose_kl.json (+ dose_kl.ckpt.pt). Usage: SCORE_THREADS=12 nice -n 10 python dose_kl.py [--stop-at 06:15] [--no-pk]"""
import ast, hashlib, json, math, os, re, resource, sys, time, platform, zipfile, torch
from collections import Counter, defaultdict
from safetensors import safe_open
from transformers import AutoTokenizer
torch.set_num_threads(int(os.environ.get("SCORE_THREADS", max(1, os.cpu_count() - 4))))
BASE = "$JLENS_HOME/claude-normtest"; JD = "$JLENS_HOME/results-whole-swift-record"; MD = "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b"
S1 = os.path.join(BASE, "stage1"); S2 = os.path.join(BASE, "step2"); TLDIR = os.path.join(BASE, "tl_out_sel")
OUTP = os.path.join(S2, "dose_kl.json"); CK = os.path.join(S2, "dose_kl.ckpt.pt"); ZIP = os.path.join(BASE, "tl_corpus", "wikitext-103-raw-v1.zip")
GATE_L = [20, 24, 28, 32, 36, 40, 44]; EPS = 1e-6; SKIP = 16; N_REC_PARTS = [131, 107, 96, 96, 93, 119]; DIM = 5120
SEED = 20261106; BSEED = 20261110; PKSEED = 20261111; NEED_GIB = 30.0; MID_MIN_GIB = 15.0; SHRUNK_SUBSETS = 10; Y_SUBSETS = 5
PIN = {"held_h.pt": "3ca0f99c", "eval.py": "cabcfb79", "stage1_fit.py": "fabece82", "pilot_hutch.py": "5fa18d9b",
       "$JLENS_HOME/whole_model_fit_record.log": "d0cf6a41", "score_stage1.json": "67449876", "score_shrink.json": "0a925a72",
       "tl_out_sel/score_tl.json": "183150ea", "tl_out_sel/tl_meta.json": "0e7c0820", "step2/fresh_h.pt": "bcb22760", "step2/passages.json": "2d302976"}
STOP_AT = sys.argv[sys.argv.index("--stop-at") + 1] if "--stop-at" in sys.argv else None; DO_PK = "--no-pk" not in sys.argv
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
peak_gib = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
def mem_avail_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"): return int(line.split()[1]) / 2**20
res = {"script_md5": md5(__file__), "label": "REPORT-ONLY (post P131 s2 step 0). Nothing here gates anything.", "verdict": "INCOMPLETE",
       "stops": [], "pins": {}, "files": {}, "deviations": [f"shrunk dose uses the first {SHRUNK_SUBSETS} of 50 subsets per k (runtime)",
       f"Y is a proxy: J22 vs J_14 (first {Y_SUBSETS} k=8 subsets) on the fresh positions, not J22 vs J54",
       "doc-sampling spread: no random 22-of-many draw exists with 16 docs; per-k SD reported, k=15 nearest",
       "p@k: real held text, not greedy continuations (post P132 a); all positions plus paper-matched subset (post P132 b)"],
       "env": {"torch": torch.__version__, "python": platform.python_version(), "threads": torch.get_num_threads(), "host": platform.node(), "note": "CPU-only; GPU not used"}}
def dump(): json.dump(res, open(OUTP, "w"), indent=1)
def stop(why): res["stops"].append(why); res["verdict"] = "NO-REPORT: " + why; dump(); log("STOP:", why); sys.exit(3)
# ---- 0 memory guard FIRST, then pins --------------------------------------------------------------------------------
_ma0 = mem_avail_gib(); res["memory"] = {"mem_available_at_start_gib": round(_ma0, 1), "required_gib": NEED_GIB}
if _ma0 < NEED_GIB: stop(f"memory guard: MemAvailable {_ma0:.1f} GiB < {NEED_GIB} (live engine protected)")
for name, want in PIN.items():
    got = md5(name if name.startswith("/") else os.path.join(BASE, name)); res["pins"][name] = got
    if not got.startswith(want): stop(f"pin {name} {got[:8]} != {want}")
S1REF = json.load(open(os.path.join(BASE, "score_stage1.json")))["report"]["per_layer_table"]["rows"]
SH = json.load(open(os.path.join(BASE, "score_shrink.json"))); RT = SH["report"]["readout_table"]["n22"]
A = {l: float(SH["report"]["frozen_step2_parameters"]["real_per_layer"][str(l)]) for l in GATE_L}
TS = json.load(open(os.path.join(TLDIR, "score_tl.json"))); TROWS = TS["report"]["per_layer_table"]; TM = json.load(open(os.path.join(TLDIR, "tl_meta.json")))
log("pins OK; frozen a", A)
# ---- shared CPU path (tl_score 368b8b09 / f2409a09, unchanged maths) ------------------------------------------------
idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
def W(n):
    with safe_open(os.path.join(MD, idx[n]), "pt") as f: return f.get_tensor(n).float()
WU = W("lm_head.weight"); g1 = 1.0 + W("model.language_model.norm.weight")
def logits_of(x):
    xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * g1
    return xn @ WU.T
held = torch.load(os.path.join(BASE, "held_h.pt")); fresh = torch.load(os.path.join(S2, "fresh_h.pt"))
def held_out_from(path):
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "HELD_OUT" for t in node.targets): return ast.literal_eval(node.value)
    raise SystemExit(f"no HELD_OUT in {path}")
HELD_OUT = held_out_from(os.path.join(BASE, "pilot_hutch.py"))
if HELD_OUT != held_out_from(os.path.join(BASE, "eval.py")): stop("HELD_OUT in pilot_hutch.py != HELD_OUT in eval.py")
tok = AutoTokenizer.from_pretrained(MD); bos = tok.bos_token_id
held_ids, nexts, hpos = [], [], []   # hpos[q] = (passage, token index t)
for p, text in enumerate(HELD_OUT):
    ids = tok(text, truncation=True, max_length=256 - (1 if bos is not None else 0), add_special_tokens=False).input_ids
    ids = ([bos] if bos is not None else []) + ids; held_ids.append(ids)
    for q in range(SKIP, len(ids) - 1): nexts.append(torch.tensor(ids[q + 1:q + 11])); hpos.append((p, q))
NH = len(nexts); NF = fresh["h_final"].shape[0]
if NH != held["h_final"].shape[0] or NH != 306: stop(f"held positions {NH} / {held['h_final'].shape[0]} != 306")
if NF != 1434: stop(f"fresh positions {NF} != 1434")
hpass, fpass = torch.tensor(held["passage"]), torch.tensor(fresh["passage"])
def final_of(hf):
    lp = torch.log_softmax(logits_of(hf), -1); return lp, lp.exp(), lp.argmax(-1)
LPH, PH, T1H = final_of(held["h_final"]); LPF, PF, T1F = final_of(fresh["h_final"])
def klvec(x, LP, P):   # per-position KL(P_final || P_lens) and top-1 agreement vector
    lg = logits_of(x); lp = torch.log_softmax(lg, -1); return (P * (LP - lp)).sum(-1), lg.argmax(-1)
def kl_h(x): k, t = klvec(x, LPH, PH); return k, (t == T1H).float()
def kl_f(x): k, t = klvec(x, LPF, PF); return k, (t == T1F).float()
def fut20_mean(x):
    t20 = logits_of(x).topk(20, -1).indices; return float(sum(float(torch.isin(n, t20[i]).float().mean()) for i, n in enumerate(nexts)) / NH)
# ---- 1 disjointness (f2409a09) --------------------------------------------------------------------------------------
corpus = json.load(open(os.path.join(BASE, "stage1_corpus.json")))
norm_t = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
def shingles(s, k=12):
    w = norm_t(s).split(); return {" ".join(w[i:i + k]) for i in range(max(0, len(w) - k + 1))}
held_sh = set().union(*[shingles(t) for t in HELD_OUT]); held_hash = {hashlib.sha256(norm_t(t).encode()).hexdigest() for t in HELD_OUT}
overlap = [i + 1 for i, d in enumerate(corpus) if hashlib.sha256(norm_t(d["text"]).encode()).hexdigest() in held_hash or shingles(d["text"]) & held_sh]
res["disjoint"] = {"stage1_docs": len(corpus), "held_out": len(HELD_OUT), "overlapping_docs": overlap}
if overlap: stop(f"Stage 1 docs overlap the held-out passages: {overlap}")
log("disjoint OK")
# ---- 2 resume (b)+(c), J22, z-precompute ----------------------------------------------------------------------------
rec_log = open("$JLENS_HOME/whole_model_fit_record.log").read()
parts = [int(x) for x in re.findall(r"n_valid=(\d+)", rec_log)]; N_REC = sum(parts)
if parts != N_REC_PARTS or N_REC != 642: stop(f"N_rec parts {parts} != {N_REC_PARTS}")
docs = sorted(d for d in os.listdir(S1) if re.fullmatch(r"doc_\d\d", d))
if docs != [f"doc_{i:02d}" for i in range(1, 17)]: stop(f"Stage 1 doc dirs are not exactly doc_01..doc_16: {docs}")
meta = [json.load(open(os.path.join(S1, d, "meta.json"))) for d in docs]; n = [int(m["n_valid"]) for m in meta]
if [m["doc"] for m in meta] != list(range(1, 17)) or min(n) <= 0: stop(f"meta docs/n_valid bad: {n}")
res["N"] = {"N_rec": N_REC, "n_valid": n, "sum_n_p": sum(n)}
ZH, ZF, J22, acc_ok = {}, {}, {}, {}   # ZH[l] = {"j6": h@J6^T, "p": [h@J_p^T]} (held); ZF same on fresh
for l in GATE_L:
    J6 = torch.load(os.path.join(JD, f"J_l_{l}.pt")).float(); res["files"][f"record/J_l_{l}.pt"] = md5(os.path.join(JD, f"J_l_{l}.pt"))
    Jl = []
    for i, d in enumerate(docs):
        p = os.path.join(S1, d, f"J_p_{l}.pt"); h = md5(p); res["files"][f"stage1/{d}/J_p_{l}.pt"] = h
        if h != meta[i]["J_p_md5"][str(l)]: stop(f"{d} J_p_{l} md5 {h[:8]} != meta")
        Jl.append(torch.load(p).float())
    ap = os.path.join(S1, f"acc_{l}.pt"); res["files"][f"stage1/acc_{l}.pt"] = md5(ap); s, tot = torch.load(ap)
    want = sum(nn * j for nn, j in zip(n, Jl)); rel = float((s.float() - want).norm() / want.norm())
    acc_ok[l] = {"tot": int(tot), "rel_diff": rel}
    if int(tot) != sum(n) or rel > 1e-5: stop(f"acc_{l}: tot {tot} vs {sum(n)}, rel {rel:.2e}")
    J22[l] = (N_REC * J6 + want) / (N_REC + sum(n))
    hh, hf_ = held["layers"][l], fresh["layers"][l]
    ZH[l] = {"j6": hh @ J6.T, "p": [hh @ j.T for j in Jl]}; ZF[l] = {"j6": hf_ @ J6.T, "p": [hf_ @ j.T for j in Jl]}
    del Jl, want, s, J6; res["memory"][f"rss_after_layer_{l}"] = peak_gib()
res["resume_b_c"] = acc_ok; log("resume (b)+(c) OK; z precomputed")
def zpool(Z, l, sel):   # state of J_n = record + selected docs, n_valid-weighted: h @ J_n^T by linearity
    return (N_REC * Z[l]["j6"] + sum(n[i] * Z[l]["p"][i] for i in sel)) / (N_REC + sum(n[i] for i in sel))
shr = lambda h, z, a: h + a * (z - h)   # h + a * h @ (J - I)^T, as step2_score / shrink_check (h + a * hD)
ALL = list(range(16)); I = torch.eye(DIM)
direct = lambda h, l: h @ J22[l].T                                # J22 state, direct (score_stage1 / tl_score)
shr_direct = lambda h, l, a: h + a * (h @ (J22[l] - I).T)         # exactly step2_score's h + a * hD
# ---- 3 controls -----------------------------------------------------------------------------------------------------
ctl = {}
for l in GATE_L:
    hh = held["layers"][l]; kl_lo = kl_h(hh)[0]; z22 = zpool(ZH, l, ALL)
    c1 = bool(torch.equal(shr(hh, z22, 0.0), hh))
    c2 = float(klvec(held["h_final"], LPH, PH)[0].abs().max())
    c3 = max(abs(float(kl_h(ZH[l]["j6"])[0].mean()) - S1REF[str(l)]["J6"]["kl"]), abs(float(kl_lo.mean()) - S1REF[str(l)]["logit"]["kl"]))
    c4 = abs(float(kl_h(z22)[0].mean()) - S1REF[str(l)]["J22"]["kl"])
    c4b = abs(float(kl_h(shr(hh, z22, A[l]))[0].mean()) - float(kl_h(shr_direct(hh, l, A[l]))[0].mean()))   # pooled route == direct route (dose uses pooled)
    zs = shr_direct(hh, l, A[l]); row = RT[str(l)][f"a={A[l]}"]
    c5 = (abs(float(kl_h(zs)[0].mean()) - row["kl"]), abs(fut20_mean(zs) - row["fut20"]))
    ctl[l] = {"c1_a0_bit_equal_h": c1, "c2_oracle_max_kl": c2, "c3_j6_logit_absdiff": c3, "c4_zpool_j22_absdiff": c4, "c4b_zpool_vs_direct_shrunk_kl": c4b, "c5_shrunk_kl_fut20_absdiff": c5}
    if not c1 or c2 > 1e-4 or c3 > 1e-4 or c4 > 1e-4 or c4b > 1e-4 or c5[0] > 1e-4 or c5[1] > 1e-6: res["controls"] = ctl; stop(f"control failed at L{l}: {ctl[l]}")
LENS = {}
for name in ("big", "shuf"):
    for l in GATE_L:
        p = os.path.join(TLDIR, f"{name}_L{l}.pt"); h = md5(p); res["files"][f"tl/{name}_L{l}.pt"] = h
        if h != TM["files"][f"{name}_L{l}.pt"]: stop(f"{name}_L{l}.pt md5 != tl_meta")
        LENS[(name, l)] = torch.load(p)
tlap = lambda name, l, h: h + h @ LENS[(name, l)]["R"].T + LENS[(name, l)]["b"]
for l in GATE_L:
    c6 = max(abs(float(kl_h(tlap(nm, l, held["layers"][l]))[0].mean()) - TROWS[str(l)][f"TL-{nm}"]["kl"]) for nm in ("big", "shuf"))
    ctl[l]["c6_tl_kl_absdiff"] = c6
    if c6 > 1e-4: res["controls"] = ctl; stop(f"control c6 failed at L{l}: {c6}")
res["controls"] = ctl; dump(); log("controls c1-c6 OK")
# ---- 4 table: held (306) and fresh (1,434) ----------------------------------------------------------------------------
tab = {"held": {}, "fresh": {}}; KLF = {"logit": [], "J6": [], "J22": [], "shrunk": []}
for l in GATE_L:
    hh, hf_ = held["layers"][l], fresh["layers"][l]
    sth = {"logit": hh, "J6": ZH[l]["j6"], "J22": direct(hh, l), "shrunk": shr_direct(hh, l, A[l]), "TL-big": tlap("big", l, hh), "TL-shuf": tlap("shuf", l, hh)}
    stf = {"logit": hf_, "J6": ZF[l]["j6"], "J22": direct(hf_, l), "shrunk": shr_direct(hf_, l, A[l])}
    tab["held"][l] = {}; tab["fresh"][l] = {}
    for k_, x in sth.items():
        kv, tv = kl_h(x); tab["held"][l][k_] = {"kl": float(kv.mean()), "top1": float(tv.mean())}
    for k_, x in stf.items():
        kv, tv = kl_f(x); tab["fresh"][l][k_] = {"kl": float(kv.mean()), "top1": float(tv.mean())}; KLF[k_].append(kv)
    log(f"table L{l} done")
KLF = {k_: torch.stack(v).mean(0) for k_, v in KLF.items()}   # mid-band mean per fresh position
gb = torch.Generator().manual_seed(BSEED); BIDX = torch.randint(0, NF, (10000, NF), generator=gb)
def ci(v): bs = v[BIDX].mean(1); return [float(bs.quantile(.025)), float(bs.quantile(.975))]
def cluster_ci(v, pas, seed):
    g = torch.Generator().manual_seed(seed); cl = torch.stack([v[pas == c].mean() for c in range(6)])
    bs = torch.stack([cl[torch.randint(0, 6, (6,), generator=g)].mean() for _ in range(10000)]); return cl.tolist(), [float(bs.quantile(.025)), float(bs.quantile(.975))]
vs_logit = {}
for k_ in ("J22", "shrunk", "J6"):
    d = KLF[k_] - KLF["logit"]; pp, cc = cluster_ci(d, fpass, BSEED + 1)
    vs_logit[k_] = {"mean": float(d.mean()), "ci95": ci(d), "per_passage": pp, "cluster_ci95_crude": cc, "better_than_logit_report_only": ci(d)[1] < 0}
pred = KLF["shrunk"].mean() - KLF["logit"].mean()
res["table"] = tab; res["fresh_vs_logit"] = vs_logit
res["prediction_check"] = {"text": "shrunk lands near logit on KL (|mid-band mean diff| <= 0.5; band set by Claude P134, after the in-sample number was seen) and below raw J22 at every mid-band layer (fresh)",
                           "mid_band_shrunk_minus_logit": float(pred),
                           "shrunk_below_J22_every_layer": all(tab["fresh"][l]["shrunk"]["kl"] < tab["fresh"][l]["J22"]["kl"] for l in GATE_L),
                           "near_logit": abs(float(pred)) <= 0.5}
dump(); log("table done; fresh shrunk - logit", float(pred))
# ---- 5 DOSE (held, checkpointed) --------------------------------------------------------------------------------------
gd = torch.Generator().manual_seed(SEED + 3)   # f2409a09's exact draw: same seed, same call order
SUBSETS = {k: [torch.randperm(16, generator=gd)[:k].tolist() for _ in range(50)] for k in range(1, 16)}
def midband_kl(sel, shrunk):
    acc = torch.zeros(NH)
    for l in GATE_L:
        z = zpool(ZH, l, sel); x = shr(held["layers"][l], z, A[l]) if shrunk else z
        acc += kl_h(x)[0]
    return acc / len(GATE_L)
REF = {"j6": midband_kl([], False), "j22": midband_kl(ALL, False), "s6": midband_kl([], True), "s22": midband_kl(ALL, True)}
dose = {"raw": {}, "shrunk": {}}
if os.path.exists(CK):
    ck = torch.load(CK)
    if ck["script_md5"] != res["script_md5"] or any(not torch.allclose(ck["ref"][k_], REF[k_], atol=1e-6) for k_ in REF): stop("checkpoint from a different script or different refs: refusing to resume")
    dose = ck["dose"]; log("resuming:", {a: {k: len(v) for k, v in d.items()} for a, d in dose.items()})
def ckpt(reason=None): torch.save({"script_md5": res["script_md5"], "ref": REF, "dose": dose, "stopped": reason}, CK)
for k in range(1, 16):
    for arm, cap in (("raw", 50), ("shrunk", SHRUNK_SUBSETS)):
        dose[arm].setdefault(k, [])
        for j in range(len(dose[arm][k]), cap):
            ma = mem_avail_gib()
            if ma < MID_MIN_GIB: ckpt(f"memory {ma:.1f}"); res["dose_stopped"] = f"mid-run guard at {arm} k={k} j={j}: {ma:.1f} GiB"; dump(); log("DOSE STOP (memory)"); sys.exit(4)
            if STOP_AT and time.strftime("%H:%M") >= STOP_AT and time.strftime("%H:%M") < "12:00":
                ckpt(f"stop-at {STOP_AT}"); res["dose_stopped"] = f"paused at {arm} k={k} j={j}"; dump(); log("DOSE PAUSE"); sys.exit(5)
            dose[arm][k].append(midband_kl(SUBSETS[k][j], arm == "shrunk")); ckpt()
    log(f"dose k={k}: raw {float(torch.stack(dose['raw'][k]).mean()):.4f} shrunk {float(torch.stack(dose['shrunk'][k]).mean()):.4f}")
# ---- 6 REPORT: slope, projection, Y, spread ---------------------------------------------------------------------------
LN = torch.tensor([math.log(6.0 + k) for k in range(17)])
def curve(arm, r0, r16):
    per_k = [r0.unsqueeze(0)] + [torch.stack(dose[arm][k]) for k in range(1, 16)] + [r16.unsqueeze(0)]   # [subsets, NH] per k
    mk = torch.stack([pk.mean(0) for pk in per_k])                                                    # [17, NH]
    def slope(m):
        M = m.mean(1); x = LN - LN.mean(); return float((x * (M - M.mean())).sum() / (x ** 2).sum())
    b = slope(mk); g = torch.Generator().manual_seed(BSEED + 2)
    sl = torch.tensor([slope(mk[:, torch.randint(0, NH, (NH,), generator=g)]) for _ in range(2000)])
    X = b * (math.log(54) - math.log(22)); Xci = [float(v * (math.log(54) - math.log(22))) for v in (sl.quantile(.025), sl.quantile(.975))]
    spread = {6 + k: float(per_k[k].mean(1).std()) for k in range(1, 16)}
    return {"n": [6 + k for k in range(17)], "mean_kl": mk.mean(1).tolist(), "slope_vs_ln_n": b, "slope_ci95_positions": [float(sl.quantile(.025)), float(sl.quantile(.975))],
            "X_projected_kl_change_22_to_54": X, "X_ci95": Xci, "doc_sampling_spread_sd_by_n": spread, "subsets_per_k": len(dose[arm][1])}
res["dose"] = {"raw": curve("raw", REF["j6"], REF["j22"]), "shrunk": curve("shrunk", REF["s6"], REF["s22"])}
dy = []
for j in range(Y_SUBSETS):
    acc = torch.zeros(NF)
    for l in GATE_L: acc += kl_f(direct(fresh["layers"][l], l))[0] - kl_f(zpool(ZF, l, SUBSETS[8][j]))[0]
    dy.append(acc / len(GATE_L))
Ys = [(c[1] - c[0]) / 2 for c in (ci(d) for d in dy)]; Y = sum(Ys) / len(Ys)
X = res["dose"]["raw"]["X_projected_kl_change_22_to_54"]
res["niz_sentence"] = (f"projected KL gain at 54 = {-X:.4f} nats (raw J, mid-band; CI of the projection {[-v for v in res['dose']['raw']['X_ci95']][::-1]}) "
                       f"vs CI half-width Y = {Y:.4f} (proxy, fresh positions)" + ("; Stage 2 could not see its own effect" if -X < Y else ""))
res["Y"] = {"per_subset_half_width": Ys, "mean": Y, "fresh_mean_kl_J22_minus_J14": [float(d.mean()) for d in dy]}
res["verdict"] = "REPORT"; res["memory"]["rss_peak_gib"] = peak_gib(); dump(); log(res["niz_sentence"])
# ---- 7 p@k row (post P132, report-only) -----------------------------------------------------------------------------------
if DO_PK:
    need = {held_ids[p][t] for p, t in hpos}; cnt = {N: defaultdict(Counter) for N in range(4)}
    BAN = [held_out_from(os.path.join(BASE, "pilot_hutch.py")), ast.literal_eval(next(node.value for node in ast.parse(open("$JLENS_HOME/whole_model_fit.py").read()).body
           if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PROMPTS" for t in node.targets))), [d["text"] for d in corpus]]
    ban_txt = [t for grp in BAN for t in grp]; ban_sh = set().union(*[shingles(t) for t in ban_txt]); ban_hash = {hashlib.sha256(norm_t(t).encode()).hexdigest() for t in ban_txt}
    res["pk_zip_sha256"] = hashlib.sha256(open(ZIP, "rb").read()).hexdigest()
    if not res["pk_zip_sha256"].startswith("91c00ae287f0d699"): stop("p@k: corpus zip sha256 mismatch")
    raw = zipfile.ZipFile(ZIP).read("wikitext-103-raw/wiki.train.raw").decode("utf-8"); arts, cur = [], []
    for line in raw.split("\n"):
        if re.fullmatch(r" = [^=].* = ", line) and cur: arts.append("\n".join(cur)); cur = []
        cur.append(line)
    if cur: arts.append("\n".join(cur))
    del raw; dropped = 0
    for a in arts:
        a = a.strip()
        if not a: continue
        if hashlib.sha256(norm_t(a).encode()).hexdigest() in ban_hash or shingles(a) & ban_sh: dropped += 1; continue
        ids = tok(a, add_special_tokens=False).input_ids
        for i, x in enumerate(ids):
            if x in need:
                for N in range(4):
                    if i + 1 + N < len(ids): cnt[N][x][ids[i + 1 + N]] += 1
    res["pk_corpus"] = {"articles": len(arts), "dropped_overlap": dropped}; del arts
    U1 = TM["unigram_top20"][0]
    probes = {}
    for l in GATE_L:
        hh = held["layers"][l]
        for nm, x in (("logit", hh), ("J22", direct(hh, l)), ("shrunk", shr_direct(hh, l, A[l])), ("TL-big", tlap("big", l, hh)), ("TL-shuf", tlap("shuf", l, hh))):
            probes.setdefault(nm, []).append(logits_of(x).argmax(-1))
    probes = {k_: torch.stack(v) for k_, v in probes.items()}   # [7, NH]
    matched = torch.tensor([T1H[q].item() == held_ids[p][t + 1] for q, (p, t) in enumerate(hpos)])
    pk = {}; g = torch.Generator().manual_seed(PKSEED)
    for N in range(4):
        valid = [q for q, (p, t) in enumerate(hpos) if q + N < NH and hpos[q + N][0] == p]
        qv = torch.tensor(valid); tgt = qv + N; sb = torch.tensor([cnt[N][held_ids[hpos[q][0]][hpos[q][1]]].most_common(1)[0][0] if cnt[N][held_ids[hpos[q][0]][hpos[q][1]]] else U1 for q in valid])
        pk[N] = {"positions": len(valid), "dropped": NH - len(valid), "paper_matched": int(matched[qv].sum())}
        for kk in (1, 5, 10):
            topk = LPH[tgt].topk(kk, -1).indices
            def hit(tok1):   # tok1: [len(valid)] -> per-position hit
                return (topk == tok1.unsqueeze(1)).any(1).float()
            base = {"unigram": hit(torch.full((len(valid),), U1)), "skip-bigram": hit(sb), "TL-shuf": torch.stack([hit(probes["TL-shuf"][li][qv]) for li in range(7)]).mean(0)}
            row = {b_: {"all": float(v.mean()), "paper_matched": float(v[matched[qv]].mean())} for b_, v in base.items()}
            for nm in ("logit", "J22", "shrunk", "TL-big"):
                v = torch.stack([hit(probes[nm][li][qv]) for li in range(7)]).mean(0); bi = torch.randint(0, len(valid), (10000, len(valid)), generator=g)
                cis = {b_: [float(q_) for q_ in (v - bv)[bi].mean(1).quantile(torch.tensor([.025, .975]))] for b_, bv in base.items()}
                row[nm] = {"all": float(v.mean()), "paper_matched": float(v[matched[qv]].mean()), "ci95_vs": cis,
                           "reads_ahead_claimable": N >= 1 and all(c[0] > 0 for c in cis.values())}
            pk[N][f"p@{kk}"] = row
        sur = {nm: float(torch.stack([-LPH[tgt, probes[nm][li][qv]] for li in range(7)]).mean()) for nm in ("logit", "J22", "shrunk", "TL-big", "TL-shuf")}
        pk[N]["surprisal_mid_band"] = sur
    res["pk_row"] = {"label": "REPORT-ONLY (post P132). Claim needs CI > 0 vs unigram AND skip-bigram AND TL-shuf at N >= 1.", "rows": pk}
res["memory"]["rss_peak_gib"] = peak_gib(); res["minutes"] = round((time.time() - T0) / 60, 1); res["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S"); dump()
log("done:", res["verdict"])
