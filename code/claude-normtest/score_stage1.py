"""J-Lens STAGE 1 scorer, CPU only (Claude 2026-11-06; spec Deron P047 + post P089; card C01). Derived from ctrl_precheck.py e8d62af3
(the CPU fut20 path that reproduced eval_record.json to 0.0 at every layer on 11-04). No model forward, no GPU, no window.
ORDER, and every step before the gate can only STOP (verdict NO-SCORE):
  0 pins:    held_h.pt == 3ca0f99c, eval.py == cabcfb79, stage1_fit.py == fabece82; md5 of every J_l / J_p / acc file read; env.
  1 disjoint: no Stage 1 doc shares a normalised text hash, or any 12-word shingle, with the 6 held-out passages (post P089 fix 3).
  2 resume (b)+(c): exactly 16 meta.json, docs 1..16, each n_valid > 0; each J_p md5 == its meta; for every SAVE layer,
             acc = (sum n_p*J_p, sum n_p) with tot == sum n_p and sum allclose to sum n_p*J_p (one count per doc: no doc
             double-counted, the partial doc 14 not counted). Report-only evidence for (a) and (d) is gathered by resume_evidence.sh.
  3 N:       N_rec == 642 == 131+107+96+96+93+119 (parsed from whole_model_fit_record.log); sum n_p == 3824; N_rec + sum n_p == 4466.
  4 positive control (post P089 fix 2): recompute eval.py's fut20 for J_6 (jlens) and logit at all 13 layers; |diff| vs
             eval_record.json < 5e-6 at every layer, else STOP.
  5 GATE (post P047 s3): per position q, g_n(q) = mean over L in {20..44 by 4} of [fut20_Jn(q,L) - fut20_logit(q,L)];
             D(q) = g_22(q) - g_6(q); paired bootstrap over the 306 positions, B = 10000, seed 20261106;
             CONTINUE iff the 95% percentile CI of mean(D) is entirely > 0, else STOP.
  6 REPORT-ONLY (none of these gates anything): 6-cluster CI ("6 clusters, crude"); per-layer table headed "not a J1 verdict"
             (fut20/top1/KL for J_22, J_6, logit, row-permuted J_22 after its must-fail pre-check); equal-weight-per-doc J_22 gate
             statistic; 8-vs-8 readout agreement at L24/40/52; the 86% new-positions limit (3824/4466) in words.
  --dose: the dose-response (n = 6+k, k = 1..16, 50 random subsets per k, same weighting), slope vs log n with a bootstrap CI.
             ~5,600 CPU readouts at ~8 s each, so it is its OWN run (GPU window or nice'd CPU), never inside the gate run.
J_22 = (N_rec*J_6 + sum_p n_p*J_p) / (N_rec + sum_p n_p), n_valid-weighted (post P089 Q1).
Usage: python score_stage1.py [--dose] [--stage1 DIR] [--out PATH]   (--stage1 / --out exist for the synthetic dry run only)"""
import ast, hashlib, json, os, re, resource, sys, time, platform, torch
from safetensors import safe_open
from transformers import AutoTokenizer
torch.set_num_threads(int(os.environ.get("SCORE_THREADS", max(1, os.cpu_count() - 4))))
BASE = "$JLENS_HOME/claude-normtest"; JD = "$JLENS_HOME/results-whole-swift-record"; MD = "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b"
S1 = sys.argv[sys.argv.index("--stage1") + 1] if "--stage1" in sys.argv else os.path.join(BASE, "stage1")
OUTP = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(BASE, "score_stage1" + ("_dose" if "--dose" in sys.argv else "") + ".json")
DRY = "--stage1" in sys.argv
LAYERS = [16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 62]; GATE_L = [20, 24, 28, 32, 36, 40, 44]; SAVE_L = [20, 24, 28, 32, 36, 40, 44, 48, 52, 56]
EPS = 1e-6; SKIP = 16; B = 10000; SEED = 20261106; N_REC_PARTS = [131, 107, 96, 96, 93, 119]
PIN = {"held_h.pt": "3ca0f99c", "eval.py": "cabcfb79", "stage1_fit.py": "fabece82", "pilot_hutch.py": "5fa18d9b",   # post P090 swap 2
       "$JLENS_HOME/whole_model_fit_record.log": "d0cf6a41"}                                                    # post P090 swap 3
DRY_PEAK_GIB = 18.0; HEADROOM_GIB = 20.0   # post P090 swap 4: measured dry-run RSS peak 17.41 GiB (streamed, 3 dry runs 11-07 00:1x), rounded up; refuse unless MemAvailable >= peak + 20 GiB (Hans stays live)
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
peak_gib = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)   # Linux: KiB
def mem_avail_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"): return int(line.split()[1]) / 2**20
res = {"script_md5": md5(__file__), "dry_run": DRY, "stage1_dir": S1, "verdict": "INCOMPLETE", "stops": [], "pins": {}, "files": {},
       "env": {"torch": torch.__version__, "python": platform.python_version(), "threads": torch.get_num_threads(), "host": platform.node(),
               "transformers": __import__("transformers").__version__, "driver_note": "CPU-only run; GPU driver not used (post P087 baseline 580.173.02)"}}
def dump(): json.dump(res, open(OUTP, "w"), indent=1)
DOSE = "--dose" in sys.argv
NEED = DRY_PEAK_GIB + HEADROOM_GIB + (17.0 if DOSE else 0.0)   # --dose keeps the 160 J_p resident (~17 GiB more)
_ma0 = None
def stop(why):
    res["stops"].append(why); res["verdict"] = "NO-SCORE: " + why; dump(); log("STOP:", why); sys.exit(3)
# ---- memory guard FIRST (post P090 swap 4: "abort BEFORE loading"): the dry-run peak 17.41 GiB is the WHOLE process, so it is
# compared against what is free before the process has loaded anything. (f5e9f22d checked AFTER ~11 GiB of WU/held_h/J_6 were
# already resident, which double-counted them; the real run stopped there at 34.5 < 38, 00:55, nothing read.)
_ma0 = mem_avail_gib(); res["memory"] = {"mem_available_at_start_gib": round(_ma0, 1), "required_gib": NEED}
if _ma0 < NEED: stop(f"memory guard: MemAvailable {_ma0:.1f} GiB at start < dry-run peak {DRY_PEAK_GIB} + {HEADROOM_GIB} GiB{' + 17 (dose)' if DOSE else ''} (live engine protected)")
# ---- 0 pins -----------------------------------------------------------------------------------------------------
for name, want in PIN.items():
    got = md5(name if name.startswith("/") else os.path.join(BASE, name)); res["pins"][name] = got
    if not got.startswith(want): stop(f"pin {name} {got[:8]} != {want}")
for l in LAYERS: res["files"][f"record/J_l_{l}.pt"] = md5(os.path.join(JD, f"J_l_{l}.pt"))
log("pins OK")
# ---- shared CPU fut20 path (ctrl_precheck.py e8d62af3, unchanged maths) -----------------------------------------
idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
def W(n):
    with safe_open(os.path.join(MD, idx[n]), "pt") as f: return f.get_tensor(n).float()
WU = W("lm_head.weight"); g1 = 1.0 + W("model.language_model.norm.weight")
held = torch.load(os.path.join(BASE, "held_h.pt")); ref = json.load(open(os.path.join(BASE, "eval_record.json")))["layers"]
def held_out_from(path):   # post P090 swap 2: ast.literal_eval of the module's HELD_OUT assignment, never eval()
    tree = ast.parse(open(path).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "HELD_OUT" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit(f"no HELD_OUT in {path}")
HELD_OUT = held_out_from(os.path.join(BASE, "pilot_hutch.py"))   # the list capture_h.py used to build held_h.pt
if HELD_OUT != held_out_from(os.path.join(BASE, "eval.py")): stop("HELD_OUT in pilot_hutch.py != HELD_OUT in eval.py")
tok = AutoTokenizer.from_pretrained(MD); bos = tok.bos_token_id
nexts = []
for text in HELD_OUT:
    ids = tok(text, truncation=True, max_length=256 - (1 if bos is not None else 0), add_special_tokens=False).input_ids
    ids = ([bos] if bos is not None else []) + ids
    for q in range(SKIP, len(ids) - 1): nexts.append(torch.tensor(ids[q + 1:q + 11]))
NPOS = len(nexts); assert NPOS == held["h_final"].shape[0] == 306, f"positions {NPOS} / held {held['h_final'].shape[0]} / expected 306"
passage = torch.tensor(held["passage"])
def logits_of(x):
    xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * g1
    return xn @ WU.T
def fut20_vec(x):   # per-position fut20 (eval.py: share of the next <=10 actual tokens in the lens top-20)
    t20 = logits_of(x).topk(20, -1).indices
    return torch.tensor([float(torch.isin(n, t20[i]).float().mean()) for i, n in enumerate(nexts)])
LP_FINAL = torch.log_softmax(logits_of(held["h_final"]), -1); TOP1_FINAL = LP_FINAL.argmax(-1)
def table_row(x):   # report-only fut20 / top1 / KL(P_final || P_lens), eval.py's definitions
    lg = logits_of(x); lp = torch.log_softmax(lg, -1)
    kl = float((LP_FINAL.exp() * (LP_FINAL - lp)).sum(-1).mean()); top1 = float((lg.argmax(-1) == TOP1_FINAL).float().mean())
    t20 = lg.topk(20, -1).indices; f = float(sum(float(torch.isin(n, t20[i]).float().mean()) for i, n in enumerate(nexts)) / NPOS)
    return {"fut20": f, "top1": top1, "kl": kl}
# ---- 1 disjointness ---------------------------------------------------------------------------------------------
corpus = json.load(open(os.path.join(BASE, "stage1_corpus.json")))
norm_t = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
def shingles(s, k=12):
    w = norm_t(s).split(); return {" ".join(w[i:i + k]) for i in range(max(0, len(w) - k + 1))}
held_sh = set().union(*[shingles(t) for t in HELD_OUT]); held_hash = {hashlib.sha256(norm_t(t).encode()).hexdigest() for t in HELD_OUT}
overlap = [i + 1 for i, d in enumerate(corpus) if hashlib.sha256(norm_t(d["text"]).encode()).hexdigest() in held_hash or shingles(d["text"]) & held_sh]
res["disjoint"] = {"stage1_docs": len(corpus), "held_out": len(HELD_OUT), "overlapping_docs": overlap}
if overlap: stop(f"Stage 1 docs overlap the held-out passages: {overlap}")
log("disjoint OK")
# ---- 2 resume (b)+(c): 16 docs, n_valid > 0, md5s, acc == sum n_p*J_p ---------------------------------------------
rec_log = open("$JLENS_HOME/whole_model_fit_record.log").read()           # step 3's record half, needed to pool while streaming
parts = [int(x) for x in re.findall(r"n_valid=(\d+)", rec_log)]; N_REC = sum(parts)
if parts != N_REC_PARTS or N_REC != 642: stop(f"N_rec parts {parts} != {N_REC_PARTS}")
J6 = {l: torch.load(os.path.join(JD, f"J_l_{l}.pt")).float() for l in LAYERS}
ma = mem_avail_gib(); res["memory"].update({"mem_available_before_Jp_gib": round(ma, 1), "rss_peak_gib_before_Jp": peak_gib()})   # report-only
docs = sorted(d for d in os.listdir(S1) if re.fullmatch(r"doc_\d\d", d))
if docs != [f"doc_{i:02d}" for i in range(1, 17)]: stop(f"Stage 1 doc dirs are not exactly doc_01..doc_16: {docs}")
meta = [json.load(open(os.path.join(S1, d, "meta.json"))) for d in docs]
n = [int(m["n_valid"]) for m in meta]
if [m["doc"] for m in meta] != list(range(1, 17)): stop("meta doc numbers are not 1..16 in order")
if min(n) <= 0: stop(f"a doc has n_valid <= 0: {n}")
res["resume_b_c"] = {"n_valid": n, "doc_genre": [m["genre"] for m in meta], "doc_ts": [m["ts"] for m in meta]}
res["N"] = {"record_parts_from_log": parts, "N_rec": N_REC, "sum_n_p": sum(n), "total": N_REC + sum(n), "new_share": round(sum(n) / (N_REC + sum(n)), 4),
            "note": "post P090 swap 1: sum n_p is REPORTED, not asserted (3824 = 16 x 239 assumes every doc fills 256 tokens; a shorter doc is legal). The acc tot == sum n_p stop stays."}
log("N OK", res["N"])
# STREAMED (memory, post P090 swap 4): each layer's 16 J_p are read once, checked against acc, folded into the pooled maps, then freed.
# Only --dose (its own run) keeps every J_p resident, because it re-pools 750 subsets.
gsp = torch.Generator().manual_seed(SEED + 2); perm = torch.randperm(16, generator=gsp).tolist(); A, Bh = sorted(perm[:8]), sorted(perm[8:])
J22, Jeq, ha, hb, Jp = {}, {}, {}, {}, {}
for i, d in enumerate(docs):
    for l in SAVE_L:
        p = os.path.join(S1, d, f"J_p_{l}.pt"); h = md5(p); res["files"][f"stage1/{d}/J_p_{l}.pt"] = h
        if h != meta[i]["J_p_md5"][str(l)]: stop(f"{d} J_p_{l} md5 {h[:8]} != meta {meta[i]['J_p_md5'][str(l)][:8]}")
acc_ok = {}
for l in SAVE_L:
    ap = os.path.join(S1, f"acc_{l}.pt"); res["files"][f"stage1/acc_{l}.pt"] = md5(ap)
    s, tot = torch.load(ap)
    Jl = [torch.load(os.path.join(S1, d, f"J_p_{l}.pt")).float() for d in docs]
    want = sum(nn * j for nn, j in zip(n, Jl)); rel = float((s.float() - want).norm() / want.norm())
    acc_ok[l] = {"tot": int(tot), "sum_n": sum(n), "rel_diff_acc_vs_sum_nJ": rel}
    if int(tot) != sum(n): stop(f"acc_{l} total {tot} != sum n_valid {sum(n)} (a doc double-counted or missing)")
    if rel > 1e-5: stop(f"acc_{l} != sum n_p*J_p (rel {rel:.2e})")
    J22[l] = (N_REC * J6[l] + want) / (N_REC + sum(n))                              # post P089 Q1
    Jeq[l] = (6 * J6[l] + sum(Jl)) / (6 + len(Jl))                                   # report-only equal-weight
    if l in (24, 40, 52):                                                            # report-only 8-vs-8 (Stage 1 docs only)
        ha[l] = sum(n[i] * Jl[i] for i in A) / sum(n[i] for i in A); hb[l] = sum(n[i] * Jl[i] for i in Bh) / sum(n[i] for i in Bh)
    if DOSE: Jp[l] = Jl
    del Jl, want, s; res["memory"][f"rss_after_layer_{l}"] = peak_gib()
res["resume_b_c"]["acc_check"] = acc_ok; log("resume (b)+(c) OK")
def pool(sel, weights=None):   # J_n over the record + the selected Stage 1 docs, n_valid-weighted (or equal-weight if weights == "equal")
    out = {}
    for l in SAVE_L:
        if weights == "equal": out[l] = (6 * J6[l] + sum(Jp[l][i] for i in sel)) / (6 + len(sel))
        else: out[l] = (N_REC * J6[l] + sum(n[i] * Jp[l][i] for i in sel)) / (N_REC + sum(n[i] for i in sel))
    return out
# ---- 4 positive control -----------------------------------------------------------------------------------------
pc = {}
for l in LAYERS:
    h = held["layers"][l]; fj = float(fut20_vec(h @ J6[l].T).mean()); fl = float(fut20_vec(h).mean())
    d = max(abs(fj - ref[str(l)]["jlens"]["fut20"]), abs(fl - ref[str(l)]["logit"]["fut20"])); pc[l] = {"jlens": fj, "logit": fl, "max_abs_diff": d}
res["positive_control"] = pc
bad = [l for l in LAYERS if pc[l]["max_abs_diff"] >= 5e-6]
if bad: stop(f"positive control: J_6/logit fut20 does not reproduce eval_record.json at layers {bad}")
log("positive control OK (J_6 reproduces eval_record.json)")
# ---- 5 GATE -----------------------------------------------------------------------------------------------------
FL = {l: fut20_vec(held["layers"][l]) for l in GATE_L}
def gvec(Jn): return torch.stack([fut20_vec(held["layers"][l] @ Jn[l].T) - FL[l] for l in GATE_L]).mean(0)
g6, g22 = gvec(J6), gvec(J22); D = g22 - g6
gen = torch.Generator().manual_seed(SEED)
bs = torch.stack([D[torch.randint(0, NPOS, (NPOS,), generator=gen)].mean() for _ in range(B)])
lo, hi = float(bs.quantile(0.025)), float(bs.quantile(0.975))
res["gate"] = {"bar": "post P047 s3: paired bootstrap 95% CI (306 positions) of [mid-band fut20 gain over logit at n=22] - [same at n=6] entirely > 0; mid-band = L20-44",
               "mean_D": float(D.mean()), "ci95": [lo, hi], "B": B, "seed": SEED, "mean_gain_n6": float(g6.mean()), "mean_gain_n22": float(g22.mean()),
               "decision": "CONTINUE" if lo > 0 else "STOP"}
res["verdict"] = res["gate"]["decision"]; dump(); log("GATE:", res["gate"]["decision"], f"mean D {D.mean():.5f} CI [{lo:.5f}, {hi:.5f}]")
# ---- 6 REPORT-ONLY ----------------------------------------------------------------------------------------------
rep = {"heading": "REPORT-ONLY. Nothing below gates anything (post P047 s4, post P089)."}
cl = torch.stack([D[passage == c].mean() for c in range(6)]); gen2 = torch.Generator().manual_seed(SEED + 1)
cbs = torch.stack([cl[torch.randint(0, 6, (6,), generator=gen2)].mean() for _ in range(B)])
rep["cluster_ci_6_clusters_crude"] = {"label": "6 clusters, crude", "per_passage_mean_D": cl.tolist(), "ci95": [float(cbs.quantile(.025)), float(cbs.quantile(.975))]}
gperm = torch.Generator().manual_seed(20261104); permJ22 = {l: J22[l][torch.randperm(5120, generator=gperm)] for l in SAVE_L}
tab = {"heading": "not a J1 verdict (post P089 Q3): per-layer readability at n=22, REPORT-ONLY", "rows": {}}
for l in SAVE_L:
    h = held["layers"][l]
    tab["rows"][l] = {"J22": table_row(h @ J22[l].T), "J6": table_row(h @ J6[l].T), "logit": table_row(h), "rowperm_J22": table_row(h @ permJ22[l].T)}
pre = {l: tab["rows"][l]["rowperm_J22"]["fut20"] for l in SAVE_L}
tab["rowperm_must_fail_precheck"] = {"rule": "fut20 < .01 AND below logit at every scored layer", "fut20": pre,
                                     "valid": all(pre[l] < .01 and pre[l] < tab["rows"][l]["logit"]["fut20"] for l in SAVE_L)}
rep["per_layer_table"] = tab
geq = gvec(Jeq); De = geq - g6
rep["equal_weight_per_doc"] = {"note": "J_6 enters as 6 equal docs (only its n-weighted mean exists)", "mean_D": float(De.mean())}
agr = {}
for l in (24, 40, 52):
    la, lb = logits_of(held["layers"][l] @ ha[l].T), logits_of(held["layers"][l] @ hb[l].T)
    t10a, t10b = la.topk(10, -1).indices, lb.topk(10, -1).indices
    agr[l] = {"top1_agree": float((la.argmax(-1) == lb.argmax(-1)).float().mean()),
              "top10_overlap": float(sum(len(set(t10a[i].tolist()) & set(t10b[i].tolist())) for i in range(NPOS)) / (10 * NPOS))}
rep["eight_vs_eight"] = {"split_seed": SEED + 2, "half_A_docs": [i + 1 for i in A], "half_B_docs": [i + 1 for i in Bh], "agreement": agr}
rep["limit_86pct_new_positions"] = (f"{sum(n)} of {N_REC + sum(n)} positions ({100 * sum(n) / (N_REC + sum(n)):.1f}%) are new, so a CONTINUE "
                                    "supports 'more data helps', and cannot separate more DOCUMENTS (diversity) from more TEXT (volume) (post P056).")
res["memory"]["rss_peak_gib"] = peak_gib(); res["report"] = rep; res["minutes"] = round((time.time() - T0) / 60, 1); res["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S"); dump()
log("report-only done; verdict", res["verdict"])
# ---- --dose (its own run) ---------------------------------------------------------------------------------------
if "--dose" in sys.argv:
    gd = torch.Generator().manual_seed(SEED + 3); dose = {0: [float(g6.mean())], 16: [float(g22.mean())]}; per = {}
    for k in range(1, 16):
        dose[k] = []
        for _ in range(50):
            sel = torch.randperm(16, generator=gd)[:k].tolist(); dose[k].append(float(gvec(pool(sel)).mean()))
        log(f"dose k={k}: mean gain {sum(dose[k]) / 50:.5f}")
    xs = torch.tensor([float(torch.log(torch.tensor(6.0 + k))) for k in range(17) for _ in dose[k]])
    ys = torch.tensor([y for k in range(17) for y in dose[k]])
    slope = float(((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()) ** 2).sum())
    gb = torch.Generator().manual_seed(SEED + 4); sl = []
    for _ in range(2000):
        ii = torch.randint(0, len(xs), (len(xs),), generator=gb); x, y = xs[ii], ys[ii]
        sl.append(float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()))
    sl = torch.tensor(sl)
    res["report"]["dose_response"] = {"n": [6 + k for k in range(17)], "mean_gain": [sum(dose[k]) / len(dose[k]) for k in range(17)],
                                      "slope_vs_log_n": slope, "slope_ci95": [float(sl.quantile(.025)), float(sl.quantile(.975))], "subsets_per_k": 50}
    res["memory"]["rss_peak_gib_dose"] = peak_gib(); res["minutes"] = round((time.time() - T0) / 60, 1); dump(); log("dose done, slope", slope)
