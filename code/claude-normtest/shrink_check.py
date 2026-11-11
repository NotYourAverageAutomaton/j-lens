"""J-Lens shrink/denoise CHEAP CHECK, CPU only (Claude 2026-11-07; pre-reg Deron P101, PREREG-shrink-check.md md5 0466a966, as amended by
PREREG-amend-1.md md5 e088c347 (Deron P103: 19-permutation rank null, max over S/T, VOID dropped); card C01).
Derived from score_stage1_DOSE.py f2409a09 (its pins, disjointness, resume (b)+(c), pooling and CPU fut20 path, unchanged maths).
No model forward, no GPU, no window. Write J = I + D. Candidates, fixed before data:
  S (shrink):   J_a = I + a*D22,          a in {0, .05, .1, .2, .3, .5, .7, 1}
  T (truncate): J_k = I + top-k SVD(D22), k in {32, 128, 512, 2048}
ORDER. Every step before the verdict can only STOP (verdict NO-SCORE):
  0 memory guard at start; pins (as f2409a09, plus score_stage1.json 67449876 for positive control 2); md5 of every file read.
  1 disjoint (as f2409a09).
  2 resume (b)+(c) on the 7 mid-band layers it reads: 16 meta.json, n_valid > 0, J_p md5 == meta, acc == sum n_p*J_p, tot == sum n_p.
  3 N_rec == 642 from the record log.
  4 positive controls: (a) J_6 and logit reproduce eval_record.json at all 13 layers (< 5e-6); (b) S at a=1 reproduces
    score_stage1.json's J22 per-layer fut20 at L20-44 (|diff| <= 1e-6); (c) a=0 is the logit lens, bit-equal per position;
    (d) T at k=5120 rebuilds J22 at L20 (rel. Frobenius error <= 1e-4) and its fut20 matches a=1 (|mean diff| <= 2e-3).
  5 tables: per-position fut20 for every candidate at every mid-band layer, for D22, for the NULL = K=19 row-permutations of D22
    (draw i uses its own generator seeded 20261104+i, one randperm(5120) per layer in L20..44 order), and D6 (report-only).
    Mid-run memory guard before every SVD: stop if MemAvailable < 15 GiB.
  6 cross-fit by passage: fold A = passages 0-2, fold B = passages 3-5. Per layer and family, the grid value with the highest
    fold-A mean fut20 scores fold B, and the reverse (ties: the first in grid order). g(q) = mean over L20..44 by 4 of
    [fut20_cand(q,L) - fut20_logit(q,L)]. Paired bootstrap over the 306 positions, B = 10000, seed 20261107; one shared index
    draw for every family and draw. CI percentiles 2.5 / 97.5 (amend-1: the max over families replaces Bonferroni).
  7 self-gating dry runs, on these same tables, before the real verdict: FAIL-WAY (every grid {0}, real and null) must read
    KILLED; PASS-WAY (a planted oracle candidate z = h_final added to the real S, against the real 19 null draws) must read
    SURVIVE. Either wrong: STOP.
  8 VERDICT at n=22 (amend-1): M = max(mean g_S, mean g_T), for the real D22 and each of the 19 null draws. SURVIVE iff
    M_real > every M_i (strict; permutation p < 1/20) AND the family attaining M_real (ties: S) has its CI entirely > 0;
    otherwise KILLED. That family is the step-2 family.
  9 REPORT-ONLY: all 19 M_i with per-family means, the rank of M_real, the old CI-only bar's false-pass count among the 19
    (S or T with CI 1.25/98.75 entirely > 0); chosen a/k per layer per fold; in-sample best (labelled in-sample); frozen step-2
    parameters for the real D22 AND for each null draw (in-sample best of the step-2 family over all 306, for amend-1 step 2);
    n=6 the same way; whether the chosen a rises from 6 to 22; 6-passage cluster CI (crude); fut20/top-1/KL for every real
    candidate at every mid-band layer.
Usage: python shrink_check.py [--selftest [SEED]]
  --selftest (amend-1): D22 is replaced by seeded Gaussian noise, norm-matched per layer to D6 = J6 - I (default seed 1), and goes
  through the full rank rule against 19 permutations of itself. It should read KILLED (1-in-20 false SURVIVE: re-seed once and
  say so; not a gate). Steps 2-3, control 4b and the n=6 report are skipped; Stage 1 J_p and J22 are never read.
  Writes score_shrink_selftest.json."""
import ast, hashlib, json, os, re, resource, sys, time, platform, torch
from safetensors import safe_open
from transformers import AutoTokenizer
torch.set_num_threads(int(os.environ.get("SCORE_THREADS", max(1, os.cpu_count() - 4))))
BASE = "$JLENS_HOME/claude-normtest"; JD = "$JLENS_HOME/results-whole-swift-record"; MD = "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b"
S1 = os.path.join(BASE, "stage1"); SELF = "--selftest" in sys.argv
OUTP = os.path.join(BASE, "score_shrink" + ("_selftest" if SELF else "") + ".json")
LAYERS = [16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 62]; GATE_L = [20, 24, 28, 32, 36, 40, 44]; SAVE_L = [20, 24, 28, 32, 36, 40, 44, 48, 52, 56]
EPS = 1e-6; SKIP = 16; B = 10000; SEED = 20261107; PERM_SEED = 20261104; N_REC_PARTS = [131, 107, 96, 96, 93, 119]; DIM = 5120
ALPHAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]; KS = [32, 128, 512, 2048]; FOLD_A = [0, 1, 2]; FOLD_B = [3, 4, 5]; QLO, QHI = 0.025, 0.975
K_NULL = 19; OLD_QLO = 0.0125   # amend-1; OLD_QLO only for the report-only false-pass count of the old bar
PIN = {"held_h.pt": "3ca0f99c", "eval.py": "cabcfb79", "stage1_fit.py": "fabece82", "pilot_hutch.py": "5fa18d9b",
       "$JLENS_HOME/whole_model_fit_record.log": "d0cf6a41", "score_stage1.json": "67449876"}
DRY_PEAK_GIB = 17.0; HEADROOM_GIB = 20.0; MID_MIN_GIB = 15.0   # selftest 98f5aa28 measured RSS peak 15.86 GiB, rounded up
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
peak_gib = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
def mem_avail_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"): return int(line.split()[1]) / 2**20
SELF_SEED = int(sys.argv[sys.argv.index("--selftest") + 1]) if SELF and len(sys.argv) > sys.argv.index("--selftest") + 1 else 1
res = {"script_md5": md5(__file__), "selftest": SELF, "prereg": "PREREG-shrink-check.md 0466a966 (Deron P101) + PREREG-amend-1.md e088c347 (Deron P103)", "verdict": "INCOMPLETE",
       "stops": [], "pins": {}, "files": {},
       "env": {"torch": torch.__version__, "python": platform.python_version(), "threads": torch.get_num_threads(), "host": platform.node(),
               "transformers": __import__("transformers").__version__, "note": "CPU-only run; GPU not used"}}
def dump(): json.dump(res, open(OUTP, "w"), indent=1)
def stop(why):
    res["stops"].append(why); res["verdict"] = "NO-SCORE: " + why; dump(); log("STOP:", why); sys.exit(3)
def mid_guard(where):
    ma = mem_avail_gib()
    if ma < MID_MIN_GIB: stop(f"mid-run memory guard at {where}: MemAvailable {ma:.1f} GiB < {MID_MIN_GIB} (live engine protected)")
# ---- 0 memory guard FIRST, then pins ----------------------------------------------------------------------------
NEED = DRY_PEAK_GIB + HEADROOM_GIB
_ma0 = mem_avail_gib(); res["memory"] = {"mem_available_at_start_gib": round(_ma0, 1), "required_gib": NEED}
if _ma0 < NEED: stop(f"memory guard: MemAvailable {_ma0:.1f} GiB at start < measured peak {DRY_PEAK_GIB} + {HEADROOM_GIB} GiB (live engine protected)")
for name, want in PIN.items():
    got = md5(name if name.startswith("/") else os.path.join(BASE, name)); res["pins"][name] = got
    if not got.startswith(want): stop(f"pin {name} {got[:8]} != {want}")
for l in LAYERS: res["files"][f"record/J_l_{l}.pt"] = md5(os.path.join(JD, f"J_l_{l}.pt"))
log("pins OK")
# ---- shared CPU fut20 path (f2409a09, unchanged maths) ----------------------------------------------------------
idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
def W(n):
    with safe_open(os.path.join(MD, idx[n]), "pt") as f: return f.get_tensor(n).float()
WU = W("lm_head.weight"); g1 = 1.0 + W("model.language_model.norm.weight")
held = torch.load(os.path.join(BASE, "held_h.pt")); ref = json.load(open(os.path.join(BASE, "eval_record.json")))["layers"]
S1REF = json.load(open(os.path.join(BASE, "score_stage1.json")))["report"]["per_layer_table"]["rows"]
def held_out_from(path):   # ast.literal_eval of the module's HELD_OUT assignment, never eval()
    tree = ast.parse(open(path).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "HELD_OUT" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit(f"no HELD_OUT in {path}")
HELD_OUT = held_out_from(os.path.join(BASE, "pilot_hutch.py"))
if HELD_OUT != held_out_from(os.path.join(BASE, "eval.py")): stop("HELD_OUT in pilot_hutch.py != HELD_OUT in eval.py")
tok = AutoTokenizer.from_pretrained(MD); bos = tok.bos_token_id
nexts = []
for text in HELD_OUT:
    ids = tok(text, truncation=True, max_length=256 - (1 if bos is not None else 0), add_special_tokens=False).input_ids
    ids = ([bos] if bos is not None else []) + ids
    for q in range(SKIP, len(ids) - 1): nexts.append(torch.tensor(ids[q + 1:q + 11]))
NPOS = len(nexts); assert NPOS == held["h_final"].shape[0] == 306, f"positions {NPOS} / held {held['h_final'].shape[0]} / expected 306"
passage = torch.tensor(held["passage"])
if sorted(set(passage.tolist())) != [0, 1, 2, 3, 4, 5]: stop(f"held['passage'] ids are not 0..5: {sorted(set(passage.tolist()))}")
inA = torch.tensor([int(p) in FOLD_A for p in passage.tolist()]); inB = ~inA
res["folds"] = {"A_passages": FOLD_A, "B_passages": FOLD_B, "A_positions": int(inA.sum()), "B_positions": int(inB.sum())}
def logits_of(x):
    xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * g1
    return xn @ WU.T
LP_FINAL = torch.log_softmax(logits_of(held["h_final"]), -1); TOP1_FINAL = LP_FINAL.argmax(-1)
def readout(x):   # per-position fut20 (eval.py) plus report-only mean top-1 and KL(P_final || P_lens)
    lg = logits_of(x); t20 = lg.topk(20, -1).indices
    f = torch.tensor([float(torch.isin(n, t20[i]).float().mean()) for i, n in enumerate(nexts)])
    lp = torch.log_softmax(lg, -1)
    return f, float((lg.argmax(-1) == TOP1_FINAL).float().mean()), float((LP_FINAL.exp() * (LP_FINAL - lp)).sum(-1).mean())
fut20_vec = lambda x: readout(x)[0]
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
# ---- 2+3 resume (b)+(c) and N, on the 7 mid-band layers this check reads; J22 pooled exactly as f2409a09 ----------
J6 = {l: torch.load(os.path.join(JD, f"J_l_{l}.pt")).float() for l in LAYERS}
I = torch.eye(DIM)
if SELF:
    gs = torch.Generator().manual_seed(SELF_SEED); D22, D6 = {}, None
    for l in GATE_L:
        R_ = torch.randn(DIM, DIM, generator=gs); D22[l] = R_ * ((J6[l] - I).norm() / R_.norm()); del R_
    J22 = {l: I + D22[l] for l in GATE_L}
    res["selftest_note"] = (f"amend-1 self-test: D22 = seeded Gaussian noise (seed {SELF_SEED}) norm-matched per layer to D6 = J6 - I; it should read "
                            "KILLED (1-in-20 false SURVIVE; re-seed once and say so; not a gate). Steps 2-3, control 4b and the n=6 report skipped; "
                            "Stage 1 J_p and J22 are never read.")
    log("SELFTEST: synthetic D22 / D6")
else:
    rec_log = open("$JLENS_HOME/whole_model_fit_record.log").read()
    parts = [int(x) for x in re.findall(r"n_valid=(\d+)", rec_log)]; N_REC = sum(parts)
    if parts != N_REC_PARTS or N_REC != 642: stop(f"N_rec parts {parts} != {N_REC_PARTS}")
    docs = sorted(d for d in os.listdir(S1) if re.fullmatch(r"doc_\d\d", d))
    if docs != [f"doc_{i:02d}" for i in range(1, 17)]: stop(f"Stage 1 doc dirs are not exactly doc_01..doc_16: {docs}")
    meta = [json.load(open(os.path.join(S1, d, "meta.json"))) for d in docs]
    n = [int(m["n_valid"]) for m in meta]
    if [m["doc"] for m in meta] != list(range(1, 17)): stop("meta doc numbers are not 1..16 in order")
    if min(n) <= 0: stop(f"a doc has n_valid <= 0: {n}")
    res["N"] = {"record_parts_from_log": parts, "N_rec": N_REC, "sum_n_p": sum(n), "total": N_REC + sum(n)}
    J22, acc_ok = {}, {}
    for l in GATE_L:
        for i, d in enumerate(docs):
            p = os.path.join(S1, d, f"J_p_{l}.pt"); h = md5(p); res["files"][f"stage1/{d}/J_p_{l}.pt"] = h
            if h != meta[i]["J_p_md5"][str(l)]: stop(f"{d} J_p_{l} md5 {h[:8]} != meta {meta[i]['J_p_md5'][str(l)][:8]}")
        ap = os.path.join(S1, f"acc_{l}.pt"); res["files"][f"stage1/acc_{l}.pt"] = md5(ap)
        s, tot = torch.load(ap)
        Jl = [torch.load(os.path.join(S1, d, f"J_p_{l}.pt")).float() for d in docs]
        want = sum(nn * j for nn, j in zip(n, Jl)); rel = float((s.float() - want).norm() / want.norm())
        acc_ok[l] = {"tot": int(tot), "sum_n": sum(n), "rel_diff_acc_vs_sum_nJ": rel}
        if int(tot) != sum(n): stop(f"acc_{l} total {tot} != sum n_valid {sum(n)} (a doc double-counted or missing)")
        if rel > 1e-5: stop(f"acc_{l} != sum n_p*J_p (rel {rel:.2e})")
        J22[l] = (N_REC * J6[l] + want) / (N_REC + sum(n))
        del Jl, want, s; res["memory"][f"rss_after_layer_{l}"] = peak_gib()
    res["resume_b_c"] = {"n_valid": n, "acc_check": acc_ok}
    D22 = {l: J22[l] - I for l in GATE_L}; D6 = {l: J6[l] - I for l in GATE_L}
    log("resume (b)+(c) OK; J22 pooled")
# ---- 4 positive controls (a)-(c) --------------------------------------------------------------------------------
pc = {}
for l in LAYERS:
    h = held["layers"][l]; fj = float(fut20_vec(h @ J6[l].T).mean()); fl = float(fut20_vec(h).mean())
    pc[l] = {"jlens": fj, "logit": fl, "max_abs_diff": max(abs(fj - ref[str(l)]["jlens"]["fut20"]), abs(fl - ref[str(l)]["logit"]["fut20"]))}
res["positive_control"] = {"a_eval_record": pc}
bad = [l for l in LAYERS if pc[l]["max_abs_diff"] >= 5e-6]
if bad: stop(f"positive control (a): J_6/logit fut20 does not reproduce eval_record.json at layers {bad}")
for l in [l for l in LAYERS if l not in GATE_L]: del J6[l]
log("positive control (a) OK")
FL = {l: fut20_vec(held["layers"][l]) for l in GATE_L}
def family_tables(D, tag, perm_gen=None):
    """Per-position fut20 for every S and T candidate at every mid-band layer, plus report-only mean top-1/KL. One SVD per layer."""
    S, T, rows = {}, {}, {}
    for l in GATE_L:
        Dl = D[l][torch.randperm(DIM, generator=perm_gen)] if perm_gen is not None else D[l]
        h = held["layers"][l]; hD = h @ Dl.T; rows[l] = {}
        S[l] = []
        for a in ALPHAS:
            f, t1, kl = readout(h + a * hD); S[l].append(f); rows[l][f"a={a}"] = {"fut20": float(f.mean()), "top1": t1, "kl": kl}
        mid_guard(f"{tag} SVD L{l}")
        U, sv, Vh = torch.linalg.svd(Dl, full_matrices=False)
        T[l] = []
        for k in KS:
            f, t1, kl = readout(h + ((h @ Vh[:k].T) * sv[:k]) @ U[:, :k].T); T[l].append(f)
            rows[l][f"k={k}"] = {"fut20": float(f.mean()), "top1": t1, "kl": kl}
        if tag == "n22" and l == 20: svd20 = (U, sv, Vh)
        del U, sv, Vh, hD, Dl; res["memory"][f"rss_{tag}_L{l}"] = peak_gib()
        log(f"{tag} L{l} tables done")
    return S, T, rows, (svd20 if tag == "n22" else None)
S22, T22, rows22, svd20 = family_tables(D22, "n22")
# (b) a=1 reproduces score_stage1.json's J22; (c) a=0 is the logit lens, bit-equal; (d) k=5120 rebuilds J22 at L20
if not SELF:
    pb = {l: {"a1": float(S22[l][ALPHAS.index(1.0)].mean()), "score_stage1_J22": S1REF[str(l)]["J22"]["fut20"]} for l in GATE_L}
    for l in GATE_L: pb[l]["abs_diff"] = abs(pb[l]["a1"] - pb[l]["score_stage1_J22"])
    res["positive_control"]["b_alpha1_vs_score_stage1"] = pb
    if any(pb[l]["abs_diff"] > 1e-6 for l in GATE_L): stop(f"positive control (b): a=1 does not reproduce score_stage1.json J22 fut20: {pb}")
pcc = {l: bool(torch.equal(S22[l][0], FL[l])) for l in GATE_L}; res["positive_control"]["c_alpha0_is_logit_bit_equal"] = pcc
if not all(pcc.values()): stop(f"positive control (c): a=0 is not bit-equal to the logit lens: {pcc}")
U, sv, Vh = svd20; Jrec = I + (U * sv) @ Vh; relrec = float((Jrec - J22[20]).norm() / J22[20].norm())
f5120 = fut20_vec(held["layers"][20] @ Jrec.T); dmean = abs(float(f5120.mean()) - float(S22[20][ALPHAS.index(1.0)].mean()))
res["positive_control"]["d_k5120_L20"] = {"rel_frobenius_err": relrec, "abs_mean_fut20_diff_vs_a1": dmean, "bars": "rel <= 1e-4, diff <= 2e-3"}
if relrec > 1e-4 or dmean > 2e-3: stop(f"positive control (d): k=5120 does not rebuild J22 at L20 (rel {relrec:.2e}, fut20 diff {dmean:.2e})")
del U, sv, Vh, Jrec, svd20; log("positive controls (b)-(d) OK" + (" [(b) skipped: selftest]" if SELF else ""))
NULL = []   # amend-1: K=19 row-permutations of D22; draw i has its own generator seeded 20261104+i
for i in range(1, K_NULL + 1):
    Sp, Tp, _, _ = family_tables(D22, f"null{i:02d}", perm_gen=torch.Generator().manual_seed(PERM_SEED + i))
    NULL.append({"seed": PERM_SEED + i, "S": Sp, "T": Tp})
if not SELF: S6, T6, rows6, _ = family_tables(D6, "n6")
# ---- 6 cross-fit + paired bootstrap ------------------------------------------------------------------------------
gb = torch.Generator().manual_seed(SEED); BIDX = torch.randint(0, NPOS, (B, NPOS), generator=gb)
def crossfit(tab, grid):
    """tab[l] = list of per-position fut20 vectors over the grid. Returns g(q) and the chosen grid value per layer per fold."""
    per_l, chosen = [], {}
    for l in GATE_L:
        M = torch.stack(tab[l])                                    # [grid, 306]
        iA = int(torch.argmax(M[:, inA].mean(1))); iB = int(torch.argmax(M[:, inB].mean(1)))   # argmax returns the first max: ties -> first in grid order
        cand = torch.where(inB, M[iA], M[iB])                      # fold B scored with A's pick, fold A with B's pick
        per_l.append(cand - FL[l]); chosen[l] = {"picked_on_A_scores_B": grid[iA], "picked_on_B_scores_A": grid[iB]}
    return torch.stack(per_l).mean(0), chosen
def ci(g):
    bs = g[BIDX].mean(1); return [float(bs.quantile(QLO)), float(bs.quantile(QHI))]
def verdict(gS, gT, null_M):
    """amend-1 rank rule: SURVIVE iff M_real > every M_i (strict) AND the family attaining M_real (ties: S) has CI entirely > 0."""
    mS, mT = float(gS.mean()), float(gT.mean()); fam = "S" if mS >= mT else "T"; M = max(mS, mT)
    c = ci(gS if fam == "S" else gT); beats = all(M > m for m in null_M)
    v = "SURVIVE" if beats and c[0] > 0 else "KILLED"
    return v, {"M_real": M, "family": fam, "family_ci_2.5_97.5": c, "beats_all_null": beats,
               "rank_of_M_real": 1 + sum(m >= M for m in null_M), "of": len(null_M) + 1}
gS, chS = crossfit(S22, ALPHAS); gT, chT = crossfit(T22, KS)
for d in NULL:
    d["gS"], d["chS"] = crossfit(d["S"], ALPHAS); d["gT"], d["chT"] = crossfit(d["T"], KS)
    d["mS"], d["mT"] = float(d["gS"].mean()), float(d["gT"].mean()); d["M"] = max(d["mS"], d["mT"])
NULL_M = [d["M"] for d in NULL]
# ---- 7 self-gating dry runs on these same tables -----------------------------------------------------------------
zero = {l: [FL[l]] for l in GATE_L}; gz = crossfit(zero, [0])[0]
v_fail, _ = verdict(gz, gz, [float(gz.mean())] * K_NULL)                      # every grid {0}, real and null: M_real = 0 = every M_i
orc = {l: S22[l] + [fut20_vec(held["h_final"])] for l in GATE_L}
v_pass, ci_pass = verdict(crossfit(orc, ALPHAS + ["oracle"])[0], gT, NULL_M)  # planted oracle in the real S, against the real 19 null draws
res["dry_runs"] = {"fail_way_grid0": v_fail, "pass_way_oracle_h_final": v_pass, "pass_way_ci": ci_pass,
                   "rule": "FAIL-WAY must read KILLED and PASS-WAY must read SURVIVE, on this file and these tables, before the real verdict"}
if v_fail != "KILLED" or v_pass != "SURVIVE": stop(f"self-gating dry runs: fail-way {v_fail} (want KILLED), pass-way {v_pass} (want SURVIVE)")
log("dry runs OK: fail-way KILLED, pass-way SURVIVE")
# ---- 8 VERDICT ---------------------------------------------------------------------------------------------------
v, rk = verdict(gS, gT, NULL_M)
mS, mT = float(gS.mean()), float(gT.mean()); step2 = rk["family"] if v == "SURVIVE" else None
res["check"] = {"bar": "PREREG-amend-1 e088c347: M = max(mean g_S, mean g_T); SURVIVE iff M_real > all 19 M_i (strict, p < .05 family-wise) AND the family attaining M_real has its cross-fitted CI (2.5/97.5) entirely > 0; else KILLED",
                "mean_g": {"S": mS, "T": mT}, "ci_2.5_97.5": {"S": ci(gS), "T": ci(gT)}, "rank_rule": rk,
                "null_M": NULL_M, "B": B, "seed": SEED, "null_seeds": [d["seed"] for d in NULL], "decision": v, "step2_family": step2}
res["verdict"] = ("SELFTEST " if SELF else "") + v; dump()
log("CHECK:", v, f"M_real {rk['M_real']:.5f} ({rk['family']}) rank {rk['rank_of_M_real']}/{rk['of']}", f"CI {rk['family_ci_2.5_97.5']}",
    f"null M max {max(NULL_M):.5f}", "step2:", step2)
# ---- 9 REPORT-ONLY -----------------------------------------------------------------------------------------------
rep = {"heading": "REPORT-ONLY. Nothing below gates anything (PREREG 0466a966)."}
rep["chosen_per_layer_per_fold"] = {"S": chS, "T": chT}
old_pass = lambda g: float(g[BIDX].mean(1).quantile(OLD_QLO)) > 0
rep["null_distribution"] = {"draws": [{"seed": d["seed"], "mean_g_S": d["mS"], "mean_g_T": d["mT"], "M": d["M"],
                                        "old_bar_pass": old_pass(d["gS"]) or old_pass(d["gT"])} for d in NULL],
                            "rank_of_M_real": rk["rank_of_M_real"], "of": rk["of"]}
rep["null_distribution"]["old_bar_false_pass_count"] = sum(x["old_bar_pass"] for x in rep["null_distribution"]["draws"])
def insample(tab, grid):
    best = {l: grid[int(torch.argmax(torch.stack(tab[l]).mean(1)))] for l in GATE_L}
    g = torch.stack([tab[l][grid.index(best[l])] - FL[l] for l in GATE_L]).mean(0)
    return best, float(g.mean())
bS, mSi = insample(S22, ALPHAS); bT, mTi = insample(T22, KS)
rep["in_sample_best_LABELLED_IN_SAMPLE"] = {"S": {"per_layer": bS, "mean_g": mSi}, "T": {"per_layer": bT, "mean_g": mTi}}
rep["frozen_step2_parameters"] = None if step2 is None else {
    "family": step2, "real_per_layer": bS if step2 == "S" else bT,
    "null_per_layer": [{"seed": d["seed"], "per_layer": insample(d[step2], ALPHAS if step2 == "S" else KS)[0]} for d in NULL],
    "rule": "refit on all 306 positions at n=22 (PREREG step 2); the 19 null draws frozen the same way (amend-1 step 2)"}
if not SELF:
    g6S, ch6S = crossfit(S6, ALPHAS); g6T, ch6T = crossfit(T6, KS)
    rep["n6_same_way"] = {"mean_g": {"S": float(g6S.mean()), "T": float(g6T.mean())}, "ci_2.5_97.5": {"S": ci(g6S), "T": ci(g6T)},
                          "chosen": {"S": ch6S, "T": ch6T}, "in_sample_best": {"S": insample(S6, ALPHAS)[0], "T": insample(T6, KS)[0]}}
    rep["alpha_rises_6_to_22"] = {"in_sample_alpha_n6": insample(S6, ALPHAS)[0], "in_sample_alpha_n22": bS,
                                  "layers_where_alpha_rises": [l for l in GATE_L if bS[l] > insample(S6, ALPHAS)[0][l]],
                                  "note": "the noise mechanism predicts a larger chosen a at n=22 than at n=6"}
gc = torch.Generator().manual_seed(SEED + 1)
def cluster_ci(g):
    cl = torch.stack([g[passage == c].mean() for c in range(6)])
    cbs = torch.stack([cl[torch.randint(0, 6, (6,), generator=gc)].mean() for _ in range(B)])
    return {"per_passage_mean_g": cl.tolist(), "ci95": [float(cbs.quantile(.025)), float(cbs.quantile(.975))]}
rep["cluster_ci_6_passages_crude"] = {"label": "6 clusters, crude", "S": cluster_ci(gS), "T": cluster_ci(gT)}
rep["readout_table"] = {"heading": "fut20 / top-1 / KL(P_final||P_lens) per candidate per mid-band layer, REPORT-ONLY, not a J1 verdict",
                        "n22": rows22, "n6": None if SELF else rows6, "logit": {l: float(FL[l].mean()) for l in GATE_L}}
res["memory"]["rss_peak_gib"] = peak_gib(); res["report"] = rep; res["minutes"] = round((time.time() - T0) / 60, 1); res["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S"); dump()
log("report-only done; verdict", res["verdict"])
