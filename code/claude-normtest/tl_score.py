"""J-Lens TUNED-LENS baseline, SCORING on CPU (Claude 2026-11-07; pre-reg Deron PREREG-tunedlens-and-Sup.md part B + post P107; card C01).
Reads the translators written by tl_train.py and scores them on the SAME 306 held positions with the SAME CPU fut20 path.
Lens: z = h + h R^T + b  (A = I + R, identity init), read out by logits_of (final norm + unembed), as in training.
ORDER. Every step before the verdict can only STOP (NO-SCORE), except a failed training control, which is VOID:
  0-3 preamble copied VERBATIM from shrink_check.py d153df93 up to "# ---- 4 positive controls" (memory guard, pins, disjointness,
      resume (b)+(c), J22 pooled as f2409a09, CPU fut20 path). J22 is needed only for the report-only rows.
  4 tl_meta.json: verdict must be TRAINED (post P107 training control passed). If it reads VOID, this writes VOID and stops; a VOID run
    is never graded. md5 of every translator file must match tl_meta.json.
  5 dry runs (pre-reg): FAIL-WAY = the identity translator (R = 0, b = 0) must give fut20 bit-equal to the logit lens at every
    mid-band layer, so its gain is 0 and it cannot pass; PASS-WAY = TL-big's output replaced by the oracle h_final must PASS.
  6 VERDICT: g_logit(q) = mean over L20..44 by 4 of [fut20_TLbig(q,L) - fut20_logit(q,L)]; g_shuf(q) = same against TL-shuf.
    Paired bootstrap over the 306 positions, B = 10000, seed 20261109, one shared index draw; CI 2.5/97.5.
    PASS iff BOTH CIs are entirely > 0. Otherwise FAIL.
  7 REPORT-ONLY: TL-22 the same bars (labelled); 6-passage cluster CI (crude); per-layer fut20 / top-1 / KL for TL-big, TL-shuf,
    TL-22, J22, logit, and the unigram top-20 (the 20 most frequent training tokens as a fixed top-20; KL not defined for it);
    ||R_l||_F vs ||D22_l||_F; agreement of trained and Jacobian corrections: |cos| of the top-1 left singular vectors of R_l and
    D22_l, and the top-16 subspace overlap ||U_R^T U_D||_F^2 / 16.
Usage: python tl_score.py [--dir tl_out_lr1]      writes <dir>/score_tl.json"""
import ast, hashlib, json, os, re, resource, sys, time, platform, torch
from safetensors import safe_open
from transformers import AutoTokenizer
torch.set_num_threads(int(os.environ.get("SCORE_THREADS", max(1, os.cpu_count() - 4))))
BASE = "$JLENS_HOME/claude-normtest"; JD = "$JLENS_HOME/results-whole-swift-record"; MD = "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b"
S1 = os.path.join(BASE, "stage1"); SELF = "--selftest" in sys.argv
TLDIR = os.path.join(BASE, sys.argv[sys.argv.index("--dir") + 1]) if "--dir" in sys.argv else os.path.join(BASE, "tl_out_lr1")
OUTP = os.path.join(TLDIR, "score_tl.json")
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
# ---- 4 training run: control verdict + md5s ---------------------------------------------------------------------
if SELF: stop("tl_score.py has no --selftest")
TM = json.load(open(os.path.join(TLDIR, "tl_meta.json"))); res["tl_meta"] = {k: TM.get(k) for k in ("script_md5", "lr", "verdict", "training_control", "corpus", "tl22_positions", "unigram_top20", "minutes")}
if TM.get("verdict") != "TRAINED":
    res["verdict"] = "VOID: training control failed or run incomplete (post P107); not graded"; dump(); log(res["verdict"]); sys.exit(5)
LENS = {}
for name in ("big", "shuf", "22"):
    for l in GATE_L:
        p = os.path.join(TLDIR, f"{name}_L{l}.pt"); h = md5(p); res["files"][f"tl/{name}_L{l}.pt"] = h
        if h != TM["files"][f"{name}_L{l}.pt"]: stop(f"{name}_L{l}.pt md5 {h[:8]} != tl_meta {TM['files'][f'{name}_L{l}.pt'][:8]}")
        LENS[(name, l)] = torch.load(p)
log("translators loaded; md5s match tl_meta.json")
apply = lambda name, l, h: h + h @ LENS[(name, l)]["R"].T + LENS[(name, l)]["b"]
FL = {l: fut20_vec(held["layers"][l]) for l in GATE_L}
F = {name: {} for name in ("big", "shuf", "22")}; rows = {}
for l in GATE_L:
    h = held["layers"][l]; rows[l] = {}
    for name in F:
        f, t1, kl = readout(apply(name, l, h)); F[name][l] = f; rows[l][f"TL-{name}"] = {"fut20": float(f.mean()), "top1": t1, "kl": kl}
    f, t1, kl = readout(h @ J22[l].T); rows[l]["J22"] = {"fut20": float(f.mean()), "top1": t1, "kl": kl}
    f, t1, kl = readout(h); rows[l]["logit"] = {"fut20": float(f.mean()), "top1": t1, "kl": kl}
    log(f"L{l} scored")
# ---- 5 dry runs ---------------------------------------------------------------------------------------------------------
Z = torch.zeros(DIM, DIM); z = torch.zeros(DIM)
ident = {l: bool(torch.equal(fut20_vec(held["layers"][l] + held["layers"][l] @ Z.T + z), FL[l])) for l in GATE_L}
gb = torch.Generator().manual_seed(20261109); BIDX = torch.randint(0, NPOS, (B, NPOS), generator=gb)
ci = lambda g: [float(g[BIDX].mean(1).quantile(QLO)), float(g[BIDX].mean(1).quantile(QHI))]
gain = lambda Fa, Fb: torch.stack([Fa[l] - Fb[l] for l in GATE_L]).mean(0)
def verdict(Fbig):
    a, b = ci(gain(Fbig, FL)), ci(gain(Fbig, F["shuf"])); return ("PASS" if a[0] > 0 and b[0] > 0 else "FAIL"), a, b
v_fail = verdict({l: fut20_vec(held["layers"][l] + held["layers"][l] @ Z.T + z) for l in GATE_L})[0]
v_pass = verdict({l: fut20_vec(held["h_final"]) for l in GATE_L})[0]
res["dry_runs"] = {"identity_bit_equal_logit": ident, "fail_way_identity": v_fail, "pass_way_oracle_h_final": v_pass}
if not all(ident.values()) or v_fail != "FAIL" or v_pass != "PASS": stop(f"dry runs: identity bit-equal {ident}, fail-way {v_fail} (want FAIL), pass-way {v_pass} (want PASS)")
log("dry runs OK")
# ---- 6 VERDICT -------------------------------------------------------------------------------------------------------------
v, ci_l, ci_s = verdict(F["big"])
res["check"] = {"bar": "PREREG-tunedlens part B: PASS iff TL-big's mid-band fut20 gain has a paired-bootstrap 95% CI (B=10000, seed 20261109) entirely > 0 over the logit lens AND over the shuffled-target lens",
                "mean_gain_vs_logit": float(gain(F["big"], FL).mean()), "ci_vs_logit": ci_l,
                "mean_gain_vs_shuf": float(gain(F["big"], F["shuf"]).mean()), "ci_vs_shuf": ci_s, "decision": v, "lr": TM["lr"]}
res["verdict"] = v; dump(); log("TL CHECK:", v, "vs logit", res["check"]["mean_gain_vs_logit"], ci_l, "| vs shuf", res["check"]["mean_gain_vs_shuf"], ci_s)
# ---- 7 REPORT-ONLY ---------------------------------------------------------------------------------------------------------
rep = {"heading": "REPORT-ONLY. Nothing below gates anything (PREREG-tunedlens part B)."}
v22, a22, b22 = verdict(F["22"])
rep["TL22_same_bars_LABELLED"] = {"label": "TL-22, equal data with J22, report-only", "decision_if_it_were_graded": v22,
                                  "mean_gain_vs_logit": float(gain(F["22"], FL).mean()), "ci_vs_logit": a22, "ci_vs_shuf": b22}
gc = torch.Generator().manual_seed(20261109 + 1); g_l = gain(F["big"], FL)
cl = torch.stack([g_l[passage == c].mean() for c in range(6)]); cbs = torch.stack([cl[torch.randint(0, 6, (6,), generator=gc)].mean() for _ in range(B)])
rep["cluster_ci_6_passages_crude"] = {"per_passage_mean_gain_vs_logit": cl.tolist(), "ci95": [float(cbs.quantile(.025)), float(cbs.quantile(.975))]}
U20 = torch.tensor(TM["unigram_top20"])
uni = torch.tensor([float(torch.isin(n, U20).float().mean()) for n in nexts])
rep["unigram_top20"] = {"tokens": TM["unigram_top20"], "fut20": float(uni.mean()), "note": "a fixed top-20 at every position and layer; KL undefined"}
rep["per_layer_table"] = rows
agree = {}
for l in GATE_L:
    Rm = LENS[("big", l)]["R"]; Dm = J22[l] - torch.eye(DIM)
    Ur = torch.linalg.svd(Rm, full_matrices=False)[0][:, :16]; Ud = torch.linalg.svd(Dm, full_matrices=False)[0][:, :16]
    agree[l] = {"fro_R": float(Rm.norm()), "fro_D22": float(Dm.norm()), "abs_cos_top1": float((Ur[:, 0] @ Ud[:, 0]).abs()),
                "top16_overlap": float((Ur.T @ Ud).pow(2).sum() / 16)}
    mid_guard(f"agreement L{l}")
rep["trained_vs_jacobian_correction"] = agree
res["memory"]["rss_peak_gib"] = peak_gib(); res["report"] = rep; res["minutes"] = round((time.time() - T0) / 60, 1); res["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S"); dump()
log("report-only done; verdict", res["verdict"])
