"""REPORT-ONLY S-up arm (Deron P106, PREREG-tunedlens-and-Sup.md md5 31174730; card C01). Prior art: LRE (Hernandez et al.,
ICLR 2024, arXiv 2308.09124, sec 3): the mean-Jacobian map is made "steeper by multiplying with a scalar constant beta (> 1)".
Our shrink grid only had a in [0, 1]. This asks whether SCALING D22 UP helps the mid-band.
Candidate: J_a = I + a*D22, a in {1.25, 1.5, 2, 3} (fixed before data). Same mid-band, same cross-fit by passage (fold A = passages
0-2, B = 3-5; ties to the first in grid order), same g(q), same shared bootstrap (B = 10000, seed 20261107), CI 2.5/97.5.
Null: the same 19 row-permutations of D22 as shrink_check d153df93 (draw i: generator seeded 20261104+i, one randperm(5120) per
layer in L20..44 order), through the identical cross-fit. M = mean g (one family).
LEAD iff M_real > every M_i (strict) AND its CI is entirely > 0; otherwise NO LEAD. A LEAD is grounds for its own pre-reg on fresh
text, NOT a pass (post P106). Self-gating dry runs on these tables: grid {0} must read NO LEAD; an oracle h_final planted in the real
grid must read LEAD. Report-only: chosen a per layer per fold, a=1 per-layer fut20 (consistency with score_shrink.json), null M_i.
Preamble (memory guard, pins, disjointness, resume (b)+(c), J22 pooling, CPU fut20 path) is copied VERBATIM from shrink_check.py
d153df93 up to "# ---- 4 positive controls". No SVD. Usage: python sup_check.py      writes score_shrink_sup.json"""
import ast, hashlib, json, os, re, resource, sys, time, platform, torch
from safetensors import safe_open
from transformers import AutoTokenizer
torch.set_num_threads(int(os.environ.get("SCORE_THREADS", max(1, os.cpu_count() - 4))))
BASE = "$JLENS_HOME/claude-normtest"; JD = "$JLENS_HOME/results-whole-swift-record"; MD = "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b"
S1 = os.path.join(BASE, "stage1"); SELF = "--selftest" in sys.argv
OUTP = os.path.join(BASE, "score_shrink_sup.json")
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
# ---- S-up arm (report-only) ---------------------------------------------------------------------------------------
if SELF: stop("sup_check.py has no --selftest")
UP = [1.25, 1.5, 2.0, 3.0]
FL = {l: fut20_vec(held["layers"][l]) for l in GATE_L}
def tables(perm_seed=None):
    g = torch.Generator().manual_seed(perm_seed) if perm_seed is not None else None; out = {}
    for l in GATE_L:
        Dl = D22[l][torch.randperm(DIM, generator=g)] if g is not None else D22[l]
        h = held["layers"][l]; hD = h @ Dl.T; out[l] = [fut20_vec(h + a * hD) for a in UP]; del hD, Dl
    return out
real = tables()
a1 = {l: float(fut20_vec(held["layers"][l] + held["layers"][l] @ D22[l].T).mean()) for l in GATE_L}   # report-only consistency
log("real tables done")
gb = torch.Generator().manual_seed(SEED); BIDX = torch.randint(0, NPOS, (B, NPOS), generator=gb)
def crossfit(tab, grid):
    per_l, chosen = [], {}
    for l in GATE_L:
        M = torch.stack(tab[l]); iA = int(torch.argmax(M[:, inA].mean(1))); iB = int(torch.argmax(M[:, inB].mean(1)))
        per_l.append(torch.where(inB, M[iA], M[iB]) - FL[l]); chosen[l] = {"picked_on_A_scores_B": grid[iA], "picked_on_B_scores_A": grid[iB]}
    return torch.stack(per_l).mean(0), chosen
ci = lambda g: [float(g[BIDX].mean(1).quantile(QLO)), float(g[BIDX].mean(1).quantile(QHI))]
NULL = []
for i in range(1, K_NULL + 1):
    mid_guard(f"null draw {i}")
    gi, _ = crossfit(tables(PERM_SEED + i), UP); NULL.append({"seed": PERM_SEED + i, "M": float(gi.mean())}); log(f"null {i:02d} M {NULL[-1]['M']:.5f}")
NULL_M = [d["M"] for d in NULL]
def lead(g, null_M):
    M = float(g.mean()); c = ci(g); return ("LEAD" if all(M > m for m in null_M) and c[0] > 0 else "NO LEAD"), M, c
gz = crossfit({l: [FL[l]] for l in GATE_L}, [0])[0]
v_fail = lead(gz, [0.0] * K_NULL)[0]
v_pass = lead(crossfit({l: real[l] + [fut20_vec(held["h_final"])] for l in GATE_L}, UP + ["oracle"])[0], NULL_M)[0]
res["dry_runs"] = {"fail_way_grid0": v_fail, "pass_way_oracle_h_final": v_pass}
if v_fail != "NO LEAD" or v_pass != "LEAD": stop(f"self-gating dry runs: fail-way {v_fail} (want NO LEAD), pass-way {v_pass} (want LEAD)")
log("dry runs OK")
g, ch = crossfit(real, UP); v, M, c = lead(g, NULL_M)
res["report_only"] = {"heading": "REPORT-ONLY (Deron P106). A LEAD is grounds for its own pre-reg on fresh text, not a pass.",
                      "decision": v, "M_real": M, "ci_2.5_97.5": c, "rank_of_M_real": 1 + sum(m >= M for m in NULL_M), "of": K_NULL + 1,
                      "null": NULL, "chosen": ch, "per_layer_mean_fut20": {l: {"logit": float(FL[l].mean()), "a=1": a1[l], "up": dict(zip(map(str, UP), [float(x.mean()) for x in real[l]]))} for l in GATE_L}}
res["verdict"] = "REPORT-ONLY " + v; res["minutes"] = round((time.time() - T0) / 60, 1); res["memory"]["rss_peak_gib"] = peak_gib(); dump()
log("S-up:", v, f"M {M:.5f} CI {c} rank {res['report_only']['rank_of_M_real']}/20")
