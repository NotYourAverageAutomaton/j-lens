"""J-Lens shrink STEP 2 SCORER, CPU only (Claude 2026-11-07; PREREG 0466a966 step 2 as amended by amend-1 e088c347 "Step 2 (fresh text),
same hole, same fix"; card C01). Reads step2/fresh_h.pt (step2_capture.py) and the frozen parameters in score_shrink.json 0a925a72
(report.frozen_step2_parameters: family S, real a per layer, each null draw's own in-sample a per layer). Nothing is refit here.
D22 is pooled exactly as shrink_check.py d153df93 (record J6 + Stage-1 acc, same pins, same resume (b)+(c)); the null draws are the same
19 row-permutations (draw i: its own generator seeded 20261104+i, one randperm(5120) per layer in L20..44 order).
ORDER. Every step before the verdict can only STOP (verdict NO-SCORE):
  0 memory guard at start (MemAvailable >= 17 + 20 GiB, as d153df93); pins; md5 of every file read; frozen family must be S.
  1 fresh_h.pt: capture_meta.json verdict CAPTURED and its md5 matches; ids re-tokenized here from passages.json (same hermetic path:
    BOS + truncation to 255) must equal the captured ids; 6 x 239 = 1434 positions; passage shas == passages.json.
  2 D22 pooled (resume (b)+(c) as d153df93).
  3 positive controls on the 306 held positions: (a) a=1 reproduces score_stage1.json J22 fut20 at L20-44 (|diff| <= 1e-6); (b) the
    real frozen a per layer reproduces score_shrink.json readout_table n22 fut20 for that a (|diff| <= 1e-6); (c) a=0 is the logit
    lens, bit-equal per position, on the fresh positions.
  4 tables on the fresh positions: logit fut20; real at its frozen a; each null draw at its own frozen a.
    g(q) = mean over L20..44 by 4 of [fut20_cand(q,L) - fut20_logit(q,L)]; M = mean g.
  5 self-gating dry runs on these tables: FAIL-WAY (real a = 0 at every layer) must read FAIL; PASS-WAY (planted oracle: h_final in
    place of the real candidate at every layer, against the real 19 null draws) must read PASS. Either wrong: STOP.
  6 VERDICT (amend-1 step 2): PASS iff M_real > every M_i (strict; permutation p < 1/20) AND the paired-bootstrap 95% CI (2.5/97.5,
    B = 10000 over the 1434 positions, seed 20261108) of g_real is entirely > 0. Anything else = FAIL ("luck").
  7 REPORT-ONLY: per-layer gains, all 19 M_i and the rank, per-passage mean g with a crude 6-cluster CI, top-1 agreement, logit fut20.
Usage: python step2_score.py      writes step2/score_step2.json"""
import ast, hashlib, json, os, re, resource, sys, time, platform, torch
from safetensors import safe_open
from transformers import AutoTokenizer
torch.set_num_threads(int(os.environ.get("SCORE_THREADS", max(1, os.cpu_count() - 4))))
BASE = "$JLENS_HOME/claude-normtest"; S2 = os.path.join(BASE, "step2"); JD = "$JLENS_HOME/results-whole-swift-record"
MD = "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b"; S1 = os.path.join(BASE, "stage1"); OUTP = os.path.join(S2, "score_step2.json")
GATE_L = [20, 24, 28, 32, 36, 40, 44]; EPS = 1e-6; SKIP = 16; MAXLEN = 256; DIM = 5120; N_REC_PARTS = [131, 107, 96, 96, 93, 119]
B = 10000; SEED = 20261108; PERM_SEED = 20261104; K_NULL = 19; QLO, QHI = 0.025, 0.975; NFRESH = 6 * (MAXLEN - 1 - SKIP)
PIN = {"held_h.pt": "3ca0f99c", "pilot_hutch.py": "5fa18d9b", "$JLENS_HOME/whole_model_fit_record.log": "d0cf6a41",
       "score_stage1.json": "67449876", "score_shrink.json": "0a925a72", "step2/passages.json": "2d302976"}
NEED = 17.0 + 20.0
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
peak_gib = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
def mem_avail_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"): return int(line.split()[1]) / 2**20
res = {"script_md5": md5(__file__), "prereg": "PREREG-shrink-check.md 0466a966 step 2 + PREREG-amend-1.md e088c347 step 2", "verdict": "INCOMPLETE",
       "stops": [], "pins": {}, "files": {}, "env": {"torch": torch.__version__, "python": platform.python_version(), "threads": torch.get_num_threads(),
       "host": platform.node(), "note": "CPU-only run; GPU not used"}}
def dump(): json.dump(res, open(OUTP, "w"), indent=1)
def stop(why): res["stops"].append(why); res["verdict"] = "NO-SCORE: " + why; dump(); log("STOP:", why); sys.exit(3)
# ---- 0 memory guard, pins, frozen parameters -----------------------------------------------------------------------
ma0 = mem_avail_gib(); res["memory"] = {"mem_available_at_start_gib": round(ma0, 1), "required_gib": NEED}
if ma0 < NEED: stop(f"memory guard: MemAvailable {ma0:.1f} GiB < {NEED} (live engine protected)")
for name, want in PIN.items():
    got = md5(name if name.startswith("/") else os.path.join(BASE, name)); res["pins"][name] = got
    if not got.startswith(want): stop(f"pin {name} {got[:8]} != {want}")
SH = json.load(open(os.path.join(BASE, "score_shrink.json"))); FZ = SH["report"]["frozen_step2_parameters"]
if SH["verdict"] != "SURVIVE" or FZ is None or FZ["family"] != "S": stop(f"score_shrink.json: verdict {SH['verdict']}, frozen family {FZ and FZ['family']} (step 2 needs SURVIVE, S)")
A_REAL = {l: float(FZ["real_per_layer"][str(l)]) for l in GATE_L}
if [d["seed"] for d in FZ["null_per_layer"]] != [PERM_SEED + i for i in range(1, K_NULL + 1)]: stop("frozen null seeds are not 20261105..20261123 in order")
A_NULL = [{l: float(d["per_layer"][str(l)]) for l in GATE_L} for d in FZ["null_per_layer"]]
res["frozen"] = {"real": A_REAL, "null": A_NULL}
log("pins OK; frozen a real", A_REAL)
# ---- shared CPU fut20 path (d153df93, unchanged maths) --------------------------------------------------------------
idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
def W(n):
    with safe_open(os.path.join(MD, idx[n]), "pt") as f: return f.get_tensor(n).float()
WU = W("lm_head.weight"); g1 = 1.0 + W("model.language_model.norm.weight")
def logits_of(x):
    xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * g1
    return xn @ WU.T
def make_readout(nexts, top1_final):
    def readout(x):   # per-position fut20 (eval.py) plus report-only mean top-1 agreement with the final layer
        lg = logits_of(x); t20 = lg.topk(20, -1).indices
        f = torch.tensor([float(torch.isin(n, t20[i]).float().mean()) for i, n in enumerate(nexts)])
        return f, None if top1_final is None else float((lg.argmax(-1) == top1_final).float().mean())
    return readout
tok = AutoTokenizer.from_pretrained(MD); bos = tok.bos_token_id
def encode(text):   # hermetic: BOS + truncation to 255 (our_jlens._encode)
    ids = tok(text, truncation=True, max_length=MAXLEN - (1 if bos is not None else 0), add_special_tokens=False).input_ids
    return ([bos] if bos is not None else []) + ids
nexts_of = lambda ids: [torch.tensor(ids[q + 1:q + 11]) for q in range(SKIP, len(ids) - 1)]
# ---- 1 fresh_h.pt ---------------------------------------------------------------------------------------------------
P = json.load(open(os.path.join(S2, "passages.json"))); CM = json.load(open(os.path.join(S2, "capture_meta.json")))
fp = os.path.join(S2, "fresh_h.pt"); res["files"]["step2/fresh_h.pt"] = md5(fp); res["files"]["step2/capture_meta.json"] = md5(os.path.join(S2, "capture_meta.json"))
if CM.get("verdict") != "CAPTURED" or CM.get("fresh_h_md5") != res["files"]["step2/fresh_h.pt"]: stop(f"capture_meta verdict {CM.get('verdict')} / fresh_h md5 mismatch")
if CM.get("passages_md5") != res["pins"]["step2/passages.json"]: stop("capture_meta passages_md5 != pinned passages.json")
for p in P:
    if hashlib.sha256(p["text"].encode()).hexdigest() != p["sha256"]: stop(f"{p['genre']} text sha256 != recorded")
FR = torch.load(fp); ids_ref = [encode(p["text"]) for p in P]
if FR["ids"] != ids_ref: stop("captured ids != ids re-tokenized here from passages.json")
nexts = [n for ids in ids_ref for n in nexts_of(ids)]
if len(nexts) != NFRESH or FR["h_final"].shape[0] != NFRESH or len(FR["passage"]) != NFRESH: stop(f"positions {len(nexts)} / fresh {FR['h_final'].shape[0]} != {NFRESH}")
passage = torch.tensor(FR["passage"]); res["fresh"] = {"positions": NFRESH, "passages": [{"genre": p["genre"], "sha256": p["sha256"][:8]} for p in P]}
fresh_read = make_readout(nexts, logits_of(FR["h_final"]).argmax(-1))
log("fresh_h OK:", NFRESH, "positions")
# ---- 2 D22 pooled exactly as d153df93 -------------------------------------------------------------------------------
I = torch.eye(DIM); J6 = {l: torch.load(os.path.join(JD, f"J_l_{l}.pt")).float() for l in GATE_L}
for l in GATE_L: res["files"][f"record/J_l_{l}.pt"] = md5(os.path.join(JD, f"J_l_{l}.pt"))
parts = [int(x) for x in re.findall(r"n_valid=(\d+)", open("$JLENS_HOME/whole_model_fit_record.log").read())]; N_REC = sum(parts)
if parts != N_REC_PARTS or N_REC != 642: stop(f"N_rec parts {parts} != {N_REC_PARTS}")
docs = sorted(d for d in os.listdir(S1) if re.fullmatch(r"doc_\d\d", d))
if docs != [f"doc_{i:02d}" for i in range(1, 17)]: stop(f"Stage 1 doc dirs are not exactly doc_01..doc_16: {docs}")
meta = [json.load(open(os.path.join(S1, d, "meta.json"))) for d in docs]; n = [int(m["n_valid"]) for m in meta]
if [m["doc"] for m in meta] != list(range(1, 17)) or min(n) <= 0: stop(f"meta docs / n_valid bad: {n}")
D22 = {}
for l in GATE_L:
    for i, d in enumerate(docs):
        p = os.path.join(S1, d, f"J_p_{l}.pt"); h = md5(p); res["files"][f"stage1/{d}/J_p_{l}.pt"] = h
        if h != meta[i]["J_p_md5"][str(l)]: stop(f"{d} J_p_{l} md5 {h[:8]} != meta")
    ap = os.path.join(S1, f"acc_{l}.pt"); res["files"][f"stage1/acc_{l}.pt"] = md5(ap); s, tot = torch.load(ap)
    Jl = [torch.load(os.path.join(S1, d, f"J_p_{l}.pt")).float() for d in docs]
    want = sum(nn * j for nn, j in zip(n, Jl)); rel = float((s.float() - want).norm() / want.norm())
    if int(tot) != sum(n) or rel > 1e-5: stop(f"acc_{l}: total {tot} vs {sum(n)}, rel {rel:.2e}")
    D22[l] = (N_REC * J6[l] + want) / (N_REC + sum(n)) - I
    del Jl, want, s; res["memory"][f"rss_after_layer_{l}"] = peak_gib()
del J6; log("D22 pooled")
# ---- 3 positive controls --------------------------------------------------------------------------------------------
held = torch.load(os.path.join(BASE, "held_h.pt"))
def held_out_from(path):   # ast.literal_eval of the module's HELD_OUT assignment, never eval()
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "HELD_OUT" for t in node.targets):
            return ast.literal_eval(node.value)
held_nexts = [n for t in held_out_from(os.path.join(BASE, "pilot_hutch.py")) for n in nexts_of(encode(t))]
if len(held_nexts) != 306 or held["h_final"].shape[0] != 306: stop(f"held positions {len(held_nexts)} != 306")
held_read = make_readout(held_nexts, None)
S1REF = json.load(open(os.path.join(BASE, "score_stage1.json")))["report"]["per_layer_table"]["rows"]; RT = SH["report"]["readout_table"]["n22"]
pc = {}
for l in GATE_L:
    h = held["layers"][l]; hD = h @ D22[l].T
    f1 = float(held_read(h + hD)[0].mean()); fa = float(held_read(h + A_REAL[l] * hD)[0].mean())
    pc[l] = {"a1": f1, "score_stage1_J22": S1REF[str(l)]["J22"]["fut20"], "a_frozen": fa, "score_shrink_row": RT[str(l)][f"a={A_REAL[l]}"]["fut20"]}
    pc[l]["diff_a1"] = abs(f1 - pc[l]["score_stage1_J22"]); pc[l]["diff_frozen"] = abs(fa - pc[l]["score_shrink_row"])
res["positive_control"] = {"held_306": pc}
if any(pc[l]["diff_a1"] > 1e-6 or pc[l]["diff_frozen"] > 1e-6 for l in GATE_L): stop(f"positive control (a)/(b) on held positions: {pc}")
del held; log("positive controls (a)(b) OK")
# ---- 4 tables on the fresh positions --------------------------------------------------------------------------------
FL, REAL, NULLT, top1 = {}, {}, [{} for _ in range(K_NULL)], {"logit": {}, "real": {}}
for l in GATE_L:
    h = FR["layers"][l]; FL[l], top1["logit"][l] = fresh_read(h)
    hD = h @ D22[l].T
    if not torch.equal(fresh_read(h + 0.0 * hD)[0], FL[l]): stop(f"positive control (c): a=0 is not bit-equal to the logit lens at L{l}")
    REAL[l], top1["real"][l] = fresh_read(h + A_REAL[l] * hD) if A_REAL[l] != 0 else (FL[l], top1["logit"][l])
    log(f"L{l} logit {float(FL[l].mean()):.4f} real(a={A_REAL[l]}) {float(REAL[l].mean()):.4f}")
GENS = [torch.Generator().manual_seed(PERM_SEED + i) for i in range(1, K_NULL + 1)]
for l in GATE_L:   # each generator draws one randperm per layer in L20..44 order, as d153df93
    h = FR["layers"][l]
    for i in range(K_NULL):
        perm = torch.randperm(DIM, generator=GENS[i]); a = A_NULL[i][l]
        NULLT[i][l] = fresh_read(h + a * (h @ D22[l][perm].T))[0] if a != 0 else FL[l]
    if mem_avail_gib() < 15: stop(f"mid-run memory guard at L{l}: MemAvailable < 15 GiB")
    log(f"L{l} null tables done"); res["memory"][f"rss_null_L{l}"] = peak_gib()
g_of = lambda tab: torch.stack([tab[l] - FL[l] for l in GATE_L]).mean(0)
gR = g_of(REAL); gN = [g_of(t) for t in NULLT]; NULL_M = [float(g.mean()) for g in gN]
gb = torch.Generator().manual_seed(SEED); BIDX = torch.randint(0, NFRESH, (B, NFRESH), generator=gb)
def ci(g): bs = g[BIDX].mean(1); return [float(bs.quantile(QLO)), float(bs.quantile(QHI))]
def verdict(g, null_M):
    M = float(g.mean()); c = ci(g); beats = all(M > m for m in null_M)
    return ("PASS" if beats and c[0] > 0 else "FAIL"), {"M_real": M, "ci_2.5_97.5": c, "beats_all_null": beats,
            "rank_of_M_real": 1 + sum(m >= M for m in null_M), "of": len(null_M) + 1}
# ---- 5 self-gating dry runs -----------------------------------------------------------------------------------------
v_fail, _ = verdict(g_of(FL), NULL_M)
orc = fresh_read(FR["h_final"])[0]; v_pass, rk_pass = verdict(g_of({l: orc for l in GATE_L}), NULL_M)
res["dry_runs"] = {"fail_way_a0": v_fail, "pass_way_oracle_h_final": v_pass, "pass_way": rk_pass,
                   "rule": "FAIL-WAY must read FAIL and PASS-WAY must read PASS, on these tables, before the real verdict"}
if v_fail != "FAIL" or v_pass != "PASS": stop(f"self-gating dry runs: fail-way {v_fail} (want FAIL), pass-way {v_pass} (want PASS)")
log("dry runs OK: fail-way FAIL, pass-way PASS")
# ---- 6 VERDICT ------------------------------------------------------------------------------------------------------
v, rk = verdict(gR, NULL_M)
res["check"] = {"bar": "PREREG-amend-1 e088c347 step 2: PASS iff M_real > all 19 M_i (strict, p < .05) AND the 95% CI (2.5/97.5) of g_real is entirely > 0; else FAIL",
                "rank_rule": rk, "null_M": NULL_M, "B": B, "seed": SEED, "null_seeds": [PERM_SEED + i for i in range(1, K_NULL + 1)], "decision": v}
res["verdict"] = v; dump()
log("STEP 2:", v, f"M_real {rk['M_real']:.5f} rank {rk['rank_of_M_real']}/{rk['of']} CI {rk['ci_2.5_97.5']} null M max {max(NULL_M):.5f}")
# ---- 7 REPORT-ONLY --------------------------------------------------------------------------------------------------
gc = torch.Generator().manual_seed(SEED + 1)
cl = torch.stack([gR[passage == c].mean() for c in range(6)])
cbs = torch.stack([cl[torch.randint(0, 6, (6,), generator=gc)].mean() for _ in range(B)])
res["report"] = {"heading": "REPORT-ONLY. Nothing below gates anything.",
                 "per_layer": {l: {"a": A_REAL[l], "logit_fut20": float(FL[l].mean()), "real_fut20": float(REAL[l].mean()),
                                   "gain": float((REAL[l] - FL[l]).mean()), "top1_logit": top1["logit"][l], "top1_real": top1["real"][l]} for l in GATE_L},
                 "null_draws": [{"seed": PERM_SEED + i + 1, "a": A_NULL[i], "M": NULL_M[i]} for i in range(K_NULL)],
                 "per_passage_mean_g": {P[c]["genre"]: float(cl[c]) for c in range(6)},
                 "cluster_ci_6_passages_crude": [float(cbs.quantile(.025)), float(cbs.quantile(.975))]}
res["memory"]["rss_peak_gib"] = peak_gib(); res["minutes"] = round((time.time() - T0) / 60, 1); res["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S"); dump()
log("report-only done; verdict", res["verdict"])
