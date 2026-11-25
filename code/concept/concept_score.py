"""J54 concept test, SCORE half: CPU only, file-only (2026-11-18; FROZEN BAR v4 + v4.1 + v4.2, a reviewer, bar file md5 2217480f,
written before any model data; the pre-registration card). The bar lives in gate() and nowhere else.
BAR (v4 + v4.1 + v4.2, verbatim in substance):
  OBJECT   J54 at L20..44 by 4 (no refit). Lens = softmax(W_U norm(J h)). Logit lens = the same path with J = I.
  POSITION R1; INTERMEDIATES R3; DECOY R4, drawn on the kept-before-filter list (all in concept_common.py).
  FILTER   R2 (v4.2): the WHOLE answer. The captured greedy continuation (up to 8 new tokens), decoded, leading whitespace
           stripped, must start with the target (exact for numbers, casefold for words) and the next character must be absent
           or not alphanumeric. multihop, multilingual, order-ops; poetry with its rhyme word; typo and association unfiltered.
  SCORE    rank of each intermediate in each lens at each of the 7 layers; best rank over the 7 layers; pass@k for
           k in {1,2,5,10,20,50,100,200,500,1000}; AUC = mean of pass@k over those 10 k; item score = mean over its intermediates.
  GATE     PASS iff G1 and G2, and G3 does not void:
           G1  AUC(J54) - AUC(logit): 95% CI low > 0.
           G2  AUC(J54, own intermediates) - AUC(J54, decoy): 95% CI low > 0.
           G3  CONTROL: row-permuted J54 (seed 20261104): AUC(rowperm) - AUC(logit) must NOT have CI low > 0; if it does, VOID.
           Bootstrap R5: items, resampled STRATIFIED within set, B 10000, seed 20261119, one draw shared by G1, G2, G3;
           statistic = the pooled mean over items.
  UNDERPOWERED (not FAIL) if fewer than 150 items survive in total.
  STOPS (NO-SCORE, never FAIL): pins mismatch; J = I path not bit-equal to the plain unembed readout on 5 items; any non-finite
           rank; fewer than 7 layers; and (review note #4) any lens logit row with zero variance, which strictly-greater ranking
           would score as rank 1 for every token.
  REPORT-ONLY (changes nothing): per-set rows; J22 and shrunk-J54 rows; per-layer rows; the unstratified bootstrap; per-set CIs;
           the v4.1 first-token filter count; and (v4.2) EVERY result split by read position, under 16 and 16 or over.
  READING  (v4.2, fixed before data): if the gate FAILS and the 16-or-over half alone would pass, the verdict is still FAIL.
           The report says so and names our position skip as a candidate cause. No re-grade of this run.
  NO FORECAST NUMBER. No re-run, no metric swap.
Usage: python concept_score.py            (score)
       CONCEPT_H_DIR=<dir> python concept_score.py   (plumbing dry run: reads capture files from <dir>, writes there, and
                                                      stamps every verdict DRY-RUN; the real output file is never touched)
       python concept_score.py --selftest (planted ranks; no files read)"""
import hashlib, json, os, platform, resource, sys, time
import numpy as np
B = 10000; BSEED = 20261119; RPSEED = 20261104; KS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]; MIN_ITEMS = 150
GATE_L = [20, 24, 28, 32, 36, 40, 44]; EPS = 1e-6; DIM = 5120

def auc(rank):
    """AUC of one rank (or an array of ranks): the mean of pass@k over KS. 1 = always rank 1."""
    r = np.asarray(rank, dtype=np.float64)[..., None]; return (r <= np.asarray(KS)).mean(-1)

def boot_idx(strata):
    """R5: one stratified draw, shared by every statistic. Returns a list of (member_indices, draws[B, n_s])."""
    rng = np.random.default_rng(BSEED); out = []
    for s in sorted(set(strata)):
        m = np.flatnonzero(np.asarray(strata) == s); out.append((m, rng.integers(0, len(m), (B, len(m)))))
    return out

def ci(diff, draws):
    """95% CI of the pooled mean over items under the stratified draw."""
    tot = np.zeros(B)
    for m, d in draws: tot += diff[m][d].sum(1)
    bs = tot / len(diff); return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

def gate(a54, alog, adec, arp, strata):   # per-item AUC arrays. Returns the decision dict. The ONLY place the bar lives.
    a54, alog, adec, arp = (np.asarray(x, dtype=np.float64) for x in (a54, alog, adec, arp)); n = len(a54)
    if not all(np.isfinite(x).all() for x in (a54, alog, adec, arp)): return {"decision": "NO-SCORE", "why": "non-finite item score", "n_items": n}
    if n < MIN_ITEMS: return {"decision": "UNDERPOWERED", "n_items": n, "min_items": MIN_ITEMS}
    dr = boot_idx(strata)
    g1, g2, g3 = a54 - alog, a54 - adec, arp - alog
    c1, c2, c3 = ci(g1, dr), ci(g2, dr), ci(g3, dr)
    r = {"n_items": n, "B": B, "seed": BSEED, "min_items": MIN_ITEMS,
         "G1": {"mean": float(g1.mean()), "ci95": c1, "pass": c1[0] > 0},
         "G2": {"mean": float(g2.mean()), "ci95": c2, "pass": c2[0] > 0},
         "G3": {"mean": float(g3.mean()), "ci95": c3, "void": c3[0] > 0}}
    r["decision"] = "VOID" if r["G3"]["void"] else ("PASS" if r["G1"]["pass"] and r["G2"]["pass"] else "FAIL")
    return r

def best_rank(logits_row, ids):
    """R3: the best (lowest) rank among the allowed token ids over the full vocabulary; 1 = top. Rank = 1 + the number of
    tokens with a STRICTLY higher logit, so tied tokens share the better rank (the same rule for every lens and for the decoy)."""
    return min(int((logits_row > logits_row[i]).sum()) + 1 for i in ids)

if "--selftest" in sys.argv:
    rng = np.random.default_rng(1); V = 248320; N = 300; strata = ["a"] * 100 + ["b"] * 120 + ["c"] * 80
    rnd = lambda n=N: auc(rng.integers(1, V + 1, n)); one = lambda n=N: auc(np.ones(n))
    mid = lambda n=N: auc(rng.integers(1, 2001, n))                    # a lens that is often in the top 1000
    lg = mid()
    cases = {
        "all-rank-1 lens vs a random lens, decoy random, rowperm random -> PASS": (one(), rnd(), rnd(), rnd(), strata, "PASS"),
        "lens EQUAL to logit (bar's own selftest) -> FAIL on G1": (lg, lg, rnd(), rnd(), strata, "FAIL"),
        "lens beats logit but the decoy ranks as high as the true step -> FAIL on G2": (one(), rnd(), one(), rnd(), strata, "FAIL"),
        "lens worse than logit -> FAIL": (rnd(), mid(), rnd(), rnd(), strata, "FAIL"),
        "row-permuted lens ALSO beats logit -> VOID": (one(), rnd(), rnd(), one(), strata, "VOID"),
        "void outranks a pass (G1, G2 true, G3 fires) -> VOID": (one(), rnd(), rnd(), mid(), strata, "VOID"),
        "149 items -> UNDERPOWERED, not FAIL": (one(149), rnd(149), rnd(149), rnd(149), strata[:149], "UNDERPOWERED"),
        "a NaN item score -> NO-SCORE, never FAIL": (np.r_[one(N - 1), np.nan], rnd(), rnd(), rnd(), strata, "NO-SCORE"),
        "mean gain over logit is positive but its CI includes 0 -> FAIL on G1 (the bar is CI low > 0, not mean > 0)":
            (np.clip(0.5 + 0.004 + rng.normal(0, 0.25, N), 0, 1), np.full(N, 0.5), np.zeros(N), np.zeros(N), strata, "FAIL"),
        "lens beats logit, and beats the decoy on average, but that CI includes 0 -> FAIL on G2":
            (one(), rnd(), np.clip(one() - 0.004 + rng.normal(0, 0.25, N), 0, 2), rnd(), strata, "FAIL"),
    }
    ok = True
    for name, (a, l, d, r, st, want) in cases.items():
        res = gate(a, l, d, r, st); good = res["decision"] == want; ok &= good
        print(("  ok  " if good else "  FAIL"), name, "->", res["decision"], {k: [round(x, 4) for x in res[k]["ci95"]] for k in ("G1", "G2", "G3") if k in res})
    # the effect lives in ONE stratum only: the stratified statistic must still be the pooled mean over items
    a = np.r_[one(100), lg[100:]]; res = gate(a, lg, rnd(), rnd(), strata); want_mean = float((a - lg).mean())
    good = abs(res["G1"]["mean"] - want_mean) < 1e-12 and res["G1"]["ci95"][0] < want_mean < res["G1"]["ci95"][1]; ok &= good
    print(("  ok  " if good else "  FAIL"), "effect in one set only: G1 mean is the pooled item mean and sits inside its CI", round(want_mean, 4), [round(x, 4) for x in res["G1"]["ci95"]])
    # the seeds are part of the bar: a fingerprint of each stream, so a changed seed fails here
    fpb = boot_idx(["x"] * 7)[0][1][0].tolist() == [1, 0, 3, 2, 2, 3, 1]; fpr = np.random.default_rng(RPSEED).permutation(DIM)[:5].tolist() == [4914, 1125, 1657, 5020, 4924]
    for name, g in (("bootstrap seed 20261119 gives its known first draw", fpb), ("row-permutation seed 20261104 gives its known first five", fpr)): ok &= g; print(("  ok  " if g else "  FAIL"), name)
    # auc and best_rank on planted values
    import torch
    row = torch.tensor([0.1, 5.0, 3.0, 3.0, -1.0]); t = [("rank of the top logit is 1", best_rank(row, [1]) == 1), ("a tie counts as tied-best (rank 2, not 3)", best_rank(row, [2]) == 2),
        ("best over forms takes the lower rank", best_rank(row, [4, 2]) == 2), ("last of five is rank 5", best_rank(row, [4]) == 5),
        ("auc(1) = 1", float(auc(1)) == 1.0), ("auc(1001) = 0", float(auc(1001)) == 0.0), ("auc(10) = 0.7", abs(float(auc(10)) - 0.7) < 1e-12), ("auc(1000) = 0.1", abs(float(auc(1000)) - 0.1) < 1e-12)]
    for name, g in t: ok &= g; print(("  ok  " if g else "  FAIL"), name)
    # R2 (v4.2), a reviewer's plants (22634 #2) and a few more
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import concept_common as cc
    r2 = [("20", "25", False), ("20", "200", False), ("20", " 2", False), ("20", " 20.", True), ("20", "20", True), ("20", " 20\n", True), ("20", "20 apples", True), ("20", "2 0", False),
          ("death", "Death", True), ("death", "deaths", False), ("death", " death,", True), ("death", "", False), ("Atlantic", " atlantic Ocean", True), ("Atlantic", " Atlantica", False),
          ("pequeño", " Pequeño\"", True), ("pequeño", " pequeña", False), ("5", " 5", True), ("5", "55", False), ("5", "5th", False)]
    for tgt, txt, want in r2:
        g = cc.answer_ok(tgt, txt) is want; ok &= g; print(("  ok  " if g else "  FAIL"), f"R2 target {tgt!r} continuation {txt!r} -> {'kept' if want else 'dropped'}")
    print("SELFTEST OK" if ok else "SELFTEST FAILED"); sys.exit(0 if ok else 1)

import torch
from safetensors import safe_open
torch.set_num_threads(int(os.environ.get("SCORE_THREADS", max(1, os.cpu_count() - 4))))
BASE = "$JLENS_HOME/claude-normtest"; CD = os.path.join(BASE, "concept"); JD = "$JLENS_HOME/results-whole-swift-record"
MD = "/home/<user>/models/bf16/Swift-1.5-Qwen3.8-27b"; DATA = os.path.join(BASE, "concept-data-581d3986"); S1, S2 = os.path.join(BASE, "stage1"), os.path.join(BASE, "stage2")
HD = os.environ.get("CONCEPT_H_DIR", CD); DRY = HD != CD      # DRY: capture files read from another folder (synthetic states, plumbing only)
BAR = os.path.join(CD, "jlens-j54-concept-test-prereg.md"); OUTP = os.path.join(HD, "concept_score.json"); NEED_GIB = 30.0
PIN = {BAR: "2217480f", os.path.join(DATA, "MD5SUMS"): "0c584676", os.path.join(BASE, "stage2_score.json"): "be917016", os.path.join(BASE, "score_shrink.json"): "0a925a72"}
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
sys.path.insert(0, CD); import concept_common as cc
res = {"script_md5": md5(__file__), "common_md5": md5(cc.__file__), "bar": "FROZEN BAR v4 + v4.1 + v4.2, file md5 2217480f", "verdict": "INCOMPLETE", "dry_synthetic": DRY, "stops": [], "pins": {}, "files": {},
       "env": {"torch": torch.__version__, "python": platform.python_version(), "threads": torch.get_num_threads(), "host": platform.node()}}
def dump(): json.dump(res, open(OUTP, "w"), indent=1)
def stop(why): res["stops"].append(why); res["verdict"] = "NO-SCORE: " + why; dump(); log("STOP:", why); sys.exit(3)
def mem_avail_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"): return int(line.split()[1]) / 2**20
# ---- 0 memory guard, pins --------------------------------------------------------------------------------------------------
ma0 = mem_avail_gib(); res["memory"] = {"mem_available_at_start_gib": round(ma0, 1), "required_gib": NEED_GIB}
if ma0 < NEED_GIB: stop(f"memory guard: MemAvailable {ma0:.1f} GiB < {NEED_GIB} (live engine protected)")
for p, want in PIN.items():
    got = md5(p); res["pins"][p] = got
    if not got.startswith(want): stop(f"pin {p} {got[:8]} != {want}")
CM = json.load(open(os.path.join(HD, "capture_meta.json")))
if CM.get("verdict") != "CAPTURED": stop(f"capture_meta verdict is {CM.get('verdict')!r}, not CAPTURED")
if CM["common_md5"] != res["common_md5"]: stop("concept_common.py changed between capture and score: the item rules must be one definition")
hp = os.path.join(HD, "concept_h.pt"); res["pins"][hp] = md5(hp)
if res["pins"][hp] != CM["concept_h_md5"]: stop("concept_h.pt md5 != capture_meta")
if len(GATE_L) != 7: stop("fewer than 7 layers")
S2S = json.load(open(os.path.join(BASE, "stage2_score.json"))); N_REC, T1, T2 = S2S["N"]["N_rec"], S2S["N"]["n1"], S2S["N"]["n2"]
if S2S["N"]["docs_in_J54"] != 54 or S2S["stage2_progress"]["complete"] is not True: stop("stage2_score.json does not describe a complete J54")
log("pins OK")
# ---- 1 items: R1, R3, R4 (shared module, on the kept-before-filter list), then R2 from the captured continuation ---------------
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MD); items = cc.load_items(DATA); kept0, drops = cc.prepare(tok, items)
Hc = torch.load(hp)
if Hc["idx"] != [it["idx"] for it in kept0]: stop("captured item order != the rules' kept list")
if sorted(Hc["layers"].keys()) != GATE_L: stop("captured layers != the 7 gate layers")
idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
def W(n):
    with safe_open(os.path.join(MD, idx[n]), "pt") as f: return f.get_tensor(n).float()
WU = W("lm_head.weight"); g1 = 1.0 + W("model.language_model.norm.weight")
def logits_of(x): return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * g1) @ WU.T
dec = cc.decoys(kept0)                                                # R4 on the kept-BEFORE-filter list (review note #3)
djs = os.path.join(CD, "decoys.json")                                # the list --list-only wrote before any model data: must be this one
want = json.load(open(djs)); got = {str(k): (None if v is None else {"word": v[0], "form_ids": v[1], "partner_idx": v[2], "fallback": v[3]}) for k, v in dec.items()}
if want != got: stop("the decoy list differs from decoys.json written before any model data")
res["pins"][djs] = md5(djs)
top_cpu = logits_of(Hc["h_final_last"]).argmax(-1).tolist(); top1 = Hc["top1"]
ok_r2 = [it["set"] not in cc.FILTERED or cc.answer_ok(cc.target_text(it), tok.decode(Hc["cont_ids"][i])) for i, it in enumerate(kept0)]
res["top1"] = {"heading": "REPORT-ONLY since v4.2", "next_token_head": CM.get("next_token_head"), "cpu_vs_capture_disagree": sum(1 for a, b in zip(top_cpu, top1) if a != b),
               "v41_first_token_filter_would_keep": sum(1 for i, it in enumerate(kept0) if it["set"] not in cc.FILTERED or top1[i] in it["tgt"]),
               "v42_whole_answer_keeps": int(sum(ok_r2))}
keep = [i for i, v in enumerate(ok_r2) if v]
kept = [kept0[i] for i in keep]; sel = torch.tensor(keep)
res["counts"] = {"items_total": len(items), "kept_before_filter": {s: sum(1 for it in kept0 if it["set"] == s) for s in cc.SETS}, "drops_before_run": drops,
                 "kept": {s: sum(1 for it in kept if it["set"] == s) for s in cc.SETS}, "kept_total": len(kept)}
log("kept", res["counts"]["kept"], "total", len(kept))
nodec = [it["name"] for it in kept if dec[it["idx"]] is None]
if nodec: stop(f"{len(nodec)} kept items have no decoy in their set: {nodec[:5]}")
res["counts"]["decoys"] = {"drawn_on": "kept before filter", "from_fallback_partner_among_kept": sum(1 for it in kept if dec[it["idx"]][3]),
                           "partner_was_filtered_out_among_kept": sum(1 for it in kept if dec[it["idx"]][2] not in {x["idx"] for x in kept})}
# ---- 2 ranks at the 7 layers --------------------------------------------------------------------------------------------
SH = json.load(open(os.path.join(BASE, "score_shrink.json"))); A = {l: float(SH["report"]["frozen_step2_parameters"]["real_per_layer"][str(l)]) for l in GATE_L}
perm = torch.from_numpy(np.random.default_rng(RPSEED).permutation(DIM)); LENSES = ["logit", "J54", "rowperm", "J22", "shrunk54"]
n = len(kept); own = {k: [[[] for _ in it["inter"]] for it in kept] for k in LENSES}; dcy = [[] for _ in kept]     # ranks per layer
eye_ok = None; I5 = torch.eye(DIM)
def acc(d, l, label, tot_want):
    p = os.path.join(d, f"acc_{l}.pt"); h = md5(p); res["files"][f"{label}/acc_{l}.pt"] = h
    if h != S2S["files"][f"{label}/acc_{l}.pt"]: stop(f"{label}/acc_{l}.pt md5 != the file Stage 2 was scored on")
    a = torch.load(p)
    if int(a[1]) != tot_want: stop(f"{label} acc_{l}: tot {int(a[1])} != {tot_want}")
    return a[0].float()
for l in GATE_L:
    jp = os.path.join(JD, f"J_l_{l}.pt"); h = md5(jp); res["files"][f"record/J_l_{l}.pt"] = h
    if h != S2S["files"][f"record/J_l_{l}.pt"]: stop(f"record/J_l_{l}.pt md5 != the file Stage 2 was scored on")
    J6 = torch.load(jp).float(); s1 = acc(S1, l, "stage1", T1); s2 = acc(S2, l, "stage2", T2)
    J22 = (N_REC * J6 + s1) / (N_REC + T1); J54 = (N_REC * J6 + s1 + s2) / (N_REC + T1 + T2); del J6, s1, s2
    H = Hc["layers"][l][sel]
    if l == GATE_L[0]:                                                 # STOP: the J = I path must be bit-equal to the plain unembed readout, 5 items
        eye_ok = bool(torch.equal(logits_of(H[:5] @ I5.T), logits_of(H[:5])))
        if not eye_ok: stop("J = I path is not bit-equal to the plain unembed readout on 5 items")
    X = {"logit": H, "J54": H @ J54.T, "rowperm": H @ J54[perm].T, "J22": H @ J22.T, "shrunk54": H + A[l] * (H @ (J54 - I5).T)}
    del J22, J54
    for k in LENSES:
        lg = logits_of(X[k])
        if not bool(torch.isfinite(lg).all()): stop(f"non-finite logits: lens {k} layer {l}")
        if bool((lg.std(-1) == 0).any()): stop(f"a logit row with zero variance: lens {k} layer {l} (ties would rank every token 1)")
        for i, it in enumerate(kept):
            for j, (_, f) in enumerate(it["inter"]): own[k][i][j].append(best_rank(lg[i], f))
            if k == "J54": dcy[i].append(best_rank(lg[i], dec[it["idx"]][1]))
        del lg
    log(f"L{l} ranked")
res["controls"] = {"J_eq_I_bit_equal_on_5_items": eye_ok, "rowperm_seed": RPSEED, "decoy_seed": cc.DSEED}
# ---- 3 item scores and the gate ---------------------------------------------------------------------------------------------
def item_auc(rk):       # rk[i][j] = the 7 per-layer ranks of intermediate j of item i. Best over layers, AUC, mean over intermediates.
    return np.array([np.mean([float(auc(min(r))) for r in it]) for it in rk])
IA = {k: item_auc(own[k]) for k in LENSES}; DA = np.array([float(auc(min(r))) for r in dcy]); strata = [it["set"] for it in kept]
res["gate"] = gate(IA["J54"], IA["logit"], DA, IA["rowperm"], strata); res["verdict"] = ("DRY-RUN ON SYNTHETIC STATES, NOT A RESULT: " if DRY else "") + res["gate"]["decision"]
# ---- 4 REPORT-ONLY ----------------------------------------------------------------------------------------------------------
rep = {"heading": "REPORT-ONLY. Nothing below changes the verdict.", "auc_mean": {k: float(IA[k].mean()) for k in LENSES} | {"J54_decoy": float(DA.mean())}}
def pair(a, b, st): return {"mean": float((a - b).mean()), "ci95": ci(a - b, boot_idx(st)) if len(a) >= 2 else None, "n": int(len(a))}
rep["per_set"] = {}
for s in cc.SETS:
    m = np.array([x == s for x in strata])
    if m.sum(): rep["per_set"][s] = {"n": int(m.sum()), "auc": {k: float(IA[k][m].mean()) for k in LENSES}, "J54_minus_logit": pair(IA["J54"][m], IA["logit"][m], [s] * int(m.sum())),
                                     "J54_minus_decoy": pair(IA["J54"][m], DA[m], [s] * int(m.sum()))}
rep["J22_minus_logit"] = pair(IA["J22"], IA["logit"], strata); rep["shrunk54_minus_logit"] = pair(IA["shrunk54"], IA["logit"], strata)
rep["J54_minus_J22"] = pair(IA["J54"], IA["J22"], strata)
rep["unstratified"] = {"J54_minus_logit": pair(IA["J54"], IA["logit"], ["all"] * n), "J54_minus_decoy": pair(IA["J54"], DA, ["all"] * n)}
rep["per_layer_auc"] = {str(l): {k: float(np.mean([np.mean([float(auc(r[li])) for r in it]) for it in own[k]])) for k in LENSES} for li, l in enumerate(GATE_L)}
rep["pass_at_k"] = {k: {str(kk): float(np.mean([np.mean([min(r) <= kk for r in it]) for it in own[k]])) for kk in KS} for k in LENSES}
pm = np.array([Hc["pos_model"][i] for i in keep]); early = pm < 16   # index in the BOS sequence, as SKIP_FIRST_N_POSITIONS counts
def half(m):
    st = [x for x, y in zip(strata, m) if y]
    if not m.sum(): return {"n": 0}
    return {"n": int(m.sum()), "per_set_n": {s: int(sum(1 for x in st if x == s)) for s in cc.SETS}, "auc": {k: float(IA[k][m].mean()) for k in LENSES} | {"J54_decoy": float(DA[m].mean())},
            "gate_on_this_half_alone (REPORT-ONLY, never the verdict)": gate(IA["J54"][m], IA["logit"][m], DA[m], IA["rowperm"][m], st),
            "J22_minus_logit": pair(IA["J22"][m], IA["logit"][m], st), "shrunk54_minus_logit": pair(IA["shrunk54"][m], IA["logit"][m], st)}
bp = {"under_16": half(early), "16_and_over": half(~early)}
late = bp["16_and_over"].get("gate_on_this_half_alone (REPORT-ONLY, never the verdict)", {}).get("decision")
bp["reading (v4.2, fixed before data)"] = ("The gate FAILED and the 16-or-over half alone would PASS. The verdict is still FAIL. Our position skip (the lens was fitted on "
    "positions 16 and over only) is a candidate cause; the next step is a refit on all positions, under its own new bar. No re-grade of this run."
    if res["gate"]["decision"] == "FAIL" and late == "PASS" else "The 16-or-over half alone does not change the reading of the verdict.")
rep["by_read_position (v4.2: report-only; our lens was fitted on positions 16 and over only)"] = bp
rep["paper_ordering"] = "the paper reports J beat logit on all six sets, modest on multihop and association; compare per_set above"
res["report"] = rep; res["minutes"] = round((time.time() - T0) / 60, 1); res["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
res["memory"]["rss_peak_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2); dump()
g = res["gate"]; log("GATE:", g["decision"], {k: (round(g[k]["mean"], 4), [round(x, 4) for x in g[k]["ci95"]]) for k in ("G1", "G2", "G3") if k in g}); log("done:", res["verdict"])
