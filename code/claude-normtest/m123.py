"""Jett round-1 measurements M1 and M3, REPORT-ONLY, CPU from files (Claude 2026-11-07; definitions fixed by Deron P112 before any
measurement; Jett's pinned prediction PREDICTION-ADDENDUM.md md5 4e85fd19; card C01). M2 is NOT computed here: with only the 306
held states Sigma_h has rank <= 305, so PR(M) <= 305 by construction -> UNMEASURABLE until >= ~20k states are captured (post P112 item 2).
Layers 24, 40, 52 (Jett's). Per-doc RECORD maps do not exist (results-whole-swift-record holds only the 47 pooled J_l_16..62.pt;
checked 11-07), so both measurements use the 16 Stage-1 docs:
  halves: the same 8/8 split as score_stage1 f2409a09's 8-vs-8 report (torch.randperm(16) under seed 20261106+2), n_valid-weighted means
          J8a, J8b; J16 = n_valid-weighted mean of all 16. D16 = J16 - I.
  M1  (post P112 item 1): sigma_D = singular values of D16; sigma_N = singular values of (J8a - J8b)/2 (the noise of the mean of the two
      halves); both sorted descending; corrected s_i^2 = max(sigma_D_i^2 - sigma_N_i^2, 0) (index-matched quadrature subtraction);
      PR = (sum s^2)^2 / sum s^4. Also reported: raw PR of D16 and PR of the noise. LIMIT: the noise mixes position and doc variance.
      Jett's intervals (hit iff PR_corrected inside): L52 1,700-2,600; L40 2,400-3,300; L24 3,000-3,900.
  M3  (as Jett defined it): ||J8a - J8b||_F / ||J16||_F (||I||_F = 71.6 dominates it, post P112), and also / ||D16||_F, report-only.
      Jett's interval (hit iff inside): 0.10-0.25 per layer, on the ||J16|| denominator.
Pins: stage1 meta md5 for every J_p read (as f2409a09). Memory guard at start (MemAvailable >= 20 GiB) and before each layer (>= 15 GiB).
Usage: python m123.py      writes m123.json"""
import hashlib, json, os, re, resource, sys, time, platform, torch
torch.set_num_threads(int(os.environ.get("SCORE_THREADS", max(1, os.cpu_count() - 4))))
BASE = "$JLENS_HOME/claude-normtest"; S1 = os.path.join(BASE, "stage1"); OUTP = os.path.join(BASE, "m123.json")
LAYERS = [24, 40, 52]; DIM = 5120; SPLIT_SEED = 20261106 + 2
JETT = {"M1": {52: (1700, 2600), 40: (2400, 3300), 24: (3000, 3900)}, "M3": (0.10, 0.25)}
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
def mem_avail_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"): return int(line.split()[1]) / 2**20
res = {"script_md5": md5(__file__), "prereg": "Deron P112 definitions; Jett PREDICTION-ADDENDUM 4e85fd19", "verdict": "INCOMPLETE", "stops": [], "files": {},
       "env": {"torch": torch.__version__, "python": platform.python_version(), "threads": torch.get_num_threads(), "host": platform.node()},
       "M2": "UNMEASURABLE with the 306 held states (rank(Sigma_h) <= 305 caps PR(M) at 305); needs >= ~20k captured states (post P112 item 2)"}
def dump(): json.dump(res, open(OUTP, "w"), indent=1)
def stop(why): res["stops"].append(why); res["verdict"] = "NO-SCORE: " + why; dump(); log("STOP:", why); sys.exit(3)
if mem_avail_gib() < 20: stop(f"memory guard at start: MemAvailable {mem_avail_gib():.1f} GiB < 20")
docs = sorted(d for d in os.listdir(S1) if re.fullmatch(r"doc_\d\d", d))
if docs != [f"doc_{i:02d}" for i in range(1, 17)]: stop(f"Stage 1 doc dirs are not exactly doc_01..doc_16: {docs}")
meta = [json.load(open(os.path.join(S1, d, "meta.json"))) for d in docs]; n = [int(m["n_valid"]) for m in meta]
if min(n) <= 0: stop(f"a doc has n_valid <= 0: {n}")
perm = torch.randperm(16, generator=torch.Generator().manual_seed(SPLIT_SEED)).tolist(); A, Bh = sorted(perm[:8]), sorted(perm[8:])
res["split"] = {"seed": SPLIT_SEED, "half_A_docs": [i + 1 for i in A], "half_B_docs": [i + 1 for i in Bh], "n_valid": n}
PR = lambda s2: float(s2.sum() ** 2 / (s2 ** 2).sum()) if float(s2.sum()) > 0 else 0.0
I = torch.eye(DIM); out = {}
for l in LAYERS:
    ma = mem_avail_gib()
    if ma < 15: stop(f"mid-run memory guard before L{l}: MemAvailable {ma:.1f} GiB < 15")
    Jp = []
    for i, d in enumerate(docs):
        p = os.path.join(S1, d, f"J_p_{l}.pt"); h = md5(p); res["files"][f"stage1/{d}/J_p_{l}.pt"] = h
        if h != meta[i]["J_p_md5"][str(l)]: stop(f"{d} J_p_{l} md5 {h[:8]} != meta")
        Jp.append(torch.load(p).float())
    wavg = lambda idx: sum(n[i] * Jp[i] for i in idx) / sum(n[i] for i in idx)
    J16, Ja, Jb = wavg(range(16)), wavg(A), wavg(Bh); del Jp
    D16 = J16 - I; N = (Ja - Jb) / 2
    sD = torch.linalg.svdvals(D16); sN = torch.linalg.svdvals(N)
    s2 = (sD ** 2 - sN ** 2).clamp(min=0)
    pr_c, pr_raw, pr_n = PR(s2), PR(sD ** 2), PR(sN ** 2)
    m3_full = float((Ja - Jb).norm() / J16.norm()); m3_delta = float((Ja - Jb).norm() / D16.norm())
    lo1, hi1 = JETT["M1"][l]; lo3, hi3 = JETT["M3"]
    out[l] = {"M1": {"PR_corrected": pr_c, "PR_raw_D16": pr_raw, "PR_noise": pr_n, "jett_interval": [lo1, hi1], "hit": lo1 <= pr_c <= hi1,
                     "top10_sigma_D": sD[:10].tolist(), "top10_sigma_noise": sN[:10].tolist(), "share_of_D_energy_above_noise": float(s2.sum() / (sD ** 2).sum())},
              "M3": {"ratio_over_J16": m3_full, "jett_interval": [lo3, hi3], "hit": lo3 <= m3_full <= hi3, "ratio_over_D16_report_only": m3_delta,
                     "fro_J16": float(J16.norm()), "fro_D16": float(D16.norm()), "fro_I": float(I.norm())}}
    log(f"L{l}: M1 PR corrected {pr_c:.0f} (raw {pr_raw:.0f}, noise {pr_n:.0f}) hit={out[l]['M1']['hit']} | M3 {m3_full:.4f} hit={out[l]['M3']['hit']} (over D16 {m3_delta:.3f})")
    del J16, Ja, Jb, D16, N, sD, sN, s2
res["layers"] = out; res["limits"] = ["M1 noise mixes position and doc variance (post P112)", "16 Stage-1 docs, not the 6-doc record (per-doc record maps do not exist)",
                                      "index-matched quadrature subtraction is a heuristic for spectra of a sum (as defined in post P112)"]
res["verdict"] = "REPORT-ONLY DONE"; res["minutes"] = round((time.time() - T0) / 60, 1); res["rss_peak_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
res["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S"); dump(); log("done")
