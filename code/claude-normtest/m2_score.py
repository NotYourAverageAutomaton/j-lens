"""J-Lens M2, REPORT-ONLY, CPU from files (Claude 2026-11-07; estimator Deron P112 item 2, states pinned post P123, captured by m2_capture
d0191bbf; Jett's pinned prediction PREDICTION-ADDENDUM 4e85fd19; card C01).
Per layer l in 24, 40, 52:
  Sigma_h = covariance of the captured states at l (centred, /(N-1)); Sigma^{1/2} by eigh, eigenvalues clipped at 0.
  x_bar = A h_bar (h_bar = mean state); J_rms(x) = diag(g1) (I/r - x x^T / (d r^3)), r = sqrt(mean(x^2) + 1e-6), g1 = 1 + norm.weight
  (the readout's RMSNorm, as every fut20 path); W = lm_head (fp32).
  M = W J_rms(x_bar) (A - I) Sigma^{1/2};  PR(M) = (sum s^2)^2 / sum s^4, s = singular values of M, from eigvalsh(M^T M) (5120 x 5120;
  W enters only as G = W^T W).
  PRIMARY: A = J_6 (record J_l, pinned by score_shrink.json's md5). REPORT-ONLY: A = J_22 (pooled exactly as shrink_check d153df93), and
  the per-state J_rms mean over q of J_rms(A h_q) in place of J_rms(x_bar), with A = J_6.
N-STABILITY (post P123): the primary PR also at N/2 = the first 42 articles; if |PR_N/2 - PR_N| / PR_N > 0.15 the layer's M2 is labelled
  NOT CONVERGED and Jett's hit/miss is graded with that label. Also reported: rank of Sigma_h (eigenvalues > 1e-7 * max) and its floor.
Jett's intervals (hit iff PR_primary inside): L52 150-350; L40 350-700; L24 700-1,400.
STOPs: memory guard (MemAvailable >= 20 GiB start, >= 10 before each layer); m2_capture_meta CAPTURED with matching md5, 20,076 states
from 84 articles; record J_l md5 == score_shrink.json's; Stage-1 resume (b)+(c) for J_22 as d153df93; machinery controls before any data:
  (i) PR of a spectrum with k equal values == k; (ii) the Gram route equals explicit SVD on a seeded 300 x 200 matrix (rel 1e-6);
  (iii) J_rms matches torch.autograd.functional.jacobian of the RMSNorm at x_bar for L24 (rel 1e-5).
CENTRED (Deron P124, report-only): a uniform logit shift changes no softmax and no top-k but carries energy in W^T W. So also PR(M) with
  centred logits, G_c = W^T (I - 11^T/V) W = G - V w_bar w_bar^T (w_bar = mean row of W), primary A = J_6, at N and N/2; plus the factor
  spectra alone: PR of eig(G), PR of eig(G_c), PR of eig(Sigma_h), and the share of tr(G) on w_bar's direction. Nothing gates on these.
Usage: python m2_score.py      writes m2_v2.json (95a1dd02 wrote m2.json; this file leaves it untouched)"""
import hashlib, json, os, re, resource, sys, time, platform, torch
from safetensors import safe_open
torch.set_num_threads(int(os.environ.get("SCORE_THREADS", max(1, os.cpu_count() - 4))))
BASE = "$JLENS_HOME/claude-normtest"; JD = "$JLENS_HOME/results-whole-swift-record"; S1 = os.path.join(BASE, "stage1")
MD = "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b"; OUTP = os.path.join(BASE, "m2_v2.json")
M2_L = [24, 40, 52]; DIM = 5120; EPS = 1e-6; N_WANT = 20076; ARTS = 84; HALF_ARTS = 42; CONV_TOL = 0.15; N_REC_PARTS = [131, 107, 96, 96, 93, 119]
JETT = {52: (150, 350), 40: (350, 700), 24: (700, 1400)}
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
def mem_avail_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"): return int(line.split()[1]) / 2**20
res = {"script_md5": md5(__file__), "prereg": "Deron P112 item 2 + post P123; Jett PREDICTION-ADDENDUM 4e85fd19", "verdict": "INCOMPLETE", "stops": [], "files": {},
       "env": {"torch": torch.__version__, "python": platform.python_version(), "threads": torch.get_num_threads(), "host": platform.node()}}
def dump(): json.dump(res, open(OUTP, "w"), indent=1)
def stop(why): res["stops"].append(why); res["verdict"] = "NO-SCORE: " + why; dump(); log("STOP:", why); sys.exit(3)
if mem_avail_gib() < 20: stop(f"memory guard at start: MemAvailable {mem_avail_gib():.1f} GiB < 20")
PR = lambda e: float(e.sum() ** 2 / (e ** 2).sum()) if float(e.sum()) > 0 else 0.0   # e = s^2 (eigenvalues of M^T M), clipped >= 0
# ---- machinery controls (before any data) ------------------------------------------------------------------------------
c1 = PR(torch.tensor([3.0] * 37 + [0.0] * 10))
gm = torch.Generator().manual_seed(20261108); Mt = torch.randn(300, 200, generator=gm, dtype=torch.float64)
e_gram = torch.linalg.eigvalsh(Mt.T @ Mt).clamp(min=0); e_svd = torch.linalg.svdvals(Mt) ** 2
c2 = abs(PR(e_gram) - PR(e_svd)) / PR(e_svd)
res["controls"] = {"i_pr_k_equal_37": c1, "ii_gram_vs_svd_rel": c2}
if abs(c1 - 37) > 1e-9 or c2 > 1e-6: stop(f"machinery controls: PR(37 equal) {c1}, gram-vs-svd rel {c2:.2e}")
# ---- states ------------------------------------------------------------------------------------------------------------
CM = json.load(open(os.path.join(BASE, "m2_capture_meta.json"))); sp = os.path.join(BASE, "m2_states.pt"); res["files"]["m2_states.pt"] = md5(sp)
if CM.get("verdict") != "CAPTURED" or CM.get("m2_states_md5") != res["files"]["m2_states.pt"]: stop(f"m2_capture_meta verdict {CM.get('verdict')} / md5 mismatch")
ST = torch.load(sp); art = torch.tensor(ST["article"])
if ST["h_final"].shape[0] != N_WANT or len(ST["used_article_indices"]) != ARTS: stop(f"states {ST['h_final'].shape[0]} / articles {len(ST['used_article_indices'])} != {N_WANT} / {ARTS}")
half = art < HALF_ARTS; res["N"] = {"full": N_WANT, "half": int(half.sum()), "half_rule": f"first {HALF_ARTS} articles"}
# ---- readout pieces ----------------------------------------------------------------------------------------------------
idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
def Wt(n):
    with safe_open(os.path.join(MD, idx[n]), "pt") as f: return f.get_tensor(n).float()
g1 = 1.0 + Wt("model.language_model.norm.weight"); WU = Wt("lm_head.weight")
G = torch.zeros(DIM, DIM, dtype=torch.float64); wsum = torch.zeros(DIM, dtype=torch.float64); V = WU.shape[0]
for s in range(0, V, 16384): w = WU[s:s + 16384].double(); G += w.T @ w; wsum += w.sum(0)
del WU; wbar = wsum / V; Gc = (G - V * torch.outer(wbar, wbar)).float(); G = G.float(); log("G = W^T W and centred G_c built")
eG = torch.linalg.eigvalsh(G.double()).clamp(min=0); eGc = torch.linalg.eigvalsh(((Gc + Gc.T) / 2).double()).clamp(min=0)
res["factor_spectra"] = {"PR_G": PR(eG), "PR_G_centred": PR(eGc), "share_of_trG_on_wbar_direction": float(V * wbar.dot(wbar) / G.double().trace()),
                         "top6_eig_G_over_max": (torch.sort(eG, descending=True).values[:6] / eG.max()).tolist(),
                         "top6_eig_Gc_over_max": (torch.sort(eGc, descending=True).values[:6] / eGc.max()).tolist()}
log("factor spectra:", {k: v for k, v in res["factor_spectra"].items() if not k.startswith("top")})
def jrms(x):   # Jacobian of g1 * x / r at x (d x d)
    r = torch.sqrt(x.pow(2).mean() + EPS); return g1[:, None] * (torch.eye(DIM) / r - torch.outer(x, x) / (DIM * r ** 3))
def jrms_mean(X):   # mean over rows q of jrms(X[q]) = diag(g1) (mean(1/r) I - (1/(d N)) sum_q x_q x_q^T / r_q^3)
    r = torch.sqrt(X.pow(2).mean(1) + EPS); C = (X * (1.0 / r ** 3)[:, None]).T @ X / X.shape[0]
    return g1[:, None] * (torch.eye(DIM) * float((1.0 / r).mean()) - C / DIM)
# ---- J_6 (pinned) and J_22 (pooled as d153df93) ------------------------------------------------------------------------
SH = json.load(open(os.path.join(BASE, "score_shrink.json")))
parts = [int(x) for x in re.findall(r"n_valid=(\d+)", open("$JLENS_HOME/whole_model_fit_record.log").read())]; N_REC = sum(parts)
if parts != N_REC_PARTS: stop(f"N_rec parts {parts} != {N_REC_PARTS}")
docs = sorted(d for d in os.listdir(S1) if re.fullmatch(r"doc_\d\d", d))
meta1 = [json.load(open(os.path.join(S1, d, "meta.json"))) for d in docs]; n = [int(m["n_valid"]) for m in meta1]
if docs != [f"doc_{i:02d}" for i in range(1, 17)] or min(n) <= 0: stop(f"Stage 1 docs / n_valid bad: {docs} {n}")
def J_of(l):
    p = os.path.join(JD, f"J_l_{l}.pt"); h = md5(p); res["files"][f"record/J_l_{l}.pt"] = h
    if h != SH["files"].get(f"record/J_l_{l}.pt"): stop(f"record J_l_{l} md5 {h[:8]} != score_shrink.json's")
    J6 = torch.load(p).float()
    for i, d in enumerate(docs):
        q = os.path.join(S1, d, f"J_p_{l}.pt"); hq = md5(q); res["files"][f"stage1/{d}/J_p_{l}.pt"] = hq
        if hq != meta1[i]["J_p_md5"][str(l)]: stop(f"{d} J_p_{l} md5 {hq[:8]} != meta")
    s, tot = torch.load(os.path.join(S1, f"acc_{l}.pt"))
    want = sum(nn * torch.load(os.path.join(S1, d, f"J_p_{l}.pt")).float() for nn, d in zip(n, docs)); rel = float((s.float() - want).norm() / want.norm())
    if int(tot) != sum(n) or rel > 1e-5: stop(f"acc_{l}: total {tot} vs {sum(n)}, rel {rel:.2e}")
    return J6, (N_REC * J6 + want) / (N_REC + sum(n))
I = torch.eye(DIM)
def sqrt_cov(H):
    Hc = H - H.mean(0); S = (Hc.T @ Hc) / (H.shape[0] - 1); e, V = torch.linalg.eigh(S.double())
    ec = e.clamp(min=0); return (V * ec.sqrt()) @ V.T, e
def pr_M(J, D, R, GG=None):   # eigenvalues of M^T M = R D^T J^T G J D R, PR (GG: G_c for the centred report)
    GG = G if GG is None else GG
    K = R @ (D.T.double() @ (J.T.double() @ (GG.double() @ (J.double() @ (D.double() @ R)))))
    e = torch.linalg.eigvalsh((K + K.T) / 2).clamp(min=0); return PR(e), e
out = {}
for l in M2_L:
    ma = mem_avail_gib()
    if ma < 10: stop(f"mid-run memory guard before L{l}: MemAvailable {ma:.1f} GiB < 10")
    J6, J22 = J_of(l); D6, D22 = J6 - I, J22 - I; H = ST["layers"][l].float()
    Rf, ef = sqrt_cov(H); Rh, eh = sqrt_cov(H[half])
    xb = J6 @ H.mean(0); Jx = jrms(xb)
    if l == M2_L[0]:   # control (iii): analytic J_rms == autograd at x_bar
        f = lambda x: g1 * x * torch.rsqrt(x.pow(2).mean() + EPS); Ja = torch.autograd.functional.jacobian(f, xb)
        c3 = float((Ja - Jx).norm() / Ja.norm()); res["controls"]["iii_jrms_vs_autograd_rel_L24"] = c3
        if c3 > 1e-5: stop(f"control (iii): analytic J_rms differs from autograd (rel {c3:.2e})")
    pr_full, eM = pr_M(Jx, D6, Rf)
    xh = J6 @ H[half].mean(0); pr_half, _ = pr_M(jrms(xh), D6, Rh)
    rel = abs(pr_half - pr_full) / pr_full; label = "CONVERGED" if rel <= CONV_TOL else "NOT CONVERGED"
    pr_22, _ = pr_M(jrms(J22 @ H.mean(0)), D22, Rf)
    pr_ps, _ = pr_M(jrms_mean(H @ J6.T), D6, Rf)
    pr_c, eMc = pr_M(Jx, D6, Rf, Gc); pr_ch, _ = pr_M(jrms(xh), D6, Rh, Gc)
    rank = int((ef > 1e-7 * ef.max()).sum()); lo, hi = JETT[l]
    out[l] = {"PR_primary_J6": pr_full, "PR_half_N": pr_half, "rel_change_half_vs_full": rel, "label": label,
              "jett_interval": [lo, hi], "hit": lo <= pr_full <= hi, "graded_as": f"{'HIT' if lo <= pr_full <= hi else 'MISS'} ({label})",
              "centred_report_only": {"PR_centred_J6": pr_c, "PR_centred_half_N": pr_ch, "rel_change": abs(pr_ch - pr_c) / pr_c,
                                      "top10_s2_Mc": torch.sort(eMc, descending=True).values[:10].tolist(), "PR_Sigma_h": PR(ef.clamp(min=0))},
              "report_only": {"PR_J22": pr_22, "PR_per_state_jrms_mean_J6": pr_ps, "top10_s2_M": torch.sort(eM, descending=True).values[:10].tolist()},
              "sigma_h": {"rank_gt_1e-7_max": rank, "eig_max": float(ef.max()), "eig_min": float(ef.min()), "half_rank": int((eh > 1e-7 * eh.max()).sum())}}
    log(f"L{l}: centred PR(M) {pr_c:.1f} (N/2 {pr_ch:.1f}) | PR(Sigma_h) {PR(ef.clamp(min=0)):.0f}")
    log(f"L{l}: PR(M) {pr_full:.0f} (N/2 {pr_half:.0f}, {rel:.1%}, {label}) Jett {lo}-{hi} -> {out[l]['graded_as']} | J22 {pr_22:.0f} | per-state {pr_ps:.0f} | rank Sigma {rank}")
    del J6, J22, D6, D22, H, Rf, Rh; res["layers"] = out; dump()
res["limits"] = ["one model (Swift); WikiText-103 states, 256-token windows (post P123)", "N/d ~ 3.9: sample-covariance spectrum still biased; see the N/2 label",
                 "M2 is report-only; it grades Jett's interval prediction, nothing in the lab gates on it"]
res["verdict"] = "REPORT-ONLY DONE"; res["minutes"] = round((time.time() - T0) / 60, 1)
res["rss_peak_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2); res["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S"); dump(); log("done")
