"""Weight diff (bars Deron P051 / vault 09fc513: size-normalised C>=1.6, layer bootstrap, every-4th down_proj E>=.5 in >=2/3): Swift-1.5 vs base Qwen3.8-27B (Claude 2026-11-04; Niz TG 17:00/17:04; pre-registration Deron, see post P050). CPU only.
Streams tensor by tensor from both safetensors checkpoints (never more than two tensors in RAM). Text decoder only (no visual/mtp).
Per tensor: rel = ||Ws - Wb||_F / ||Wb||_F, cos(Ws, Wb), shapes must match. Per layer x module rollups. On a seeded sample of
large 2-D matrices: share of ||ΔW||^2 in the top 1% of singular values (low-rank test).
Writes weight_diff.json + weight_diff_tensors.jsonl. Usage: python weight_diff.py [--smoke]"""
import json, os, re, sys, time, random, torch
from safetensors import safe_open
torch.set_num_threads(max(1, os.cpu_count() - 4))
S = "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b"; B = "$MODELS_DIR/bf16/Qwen3.8-27B"; OUT = "$JLENS_HOME/claude-normtest"
SMOKE = "--smoke" in sys.argv; random.seed(20261104)
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
mapS = json.load(open(os.path.join(S, "model.safetensors.index.json")))["weight_map"]
mapB = json.load(open(os.path.join(B, "model.safetensors.index.json")))["weight_map"]
keep = lambda k: not re.search(r"(^|\.)(visual|mtp)\.", k)
keysS = {k for k in mapS if keep(k)}; keysB = {k for k in mapB if keep(k)}
res = {"only_in_swift": sorted(keysS - keysB)[:50], "only_in_base": sorted(keysB - keysS)[:50], "n_common": len(keysS & keysB)}
log("common", res["n_common"], "only swift", len(keysS - keysB), "only base", len(keysB - keysS))
keys = sorted(keysS & keysB, key=lambda k: (mapS[k], k))
if SMOKE: keys = keys[:12]
def layer_of(k):
    m = re.search(r"\.layers\.(\d+)\.", k); return int(m.group(1)) if m else -1
def module_of(k):
    for pat, name in ((r"self_attn\.q_proj", "attn.q"), (r"self_attn\.k_proj", "attn.k"), (r"self_attn\.v_proj", "attn.v"), (r"self_attn\.o_proj", "attn.o"),
                      (r"linear_attn\.in_proj", "gdn.in_proj"), (r"linear_attn\.out_proj", "gdn.out_proj"), (r"linear_attn\.", "gdn.other"),
                      (r"mlp\.gate_proj", "mlp.gate"), (r"mlp\.up_proj", "mlp.up"), (r"mlp\.down_proj", "mlp.down"), (r"norm", "norm"),
                      (r"embed_tokens", "embed"), (r"lm_head", "lm_head")):
        if re.search(pat, k): return name
    return "other"
handles = {}
def mem_free_gb():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable:"): return int(l.split()[1]) / 2**20
    return 0.0
def get(root, mp, k):
    f = os.path.join(root, mp[k])
    if (root, f) not in handles: handles[(root, f)] = safe_open(f, "pt")
    return handles[(root, f)].get_tensor(k).float()
rows = []; big = []
with open(os.path.join(OUT, "weight_diff_tensors" + ("_smoke" if SMOKE else "") + ".jsonl"), "w") as fo:
    for i, k in enumerate(keys):
        if mem_free_gb() < 20: log("ABORT: MemAvailable < 20 GB (Deron P051 safety)"); sys.exit(3)
        ws, wb = get(S, mapS, k), get(B, mapB, k)
        if ws.shape != wb.shape: r = {"k": k, "shape_mismatch": [list(ws.shape), list(wb.shape)]}
        else:
            nd2 = nb2 = dot = ns2 = 0.0
            for a, b in zip(ws.split(16384), wb.split(16384)):       # chunked: embed/lm_head are 248k x 5120
                dd = a - b; nd2 += float(dd.pow(2).sum()); nb2 += float(b.pow(2).sum()); ns2 += float(a.pow(2).sum()); dot += float((a * b).sum())
            r = {"k": k, "layer": layer_of(k), "module": module_of(k), "shape": list(ws.shape), "numel": ws.numel(),
                 "rel": (nd2 ** .5) / max(nb2 ** .5, 1e-30), "cos": dot / max((ns2 * nb2) ** .5, 1e-30), "dnorm2": nd2, "bnorm2": nb2}
        rows.append(r); fo.write(json.dumps(r) + "\n")
        if i % 50 == 0: log(f"{i}/{len(keys)} {k} rel={r.get('rel', float('nan')):.3e}")
        del ws, wb
tot = sum(r.get("dnorm2", 0) for r in rows)
by_layer = {}
for r in rows:
    if "dnorm2" in r and r["layer"] >= 0: by_layer.setdefault(r["layer"], []).append(r)
L = {l: {"dnorm2": sum(x["dnorm2"] for x in v), "bnorm2": sum(x["bnorm2"] for x in v), "rel_mean": sum(x["rel"] for x in v) / len(v)} for l, v in sorted(by_layer.items())}
res["by_layer"] = L
res["by_module"] = {}
for r in rows:
    if "dnorm2" in r: m = res["by_module"].setdefault(r["module"], {"dnorm2": 0, "rels": []}); m["dnorm2"] += r["dnorm2"]; m["rels"].append(r["rel"])
res["by_module"] = {m: {"share_of_delta": v["dnorm2"] / max(tot, 1e-30), "rel_mean": sum(v["rels"]) / len(v["rels"]), "n": len(v["rels"])} for m, v in res["by_module"].items()}
top10 = sorted(L, key=lambda l: L[l]["dnorm2"], reverse=True)[:10]
totL = sum(L[l]["dnorm2"] for l in L); totB = sum(L[l]["bnorm2"] for l in L)
dsh = sum(L[l]["dnorm2"] for l in top10) / max(totL, 1e-30); bsh = sum(L[l]["bnorm2"] for l in top10) / max(totB, 1e-30)
res["H_concentrate"] = {"top10_layers": top10, "top10_share_dW2": dsh, "top10_share_Wb2": bsh, "C": dsh / max(bsh, 1e-30), "pass": dsh / max(bsh, 1e-30) >= 1.6}
# (b) H-mid: band 20-44 vs outside, layer-level rel_mean, bootstrap over LAYERS (two-sided 95% CI excludes 0)
inb = [L[l]["rel_mean"] for l in L if 20 <= l <= 44]; outb = [L[l]["rel_mean"] for l in L if not 20 <= l <= 44]
g = torch.Generator().manual_seed(20261104); diffs = []
tin, tout = torch.tensor(inb), torch.tensor(outb)
for _ in range(10000):
    diffs.append(float(tin[torch.randint(len(inb), (len(inb),), generator=g)].mean() - tout[torch.randint(len(outb), (len(outb),), generator=g)].mean()))
diffs.sort(); lo, hi = diffs[249], diffs[9749]
res["H_mid"] = {"band_mean": float(tin.mean()), "outside_mean": float(tout.mean()), "diff": float(tin.mean() - tout.mean()), "ci95": [lo, hi], "pass": lo > 0 or hi < 0}
res["layer_rel_mean"] = {l: L[l]["rel_mean"] for l in L}
# low-rank test on a seeded sample of large matrices (SVD is the slow part)
sample = [k for k in keys if module_of(k) == "mlp.down" and layer_of(k) % 4 == 0][: (2 if SMOKE else 99)]; lr = {}
for k in sample:
    d = get(S, mapS, k) - get(B, mapB, k); s = torch.linalg.svdvals(d); e = s.pow(2); n1 = max(1, int(0.01 * len(s)))
    lr[k] = {"module": module_of(k), "layer": layer_of(k), "top1pct_share": float(e[:n1].sum() / e.sum()), "top16_share": float(e[:16].sum() / e.sum())}
    log("svd", k, f"{lr[k]['top1pct_share']:.3f}")
res["lowrank_sample"] = lr; hitsE = sum(1 for v in lr.values() if v["top1pct_share"] >= .50)
res["H_lowrank"] = {"n_sampled": len(lr), "n_E_ge_50": hitsE, "pass": len(lr) > 0 and hitsE >= (2 / 3) * len(lr)}; res["total_delta_norm2"] = tot; res["minutes"] = round((time.time() - T0) / 60, 1)
json.dump(res, open(os.path.join(OUT, "weight_diff" + ("_smoke" if SMOKE else "") + ".json"), "w"), indent=1)
log("DONE", "C", round(res["H_concentrate"]["C"], 3), "H_mid", res["H_mid"]["pass"], "H_lowrank", res["H_lowrank"]["pass"], "| modules", {m: round(v["share_of_delta"], 3) for m, v in res["by_module"].items()})
