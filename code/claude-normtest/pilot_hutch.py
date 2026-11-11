"""Claude J-Lens re-fit PILOT (2026-11-04), design Deron 0f2b29a / 1a70281 (card C01). ONE model load.
Writes only to claude-normtest/. Does not modify Qwen's files or the record fit's J_l files.

Question: how many TOTAL random probes K does a Hutchinson-style estimate of J_l need before its READOUT matches
the exact J_l's readout? Reference = the record fit's exact J_l (results-whole-swift-record, mean over the same 6
prompts, n_valid-weighted), so no exact recompute (Claude P038).

Estimator (per prompt p, layer l): cotangent v (Rademacher, same v at every valid target position, exactly where the
exact fit puts its one-hot) -> backward -> g = mean over valid source positions of grad at h_l = v^T J_p.
J_p ~= (1/k) sum v (v^T J_p)   [unbiased: E[v v^T] = I].  Jhat = n_valid-weighted mean over prompts.
Matrix cos vs exact is reported next to the formula 1/sqrt(1+(d-1)/K) (Claude P035): if the code is wrong, cos will
not track the formula, so the formula is this script's self-check.

Sweep: per-prompt k doubles (cumulative, probes are reused), K = 6k. After each round, on HELD-OUT text: top-1
agreement (argmax Jhat == argmax exact) and top-10 overlap (|top10 n top10| / 10) at L24/L40/L52.
FIXED floor (1a70281): top-1 >= 0.90 AND top-10 >= 0.95 at every pilot layer -> K* = that K. Stop there.
Budget cap (Claude card C01 comment): K = 16000 (= N 1000 x k 16). Reaching the cap without clearing the floor
=> K* > budget => escalate/STOP by the rule. Wall-clock cap --max-hours.
Then: peak-GiB probe at long seq (memory only, not a fit).
Usage: python pilot_hutch.py [--smoke] [--max-hours 5]"""
import json, math, os, sys, time, torch
sys.path.insert(0, "$JLENS_HOME")
import whole_model_fit as wmf
import importlib.util
spec = importlib.util.spec_from_file_location("our_jlens", "$JLENS_HOME/our_jlens.py")
oj = importlib.util.module_from_spec(spec); sys.modules[spec.name] = oj; spec.loader.exec_module(oj)
from transformers import AutoTokenizer

SMOKE = "--smoke" in sys.argv
MAX_H = float(sys.argv[sys.argv.index("--max-hours") + 1]) if "--max-hours" in sys.argv else 5.0
OUT = "$JLENS_HOME/claude-normtest"; MD = "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b"
JD = "$JLENS_HOME/results-whole-swift-record"
RES = os.path.join(OUT, "pilot_hutch_smoke.json" if SMOKE else "pilot_hutch.json")
SKIP = oj.SKIP_FIRST_N_POSITIONS
LAYERS = [24, 40, 52]; ROOT = 16; DIM_BATCH = 2; MAX_SEQ = 150
BUDGET_K = 16000
FLOOR_TOP1, FLOOR_TOP10 = 0.90, 0.95
SCHED = [16, 32, 64, 128, 256, 512, 1024, 2048, BUDGET_K // 6]   # cumulative per-prompt k; last = budget
PROMPTS = wmf.PROMPTS
HELD_OUT = [  # same held-out set as eval.py (never in any fit)
    "The committee met on a Thursday to decide whether the old bridge should be repaired or replaced. The engineers had "
    "measured the cracks twice, and both reports agreed that the steel was sound but the concrete deck was failing. Replacing "
    "the whole structure would take three years and close the only road to the hospital, so the vote was not close.",
    "To make the bread rise properly, the dough needs warmth, time, and patience. She mixed the flour with water and a pinch of "
    "salt, folded in the yeast, and covered the bowl with a damp towel. Two hours later it had doubled in size, and the kitchen "
    "smelled the way her grandmother's house used to smell on Sunday mornings.",
    "In 1969 the first humans walked on the Moon, and the photographs they brought back changed how people saw the Earth. Seen "
    "from a quarter of a million miles away, the planet looked small and fragile, a blue marble with no visible borders between "
    "countries. Many historians argue that image helped start the modern environmental movement.",
    "The function takes a list of integers and returns a new list containing only the even numbers, in their original order. "
    "If the input list is empty, it returns an empty list. A common mistake is to modify the list while iterating over it, which "
    "can skip elements; building a fresh list avoids that problem entirely and keeps the code easy to test.",
    "He had promised himself he would not cry at the airport, but when his daughter turned at the security gate and waved one "
    "last time, the promise broke. She was nineteen, moving across the ocean for a job she had worked years to get, and he was "
    "proud of her in a way that hurt. On the drive home he kept the radio off.",
    "Photosynthesis converts light energy into chemical energy stored in sugar. Inside the chloroplast, chlorophyll absorbs mostly "
    "red and blue light and reflects green, which is why leaves look green to us. The process uses carbon dioxide and water and "
    "releases oxygen, which is the source of nearly all the oxygen in the air we breathe.",
]
if SMOKE:
    PROMPTS, HELD_OUT, SCHED, LAYERS = PROMPTS[:2], HELD_OUT[:1], [4, 8], [24, 52]

T0 = time.time()
def log(*a): print(f"[{(time.time()-T0)/60:6.1f}m]", *a, flush=True)
res = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "design": "Deron 0f2b29a/1a70281; Claude pilot", "smoke": SMOKE,
       "layers": LAYERS, "root": ROOT, "dim_batch": DIM_BATCH, "max_seq": MAX_SEQ, "n_prompts": len(PROMPTS),
       "floor": {"top1": FLOOR_TOP1, "top10": FLOOR_TOP10}, "budget_K": BUDGET_K, "rounds": [], "verdict": None}
def save():
    with open(RES, "w") as f: json.dump(res, f, indent=1)

dev = torch.device("cuda:0")
model, tc = wmf.load_whole_model(MD, dev, log)
tok = AutoTokenizer.from_pretrained(MD)
lens = oj.OurJacobianLens(model, tok, add_bos=True)
def _to_fp32(mod, args, kwargs):
    if args: return (args[0].float(),) + tuple(args[1:]), kwargs
    kwargs["hidden_states"] = kwargs["hidden_states"].float(); return args, kwargs
lens.layers[42].register_forward_pre_hook(_to_fp32, with_kwargs=True)
d = lens.d_model; TGT = lens.n_layers - 1
res["load_s"] = round(time.time() - T0)

def readout(h, Jm):   # bug-#6-fixed transport: fp32 throughout (as eval.py)
    return lens.lm_head(lens.final_norm(h.float() @ Jm.T)).float()

# ---- exact reference + held-out h (computed once) ----
EX = {l: torch.load(os.path.join(JD, f"J_l_{l}.pt"), map_location=dev).float() for l in LAYERS}
HO = {l: [] for l in LAYERS}
cap = {}
def mkcap(l):
    def hook(m, a, out): cap[l] = (out[0] if isinstance(out, tuple) else out).detach()
    return hook
hs = [lens.layers[l].register_forward_hook(mkcap(l)) for l in LAYERS]
with torch.no_grad():
    for text in HELD_OUT:
        ids = lens._encode(text, 256); model(input_ids=ids, use_cache=False)
        for l in LAYERS: HO[l].append(cap[l][0, SKIP:ids.shape[1] - 1].float())
for h in hs: h.remove()
HO = {l: torch.cat(v) for l, v in HO.items()}
with torch.no_grad():
    REF1 = {l: readout(HO[l], EX[l]).argmax(-1) for l in LAYERS}
    REF10 = {l: readout(HO[l], EX[l]).topk(10, -1).indices for l in LAYERS}
res["n_heldout_positions"] = int(HO[LAYERS[0]].shape[0])
log(f"reference ready: {res['n_heldout_positions']} held-out positions, layers {LAYERS}")

# ---- probe accumulators ----
acc = [{l: torch.zeros(d, d) for l in LAYERS} for _ in PROMPTS]
kdone = [0] * len(PROMPTS); nval = [0] * len(PROMPTS)
gen = torch.Generator(device=dev); gen.manual_seed(20261104)
t_probe = 0.0; n_probe = 0

@torch.enable_grad()
def add_probes(p, k_new):
    global t_probe, n_probe
    ids = lens._encode(PROMPTS[p], MAX_SEQ); seq = ids.shape[1]
    valid = torch.zeros(seq, dtype=torch.bool); valid[SKIP:seq - 1] = True; nval[p] = int(valid.sum())
    handles, store = lens._capture_residuals(LAYERS, TGT, grad_root_layer=ROOT)
    try:
        model(input_ids=ids.expand(DIM_BATCH, -1), use_cache=False)
        h_tgt = store[TGT]
        n_back = math.ceil(k_new / DIM_BATCH)
        for blk in range(n_back):
            t = time.time()
            V = (torch.randint(0, 2, (DIM_BATCH, d), generator=gen, device=dev) * 2 - 1).to(h_tgt.dtype)
            cot = torch.zeros_like(h_tgt); cot[:, valid, :] = V[:, None, :]
            for l in LAYERS:
                if store[l].grad is not None: store[l].grad.zero_()
            torch.autograd.backward(h_tgt, cot, retain_graph=(blk < n_back - 1))
            for l in LAYERS:
                g = store[l].grad
                assert g is not None, f"no grad at layer {l} (root {ROOT})"
                G = g[:, valid, :].mean(dim=1).float()          # [B, d] = v^T J_p
                acc[p][l] += (V.float().T @ G).cpu()             # sum v (v^T J_p)
            torch.cuda.synchronize(); t_probe += time.time() - t; n_probe += DIM_BATCH
        kdone[p] += n_back * DIM_BATCH
    finally:
        for hd in handles: hd.remove()

def estimate(l):
    W = sum(nval)
    return sum((acc[p][l] / kdone[p]) * (nval[p] / W) for p in range(len(PROMPTS))).to(dev)

def cosF(A, B): return float((A * B).sum() / A.norm() / B.norm())

verdict = None
for k_target in SCHED:
    for p in range(len(PROMPTS)):
        if kdone[p] < k_target: add_probes(p, k_target - kdone[p])
    K = sum(kdone)
    row = {"k_per_prompt": kdone[0], "K": K, "pred_cos": round(1 / math.sqrt(1 + (d - 1) / K), 4),
           "s_per_probe": round(t_probe / max(n_probe, 1), 3),
           "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1), "layers": {}}
    ok = True
    with torch.no_grad():
        for l in LAYERS:
            Jh = estimate(l); lg = readout(HO[l], Jh)
            t1 = float((lg.argmax(-1) == REF1[l]).float().mean())
            t10 = lg.topk(10, -1).indices
            ov = float(torch.stack([torch.isin(t10[i], REF10[l][i]).float().mean() for i in range(t10.shape[0])]).mean())
            row["layers"][l] = {"matrix_cos": round(cosF(Jh, EX[l]), 4), "top1": round(t1, 4), "top10": round(ov, 4)}
            ok = ok and t1 >= FLOOR_TOP1 and ov >= FLOOR_TOP10
            del Jh, lg
    res["rounds"].append(row); save()
    log(json.dumps(row))
    if ok:
        verdict = f"K*={K} clears the floor at every pilot layer; K* <= budget {BUDGET_K} -> PROCEED (Deron adjudicates)"; break
    if (time.time() - T0) / 3600 > MAX_H:
        verdict = f"wall-clock cap {MAX_H} h hit at K={K} without clearing the floor -> K* unresolved; report"; break
if verdict is None:
    verdict = f"floor NOT cleared at the budget K={sum(kdone)} -> K* > budget -> escalate (Hutch++/low-rank) or STOP by 1a70281"
res["verdict"] = verdict; save(); log("VERDICT:", verdict)

# ---- peak-GiB probe at long seq (memory only; one forward + one backward, no fit) ----
del acc; torch.cuda.empty_cache()
LONG = " ".join(PROMPTS + HELD_OUT)
res["mem_probe"] = []
for seq_len, db in ((256, 1), (256, 2), (512, 1), (1024, 1)) if not SMOKE else ((256, 2),):
    torch.cuda.reset_peak_memory_stats()
    rec = {"seq": seq_len, "dim_batch": db}
    try:
        with torch.enable_grad():
            ids = lens._encode(LONG, seq_len); rec["tokens"] = int(ids.shape[1])
            handles, store = lens._capture_residuals(LAYERS, TGT, grad_root_layer=ROOT)
            try:
                model(input_ids=ids.expand(db, -1), use_cache=False)
                cot = torch.ones_like(store[TGT]); t = time.time()
                torch.autograd.backward(store[TGT], cot); torch.cuda.synchronize()
                rec["backward_s"] = round(time.time() - t, 2)
            finally:
                for hd in handles: hd.remove()
        rec["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 1); rec["ok"] = True
    except torch.cuda.OutOfMemoryError as e:
        rec["ok"] = False; rec["error"] = "OOM"; rec["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
    store = None; torch.cuda.empty_cache()
    res["mem_probe"].append(rec); save(); log("mem", json.dumps(rec))
res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S"); res["total_min"] = round((time.time() - T0) / 60, 1); save()
log("DONE", RES)
