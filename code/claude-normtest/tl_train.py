"""J-Lens TUNED-LENS baseline, TRAINING pass on the GPU host GPU (Claude 2026-11-07; pre-reg Deron PREREG-tunedlens-and-Sup.md part B, as amended
by post P107: training positive control + one fallback lr; card C01). Built from the paper, not its package.
PAPER (Belrose et al. 2023, "Eliciting Latent Predictions from Transformers with the Tuned Lens", arXiv 2303.08112v5, md5 b480d685):
  sec 3, eq. 8:  "TunedLens_l(h_l) = LogitLens(A_l h_l + b_l)"; "(A_l, b_l)" = "the translator for layer l"
  sec 3, eq. 9:  argmin E[ D_KL( f_>l(h_l) || TunedLens_l(h_l) ) ]  ("a distillation loss, using the final layer distribution as a soft label")
  sec 3, impl.:  "SGD with Nesterov momentum, with a linear learning rate decay schedule over 250 training steps ... base learning rate
                  of 1.0 ... clip gradients to a norm of 1 ... total batch size of 2^18 tokens per optimizer step. We initialize all
                  translators to the identity transform, and use a weight decay of 10^-3."
DEVIATIONS, declared (Deron P107 accepted the budget):
  - 250 steps x 4096 tokens (1,024,000 tokens) instead of 250 x 2^18. Every training token is used once.
  - lr (amended, Deron P110): the paper's lr 1.0 DIVERGES on this model at this batch, measured at smoke (val KL 7.3 -> 33 in
    2 steps; mean ||h|| 65-114 at L20-44 is a plausible reason, report-only). So lr is CHOSEN ON THE VALIDATION SLICE, never on
    the 306: the 16,384 validation positions are split by order into val-A (first 8,192) and val-B (last 8,192). Probes: TL-big
    alone, 25 steps (= exactly the first 25 steps of the full run: same data, same shuffles, same schedule over 250) at lr in
    {0.125, 0.0156, 0.00195, 0.000244}; pick the lowest mean (over the 7 layers) val-A KL at step 25, ties to the larger lr. If
    no grid lr has mean val-A KL below the logit lens's mean val-A KL: VOID (no working lr in the grid), no full run. TL-shuf and
    TL-22 use the chosen lr.
  - Not specified by the paper, chosen here: momentum 0.9; identity init as A = I + R with R = 0 (so weight decay shrinks R, i.e.
    toward the identity, not toward the zero map); gradient clip to norm 1 PER translator (A and b of one layer). Captures run in
    plain fp32 (TF32 off), so captured h matches held_h's capture conditions (checked by the consistency gate).
  - Speed (post P113, as amended by Deron P114): TF32 matmuls in the optimiser's forward/backward (train_step) ONLY (10-bit mantissa
    inputs, fp32 accumulate). kl_rows (val-A lr pick, val-B training control) and every model forward stay plain fp32. The probes'
    4 lr probes train side by side on ONE capture pass of REST (post P115; the post P114 host cache hit the 6 GiB guard), each with its own lens.
  - Corpus: WikiText-103 raw train (wikitext-103-raw-v1.zip sha256 91c00ae2...), not the Pile validation set. Articles in a fixed
    seeded order (seed 20261107), each truncated to 512 tokens (BOS + 511), positions 16..seq-2 used (SKIP as every J-Lens capture).
    An article is dropped if it shares any 12-word shingle or the normalised sha with the 6 held passages, the 6 record prompts or
    the 16 Stage-1 docs. The first 16,384 positions are the VALIDATION slice (never trained on); the next 1,024,000 are training.
OBJECTS (mid-band layers 20..44 by 4; read out through the same final norm + unembed as logits_of, fp32 copies of the weights):
  TL-big   trained on the 1,024,000 training positions, KL(P_final || P_lens) at the same position.
  TL-shuf  same positions, same steps, same seed; each position's target P_final replaced by another position's (a fixed random
           permutation within each 4096 batch). The shuffled-target control.
  TL-22    trained only on the 22 J-Lens docs' positions (6 record prompts at max_seq 150 + 16 Stage-1 docs at max_seq 256, positions
           16..seq-2 = 4466 positions), 250 steps x 4096 sampled with replacement (seed 20261108). REPORT-ONLY.
TRAINING POSITIVE CONTROL (post P107, judged on val-B only per post P110): after training, TL-big's mean KL(P_final || P_lens) on val-B must be
  below the logit lens's at EVERY mid-band layer. If not: VOID (training failed, not a null); tl_score.py refuses to grade a VOID run.
  val-B and val-A KL are logged every 25 steps (the curves).
MEMORY (post P111): every captured h (validation slice, training chunks, TL-22 rows) is held in HOST tensors and moved to the GPU one
  1,024-row micro-batch at a time. On the GPU host host and GPU share physical memory, but the 0.70 CUDA cap (wmf loader) counts only CUDA
  allocations: f254a0c0's lr 1.0 run hit it at the first chunk (82.4 of 85.2 GiB allocated). Same maths, same order, same seeds.
GUARDS: model loads only if MemAvailable >= 80 GiB (Hans's engine must be stopped; the bf16 model is ~54 GB); read-only forwards,
  no weight edits; wmf.check_provenance as stage1_fit.py. Before each training-chunk capture: stop cleanly (meta dumped) if
  MemAvailable < 6 GiB (Deron P108). Chunks are 8 steps (32,768 positions).
CAPTURE-CONSISTENCY STOP (Deron P108): right after the model loads, the 6 HELD_OUT passages are captured with this script's
  capture() at max length 256 (held_h.pt's) and compared with held_h["layers"][l] at the 7 mid-band layers and with h_final.
  STOP if any relative Frobenius difference ||a - b|| / ||b|| exceeds 1e-3; all values are logged. They are captured only, never
  trained on.
OUTPUT tl_out_sel[_smoke]/: translators {big,shuf,22}_L<l>.pt = {"R": [5120,5120] fp32, "b": [5120] fp32}; tl_meta.json (md5 of every
  file, the article list, token counts, unigram top-20, the KL curves, the control verdict, timings).
PROBE-PEAK (Deron P121, after 22:01's OOM in the side-by-side probes at the 85.18 GiB cap): MICRO 1024 -> 512 for the full run (same
  sums, gradient accumulated; only float summation order changes) and expandable_segments. --probe-peak runs the FULL config (BATCH 4096,
  MICRO 512, VAL_N 16384, 4 lenses side by side) for 2 probe steps, logs torch.cuda.max_memory_allocated, device free memory
  (cudaMemGetInfo, which on the GPU host's unified memory also counts host use) after load and its minimum during the probes, and the other GPU
  processes (pid, MiB) at that minimum; then exits. Verdict PEAK-OK, or NO-RUN if peak allocated > 80 GiB. The full run then STOPs
  after load unless tl_out_sel_peak/tl_meta.json is PEAK-OK from this same script md5 and device free now >= max(the smoke's probe peak
  allocated - its allocated just before the probes, its free after load - its minimum free during the probes) + 4 GiB (Deron P122).
PAIRS (Deron, after window 4's probe-peak NO-RUN 23:18: peak 83.45 GiB, model + val alone 72.84): the 4 lr probes run as 2 PAIRS, each
  pair on its own identical pass over REST (same seeds; capture bit-equal, consistency 0.0); each layer's gradient is freed right after its
  optimizer step (momentum kept), so a lens holds R + momentum only; MICRO 256 everywhere (summation order only); BIG and SHUF are saved and
  dropped from the GPU before TL-22. Same maths, same seeds, same outputs.
Usage: python tl_train.py [--smoke | --probe-peak]"""
import hashlib, json, os, re, subprocess, sys, time, zipfile
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # before torch is imported (Deron P121)
import torch
sys.path.insert(0, "$JLENS_HOME")
import whole_model_fit as wmf
import importlib.util
spec = importlib.util.spec_from_file_location("our_jlens", "$JLENS_HOME/our_jlens.py")
oj = importlib.util.module_from_spec(spec); sys.modules[spec.name] = oj; spec.loader.exec_module(oj)
from transformers import AutoTokenizer
import ast
SMOKE = "--smoke" in sys.argv; PEAK = "--probe-peak" in sys.argv; LR = None   # chosen on val-A (Deron P110)
LR_GRID = [0.125, 0.0156, 0.00195, 0.000244]
BASE = "$JLENS_HOME/claude-normtest"; MD = "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b"
ZIP = os.path.join(BASE, "tl_corpus", "wikitext-103-raw-v1.zip"); ZIP_SHA = "91c00ae287f0d699"
OUT = os.path.join(BASE, "tl_out_sel" + ("_smoke" if SMOKE else "_peak" if PEAK else "")); os.makedirs(OUT, exist_ok=True)
GATE_L = [20, 24, 28, 32, 36, 40, 44]; SKIP = oj.SKIP_FIRST_N_POSITIONS; MAXLEN = 512; DIM = 5120
STEPS = 250 if not SMOKE else 4; BATCH = 4096 if not SMOKE else 512; MICRO = 256
PROBE_STEPS = 25 if not (SMOKE or PEAK) else 2
PEAK_MAX_GIB = 80.0; FREE_MARGIN_GIB = 4.0   # Deron P121
VAL_N = 16384 if not SMOKE else 1024; TRAIN_N = STEPS * BATCH; CHUNK_STEPS = 8; CURVE_EVERY = 25 if not SMOKE else 2   # 8 (Deron P108: memory)
MID_MIN_GIB = 6.0; CONSIST_MAX = 1e-3   # Deron P108: mid-run guard before each chunk capture; capture-consistency STOP bar
MOMENTUM = 0.9; WD = 1e-3; CLIP = 1.0; SEED = 20261107; SEED22 = 20261108; MEM_NEED_GIB = 80.0 if not SMOKE else 60.0
T0 = time.time(); log = lambda *a: print(f"[{(time.time()-T0)/60:6.1f}m]", *a, flush=True)
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
meta = {"script_md5": md5(__file__), "lr": LR, "smoke": SMOKE, "probe_peak": PEAK, "micro": MICRO, "steps": STEPS, "batch": BATCH, "momentum": MOMENTUM, "weight_decay": WD,
        "clip_per_translator": CLIP, "seed": SEED, "seed_tl22": SEED22, "verdict": "INCOMPLETE", "files": {}}
def dump(): json.dump(meta, open(os.path.join(OUT, "tl_meta.json"), "w"), indent=1)
def stop(why): meta["verdict"] = "NO-RUN: " + why; dump(); log("STOP:", why); sys.exit(3)
dev_free_gib = lambda: torch.cuda.mem_get_info()[0] / 2**30
def gpu_procs():   # other processes on the device (pid, MiB), as nvidia-smi reports them
    try: out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
    except Exception as e: return f"nvidia-smi failed: {e}"
    return [{"pid": int(a), "mib": int(b)} for a, b in (l.split(",") for l in out.strip().splitlines() if l.strip()) if int(a) != os.getpid()]
def mem_avail_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"): return int(line.split()[1]) / 2**20
# ---- pins + memory guard ------------------------------------------------------------------------------------------
sha = hashlib.sha256(open(ZIP, "rb").read()).hexdigest()
if not sha.startswith(ZIP_SHA): stop(f"corpus zip sha256 {sha[:16]} != {ZIP_SHA}")
for name, want in {"held_h.pt": "3ca0f99c", "pilot_hutch.py": "5fa18d9b", "eval.py": "cabcfb79"}.items():
    got = md5(os.path.join(BASE, name)); meta["files"][name] = got
    if not got.startswith(want): stop(f"pin {name} {got[:8]} != {want}")
for p in ("$JLENS_HOME/our_jlens.py", "$JLENS_HOME/whole_model_fit.py", os.path.join(BASE, "stage1_corpus.json")): meta["files"][p] = md5(p)
ma = mem_avail_gib(); meta["mem_available_at_start_gib"] = round(ma, 1)
if ma < MEM_NEED_GIB: stop(f"memory guard: MemAvailable {ma:.1f} GiB < {MEM_NEED_GIB} (stop Hans's engine first)")
# ---- texts that must not appear in training: held passages, record prompts, Stage-1 docs ---------------------------
def assigned_list(path, name):
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets): return ast.literal_eval(node.value)
    raise SystemExit(f"no {name} in {path}")
HELD_OUT = assigned_list(os.path.join(BASE, "pilot_hutch.py"), "HELD_OUT")
RECORD = assigned_list("$JLENS_HOME/whole_model_fit.py", "PROMPTS")
STAGE1 = [d["text"] for d in json.load(open(os.path.join(BASE, "stage1_corpus.json")))]
if len(HELD_OUT) != 6 or len(RECORD) != 6 or len(STAGE1) != 16: stop(f"text counts {len(HELD_OUT)}/{len(RECORD)}/{len(STAGE1)} != 6/6/16")
norm_t = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
def shingles(s, k=12):
    w = norm_t(s).split(); return {" ".join(w[i:i + k]) for i in range(max(0, len(w) - k + 1))}
BAN_SH = set().union(*[shingles(t) for t in HELD_OUT + RECORD + STAGE1]); BAN_HASH = {hashlib.sha256(norm_t(t).encode()).hexdigest() for t in HELD_OUT + RECORD + STAGE1}
# ---- corpus: WikiText-103 articles, fixed seeded order --------------------------------------------------------------
raw = zipfile.ZipFile(ZIP).read("wikitext-103-raw/wiki.train.raw").decode("utf-8")
arts, cur = [], []
for line in raw.split("\n"):
    if re.fullmatch(r" = [^=].* = ", line) and cur: arts.append("\n".join(cur)); cur = []
    cur.append(line)
if cur: arts.append("\n".join(cur))
arts = [a.strip() for a in arts if len(a.split()) >= 400]   # long enough to fill 512 tokens
g = torch.Generator().manual_seed(SEED); order = torch.randperm(len(arts), generator=g).tolist()
meta["corpus"] = {"zip_sha256": sha, "articles_total_ge400_words": len(arts)}
del raw
# ---- model (same path as capture_h.py / stage1_fit.py) --------------------------------------------------------------
wmf.check_provenance(MD)
dev = torch.device("cuda:0"); model, tc = wmf.load_whole_model(MD, dev, log)
# EVICT (window 5, 00:09: device-free gate STOP at 7.5 GiB with ~39 GiB of the model files' clean page cache resident; on the GPU host's unified memory
# cudaMemGetInfo counts page cache as used). Drop the model files' cached pages now that the weights are on the device (clean pages only, no root),
# BEFORE measuring, in every mode, so --probe-peak and the full run measure the same way.
import gc; gc.collect()   # release any safetensors mmaps the loader still holds (mapped pages are not dropped)
_cached = lambda: int(next(l for l in open("/proc/meminfo") if l.startswith("Cached:")).split()[1]) / 2**20
meta["cached_gib_before_evict"] = round(_cached(), 1); _ne = 0
for _f in os.listdir(MD):
    try: _fd = os.open(os.path.join(MD, _f), os.O_RDONLY); os.posix_fadvise(_fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(_fd); _ne += 1
    except OSError: pass
time.sleep(5); meta["evicted_model_files_after_load"] = _ne; meta["cached_gib_after_evict"] = round(_cached(), 1)
meta["device_free_after_load_gib"] = round(dev_free_gib(), 2); meta["gpu_procs_after_load"] = gpu_procs(); dump()
log("device free after load", meta["device_free_after_load_gib"], "GiB; other GPU processes", meta["gpu_procs_after_load"])
if not (SMOKE or PEAK):   # Deron P121: gate on device-free headroom measured by --probe-peak, not only on our 0.70 cap
    try: pk = json.load(open(os.path.join(BASE, "tl_out_sel_peak", "tl_meta.json")))
    except OSError: stop("no tl_out_sel_peak/tl_meta.json: run --probe-peak first")
    if pk.get("verdict") != "PEAK-OK" or pk.get("script_md5") != meta["script_md5"]: stop(f"probe-peak verdict {pk.get('verdict')} / script md5 {str(pk.get('script_md5'))[:8]} != this file")
    # post P122: free is sampled AFTER each train_step, so it misses the in-backward transient; the allocator peak sees it
    need = max(pk["probe_peak_allocated_gib"] - pk["alloc_pre_probe_gib"], pk["device_free_after_load_gib"] - pk["probe_min_device_free_gib"]) + FREE_MARGIN_GIB
    meta["device_free_gate"] = {"need_gib": round(need, 2), "have_gib": meta["device_free_after_load_gib"], "from_peak_run": pk.get("ts")}
    if meta["device_free_after_load_gib"] < need: stop(f"device-free gate: {meta['device_free_after_load_gib']:.1f} GiB free after load < {need:.1f} (probe use + {FREE_MARGIN_GIB})")
tok = AutoTokenizer.from_pretrained(MD); lens = oj.OurJacobianLens(model, tok, add_bos=True)
def _to_fp32(mod, args, kwargs):
    if args: return (args[0].float(),) + tuple(args[1:]), kwargs
    kwargs["hidden_states"] = kwargs["hidden_states"].float(); return args, kwargs
lens.layers[42].register_forward_pre_hook(_to_fp32, with_kwargs=True)
FINAL = lens.n_layers - 1; cap = {}
def mk(l):
    def hook(m, a, out): cap[l] = (out[0] if isinstance(out, tuple) else out).detach()
    return hook
hooks = [lens.layers[l].register_forward_hook(mk(l)) for l in GATE_L + [FINAL]]
# fp32 readout weights, identical definition to the CPU logits_of (EPS 1e-6, g1 = 1 + norm.weight)
from safetensors import safe_open
idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
def W(n):
    with safe_open(os.path.join(MD, idx[n]), "pt") as f: return f.get_tensor(n).float()
WU = W("lm_head.weight").to(dev); g1 = (1.0 + W("model.language_model.norm.weight")).to(dev); EPS = 1e-6
def logits_of(x):
    xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * g1
    return xn @ WU.T
torch.backends.cuda.matmul.allow_tf32 = False   # capture must match held_h.pt's conditions
class tf32:   # P113/P114: TF32 inside train_step ONLY; val-KL (kl_rows) and every model forward (capture) run with it OFF
    def __enter__(self): torch.backends.cuda.matmul.allow_tf32 = True
    def __exit__(self, *a): torch.backends.cuda.matmul.allow_tf32 = False
def capture(text, maxlen):
    ids = lens._encode(text, maxlen); seq = ids.shape[1]
    if seq <= SKIP + 1: return None
    torch.backends.cuda.matmul.allow_tf32 = False   # capture: plain fp32, as held_h.pt (checked by the consistency gate)
    with torch.no_grad(): model(input_ids=ids, use_cache=False)
    return {l: cap[l][0, SKIP:seq - 1].float().cpu() for l in GATE_L + [FINAL]}, ids[0, SKIP:seq - 1].cpu()   # host side (post P111: the 0.70 CUDA cap)
# ---- capture-consistency STOP (Deron P108): this capture() must reproduce held_h.pt on the 6 held passages -----------------
HELD = torch.load(os.path.join(BASE, "held_h.pt")); cc = {l: [] for l in GATE_L + [FINAL]}
for text in HELD_OUT:
    h, _ = capture(text, 256)
    for l in cc: cc[l].append(h[l].cpu())
cons = {}
for l in GATE_L + [FINAL]:
    a = torch.cat(cc[l]); b = HELD["h_final"] if l == FINAL else HELD["layers"][l]
    if a.shape != b.shape: stop(f"capture-consistency: layer {l} shape {tuple(a.shape)} != held_h {tuple(b.shape)}")
    cons["h_final" if l == FINAL else str(l)] = float((a - b).norm() / b.norm())
meta["capture_consistency"] = {"bar": f"max rel Frobenius <= {CONSIST_MAX}", "rel_frobenius": cons, "max": max(cons.values())}; dump()
log("capture consistency vs held_h.pt:", cons)
if max(cons.values()) > CONSIST_MAX: stop(f"capture-consistency: max rel Frobenius {max(cons.values()):.2e} > {CONSIST_MAX} (training and scoring h would differ)")
del HELD, cc
# ---- article stream -> validation slice, then training chunks --------------------------------------------------------
used, dropped = [], 0; ai = iter(order); unigram = torch.zeros(WU.shape[0], dtype=torch.long)
def next_article():
    global dropped
    for i in ai:
        a = arts[i]
        if hashlib.sha256(norm_t(a).encode()).hexdigest() in BAN_HASH or shingles(a) & BAN_SH: dropped += 1; continue
        r = capture(a, MAXLEN)
        if r is None: continue
        used.append(i); return r
    stop("corpus exhausted")
def take(n):
    hs = {l: [] for l in GATE_L + [FINAL]}; ids = []; got = 0
    while got < n:
        h, t = next_article()
        for l in hs: hs[l].append(h[l])
        ids.append(t); got += t.shape[0]
    for l in list(hs):                                   # one layer at a time, so only one extra copy is ever alive
        t = torch.cat(hs[l]); hs[l] = t[:n].clone() if t.shape[0] > n else t; del t
    ids = torch.cat(ids)[:n]                             # the tail of the last article is dropped (declared)
    return hs, ids
log("capturing validation slice"); VAL, val_ids = take(VAL_N)
REST = list(ai)   # the article order after the validation slice; every probe and the full run restart from here (same data)
VA = {l: v[:VAL_N // 2] for l, v in VAL.items()}; VB = {l: v[VAL_N // 2:] for l, v in VAL.items()}   # post P110 split by order
del VAL
def kl_rows(h_src, Rb, hfin):   # mean KL(P_final || P_lens) over rows, micro-batched, no grad; plain fp32 (Deron P114: it picks lr + judges the control)
    tot = 0.0
    with torch.no_grad():
        for s in range(0, h_src.shape[0], MICRO):
            hb = h_src[s:s + MICRO].to(dev); z = hb if Rb is None else hb + hb @ Rb[0].T + Rb[1]
            lp_f = torch.log_softmax(logits_of(hfin[s:s + MICRO].to(dev)), -1); lp = torch.log_softmax(logits_of(z), -1)
            tot += float((lp_f.exp() * (lp_f - lp)).sum())
    return tot / h_src.shape[0]
VA_LOGIT_KL = {l: kl_rows(VA[l], None, VA[FINAL]) for l in GATE_L}; VB_LOGIT_KL = {l: kl_rows(VB[l], None, VB[FINAL]) for l in GATE_L}
meta["valA_logit_kl"] = VA_LOGIT_KL; meta["valB_logit_kl"] = VB_LOGIT_KL; log("val-A logit KL", VA_LOGIT_KL, "| val-B", VB_LOGIT_KL)
# ---- translators + optimizers ---------------------------------------------------------------------------------------
def new_lens(lr):
    P = {l: (torch.zeros(DIM, DIM, device=dev, requires_grad=True), torch.zeros(DIM, device=dev, requires_grad=True)) for l in GATE_L}
    opt = {l: torch.optim.SGD(list(P[l]), lr=lr, momentum=MOMENTUM, nesterov=True, weight_decay=WD) for l in GATE_L}
    sch = {l: torch.optim.lr_scheduler.LambdaLR(opt[l], lambda s: max(0.0, 1.0 - s / STEPS)) for l in GATE_L}
    return P, opt, sch
def train_step(P, opt, sch, H, hfin_rows, perm=None):
    """One optimizer step on BATCH rows: KL(P_final || P_lens) per layer, gradient accumulated over MICRO slices, clip per translator."""
    losses = {}
    with tf32():
        return _train_step(P, opt, sch, H, hfin_rows, perm, losses)
def _train_step(P, opt, sch, H, hfin_rows, perm, losses):
    for l in GATE_L:
        opt[l].zero_grad(set_to_none=True); R, b = P[l]; tot = 0.0
        for s in range(0, BATCH, MICRO):
            hb = H[l][s:s + MICRO].to(dev); tgt = (hfin_rows[s:s + MICRO] if perm is None else hfin_rows[perm[s:s + MICRO]]).to(dev)
            with torch.no_grad(): lp_f = torch.log_softmax(logits_of(tgt), -1)
            lp = torch.log_softmax(logits_of(hb + hb @ R.T + b), -1)
            loss = (lp_f.exp() * (lp_f - lp)).sum() / BATCH; loss.backward(); tot += float(loss)
        torch.nn.utils.clip_grad_norm_([R, b], CLIP); opt[l].step(); sch[l].step(); losses[l] = tot
        R.grad = None; b.grad = None   # PAIRS: free the gradient now (zero_grad(set_to_none) would at the next step anyway)
    return losses
# ---- lr probes on val-A (Deron P110): each probe = the first PROBE_STEPS steps of the full run, TL-big only ---------------------
# post P115: the 4 probes train SIDE BY SIDE on one capture pass of REST (no cache: post P114's cache hit the 6 GiB host guard). Each probe has
# its own lens; all see identical chunks, identical chunk shuffles and the same generator draws as the full run's first 25 steps.
def probe_all(lrs):
    global ai
    ai = iter(REST); P = {lr: new_lens(lr) for lr in lrs}; gp = torch.Generator(device="cpu").manual_seed(SEED + 1); st = 0
    while st < PROBE_STEPS:
        n = min(CHUNK_STEPS, STEPS - st) * BATCH
        ma = mem_avail_gib()
        if ma < MID_MIN_GIB: stop(f"mid-run memory guard in the lr probes at step {st}: MemAvailable {ma:.1f} GiB < {MID_MIN_GIB}")
        H, _ = take(n); sh = torch.randperm(n, generator=gp)
        for l in list(H): H[l] = H[l][sh]
        for k in range(n // BATCH):
            if st >= PROBE_STEPS: break
            rows = slice(k * BATCH, (k + 1) * BATCH); _ = torch.randperm(BATCH, generator=gp)   # keep the generator in step with the full run
            Hk = {l: H[l][rows] for l in GATE_L}; fin = H[FINAL][rows]
            for lr in lrs:
                train_step(*P[lr], Hk, fin); f = dev_free_gib()
                if f < meta.get("probe_min_device_free_gib", 1e9): meta["probe_min_device_free_gib"] = round(f, 2); meta["gpu_procs_at_min_free"] = gpu_procs()
            st += 1
        del H; torch.cuda.empty_cache()
    out = {lr: {l: kl_rows(VA[l], (P[lr][0][l][0].detach(), P[lr][0][l][1].detach()), VA[FINAL]) for l in GATE_L} for lr in lrs}
    del P; torch.cuda.empty_cache(); return out
torch.cuda.reset_peak_memory_stats(); meta["alloc_pre_probe_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)   # Deron P122
PROBE_OUT = {}
for pair in (LR_GRID[:2], LR_GRID[2:]): PROBE_OUT.update(probe_all(pair))   # PAIRS: 2 lenses on the GPU at a time
PROBE_OUT = {lr: PROBE_OUT[lr] for lr in LR_GRID}
meta["probe_peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2); dump()
log("probe peak allocated", meta["probe_peak_allocated_gib"], "GiB; min device free", meta.get("probe_min_device_free_gib"), "GiB; procs", meta.get("gpu_procs_at_min_free"))
if PEAK:
    if meta["probe_peak_allocated_gib"] > PEAK_MAX_GIB: stop(f"probe peak allocated {meta['probe_peak_allocated_gib']} GiB > {PEAK_MAX_GIB}")
    meta["verdict"] = "PEAK-OK"; meta["minutes"] = round((time.time() - T0) / 60, 1); meta["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S"); dump()
    log("done: PEAK-OK"); sys.exit(0)
sel = {}
for lr, kl in PROBE_OUT.items():
    sel[str(lr)] = {"valA_kl": kl, "mean": sum(kl.values()) / len(kl)}; log(f"probe lr {lr}: val-A KL", {l: round(v, 4) for l, v in kl.items()}, "mean", round(sel[str(lr)]["mean"], 4))
meta["lr_selection"] = {"grid": LR_GRID, "probe_steps": PROBE_STEPS, "probes": sel, "mode": "side by side, one capture pass (post P115)"}; dump()
best = min(LR_GRID, key=lambda x: (sel[str(x)]["mean"], -x)); logit_mean = sum(VA_LOGIT_KL.values()) / len(VA_LOGIT_KL)
meta["lr_selection"].update({"chosen": best, "chosen_mean_valA_kl": sel[str(best)]["mean"], "logit_mean_valA_kl": logit_mean,
                             "rule": "lowest mean val-A KL at step PROBE_STEPS, ties to the larger lr; VOID if none beats logit's mean val-A KL"})
if sel[str(best)]["mean"] >= logit_mean:
    meta["verdict"] = "VOID: no working lr in the grid (post P110)"; dump(); log("VOID: no grid lr beats the logit lens on val-A"); sys.exit(0)
LR = best; meta["lr"] = LR; dump(); log(f"chosen lr {LR}")
# ---- full run at the chosen lr, from the same start ---------------------------------------------------------------------------
ai = iter(REST); used.clear(); dropped = 0
BIG, SHUF = new_lens(LR), new_lens(LR)
curve = {"step": [], "valB_kl_big": {l: [] for l in GATE_L}, "valA_kl_big": {l: [] for l in GATE_L}, "train_kl_big": {l: [] for l in GATE_L}, "train_kl_shuf": {l: [] for l in GATE_L}}
gs = torch.Generator(device="cpu").manual_seed(SEED + 1); step = 0
while step < STEPS:
    n = min(CHUNK_STEPS, STEPS - step) * BATCH
    ma = mem_avail_gib()
    if ma < MID_MIN_GIB: meta["curve"] = curve; stop(f"mid-run memory guard before chunk at step {step}: MemAvailable {ma:.1f} GiB < {MID_MIN_GIB}")
    log(f"capturing training chunk for steps {step}..{step + n // BATCH - 1} (MemAvailable {ma:.1f} GiB)"); H, ids = take(n)
    unigram += torch.bincount(ids.cpu(), minlength=unigram.shape[0])
    sh = torch.randperm(n, generator=gs)
    for l in list(H): H[l] = H[l][sh]                    # mix articles inside every batch; one layer at a time (host side)
    for k in range(n // BATCH):
        rows = slice(k * BATCH, (k + 1) * BATCH); Hk = {l: H[l][rows] for l in GATE_L}; fin = H[FINAL][rows]
        perm = torch.randperm(BATCH, generator=gs)
        lb = train_step(*BIG, Hk, fin); ls = train_step(*SHUF, Hk, fin, perm=perm); step += 1
        for l in GATE_L: curve["train_kl_big"][l].append(lb[l]); curve["train_kl_shuf"][l].append(ls[l])
        if step % CURVE_EVERY == 0 or step == STEPS:
            curve["step"].append(step)
            for l in GATE_L:
                Rb = (BIG[0][l][0].detach(), BIG[0][l][1].detach())
                curve["valB_kl_big"][l].append(kl_rows(VB[l], Rb, VB[FINAL])); curve["valA_kl_big"][l].append(kl_rows(VA[l], Rb, VA[FINAL]))
            log(f"step {step}: val-B KL big", {l: round(curve["valB_kl_big"][l][-1], 4) for l in GATE_L}); meta["curve"] = curve; dump()
    del H, ids, Hk, fin; torch.cuda.empty_cache()
meta["corpus"].update({"articles_used": len(used), "articles_dropped_overlap": dropped, "train_positions": TRAIN_N, "val_positions": VAL_N,
                       "used_article_indices_first_50": used[:50]})
meta["unigram_top20"] = torch.topk(unigram, 20).indices.tolist()
# ---- training positive control (post P107) --------------------------------------------------------------------------------
VAL_BIG_KL = {l: curve["valB_kl_big"][l][-1] for l in GATE_L}
ok = {l: VAL_BIG_KL[l] < VB_LOGIT_KL[l] for l in GATE_L}
meta["training_control"] = {"rule": "post P107: TL-big val KL(P_final||P_lens) < logit lens val KL at EVERY mid-band layer, else VOID",
                            "judged_on": "val-B (post P110)", "valB_kl_big": VAL_BIG_KL, "valB_kl_logit": VB_LOGIT_KL, "pass_per_layer": ok, "pass": all(ok.values())}
log("training control:", "PASS" if all(ok.values()) else "FAIL", ok)
# ---- save BIG + SHUF, then drop them from the GPU before TL-22 (PAIRS) ----------------------------------------------------
for name, (P, _, _) in (("big", BIG), ("shuf", SHUF)):
    for l in GATE_L:
        p = os.path.join(OUT, f"{name}_L{l}.pt"); torch.save({"R": P[l][0].detach().cpu(), "b": P[l][1].detach().cpu()}, p); meta["files"][f"{name}_L{l}.pt"] = md5(p)
del BIG, SHUF; torch.cuda.empty_cache(); dump()
# ---- TL-22 (report-only): the 22 J-Lens docs' positions, same lr -------------------------------------------------------
H22 = {l: [] for l in GATE_L + [FINAL]}
for text, ml in [(t, 150) for t in RECORD] + [(t, 256) for t in STAGE1]:
    h, _ = capture(text, ml)
    for l in H22: H22[l].append(h[l])
H22 = {l: torch.cat(v) for l, v in H22.items()}; meta["tl22_positions"] = int(H22[FINAL].shape[0])
if not SMOKE and meta["tl22_positions"] != 4466: stop(f"TL-22 positions {meta['tl22_positions']} != 4466 (642 record + 3824 Stage 1)")
T22 = new_lens(LR); g22 = torch.Generator(device="cpu").manual_seed(SEED22)
for s in range(STEPS):
    rows = torch.randint(0, H22[FINAL].shape[0], (BATCH,), generator=g22)
    train_step(*T22, {l: H22[l][rows] for l in GATE_L}, H22[FINAL][rows])
meta["tl22_valB_kl"] = {l: kl_rows(VB[l], (T22[0][l][0].detach(), T22[0][l][1].detach()), VB[FINAL]) for l in GATE_L}
for h in hooks: h.remove()
# ---- save -------------------------------------------------------------------------------------------------------------
for name, (P, _, _) in (("22", T22),):
    for l in GATE_L:
        p = os.path.join(OUT, f"{name}_L{l}.pt"); torch.save({"R": P[l][0].detach().cpu(), "b": P[l][1].detach().cpu()}, p); meta["files"][f"{name}_L{l}.pt"] = md5(p)
meta["verdict"] = "TRAINED" if meta["training_control"]["pass"] else "VOID: training control failed (post P107)"
meta["minutes"] = round((time.time() - T0) / 60, 1); meta["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S"); dump()
log("done:", meta["verdict"])
