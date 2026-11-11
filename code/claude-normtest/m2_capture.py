"""J-Lens M2 state capture (Claude 2026-11-07; estimator Deron P112 item 2, state source pinned Deron P123; card C01). GPU, read-only
forwards, the SAME path as tl_train / step2_capture: wmf loader + check_provenance, our_jlens _encode with BOS, the L42 fp32 pre-hook,
forward hooks on layer outputs, TF32 off. Captures only; the PR(M) estimator runs later on CPU from this file.
SOURCE (post P123): WikiText-103 train (zip sha256 91c00ae2), articles >= 400 words in tl_train's seeded order (seed 20261107), the same
disjointness drop (held 6 + record 6 + Stage-1 16; 12-word shingles + normalised-text sha256), at 256 tokens (BOS + 255), positions
16..254 = 239 per article. Take kept articles in order until >= 20,000 states (84 articles = 20,076).
Gates, each a STOP: zip sha; held_h 3ca0f99c / pilot_hutch 5fa18d9b pins; text counts 6/6/16; MemAvailable >= 80 GiB; capture consistency:
the 6 HELD_OUT re-captured here must equal held_h.pt (rel Frobenius <= 1e-3) at L24, L40, L52 and h_final (post P123: held_h has L52); every
article yields exactly 239 positions.
Output m2_states.pt = {"layers": {24|40|52: [N,5120] fp16}, "h_final": [N,5120] fp16, "article": [N] (0-based order in this capture),
"used_article_indices": [84 corpus indices]} plus m2_capture_meta.json (md5s, consistency values, counts, dropped).
Usage: python m2_capture.py"""
import ast, hashlib, json, os, re, sys, time, zipfile, torch
sys.path.insert(0, "$JLENS_HOME")
import whole_model_fit as wmf
import importlib.util
spec = importlib.util.spec_from_file_location("our_jlens", "$JLENS_HOME/our_jlens.py")
oj = importlib.util.module_from_spec(spec); sys.modules[spec.name] = oj; spec.loader.exec_module(oj)
BASE = "$JLENS_HOME/claude-normtest"; MD = "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b"
ZIP = os.path.join(BASE, "tl_corpus", "wikitext-103-raw-v1.zip"); ZIP_SHA = "91c00ae287f0d699"
M2_L = [24, 40, 52]; SKIP = oj.SKIP_FIRST_N_POSITIONS; MAXLEN = 256; NPER = MAXLEN - 1 - SKIP; N_MIN = 20000; SEED = 20261107; CONSIST_MAX = 1e-3
OUTP = os.path.join(BASE, "m2_states.pt"); METAP = os.path.join(BASE, "m2_capture_meta.json")
T0 = time.time(); log = lambda *a: print(f"[{(time.time()-T0)/60:6.1f}m]", *a, flush=True)
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
meta = {"script_md5": md5(__file__), "prereg": "Deron P112 item 2 + post P123", "verdict": "INCOMPLETE", "files": {}}
def dump(): json.dump(meta, open(METAP, "w"), indent=1)
def stop(why): meta["verdict"] = "NO-RUN: " + why; dump(); log("STOP:", why); sys.exit(3)
def mem_avail_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"): return int(line.split()[1]) / 2**20
# ---- pins, memory guard -------------------------------------------------------------------------------------------------
sha = hashlib.sha256(open(ZIP, "rb").read()).hexdigest(); meta["zip_sha256"] = sha
if not sha.startswith(ZIP_SHA): stop(f"corpus zip sha256 {sha[:16]} != {ZIP_SHA}")
for name, want in {"held_h.pt": "3ca0f99c", "pilot_hutch.py": "5fa18d9b"}.items():
    got = md5(os.path.join(BASE, name)); meta["files"][name] = got
    if not got.startswith(want): stop(f"pin {name} {got[:8]} != {want}")
for p in ("$JLENS_HOME/our_jlens.py", "$JLENS_HOME/whole_model_fit.py", os.path.join(BASE, "stage1_corpus.json")): meta["files"][p] = md5(p)
ma = mem_avail_gib(); meta["mem_available_at_start_gib"] = round(ma, 1)
if ma < 80: stop(f"memory guard: MemAvailable {ma:.1f} GiB < 80 (stop Hans's engine first)")
# ---- banned texts + corpus order, exactly as tl_train ------------------------------------------------------------------
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
raw = zipfile.ZipFile(ZIP).read("wikitext-103-raw/wiki.train.raw").decode("utf-8")
arts, cur = [], []
for line in raw.split("\n"):
    if re.fullmatch(r" = [^=].* = ", line) and cur: arts.append("\n".join(cur)); cur = []
    cur.append(line)
if cur: arts.append("\n".join(cur))
arts = [a.strip() for a in arts if len(a.split()) >= 400]
order = torch.randperm(len(arts), generator=torch.Generator().manual_seed(SEED)).tolist(); del raw
meta["corpus"] = {"articles_total_ge400_words": len(arts)}
# ---- model (same path as tl_train / step2_capture) ---------------------------------------------------------------------
wmf.check_provenance(MD)
from transformers import AutoTokenizer
dev = torch.device("cuda:0"); model, tc = wmf.load_whole_model(MD, dev, log)
tok = AutoTokenizer.from_pretrained(MD); lens = oj.OurJacobianLens(model, tok, add_bos=True)
def _to_fp32(mod, args, kwargs):
    if args: return (args[0].float(),) + tuple(args[1:]), kwargs
    kwargs["hidden_states"] = kwargs["hidden_states"].float(); return args, kwargs
lens.layers[42].register_forward_pre_hook(_to_fp32, with_kwargs=True)
torch.backends.cuda.matmul.allow_tf32 = False
FINAL = lens.n_layers - 1; cap = {}
def mk(l):
    def hook(m, a, out): cap[l] = (out[0] if isinstance(out, tuple) else out).detach()
    return hook
hooks = [lens.layers[l].register_forward_hook(mk(l)) for l in M2_L + [FINAL]]
def capture(text):
    ids = lens._encode(text, MAXLEN); seq = ids.shape[1]
    with torch.no_grad(): model(input_ids=ids, use_cache=False)
    return {l: cap[l][0, SKIP:seq - 1].float().cpu() for l in M2_L + [FINAL]}, seq
# ---- capture-consistency STOP at L24, L40, L52, h_final (post P123) ---------------------------------------------------------
HELD = torch.load(os.path.join(BASE, "held_h.pt")); cc = {l: [] for l in M2_L + [FINAL]}
for text in HELD_OUT:
    h, _ = capture(text)
    for l in cc: cc[l].append(h[l])
cons = {}
for l in M2_L + [FINAL]:
    a = torch.cat(cc[l]); b = HELD["h_final"] if l == FINAL else HELD["layers"][l]
    if a.shape != b.shape: stop(f"capture-consistency: layer {l} shape {tuple(a.shape)} != held_h {tuple(b.shape)}")
    cons["h_final" if l == FINAL else str(l)] = float((a - b).norm() / b.norm())
meta["capture_consistency"] = {"bar": f"max rel Frobenius <= {CONSIST_MAX}", "rel_frobenius": cons, "max": max(cons.values())}; dump()
log("capture consistency vs held_h.pt:", cons)
if max(cons.values()) > CONSIST_MAX: stop(f"capture-consistency: max rel Frobenius {max(cons.values()):.2e} > {CONSIST_MAX}")
del HELD, cc
# ---- articles in order until >= N_MIN states ---------------------------------------------------------------------------
out = {l: [] for l in M2_L + [FINAL]}; art, used, dropped, n = [], [], 0, 0
for i in order:
    if n >= N_MIN: break
    a = arts[i]
    if hashlib.sha256(norm_t(a).encode()).hexdigest() in BAN_HASH or shingles(a) & BAN_SH: dropped += 1; continue
    h, seq = capture(a)
    if seq != MAXLEN or h[FINAL].shape[0] != NPER: stop(f"article {i}: seq {seq}, positions {h[FINAL].shape[0]} (want {MAXLEN}, {NPER})")
    for l in out: out[l].append(h[l].half())
    art += [len(used)] * NPER; used.append(i); n += NPER
    if len(used) % 20 == 0: log(f"{len(used)} articles, {n} states (MemAvailable {mem_avail_gib():.1f} GiB)")
for h_ in hooks: h_.remove()
states = {"layers": {l: torch.cat(out[l]) for l in M2_L}, "h_final": torch.cat(out[FINAL]), "article": art, "used_article_indices": used}
torch.save(states, OUTP)
meta.update({"m2_states_md5": md5(OUTP), "states": n, "articles": len(used), "per_article": NPER, "dropped_overlap": dropped,
             "used_article_indices": used, "verdict": "CAPTURED", "minutes": round((time.time() - T0) / 60, 1), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}); dump()
log("done:", meta["verdict"], n, "states from", len(used), "articles; m2_states.pt", meta["m2_states_md5"][:8])
