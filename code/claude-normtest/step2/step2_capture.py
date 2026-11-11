"""J-Lens shrink STEP 2: capture the six fresh passages (Claude 2026-11-07; PREREG 0466a966 step 2 + amend-1 e088c347; passages Deron
PASS post P119, markup stripped per post P119; card C01). GPU, read-only forwards, the SAME path as capture_h.py / tl_train.py:
wmf loader + check_provenance, our_jlens _encode with BOS, the L42 fp32 pre-hook, forward hooks on layer outputs, TF32 off.
Gates, each a STOP:
  - passages.json: exactly 6, each text's sha256 == its recorded sha256.
  - capture consistency: the 6 HELD_OUT passages re-captured here must equal held_h.pt (rel Frobenius <= 1e-3) at the 7 mid-band
    layers and h_final (as tl_train f254a0c0), so the fresh h are produced exactly like the h every lens was scored on.
  - each fresh passage yields exactly 239 positions (256 tokens, positions 16..254).
Output step2/fresh_h.pt = {"layers": {l: [N,5120] fp32 for l in 20..44 by 4}, "h_final": [N,5120], "passage": [N], "ids": [[256] per
passage], "pos": [N]} plus step2/capture_meta.json (md5s, the consistency values, counts). Memory guard: MemAvailable >= 80 GiB.
Usage: python step2_capture.py"""
import hashlib, json, os, sys, time, torch
sys.path.insert(0, "$JLENS_HOME")
import whole_model_fit as wmf
import importlib.util
spec = importlib.util.spec_from_file_location("our_jlens", "$JLENS_HOME/our_jlens.py")
oj = importlib.util.module_from_spec(spec); sys.modules[spec.name] = oj; spec.loader.exec_module(oj)
from transformers import AutoTokenizer
import ast
BASE = "$JLENS_HOME/claude-normtest"; S2 = os.path.join(BASE, "step2"); MD = "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b"
GATE_L = [20, 24, 28, 32, 36, 40, 44]; SKIP = oj.SKIP_FIRST_N_POSITIONS; MAXLEN = 256; CONSIST_MAX = 1e-3
T0 = time.time(); log = lambda *a: print(f"[{(time.time()-T0)/60:6.1f}m]", *a, flush=True)
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
meta = {"script_md5": md5(__file__), "verdict": "INCOMPLETE"}
def dump(): json.dump(meta, open(os.path.join(S2, "capture_meta.json"), "w"), indent=1)
def stop(why): meta["verdict"] = "NO-RUN: " + why; dump(); log("STOP:", why); sys.exit(3)
def mem_avail_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"): return int(line.split()[1]) / 2**20
P = json.load(open(os.path.join(S2, "passages.json"))); meta["passages_md5"] = md5(os.path.join(S2, "passages.json"))
if len(P) != 6: stop(f"{len(P)} passages, not 6")
for p in P:
    if hashlib.sha256(p["text"].encode()).hexdigest() != p["sha256"]: stop(f"{p['genre']} text sha256 != recorded")
for name, want in {"held_h.pt": "3ca0f99c", "pilot_hutch.py": "5fa18d9b"}.items():
    got = md5(os.path.join(BASE, name)); meta[name] = got
    if not got.startswith(want): stop(f"pin {name} {got[:8]} != {want}")
ma = mem_avail_gib(); meta["mem_available_at_start_gib"] = round(ma, 1)
if ma < 80: stop(f"memory guard: MemAvailable {ma:.1f} GiB < 80 (stop Hans's engine first)")
wmf.check_provenance(MD)
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
hooks = [lens.layers[l].register_forward_hook(mk(l)) for l in GATE_L + [FINAL]]
def capture(text):
    ids = lens._encode(text, MAXLEN); seq = ids.shape[1]
    with torch.no_grad(): model(input_ids=ids, use_cache=False)
    return {l: cap[l][0, SKIP:seq - 1].float().cpu() for l in GATE_L + [FINAL]}, ids[0].cpu(), seq
# consistency gate (as tl_train f254a0c0)
def assigned_list(path, name):
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets): return ast.literal_eval(node.value)
HELD = torch.load(os.path.join(BASE, "held_h.pt")); cc = {l: [] for l in GATE_L + [FINAL]}
for text in assigned_list(os.path.join(BASE, "pilot_hutch.py"), "HELD_OUT"):
    h, _, _ = capture(text)
    for l in cc: cc[l].append(h[l])
cons = {}
for l in GATE_L + [FINAL]:
    a = torch.cat(cc[l]); b = HELD["h_final"] if l == FINAL else HELD["layers"][l]
    if a.shape != b.shape: stop(f"capture-consistency: layer {l} shape {tuple(a.shape)} != held_h {tuple(b.shape)}")
    cons["h_final" if l == FINAL else str(l)] = float((a - b).norm() / b.norm())
meta["capture_consistency"] = {"bar": f"max rel Frobenius <= {CONSIST_MAX}", "rel_frobenius": cons, "max": max(cons.values())}; dump()
log("capture consistency vs held_h.pt:", cons)
if max(cons.values()) > CONSIST_MAX: stop(f"capture-consistency: max rel Frobenius {max(cons.values()):.2e} > {CONSIST_MAX}")
del HELD, cc
# the six fresh passages
out = {"layers": {l: [] for l in GATE_L}, "h_final": [], "passage": [], "ids": [], "pos": []}
for i, p in enumerate(P):
    h, ids, seq = capture(p["text"])
    n = h[FINAL].shape[0]
    if seq != MAXLEN or n != MAXLEN - 1 - SKIP: stop(f"{p['genre']}: seq {seq}, positions {n} (want {MAXLEN}, {MAXLEN - 1 - SKIP})")
    for l in GATE_L: out["layers"][l].append(h[l])
    out["h_final"].append(h[FINAL]); out["passage"] += [i] * n; out["ids"].append(ids.tolist()); out["pos"] += list(range(SKIP, seq - 1))
    log(f"{p['genre']}: {n} positions")
for h_ in hooks: h_.remove()
out["layers"] = {l: torch.cat(v) for l, v in out["layers"].items()}; out["h_final"] = torch.cat(out["h_final"])
fp = os.path.join(S2, "fresh_h.pt"); torch.save(out, fp)
meta.update({"fresh_h_md5": md5(fp), "positions": int(out["h_final"].shape[0]), "per_passage": [MAXLEN - 1 - SKIP] * 6,
             "verdict": "CAPTURED", "minutes": round((time.time() - T0) / 60, 1), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}); dump()
log("done:", meta["verdict"], meta["positions"], "positions, fresh_h.pt", meta["fresh_h_md5"][:8])
