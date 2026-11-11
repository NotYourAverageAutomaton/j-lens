"""Claude 2026-11-04 (card C01, Deron P042 low-rank gate; Claude P043). ONE forward-only window, saved for good, so the
low-rank gate, every r, both U choices and the L52 scorer stop-and-look run on CPU afterwards with no GPU window.
Writes only to claude-normtest/. Does not modify Qwen's files or the record fit's J_l files.
  held_h.pt      : {"layers": {l: [N,5120] fp32 for l in 16..62}, "h_final": [N,5120] fp32 (layer 63 output, pre-norm),
                    "ids": [...], "pos": [...], "passage": [...]}  N = held-out positions >= SKIP (eval.py's 6 passages)
  nonfit_hfinal.pt: {"h_final": [M,5120] fp32, "n_docs": 50, "src": "emotion corpus-claude-v1 (never in any fit)"}
  capture_meta.json: md5s, counts, timings."""
import hashlib, json, os, sys, time, torch
sys.path.insert(0, "$JLENS_HOME")
import whole_model_fit as wmf
import importlib.util
spec = importlib.util.spec_from_file_location("our_jlens", "$JLENS_HOME/our_jlens.py")
oj = importlib.util.module_from_spec(spec); sys.modules[spec.name] = oj; spec.loader.exec_module(oj)
from transformers import AutoTokenizer
sys.argv += ["--smoke"] if False else []
OUT = "$JLENS_HOME/claude-normtest"; MD = "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b"; SKIP = oj.SKIP_FIRST_N_POSITIONS
SMOKE = "--smoke" in sys.argv
src = open(os.path.join(OUT, "pilot_hutch.py")).read()
HELD_OUT = eval(src[src.index("HELD_OUT = ["):].split("\n]\n")[0].split("=", 1)[1] + "\n]")   # the SAME 6 passages as eval/pilot
STORIES = []
cdir = "$AUX_DIR/corpus-claude-v1/corpus"
rows = [r for r in (json.loads(l) for f in sorted(os.listdir(cdir)) for l in open(os.path.join(cdir, f)) if l.strip()) if "story" in r]   # 3580 stories; the 300 dialogue rows have no story field
step = max(1, len(rows) // 50); STORIES = [r["story"] for r in rows[::step]][:50]
if SMOKE: HELD_OUT, STORIES = HELD_OUT[:1], STORIES[:2]
T0 = time.time(); log = lambda *a: print(f"[{(time.time()-T0)/60:5.1f}m]", *a, flush=True)
dev = torch.device("cuda:0")
model, tc = wmf.load_whole_model(MD, dev, log)
tok = AutoTokenizer.from_pretrained(MD)
lens = oj.OurJacobianLens(model, tok, add_bos=True)
def _to_fp32(mod, args, kwargs):
    if args: return (args[0].float(),) + tuple(args[1:]), kwargs
    kwargs["hidden_states"] = kwargs["hidden_states"].float(); return args, kwargs
lens.layers[42].register_forward_pre_hook(_to_fp32, with_kwargs=True)
LAYERS = list(range(16, 63)); FINAL = lens.n_layers - 1
cap = {}
def mk(l):
    def hook(m, a, out): cap[l] = (out[0] if isinstance(out, tuple) else out).detach()
    return hook
hs = [lens.layers[l].register_forward_hook(mk(l)) for l in LAYERS + [FINAL]]
held = {"layers": {l: [] for l in LAYERS}, "h_final": [], "ids": [], "pos": [], "passage": []}
with torch.no_grad():
    for pi, text in enumerate(HELD_OUT):
        ids = lens._encode(text, 256); seq = ids.shape[1]; model(input_ids=ids, use_cache=False)
        for l in LAYERS: held["layers"][l].append(cap[l][0, SKIP:seq - 1].float().cpu())
        held["h_final"].append(cap[FINAL][0, SKIP:seq - 1].float().cpu())
        held["ids"] += ids[0].tolist() if pi == 0 else []; held["pos"] += list(range(SKIP, seq - 1)); held["passage"] += [pi] * (seq - 1 - SKIP)
    nonfit = []
    for s in STORIES:
        ids = lens._encode(s, 256); seq = ids.shape[1]
        if seq <= SKIP + 1: continue
        model(input_ids=ids, use_cache=False); nonfit.append(cap[FINAL][0, SKIP:seq - 1].float().cpu())
for h in hs: h.remove()
held["layers"] = {l: torch.cat(v) for l, v in held["layers"].items()}; held["h_final"] = torch.cat(held["h_final"])
# Deron P044 cond.2: the GPU reference. The exact-J readout ids at L24/40/52, computed with the PILOT's own code path
# (lens.final_norm + lens.lm_head on device, fp32 transport). The CPU gate must reproduce these ids exactly, or it stops.
JD = "$JLENS_HOME/results-whole-swift-record"; held["gpu_ref"] = {}
with torch.no_grad():
    for l in (24, 40, 52):
        J = torch.load(os.path.join(JD, f"J_l_{l}.pt"), map_location=dev).float(); h = held["layers"][l].to(dev)
        lg = lens.lm_head(lens.final_norm(h.float() @ J.T)).float()
        held["gpu_ref"][l] = {"top1": lg.argmax(-1).cpu(), "top10": lg.topk(10, -1).indices.cpu(), "top2_gap": (lg.topk(2, -1).values[:, 0] - lg.topk(2, -1).values[:, 1]).cpu()}
        del J, h, lg
tag = "_smoke" if SMOKE else ""
p1 = os.path.join(OUT, f"held_h{tag}.pt"); torch.save(held, p1)
p2 = os.path.join(OUT, f"nonfit_hfinal{tag}.pt"); torch.save({"h_final": torch.cat(nonfit), "n_docs": len(nonfit), "src": "emotion corpus-claude-v1 (never in any fit)"}, p2)
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
meta = {"held_positions": int(held["h_final"].shape[0]), "nonfit_positions": int(sum(x.shape[0] for x in nonfit)), "nonfit_docs": len(nonfit),
        "layers": [LAYERS[0], LAYERS[-1]], "final_layer": FINAL, "md5": {os.path.basename(p1): md5(p1), os.path.basename(p2): md5(p2)},
        "minutes": round((time.time() - T0) / 60, 1), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
json.dump(meta, open(os.path.join(OUT, f"capture_meta{tag}.json"), "w"), indent=1); log("DONE", json.dumps(meta))
