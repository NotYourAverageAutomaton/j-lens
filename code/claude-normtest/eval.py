"""Claude post-fit eval for the J-Lens record fit (2026-11-03). ONE model load. Writes only to claude-normtest/.
Does not modify Qwen's files or the record fit's J_l files.
 1. Readouts (the record script's readout step, with bug #6 fixed: transport kept in fp32 because final_norm and
    lm_head are fp32): top-20 tokens at readout layers x 4 positions x the 6 fit prompts -> readouts.jsonl
 2. J1 test on HELD-OUT text (not the fit prompts): per layer, J-lens vs logit lens vs a wrong-layer-J control:
    - kl:     mean KL(P_final || P_lens) over positions >= 16 (lower = closer to the model's own output)
    - top1:   agreement of the lens top-1 with the final top-1 next token
    - fut20:  fraction of the next 10 actual tokens that appear in the lens top-20 (what the model is primed to say)
Usage: python eval.py J_DIR [--smoke]"""
import json, math, os, sys, time, torch
sys.path.insert(0, "$JLENS_HOME")
import whole_model_fit as wmf
import importlib.util
spec = importlib.util.spec_from_file_location("our_jlens", "$JLENS_HOME/our_jlens.py")
oj = importlib.util.module_from_spec(spec); sys.modules[spec.name] = oj; spec.loader.exec_module(oj)
from transformers import AutoTokenizer
JD = sys.argv[1]; SMOKE = "--smoke" in sys.argv
OUT = "$JLENS_HOME/claude-normtest"; MD = "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b"
SKIP = oj.SKIP_FIRST_N_POSITIONS
HELD_OUT = [  # written by Claude for this eval; never used in any fit
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
LAYERS = [16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 62]
READOUT_LAYERS = [16, 20, 24, 28, 32, 36, 40, 44, 48, 54, 60, 62]
if SMOKE:
    HELD_OUT, LAYERS, READOUT_LAYERS = HELD_OUT[:1], [16, 44], [16, 44]
dev = torch.device("cuda:0")
model, tc = wmf.load_whole_model(MD, dev, lambda s: print(s, flush=True))
tok = AutoTokenizer.from_pretrained(MD)
lens = oj.OurJacobianLens(model, tok, add_bos=True)
def _to_fp32(mod, args, kwargs):
    if args: return (args[0].float(),) + tuple(args[1:]), kwargs
    kwargs["hidden_states"] = kwargs["hidden_states"].float(); return args, kwargs
lens.layers[42].register_forward_pre_hook(_to_fp32, with_kwargs=True)
store = {}
def mk(l):
    def hook(m, a, out): store[l] = (out[0] if isinstance(out, tuple) else out).detach()
    return hook
need = sorted(set(LAYERS) | set(READOUT_LAYERS))
for l in need: lens.layers[l].register_forward_hook(mk(l))
Jc = {}
def J(l):
    if l not in Jc: Jc[l] = torch.load(os.path.join(JD, f"J_l_{l}.pt"), map_location=dev).float()
    return Jc[l]
def readout(h, Jm):   # bug-#6-fixed transport: everything fp32
    t = h.float() @ Jm.T if Jm is not None else h.float()
    return lens.lm_head(lens.final_norm(t)).float()
CTRL = {l: (62 if l < 40 else 16) for l in LAYERS}   # wrong-layer J control
res = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "j_dir": JD, "smoke": SMOKE, "layers": {}, "n_positions": 0}
acc = {l: {k: [0.0, 0.0, 0.0] for k in ("jlens", "logit", "ctrl")} for l in LAYERS}   # kl, top1, fut20 sums
npos = 0; nfut = 0
with torch.no_grad():   # SELFCHECK: J = identity must reproduce the logit lens exactly (else the comparison is broken)
    ids0 = lens._encode(HELD_OUT[0], 256); model(input_ids=ids0, use_cache=False)
    h0 = store[LAYERS[0]][0]; eye = torch.eye(h0.shape[-1], device=dev)
    d = (readout(h0, eye) - readout(h0, None)).abs().max().item()
    res["selfcheck_identity_maxdiff"] = d
    print(f"SELFCHECK identity-J vs logit lens max|diff| = {d:.3e}", flush=True)
    del eye
with torch.no_grad():
    for text in HELD_OUT:
        ids = lens._encode(text, 256); seq = ids.shape[1]
        logits = model(input_ids=ids, use_cache=False).logits[0].float()
        lp_final = torch.log_softmax(logits, -1); top1_final = logits.argmax(-1)
        pos = list(range(SKIP, seq - 1)); npos += len(pos)
        for l in LAYERS:
            h = store[l][0]
            for name, Jm in (("jlens", J(l)), ("logit", None), ("ctrl", J(CTRL[l]))):
                lg = readout(h, Jm); lp = torch.log_softmax(lg, -1)
                p = torch.tensor(pos, device=dev)
                kl = (lp_final[p].exp() * (lp_final[p] - lp[p])).sum(-1)
                acc[l][name][0] += kl.sum().item()
                acc[l][name][1] += (lg[p].argmax(-1) == top1_final[p]).float().sum().item()
                t20 = lg[p].topk(20, -1).indices
                f = 0
                for i, q in enumerate(pos):
                    nxt = ids[0, q + 1: q + 11]
                    f += torch.isin(nxt, t20[i]).float().mean().item()
                acc[l][name][2] += f
            if l == LAYERS[0]: nfut += sum(1 for q in pos if q + 1 < seq)
for l in LAYERS:
    res["layers"][l] = {k: {"kl": acc[l][k][0] / npos, "top1": acc[l][k][1] / npos, "fut20": acc[l][k][2] / max(nfut, 1)}
                        for k in acc[l]}
    r = res["layers"][l]
    print(f"L{l:2d}  KL j/logit/ctrl {r['jlens']['kl']:.2f}/{r['logit']['kl']:.2f}/{r['ctrl']['kl']:.2f}  "
          f"top1 {r['jlens']['top1']:.3f}/{r['logit']['top1']:.3f}/{r['ctrl']['top1']:.3f}  "
          f"fut20 {r['jlens']['fut20']:.3f}/{r['logit']['fut20']:.3f}/{r['ctrl']['fut20']:.3f}", flush=True)
res["n_positions"] = npos
# readouts on the fit prompts (the record script's step, dtype fixed)
rows = []
with torch.no_grad():
    for pr in (wmf.PROMPTS[:1] if SMOKE else wmf.PROMPTS):
        ids = lens._encode(pr, 150); seq = ids.shape[1]
        model(input_ids=ids, use_cache=False)
        vpos = list(range(SKIP, seq - 1)); vpos = vpos[:: max(1, len(vpos) // 4)]
        for l in READOUT_LAYERS:
            lg = readout(store[l][0], J(l)); lgl = readout(store[l][0], None)
            for q in vpos:
                rows.append({"prompt": pr[:60], "layer": l, "pos": q, "context_tail": tok.decode(ids[0, max(0, q - 8): q + 1]),
                             "jlens_top20": [tok.decode([int(x)]) for x in lg[q].topk(20).indices],
                             "logit_top20": [tok.decode([int(x)]) for x in lgl[q].topk(20).indices]})
tag = "smoke" if SMOKE else "record"
open(f"{OUT}/eval_{tag}.json", "w").write(json.dumps(res, indent=2))
with open(f"{OUT}/readouts_{tag}.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(f"EVAL DONE {tag}: {npos} positions, {len(rows)} readout rows -> {OUT}", flush=True)
