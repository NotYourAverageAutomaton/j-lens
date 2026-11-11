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
JD = sys.argv[1]; SMOKE = False
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
LAYERS = [24, 32, 40, 52, 60]
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

# H-mem (pre-stated post P028 BEFORE this ran): fit-passage theme words, fixed list
THEME = """lighthouse lantern lamp vigil patrol patrols climb climbed climbing keeper beacon dawn routine glass lens ledger
gel gels gene genes allele alleles mutation mutant dna band bands researcher reconcile protein assay
risk risks stakes threat threats menace danger trade tradeoff path paths
series converge converges convergence ratio limit proof prove divergent sum
silence silent argument quarrel quiet fight
map maps territory scale smaller represent useful
值班 巡检 巡查 灯塔 每日 风险 威胁 基因""".split()
TH = set(w.lower() for w in THEME)
norm = lambda t: t.strip().lower().strip(".,;:!?'\"()[]{}")
out = {"theme_words": len(TH), "layers": {}}
with torch.no_grad():
    for l in LAYERS:
        hits = {"jlens": 0, "logit": 0}; tot = 0; ex = []
        for ti, text in enumerate(HELD_OUT):
            ids = lens._encode(text, 256); seq = ids.shape[1]; model(input_ids=ids, use_cache=False)
            h = store[l][0]
            for q in range(SKIP, seq - 1, 4):
                jt = [tok.decode([int(x)]) for x in readout(h[q:q+1], J(l))[0].topk(20).indices]
                lt = [tok.decode([int(x)]) for x in readout(h[q:q+1], None)[0].topk(20).indices]
                hits["jlens"] += sum(norm(t) in TH for t in jt); hits["logit"] += sum(norm(t) in TH for t in lt); tot += 20
                if q == SKIP + 20 and ti < 3: ex.append({"text": text[:40], "jlens": jt[:10], "logit": lt[:10]})
        out["layers"][l] = {"jlens_theme_rate": hits["jlens"] / tot, "logit_theme_rate": hits["logit"] / tot, "n_tokens": tot, "examples": ex}
        print(f"L{l}: theme-word rate  J {hits['jlens']/tot:.4f}  logit {hits['logit']/tot:.4f}  (n={tot})", flush=True)
        for e in ex: print("   ", e["text"], "| J:", e["jlens"], "| logit:", e["logit"], flush=True)
open(f"{OUT}/memcheck.json", "w").write(json.dumps(out, indent=2, ensure_ascii=False)); print("MEMCHECK DONE", flush=True)
