"""J-Lens CONTEXT TABLE (Claude 2026-11-08; Deron P127 ask 1; card C01). REPORT-ONLY: nothing here gates anything. CPU, file-only.
One table on the SAME 306 held positions, per mid-band layer L20..44 by 4: fut20 and top-1 for
  shrunk  = the frozen-a shrunk lens (family S, a per layer = score_shrink.json report.frozen_step2_parameters.real_per_layer, the step-2 a)
  logit, TL-big, TL-shuf, unigram (the fixed top-20 from tl_meta; its top-1 = the single most frequent training token, unigram_top20[0]).
SOURCES. Every cell but one is READ from a graded file, never recomputed:
  shrunk          score_shrink.json 0a925a72 report.readout_table.n22["a=<frozen a>"]      (step2_score control (b) reproduced these rows)
  logit / TL-big / TL-shuf   tl_out_sel/score_tl.json 183150ea report.per_layer_table
  unigram fut20   score_tl.json report.unigram_top20.fut20 (one value; a fixed top-20 does not depend on the layer)
  unigram top-1   COMPUTED here: mean(TOP1_FINAL == unigram_top20[0]), TOP1_FINAL built by the tl_score 368b8b09 lines copied verbatim
                  (logits_of = final RMSNorm with g1 = 1 + norm.weight, then lm_head; argmax of log_softmax of h_final).
CONTROLS, each a STOP (so the one computed cell sits on the same positions and the same top-1 path as the read cells):
  c1 cross-file: score_shrink n22 "a=0.0" (fut20, top1) == score_tl logit (fut20, top1) at every layer, |diff| <= 1e-6 (a=0 IS the logit lens).
  c2 positions: unigram fut20 recomputed here from nexts == score_tl unigram fut20, |diff| <= 1e-6.
  c3 top-1 path: logit lens recomputed here from held_h.pt (fut20, top1) == score_tl logit row at every layer, |diff| <= 1e-6.
  pins: held_h 3ca0f99c, pilot_hutch 5fa18d9b, eval.py cabcfb79, score_shrink 0a925a72, score_tl 183150ea, tl_meta 0e7c0820; HELD_OUT equal in
  pilot_hutch and eval.py; 306 positions. Memory guard MemAvailable >= 12 GiB (lm_head fp32 ~3 GiB; pad >= 8 GiB per post P127).
Output step2/ctx_table.json (+ the markdown table in the log). Usage: SCORE_THREADS=4 nice -n 10 python ctx_table.py"""
import ast, hashlib, json, os, platform, sys, time, torch
from safetensors import safe_open
from transformers import AutoTokenizer
torch.set_num_threads(int(os.environ.get("SCORE_THREADS", 4)))
BASE = "$JLENS_HOME/claude-normtest"; MD = "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b"; TLDIR = os.path.join(BASE, "tl_out_sel")
OUTP = os.path.join(BASE, "step2", "ctx_table.json")
GATE_L = [20, 24, 28, 32, 36, 40, 44]; EPS = 1e-6; SKIP = 16; TOL = 1e-6; NEED_GIB = 12.0
PIN = {"held_h.pt": "3ca0f99c", "pilot_hutch.py": "5fa18d9b", "eval.py": "cabcfb79", "score_shrink.json": "0a925a72",
       "tl_out_sel/score_tl.json": "183150ea", "tl_out_sel/tl_meta.json": "0e7c0820"}
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
res = {"script_md5": md5(__file__), "ask": "Deron P127 ask 1", "label": "REPORT-ONLY. Nothing here gates anything.", "verdict": "INCOMPLETE",
       "stops": [], "pins": {}, "env": {"torch": torch.__version__, "python": platform.python_version(), "threads": torch.get_num_threads(),
       "host": platform.node(), "note": "CPU-only, file-only; GPU not used"}}
def dump(): json.dump(res, open(OUTP, "w"), indent=1)
def stop(why): res["stops"].append(why); res["verdict"] = "NO-TABLE: " + why; dump(); log("STOP:", why); sys.exit(3)
def mem_avail_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"): return int(line.split()[1]) / 2**20
# ---- memory guard, pins -----------------------------------------------------------------------------------------
ma = mem_avail_gib(); res["memory"] = {"mem_available_at_start_gib": round(ma, 1), "required_gib": NEED_GIB}
if ma < NEED_GIB: stop(f"memory guard: MemAvailable {ma:.1f} GiB < {NEED_GIB} (live engine protected)")
for name, want in PIN.items():
    got = md5(os.path.join(BASE, name)); res["pins"][name] = got
    if not got.startswith(want): stop(f"pin {name} {got[:8]} != {want}")
SH = json.load(open(os.path.join(BASE, "score_shrink.json"))); FZ = SH["report"]["frozen_step2_parameters"]
if FZ is None or FZ["family"] != "S": stop(f"frozen family {FZ and FZ['family']} != S")
A = {l: float(FZ["real_per_layer"][str(l)]) for l in GATE_L}; RT = SH["report"]["readout_table"]["n22"]
TS = json.load(open(os.path.join(TLDIR, "score_tl.json"))); ROWS = TS["report"]["per_layer_table"]; UNI = TS["report"]["unigram_top20"]
TM = json.load(open(os.path.join(TLDIR, "tl_meta.json")))
if TM["unigram_top20"] != UNI["tokens"]: stop("unigram_top20 in tl_meta != score_tl")
log("pins OK; frozen a", A)
# ---- c1 cross-file: shrink a=0 == score_tl logit -----------------------------------------------------------------
c1 = {l: {k: abs(RT[str(l)]["a=0.0"][k] - ROWS[str(l)]["logit"][k]) for k in ("fut20", "top1")} for l in GATE_L}
res["controls"] = {"c1_shrink_a0_vs_tl_logit_absdiff": c1}
if any(v > TOL for d in c1.values() for v in d.values()): stop(f"c1: shrink a=0 != score_tl logit {c1}")
# ---- held positions + top-1 path (tl_score 368b8b09 lines, verbatim) ----------------------------------------------
def held_out_from(path):   # ast.literal_eval of the module's HELD_OUT assignment, never eval()
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "HELD_OUT" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit(f"no HELD_OUT in {path}")
HELD_OUT = held_out_from(os.path.join(BASE, "pilot_hutch.py"))
if HELD_OUT != held_out_from(os.path.join(BASE, "eval.py")): stop("HELD_OUT in pilot_hutch.py != HELD_OUT in eval.py")
held = torch.load(os.path.join(BASE, "held_h.pt"))
tok = AutoTokenizer.from_pretrained(MD); bos = tok.bos_token_id
nexts = []
for text in HELD_OUT:
    ids = tok(text, truncation=True, max_length=256 - (1 if bos is not None else 0), add_special_tokens=False).input_ids
    ids = ([bos] if bos is not None else []) + ids
    for q in range(SKIP, len(ids) - 1): nexts.append(torch.tensor(ids[q + 1:q + 11]))
NPOS = len(nexts)
if not NPOS == held["h_final"].shape[0] == 306: stop(f"positions {NPOS} / held {held['h_final'].shape[0]} / expected 306")
idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]
def W(n):
    with safe_open(os.path.join(MD, idx[n]), "pt") as f: return f.get_tensor(n).float()
WU = W("lm_head.weight"); g1 = 1.0 + W("model.language_model.norm.weight")
def logits_of(x):
    xn = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * g1
    return xn @ WU.T
LP_FINAL = torch.log_softmax(logits_of(held["h_final"]), -1); TOP1_FINAL = LP_FINAL.argmax(-1)
def readout(x):   # fut20 per position + mean top-1 agreement with the final layer (tl_score's readout without KL)
    lg = logits_of(x); t20 = lg.topk(20, -1).indices
    f = torch.tensor([float(torch.isin(n, t20[i]).float().mean()) for i, n in enumerate(nexts)])
    return f, float((lg.argmax(-1) == TOP1_FINAL).float().mean())
# ---- c2 positions, c3 top-1 path -----------------------------------------------------------------------------------
U20 = torch.tensor(UNI["tokens"])
uni = torch.tensor([float(torch.isin(n, U20).float().mean()) for n in nexts])
res["controls"]["c2_unigram_fut20_absdiff"] = abs(float(uni.mean()) - UNI["fut20"])
if res["controls"]["c2_unigram_fut20_absdiff"] > TOL: stop(f"c2: unigram fut20 {float(uni.mean())} != score_tl {UNI['fut20']}")
c3 = {}
for l in GATE_L:
    f, t1 = readout(held["layers"][l]); c3[l] = {"fut20": abs(float(f.mean()) - ROWS[str(l)]["logit"]["fut20"]), "top1": abs(t1 - ROWS[str(l)]["logit"]["top1"])}
res["controls"]["c3_logit_recomputed_vs_tl_absdiff"] = c3
if any(v > TOL for d in c3.values() for v in d.values()): stop(f"c3: recomputed logit != score_tl logit {c3}")
log("controls c1 c2 c3 OK")
uni_top1 = float((TOP1_FINAL == U20[0]).float().mean())
# ---- the table ------------------------------------------------------------------------------------------------------
table = {}
for l in GATE_L:
    s = RT[str(l)][f"a={A[l]}"]; r = ROWS[str(l)]
    table[l] = {"shrunk": {"a": A[l], "fut20": s["fut20"], "top1": s["top1"]},
                "logit": {k: r["logit"][k] for k in ("fut20", "top1")},
                "unigram": {"fut20": UNI["fut20"], "top1": uni_top1},
                "TL-big": {k: r["TL-big"][k] for k in ("fut20", "top1")},
                "TL-shuf": {k: r["TL-shuf"][k] for k in ("fut20", "top1")}}
res["table"] = table
res["unigram_top1_token"] = int(U20[0]); res["unigram_top1_token_text"] = tok.decode([int(U20[0])])
cols = ["shrunk", "logit", "unigram", "TL-big", "TL-shuf"]
md = ["| layer | a | " + " | ".join(f"{c} fut20 | {c} top-1" for c in cols) + " |", "|" + "---|" * (2 + 2 * len(cols))]
for l in GATE_L:
    md.append(f"| L{l} | {A[l]} | " + " | ".join(f"{table[l][c]['fut20']:.3f} | {table[l][c]['top1']:.3f}" for c in cols) + " |")
res["markdown"] = "\n".join(md)
res.update({"verdict": "TABLE", "minutes": round((time.time() - T0) / 60, 1), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}); dump()
print(res["markdown"], flush=True)
log("done: TABLE (report-only); ctx_table.json", md5(OUTP)[:8])
