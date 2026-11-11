"""J-Lens STAGE 1 fit (Claude 2026-11-04; bars Deron P047 / vault b5274ab; card C01). Resumable across windows.
Same code path as the record fit (whole_model_fit loader + L42 fp32 pre-hook + our_jlens.fit_prompt, grad root 16), on 16 NEW
documents from 16 genres (stage1_corpus.json), seq 256, dim_batch 1 (the only seq-256 setting that fits: post P041 mem probe).
Per document, saved the moment it finishes (so a window end or crash loses at most the doc in flight):
  stage1/doc_XX/J_p_<l>.pt for l in SAVE_LAYERS (n_valid-WEIGHTED is applied at merge time; J_p here is the per-doc mean)
  stage1/doc_XX/meta.json  {source, genre, n_valid, tokens, seconds, md5 of each J_p}
  stage1/acc_<l>.pt        running (sum J_p * n_valid, sum n_valid) for ALL layers 16..62 (Deron: keep running sums)
  stage1/progress.json     done doc indices
--deadline <epoch>: don't START a doc that can't finish before it (needs ~1.6 h of margin). Usage: python stage1_fit.py [--deadline N] [--smoke]"""
import hashlib, json, os, sys, time, torch
sys.path.insert(0, "$JLENS_HOME")
import whole_model_fit as wmf
import importlib.util
spec = importlib.util.spec_from_file_location("our_jlens", "$JLENS_HOME/our_jlens.py")
oj = importlib.util.module_from_spec(spec); sys.modules[spec.name] = oj; spec.loader.exec_module(oj)
from transformers import AutoTokenizer
SMOKE = "--smoke" in sys.argv
DEADLINE = float(sys.argv[sys.argv.index("--deadline") + 1]) if "--deadline" in sys.argv else float("inf")
BASE = "$JLENS_HOME/claude-normtest"; OUT = os.path.join(BASE, "stage1" + ("_smoke" if SMOKE else "")); os.makedirs(OUT, exist_ok=True)
MD = "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b"
LAYERS = list(range(16, 63)); SAVE_LAYERS = [20, 24, 28, 32, 36, 40, 44, 48, 52, 56]
MAX_SEQ = 256 if not SMOKE else 24; DIM_BATCH = 1 if not SMOKE else 8; DOC_SECONDS_EST = 1.6 * 3600   # smoke: 640 backwards, tests the code path, not numbers
# (21:4x: smoke at DIM_BATCH 64 OOMed: 64 x 24 = 1536 seq-equivalents vs the full run's 1 x 256; keep dim_batch*seq <= 256)
docs = json.load(open(os.path.join(BASE, "stage1_corpus.json")))
if SMOKE: docs, LAYERS, SAVE_LAYERS = docs[:1], [16, 20, 52], [20, 52]
prog_p = os.path.join(OUT, "progress.json"); prog = json.load(open(prog_p)) if os.path.exists(prog_p) else {"done": []}
T0 = time.time(); log = lambda *a: print(f"[{(time.time()-T0)/60:6.1f}m]", *a, flush=True)
todo = [i for i in range(len(docs)) if i not in prog["done"]]
log(f"stage1: {len(prog['done'])} done, {len(todo)} to go; deadline {'none' if DEADLINE == float('inf') else time.strftime('%H:%M', time.localtime(DEADLINE))}")
if not todo: log("nothing to do"); sys.exit(0)
if time.time() + (60 if SMOKE else DOC_SECONDS_EST) > DEADLINE: log("not enough time before the deadline for one doc; exiting"); sys.exit(0)
wmf.check_provenance(MD)
dev = torch.device("cuda:0"); model, tc = wmf.load_whole_model(MD, dev, log)
lang = model.model.language_model if hasattr(model.model, "language_model") else model.model
lang.layers[42].register_forward_pre_hook(
    lambda m, a, k: ((a[0].float(),) + a[1:], k) if a else (a, {**k, "hidden_states": k["hidden_states"].float()}), with_kwargs=True)
tok = AutoTokenizer.from_pretrained(MD); lens = oj.OurJacobianLens(model, tok, add_bos=True)
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
for i in todo:
    if time.time() + (60 if SMOKE else DOC_SECONDS_EST) > DEADLINE: log(f"deadline: stopping before doc {i+1}"); break
    d = docs[i]; t = time.time(); ddir = os.path.join(OUT, f"doc_{i+1:02d}"); os.makedirs(ddir, exist_ok=True)
    ntok = int(lens._encode(d["text"], MAX_SEQ).shape[1])
    Js, n_valid = lens.fit_prompt(d["text"], LAYERS, dim_batch=DIM_BATCH, max_seq_len=MAX_SEQ, grad_root_layer=LAYERS[0])
    if n_valid == 0: log(f"doc {i+1} SKIPPED (too short)"); continue
    sums = {}
    for l in SAVE_LAYERS:
        p = os.path.join(ddir, f"J_p_{l}.pt"); torch.save(Js[l].cpu(), p); sums[l] = md5(p)
    for l in LAYERS:                                   # running sums, written before progress so progress never runs ahead
        ap = os.path.join(OUT, f"acc_{l}.pt")
        acc, tot = torch.load(ap) if os.path.exists(ap) else (torch.zeros_like(Js[l].cpu()), 0)
        torch.save((acc + Js[l].cpu() * n_valid, tot + n_valid), ap + ".tmp"); os.replace(ap + ".tmp", ap)
    meta = {"doc": i + 1, "source": d["source"], "genre": d["genre"], "url": d["url"], "tokens_in_fit": ntok, "n_valid": n_valid,
            "seconds": round(time.time() - t), "J_p_md5": sums, "max_seq": MAX_SEQ, "dim_batch": DIM_BATCH, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(meta, open(os.path.join(ddir, "meta.json"), "w"), indent=1)
    prog["done"].append(i); json.dump(prog, open(prog_p + ".tmp", "w")); os.replace(prog_p + ".tmp", prog_p)
    del Js; torch.cuda.empty_cache()
    log(f"doc {i+1}/{len(docs)} {d['genre']}: n_valid={n_valid} tokens={ntok} {meta['seconds']}s | done {len(prog['done'])}")
log("STAGE1 window end:", len(prog["done"]), "of", len(docs), "docs done")
