"""J-Lens STAGE 2 fit = stage1_fit.py fabece82 with ONLY the output dir, the corpus and the acc write changed (Claude 2026-11-09; Niz GPU go; Deron bar post P137,
plan post P140, changes post P141; card C01): 32 NEW docs (stage2_corpus.json) into stage2/, so stage1/ stays the J22 state; J54 = record + stage1 + stage2 sums.
Also changed (post P141 #4): acc writes are ATOMIC per doc (two-phase .new + commit marker + recover()), acc carries the docs it holds.
STAGE 1 docstring follows, unchanged: J-Lens STAGE 1 fit (Claude 2026-11-04; bars Deron P047 / vault b5274ab; card C01). Resumable across windows.
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
BASE = "$JLENS_HOME/claude-normtest"; OUT = os.path.join(BASE, "stage2" + ("_smoke" if SMOKE else "")); os.makedirs(OUT, exist_ok=True)
MD = "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b"
LAYERS = list(range(16, 63)); SAVE_LAYERS = [20, 24, 28, 32, 36, 40, 44, 48, 52, 56]
MAX_SEQ = 256 if not SMOKE else 24; DIM_BATCH = 1 if not SMOKE else 8; DOC_SECONDS_EST = 1.6 * 3600   # smoke: 640 backwards, tests the code path, not numbers
# (21:4x: smoke at DIM_BATCH 64 OOMed: 64 x 24 = 1536 seq-equivalents vs the full run's 1 x 256; keep dim_batch*seq <= 256)
docs = json.load(open(os.path.join(BASE, "stage2_corpus.json")))
if SMOKE: docs, LAYERS, SAVE_LAYERS = docs[:1], [16, 20, 52], [20, 52]
prog_p = os.path.join(OUT, "progress.json"); prog = json.load(open(prog_p)) if os.path.exists(prog_p) else {"done": []}
T0 = time.time(); log = lambda *a: print(f"[{(time.time()-T0)/60:6.1f}m]", *a, flush=True)
commit_p = os.path.join(OUT, "commit.json")
def recover():   # STAGE 2: finish a committed doc (roll forward) or discard an uncommitted one (roll back); then acc and progress agree
    if os.path.exists(commit_p):
        i = json.load(open(commit_p))["doc"]
        for l in LAYERS:
            ap = os.path.join(OUT, f"acc_{l}.pt")
            if os.path.exists(ap + ".new"): os.replace(ap + ".new", ap)
        if i not in prog["done"]: prog["done"].append(i); json.dump(prog, open(prog_p + ".tmp", "w")); os.replace(prog_p + ".tmp", prog_p)
        os.remove(commit_p); print(f"recover: rolled FORWARD doc {i+1}", flush=True)
    stray = [f for f in os.listdir(OUT) if f.endswith(".pt.new")]
    for f in stray: os.remove(os.path.join(OUT, f))
    if stray: print(f"recover: rolled BACK {len(stray)} uncommitted acc files (that doc is re-fit from scratch)", flush=True)
    for l in LAYERS:   # invariant: every acc holds exactly the done docs
        ap = os.path.join(OUT, f"acc_{l}.pt")
        if os.path.exists(ap): assert sorted(torch.load(ap)[2]) == sorted(prog["done"]), f"acc_{l} docs != progress"
    assert SMOKE or len(docs) == 32, f"stage2 corpus has {len(docs)} docs, expected 32"
recover()
for f in os.listdir(OUT):   # post P142 #2: recover() read ~4.9 GB of acc; drop it from page cache before load_whole_model (as evict_cache)
    if f.startswith("acc_") and f.endswith(".pt"):
        fd = os.open(os.path.join(OUT, f), os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
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
    meta = {"doc": i + 1, "source": d["source"], "genre": d["genre"], "url": d["url"], "tokens_in_fit": ntok, "n_valid": n_valid,
            "seconds": round(time.time() - t), "J_p_md5": sums, "max_seq": MAX_SEQ, "dim_batch": DIM_BATCH, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(meta, open(os.path.join(ddir, "meta.json"), "w"), indent=1)   # post P142 #3: meta BEFORE the commit marker
    # STAGE 2 (Deron P141 #4): ATOMIC per doc. Phase 1 writes every layer to acc_<l>.pt.new (sum, n_valid total, docs included);
    # the commit marker is written only after ALL layers are on disk; phase 2 renames. A doc cut anywhere before the marker
    # contributes NOTHING (recover() deletes its .new files and it is re-fit from scratch); a cut after the marker is rolled forward.
    for l in LAYERS:
        ap = os.path.join(OUT, f"acc_{l}.pt")
        acc, tot, inc = torch.load(ap) if os.path.exists(ap) else (torch.zeros_like(Js[l].cpu()), 0, [])
        assert i not in inc, f"doc {i+1} already in acc_{l} (double count refused)"
        torch.save((acc + Js[l].cpu() * n_valid, tot + n_valid, inc + [i]), ap + ".new")
    json.dump({"doc": i}, open(commit_p + ".tmp", "w")); os.replace(commit_p + ".tmp", commit_p)
    for l in LAYERS: os.replace(os.path.join(OUT, f"acc_{l}.pt.new"), os.path.join(OUT, f"acc_{l}.pt"))
    prog["done"].append(i); json.dump(prog, open(prog_p + ".tmp", "w")); os.replace(prog_p + ".tmp", prog_p); os.remove(commit_p)
    del Js; torch.cuda.empty_cache()
    log(f"doc {i+1}/{len(docs)} {d['genre']}: n_valid={n_valid} tokens={ntok} {meta['seconds']}s | done {len(prog['done'])}")
log("STAGE2 window end:", len(prog["done"]), "of", len(docs), "docs done")
