"""Claude fit-memory probe (2026-11-03). Does NOT modify Qwen's or Deron's files and writes no fit
results. Loads the model with whole_model_fit.load_whole_model (fraction 0.70, unchanged), then for
grad_root_layer=42 tries dim_batch 16 -> 8 -> 4: one forward with graph + 2 backward blocks, the same
steps as our_jlens.fit_prompt. Reports peak GiB and seconds per backward block, for a time estimate."""
import json, math, sys, time, torch
sys.path.insert(0, "$JLENS_HOME")
import importlib.util
import whole_model_fit as wmf
spec = importlib.util.spec_from_file_location("our_jlens", "$JLENS_HOME/our_jlens.py")
oj = importlib.util.module_from_spec(spec); sys.modules[spec.name] = oj; spec.loader.exec_module(oj)
from transformers import AutoTokenizer
G = 1024 ** 3
MD = "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b"
dev = torch.device("cuda:0")
model, tc = wmf.load_whole_model(MD, dev, lambda s: print(s, flush=True))
tok = AutoTokenizer.from_pretrained(MD)
lens = oj.OurJacobianLens(model, tok, add_bos=True)
# the 41->42 boundary cast that FullModel.forward did (h = h.to(wdt)) and HF's forward does not
def _to_fp32(mod, args, kwargs):
    if args:
        return (args[0].float(),) + tuple(args[1:]), kwargs
    kwargs["hidden_states"] = kwargs["hidden_states"].float(); return args, kwargs
lens.layers[42].register_forward_pre_hook(_to_fp32, with_kwargs=True)
with torch.no_grad():
    t0 = time.time(); o = model(input_ids=lens._encode(wmf.PROMPTS[0], 256), use_cache=False)
    print(f"SMOKE no-grad forward ok {time.time()-t0:.1f}s logits {tuple(o.logits.shape)} {o.logits.dtype} finite={bool(torch.isfinite(o.logits).all())}", flush=True)
    del o
src, tgt, root = list(range(42, 63)), lens.n_layers - 1, 42
ids = lens._encode(wmf.PROMPTS[0], 256)
seq = ids.shape[1]; valid = torch.zeros(seq, dtype=torch.bool); valid[oj.SKIP_FIRST_N_POSITIONS:seq-1] = True
base = torch.cuda.memory_allocated() / G
res = {"ts": time.strftime("%H:%M:%S"), "seq": seq, "base_gib": base, "n_prompts": len(wmf.PROMPTS), "trials": []}
for db in (16, 8, 4):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t = {"dim_batch": db, "grad_root_layer": root}
    handles, store = lens._capture_residuals(src, tgt, grad_root_layer=root)
    try:
        with torch.enable_grad():
            t0 = time.time(); model(input_ids=ids.expand(db, -1), use_cache=False)
            t["fwd_s"] = round(time.time() - t0, 1); t["peak_fwd_gib"] = torch.cuda.max_memory_allocated() / G
            h = store[tgt]; bt = []
            for blk in range(2):
                cot = torch.zeros_like(h)
                for b in range(db): cot[b, valid, blk * db + b] = 1.0
                torch.cuda.synchronize(); t0 = time.time()
                torch.autograd.backward(h, cot, retain_graph=(blk == 0))
                torch.cuda.synchronize(); bt.append(time.time() - t0)
                for l in src:
                    if store[l].grad is not None: store[l].grad.zero_()
        n_back = math.ceil(lens.d_model / db)
        t.update(ok=True, back_s=[round(x, 2) for x in bt], peak_gib=torch.cuda.max_memory_allocated() / G,
                 n_back_per_prompt=n_back,
                 est_total_min=round((t["fwd_s"] + n_back * bt[-1]) * len(wmf.PROMPTS) / 60, 1))
    except torch.OutOfMemoryError as e:
        t.update(ok=False, err="OOM " + str(e)[:160])
    except Exception as e:
        import traceback; t.update(ok=False, err=traceback.format_exc()[-600:]); print("TRIAL-ERR " + json.dumps(t), flush=True); res["trials"].append(t); break
    finally:
        for hd in handles: hd.remove()
        store = None; h = None; cot = None
    res["trials"].append(t); print("TRIAL " + json.dumps(t), flush=True)
    if t.get("ok"): break
print("FITPROBE " + json.dumps(res), flush=True)
open("$JLENS_HOME/claude-normtest/fitprobe.json", "w").write(json.dumps(res, indent=2))
