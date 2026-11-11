"""PART C, the J-Lens gate (Claude 2026-11-06; spec Deron P085; card C01). Derived from diag_layers.py adbb7c75 (the v3 Deron passed), --gate-only path ONLY.
(C) Deron P064/P066/P067 GATE for J-Lens, TWO forwards: C1 = Stage 1's OWN forward (lens._encode + _capture_residuals grad-root 16 + L42 pre-hook + model(input_ids), grad on) vs
the reference; C2 = HF top-level model(input_ids) at 256 + L42 pre-hook (the held_h.pt path J1 scores on). On 3 Stage-1 fit docs (stage1_corpus.json idx 0/7/15) at Stage 1's length (bos + 255 = 256 tokens).
Changes from adbb7c75, per post P085:
 1. REFERENCE = c4_check_v2.py 061f2897's reference(ids, 42, "b"), copied VERBATIM (sdpa, no mask, fp32 from 42), with its keep_mask helper. scl.full_reference is no longer called.
 2. SAME-KERNEL FIRST: the capture's L3 impl (read off the live module, as 061f2897 does) must be sdpa before any forward runs; the ref's L3 impl must be sdpa on every run. Else UNMEASURABLE.
 3. BAR unchanged (P064/P067): final-norm AND logits rel_l2 <= 2e-3 on docs 0/7/15 at 256, for BOTH C1 and C2. PASS needs C1 AND C2.
 4. Signed C1 backend clause, in-script: per doc, log C1-vs-C2 final-norm rel and token-0 rel (final norm, token 0, vs the reference). If C2 passes, C1 misses, every C1
    final-norm and logits rel is <= 2e-2, C1's token-0 rel is exactly 0.0 on every doc, and C1-vs-C2 > 2e-3 on every doc C1 misses, then C1 = UNMEASURABLE. Any other C1 miss = FAIL.
 5. Memory (v3 Part C OOMed): one doc x forward at a time. The capture outputs are copied out and the forward (and C1's grad graph + residual store) is freed BEFORE the
    reference runs. The json is dumped after EVERY run. Peak GiB is logged per run (reset between runs) and overall.
 6. The json carries script_md5 + verdict, per model. Swift is required (Stage 1 fit on it).
Writes capture-<m>/part_c.json (capture-<m> as in adbb7c75; nothing else there is touched). Usage: python part_c.py --model swift|base"""
import copy, gc, hashlib, json, os, sys, time, torch
import torch.nn.functional as F
from safetensors import safe_open
sys.path.insert(0, "$JLENS_HOME")
import whole_model_fit as wmf
import shape_c_loader as scl
from transformers import AutoConfig, AutoTokenizer
import importlib.util
_sp = importlib.util.spec_from_file_location("our_jlens", "$JLENS_HOME/our_jlens.py")
oj = importlib.util.module_from_spec(_sp); sys.modules[_sp.name] = oj; _sp.loader.exec_module(oj)
WHICH = sys.argv[sys.argv.index("--model") + 1]
MD = {"swift": "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b", "base": "$MODELS_DIR/bf16/Qwen3.8-27B"}[WHICH]
CAP = f"$AUX_DIR/capture-{WHICH}"; OUT_NAME = "part_c.json"; BAR = 2e-3; C1_BACKEND_MAX = 2e-2; START = 50
md5f = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
T0 = time.time(); log = lambda *a: print(f"[{(time.time()-T0)/60:5.1f}m]", *a, flush=True)
dev = torch.device("cuda:0"); model, tc = wmf.load_whole_model(MD, dev, log)
lang = model.model.language_model if hasattr(model.model, "language_model") else model.model
tok = AutoTokenizer.from_pretrained(MD); bos = tok.bos_token_id
config = scl.text_cfg(AutoConfig.from_pretrained(MD)); kidx = scl.build_key_index(MD)
ltypes = list(config.layer_types); assert len(ltypes) == scl.N_LAYERS == 64
rotary = scl.make_rotary(config, scl.find_rotary_class(scl.import_qwen35())).to(dev)
lm_w = scl.get_tensor(MD, "lm_head.weight", kidx, dev).float()
NL = len(lang.layers); cap = {}
lang.layers[42].register_forward_pre_hook(lambda m, a, k: ((a[0].float(),) + a[1:], k) if a else (a, {**k, "hidden_states": k["hidden_states"].float()}), with_kwargs=True)   # always on: Part C ran with PRE on in adbb7c75
lang.norm.register_forward_hook(lambda m, a, out: cap.__setitem__("norm", out.detach().float()))
def two(x):   # Deron P068 (1): pin every compared tensor to [T, d]; never let rel() broadcast
    x = x.float(); assert x.dim() in (2, 3), f"unexpected rank {tuple(x.shape)}"
    if x.dim() == 3: assert x.shape[0] == 1, f"batch != 1: {tuple(x.shape)}"; x = x[0]
    return x
def rel(a, b):
    assert a.shape == b.shape, f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"
    return float((a - b).norm() / b.norm())
# ---- VERBATIM from c4_check_v2.py 061f2897 (lines 67-96): keep_mask + reference ----
def keep_mask(T, diag):   # bool [1,1,T,T]; True = takes part in attention (torch SDPA); masked = triu(ones, diagonal=diag)
    return ~torch.triu(torch.ones(T, T, dtype=torch.bool, device=dev), diagonal=diag)[None, None]
def reference(ids, fp, arm):
    """LOCAL copy of full_reference (79a102cd 307-346) with sdpa. arm 'b' = no mask (HF is_causal); 'c' = keep_mask(1); 'd' = keep_mask(2)."""
    m = scl.import_qwen35(); cfg = copy.deepcopy(config); cfg._attn_implementation = "sdpa"; T = ids.shape[1]
    km = {"b": None, "c": keep_mask(T, 1), "d": keep_mask(T, 2)}[arm]
    h = F.embedding(ids, scl.get_tensor(MD, "model.language_model.embed_tokens.weight", kidx, dev))
    position_ids = scl.text_position_ids(T, dev); cos, sin = scl.rotary_call(rotary, h, position_ids); keep = {}
    for i in range(scl.N_LAYERS):
        fp32_layer = fp is not None and i >= fp
        if fp32_layer: h = h.float(); cos = cos.float(); sin = sin.float()
        layer = m.Qwen3_5DecoderLayer(cfg, i).to(dev)
        if fp32_layer: layer = layer.to(torch.float32)
        prefix = f"model.language_model.layers.{i}."
        for key, shard in kidx.items():
            if key.startswith(prefix):
                with safe_open(shard, framework="pt") as f:
                    t = f.get_tensor(key).to(dev)
                    if fp32_layer: t = t.to(torch.float32)
                    scl.set_attr(layer, key[len(prefix):], t)
        kw = {"attention_mask": km} if (km is not None and ltypes[i] == "full_attention") else {}
        if i == 3: impl = str(layer.self_attn.config._attn_implementation)
        h = layer(h, position_embeddings=(cos, sin), position_ids=position_ids, **kw)
        if i in (3, 42, 62): keep[i] = h.detach()
        del layer; gc.collect(); torch.cuda.empty_cache()
    norm_w = scl.get_tensor(MD, "model.language_model.norm.weight", kidx, dev)
    if fp is not None: h = h.float(); norm_w = norm_w.float()
    hr = scl.rms_norm_ref(h, norm_w, config.rms_norm_eps)[0].float()
    pool = {l: keep[l][0, START:].float().mean(0).cpu() for l in (42, 62)}   # independent pooling: token START onward, batch row 0
    return hr, pool, keep[3][0].float(), impl
# ---- end verbatim ----
CAPTURE_IMPL = str(lang.layers[3].self_attn.config._attn_implementation)   # 061f2897 line 97
res = {"model": WHICH, "capture_attn_impl_L3": CAPTURE_IMPL, "bars": "Deron P064/P067 (2e-3, C1 AND C2) + post P085 (verbatim 061f2897 reference b, same-kernel first, C1 backend clause)",
       "script_md5": md5f(__file__), "derived_from": "diag_layers.py adbb7c75 (gate-only path)", "reference_from": "c4_check_v2.py 061f2897 reference(ids, 42, 'b'), verbatim",
       "our_jlens_md5": md5f("$JLENS_HOME/our_jlens.py"), "shape_c_loader_md5": md5f(scl.__file__), "whole_model_fit_md5": md5f(wmf.__file__),
       "torch": torch.__version__, "transformers": __import__("transformers").__version__, "runs": [], "verdict": "INCOMPLETE (crashed or still running)"}
dump = lambda: json.dump(res, open(os.path.join(CAP, OUT_NAME), "w"), indent=1)
if CAPTURE_IMPL != "sdpa":   # post P085 (2): same-kernel FIRST, before any forward
    res["verdict"] = f"UNMEASURABLE (capture L3 impl {CAPTURE_IMPL}, not sdpa)"; dump(); log(res["verdict"]); sys.exit(0)
dump()
s1 = json.load(open("$JLENS_HOME/claude-normtest/stage1_corpus.json"))
lens = oj.OurJacobianLens(model, tok, add_bos=True)
def capture(ids, stage1_forward):
    """One forward; returns final-norm [T,d] and logits [T,V] as fp32 copies, with the forward, its graph and C1's residual store freed."""
    cap.clear()
    if stage1_forward:   # Deron P066 (3): Stage 1's OWN forward = lens._capture_residuals hooks (grad root 16) + model(input_ids) with grad on
        hs, store = lens._capture_residuals(list(range(16, 63)), NL - 1, grad_root_layer=16)
        try:
            with torch.enable_grad(): o = model(input_ids=ids, use_cache=False)
            lg = o.logits[0].detach().float().clone()
        finally:
            for hd in hs: hd.remove()
        del o, store
    else:
        with torch.no_grad(): o = model(input_ids=ids, use_cache=False)
        lg = o.logits[0].float().clone(); del o
    nm = two(cap.pop("norm")).clone(); cap.clear(); gc.collect(); torch.cuda.empty_cache()   # post P085 (5): free BEFORE the reference runs
    return nm, lg
final_norms = {}
for di in (0, 7, 15):
    ids = lens._encode(s1[di]["text"], 256); assert ids.shape[1] == 256, f"doc {di}: ntok {ids.shape[1]} != 256"
    for tag, s1f in (("C1_stage1_own_forward", True), ("C2_hf_toplevel_heldh_path", False)):
        log(f"C: {tag} doc {di} ({s1[di]['genre']}) vs 061f2897 reference(ids, 42, 'b')")
        torch.cuda.reset_peak_memory_stats()
        nm, lg = capture(ids, s1f); peak_fwd = round(torch.cuda.max_memory_allocated() / 2**30, 1)
        with torch.no_grad(): hr, _pool, _r3, impl = reference(ids, 42, "b")
        r = {"doc": di, "genre": s1[di]["genre"], "forward": tag, "ntok": int(ids.shape[1]), "ref_L3_impl_seen": impl,
             "final_norm_rel": rel(nm, two(hr)), "logits_rel": rel(two(lg), two(hr) @ lm_w.T), "token0_final_norm_rel": rel(nm[0], two(hr)[0]),
             "peak_gib_forward": peak_fwd, "peak_gib_run": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        final_norms[(di, tag)] = nm.cpu()
        if tag.startswith("C2"):   # post P085 (4): C1-vs-C2 final-norm rel, per doc
            r["C1_vs_C2_final_norm_rel"] = rel(final_norms[(di, "C1_stage1_own_forward")], final_norms[(di, tag)])
            next(x for x in res["runs"] if x["doc"] == di and x["forward"].startswith("C1"))["C1_vs_C2_final_norm_rel"] = r["C1_vs_C2_final_norm_rel"]
        res["runs"].append(r); del nm, lg, hr, _pool, _r3; gc.collect(); torch.cuda.empty_cache()
        log(json.dumps(r)); dump()   # post P085 (5): dump after EVERY run
R = res["runs"]; C1 = [r for r in R if r["forward"].startswith("C1")]; C2 = [r for r in R if r["forward"].startswith("C2")]
ok = lambda rs: all(r["final_norm_rel"] <= BAR and r["logits_rel"] <= BAR for r in rs)
res["same_kernel"] = CAPTURE_IMPL == "sdpa" and all(r["ref_L3_impl_seen"] == "sdpa" for r in R)
res["C1_PASS"] = ok(C1); res["C2_PASS"] = ok(C2)
miss = [r for r in C1 if not (r["final_norm_rel"] <= BAR and r["logits_rel"] <= BAR)]
res["C1_backend_clause"] = bool(res["C2_PASS"] and miss
    and all(r["final_norm_rel"] <= C1_BACKEND_MAX and r["logits_rel"] <= C1_BACKEND_MAX for r in C1)
    and all(r["token0_final_norm_rel"] == 0.0 for r in C1)
    and all(r["C1_vs_C2_final_norm_rel"] > BAR for r in miss))
res["C1_outcome"] = "PASS" if res["C1_PASS"] else ("UNMEASURABLE (signed backend clause, post P085 item 4)" if res["C1_backend_clause"] else "FAIL")
res["verdict"] = ("UNMEASURABLE (not same-kernel: capture %s / ref %s)" % (CAPTURE_IMPL, [r["ref_L3_impl_seen"] for r in R]) if not res["same_kernel"] else
                  "PASS" if res["C1_PASS"] and res["C2_PASS"] else
                  "UNMEASURABLE (C1 backend clause; C2 PASS)" if res["C1_backend_clause"] else "FAIL")
res["peak_gib"] = round(max(r["peak_gib_run"] for r in R), 1); res["minutes"] = round((time.time() - T0) / 60, 1); res["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
dump(); log("Part C", WHICH, res["verdict"], "| C1", res["C1_outcome"], "| C2", "PASS" if res["C2_PASS"] else "FAIL", "| peak GiB", res["peak_gib"])
