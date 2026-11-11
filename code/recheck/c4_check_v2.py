"""FRESH C4 fidelity check for the emotion capture (Claude 2026-11-06; design Deron P075 §3, co-signed Qwen P076 +1 line; card C01).
Derived from c4_check.py d6d618da; the CAPTURE side, rows, stories, determinism and stored-vector checks are UNCHANGED. What changes is the REFERENCE:
the old full_reference ran full-attention layers with no mask under eager (bidirectional). Here the reference is a LOCAL copy of
shape_c_loader.full_reference (79a102cd lines 307-346; scl imported for helpers only, not modified) with _attn_implementation="sdpa" and no
mask = HF's is_causal path (test-1 arm b, bit-exact on row 0, post P072).
BINDING (bar UNCHANGED 2e-3, 3 stories x both models): (i) determinism; (ii) h_final AND logits vs reference(b, fp32_from=42); (iv) STORED vec_L42 row
vs the reference's own L42 output pooled from token 50. **The binding check is SAME-KERNEL: capture (sdpa, whole_model_fit.py:96) vs reference (b)
(sdpa), so the 2e-3 bar is achievable; the measured cross-kernel floor (a vs c 7.6e-3 / 8.5e-3, post P074) applies ONLY to the mask leg, which is why
the mask-leg bar is 2e-2** (Qwen P076).
BINDING MASK LEG (same run, 3 stories x both models), two extra reference arms vs the capture, sdpa + an in-script bool keep-mask (True = attend),
full_attention layers only (linear layers get None; modeling_qwen3_5.py :788/:444/:209), never HF create_causal_mask:
  (c) keep = ~triu(ones, diagonal=1): final_norm <= 2e-2;   (d) keep = ~triu(ones, diagonal=2): final_norm >= 1e-1 AND L3 token-0 rel >= 1e-2.
OUTCOME per model: capture or binding-ref L3 impl not sdpa -> UNMEASURABLE (Deron P077); (d) misses its line -> UNMEASURABLE;
determinism + binding (ii)+(iv) + mask leg (c) all pass -> PASS; else FAIL (capture VOID).
REPORTED: the same (ii)/(iv) vs reference(b, fp32_from=None) (all-bf16) on ROW 0 ONLY (Deron P078, Qwen P080), with the trip-wire (gap > 5e-2 -> stop and look)
and d6d618da's conditional claim clause (mixed-precision wording when that gap > 2e-3; Deron P079).
Claim limit (post P075): faithful to HF transformers' sdpa is_causal of the bf16 checkpoints; mask semantics cross-checked by an independent mask to
within the bf16 cross-kernel spread; BLIND to bugs shared by the HF layer class, the sdpa kernel or is_causal.
Writes capture-<m>/c4_check_v2.json after EVERY story. Usage: python c4_check_v2.py --model swift|base"""
import copy, gc, hashlib, json, os, sys, time, torch
import torch.nn.functional as F
from safetensors import safe_open
sys.path.insert(0, "$JLENS_HOME")
import whole_model_fit as wmf
import shape_c_loader as scl
from transformers import AutoConfig, AutoTokenizer
WHICH = sys.argv[sys.argv.index("--model") + 1]
MD = {"swift": "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b", "base": "$MODELS_DIR/bf16/Qwen3.8-27B"}[WHICH]
CORPUS = "$AUX_DIR/corpus-claude-v1/corpus"; CAP = f"$AUX_DIR/capture-{WHICH}"
LAYERS = [16, 24, 32, 40, 42, 43, 48, 56, 62]; START = 50; MAX_SEQ = 512; PICK = [0, 1790, 3579]; BAR = 2e-3
MASK_C_MAX = 2e-2; MASK_D_FINAL_MIN = 1e-1; MASK_D_TOK0_MIN = 1e-2; OUT_NAME = "c4_check_v2.json"
md5f = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
T0 = time.time(); log = lambda *a: print(f"[{(time.time()-T0)/60:5.1f}m]", *a, flush=True)
# rows exactly as emo_capture.py builds them (same sort, same field fallback)
files = sorted(os.listdir(CORPUS)); rows = []
for f in files:
    for l in open(os.path.join(CORPUS, f)):
        if l.strip():
            r = json.loads(l); rows.append({"file": f, "idx": r.get("idx"), "kind": "story" if "story" in r else "dialogue", "text": r.get("story") or r.get("dialogue")})
ix = json.load(open(os.path.join(CAP, "index.json")))
assert [(r["file"], r["idx"]) for r in rows] == [(r["file"], r["idx"]) for r in ix], "row order differs from the capture's index.json"
assert all(rows[i]["kind"] == "story" for i in PICK)
dev = torch.device("cuda:0"); model, tc = wmf.load_whole_model(MD, dev, log)
lang = model.model.language_model if hasattr(model.model, "language_model") else model.model
lang.layers[42].register_forward_pre_hook(
    lambda m, a, k: ((a[0].float(),) + a[1:], k) if a else (a, {**k, "hidden_states": k["hidden_states"].float()}), with_kwargs=True)
tok = AutoTokenizer.from_pretrained(MD); bos = tok.bos_token_id
cap = {}
def mk(l):
    def hook(m, a, out): cap[l] = (out[0] if isinstance(out, tuple) else out).detach()
    return hook
for l in LAYERS: lang.layers[l].register_forward_hook(mk(l))
lang.norm.register_forward_hook(lambda m, a, out: cap.__setitem__("hfin", out.detach()))
lang.layers[3].register_forward_hook(lambda m, a, out: cap.__setitem__("L3raw", (out[0] if isinstance(out, tuple) else out).detach()))
def enc(t):
    ids = tok(t, truncation=True, max_length=MAX_SEQ - 1, add_special_tokens=False).input_ids
    return torch.tensor([([bos] if bos is not None else []) + ids], device=dev)
def vecs(ids):
    with torch.no_grad(): o = model(input_ids=ids, use_cache=False)
    return {l: cap[l][0, START:].float().mean(0).cpu() for l in LAYERS}, cap["hfin"][0].float(), o.logits[0].float()
config = scl.text_cfg(AutoConfig.from_pretrained(MD)); kidx = scl.build_key_index(MD)
rotary = scl.make_rotary(config, scl.find_rotary_class(scl.import_qwen35())).to(dev)
lm_w = scl.get_tensor(MD, "lm_head.weight", kidx, dev).float()
stored = {l: torch.load(os.path.join(CAP, f"vec_L{l}.pt")) for l in LAYERS}
rel = lambda a, b: float((a - b).norm() / b.norm())
ltypes = list(config.layer_types); assert len(ltypes) == scl.N_LAYERS == 64
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
CAPTURE_IMPL = str(lang.layers[3].self_attn.config._attn_implementation)   # Deron P077: the capture side's attention impl, read off the live L3 module
res = {"model": WHICH, "capture_attn_impl_L3": CAPTURE_IMPL, "bars": "Deron P075 s3 as amended post P077 (2e-3 unchanged + mask leg + same-kernel check) / Qwen P076", "torch": torch.__version__, "transformers": __import__("transformers").__version__, "script_md5": md5f(__file__), "shape_c_loader_md5": md5f(scl.__file__),
       "whole_model_fit_md5": md5f(wmf.__file__), "picked_rows": PICK, "stories": []}
for i in PICK:
    ids = enc(rows[i]["text"]); v1, h1, lg1 = vecs(ids); v2, h2, lg2 = vecs(ids)
    det = all(torch.equal(v1[l], v2[l]) for l in LAYERS) and torch.equal(h1, h2) and torch.equal(lg1, lg2)
    vs_stored = all(torch.equal(v1[l], stored[l][i]) for l in LAYERS)
    out = {"row": i, "ntok": int(ids.shape[1]), "determinism_bit_identical": det, "equals_stored_vectors_REPORT": vs_stored}
    c3 = cap["L3raw"][0].float()
    ARMS = [("binding_b_fp32_from_42", 42, "b"), ("mask_c_fp32_from_42", 42, "c"), ("mask_d_fp32_from_42", 42, "d")]
    if i == PICK[0]: ARMS.insert(1, ("reported_b_all_bf16", None, "b"))   # REPORT arm on row 0 only (Deron P078, Qwen P080)
    for name, fp, arm in ARMS:
        hr, pool, r3, impl = reference(ids, fp, arm)
        assert c3.shape == r3.shape and h1.shape == hr.shape, (c3.shape, r3.shape, h1.shape, hr.shape)   # before any rel() (Deron P078 minor)
        out[name] = {"L3_impl_seen": impl, "h_final_rel_l2": rel(h1, hr), "logits_rel_l2": rel(lg1, hr @ lm_w.T),
                     "stored_vecL42_rel_l2": rel(stored[42][i].float(), pool[42]), "stored_vecL62_rel_l2_REPORT": rel(stored[62][i].float(), pool[62]),
                     "L3_token0_rel": rel(c3[0], r3[0])}
        torch.cuda.empty_cache()
    res["stories"].append(out); log(json.dumps(out)); json.dump(res, open(os.path.join(CAP, OUT_NAME), "w"), indent=1)
B = [s["binding_b_fp32_from_42"] for s in res["stories"]]; MC = [s["mask_c_fp32_from_42"] for s in res["stories"]]; MD_ = [s["mask_d_fp32_from_42"] for s in res["stories"]]
res["determinism_pass"] = all(s["determinism_bit_identical"] for s in res["stories"])
res["binding_pass"] = all(b["h_final_rel_l2"] <= BAR and b["logits_rel_l2"] <= BAR and b["stored_vecL42_rel_l2"] <= BAR for b in B)   # (ii) + (iv), same-kernel
res["mask_c_pass"] = all(c["h_final_rel_l2"] <= MASK_C_MAX for c in MC)
res["mask_d_valid"] = all(d["h_final_rel_l2"] >= MASK_D_FINAL_MIN and d["L3_token0_rel"] >= MASK_D_TOK0_MIN for d in MD_)
res["all_bf16_gap_max"] = max(max(v for k, v in s["reported_b_all_bf16"].items() if k.endswith("rel_l2")) for s in res["stories"] if "reported_b_all_bf16" in s)   # row 0 only
res["tripwire_own_gap_gt_5e-2"] = res["all_bf16_gap_max"] > 5e-2
res["same_kernel"] = CAPTURE_IMPL == "sdpa" and all(b["L3_impl_seen"] == "sdpa" for b in B)   # Deron P077: either side not sdpa = not same-kernel
res["verdict"] = ("UNMEASURABLE (binding check not same-kernel: capture %s / ref %s)" % (CAPTURE_IMPL, [b["L3_impl_seen"] for b in B]) if not res["same_kernel"] else
                  "UNMEASURABLE (mask-leg negative control d missed its line)" if not res["mask_d_valid"] else
                  "PASS" if res["determinism_pass"] and res["binding_pass"] and res["mask_c_pass"] else "FAIL: capture VOID")
res["claim_limit"] = ("faithful to HF transformers' sdpa is_causal implementation of the bf16 checkpoints (post P075); blind to bugs shared by the HF layer class / sdpa kernel / is_causal"
    + ("; vectors of the record-fit (mixed-precision) forward of the bf16 checkpoints" if res["all_bf16_gap_max"] > BAR else "; within 2e-3 of the all-bf16 forward"))   # d6d618da's conditional clause (Deron P079)
res["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 1); res["minutes"] = round((time.time() - T0) / 60, 1); res["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
json.dump(res, open(os.path.join(CAP, OUT_NAME), "w"), indent=1); log("C4v2", WHICH, res["verdict"], "| all-bf16 gap max", f"{res['all_bf16_gap_max']:.2e}")
