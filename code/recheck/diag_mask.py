"""C4 decisive test (Deron P071): is the C4/diag split caused by the REFERENCE running full-attention layers WITHOUT a causal mask?
Forward only, part A only (story row 0 exactly as C4 ran it: c4_enc, ntok 125), per model (--model swift|base).
Capture side = C4's own path, unchanged: wmf.load_whole_model + L42 fp32 pre-hook + HF top-level model(input_ids) (builds its own causal mask).
Reference side = a LOCAL copy of shape_c_loader.full_reference (79a102cd) with fp32_from=42, run in three arms:
  (u) unfixed: no mask (identical to full_reference; reproduces the diag; also logs per-token rel at the LAST 8 positions of L3)
  (a) eager + an explicit additive causal mask built HERE with torch.triu (0 on/below the diagonal, finfo(dtype).min above), in the
      dtype of each layer's activations (bf16 for 0..41, fp32 for 42..63), passed ONLY to full_attention layers (linear-attention
      layers get no mask, as in full_reference). HF create_causal_mask is NOT used (it is the capture's own code; Deron P071).
  (b) _attn_implementation="sdpa" on a deep copy of the config, no mask (cross-check of (a)).
shape_c_loader.py is imported for helpers only and is NOT modified. Per arm: per-layer rel_l2 0..63, first layer over 1e-3, per-token first 8,
final_norm and logits rel_l2. Also (a) vs (b) per-layer and final agreement. Predictions (Deron P071, written before the run): in (a) and (b)
L0..L2 unchanged <1e-3, L3..63 <1e-3, final_norm and logits <=2e-3, (a) vs (b) <=1e-3; either arm above 2e-3 kills the hypothesis.
json is dumped after EVERY arm (Deron P069 note 2). Writes capture-<m>/diag_mask.json. Usage: python diag_mask.py --model swift|base"""
import copy, gc, hashlib, json, os, sys, time, torch
import torch.nn.functional as F
sys.path.insert(0, "$JLENS_HOME")
import whole_model_fit as wmf
import shape_c_loader as scl
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
WHICH = sys.argv[sys.argv.index("--model") + 1]
MD = {"swift": "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b", "base": "$MODELS_DIR/bf16/Qwen3.8-27B"}[WHICH]
CORPUS = "$AUX_DIR/corpus-claude-v1/corpus"; CAP = f"$AUX_DIR/capture-{WHICH}"; OUT = os.path.join(CAP, "diag_mask.json")
FP = 42
md5f = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
T0 = time.time(); log = lambda *a: print(f"[{(time.time()-T0)/60:5.1f}m]", *a, flush=True)
files = sorted(os.listdir(CORPUS)); rows = []
for f in files:
    for l in open(os.path.join(CORPUS, f)):
        if l.strip():
            r = json.loads(l); rows.append({"text": r.get("story") or r.get("dialogue")})
res = {"model": WHICH, "script_md5": md5f(__file__), "shape_c_loader_md5": md5f(scl.__file__), "whole_model_fit_md5": md5f(wmf.__file__),
       "transformers": __import__("transformers").__version__, "bars": "Deron P071", "arms": {}}
def dump(): json.dump(res, open(OUT, "w"), indent=1)
dev = torch.device("cuda:0"); model, tc = wmf.load_whole_model(MD, dev, log)
lang = model.model.language_model if hasattr(model.model, "language_model") else model.model
# C4's capture side, verbatim (c4_check.py d6d618da line 35): the L42 fp32 pre-hook, always on
lang.layers[42].register_forward_pre_hook(
    lambda m, a, k: ((a[0].float(),) + a[1:], k) if a else (a, {**k, "hidden_states": k["hidden_states"].float()}), with_kwargs=True)
tok = AutoTokenizer.from_pretrained(MD); bos = tok.bos_token_id
config = scl.text_cfg(AutoConfig.from_pretrained(MD)); kidx = scl.build_key_index(MD)
ltypes = list(config.layer_types); NL = len(lang.layers); assert NL == scl.N_LAYERS == len(ltypes) == 64
rotary = scl.make_rotary(config, scl.find_rotary_class(scl.import_qwen35())).to(dev)
lm_w = scl.get_tensor(MD, "lm_head.weight", kidx, dev).float()
res["weight_dtypes"] = {str(i): str(next(lang.layers[i].parameters()).dtype) for i in [0, 3, 41, 42, 50, 63]}
res["config_attn_implementation"] = str(getattr(config, "_attn_implementation", None)); dump()
def two(x):   # pin every compared tensor to [T, d] (Deron P068)
    x = x.float(); assert x.dim() in (2, 3), f"unexpected rank {tuple(x.shape)}"
    if x.dim() == 3: assert x.shape[0] == 1, f"batch != 1: {tuple(x.shape)}"; x = x[0]
    return x
def rel(a, b):
    assert a.shape == b.shape, f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"
    return float((a - b).norm() / b.norm())
def c4_enc(t):   # VERBATIM c4_check.py d6d618da enc()
    ids = tok(t, truncation=True, max_length=512 - 1, add_special_tokens=False).input_ids
    return torch.tensor([([bos] if bos is not None else []) + ids], device=dev)
ids = c4_enc(rows[0]["text"]); T = ids.shape[1]
c4n = json.load(open(os.path.join(CAP, "c4_check.json")))["stories"][0]["ntok"]
res["ids"] = {"ntok": T, "first4": ids[0, :4].tolist(), "c4_json_ntok_row0": c4n}; assert T == c4n, f"ntok {T} != C4 {c4n}"
# ---- capture (once; deterministic per C4) ----
cap = {}
hs = [lang.layers[i].register_forward_hook((lambda i: lambda m, a, o: cap.__setitem__(i, (o[0] if isinstance(o, tuple) else o).detach().float()))(i)) for i in range(NL)]
hs.append(lang.norm.register_forward_hook(lambda m, a, o: cap.__setitem__("norm", o.detach().float())))
log("capture: HF top-level model(input_ids), C4 path")
with torch.no_grad(): lg_c = model(input_ids=ids, use_cache=False).logits[0].float()
for h in hs: h.remove()
capt = {i: two(cap[i]) for i in range(NL)}; capn = two(cap["norm"]); del cap
for i in range(NL): assert capt[i].shape[0] == T
def causal_mask(dtype):   # additive [1,1,T,T]: 0 on/below the diagonal, dtype min strictly above (Deron P071 arm a)
    m = torch.triu(torch.ones(T, T, dtype=torch.bool, device=dev), diagonal=1)
    return torch.zeros(1, 1, T, T, dtype=dtype, device=dev).masked_fill(m, torch.finfo(dtype).min)
def reference(arm):
    """LOCAL copy of shape_c_loader.full_reference(fp32_from=42) (79a102cd lines 307-346); the ONLY changes are the arm switches."""
    m = scl.import_qwen35(); cfg = config
    if arm == "b_sdpa": cfg = copy.deepcopy(config); cfg._attn_implementation = "sdpa"
    outs = []
    embed_w = scl.get_tensor(MD, "model.language_model.embed_tokens.weight", kidx, dev)
    h = F.embedding(ids, embed_w)
    position_ids = scl.text_position_ids(T, dev)
    cos, sin = scl.rotary_call(rotary, h, position_ids)
    for i in range(scl.N_LAYERS):
        fp32_layer = i >= FP
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
        kw = {}
        if arm == "a_eager_mask" and ltypes[i] == "full_attention": kw["attention_mask"] = causal_mask(h.dtype)
        if i == 3: res["arms"].setdefault(arm, {})["L3_attn_impl_seen"] = str(getattr(getattr(getattr(layer, "self_attn", None), "config", None), "_attn_implementation", "n/a"))
        h = layer(h, position_embeddings=(cos, sin), position_ids=position_ids, **kw)
        outs.append(two(h.detach()))
        del layer; gc.collect(); torch.cuda.empty_cache()
    norm_w = scl.get_tensor(MD, "model.language_model.norm.weight", kidx, dev).float()
    hn = scl.rms_norm_ref(h.float(), norm_w, config.rms_norm_eps)
    return outs, two(hn)
refs = {}
for arm in ("u_unfixed", "a_eager_mask", "b_sdpa"):
    log(f"reference arm {arm}")
    with torch.no_grad(): outs, hn = reference(arm)
    per = [{"layer": i, "type": ltypes[i], "rel_l2": rel(capt[i], outs[i]),
            "per_token_first8": [rel(capt[i][t], outs[i][t]) for t in range(8)]} for i in range(NL)]
    r = res["arms"].setdefault(arm, {})
    r.update({"per_layer": per, "first_over_1e-3": next((p for p in per if p["rel_l2"] > 1e-3), None),
              "max_rel_L0_L2": max(p["rel_l2"] for p in per[:3]), "max_rel_L3_L63": max(p["rel_l2"] for p in per[3:]),
              "final_norm_rel": rel(capn, hn), "logits_rel": rel(lg_c, hn @ lm_w.T)})
    if arm == "u_unfixed": r["L3_per_token_last8"] = [rel(capt[3][t], outs[3][t]) for t in range(T - 8, T)]
    refs[arm] = (outs, hn)
    log(json.dumps({k: v for k, v in r.items() if k != "per_layer"})); dump()
(oa, ha), (ob, hb) = refs["a_eager_mask"], refs["b_sdpa"]
res["a_vs_b"] = {"max_layer_rel": max(rel(oa[i], ob[i]) for i in range(NL)), "final_norm_rel": rel(ha, hb)}
A, B = res["arms"]["a_eager_mask"], res["arms"]["b_sdpa"]
res["verdict_vs_predictions"] = {
    "a_L0_L2_lt_1e-3": A["max_rel_L0_L2"] < 1e-3, "a_L3_L63_lt_1e-3": A["max_rel_L3_L63"] < 1e-3,
    "a_final_and_logits_le_2e-3": A["final_norm_rel"] <= 2e-3 and A["logits_rel"] <= 2e-3,
    "b_L0_L2_lt_1e-3": B["max_rel_L0_L2"] < 1e-3, "b_L3_L63_lt_1e-3": B["max_rel_L3_L63"] < 1e-3,
    "b_final_and_logits_le_2e-3": B["final_norm_rel"] <= 2e-3 and B["logits_rel"] <= 2e-3,
    "a_vs_b_le_1e-3": res["a_vs_b"]["max_layer_rel"] <= 1e-3 and res["a_vs_b"]["final_norm_rel"] <= 1e-3,
    "note": "report only; Deron adjudicates. Either arm final/logits > 2e-3 = hypothesis dead (post P071)."}
log(json.dumps({"a_vs_b": res["a_vs_b"], "verdict_vs_predictions": res["verdict_vs_predictions"]})); dump(); log("done", WHICH)
