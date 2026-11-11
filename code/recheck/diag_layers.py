"""C4 step 2 (Deron P063): find the FIRST layer where the capture path and the independent reference part.
Forward only. Per model (--model swift|base): (A) story row 0 as C4 ran it (ntok 125); (B) a 255-token input (rows 0+1790 joined, cut
to 255) since the reference passed at short seq on 11-03. Both in the ALL-BF16 regime (no L42 fp32 pre-hook on the capture side,
fp32_from=None on the reference), so any per-layer gap is structural, not precision. Per layer 0..63: rel_l2 of the full [T, d] output AND per token over the first 8 positions (mask bugs are position-dependent; Deron P066)
(capture via a forward hook on lang.layers[i]; reference via the same global Qwen3_5DecoderLayer hook c4_check used, counting 0..63).
Also: the embedding output, final norm, logits. Names the first layer with rel_l2 > 1e-3 and its type from config.layer_types.
(C) Deron P064/P066/P067 GATE for J-Lens, TWO forwards: C1 = Stage 1's OWN forward (lens._encode + _capture_residuals grad-root 16 + L42 pre-hook + model(input_ids), grad on) vs
full_reference(fp32_from=42); C2 = HF top-level model(input_ids) at 256 + L42 pre-hook (the held_h.pt path J1 scores on). On 3 Stage-1 fit docs (stage1_corpus.json idx 0/7/15) at Stage 1's length (bos + 255 = 256 tokens).
BAR per forward: final-norm AND logits rel_l2 <= 2e-3 on all 3. PASS needs C1 AND C2; else Stage 1 + J1 are UNMEASURABLE. Per-layer table 0..63 too.
Writes capture-<m>/diag_layers.json with this script's md5. A/B: diagnosis only. Usage: python diag_layers.py --model swift|base [--gate-only]"""
import hashlib, json, os, sys, time, torch
sys.path.insert(0, "$JLENS_HOME")
import whole_model_fit as wmf
import shape_c_loader as scl
from transformers import AutoConfig, AutoTokenizer
import importlib.util
_sp = importlib.util.spec_from_file_location("our_jlens", "$JLENS_HOME/our_jlens.py")
oj = importlib.util.module_from_spec(_sp); sys.modules[_sp.name] = oj; _sp.loader.exec_module(oj)
WHICH = sys.argv[sys.argv.index("--model") + 1]
MD = {"swift": "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b", "base": "$MODELS_DIR/bf16/Qwen3.8-27B"}[WHICH]
CORPUS = "$AUX_DIR/corpus-claude-v1/corpus"; CAP = f"$AUX_DIR/capture-{WHICH}"
md5f = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
T0 = time.time(); log = lambda *a: print(f"[{(time.time()-T0)/60:5.1f}m]", *a, flush=True)
files = sorted(os.listdir(CORPUS)); rows = []
for f in files:
    for l in open(os.path.join(CORPUS, f)):
        if l.strip():
            r = json.loads(l); rows.append({"text": r.get("story") or r.get("dialogue")})
dev = torch.device("cuda:0"); model, tc = wmf.load_whole_model(MD, dev, log)
lang = model.model.language_model if hasattr(model.model, "language_model") else model.model
tok = AutoTokenizer.from_pretrained(MD); bos = tok.bos_token_id
cfgfull = AutoConfig.from_pretrained(MD); config = scl.text_cfg(cfgfull); kidx = scl.build_key_index(MD)
ltypes = getattr(config, "layer_types", None) or [None] * 64
rotary = scl.make_rotary(config, scl.find_rotary_class(scl.import_qwen35())).to(dev)
lm_w = scl.get_tensor(MD, "lm_head.weight", kidx, dev).float()
NL = len(lang.layers); cap = {}; PRE = {"on": False}
lang.layers[42].register_forward_pre_hook(lambda m, a, k: (((a[0].float(),) + a[1:], k) if a else (a, {**k, "hidden_states": k["hidden_states"].float()})) if PRE["on"] else None, with_kwargs=True)
for i in range(NL): lang.layers[i].register_forward_hook((lambda i: lambda m, a, out: cap.__setitem__(i, (out[0] if isinstance(out, tuple) else out).detach().float()))(i))
lang.embed_tokens.register_forward_hook(lambda m, a, out: cap.__setitem__("emb", out.detach().float()))
lang.norm.register_forward_hook(lambda m, a, out: cap.__setitem__("norm", out.detach().float()))
lang.rotary_emb.register_forward_hook(lambda m, a, k, out: cap.__setitem__("pos", a[1] if len(a) > 1 else k.get("position_ids")), with_kwargs=True)   # (x, position_ids) positional or kw
QDL = scl.import_qwen35().Qwen3_5DecoderLayer
def two(x):   # Deron P068 (1): pin every compared tensor to [T, d]; never let rel() broadcast
    x = x.float(); assert x.dim() in (2, 3), f"unexpected rank {tuple(x.shape)}"
    if x.dim() == 3: assert x.shape[0] == 1, f"batch != 1: {tuple(x.shape)}"; x = x[0]
    return x
def rel(a, b):
    assert a.shape == b.shape, f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"
    return float((a - b).norm() / b.norm())
def c4_enc(t):   # Deron P068 (2): VERBATIM copy of c4_check.py d6d618da enc(); C4 passed this SAME ids tensor to its capture AND reference sides
    ids = tok(t, truncation=True, max_length=512 - 1, add_special_tokens=False).input_ids
    return torch.tensor([([bos] if bos is not None else []) + ids], device=dev)
def ids_of(text, n=None):
    ids = tok(text, add_special_tokens=False).input_ids
    ids = ([bos] if bos is not None else []) + ids
    return torch.tensor([ids[:n] if n else ids[:511]], device=dev)
def run(ids, fp=None, stage1_forward=False):
    if stage1_forward:   # Deron P066 (3): Stage 1's OWN forward = lens._capture_residuals hooks (grad root 16) + model(input_ids) with grad on
        hs, _store = lens._capture_residuals(list(range(16, 63)), NL - 1, grad_root_layer=16)
        try:
            with torch.enable_grad(): o = model(input_ids=ids, use_cache=False)
        finally:
            for hd in hs: hd.remove()
    else:
        with torch.no_grad(): o = model(input_ids=ids, use_cache=False)
    capt = {k: v for k, v in cap.items()}; lg_c = o.logits[0].float()
    got, n = {}, [0]
    def gh(mod, a, out):
        if isinstance(mod, QDL): got[n[0]] = (out[0] if isinstance(out, tuple) else out).detach().float(); n[0] += 1
    h = torch.nn.modules.module.register_module_forward_hook(gh)
    try:
        with torch.no_grad(): hr = scl.full_reference(MD, ids, kidx, config, rotary, dev, fp32_from=fp)[0].float()
    finally: h.remove()
    assert n[0] == NL, f"reference ran {n[0]} layers, expected {NL}"
    per = []
    for i in range(NL):
        a, b = two(capt[i]), two(got[i]); assert a.shape[0] == ids.shape[1], f"layer {i}: T {a.shape[0]} != ntok {ids.shape[1]}"
        per.append({"layer": i, "type": ltypes[i], "rel_l2": rel(a, b), "per_token_first8": [rel(a[t], b[t]) for t in range(min(8, a.shape[0]))]})
    pos_ref = scl.text_position_ids(ids.shape[1], dev); pos_cap = capt.get("pos")
    ident = {"ids_first4": ids[0, :4].tolist(), "bos_id": bos, "starts_with_bos": bool(ids[0, 0].item() == bos), "same_ids_tensor_both_sides": True,
             "pos_capture_shape": list(pos_cap.shape) if pos_cap is not None else None,
             "pos_identical": (pos_cap is not None and pos_cap.shape == pos_ref.shape and bool(torch.equal(pos_cap.to(pos_ref.device).long(), pos_ref.long()))),
             "pooling": "none here: full [T,d] compared; C4 pooled from START=50 on BOTH sides (same constant)"}
    first = next((p for p in per if p["rel_l2"] > 1e-3), None)
    out = {"ntok": int(ids.shape[1]), "identity": ident, "first_over_1e-3": first, "per_layer": per,
           "final_norm_rel": rel(two(capt["norm"]), two(hr)), "logits_rel": rel(two(lg_c), two(hr) @ lm_w.T),
           "ref_out_shape": list(hr.shape), "cap_norm_shape": list(capt["norm"].shape)}
    torch.cuda.empty_cache(); return out
res = {"model": WHICH, "weight_dtypes": {str(i): str(next(lang.layers[i].parameters()).dtype) for i in [0, 41, 42, 50, 63]}, "our_jlens_md5": md5f("$JLENS_HOME/our_jlens.py"),
       "held_h_provenance": "capture_h.py: wmf.load_whole_model (fceeffd8) + HF top-level model(input_ids) at lens._encode(text, 256), 11-04 16:26 (Claude P065)", "script_md5": md5f(__file__), "shape_c_loader_md5": md5f(scl.__file__), "whole_model_fit_md5": md5f(wmf.__file__),
       "transformers": __import__("transformers").__version__, "regime": "C4 regimes exactly: capture = load_whole_model + L42 fp32 pre-hook ON for every part; reference fp32_from=42 (A_row0, B_255, C) and fp32_from=None (A_row0_ref_bf16)"}
GATE_ONLY = "--gate-only" in sys.argv
if not GATE_ONLY:
  log("A: row 0 as C4"); _a = c4_enc(rows[0]["text"]); _b = ids_of(rows[0]["text"])
  _c4j = json.load(open(os.path.join(CAP, "c4_check.json"))); _c4n = _c4j["stories"][0]["ntok"]
  res["A_ids_check"] = {"c4_enc_len": int(_a.shape[1]), "c4_enc_first4": _a[0, :4].tolist(), "ids_of_len": int(_b.shape[1]), "ids_of_first4": _b[0, :4].tolist(),
      "c4_json_ntok_row0": _c4n, "equal": bool(torch.equal(_a, _b)), "matches_c4_json_ntok": int(_a.shape[1]) == _c4n,
      "c4_sides": "c4_check.py builds ids ONCE (enc) and passes the same tensor to vecs() (capture) and reference() (reference): identical by construction"}
  assert res["A_ids_check"]["matches_c4_json_ntok"], f"C4 ntok mismatch: {res['A_ids_check']}"
  PRE["on"] = True   # C4's capture side ALWAYS had the L42 fp32 pre-hook (c4_check.py d6d618da line 35): load_whole_model remaps 42..63 to fp32, so without it L42 gets bf16 into fp32 weights and crashes (11-06 01:00 run)
  res["A_row0"] = run(_a, fp=42); log(json.dumps({k: v for k, v in res["A_row0"].items() if k != "per_layer"}))   # = C4 "binding_fp32_from_42"
  res["A_row0_ref_bf16"] = run(_a, fp=None); log(json.dumps({k: v for k, v in res["A_row0_ref_bf16"].items() if k != "per_layer"}))   # = C4 "reported_all_bf16" (reference side only)
  log("B: 255 tokens"); res["B_255"] = run(ids_of(rows[0]["text"] + "\n\n" + rows[1790]["text"], 255), fp=42); log(json.dumps({k: v for k, v in res["B_255"].items() if k != "per_layer"}))
s1 = json.load(open("$JLENS_HOME/claude-normtest/stage1_corpus.json")); PRE["on"] = True; G = []
lens = oj.OurJacobianLens(model, tok, add_bos=True)
for di in (0, 7, 15):
    ids = lens._encode(s1[di]["text"], 256)
    for tag, s1f in (("C1_stage1_own_forward", True), ("C2_hf_toplevel_heldh_path", False)):
        log(f"C: {tag} doc {di} ({s1[di]['genre']}) vs fp32_from=42")
        r = run(ids, fp=42, stage1_forward=s1f); r["doc"] = di; r["genre"] = s1[di]["genre"]; r["forward"] = tag; G.append(r)
        log(json.dumps({k: v for k, v in r.items() if k not in ("per_layer",)}))
PRE["on"] = False
ok = lambda tag: all(g["final_norm_rel"] <= 2e-3 and g["logits_rel"] <= 2e-3 for g in G if g["forward"] == tag)
res["C_jlens_gate"] = {"bar": "final-norm AND logits rel_l2 <= 2e-3 on docs 0/7/15, for BOTH C1 and C2 (Deron P064/P067)", "runs": G,
    "C1_PASS": ok("C1_stage1_own_forward"), "C2_PASS": ok("C2_hf_toplevel_heldh_path")}
res["C_jlens_gate"]["PASS"] = res["C_jlens_gate"]["C1_PASS"] and res["C_jlens_gate"]["C2_PASS"]
log("C gate PASS" if res["C_jlens_gate"]["PASS"] else "C gate FAIL")
res["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 1); res["minutes"] = round((time.time() - T0) / 60, 1); res["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
json.dump(res, open(os.path.join(CAP, "diag_layers.json"), "w"), indent=1); log("done", WHICH)
