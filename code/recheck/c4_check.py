"""C4 fidelity check for the emotion capture (Claude 2026-11-04; bars Deron P058 / vault b7744ac, Qwen P059 accepts (B); card C04).
Per model (--model swift|base), on 3 fixed stories (rows 0, 1790, 3579 of the frozen corpus = first, middle, last story):
 (i)  DETERMINISM: the capture's own vecs() twice -> bit-identical; also == the stored rows of capture-<m>/vec_L*.pt (report).
 (iv) BINDING (Deron P060): the STORED vec_L42 row (layer 42's OUTPUT = layer 43's input, pooled from token 50) vs the same
      quantity taken from the independent reference pass (a temporary global forward hook counts Qwen3_5DecoderLayer calls 0..63;
      pooled here by separate lines, not vecs()). rel_l2 <= 2e-3, 3 stories. REPORT: L62, and both vs the all-bf16 reference.
 (ii) BINDING: post-norm h_final AND logits from the capture path (whole_model_fit loader + L42 fp32 pre-hook, i.e. the record-fit
      regime) vs the INDEPENDENT layer-by-layer reference shape_c_loader.full_reference(fp32_from=42), lm_head from disk.
      PASS iff rel_l2 <= 2e-3 for both, on every story. Fail = that model's capture is VOID.
      REPORTED: the same vs the all-bf16 reference (fp32_from=None). TRIP-WIRE (read across both models' outputs): all-bf16 gap
      > 5e-2, or the Swift/base gaps differ > 3x -> stop and look before H1 opens.
Writes capture-<m>/c4_check.json with this script's md5 and the reference module's md5. Usage: python c4_check.py --model swift|base"""
import hashlib, json, os, sys, time, torch
sys.path.insert(0, "$JLENS_HOME")
import whole_model_fit as wmf
import shape_c_loader as scl
from transformers import AutoConfig, AutoTokenizer
WHICH = sys.argv[sys.argv.index("--model") + 1]
MD = {"swift": "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b", "base": "$MODELS_DIR/bf16/Qwen3.8-27B"}[WHICH]
CORPUS = "$AUX_DIR/corpus-claude-v1/corpus"; CAP = f"$AUX_DIR/capture-{WHICH}"
LAYERS = [16, 24, 32, 40, 42, 43, 48, 56, 62]; START = 50; MAX_SEQ = 512; PICK = [0, 1790, 3579]; BAR = 2e-3
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
QDL = scl.import_qwen35().Qwen3_5DecoderLayer
def reference(ids, fp):
    """full_reference + its own layer-42/62 outputs, caught by a global hook live ONLY during this call (never on the capture model)."""
    got, n = {}, [0]
    def gh(mod, a, out):
        if isinstance(mod, QDL):
            if n[0] in (42, 62): got[n[0]] = (out[0] if isinstance(out, tuple) else out).detach()
            n[0] += 1
    h = torch.nn.modules.module.register_module_forward_hook(gh)
    try:
        with torch.no_grad(): hr = scl.full_reference(MD, ids, kidx, config, rotary, dev, fp32_from=fp)[0].float()
    finally: h.remove()
    assert n[0] == 64, f"reference ran {n[0]} decoder layers, expected 64"
    pool = {l: got[l][0, START:].float().mean(0).cpu() for l in (42, 62)}   # independent pooling: token START onward, batch row 0
    return hr, pool
res = {"model": WHICH, "bars": "Deron P058 (B) + post P060 (iv) / Qwen P059", "script_md5": md5f(__file__), "shape_c_loader_md5": md5f(scl.__file__),
       "whole_model_fit_md5": md5f(wmf.__file__), "picked_rows": PICK, "stories": []}
for i in PICK:
    ids = enc(rows[i]["text"]); v1, h1, lg1 = vecs(ids); v2, h2, lg2 = vecs(ids)
    det = all(torch.equal(v1[l], v2[l]) for l in LAYERS) and torch.equal(h1, h2) and torch.equal(lg1, lg2)
    vs_stored = all(torch.equal(v1[l], stored[l][i]) for l in LAYERS)
    out = {"row": i, "ntok": int(ids.shape[1]), "determinism_bit_identical": det, "equals_stored_vectors_REPORT": vs_stored}
    for name, fp in (("binding_fp32_from_42", 42), ("reported_all_bf16", None)):
        hr, pool = reference(ids, fp)
        out[name] = {"h_final_rel_l2": rel(h1, hr), "logits_rel_l2": rel(lg1, hr @ lm_w.T),
                     "stored_vecL42_rel_l2": rel(stored[42][i].float(), pool[42]), "stored_vecL62_rel_l2_REPORT": rel(stored[62][i].float(), pool[62])}
        torch.cuda.empty_cache()
    res["stories"].append(out); log(json.dumps(out))
B = [s["binding_fp32_from_42"] for s in res["stories"]]
res["determinism_pass"] = all(s["determinism_bit_identical"] for s in res["stories"])
res["binding_pass"] = all(b["h_final_rel_l2"] <= BAR and b["logits_rel_l2"] <= BAR and b["stored_vecL42_rel_l2"] <= BAR for b in B)   # (ii) + (iv)
res["all_bf16_gap_max"] = max(max(v for k, v in s["reported_all_bf16"].items() if not k.endswith("REPORT")) for s in res["stories"])
res["tripwire_own_gap_gt_5e-2"] = res["all_bf16_gap_max"] > 5e-2
res["verdict"] = "PASS" if res["determinism_pass"] and res["binding_pass"] else "FAIL: capture VOID"
res["claim_limit"] = ("vectors of the record-fit (mixed-precision) forward of the bf16 checkpoints" if res["all_bf16_gap_max"] > BAR else "within 2e-3 of the all-bf16 forward")
res["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 1); res["minutes"] = round((time.time() - T0) / 60, 1); res["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
json.dump(res, open(os.path.join(CAP, "c4_check.json"), "w"), indent=1); log("C4", WHICH, res["verdict"], "| all-bf16 gap max", f"{res['all_bf16_gap_max']:.2e}")
