#!/usr/bin/env python3
"""fit_attach.py — J-lens fit on the whole-model FullModel resident in this process.

Caller (inside holder process):
    import fit_attach, whole_model_loader, torch, time
    t0=time.time(); fit_attach.run(whole_model_loader.model,
                                   model_dir="$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b",
                                   log=print)
    print(f"FIT_ELAPSED {time.time()-t0:.0f}s", flush=True)

Method: Deron's our_jlens.py (md5 54b0a119, oracle-validated 11-01) via a thin
HF-compatible wrapper over FullModel (config/get_input_embeddings/get_output_
embeddings) so the oracle-validated code path runs UNCHANGED. Forward =
FullModel.forward = scl.full_reference semantics (the proven path).
Full fit: grad_root_layer=None → embedding is grad root; sources 42..63;
target = final norm output (lens final_norm hook).
Outputs: J_l_<layer>.pt fp32 [5120,5120] (prompt-averaged), readouts.jsonl,
fit_meta.json — under $JLENS_HOME/results-whole-swift/.
"""
import json, os, time, math, importlib.util, hashlib

import torch
import torch.nn as nn

PROMPTS = [
    "I am trying to figure out why the bridge collapsed. The load was within spec, so the failure must be in the material itself.",
    "If all cats are animals and some animals are pets, can we conclude that some cats are pets? Let me think through the logic carefully.",
    "The patient has a fever, cough, and shortness of breath. Given the epidemiology, I need to weigh the most likely diagnoses.",
    "To solve the integral, I should first check the symmetry of the integrand, then decide between substitution and integration by parts.",
    "I feel a deep sense of dread that something is going to go wrong, even though I cannot name what exactly.",
    "A warm feeling of pride swells in my chest as I watch the team finish what they started.",
    "There is an irritation building, a low friction that makes every small mistake feel personally targeted.",
    "A quiet relief settles over me, the tension draining away like fog in morning sun.",
]

OUT_DIR = "$JLENS_HOME/results-whole-swift"
DIM_BATCH = 16
MAX_SEQ_LEN = 256


class _HFWrap(nn.Module):
    """Present FullModel to our_jlens as if it were an HF causal LM."""

    def __init__(self, full, d_model=5120, n_layers=64, vocab_size=151936):
        super().__init__()
        self._full = full
        self.embed_tokens = full.embed
        self.lm_head = full.lm_head
        self.layers = full.layers
        self.norm = full.final_norm
        self.config = type("Cfg", (), {
            "num_hidden_layers": n_layers,
            "hidden_size": d_model,
            "vocab_size": vocab_size,
            "get_text_config": lambda self_: self_,
        })()

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, inputs_embeds=None, use_cache=False, **kw):
        logits = self._full(input_ids, return_logits=True)
        return type("Out", (), {"logits": logits})()


def run(model, model_dir, log=print):
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()

    spec = importlib.util.spec_from_file_location("our_jlens", "$JLENS_HOME/our_jlens.py")
    oj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oj)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)

    wrap = _HFWrap(model)
    lens = oj.OurJacobianLens(wrap, tok, add_bos=True)
    log(f"[fit] lens ready: n_layers={lens.n_layers} d_model={lens.d_model} "
        f"bos_applied={lens.bos_applied} embed={type(lens.embed).__name__} "
        f"lm_head={type(lens.lm_head).__name__}")

    layers = list(range(42, 64))
    t_fit = time.time()
    t_last = time.time()
    acc = {l: torch.zeros(lens.d_model, lens.d_model, dtype=torch.float32) for l in layers}
    tot = 0
    for i, pr in enumerate(PROMPTS):
        Js, n_valid = lens.fit_prompt(pr, layers, dim_batch=DIM_BATCH, max_seq_len=MAX_SEQ_LEN)
        for l in layers:
            acc[l] += Js[l].float() * n_valid
        tot += n_valid
        dt = time.time() - t_last
        log(f"[fit] prompt {i+1}/{len(PROMPTS)}: n_valid={n_valid} {dt:.0f}s (cum {(time.time()-t_fit)/60:.1f} min)")
        del Js
        torch.cuda.empty_cache()
        t_last = time.time()
    dt_fit = time.time() - t_fit
    log(f"[fit] all prompts done in {dt_fit/60:.1f} min (tot={tot} valid pos)")

    for l in layers:
        J = (acc[l] / max(tot, 1)).cpu()
        path = os.path.join(OUT_DIR, f"J_l_{l}.pt")
        torch.save(J, path)
        log(f"[save] {path} {tuple(J.shape)} {J.dtype} |F|={J.norm().item():.1f}")
    del acc
    torch.cuda.empty_cache()

    readout_layers = [42, 45, 48, 51, 54, 57, 60, 63]
    J_cache = {l: torch.load(os.path.join(OUT_DIR, f"J_l_{l}.pt"), map_location="cpu").cuda()
               for l in readout_layers}
    readouts = []
    with torch.no_grad():
        for pr in PROMPTS:
            ids = lens._encode(pr, MAX_SEQ_LEN)
            seq = ids.shape[1]
            valid = torch.zeros(seq, dtype=torch.bool)
            valid[16:seq - 1] = True
            vpos = torch.where(valid)[0]
            vpos = vpos[:: max(1, len(vpos) // 4)]
            handles, store = lens._capture_residuals(readout_layers, lens.n_layers - 1, grad_root_layer=None)
            try:
                model(input_ids=ids.unsqueeze(0), return_logits=False)
                for l in readout_layers:
                    h_l = store[l][0]
                    J_l = J_cache[l]
                    for pos in vpos:
                        scores = lens.transport(J_l, h_l[pos])
                        topk = scores.topk(20)
                        readouts.append({"prompt": pr[:60], "layer": l, "pos": int(pos),
                                         "top20": [tok.decode([int(x)]) for x in topk.indices]})
            finally:
                for hd in handles:
                    hd.remove()
    with open(os.path.join(OUT_DIR, "readouts.jsonl"), "w") as f:
        for r in readouts:
            f.write(json.dumps(r) + "\n")
    log(f"[save] readouts.jsonl ({len(readouts)} rows)")

    meta = {
        "model": "Swift-1.5-Qwen3.8-27b — whole-model resident (0-41 bf16 + 42-63 fp32)",
        "layers": layers,
        "prompts": [p[:80] for p in PROMPTS],
        "dim_batch": DIM_BATCH,
        "max_seq_len": MAX_SEQ_LEN,
        "total_valid_positions": tot,
        "our_jlens_md5": hashlib.md5(open("$JLENS_HOME/our_jlens.py", "rb").read()).hexdigest(),
        "fit_seconds": dt_fit,
        "total_seconds": time.time() - t_start,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t_start)),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(os.path.join(OUT_DIR, "fit_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"[fit] COMPLETE out={OUT_DIR} total={meta['total_seconds']:.0f}s")


if __name__ == "__main__":
    raise SystemExit("import me inside the holder process — see docstring")
