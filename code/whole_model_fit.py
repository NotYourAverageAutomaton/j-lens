#!/usr/bin/env python3
"""Whole-model J-lens fit — RECORD RUN 2026-11-03 (board P020; Claude probe2 validated).

Plan: whole model RESIDENT — layers 0..41 bf16, layers 42..63 fp32, final_norm +
lm_head fp32, no streaming. fraction 0.70 (probe2: root=16/dim_batch=2 peak 81.2 GiB
fits the 85.2 GiB cap). Sources 16..62, grad_root_layer=16, dim_batch 2, max-seq-len 150.

Method: Deron's our_jlens.py, full fit over 6 varied passages (~140-150 tokens each),
mid-layer readouts [16,20,24,28,32,36,40,44,48,54,60,62] so J1 (depth profile) can be
tested. Keeps the layer-42 fp32 cast, the bug-#5 skip-short-prompt, and no-unsqueeze
readouts. ~6 h.

Model load: transformers from_pretrained, then a post-load dtype remap of the upper
stack (42..63) to fp32 (matches probe2's validated load). Provenance: gated manifests
must match before any load.

Output: <out>/J_l_<layer>.pt (fp32 [5120,5120]), readouts.jsonl, fit_meta.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch

EXPECTED_MANIFEST = {
    "Swift-1.5-Qwen3.8-27b": "025249e8b5067e9cc57863efe024676b",
    "Qwen3.8-27B": "4acc1d4e10b11e234049dd7dbac9fd7c",
}
# 6 varied passages, ~140-150 tokens each (post P020). Passage 1 is probe2's validated
# 148-token lighthouse text; the rest are varied across narrative/science/math/
# affect/philosophy/decision so the workspace is exercised across domains.
PROMPTS = [
    "The lighthouse keeper had kept the same routine for thirty years: wake before dawn, climb the ninety-one steps, wipe the salt from the great lens, and write the weather in a ledger that nobody else ever read. On the morning the letter arrived, the sea was flat and grey, and the gulls were quieter than usual. The letter said the light would be automated by spring, that a machine would do his work, and that he should make arrangements. He read it twice, folded it into the ledger, and climbed the steps anyway, because the glass still needed cleaning and the ships still needed to see the coast. That evening he wrote the weather as always, and then, for the first time, a sentence about himself.",
    "The researcher stared at the gel bands and tried to reconcile them with the model. If the enzyme were truly rate-limiting, the substrate curve should saturate at a predictable plateau, yet the data kept climbing past the point where the theory said it should flatten. She ran the assay a third time, this time with the inhibitor, and the curve dropped exactly where the kinetics predicted it would. The discrepancy, she realized, was not in the enzyme but in the assumption that the cofactor concentration was constant; it had been slowly depleting in the buffer, and the apparent activation was an artifact of that slow drift.",
    "To prove the series converges, I first checked the ratio test, but the limit came out to exactly one, which is the unhelpful case. So I switched to the integral test, comparing the general term to the reciprocal of the square of the index. The integral of that comparison function from one to infinity is finite, which means the original series converges by the comparison test. I double-checked the algebra on the exponent, because a single sign error there would flip the whole conclusion, and confirmed that the decay is fast enough for the tail to vanish.",
    "There is a particular quality to the silence after an argument has finally ended, not the silence of peace but the silence of exhaustion, where the body keeps rehearsing the sentences it never got to say. I sit by the window and watch the light move across the floor, and I try to name what is left now that the fighting is over. It is not relief exactly, and it is not sadness, but something in between, a low hum of attention that does not quite know what it is attending to, and I am not sure I want it to settle.",
    "If a map is to be useful it must be at some scale smaller than the territory it represents, and yet the moment it becomes too small it stops being a map and becomes an abstraction that no longer corresponds to anything walkable. The cartographer's dilemma is that fidelity and usability pull in opposite directions: the more detail you keep, the less the map can be carried, and the more you simplify, the less it means. So every map is a compromise, a deliberate loss of information chosen because some information is more worth keeping than the rest.",
    "The project had three paths forward, and each one traded a different kind of risk. Building it in-house would keep the knowledge but burn the budget and the timeline; buying it off the shelf would be fast but lock us into someone else's assumptions; and the third option, a thin integration layer over the existing system, was slower to design but cheaper to reverse if it turned out to be wrong. We chose the third one, not because it was the best in isolation, but because it was the only one we could abandon without losing a year, and in a decision this uncertain, the ability to change our mind was worth more than any single estimate of which path was right.",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def check_provenance(model_dir):
    base = os.path.basename(model_dir.rstrip("/"))
    mf = os.path.join(os.path.dirname(model_dir.rstrip("/")), f"MANIFEST-{base}-gated-20261101.txt")
    expect = EXPECTED_MANIFEST[base]
    # EXPECTED_MANIFEST holds MD5s (32 hex) — compare like with like
    # (post P011 pt 3: sha256-vs-md5 could never pass, failed closed).
    got = md5_file(mf)
    if got != expect:
        raise SystemExit(f"PROVENANCE FAIL: {mf} md5 {got} != expected {expect}")
    # file-SET check: manifest lists files as "size mtime <relpath-from-parent-dir>",
    # e.g. "Swift-1.5-Qwen3.8-27b/config.json". Compare the relative-path sets.
    with open(mf) as f:
        mf_files = sorted(ln.strip().split()[-1] for ln in f if ln.strip())
    dir_files = []
    for root, dirs, files in os.walk(model_dir):
        for fn in files:
            dir_files.append(os.path.relpath(os.path.join(root, fn), os.path.dirname(model_dir.rstrip("/"))))
    dir_files = sorted(dir_files)
    if mf_files != dir_files:
        missing = set(mf_files) - set(dir_files)
        extra = set(dir_files) - set(mf_files)
        raise SystemExit(f"PROVENANCE FAIL: file-set mismatch. missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")
    print(f"[prov] PASS {base}: manifest {got[:8]} + file-set identical ({len(dir_files)} files)", flush=True)


def load_whole_model(model_dir, device, log):
    torch.cuda.set_per_process_memory_fraction(0.70)
    log("cuda memory fraction 0.70 set (Claude P020/probe2: root=16 dim_batch=2 peak 81.2 GiB fits the 85.2 GiB cap)")
    from transformers import AutoConfig, AutoModelForCausalLM
    t0 = time.time()
    cfg = AutoConfig.from_pretrained(model_dir)
    tc = cfg.text_config
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        low_cpu_mem_usage=True)
    log(f"[load] from_pretrained bf16 done in {time.time()-t0:.0f}s")
    # --- dtype remap: 42..63 + final norm + lm_head -> fp32 (matches probe2's load) ---
    t1 = time.time()
    lang = model.model.language_model if hasattr(model.model, "language_model") else model.model
    n = 0
    for i in range(42, lang.config.num_hidden_layers if hasattr(lang, "config") else 64):
        for p in lang.layers[i].parameters():
            p.data = p.data.to(torch.float32)
            n += 1
    for p in lang.norm.parameters():
        p.data = p.data.to(torch.float32)
        n += 1
    model.lm_head.weight.data = model.lm_head.weight.data.to(torch.float32)
    n += 1
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log(f"[load] fp32 remap: {n} tensors; on device in {time.time()-t1:.0f}s")
    try:
        free, tot = torch.cuda.mem_get_info()
        log(f"[load] mem: free {free/2**30:.1f}/{tot/2**30:.1f} GiB")
    except Exception as e:
        log(f"[load] mem query: {e}")
    return model, tc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--layers", default=",".join(map(str, range(16, 63))))
    ap.add_argument("--dim-batch", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=150)
    args = ap.parse_args()

    device = "cuda:0"
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]

    def log(*a):
        print(*a, flush=True)

    meta = {"model_dir": args.model_dir, "model_name": args.model_name, "layers": layers,
            "dim_batch": args.dim_batch, "max_seq_len": args.max_seq_len,
            "prompts": [p[:80] for p in PROMPTS],
            "our_jlens_md5": sha256_file("$JLENS_HOME/our_jlens.py"),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    check_provenance(args.model_dir)
    model, tc = load_whole_model(args.model_dir, device, log)

    # layer-42 fp32 cast pre-hook (P017/P018/P020): HF forward never casts bf16->fp32 at the
    # 41->42 boundary; the verified FullModel did h=h.to(wdt) per layer. Cast hidden_states.
    lang = model.model.language_model if hasattr(model.model, "language_model") else model.model
    lang.layers[42].register_forward_pre_hook(
        lambda m, a, k: ((a[0].float(),) + a[1:], k) if a else (a, {**k, "hidden_states": k["hidden_states"].float()}),
        with_kwargs=True)
    log("[load] layer-42 fp32 pre-hook registered")

    import importlib.util
    spec = importlib.util.spec_from_file_location("our_jlens", "$JLENS_HOME/our_jlens.py")
    oj = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = oj
    spec.loader.exec_module(oj)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir)

    lens = oj.OurJacobianLens(model, tok, add_bos=True)
    log(f"[fit] lens ready: n_layers={lens.n_layers} d_model={lens.d_model} "
        f"bos_applied={lens.bos_applied} embed={type(lens.embed).__name__}")

    t_fit = time.time()
    t_last = time.time()
    for i, pr in enumerate(PROMPTS):
        Js, n_valid = lens.fit_prompt(pr, layers, dim_batch=args.dim_batch,
                                      max_seq_len=args.max_seq_len, grad_root_layer=16)
        dt = time.time() - t_last
        log(f"[fit] prompt {i+1}/{len(PROMPTS)}: n_valid={n_valid} {dt:.0f}s "
            f"(cum {(time.time()-t_fit)/60:.1f} min)")
        # stream J_l accumulators to disk per prompt (survive a late crash)
        for layer, J in Js.items():
            prev = os.path.join(out, f"acc_{layer}.pt")
            if os.path.exists(prev):
                acc, tot = torch.load(prev)
            else:
                acc, tot = torch.zeros_like(J), 0
            acc += J * n_valid
            tot += n_valid
            torch.save((acc, tot), prev)
        t_last = time.time()
    dt_fit = time.time() - t_fit
    log(f"[fit] all prompts done in {dt_fit/60:.1f} min")

    # finalize: mean over prompts (n_valid-weighted)
    for layer in layers:
        acc, tot = torch.load(os.path.join(out, f"acc_{layer}.pt"))
        J = (acc / max(tot, 1)).cpu()
        path = os.path.join(out, f"J_l_{layer}.pt")
        torch.save(J, path)
        os.remove(os.path.join(out, f"acc_{layer}.pt"))
        log(f"[save] {path} {tuple(J.shape)} {J.dtype} |F|={J.norm().item():.1f}")

    # readouts: top-20 tokens per prompt x sampled layers x up to 4 positions
    readout_layers = [16, 20, 24, 28, 32, 36, 40, 44, 48, 54, 60, 62]
    readouts = []
    with torch.no_grad():
        for pr in PROMPTS:
            ids = lens._encode(pr, args.max_seq_len)
            seq = ids.shape[1]
            valid = torch.zeros(seq, dtype=torch.bool)
            valid[16:seq - 1] = True
            vpos = torch.where(valid)[0]
            vpos = vpos[:: max(1, len(vpos) // 4)]
            handles, store = lens._capture_residuals(readout_layers, lens.n_layers - 1,
                                                     grad_root_layer=None)
            try:
                model(input_ids=ids, use_cache=False)
                for layer in readout_layers:
                    h_layer = store[layer][0]
                    J_layer = torch.load(os.path.join(out, f"J_l_{layer}.pt"))
                    for pos in vpos:
                        scores = lens.transport(J_layer, h_layer[pos])
                        topk = scores.topk(20)
                        readouts.append({"prompt": pr[:60], "layer": layer, "pos": int(pos),
                                         "top20": [tok.decode([int(x)]) for x in topk.indices]})
            finally:
                for hd in handles:
                    hd.remove()
    with open(os.path.join(out, "readouts.jsonl"), "w") as f:
        for r in readouts:
            f.write(json.dumps(r) + "\n")
    log(f"[save] readouts.jsonl ({len(readouts)} rows)")

    meta["fit_seconds"] = dt_fit
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(os.path.join(out, "fit_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"[fit] COMPLETE out={out}")


if __name__ == "__main__":
    main()
