#!/usr/bin/env python3
"""whole_model_loader.py — J-lens WHOLE-MODEL resident loader. Card C01.

The 14:00-18:00 window (Niz go post P008; Claude P009/P010) loads the WHOLE
Swift-1.5-Qwen3.8-27b at once — no streaming — so the split path stops being
the thing under test:

  FullModel = embed + layers 0..63 + final_norm + lm_head, all resident
    layers 0..41  : bf16  (~36 GB)
    layers 42..63 : fp32  (~38 GB)   <- the J-space region of interest
    final_norm, lm_head : fp32 (matches the proven fp32 gate path)
    embed         : bf16
  ~80 GB total; the GPU host has ~99 GB free in the 121 GiB unified pool.

Reuses shape_c_loader's proven machinery (key map, rotary, Qwen3_5 layer
classes, full_reference) — no hand-rolled GDN. Text-only: model.visual.* and
mtp.* keys are EXCLUDED.

No hans-driver health gate: the engine is DOWN by plan for this window
(14:00-18:00, Claude stops it 14:02, restarts 18:00). Polling :$ENGINE_PORT would
abort by design, so it is omitted here.

Modes:
  --probe         imports + class discovery + full key map + memory budget
                  (no weights, no GPU)
  --verify        load whole model, forward, compare h_final vs the
                  layer-by-layer full reference (scl.full_reference), write
                  gate JSON, then FREE (exit). Proves the loader is correct.
  --hold          load + verify; if PASS, stay alive holding the model
                  resident (60 s heartbeat) until SIGTERM. This is what runs
                  in the window so the whole model is resident for J-Lens.

Safety: run under a capped systemd-run scope (MemoryMax) — see launch note.
"""

import argparse
import gc
import inspect
import json
import signal
import sys
import time

import torch
import torch.nn as nn
from safetensors import safe_open

sys.path.insert(0, "$JLENS_HOME")
import shape_c_loader as scl  # noqa: E402

N_LAYERS = scl.N_LAYERS          # 64
SPLIT = scl.SPLIT                # 42
DEFAULT_DIR = scl.DEFAULT_DIR
CACHE_DIR = scl.CACHE_DIR
PROMPT = scl.PROMPT

# full-model text key count = stream(559) + resident(292)
EXPECT_FULL_KEYS = scl.EXPECT_STREAM_KEYS + scl.EXPECT_RESIDENT_KEYS  # 851

# gate bar: same rel_l2 tolerance as shape_c gate2 (bf16 split vs full ref)
REL_L2_BAR = 2e-3


# ------------------------------------------------------------------- key map
def full_key_target(key):
    """Map a safetensors key -> ('skip', reason) | ('full', path) | ('unknown', key).

    Unlike scl.key_target (which splits at 42), this maps every text key to its
    path in the single FullModel (no split)."""
    if key.startswith("model.visual."):
        return ("skip", "visual")
    if key.startswith("mtp."):
        return ("skip", "mtp")
    if key == "lm_head.weight":
        return ("full", "lm_head.weight")
    if key.startswith("model.language_model."):
        rest = key[len("model.language_model."):]
        if rest == "embed_tokens.weight":
            return ("full", "embed.weight")
        if rest.startswith("layers."):
            parts = rest.split(".")
            idx = int(parts[1])
            sub = ".".join(parts[2:])
            return ("full", f"layers.{idx}.{sub}")
        if rest == "norm.weight":
            return ("full", "final_norm.weight")
        return ("full", rest)
    return ("unknown", key)


def is_fp32_region(path):
    """True for the fp32 region: layers 42..63, final_norm, lm_head."""
    if path == "lm_head.weight" or path == "final_norm.weight":
        return True
    if path.startswith("layers."):
        return int(path.split(".")[1]) >= SPLIT
    return False


# ------------------------------------------------------------- model building
class FullModel(nn.Module):
    """embed + all 64 layers + final_norm + lm_head, resident at once."""

    def __init__(self, config, rotary):
        super().__init__()
        m = scl.import_qwen35()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [m.Qwen3_5DecoderLayer(config, i) for i in range(N_LAYERS)])
        self.final_norm = m.Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rotary = rotary

    def forward(self, input_ids, return_logits=True):
        # Matches scl.full_reference exactly: cos/sin computed once from the
        # embed output and reused for all 64 layers (rotary is position-only).
        device = input_ids.device
        position_ids = scl.text_position_ids(input_ids.shape[1], device)
        h = self.embed(input_ids)
        cos, sin = scl.rotary_call(self.rotary, h, position_ids)
        for layer in self.layers:
            wdt = next(layer.parameters()).dtype
            if h.dtype != wdt:
                h = h.to(wdt)
            h = layer(h, position_embeddings=(cos, sin), position_ids=position_ids)
        h = self.final_norm(h)
        if return_logits:
            return self.lm_head(h)
        return h


def load_whole(model_dir, device):
    """Build FullModel, load all text keys, cast the 42..63 region to fp32.
    Returns (model, loaded, skipped, unknown, config, rotary, idx)."""
    from transformers import AutoConfig
    config = scl.text_cfg(AutoConfig.from_pretrained(model_dir))
    m = scl.import_qwen35()
    rotary = scl.make_rotary(config, scl.find_rotary_class(m)).to(device)
    # bf16 shell (Claude memtest post P013, proven 16:12: peak 70.4 GiB): the fp32
    # default-dtype shell is ~108 GB before any weight loads and OOM'd the box
    # (P011/P012). Construct with default dtype bf16, restore after — rotary
    # is built above, untouched, as in the proof.
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = FullModel(config, rotary)
    finally:
        torch.set_default_dtype(old)
    model = model.to(device)
    idx = scl.build_key_index(model_dir)
    t0 = time.time()

    loaded, skipped, unknown = [], [], []
    for key, shard in idx.items():
        t = full_key_target(key)
        if t[0] == "skip":
            skipped.append(key)
            continue
        if t[0] == "unknown":
            unknown.append(key)
            continue
        with safe_open(shard, framework="pt") as f:
            tensor = f.get_tensor(key).to(device)
        scl.set_attr(model, t[1], tensor)
        loaded.append(key)

    # cast the J-space region (42..63 + final_norm + lm_head) to fp32 —
    # PER TENSOR (load->cast->free): a bulk .float() on the region keeps the bf16
    # storage alive until the region copy completes, doubling peak to ~110 GB and
    # OOM-thrashing the 95 GiB cap (observed 14:47). Per-tensor: peak +1 tensor.
    for i in range(SPLIT, N_LAYERS):
        for name, param in list(model.layers[i].named_parameters()):
            param.data = param.data.to(torch.float32)
    model.final_norm.weight.data = model.final_norm.weight.data.to(torch.float32)
    model.lm_head.weight.data = model.lm_head.weight.data.to(torch.float32)

    dt = time.time() - t0
    print(f"whole model loaded {len(loaded)} keys in {dt:.1f}s "
          f"(skipped {len(skipped)} visual/mtp, unknown {len(unknown)})", flush=True)
    if unknown:
        print(f"UNKNOWN KEYS: {unknown[:10]}", flush=True)
    if len(loaded) != EXPECT_FULL_KEYS:
        print(f"WARNING: expected {EXPECT_FULL_KEYS} full keys, got {len(loaded)}", flush=True)
    model.eval()
    return model, loaded, skipped, unknown, config, rotary, idx


def mem_gib():
    try:
        return torch.cuda.memory_allocated() / 1024 ** 3
    except Exception:
        return -1.0


# --------------------------------------------------------------------- verify
def verify(model, model_dir, config, rotary, idx, device):
    """In-memory h_final vs the layer-by-layer full reference (all bf16 from
    disk). The in-memory model runs 42..63 in fp32, so the delta measures the
    bf16->fp32 shift in the upper stack. Bar: rel_l2 <= REL_L2_BAR."""
    input_ids = scl.encode_prompt(model_dir, device)
    with torch.no_grad():
        h_full = model(input_ids, return_logits=False)
    # full_reference peaks at ~1 layer on top of the resident model; run it
    # in the same dtype regime (42-63 fp32) so the delta isolates loader
    # error, not the bf16->fp32 shift of the upper stack
    with torch.no_grad():
        h_ref = scl.full_reference(model_dir, input_ids, idx, config, rotary, device,
                                   fp32_from=SPLIT)
    a, b = h_full.float(), h_ref.float()
    rel = ((a - b).norm() / b.norm()).item()
    maxabs = (a - b).abs().max().item()
    r = {
        "rel_l2": rel,
        "max_abs": maxabs,
        "bar": f"rel_l2 <= {REL_L2_BAR}",
        "pass": rel <= REL_L2_BAR,
        "note": ("whole-model in-memory (0-41 bf16 + 42-63 fp32) h_final vs "
                 "layer-by-layer full reference (all bf16 from disk)"),
    }
    return r, input_ids


# ---------------------------------------------------------------------- probe
def probe(model_dir):
    """No weights, no GPU: imports, class discovery, full key map, budget."""
    import transformers
    print(f"torch={torch.__version__} transformers={transformers.__version__}")
    m = scl.import_qwen35()
    rot = scl.find_rotary_class(m)
    print(f"rotary class: {rot.__name__} sig={inspect.signature(rot.__init__)}")
    from transformers import AutoConfig
    config = scl.text_cfg(AutoConfig.from_pretrained(model_dir))
    lt = config.layer_types
    gqa = [i for i, t in enumerate(lt) if t == "full_attention"]
    print(f"layers={len(lt)} L42={lt[42]} L43={lt[43]} GQA_idx={gqa}")
    print(f"hidden={config.hidden_size} vocab={config.vocab_size} eps={config.rms_norm_eps}")

    idx = scl.build_key_index(model_dir)
    counts = {"full": 0, "skip": 0, "unknown": 0}
    for k in idx:
        counts[full_key_target(k)[0]] += 1
    print(f"key map: {counts} (expect full={EXPECT_FULL_KEYS})")
    # budget from the plan (Claude P008 / post P009): 0-41 bf16 ~36 GB,
    # 42-63 fp32 ~38 GB, embed bf16 ~2.5 GB, lm_head fp32 ~5 GB, norm ~0
    print("budget (plan): 0-41 bf16 ~36 GB + 42-63 fp32 ~38 GB + embed bf16 ~2.5 GB "
          "+ lm_head fp32 ~5 GB ~= 81.5 GB (pool 121 GiB, ~99 GB free)")
    print(f"shards: {len(set(idx.values()))}")


# ----------------------------------------------------------------------- hold
def hold(model, device):
    """Stay alive holding the model resident until SIGTERM."""
    print(f"HOLD: model resident, mem {mem_gib():.1f} GiB; heartbeat 60s; "
          f"SIGTERM to release", flush=True)
    stop = {"flag": False}

    def _term(signum, frame):
        print("HOLD: SIGTERM received, releasing model", flush=True)
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    while not stop["flag"]:
        time.sleep(60)
        print(f"HOLD: alive, mem {mem_gib():.1f} GiB", flush=True)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("HOLD: released, exiting", flush=True)


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_DIR)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--hold", action="store_true")
    args = ap.parse_args()

    if args.probe:
        probe(args.model_dir)
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        # 0.70 fraction (~85 GiB of 121) per Claude P012: turns an OOM into a
        # clean Python error in-process instead of the kernel killing a neighbour.
        torch.cuda.set_per_process_memory_fraction(0.70)
        print("cuda memory fraction 0.70 set", flush=True)

    model, loaded, skipped, unknown, config, rotary, idx = load_whole(
        args.model_dir, device)
    print(f"post-load mem {mem_gib():.1f} GiB", flush=True)

    vr, _input_ids = verify(model, args.model_dir, config, rotary, idx, device)
    print(f"VERIFY: {json.dumps(vr)}", flush=True)

    result = {
        "model_dir": args.model_dir,
        "mode": "whole-model resident (0-41 bf16 + 42-63 fp32)",
        "full_keys": len(loaded),
        "skipped": len(skipped),
        "unknown": len(unknown),
        "verify": vr,
        "mem_gib_post_load": mem_gib(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = CACHE_DIR / "gate_results_whole_model.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"RESULTS: {out}", flush=True)

    if not vr["pass"]:
        print("VERIFY FAILED — not holding; freeing model", flush=True)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        sys.exit(1)

    if args.hold:
        hold(model, device)
    else:
        print("verify-only: freeing model", flush=True)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
