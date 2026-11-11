#!/usr/bin/env python3
"""shape_c_loader.py — J-lens shape (c) loader. Card C02.

Splits a Qwen3_5 27B (64 layers; GDN/GQA hybrid; L42=GDN, L43=GQA) at layer 42:

  StreamModel   = embed_tokens + layers 0..41        (streamed once, no grad)
  ResidentModel = layers 42..63 + final_norm + lm_head  (~19.27 GB bf16)

Reuses transformers' reference Qwen3_5 layer classes — no hand-rolled GDN.
Text-only: model.visual.* keys are EXCLUDED.

Phases (sequential; peak ~37 GB, fits the ~53 GiB the GPU host budget):
  A  stream:   load 0..41, run prompts, cache h_l (final valid position) per prompt, free
  B  resident: load 42..63 + final_norm + lm_head, verify key coverage,
               gate 1 (GDN-42 finite difference), gate 2 (split h_final vs full reference)

Modes:
  --probe     imports + class discovery + key map + budget (no weights, no GPU)
  --stream    phase A only
  --gates     phase B only (needs the h_l cache from --stream)
  --all       A then B

Safety:
  * hans-driver health gate: GET :$ENGINE_PORT/health before each phase + 30 s watch
    thread; abort if it drops.
  * run under a capped systemd-run unit (see card C02 comment).
"""

import argparse
import gc
import inspect
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

SPLIT = 42
N_LAYERS = 64
HEALTH_URL = "http://127.0.0.1:$ENGINE_PORT/health"
DEFAULT_DIR = "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b"
CACHE_DIR = Path("$JLENS_HOME")

PROMPT = ("The quick brown fox jumps over the lazy dog. "
          "Explain gated delta net attention in one sentence.")

# expected key counts from the 09:47 header probe (card C02)
EXPECT_STREAM_KEYS = 559
EXPECT_RESIDENT_KEYS = 292


# ---------------------------------------------------------------- health gate

def health_ok():
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


class HealthWatch:
    def __init__(self, interval=30):
        self.interval = interval
        self.stopped = False
        self._t = None

    def start(self):
        def loop():
            while not self.stopped:
                time.sleep(self.interval)
                if not health_ok():
                    print("HEALTH WATCH: hans-driver /health DROPPED", flush=True)
                    self.stopped = True
        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()

    def check(self, phase):
        if self.stopped or not health_ok():
            sys.exit(f"ABORT: hans-driver health dropped during {phase}")

    def end(self):
        self.stopped = True


# ------------------------------------------------------------------- key map

def key_target(key):
    """Map a safetensors key -> ('skip', reason) | ('stream'|'resident', path) | ('unknown', key)."""
    if key.startswith("model.visual."):
        return ("skip", "visual")
    if key.startswith("mtp."):
        return ("skip", "mtp")
    if key == "lm_head.weight":
        return ("resident", "lm_head.weight")
    if key.startswith("model.language_model."):
        rest = key[len("model.language_model."):]
        if rest == "embed_tokens.weight":
            return ("stream", "embed.weight")
        if rest.startswith("layers."):
            parts = rest.split(".")
            idx = int(parts[1])
            sub = ".".join(parts[2:])
            if idx < SPLIT:
                return ("stream", f"layers.{idx}.{sub}")
            return ("resident", f"layers.{idx - SPLIT}.{sub}")
        # top-level non-layer language keys -> resident (norm -> final_norm attr)
        if rest == "norm.weight":
            return ("resident", "final_norm.weight")
        return ("resident", rest)
    return ("unknown", key)


def build_key_index(model_dir):
    idx = {}
    for shard in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            for k in f.keys():
                idx[k] = str(shard)
    return idx


def set_attr(model, path, tensor):
    parts = path.split(".")
    obj = model
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    leaf = parts[-1]
    cur = getattr(obj, leaf, None)
    if isinstance(cur, nn.Parameter):
        setattr(obj, leaf, nn.Parameter(tensor.detach(), requires_grad=False))
    else:
        setattr(obj, leaf, tensor.detach())


def load_into(model, model_dir, which, idx, device):
    """Load all keys mapped to `which` into model. Tensors land on `device`
    (a CPU tensor here would device-mismatch the on-device forward).
    Returns (loaded, skipped, unknown)."""
    loaded, skipped, unknown = [], [], []
    for key, shard in idx.items():
        t = key_target(key)
        if t[0] == "skip":
            skipped.append(key)
            continue
        if t[0] == "unknown":
            unknown.append(key)
            continue
        if t[0] != which:
            continue
        with safe_open(shard, framework="pt") as f:
            tensor = f.get_tensor(key).to(device)
        set_attr(model, t[1], tensor)
        loaded.append(key)
    return loaded, skipped, unknown


def get_tensor(model_dir, key, idx, device):
    with safe_open(idx[key], framework="pt") as f:
        return f.get_tensor(key).to(device)


# ------------------------------------------------------------- model building

def import_qwen35():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    return m


def find_rotary_class(m):
    cands = [getattr(m, n) for n in dir(m)
             if "RotaryEmbedding" in n and isinstance(getattr(m, n), type)]
    if not cands:
        raise RuntimeError("no RotaryEmbedding class in qwen3_5 module")
    return cands[0]


def text_cfg(config):
    """Composite multimodal config -> language-model sub-config (has layer_types)."""
    return getattr(config, "text_config", config)


def make_rotary(config, cls):
    last_err = None
    for attempt in (lambda: cls(config),
                    lambda: cls(config.rope_parameters) if hasattr(config, "rope_parameters") else None):
        try:
            if attempt is None:
                continue
            return attempt()
        except Exception as e:  # try next signature
            last_err = e
    raise RuntimeError(f"cannot instantiate rotary embedding: {last_err}")


def rotary_call(rot, h, position_ids):
    out = rot(h, position_ids)
    if isinstance(out, tuple):
        return out[0], out[1]
    return out.cos, out.sin


def text_position_ids(seq_len, device):
    """Qwen3_5 rotary is mrope-style: position_ids shape (3, bs, seq).
    Pure text: all three rows (T/H/W) are the same arange."""
    return torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(0).expand(3, 1, seq_len)


class StreamModel(nn.Module):
    def __init__(self, config, rotary):
        super().__init__()
        m = import_qwen35()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [m.Qwen3_5DecoderLayer(config, i) for i in range(SPLIT)])
        self.rotary = rotary

    def forward(self, input_ids):
        device = input_ids.device
        position_ids = text_position_ids(input_ids.shape[1], device)
        h = self.embed(input_ids)
        cos, sin = rotary_call(self.rotary, h, position_ids)
        for layer in self.layers:
            h = layer(h, position_embeddings=(cos, sin), position_ids=position_ids)
        return h


class ResidentModel(nn.Module):
    def __init__(self, config, rotary):
        super().__init__()
        m = import_qwen35()
        self.layers = nn.ModuleList(
            [m.Qwen3_5DecoderLayer(config, i + SPLIT) for i in range(N_LAYERS - SPLIT)])
        self.final_norm = m.Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rotary = rotary

    def forward(self, h, return_logits=True, pos_offset=0):
        position_ids = text_position_ids(h.shape[1], h.device) + pos_offset
        cos, sin = rotary_call(self.rotary, h, position_ids)
        for layer in self.layers:
            wdt = next(layer.parameters()).dtype
            if h.dtype != wdt:
                h = h.to(wdt)
            h = layer(h, position_embeddings=(cos, sin), position_ids=position_ids)
        h = self.final_norm(h)
        if return_logits:
            return self.lm_head(h)
        return h


# --------------------------------------------------------------------- gates

def finite_diff_direction(model, h, param, eps, seed):
    """Forward + central finite-diff direction of model output w.r.t. `param`."""
    g = torch.Generator(device=param.device).manual_seed(seed)
    e = torch.randn(param.shape, generator=g, device=param.device, dtype=param.dtype)
    base = param.detach().clone()
    with torch.no_grad():
        y0 = model(h, return_logits=False).float()
        param.add_(eps * e)
        y_p = model(h, return_logits=False).float()
        param.sub_(2 * eps * e)
        y_m = model(h, return_logits=False).float()
        param.copy_(base)
    d_fwd = (y_p - y0) / eps
    d_cen = (y_p - y_m) / (2 * eps)
    return d_fwd, d_cen


def cos_sim(a, b):
    return F.cosine_similarity(a.flatten().unsqueeze(0),
                               b.flatten().unsqueeze(0)).item()


def gate1(resident, h):
    """GDN-42 finite-difference gate.
    PASS: cos(fwd, central) for a layer-42 (GDN) weight >= 0.99,
          AND the wrong-layer control (layer 43, GQA) direction is distinct
          (cos(central42, central43) < 0.99)."""
    w42 = resident.layers[0].linear_attn.in_proj_qkv.weight
    w43 = resident.layers[1].self_attn.q_proj.weight
    eps = 0.01 * w42.abs().mean().item()
    f42, c42 = finite_diff_direction(resident, h, w42, eps, seed=0)
    f43, c43 = finite_diff_direction(resident, h, w43, eps, seed=1)
    r = {
        "cos_fwd_central_L42_GDN": cos_sim(f42, c42),
        "cos_L42_vs_L43_control": cos_sim(c42, c43),
        "eps": eps,
        "pass": cos_sim(f42, c42) >= 0.99 and cos_sim(c42, c43) < 0.99,
    }
    return r


def rms_norm_ref(h, weight, eps):
    h32 = h.float()
    return (h32 * torch.rsqrt(h32.pow(2).mean(-1, keepdim=True) + eps)
            * (1.0 + weight.float())).to(h.dtype)


def full_reference(model_dir, input_ids, idx, config, rotary, device, fp32_from=None):
    """Full 64-layer forward, layer-by-layer from disk (peak ~1 layer in RAM).

    fp32_from: if set, layers fp32_from..N_LAYERS-1 are run in fp32 (weights
    cast on load, activations kept fp32 through to final_norm) to match a
    mixed-dtype in-memory model. None = all-bf16 (original behavior)."""
    m = import_qwen35()
    embed_w = get_tensor(model_dir, "model.language_model.embed_tokens.weight", idx, device)
    h = F.embedding(input_ids, embed_w)
    position_ids = text_position_ids(input_ids.shape[1], device)
    cos, sin = rotary_call(rotary, h, position_ids)
    for i in range(N_LAYERS):
        fp32_layer = fp32_from is not None and i >= fp32_from
        if fp32_layer:
            h = h.float()
            cos = cos.float()
            sin = sin.float()
        layer = m.Qwen3_5DecoderLayer(config, i).to(device)
        if fp32_layer:
            layer = layer.to(torch.float32)
        prefix = f"model.language_model.layers.{i}."
        for key, shard in idx.items():
            if key.startswith(prefix):
                with safe_open(shard, framework="pt") as f:
                    t = f.get_tensor(key).to(device)
                    if fp32_layer:
                        t = t.to(torch.float32)
                    set_attr(layer, key[len(prefix):], t)
        h = layer(h, position_embeddings=(cos, sin), position_ids=position_ids)
        del layer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    norm_w = get_tensor(model_dir, "model.language_model.norm.weight", idx, device)
    if fp32_from is not None:
        h = h.float()
        norm_w = norm_w.float()
    h = rms_norm_ref(h, norm_w, config.rms_norm_eps)
    return h


def gate2(h_split, model_dir, input_ids, idx, config, rotary, device):
    """Split h_final vs full-model reference, bf16 tolerance.
    Caller frees the resident model BEFORE calling, so peak = baseline + 1 layer."""
    with torch.no_grad():
        h_full = full_reference(model_dir, input_ids, idx, config, rotary, device)
    a, b = h_split.float(), h_full.float()
    rel = ((a - b).norm() / b.norm()).item()
    maxabs = (a - b).abs().max().item()
    return {"rel_l2": rel, "max_abs": maxabs, "pass": rel <= 2e-3}


# --------------------------------------------------------------------- phases

def get_tokenizer(model_dir):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    except Exception as e:
        print(f"tokenizer unavailable ({e}); using seeded random ids", flush=True)
        return None


def encode_prompt(model_dir, device):
    tok = get_tokenizer(model_dir)
    if tok is not None:
        ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
    else:
        g = torch.Generator(device=device).manual_seed(0)
        ids = torch.randint(0, 248320, (1, 48), generator=g, device=device)
    return ids


def phase_stream(model_dir, device, watch):
    """True layer-by-layer streaming: peak = baseline + embed + 1 layer.
    (Loading all 42 layers at once = 36.29 GB; OOM-killed the GB10 unified
    pool on 11-02 11:42 — the scope cap competes with hans-driver for the
    SAME 121 GiB pool, and only ~54 GiB was free.)"""
    watch.check("stream-load")
    from transformers import AutoConfig
    config = text_cfg(AutoConfig.from_pretrained(model_dir))
    m = import_qwen35()
    rotary = make_rotary(config, find_rotary_class(m)).to(device)
    idx = build_key_index(model_dir)
    t0 = time.time()

    unknown = [k for k in idx if key_target(k)[0] == "unknown"]
    if unknown:
        print(f"UNKNOWN KEYS: {unknown[:10]}", flush=True)

    # embed (1 key, ~2.5 GB) -> h, then free
    embed_w = get_tensor(model_dir, "model.language_model.embed_tokens.weight", idx, device)
    input_ids = encode_prompt(model_dir, device)
    h = F.embedding(input_ids, embed_w)
    del embed_w
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    loaded = 1

    position_ids = text_position_ids(input_ids.shape[1], device)
    cos, sin = rotary_call(rotary, h, position_ids)
    with torch.no_grad():
        for i in range(SPLIT):
            watch.check(f"stream-layer-{i}")
            layer = m.Qwen3_5DecoderLayer(config, i).to(device)
            prefix = f"model.language_model.layers.{i}."
            for key, shard in idx.items():
                if key.startswith(prefix):
                    with safe_open(shard, framework="pt") as f:
                        set_attr(layer, key[len(prefix):], f.get_tensor(key).to(device))
                    loaded += 1
            h = layer(h, position_embeddings=(cos, sin), position_ids=position_ids)
            del layer
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print(f"stream ran {SPLIT} layers, {loaded} keys in {time.time()-t0:.1f}s", flush=True)
    if loaded != EXPECT_STREAM_KEYS:
        print(f"WARNING: expected {EXPECT_STREAM_KEYS} stream keys, got {loaded}", flush=True)

    h_l = h[0, -1].cpu()  # final valid position, one tensor per prompt
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"hl_cache_{Path(model_dir).name}.pt"
    torch.save({"prompt": PROMPT, "n_tokens": int(input_ids.shape[1]),
                "input_ids": input_ids.cpu(), "h_l": h_l}, cache_path)
    print(f"h_l cached: {cache_path} shape={tuple(h_l.shape)} dtype={h_l.dtype}", flush=True)
    del h
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cache_path


def phase_gates(model_dir, device, watch):
    watch.check("resident-load")
    from transformers import AutoConfig
    config = text_cfg(AutoConfig.from_pretrained(model_dir))
    m = import_qwen35()
    rotary = make_rotary(config, find_rotary_class(m)).to(device)
    resident = ResidentModel(config, rotary).to(device)
    idx = build_key_index(model_dir)
    t0 = time.time()
    loaded, skipped, unknown = load_into(resident, model_dir, "resident", idx, device)
    print(f"resident loaded {len(loaded)} keys in {time.time()-t0:.1f}s "
          f"(skipped {len(skipped)} visual, unknown {len(unknown)})", flush=True)
    if unknown:
        print(f"UNKNOWN KEYS: {unknown[:10]}", flush=True)
    if len(loaded) != EXPECT_RESIDENT_KEYS:
        print(f"WARNING: expected {EXPECT_RESIDENT_KEYS} resident keys, got {len(loaded)}", flush=True)
    resident.eval()

    cache_path = CACHE_DIR / f"hl_cache_{Path(model_dir).name}.pt"
    cache = torch.load(cache_path, map_location="cpu")
    h_l = cache["h_l"].to(device).unsqueeze(0).unsqueeze(0)  # [1,1,H]
    input_ids = cache["input_ids"].to(device)

    watch.check("gate1")
    g1 = gate1(resident, h_l)
    print(f"GATE1: {json.dumps(g1)}", flush=True)

    watch.check("gate2")
    with torch.no_grad():
        h_split = resident(h_l, return_logits=False, pos_offset=int(cache["n_tokens"]) - 1)
    del resident  # free 19.27 GB before the full 64-layer reference
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    g2 = gate2(h_split, model_dir, input_ids, idx, config, rotary, device)
    print(f"GATE2: {json.dumps(g2)}", flush=True)

    result = {"model_dir": model_dir, "gate1": g1, "gate2": g2,
              "resident_keys": len(loaded), "stream_keys_expected": EXPECT_STREAM_KEYS}
    out = CACHE_DIR / "gate_results.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"RESULTS: {out}", flush=True)
    return result


def probe(model_dir):
    """No weights, no GPU: imports, class discovery, key map, budget."""
    import transformers
    print(f"torch={torch.__version__} transformers={transformers.__version__}")
    m = import_qwen35()
    rot = find_rotary_class(m)
    print(f"rotary class: {rot.__name__} sig={inspect.signature(rot.__init__)}")
    from transformers import AutoConfig
    config = text_cfg(AutoConfig.from_pretrained(model_dir))
    lt = config.layer_types
    gqa = [i for i, t in enumerate(lt) if t == "full_attention"]
    print(f"layers={len(lt)} L42={lt[42]} L43={lt[43]} GQA_idx={gqa}")
    print(f"hidden={config.hidden_size} vocab={config.vocab_size} eps={config.rms_norm_eps}")
    idx = build_key_index(model_dir)
    counts = {"stream": 0, "resident": 0, "skip": 0, "unknown": 0}
    top = set()
    for k in idx:
        t = key_target(k)
        counts[t[0]] += 1
        if t[0] == "resident" and not t[1].startswith("layers.") and t[1] != "lm_head.weight":
            top.add(t[1].split(".")[0])
    print(f"key map: {counts} (expect stream={EXPECT_STREAM_KEYS} resident={EXPECT_RESIDENT_KEYS})")
    print(f"resident top-level non-layer keys: {sorted(top)}")
    print(f"shards: {len(set(idx.values()))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_DIR)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.probe:
        probe(args.model_dir)
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    watch = HealthWatch()
    watch.start()
    try:
        if args.stream or args.all:
            phase_stream(args.model_dir, device, watch)
        if args.gates or args.all:
            phase_gates(args.model_dir, device, watch)
        if not (args.stream or args.gates or args.all):
            ap.print_help()
    finally:
        watch.end()


if __name__ == "__main__":
    main()
