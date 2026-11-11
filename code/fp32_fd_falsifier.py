#!/usr/bin/env python3
"""fp32_fd_falsifier.py — Deron's fp32-FD falsifier for gate 1. Card C02.

Deron P002 ruling, point 2 (verbatim intent): "cast resident 42,43 -> e/nudge/
3 forwards fp32". SINGLE VARIABLE vs the bf16 gate1 run (cos_fwd_central_L42 =
0.394): the two PERTURBED layers (global 42 = resident[0], GDN; global 43 =
resident[1], GQA control) are cast to fp32. The rest of the resident stays bf16.
h_l stays bf16 (matches the bf16 run).

Bar (Deron P002): cos(fwd,central) L42 >= 0.99 AND wrong-layer control stays
LOW / non-saturating (< 0.99). Clear -> gate 1 PASS, go to gate 2. Low -> bf16
not the cause, escalate to JVP.

Memory: bf16 resident (~19.3 GB) + 2 layers upcast to fp32 (~+3.4 GB) = ~22.7 GB.
Fits the ~56 GiB pool comfortably (the full-22-layer-fp32 variant peaked 33 GiB
and hit NV_ERR_NO_MEMORY with VLLM co-resident — rejected).

NOTE: this isolates the dtype of the PERTURBED layers only. The downstream bf16
layers (44..63) still re-quantize the propagated signal, so a PASS here means
the bf16 rounding in layers 42,43 was a dominant factor; a FAIL means escalate
to JVP (the autograd check, which has no FD-resolution issue). The position_ids
bug (resident forward uses pos 0, not 42..47) is present but CANCELS in the
cosine (fwd and central share the same position) — it is gate 2's problem.
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "$JLENS_HOME")
import shape_c_loader as scl

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_DIR = scl.DEFAULT_DIR
CACHE = scl.CACHE_DIR / f"hl_cache_{Path(MODEL_DIR).name}.pt"


def mem_gib():
    try:
        return torch.cuda.memory_allocated() / 1024 ** 3
    except Exception:
        return -1.0


def load_resident_mixed(model_dir, device):
    """Load resident in bf16, then cast ONLY layers 0,1 (global 42,43) to fp32.
    Returns (resident, loaded, unknown)."""
    from transformers import AutoConfig
    config = scl.text_cfg(AutoConfig.from_pretrained(model_dir))
    m = scl.import_qwen35()
    rotary = scl.make_rotary(config, scl.find_rotary_class(m)).to(device)
    resident = scl.ResidentModel(config, rotary).to(device)
    idx = scl.build_key_index(model_dir)
    t0 = time.time()
    loaded, skipped, unknown = scl.load_into(resident, model_dir, "resident", idx, device)
    # single variable: cast ONLY the two perturbed layers to fp32
    resident.layers[0].float()
    resident.layers[1].float()
    print(f"resident loaded {len(loaded)} keys in {time.time()-t0:.1f}s "
          f"(layers 0,1 -> fp32; mem {mem_gib():.1f} GiB)", flush=True)
    resident.eval()
    return resident, loaded, unknown


def main():
    watch = scl.HealthWatch()
    watch.start()
    try:
        watch.check("resident-load")
        resident, loaded, unknown = load_resident_mixed(MODEL_DIR, DEVICE)
        print(f"loaded={len(loaded)} unknown={len(unknown)}", flush=True)

        cache = torch.load(CACHE, map_location="cpu")
        h_l = cache["h_l"].to(DEVICE).unsqueeze(0).unsqueeze(0)  # [1,1,H] bf16 (matches bf16 run)
        print(f"h_l {tuple(h_l.shape)} {h_l.dtype} (mem {mem_gib():.1f} GiB)", flush=True)

        # FD — same eps rule as gate1 (0.01 * mean|w42|); w42 is now fp32
        w42 = resident.layers[0].linear_attn.in_proj_qkv.weight
        w43 = resident.layers[1].self_attn.q_proj.weight
        eps = 0.01 * w42.abs().mean().item()
        print(f"eps={eps:.6g} w42.dtype={w42.dtype} w43.dtype={w43.dtype}", flush=True)

        watch.check("fd-L42")
        f42, c42 = scl.finite_diff_direction(resident, h_l, w42, eps, seed=0)
        print(f"L42 FD done (mem {mem_gib():.1f} GiB)", flush=True)
        watch.check("fd-L43")
        f43, c43 = scl.finite_diff_direction(resident, h_l, w43, eps, seed=1)
        print(f"L43 FD done (mem {mem_gib():.1f} GiB)", flush=True)

        cos_fc_42 = scl.cos_sim(f42, c42)
        cos_ctrl = scl.cos_sim(c42, c43)
        r = {
            "dtype": "mixed (layers 42,43 fp32; rest bf16; h_l bf16)",
            "cos_fwd_central_L42_GDN": cos_fc_42,
            "cos_L42_vs_L43_control": cos_ctrl,
            "eps": eps,
            "bar": "cos_fwd_central_L42 >= 0.99 AND control < 0.99",
            "pass": cos_fc_42 >= 0.99 and cos_ctrl < 0.99,
            "bf16_baseline_cos": 0.3944758474826813,
            "note": "isolates dtype of perturbed layers only; downstream bf16 re-quantizes. "
                    "pos_ids bug cancels in cosine (gate2's problem).",
        }
        out = scl.CACHE_DIR / "gate_results_fp32_fd.json"
        out.write_text(json.dumps(r, indent=2))
        print(f"FP32_FD: {json.dumps(r)}", flush=True)
        print(f"RESULTS: {out}", flush=True)
    finally:
        watch.end()


if __name__ == "__main__":
    main()
