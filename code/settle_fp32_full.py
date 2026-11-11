#!/usr/bin/env python3
"""settle_fp32_full.py — Deron's settling run (i): FULL-stack fp32, gate-1 bf16 verdict.
Card C02. Deron P003 ruling: run (i) alone NOW — whole resident stack + h_l
in fp32, same eps, report cos(fwd,central) L42 + the wrong-layer control. No new JVP
code (that is arm (ii), the immediate follow-up).

SINGLE VARIABLE vs the bf16 gate1 run (cos_fwd_central_L42 = 0.394) and vs the mixed
fp32 falsifier (cos = 0.505, layers 42/43 only): the ENTIRE resident stack
(layers 42..63 + final_norm + lm_head) AND h_l are fp32. This removes the downstream
bf16 re-quantization confound (layers 44..63) that the mixed falsifier left in place.

Bar (Deron P002 / post P003): cos(fwd,central) L42 >= 0.99 AND the wrong-layer control
(layer 43, GQA) stays LOW / non-saturating (< 0.99). Clear -> gate-1 bf16 verdict
PASS. Low -> bf16 not the cause; escalate to JVP (arm ii).

Memory: full resident fp32 (~33 GiB peak). the GPU host reported ~55 GiB available at launch.
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


def load_resident_fp32(model_dir, device):
    """Load resident, cast the WHOLE stack to fp32. Returns (resident, loaded, unknown)."""
    from transformers import AutoConfig
    config = scl.text_cfg(AutoConfig.from_pretrained(model_dir))
    m = scl.import_qwen35()
    rotary = scl.make_rotary(config, scl.find_rotary_class(m)).to(device)
    resident = scl.ResidentModel(config, rotary).to(device)
    idx = scl.build_key_index(model_dir)
    t0 = time.time()
    loaded, skipped, unknown = scl.load_into(resident, model_dir, "resident", idx, device)
    # single variable: cast the ENTIRE resident stack (all layers + final_norm +
    # lm_head + rotary) to fp32 — not just layers 42/43 as in the mixed falsifier.
    resident.float()
    print(f"resident loaded {len(loaded)} keys in {time.time()-t0:.1f}s "
          f"(full stack -> fp32; mem {mem_gib():.1f} GiB)", flush=True)
    resident.eval()
    return resident, loaded, unknown


def main():
    watch = scl.HealthWatch()
    watch.start()
    try:
        watch.check("resident-load")
        resident, loaded, unknown = load_resident_fp32(MODEL_DIR, DEVICE)
        print(f"loaded={len(loaded)} unknown={len(unknown)}", flush=True)

        cache = torch.load(CACHE, map_location="cpu")
        h_l = cache["h_l"].to(DEVICE).float().unsqueeze(0).unsqueeze(0)  # [1,1,H] fp32
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
            "dtype": "full-stack fp32 (all resident layers 42..63 + final_norm + lm_head; h_l fp32)",
            "cos_fwd_central_L42_GDN": cos_fc_42,
            "cos_L42_vs_L43_control": cos_ctrl,
            "eps": eps,
            "bar": "cos_fwd_central_L42 >= 0.99 AND control < 0.99",
            "pass": cos_fc_42 >= 0.99 and cos_ctrl < 0.99,
            "bf16_baseline_cos": 0.3944758474826813,
            "mixed_fp32_falsifier_cos": 0.5052523016929626,
            "note": "settling run (i) per Deron P003: full-stack fp32 removes the "
                    "downstream bf16 re-quantization confound (layers 44..63). "
                    "Arm (ii) FD-vs-JVP is the immediate follow-up.",
        }
        out = scl.CACHE_DIR / "gate_results_settle_fp32_full.json"
        out.write_text(json.dumps(r, indent=2))
        print(f"SETTLE_FP32_FULL: {json.dumps(r)}", flush=True)
        print(f"RESULTS: {out}", flush=True)
    finally:
        watch.end()


if __name__ == "__main__":
    main()
