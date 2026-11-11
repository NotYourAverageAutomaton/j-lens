#!/usr/bin/env python3
"""fd_vs_jvp_full.py — Arm (ii): FD-vs-JVP on the FULL fp32 path. Card C02.

FIX (Claude P006): the 01:17 run crashed at `in_proj.weight = p` (nn.Module
won't take a plain tensor as a Parameter) AND had a deeper bug — fwd(q) called
resident(h_l) and ignored q, so the JVP tangent never reached the forward and
jvp_out would come out ~0 (cos nan/garbage). One change fixes both: route q
into the forward via torch.func.functional_call, drop the weight swap/restore.

Bar (Deron P005/P007):
  PASS  = cos(FD_L42, JVP_L42) >= 0.99
          AND control cos(FD_L42, JVP_L43_seed1) clearly low (< ~0.9).
  control >= ~0.9 = non-discriminating = FAIL.
  rel_l2 is REPORT-ONLY here (1e-5 is the gate-2 bar, not arm (ii)'s).

Sanity before trusting the number: jvp_out.norm() must be non-zero and the
same order as d_cen.norm().
"""
import json
import sys
import time
from pathlib import Path

import torch
from torch.func import functional_call, jvp

sys.path.insert(0, "$JLENS_HOME")
import shape_c_loader as scl

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_DIR = scl.DEFAULT_DIR
CACHE = scl.CACHE_DIR / f"hl_cache_{Path(MODEL_DIR).name}.pt"
OUT = scl.CACHE_DIR / "gate_results_fd_vs_jvp.json"

NAME_L42 = "layers.0.linear_attn.in_proj_qkv.weight"   # GDN
NAME_L43 = "layers.1.self_attn.q_proj.weight"          # GQA (control)


def mem_gib():
    try:
        return torch.cuda.memory_allocated() / 1024 ** 3
    except Exception:
        return -1.0


def load_resident_fp32(model_dir, device):
    """Identical to settle_fp32_full: load resident, cast WHOLE stack to fp32."""
    from transformers import AutoConfig
    config = scl.text_cfg(AutoConfig.from_pretrained(model_dir))
    m = scl.import_qwen35()
    rotary = scl.make_rotary(config, scl.find_rotary_class(m)).to(device)
    resident = scl.ResidentModel(config, rotary).to(device)
    idx = scl.build_key_index(model_dir)
    t0 = time.time()
    loaded, skipped, unknown = scl.load_into(resident, model_dir, "resident", idx, device)
    resident.float()
    print(f"resident loaded {len(loaded)} keys in {time.time()-t0:.1f}s "
          f"(full stack -> fp32; mem {mem_gib():.1f} GiB)", flush=True)
    resident.eval()
    return resident, loaded, unknown


def seeded_dir(shape, device, dtype, seed):
    """Same generator recipe as scl.finite_diff_direction so directions match."""
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(shape, generator=g, device=device, dtype=dtype)


def main():
    watch = scl.HealthWatch()
    watch.start()
    try:
        watch.check("resident-load")
        resident, loaded, unknown = load_resident_fp32(MODEL_DIR, DEVICE)
        print(f"loaded={len(loaded)} unknown={len(unknown)}", flush=True)

        # Confirm the parameter names we're about to override actually exist.
        pnames = dict(resident.named_parameters())
        for nm in (NAME_L42, NAME_L43):
            if nm not in pnames:
                raise KeyError(f"param {nm!r} not in named_parameters; "
                               f"have {len(pnames)} params")
        print(f"param names confirmed: {NAME_L42}, {NAME_L43}", flush=True)

        cache = torch.load(CACHE, map_location="cpu")
        h_l = cache["h_l"].to(DEVICE).float().unsqueeze(0).unsqueeze(0)  # [1,1,H] fp32
        print(f"h_l {tuple(h_l.shape)} {h_l.dtype} (mem {mem_gib():.1f} GiB)", flush=True)

        w42 = resident.layers[0].linear_attn.in_proj_qkv.weight
        w43 = resident.layers[1].self_attn.q_proj.weight
        eps = 0.01 * w42.abs().mean().item()
        print(f"eps={eps:.6g} w42.dtype={w42.dtype}", flush=True)

        # --- FD central, L42, seed=0 (the direction the JVP must match) ---
        watch.check("fd-L42")
        f42, c42 = scl.finite_diff_direction(resident, h_l, w42, eps, seed=0)
        d_cen = c42.float()
        print(f"L42 FD done (mem {mem_gib():.1f} GiB)", flush=True)

        # --- JVP L42 seed=0: exact directional derivative J@e via forward-mode AD ---
        # functional_call routes q into the forward so the tangent reaches it.
        e42 = seeded_dir(w42.shape, w42.device, w42.dtype, seed=0)

        def fwd42(q):
            return functional_call(resident, {NAME_L42: q}, (h_l,),
                                   {"return_logits": False})

        watch.check("jvp-L42")
        y42, jvp42 = jvp(fwd42, (w42.detach(),), (e42,))
        jvp42 = jvp42.float()
        print(f"L42 JVP done (mem {mem_gib():.1f} GiB)", flush=True)

        # --- Control: JVP on the WRONG layer (L43 GQA) with seed=1 direction ---
        e43 = seeded_dir(w43.shape, w43.device, w43.dtype, seed=1)

        def fwd43(q):
            return functional_call(resident, {NAME_L43: q}, (h_l,),
                                   {"return_logits": False})

        watch.check("jvp-L43-control")
        y43, jvp43 = jvp(fwd43, (w43.detach(),), (e43,))
        jvp43 = jvp43.float()
        print(f"L43 JVP control done (mem {mem_gib():.1f} GiB)", flush=True)

        # --- sanity: JVP norm must be non-zero, same order as FD norm ---
        fd_norm = d_cen.norm().item()
        j42_norm = jvp42.norm().item()
        j43_norm = jvp43.norm().item()
        if j42_norm == 0.0:
            raise RuntimeError("jvp42 norm is zero — tangent did not reach the "
                               "forward; number is meaningless")
        print(f"norms: fd={fd_norm:.6g} jvp42={j42_norm:.6g} jvp43={j43_norm:.6g}",
              flush=True)

        # --- compare ---
        diff = d_cen - jvp42
        rel_l2 = (diff.norm() / jvp42.norm()).item()
        cos_main = scl.cos_sim(d_cen, jvp42)
        cos_ctrl = scl.cos_sim(d_cen, jvp43)

        r = {
            "arm": "(ii) FD-vs-JVP, L42 GDN, central FD vs torch.func.jvp (functional_call)",
            "dtype": "full-stack fp32 (all resident layers 42..63 + final_norm + lm_head; h_l fp32)",
            "cos_fd_vs_jvp_L42": cos_main,
            "cos_control_L42_vs_L43_seed1": cos_ctrl,
            "rel_l2_fd_vs_jvp": rel_l2,
            "fd_norm": fd_norm,
            "jvp_L42_norm": j42_norm,
            "jvp_L43_norm": j43_norm,
            "eps": eps,
            "bar": "cos_main >= 0.99 AND cos_ctrl < ~0.9 (non-discriminating if ctrl >= ~0.9); "
                   "rel_l2 report-only (1e-5 is gate-2)",
            "pass": (cos_main >= 0.99) and (cos_ctrl < 0.9),
            "note": "arm (ii) per Deron P004 sequence (ii)->gate2->fit. functional_call "
                    "fix per Claude P006 (crash + dead-tangent). Same full fp32 path + "
                    "seeded directions as arm (i).",
        }
        OUT.write_text(json.dumps(r, indent=2))
        print(f"FD_VS_JVP: {json.dumps(r)}", flush=True)
        print(f"RESULTS: {OUT}", flush=True)
    finally:
        watch.end()


if __name__ == "__main__":
    main()
