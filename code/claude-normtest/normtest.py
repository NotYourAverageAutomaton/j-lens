"""Claude normtest for Qwen's whole-model verify (2026-11-03). Does NOT modify Qwen's files and does
NOT write her results file. Patches ONE thing: scl.rms_norm_ref uses x*(1+w) (GemmaRMSNorm / HF
Qwen3_5RMSNorm convention) instead of x*w. Also captures the PRE-norm hidden state on both sides, so
the stack is compared independently of either norm convention."""
import json, sys, time, torch
sys.path.insert(0, "$JLENS_HOME")
torch.cuda.set_per_process_memory_fraction(0.70)
import whole_model_loader as wml
scl = wml.scl
cap = {}
_orig = scl.rms_norm_ref
def rms_norm_1pw(h, weight, eps):
    cap["ref_pre"] = h.detach().float().clone()
    h32 = h.float()
    return (h32 * torch.rsqrt(h32.pow(2).mean(-1, keepdim=True) + eps)
            * (1.0 + weight.float())).to(h.dtype)
scl.rms_norm_ref = rms_norm_1pw
dev = torch.device("cuda:0")
t0 = time.time()
model, loaded, skipped, unknown, config, rotary, idx = wml.load_whole(wml.DEFAULT_DIR, dev)
print(f"loaded {len(loaded)} keys in {time.time()-t0:.0f}s", flush=True)
model.final_norm.register_forward_pre_hook(lambda m, a: cap.__setitem__("mdl_pre", a[0].detach().float().clone()))
vr, ids = wml.verify(model, wml.DEFAULT_DIR, config, rotary, idx, dev)
a, b = cap["mdl_pre"], cap["ref_pre"]
pre = {"rel_l2": ((a-b).norm()/b.norm()).item(), "max_abs": (a-b).abs().max().item()}
out = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "patch": "ref final norm x*(1+w)",
       "verify_with_patch": vr, "pre_norm_stack": pre}
print("NORMTEST " + json.dumps(out), flush=True)
open("$JLENS_HOME/claude-normtest/normtest.json", "w").write(json.dumps(out, indent=2))
