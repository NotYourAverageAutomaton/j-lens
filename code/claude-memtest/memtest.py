"""Claude memtest for Qwen's whole_model_loader (2026-11-03, Niz 16:03 "relieve the memory bug").
Does NOT modify Qwen's file. Monkeypatches ONE thing: FullModel is constructed with default dtype
bf16 (rotary is built before it, unchanged), so the empty shell is ~54 GB instead of ~108 GB fp32.
Then: load_whole (Qwen's code, per-tensor fp32 cast of 42-63 unchanged), one forward, report peak."""
import json, sys, time, torch
sys.path.insert(0, "$JLENS_HOME")
torch.cuda.set_per_process_memory_fraction(0.70)
import whole_model_loader as wml

Base = wml.FullModel
class FullModelBF16Shell(Base):
    def __init__(self, *a, **k):
        old = torch.get_default_dtype(); torch.set_default_dtype(torch.bfloat16)
        try:
            super().__init__(*a, **k)
        finally:
            torch.set_default_dtype(old)
wml.FullModel = FullModelBF16Shell

G = 1024 ** 3
dev = torch.device("cuda:0")
r = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "cap_fraction": 0.70,
     "total_gib": torch.cuda.get_device_properties(0).total_memory / G}
t0 = time.time()
model, loaded, skipped, unknown, config, rotary, idx = wml.load_whole(wml.DEFAULT_DIR, dev)
r.update(load_s=round(time.time() - t0, 1), keys=len(loaded), unknown=len(unknown),
         alloc_after_load_gib=torch.cuda.memory_allocated() / G,
         peak_after_load_gib=torch.cuda.max_memory_allocated() / G)
dt = {}
for i, layer in enumerate(model.layers):
    for p in layer.parameters():
        k = f"{'0-41' if i < wml.SPLIT else '42-63'}:{str(p.dtype).replace('torch.', '')}"
        dt[k] = dt.get(k, 0) + p.numel()
r["param_dtypes"] = dt
ids = wml.scl.encode_prompt(wml.DEFAULT_DIR, dev)
with torch.no_grad():
    h = model(ids, return_logits=False)
r.update(seq=int(ids.shape[1]), h_dtype=str(h.dtype), h_finite=bool(torch.isfinite(h).all()),
         h_norm=float(h.float().norm()), peak_after_forward_gib=torch.cuda.max_memory_allocated() / G)
print("MEMTEST " + json.dumps(r), flush=True)
open("$JLENS_HOME/claude-memtest/memtest.json", "w").write(json.dumps(r, indent=2))
