"""Claude resume driver for the J-Lens record fit (2026-11-03, Niz 23:09: no monkey in the middle).
Runs Qwen's whole_model_fit.main() UNCHANGED, but with PROMPTS = PROMPTS[K:], into a SEPARATE out dir.
Merge afterwards with merge.py: acc and tot are running sums, so (acc_a+acc_b)/(tot_a+tot_b) is exact.
Usage: python resume.py K OUTDIR   (K = number of prompts already complete in EVERY layer of the first run)"""
import sys
sys.path.insert(0, "$JLENS_HOME")
K, OUT = int(sys.argv[1]), sys.argv[2]
import os
_rm = os.remove   # main()'s finalize deletes acc_*.pt; keep them, merge.py needs the running sums
os.remove = lambda p: None if os.path.basename(p).startswith("acc_") else _rm(p)
import whole_model_fit as wmf
full = list(wmf.PROMPTS)
wmf.PROMPTS = full[K:]
print(f"[resume] skipping {K} done prompts; running {len(wmf.PROMPTS)} of {len(full)} into {OUT}", flush=True)
sys.argv = ["whole_model_fit.py", "--model-dir", "$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b",
            "--model-name", "Swift-1.5-Qwen3.8-27b", "--out-dir", OUT,
            "--layers", ",".join(map(str, range(16, 63))), "--dim-batch", "2", "--max-seq-len", "150"]
wmf.main()
