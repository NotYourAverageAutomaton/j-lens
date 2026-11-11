import importlib.util, sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU

# 1. EXACT import block from whole_model_fit.py (post-P015 fix)
spec = importlib.util.spec_from_file_location("our_jlens", "$JLENS_HOME/our_jlens.py")
oj = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oj          # the post P015 fix
spec.loader.exec_module(oj)
has_lens = hasattr(oj, "OurJacobianLens")
print("[import] our_jlens OK | OurJacobianLens defined=" + str(has_lens))

# 2. Prompt build: tokenize a sample prompt on CPU (the input the lens consumes)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("$MODELS_DIR/bf16/Swift-1.5-Qwen3.8-27b")
prompt = "I am trying to figure out why the bridge collapsed. The load was within spec, so the failure must be in the material itself."
ids = tok(prompt, return_tensors="pt")
shape = tuple(ids["input_ids"].shape)
first5 = ids["input_ids"][0,:5].tolist()
print("[prompt-build] input_ids shape=" + str(shape) + " first5=" + str(first5))
print("DRYRUN_OK")
