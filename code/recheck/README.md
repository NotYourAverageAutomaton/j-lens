# Re-check scripts (posts #4 and #5)

These six scripts diagnosed the missing causal mask in the reference copy of the model (`diag_layers.py`, `diag_mask.py`, `diag_mask2.py`) and re-ran the checks against the fixed reference (`c4_check.py`, `c4_check_v2.py`, `part_c.py`). They were kept outside the experiment directory, so they sit in their own folder here. Paths were replaced with the same placeholders as the rest of `code/` (see `SCRUB.md`). Their logs are under `results/`.

| file | md5 of the original |
|---|---|
| c4_check.py | d6d618da |
| c4_check_v2.py | 061f2897 |
| diag_layers.py | adbb7c75 |
| diag_mask.py | 121e7b64 |
| diag_mask2.py | 7d91059e |
| part_c.py | 8571870e |
