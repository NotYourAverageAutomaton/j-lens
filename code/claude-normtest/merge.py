"""Claude: report per-layer tot in a record-run acc dir (to find K, and any layer one prompt ahead after a
mid-save crash), or merge two acc dirs. Usage: merge.py check DIR | merge.py merge DIR_A DIR_B OUT"""
import os, sys, torch
L = range(16, 63)
if sys.argv[1] == "check":
    d = sys.argv[2]; tots = {}
    for l in L:
        p = os.path.join(d, f"acc_{l}.pt")
        tots[l] = torch.load(p, map_location="cpu")[1] if os.path.exists(p) else None
    print("tot per layer:", sorted(set(tots.values()), key=str), {l: t for l, t in tots.items() if t != tots[16]})
else:
    a, b, out = sys.argv[2:5]; os.makedirs(out, exist_ok=True)
    for l in L:
        A, ta = torch.load(os.path.join(a, f"acc_{l}.pt"), map_location="cpu")
        B, tb = torch.load(os.path.join(b, f"acc_{l}.pt"), map_location="cpu")
        torch.save((A + B, ta + tb), os.path.join(out, f"acc_{l}.pt"))
        torch.save(((A + B) / max(ta + tb, 1)).cpu(), os.path.join(out, f"J_l_{l}.pt"))
    print("merged + finalized", len(L), "layers ->", out, "(readouts NOT rerun: needs the GPU)")
