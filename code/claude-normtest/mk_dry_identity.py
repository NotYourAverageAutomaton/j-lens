# Dry run 3 (Deron P090, recommended): every J_p = I (identity). J_6 sits far below logit in-band, so J_22 ~ 0.14 J_6 + 0.86 I must
# move toward logit and the gate MUST say CONTINUE. Reads NO real Stage 1 file; n_valid invented.
import json, os, hashlib, torch
OUT = "$JLENS_HOME/claude-normtest/dry-stage1-identity"; SAVE_L = [20, 24, 28, 32, 36, 40, 44, 48, 52, 56]
os.makedirs(OUT, exist_ok=True); md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
n = [230 + 3 * i for i in range(16)]; I = torch.eye(5120)
ip = os.path.join(OUT, "_identity.pt"); torch.save(I, ip); im = md5(ip)
for i in range(16):
    d = os.path.join(OUT, f"doc_{i+1:02d}"); os.makedirs(d, exist_ok=True)
    for l in SAVE_L:
        p = os.path.join(d, f"J_p_{l}.pt")
        if not os.path.lexists(p): os.symlink(ip, p)
    json.dump({"doc": i + 1, "genre": f"synthetic-identity-{i+1}", "n_valid": n[i], "ts": "dry", "J_p_md5": {str(l): im for l in SAVE_L}}, open(os.path.join(d, "meta.json"), "w"))
for l in SAVE_L: torch.save((I * sum(n), sum(n)), os.path.join(OUT, f"acc_{l}.pt"))
print("dry identity dir ready", OUT, "sum n", sum(n))
