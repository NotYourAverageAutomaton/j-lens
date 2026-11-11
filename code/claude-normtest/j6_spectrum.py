# Report-only (Deron P091): rho = ||J6 - I|| / ||J6|| and the singular spectrum of (J6 - I) at L24/40/52, RECORD maps only (not sealed).
# Our control variate is the identity (logit lens). Effective rank = (sum s)^2 / sum s^2 (participation) and k90/k99 energy ranks.
import hashlib, json, os, torch
torch.set_num_threads(8)
JD = "$JLENS_HOME/results-whole-swift-record"; out = {}
for l in (24, 40, 52):
    p = os.path.join(JD, f"J_l_{l}.pt"); J = torch.load(p).float(); D = J - torch.eye(J.shape[0])
    s = torch.linalg.svdvals(D); e = (s ** 2).cumsum(0) / (s ** 2).sum()
    out[l] = {"md5": hashlib.md5(open(p, "rb").read()).hexdigest()[:8], "rho_fro": float(D.norm() / J.norm()),
              "eff_rank_participation": float(s.sum() ** 2 / (s ** 2).sum()), "k90": int((e < .90).sum()) + 1, "k99": int((e < .99).sum()) + 1,
              "top8_sv": [round(float(x), 3) for x in s[:8]], "sv_at_64_128_256_512_1024": [round(float(s[i]), 4) for i in (63, 127, 255, 511, 1023)]}
    print(l, json.dumps(out[l]), flush=True)
json.dump(out, open("$JLENS_HOME/claude-normtest/j6_spectrum.json", "w"), indent=1)
