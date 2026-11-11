# J-Lens #6 picture 1, NYAA look. Data: jlens-dose-points.csv f02e4a6b (Lens; every point PASS Deron P144, from dose_kl.json 6be964a0).
# Deron's label rules (post P143): source md5 printed; y = KL to the model's own final prediction, lower is better; the simplest
# method is a flat reference line; two fits labelled apart, the post-hoc one marked and its floor drawn ABOVE the flat line;
# nothing says more documents will or won't close the gap; "Stage 2 pending" marks the empty spot, with no point drawn.
import sys, os, csv, hashlib, math
sys.path.insert(0, os.path.expanduser("../tools"))
from nyaa_look import *
BLUE = "#4db8ff"; ORANGE = "#ff9d3c"; GREY = "#b9c0d4"   # the post text says blue dots, orange squares, grey line: the picture must match the words
HERE = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(HERE, "jlens-dose-points.csv")
assert hashlib.md5(open(P, "rb").read()).hexdigest().startswith("f02e4a6b"), "wrong points file"
raw, shr, flat, c = [], [], None, {}
for r in csv.reader(open(P)):
    if r[0] == "raw": raw.append((int(r[1]), float(r[2]), float(r[3]) if r[3] else None))
    elif r[0] == "shrunk": shr.append((int(r[1]), float(r[2]), float(r[3]) if r[3] else None))
    elif r[0] == "logit_lens_flat": flat = float(r[2])
    elif r[0].startswith(("registered", "post_hoc")):
        for cell in r[2:]:
            k, v = cell.split("="); c[k] = float(v)
assert len(raw) == 17 and len(shr) == 17 and flat and set(c) == {"c", "s", "a", "b"}
fig, ax0 = canvas(glows=((0.14, 0.84, CYAN, 0.22), (0.9, 0.2, PURPLE, 0.22), (0.5, 0.12, GOLD, 0.08)), seed=61)
glow_text(ax0, 0.75, 8.15, "Does our map get better with more documents?", size=34, family=HEAD, weight="black", glow=CYAN)
glow_text(ax0, 0.78, 7.5, "How far each method's prediction is from the model's own final answer, in its middle layers. Lower is better.", size=16, color=DIM)
ax = fig.add_axes([0.075, 0.335, 0.715, 0.405]); ax.set_facecolor("none")
ax.axvspan(22, 58, color=WHITE, alpha=0.035); ax.text(40, 10.72, "no measurement past 22 documents yet", color=DIM, fontsize=12.5, family=MONO, ha="center")
ax.axhline(flat, color=GREY, lw=2.2, ls=(0, (6, 4)), alpha=0.95)
xs = [22 + i * 0.5 for i in range(73)]
ax.plot(xs, [c["c"] + c["s"] * math.log(x) for x in xs], color=BLUE, lw=1.8, ls=(0, (1, 3)), alpha=0.95)
ax.plot(xs, [c["a"] + c["b"] / x for x in xs], color=MAGENTA, lw=1.6, ls=(0, (1, 3)), alpha=0.9)
xf = [6 + i * 0.25 for i in range(65)]
ax.plot(xf, [c["c"] + c["s"] * math.log(x) for x in xf], color=BLUE, lw=1.6, alpha=0.55)
ax.plot(xf, [c["a"] + c["b"] / x for x in xf], color=MAGENTA, lw=1.6, alpha=0.55, ls=(0, (5, 3)))
ax.plot([55.2, 57.6], [c["a"], c["a"]], color=MAGENTA, lw=2, alpha=0.9)
for pts, col, mk in ((raw, BLUE, "o"), (shr, ORANGE, "s")):
    for n, v, sd in pts:
        if sd is not None: ax.plot([n, n], [v - sd, v + sd], color=col, lw=1.6, alpha=0.55)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], ls="none", marker=mk, ms=8, mfc=INK, mec=col, mew=2.2)
ax.axvline(54, color=WHITE, lw=1.2, alpha=0.5); ax.text(53.2, 10.28, "Stage 2\npending", color=WHITE, fontsize=12.5, family=MONO, ha="right", va="top", alpha=0.9)
ax.set_xlim(4.5, 58); ax.set_ylim(5.9, 11.0)
ax.set_xticks([6, 10, 14, 18, 22, 30, 40, 54]); ax.set_xticklabels(["6", "10", "14", "18", "22", "30", "40", "54"], fontsize=13, color=DIM, family=MONO)
ax.set_yticks([6, 7, 8, 9, 10]); ax.set_yticklabels(["6", "7", "8", "9", "10"], fontsize=13, color=DIM, family=MONO)
ax.tick_params(length=0, pad=8); ax.grid(axis="y", color=WHITE, alpha=0.07); [s.set_visible(False) for s in ax.spines.values()]
ax.set_xlabel("documents the map is built from", color=DIM, fontsize=13, family=MONO, labelpad=10)
ax.set_ylabel("distance from the model's own answer (KL, nats)", color=DIM, fontsize=12, family=MONO, labelpad=10)
T = ax.transData
glow_text(ax, 6.6, 10.62, "our original map (blue dots)", color=BLUE, size=15, weight="bold", family=HEAD, glow=BLUE, transform=T)
glow_text(ax, 6.2, 6.25, "the blend: mostly the simplest method, a small dose of our map (orange squares)", color=ORANGE, size=13.5, weight="bold", glow=ORANGE, transform=T)
glow_text(ax, 6.2, flat + 0.2, "the simplest method (grey line): flat, it uses no documents", color=GREY, size=13, weight="bold", glow=GREY, transform=T)
lx = 0.80
fig.text(lx, 0.70, "TWO CURVES THROUGH\nTHE BLUE DOTS", color=WHITE, fontsize=12.5, family=MONO, weight="bold", va="top")
fig.text(lx, 0.615, "dotted blue: the curve we\nregistered before the run.\nIt would cross the grey line\nnear 40 documents.", color=BLUE, fontsize=12, family=BODY, va="top")
fig.text(lx, 0.475, "dotted pink: a curve we\nfitted after seeing the data.\nIt levels off at 7.35, above\nthe grey line.", color=MAGENTA, fontsize=12, family=BODY, va="top")
fig.text(lx, 0.335, "Dotted lines are not data.", color=DIM, fontsize=12, family=BODY, va="top")
glass(ax0, 0.75, 0.82, 14.5, 0.98, edge=GOLD, fill_alpha=0.05, r=0.12)
glow_text(ax0, 0.98, 1.50, "Measured on our six check passages, average over layers 20 to 44. Bars: 1 SD across document subsets (none at 6 and 22).", size=11.5, color=WHITE)
glow_text(ax0, 0.98, 1.10, "The blend's strength was chosen on these same six passages; its never-seen result is on fresh text (not shown). Source: dose_kl.json 6be964a0.", size=11.5, color=WHITE)
brand(ax0, "J-Lens", 6)
print(save(fig, os.path.join(HERE, "01-more-documents.png")))
