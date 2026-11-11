# J-Lens #5 picture 1, NYAA look. Data: score_stage1.json 67449876 (b7f9ac27, Deron GRADED CONTINUE post P093), per-layer fut20 table (report-only).
import sys, os; sys.path.insert(0, os.path.expanduser("../tools"))
from nyaa_look import *
import numpy as np
L = [20, 24, 28, 32, 36, 40, 44, 48, 52, 56]
J22 = [.0365, .0417, .0527, .0610, .0628, .0683, .0868, .1108, .1270, .1442]
J6 = [.0072, .0129, .0178, .0245, .0504, .0719, .0726, .0916, .1294, .1539]
LOG = [.0988, .1199, .1328, .1227, .0774, .0993, .1145, .0984, .0855, .1177]
fig, ax0 = canvas(glows=((0.18, 0.80, CYAN, 0.24), (0.85, 0.25, PURPLE, 0.22), (0.55, 0.15, GOLD, 0.10)), seed=21)
glow_text(ax0, 0.75, 8.15, "The bigger map reads the middle layers better", size=34, family=HEAD, weight="black", glow=CYAN)
glow_text(ax0, 0.78, 7.5, "How many of the next 10 word-pieces show up in each method's top-20 guesses, layer by layer, on text it never saw:", size=16, color=DIM)
ax = fig.add_axes([0.085, 0.27, 0.70, 0.46]); ax.set_facecolor("none")
ax.axvspan(19, 45, color=CYAN, alpha=0.05); ax.text(32, 0.158, "the middle layers we tested", color=CYAN, fontsize=13, family=MONO, ha="center", alpha=0.9)
for ys, c, ls, lw in ((LOG, GOLD, (0, (6, 4)), 2.2), (J6, PURPLE, "-", 2.2), (J22, CYAN, "-", 3.0)):
    for k, a in ((10, .05), (5, .10)): ax.plot(L, ys, color=c, lw=k, alpha=a, ls=ls)
    ax.plot(L, ys, color=c, lw=lw, ls=ls, marker="o", ms=6, mfc=INK, mec=c, mew=2)
ax.set_xlim(18, 58); ax.set_ylim(0, 0.17); ax.set_xticks(L); ax.set_xticklabels([str(l) for l in L], fontsize=13, color=DIM, family=MONO)
ax.set_yticks([0, .05, .10, .15]); ax.set_yticklabels(["0%", "5%", "10%", "15%"], fontsize=13, color=DIM, family=MONO)
ax.tick_params(length=0, pad=8); ax.grid(axis="y", color=WHITE, alpha=0.07); [s.set_visible(False) for s in ax.spines.values()]
ax.set_xlabel("layer (deeper ->)", color=DIM, fontsize=13, family=MONO, labelpad=10)
glow_text(ax, 57.2, J22[-1], "22-document map", color=CYAN, size=16, weight="bold", family=HEAD, glow=CYAN, transform=ax.transData)
glow_text(ax, 57.2, J6[-1] + 0.010, "6-document map", color=PURPLE, size=14, weight="bold", glow=PURPLE, transform=ax.transData)
glow_text(ax, 57.2, LOG[-1] - 0.012, "simple read-out", color=GOLD, size=14, weight="bold", glow=GOLD, transform=ax.transData)
glass(ax0, 0.75, 0.68, 9.6, 0.62, edge=GOLD, fill_alpha=0.05, r=0.12)
glow_text(ax0, 0.95, 0.99, "Pass mark, set before we looked: in the middle layers the 22-document map had to beat the 6-document map. It did.", size=11.5, color=WHITE)
brand(ax0, "J-Lens", 5)
print(save(fig, os.path.expanduser("./01-the-bigger-map.png")))
