# J-Lens #4 picture 1, NYAA look (v2, 11-06). Every string identical to the version Deron passed (post P081 list, post P082 PASS); only styling changed.
import sys, os; sys.path.insert(0, os.path.expanduser("../tools"))
from nyaa_look import *
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
fig, ax = canvas(glows=((0.10, 0.92, CYAN, 0.28), (0.85, 0.30, MAGENTA, 0.30), (0.45, 0.55, PURPLE, 0.12)))
glow_text(ax, 0.75, 8.1, "The bug: our answer key could read ahead", size=38, family=HEAD, weight="black", glow=CYAN)
glow_text(ax, 0.78, 7.45, "A model like this reads one word piece at a time. It is never allowed to see what comes next.", size=17, color=DIM)
words = ["The", "cat", "sat", "on", "the", "mat"]; NOW = 2
x0, w, gap, th = 4.3, 1.4, 0.16, 0.95
rows = [(5.55, "The real model", CYAN, False, "sees only what\ncame before"),
        (2.75, "Our old answer key", MAGENTA, True, "could see the words\nstill to come")]
xn = x0 + NOW*(w+gap) + w/2
# reading beam (cyan light column through both rows)
for k, a in enumerate([0.05, 0.07, 0.10]):
    ax.add_patch(Rectangle((xn - 0.95 + 0.25*k, 2.5), 1.9 - 0.5*k, 4.45, fc=CYAN, alpha=a, ec="none", zorder=1))
glow_text(ax, xn, 7.0, "reading this word", size=14, color=CYAN, family=MONO, ha="center", glow=CYAN)
for y, label, col, leak, note in rows:
    kicker(ax, 0.8, y + th/2, label, color=col)
    for i, wd in enumerate(words):
        x = x0 + i*(w+gap)
        if i < NOW:   glass(ax, x, y, w, th, edge="#9aa6c8", fill_alpha=0.08); c, s = WHITE, wd
        elif i == NOW: glass(ax, x, y, w, th, edge=CYAN, fill_alpha=0.16, glow=1.0, lw=2.2); c, s = WHITE, wd
        elif leak:    glass(ax, x, y, w, th, edge=MAGENTA, fill_alpha=0.12, glow=0.8, lw=1.8); c, s = MAGENTA, wd
        else:         glass(ax, x, y, w, th, edge=FAINT, fill_alpha=0.03, dashed=True); c, s = FAINT, "?"
        glow_text(ax, x + w/2, y + th/2, s, color=c, size=24, family=MONO, weight="bold" if i == NOW else "normal",
                  ha="center", glow=(MAGENTA if (leak and i > NOW) else (CYAN if i == NOW else None)))
    glow_text(ax, x0 + 6*(w+gap) + 0.2, y + th/2, note, size=15, color=col, glow=col)
    if leak:  # future words flowing BACK into the word being read
        for i in (3, 4, 5):
            xs = x0 + i*(w+gap) + w/2
            ax.add_patch(FancyArrowPatch((xs, y + th + 0.08), (xn + 0.2, y + th + 0.08), connectionstyle=f"arc3,rad={0.35 + 0.08*(i-3)}",
                         arrowstyle="-|>", mutation_scale=16, color=MAGENTA, lw=1.8, alpha=0.85, zorder=4))
        glow_text(ax, x0 + 4*(w+gap) + w/2, y - 0.42, "read ahead (the bug)", size=15, color=MAGENTA, family=MONO, weight="bold", ha="center", glow=MAGENTA)
# takeaway, gold-edged glass
glass(ax, 0.75, 0.85, 14.5, 1.15, edge=GOLD, fill_alpha=0.06, glow=0.5)
glow_text(ax, 1.1, 1.65, "Two earlier checks both ran on the read-ahead copy. Checks that share a flaw agree with each other.", size=16, color=WHITE)
glow_text(ax, 1.1, 1.15, "That's not proof. We've built a fixed answer key, and the re-check runs next.", size=16, color=GOLD, weight="bold", glow=GOLD)
brand(ax, "J-Lens", 4)
print(save(fig, os.path.expanduser("./01-the-answer-key-could-read-ahead.png")))
