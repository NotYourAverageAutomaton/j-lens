# J-Lens #6 picture 2, NYAA look. Subtitle + panel titles are Deron's swaps, verbatim (post P146): the score compares each method's top guess
# with the MODEL'S OWN top prediction for that spot (the model read real text; it was not left to write). The ten numbers are TYPED IN
# here, not read from the json; Deron's check (post P146) is what ties them to the file.
# Data: dose_kl.json 6be964a0, pk_row p@1 "all", N=0 (306 positions) and N=1 (300 positions),
# mid-band L20-44. Same figures as Deron's companion row (post P145). Intervals vs word frequency: only the trained lens at N=0 is
# wholly above; at N=1 every lens is wholly below.
import sys, os; sys.path.insert(0, os.path.expanduser("../tools"))
from nyaa_look import *
NAMES = ["guessing\ncommon words", "simplest\nmethod", "our map", "the blend", "a map we\ntrained"]
N0 = [.0752, .0696, .0537, .0971, .2190]
N1 = [.0733, .0176, .0157, .0190, .0095]
BLUE = "#4db8ff"; ORANGE = "#ff9d3c"; GREY = "#b9c0d4"   # same colours as picture 1: our map blue, the blend orange, the simplest method grey
COL = [GOLD, GREY, BLUE, ORANGE, MAGENTA]
fig, ax0 = canvas(glows=((0.15, 0.82, MAGENTA, 0.20), (0.88, 0.22, CYAN, 0.20), (0.5, 0.1, GOLD, 0.08)), seed=6)
glow_text(ax0, 0.75, 8.15, "Can it see what the AI will say next?", size=36, family=HEAD, weight="black", glow=MAGENTA)
glow_text(ax0, 0.78, 7.48, "How often each method's top guess, read from the model's middle layers, matched the model's own top prediction:", size=15.5, color=DIM)
for k, (vals, title, sub, x0) in enumerate(((N0, "FOR THE VERY NEXT WORD PIECE", "306 spots", 0.06), (N1, "FOR ONE WORD PIECE FURTHER AHEAD", "300 spots", 0.54))):
    ax = fig.add_axes([x0, 0.30, 0.40, 0.42]); ax.set_facecolor("none")
    for i, (v, c) in enumerate(zip(vals, COL)):
        for w, a in ((0.78, .10), (0.70, .18)): ax.bar(i, v, width=w, color=c, alpha=a)
        ax.bar(i, v, width=0.60, color=c, alpha=0.92)
        ax.text(i, v + 0.008, f"{v*100:.1f}%", color=c, fontsize=17, family=MONO, weight="bold", ha="center")
    ax.axhline(vals[0], color=GOLD, lw=1.4, ls=(0, (5, 4)), alpha=0.8)
    ax.set_ylim(0, 0.26); ax.set_xlim(-0.6, 4.6); ax.set_xticks(range(5)); ax.set_xticklabels(NAMES, fontsize=12.5, color=WHITE, family=BODY)
    ax.set_yticks([]); ax.tick_params(length=0, pad=8); [s.set_visible(False) for s in ax.spines.values()]
    fig.text(x0 + 0.005, 0.755, title, color=CYAN if k == 0 else MAGENTA, fontsize=15, family=MONO, weight="bold")
    fig.text(x0 + 0.395, 0.755, sub, color=DIM, fontsize=12, family=MONO, ha="right")
glass(ax0, 0.75, 0.86, 14.5, 0.98, edge=GOLD, fill_alpha=0.05, r=0.12)
glow_text(ax0, 0.98, 1.52, "Dashed gold line = just guessing common words. Left: only the trained map clearly beats it; the blend is too close to call.", size=13, color=WHITE)
glow_text(ax0, 0.98, 1.10, "Right: one step further ahead, every method does worse than guessing common words.", size=13, color=WHITE)
brand(ax0, "J-Lens", 6)
print(save(fig, os.path.expanduser("./02-reading-ahead.png")))
