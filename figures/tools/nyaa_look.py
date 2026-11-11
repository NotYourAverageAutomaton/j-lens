"""NYAA look for post pictures: black glass + holo-cyan / purple / magenta, touches of gold (Niz's brand, steward-profile).
Use: from nyaa_look import *; fig, ax = canvas(); ... glass(ax, ...); glow_text(ax, ...); save(fig, path)
Every post should vary the COMPOSITION, not the look: the look is the brand."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib import patheffects as pe

INK = "#05060a"; CYAN = "#3ff0ff"; PURPLE = "#a46bff"; MAGENTA = "#ff4fd8"; GOLD = "#f5c85b"
WHITE = "#eef3ff"; DIM = "#8d95b3"; FAINT = "#3a4060"
HEAD = "Inter Display"; BODY = "Inter"; MONO = "JetBrains Mono"
W, H = 16, 9

def canvas(glows=((0.12, 0.85, CYAN, 0.30), (0.92, 0.12, MAGENTA, 0.26), (0.62, 0.55, PURPLE, 0.14)), grid=True, seed=1):
    fig = plt.figure(figsize=(W, H), dpi=120, facecolor=INK)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ny, nx = 450, 800; yy, xx = np.mgrid[0:1:ny*1j, 0:1:nx*1j]
    img = np.zeros((ny, nx, 3)) + np.array(matplotlib.colors.to_rgb(INK))
    for gx, gy, col, a in glows:  # soft holo nebulae
        d = np.exp(-(((xx-gx)*1.6)**2 + (yy-gy)**2) / 0.09) * a
        img += d[..., None] * np.array(matplotlib.colors.to_rgb(col))
    rng = np.random.default_rng(seed); img += rng.normal(0, 0.008, img.shape)  # film grain
    ax.imshow(np.clip(img, 0, 1), extent=(0, W, 0, H), origin="lower", zorder=0, interpolation="bicubic")
    if grid:
        for x in np.arange(0, W, 0.5): ax.plot([x, x], [0, H], color=WHITE, alpha=0.025, lw=0.6, zorder=0.5)
        for y in np.arange(0, H, 0.5): ax.plot([0, W], [y, y], color=WHITE, alpha=0.025, lw=0.6, zorder=0.5)
    return fig, ax

def glass(ax, x, y, w, h, edge=CYAN, fill_alpha=0.07, r=0.18, glow=0.0, lw=1.4, z=2, dashed=False):
    """A frosted black-glass panel. glow>0 adds a holo halo in the edge colour."""
    for k in range(int(glow * 6)):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={0.02+0.035*k},rounding_size={r}",
                                    fc="none", ec=edge, lw=2 + 2*k, alpha=0.10 * glow / (1 + k), zorder=z-0.2))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}", fc="#ffffff", alpha=fill_alpha, ec="none", zorder=z))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}", fc="none", ec=edge, lw=lw,
                                alpha=0.9, zorder=z+0.1, ls=(0, (3, 3)) if dashed else "-"))
    ax.plot([x + r, x + w - r], [y + h - 0.03, y + h - 0.03], color="#ffffff", alpha=0.18, lw=1, zorder=z+0.2)  # top sheen

def glow_text(ax, x, y, s, color=WHITE, size=20, glow=None, family=BODY, weight="normal", ha="left", va="center", z=5, **kw):
    t = ax.text(x, y, s, color=color, fontsize=size, family=family, weight=weight, ha=ha, va=va, zorder=z, **kw)
    if glow: t.set_path_effects([pe.withStroke(linewidth=14, foreground=glow, alpha=0.045), pe.withStroke(linewidth=7, foreground=glow, alpha=0.07), pe.withStroke(linewidth=3, foreground=glow, alpha=0.10), pe.Normal()])
    return t

def kicker(ax, x, y, s, color=CYAN):  # small mono label like an instrument readout
    return glow_text(ax, x, y, s.upper(), color=color, size=13, family=MONO, weight="bold", glow=color)

def brand(ax, series, n):
    glow_text(ax, 0.75, 0.42, f"NYAA LAB  ·  {series.upper()}  #{n}", color=DIM, size=12, family=MONO)
    ax.plot([0.75, 2.0], [0.72, 0.72], color=GOLD, lw=2.2, zorder=5)

def save(fig, path):
    import os; os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, facecolor=INK, dpi=120); plt.close(fig); return path
