import json, math
from PIL import Image, ImageDraw, ImageFont
r = json.load(open("pilot_hutch.json"))["rounds"]
W, H = 1600, 900; BG = (250, 248, 243); INK = (28, 26, 23); MUTED = (110, 104, 95); GRID = (226, 221, 211); BAR = (60, 120, 70)
COL = {"24": (196, 74, 30), "40": (150, 110, 40), "52": (60, 92, 130)}
F = lambda s, b=False: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if b else ""), s)
im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
x0, x1, y0, y1 = 150, 1300, 760, 220
Ks = [p["K"] for p in r]; lo, hi = math.log2(Ks[0]), math.log2(16384)
lx = lambda k: x0 + (math.log2(k) - lo) / (hi - lo) * (x1 - x0); vy = lambda v: y0 - v * (y0 - y1)
for v in (0, .2, .4, .6, .8, 1.0):
    d.line([x0, vy(v), x1, vy(v)], fill=GRID, width=2); d.text((x0 - 16, vy(v)), f"{int(round(v * 100))}%", font=F(26), fill=MUTED, anchor="rm")
for k in (96, 384, 1536, 6144, 15996):
    d.text((lx(k), y0 + 18), f"{k:,}", font=F(24), fill=MUTED, anchor="mt")
d.text(((x0 + x1) / 2, y0 + 60), "random probes used (log scale; our budget was 16,000)", font=F(26), fill=MUTED, anchor="mt")
d.line([x0, vy(.9), x1, vy(.9)], fill=BAR, width=5); d.text((x0 + 10, vy(.9) - 8), "pass mark: 90%", font=F(28, True), fill=BAR, anchor="lb")
for L, c in COL.items():
    pts = [(lx(p["K"]), vy(p["layers"][L]["top1"])) for p in r]
    d.line(pts, fill=c, width=7, joint="curve")
    for x, y in pts: d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=c)
    v = r[-1]["layers"][L]["top1"]; d.text((x1 + 22, vy(v)), f"layer {L}: {round(v * 100)}%", font=F(28, True), fill=c, anchor="lm")
d.text((60, 50), "The shortcut got closer, but not close enough", font=F(42, True), fill=INK)
d.text((60, 116), "How often the shortcut's top guess matched the exact map's, on text it had never seen", font=F(28), fill=MUTED)
d.text((60, H - 46), "A pass needed 90% at all three layers (and 95% for its top 10) within budget. Data: pilot_hutch.json (11-04).", font=F(24), fill=MUTED)
im.save("jlens-chartC.png"); print("ok")
