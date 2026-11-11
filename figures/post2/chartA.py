import json
from PIL import Image, ImageDraw, ImageFont
d = json.load(open("eval_record.json"))["layers"]
Ls = sorted(int(k) for k in d)
J = {l: d[str(l)]["jlens"]["fut20"] * 10 for l in Ls}; B = {l: d[str(l)]["logit"]["fut20"] * 10 for l in Ls}
W, H = 1600, 900; BG = (250, 248, 243); INK = (28, 26, 23); MUTED = (110, 104, 95); GRID = (226, 221, 211)
NEW = (196, 74, 30); OLD = (60, 92, 130); BAND = (232, 240, 228)
F = lambda s, b=False: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if b else ""), s)
im = Image.new("RGB", (W, H), BG); dr = ImageDraw.Draw(im)
x0, x1, y0, y1 = 150, 1530, 760, 230          # plot box: x = layer 14..64, y = 0..2.5 words
lx = lambda l: x0 + (l - 14) / (64 - 14) * (x1 - x0); vy = lambda v: y0 - v / 3.0 * (y0 - y1)
dr.rectangle([lx(20), y1, lx(44), y0], fill=BAND)
dr.text((lx(32), y1 + 14), "where our pass mark needed the lead", font=F(26), fill=(70, 110, 70), anchor="mt")
for v in (0, 0.5, 1, 1.5, 2, 2.5, 3):
    dr.line([x0, vy(v), x1, vy(v)], fill=GRID, width=2); dr.text((x0 - 16, vy(v)), f"{v:g}", font=F(26), fill=MUTED, anchor="rm")
for l in (16, 24, 32, 40, 48, 56, 64):
    dr.text((lx(l), y0 + 18), str(l), font=F(26), fill=MUTED, anchor="mt")
dr.text(((x0 + x1) / 2, y0 + 62), "layer (1 = first, 64 = last)", font=F(26), fill=MUTED, anchor="mt")
def line(S, col):
    pts = [(lx(l), vy(S[l])) for l in Ls]; dr.line(pts, fill=col, width=7, joint="curve")
    for p in pts: dr.ellipse([p[0] - 8, p[1] - 8, p[0] + 8, p[1] + 8], fill=col)
line(B, OLD); line(J, NEW)
p = (lx(52), vy(J[52])); dr.ellipse([p[0] - 16, p[1] - 16, p[0] + 16, p[1] + 16], outline=NEW, width=5)
dr.text((p[0] - 30, p[1] - 60), "first win: layer 52", font=F(28, True), fill=NEW, anchor="rm")
dr.text((60, 50), "The new method only pulled ahead late, not in the middle", font=F(42, True), fill=INK)
dr.text((60, 118), "Of the next 10 tokens (word pieces) the model actually wrote, how many were in each method's top 20 guesses", font=F(24), fill=MUTED)
dr.text((x1, 168), "new method (J-lens)", font=F(30, True), fill=NEW, anchor="rt")
dr.text((x1 - 380, 168), "basic method", font=F(30, True), fill=OLD, anchor="rt")
dr.text((60, H - 46), "Fit on 6 documents, scored on 6 it never saw. A pass needed its biggest lead to land in the green band. Data: eval_record e02c34da.", font=F(21), fill=MUTED)
im.save("jlens-chartA.png"); print("ok")
