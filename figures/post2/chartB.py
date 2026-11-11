from PIL import Image, ImageDraw, ImageFont
W, H = 1600, 900; BG = (250, 248, 243); INK = (28, 26, 23); MUTED = (110, 104, 95); NEW = (196, 74, 30); OLD = (60, 92, 130)
F = lambda s, b=False: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if b else ""), s)
im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
d.text((60, 50), "Hypothesis, under test: we fit it on far fewer examples", font=F(42, True), fill=INK)
d.text((60, 116), "Each dot is one document the map was built from. Not proven; our next run tests it.", font=F(28), fill=MUTED)
# left: ours, 6 big dots
d.text((260, 230), "Ours", font=F(36, True), fill=NEW, anchor="mt"); d.text((260, 280), "6 documents", font=F(30), fill=MUTED, anchor="mt")
for i in range(6):
    cx = 233 + (i % 3) * 18; cy = 349 + (i // 3) * 18; d.ellipse([cx - 6.5, cy - 6.5, cx + 6.5, cy + 6.5], fill=NEW)
# right: the paper, 1000 small dots in a 40 x 25 grid
gx, gy, step, r = 660, 340, 18, 6.5
d.text((gx + 20 * step, 230), "The paper", font=F(36, True), fill=OLD, anchor="mt"); d.text((gx + 20 * step, 280), "1,000 prompts (paper, section 2.1)", font=F(30), fill=MUTED, anchor="mt")
for i in range(1000):
    cx = gx + (i % 40) * step + step / 2; cy = gy + (i // 40) * step + step / 2; d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=OLD)
d.text((60, H - 46), "Dots drawn at the same size on both sides: ours used 6 documents, the paper 1,000 prompts: about 170 times fewer.", font=F(24), fill=MUTED)
im.save("jlens-chartB.png"); print("ok")
