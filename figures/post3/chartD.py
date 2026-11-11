import sys
from PIL import Image, ImageDraw, ImageFont
done, tot = int(sys.argv[1]), 16
W, H = 1600, 900; BG = (250, 248, 243); INK = (28, 26, 23); MUTED = (110, 104, 95); GRID = (226, 221, 211); GOOD = (60, 120, 70); BLUE = (60, 92, 130)
F = lambda s, b=False: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if b else ""), s)
im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
d.text((60, 75), "The open check passed: the tool's math holds", font=F(46, True), fill=INK, anchor="lm")
d.text((60, 133), "We checked the map's own setup against a brute-force check (nudge a layer, measure the change at the output)", font=F(26), fill=MUTED, anchor="lm")
y = 225
for name in ["layers 20 to 28", "layers 28 to 36", "layers 36 to 40", "layers 40, 44 and 52, to the output"]:
    d.text((60, y), name, font=F(32), fill=INK, anchor="lm"); d.text((1300, y), "PASS", font=F(32, True), fill=GOOD, anchor="rm")
    d.line([60, y + 36, 1300, y + 36], fill=GRID, width=2); y += 74
d.text((60, y + 14), "Each pass mark was written before its run. So the 6-document map's failure is real, not a tool bug.", font=F(26), fill=INK, anchor="lm")
yb = y + 110
d.text((60, yb), "Running now: the 16-document build", font=F(32, True), fill=INK, anchor="lm")
d.rectangle([60, yb + 36, 1300, yb + 82], fill=GRID); d.rectangle([60, yb + 36, 60 + int(1240 * done / tot), yb + 82], fill=BLUE)
d.text((1320, yb + 59), f"{done} of {tot} docs", font=F(28, True), fill=BLUE, anchor="lm")
d.text((60, yb + 128), "About 80 minutes per document on one GPU. When it finishes: the same test, the same pass marks.", font=F(26), fill=MUTED, anchor="lm")
d.text((60, 862), "Checks: fd_gate v4, v4c and v5 (11-04), ruled by our fact-checker. Build progress: stage1_run.log.", font=F(24), fill=MUTED, anchor="lm")
im.save(sys.argv[2])
