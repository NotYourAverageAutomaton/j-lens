"""Step-2 passage selection, rule fixed BEFORE reading any passage (Claude 2026-11-07; Deron P118: non-famous middle sections).
For each book: strip the Gutenberg header/footer at the *** START / *** END markers; jump to 55% of the body; take the first
paragraph start (after a blank line) that opens a prose paragraph >= 200 chars with >= 60% letters; take 2000 chars from there
(the capture truncates to 256 tokens). Gutenberg markup stripped (Deron P119). Output passages.json with source URL, ebook id, offset, sha256 of the exact text."""
import hashlib, json, re, sys, os
BOOKS = [("travel-guidebook", 73849, "Baedeker, The Mediterranean: Seaports and Sea Routes (1911)"),
         ("personal-letters", 34123, "Familiar Letters of John Adams and His Wife Abigail Adams During the Revolution"),
         ("maritime-memoir", 2055, "R. H. Dana Jr., Two Years Before the Mast (1840)"),
         ("economics-treatise", 3300, "Adam Smith, The Wealth of Nations (1776)"),
         ("music-theory", 19354, "Percy Goetschius, Lessons in Music Form (1904) [substitute: Prout's Harmony not on Gutenberg]"),
         ("chess-instruction", 33870, "J. R. Capablanca, Chess Fundamentals (1921)")]
d = sys.argv[1]; out = []
for genre, eid, title in BOOKS:
    raw = open(os.path.join(d, f"books/pg{eid}.txt"), encoding="utf-8", errors="replace").read().replace("\r\n", "\n")
    s = raw.index("*** START OF"); s = raw.index("\n", s) + 1; e = raw.index("*** END OF")
    body = raw[s:e]; pos = int(len(body) * 0.55)
    for m in re.finditer(r"\n\s*\n", body[pos:]):
        st = pos + m.end(); para = body[st:].split("\n\n", 1)[0]
        letters = sum(c.isalpha() for c in para)
        if len(para) >= 200 and letters / max(1, len(para)) >= 0.6:
            text = re.sub(r"[ \t]*\n[ \t]*", "\n", body[st:st + 2000]).strip()
            # Deron P119: strip Gutenberg transcription markup mechanically (not text a reader sees). _x_ and =x= lose the
            # marks only when the marked span has no edge spaces, <= 80 chars, may wrap a line (a lone "=" in prose survives); {digits} page markers go.
            text = re.sub(r"_(\S(?:[^_]{0,80}?\S)?)_", r"\1", text)   # a marked span may wrap one line (e.g. _Jebel\nel-Herrech_), max 80 chars
            text = re.sub(r"=(\S(?:[^=]{0,80}?\S)?)=", r"\1", text)
            text = re.sub(r"\{\d+\}", "", text)
            break
    out.append({"genre": genre, "ebook": eid, "title": title, "url": f"https://www.gutenberg.org/ebooks/{eid}",
                "body_offset": st, "body_len": len(body), "frac": round(st / len(body), 4),
                "sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text})
json.dump(out, open(os.path.join(d, "passages.json"), "w"), indent=1)
for p in out: print(p["genre"], p["ebook"], "frac", p["frac"], "sha", p["sha256"][:12], "|", p["text"][:110].replace("\n", " "))
