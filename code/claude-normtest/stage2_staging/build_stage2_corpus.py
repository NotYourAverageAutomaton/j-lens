"""J-Lens STAGE 2 corpus builder (Claude 2026-11-09; Niz GPU go; Deron bar post P137, plan post P140). 32 NEW docs = 2 more per Stage-1 genre.
Rule fixed BEFORE reading any text, the same as step-2 select.py (9b97ba08) except the length: strip source boilerplate; jump to 55% of
the body; take the first paragraph start (after a blank line) that opens a paragraph >= 200 chars with >= MINLET letters (0.6 prose,
0.4 for source-code); take 3000 chars from there (Stage-1 texts are 1.4-3k; the fit truncates to 256 tokens). Gutenberg markup stripped
exactly as select.py. Each source is asserted by title text, so a wrong ebook id is a STOP, not a silent substitute.
Usage: python3 -I build_stage2_corpus.py <outdir> [spares]   -> <outdir>/raw/* and <outdir>/stage2_corpus.json (or stage2_spares.json)"""
import hashlib, json, os, re, sys, time, urllib.parse, urllib.request
OUT = sys.argv[1]; RAW = os.path.join(OUT, "raw"); os.makedirs(RAW, exist_ok=True)
G = "gutenberg"; W = "wikipedia"; U = "url"
SOURCES = [  # (genre, kind, id/title/url, must-contain, label)   two per Stage-1 genre, none a Stage-1, held, record or fresh work
    ("novel", G, 145, "Middlemarch", "Middlemarch (Eliot)"),
    ("novel", G, 2701, "Moby", "Moby-Dick (Melville)"),
    ("poetry", G, 26, "Paradise Lost", "Paradise Lost (Milton)"),
    ("poetry", G, 12242, "Dickinson", "Poems, Series One (Dickinson)"),
    ("drama", G, 2542, "Doll", "A Doll's House (Ibsen)"),
    ("drama", G, 844, "Importance of Being Earnest", "The Importance of Being Earnest (Wilde)"),
    ("scripture", G, 2800, "Koran", "The Koran (Rodwell tr.)"),
    ("scripture", G, 2388, "Bhagavad", "The Bhagavad-Gita (Arnold tr.)"),
    ("philosophy", G, 1497, "Republic", "The Republic (Plato, Jowett tr.)"),
    ("philosophy", G, 4363, "Beyond Good and Evil", "Beyond Good and Evil (Nietzsche)"),
    ("cookbook", G, 10136, "Household Management", "Book of Household Management (Beeton)"),
    ("cookbook", G, 12519, "Virginia Housewife", "The Virginia Housewife (Randolph)"),
    ("legal", G, 17150, "Hammurabi", "The Code of Hammurabi (King tr.)"),
    ("legal", G, 5983, "Justinian", "The Institutes of Justinian"),
    ("political-essay", G, 34901, "On Liberty", "On Liberty (Mill)"),
    ("political-essay", G, 147, "Common Sense", "Common Sense (Paine)"),
    ("science-19c", G, 14474, "Candle", "The Chemical History of a Candle (Faraday)"),
    ("science-19c", G, 2931, "Man's Place in Nature", "Man's Place in Nature (Huxley)"),
    ("fairy-tale", G, 1597, "Andersen", "Andersen's Fairy Tales"),
    ("fairy-tale", G, 503, "Blue Fairy Book", "The Blue Fairy Book (Lang)"),
    ("encyclopedia-geology", W, "Volcano", "volcano", "Wikipedia: Volcano"),
    ("encyclopedia-geology", W, "Glacier", "glacier", "Wikipedia: Glacier"),
    ("encyclopedia-biography", W, "Marie Curie", "Curie", "Wikipedia: Marie Curie"),
    ("encyclopedia-biography", W, "Nikola Tesla", "Tesla", "Wikipedia: Nikola Tesla"),
    ("encyclopedia-history", W, "Byzantine Empire", "Byzantine", "Wikipedia: Byzantine Empire"),
    ("encyclopedia-history", W, "Silk Road", "Silk Road", "Wikipedia: Silk Road"),
    ("encyclopedia-mathematics", W, "Fibonacci sequence", "Fibonacci", "Wikipedia: Fibonacci sequence"),
    ("encyclopedia-mathematics", W, "Pythagorean theorem", "Pythagor", "Wikipedia: Pythagorean theorem"),
    ("source-code", U, "https://raw.githubusercontent.com/python/cpython/main/Lib/shlex.py", "class shlex", "CPython stdlib shlex.py"),
    ("source-code", U, "https://raw.githubusercontent.com/python/cpython/main/Lib/heapq.py", "def heappush", "CPython stdlib heapq.py"),
    ("technical-manual", U, "https://raw.githubusercontent.com/git/git/master/Documentation/git-bisect.adoc", "bisect", "git-bisect(1) documentation"),
    ("technical-manual", U, "https://raw.githubusercontent.com/curl/curl/master/docs/MANUAL.md", "curl", "curl MANUAL"),
]
SPARES = [  # Deron P141 #3: 4 spares in FIXED order; a doc that fails validation is replaced by the next spare, and the swap is logged
    ("spare-novel", G, 84, "Frankenstein", "Frankenstein (Shelley)"),
    ("spare-essay", G, 205, "Walden", "Walden (Thoreau)"),
    ("spare-science-19c", G, 944, "Beagle", "The Voyage of the Beagle (Darwin)"),
    ("spare-fable", G, 11339, "Aesop", "Aesop's Fables (Townsend tr.)"),
]
WHICH = sys.argv[2] if len(sys.argv) > 2 else "main"
assert len(SOURCES) == 32 and all(sum(s[0] == g for s in SOURCES) == 2 for g in {s[0] for s in SOURCES}) and len({s[0] for s in SOURCES}) == 16
UA = {"User-Agent": "nyaa-jlens-corpus/1.0 (research; one fetch per source)"}
def get(url, path):
    if not os.path.exists(path):
        time.sleep(1); open(path, "wb").write(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read())
    return open(path, encoding="utf-8", errors="replace").read().replace("\r\n", "\n")
out = []
for genre, kind, ref, must, label in (SPARES if WHICH == "spares" else SOURCES):
    if kind == G:
        url = f"https://www.gutenberg.org/ebooks/{ref}"
        raw = get(f"https://www.gutenberg.org/cache/epub/{ref}/pg{ref}.txt", os.path.join(RAW, f"pg{ref}.txt"))
        if must.lower() not in raw[:3000].lower(): sys.exit(f"STOP: pg{ref} header lacks {must!r} (wrong ebook id?)")
        s = raw.index("*** START OF"); s = raw.index("\n", s) + 1; e = raw.index("*** END OF"); body = raw[s:e]
    elif kind == W:
        url = "https://en.wikipedia.org/wiki/" + ref.replace(" ", "_")
        api = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1, "format": "json", "redirects": 1, "titles": ref})
        j = json.loads(get(api, os.path.join(RAW, "wp-" + ref.replace(" ", "_") + ".json")))
        body = next(iter(j["query"]["pages"].values()))["extract"]
        body = re.sub(r"\n+==+ [^=\n]+ ==+\n+", "\n\n", body)   # section headings become paragraph breaks
        if must.lower() not in body.lower(): sys.exit(f"STOP: Wikipedia {ref!r} extract lacks {must!r}")
    else:
        url = ref; body = get(ref, os.path.join(RAW, re.sub(r"[^A-Za-z0-9.]+", "_", ref.split("//")[1])))
        if must.lower() not in body.lower(): sys.exit(f"STOP: {ref} lacks {must!r}")
    minlet = 0.4 if genre == "source-code" else 0.6; pos = int(len(body) * 0.55); st = None
    for m in re.finditer(r"\n\s*\n", body[pos:]):
        cand = pos + m.end(); para = body[cand:].split("\n\n", 1)[0]
        if len(para) >= 200 and sum(c.isalpha() for c in para) / max(1, len(para)) >= minlet: st = cand; break
    if st is None: sys.exit(f"STOP: no qualifying paragraph after 55% in {label}")
    text = body[st:st + 3000]
    if genre != "source-code":   # code keeps its indentation; prose gets select.py's whitespace and Gutenberg-markup normalisation
        text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
        text = re.sub(r"_(\S(?:[^_]{0,80}?\S)?)_", r"\1", text); text = re.sub(r"=(\S(?:[^=]{0,80}?\S)?)=", r"\1", text)
        text = re.sub(r"\{\d+\}", "", text)
    text = text.strip()
    out.append({"genre": genre, "source": label, "url": url, "body_offset": st, "body_len": len(body), "frac": round(st / len(body), 4),
                "sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text})
json.dump(out, open(os.path.join(OUT, "stage2_spares.json" if WHICH == "spares" else "stage2_corpus.json"), "w"), indent=1, ensure_ascii=False)
for p in out: print(f'{p["genre"]:24s} {p["source"][:38]:38s} frac {p["frac"]:.3f} len {len(p["text"])} sha {p["sha256"][:12]}')
print(f"wrote {len(out)} docs")
