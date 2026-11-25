"""concept_common.py: the item rules R1-R4 of FROZEN BAR v4 + v4.1 + v4.2 (a reviewer, bar file md5 2217480f), shared by
concept_capture.py and concept_score.py so both apply ONE definition (2026-11-18; the pre-registration card). No model, no torch:
tokenizer and json only. R2 is v4.2's (the WHOLE answer, from the greedy continuation); R1, R3, R4 are v4.1's."""
import json, os, re
import numpy as np
SETS = ["association", "multihop", "multilingual", "order-ops", "poetry", "typo"]
FILTERED = {"multihop", "multilingual", "order-ops", "poetry"}        # R2: sets with a target (poetry: its rhyme word)
DSEED = 20261119

def load_items(data_dir):
    """All items of the six sets, in file order. Each: {set, name, prompt, target|None, intermediates}."""
    out = []
    for s in SETS:
        for it in json.load(open(os.path.join(data_dir, "data", "evaluations", f"lens-eval-{s}.json")))["items"]:
            out.append({"set": s, "name": it["name"], "prompt": it["prompt"], "target": it.get("target"), "intermediates": list(it["intermediates"])})
    return out

def enc(tok, text):
    return tok(text, add_special_tokens=False).input_ids

def forms(tok, w):
    """R3: single-token ids among {w, ' '+w, W, ' '+W}, W = first letter capitalized. Order kept, duplicates removed."""
    W = w[:1].upper() + w[1:]; ids = []
    for f in (w, " " + w, W, " " + W):
        e = enc(tok, f)
        if len(e) == 1 and e[0] not in ids: ids.append(e[0])
    return ids

def read_pos(tok, item):
    """R1: token index into the tokenized prompt (no BOS). None = dropped before the run (poetry only)."""
    e = tok(item["prompt"], add_special_tokens=False, return_offsets_mapping=True)
    n = len(e.input_ids)
    if item["set"] != "poetry": return n - 1
    p = item["prompt"]; i1 = p.find("\n"); i2 = p.find("\n", i1 + 1) if i1 >= 0 else -1
    if i2 < 0: return None
    hit = [k for k, (a, b) in enumerate(e.offset_mapping) if a <= i2 < b]
    if len(hit) != 1: return None
    a, b = e.offset_mapping[hit[0]]
    if a <= i1 < b: return None                                       # one token holds both newlines
    return hit[0]

def target_text(item):
    """The target string of R2. Poetry: its intermediate (the rhyme word). None for sets without one."""
    return item["intermediates"][0] if item["set"] == "poetry" else item["target"]

def target_ids(tok, item):
    """REPORT-ONLY since v4.2 (was v4.1's R2): {first token of tok(target), first token of tok(' '+target)}."""
    t = target_text(item)
    if t is None: return None
    return sorted({enc(tok, t)[0], enc(tok, " " + t)[0]})

NUM = re.compile(r"[-+]?\d+(?:[.,]\d+)*")
def answer_ok(target, text):
    """R2 (v4.2), the WHOLE answer. text = the decoded greedy continuation. Strip leading whitespace. Kept if the text starts with
    the target (exact for numbers; casefold for words) AND the next character after the target is absent or not alphanumeric."""
    t = text.lstrip()
    if NUM.fullmatch(target): head = t[:len(target)] == target
    else: head = t[:len(target)].casefold() == target.casefold()
    if not head: return False
    nxt = t[len(target):len(target) + 1]
    return nxt == "" or not nxt.isalnum()

def prepare(tok, items):
    """Applies R1 and R3. Returns (kept_before_filter, drops) where each kept item gains pos, inter (list of (word, form_ids)),
    tgt (ids or None). drops counts, per set: no_position, intermediates_no_single_token, items_no_intermediate."""
    drops = {s: {"no_position": 0, "intermediates_no_single_token": 0, "items_no_intermediate": 0} for s in SETS}; kept = []
    for idx, it in enumerate(items):
        it = dict(it); it["idx"] = idx; it["pos"] = read_pos(tok, it)
        if it["pos"] is None: drops[it["set"]]["no_position"] += 1; continue
        inter = []
        for w in it["intermediates"]:
            f = forms(tok, w)
            if f: inter.append((w, f))
            else: drops[it["set"]]["intermediates_no_single_token"] += 1
        if not inter: drops[it["set"]]["items_no_intermediate"] += 1; continue
        it["inter"] = inter; it["tgt"] = target_ids(tok, it) if it["set"] in FILTERED else None; kept.append(it)
    return kept, drops

def decoys(kept):
    """R4: per set, a seeded derangement of the kept items; decoy = the partner's first single-token intermediate that does not
    appear (case-insensitive) in the item's prompt, target or intermediates; if none, the next item in permutation order.
    Called on the kept-BEFORE-filter list (review note #3), so the whole decoy list exists before any model data.
    Returns {idx: (word, form_ids, partner_idx, fallback)}; fallback = True when the first partner gave none and a later one in
    permutation order was used (counted by the caller). An item with no decoy anywhere in its set gets None."""
    out = {}
    for s in SETS:
        g = [it for it in kept if it["set"] == s]; n = len(g)
        if n < 2:
            for it in g: out[it["idx"]] = None
            continue
        k = 0
        while True:
            perm = np.random.default_rng(DSEED + k).permutation(n)
            if not any(perm[i] == i for i in range(n)): break
            k += 1
        for i, it in enumerate(g):
            hay = " ".join([it["prompt"], it["target"] or ""] + it["intermediates"]).lower(); found = None
            for t in range(n):                                         # partner perm[i]; if it gives none, the next in permutation order
                j = int(perm[(i + t) % n])
                if j == i: continue
                for w, f in g[j]["inter"]:
                    if w.lower() not in hay: found = (w, f, g[j]["idx"], j != int(perm[i])); break
                if found: break
            out[it["idx"]] = found
    return out
