"""Apply bug-#5 skip-and-log to our_jlens.py (post P020): a too-short prompt is skipped and
logged instead of raising, so one short passage can't kill a 6 h record run."""
p = "$JLENS_HOME/our_jlens.py"
s = open(p).read()
old = '        if seq_len <= skip_first + 1:\n            raise ValueError(f"prompt too short: {seq_len} <= {skip_first+1}")'
new = ('        if seq_len <= skip_first + 1:\n'
       '            print(f"[fit] SKIP prompt too short: {seq_len} <= {skip_first+1} '
       '(bug-#5 skip-and-log, post P020)", flush=True)\n'
       '            return {}, 0')
if "bug-#5 skip-and-log" in s:
    print("our_jlens.py: skip-and-log already present, no change")
elif old in s:
    open(p, "w").write(s.replace(old, new, 1))
    print("our_jlens.py: skip-and-log applied")
else:
    raise SystemExit("PATTERN NOT FOUND in our_jlens.py — aborting, no change made")
