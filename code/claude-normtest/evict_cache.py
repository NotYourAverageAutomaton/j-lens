"""Evict model files from the page cache (posix_fadvise DONTNEED; no root). Claude 2026-11-04: on the GB10, GPU memory IS system RAM;
a full page cache made the 21:42 smoke OOM at model load while nvidia-smi showed nothing. Prints MemAvailable before/after."""
import os, sys
def avail():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable:"): return int(l.split()[1]) / 2**20
def cached():
    for l in open("/proc/meminfo"):
        if l.startswith("Cached:"): return int(l.split()[1]) / 2**20
b = (avail(), cached()); n = 0
for root in sys.argv[1:]:
    for dp, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f)
            try:
                fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd); n += 1
            except OSError: pass
print(f"evict: {n} files; MemAvailable {b[0]:.1f} -> {avail():.1f} GiB; Cached {b[1]:.1f} -> {cached():.1f} GiB", flush=True)
