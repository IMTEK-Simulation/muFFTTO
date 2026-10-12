import re, sys, os, glob
T = sys.argv[1]
num = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
skip = re.compile(r"(?i)time|elapsed|took|sec\b| s\)|warning|^\s*\d+\s+\d+\s+\(")
def nums(p):
    out = []
    for line in open(p, errors="replace"):
        if skip.search(line): continue
        out += [float(x) for x in num.findall(line)]
    return out
for d in sorted(glob.glob(f"{T}/*/")):
    name = os.path.basename(d.rstrip("/"))
    c, g = f"{d}cpu/stdout.log", f"{d}gpu/stdout.log"
    if not (os.path.exists(c) and os.path.exists(g)): continue
    a, b = nums(c), nums(g)
    if not a:
        print(f"{name:62s} no numeric output"); continue
    if len(a) != len(b):
        print(f"{name:62s} DIFFERENT number of values: cpu {len(a)} gpu {len(b)}"); continue
    scale = max(abs(x) for x in a) or 1
    worst = max(abs(x - y) / max(abs(x), 1e-6 * scale) for x, y in zip(a, b))
    print(f"{name:62s} {len(a):6d} values, max rel diff {worst:.2e}")
