import re, sys, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    sys.exit("Usage: python3 plot_speedup.py times.txt")

pat = re.compile(r"P=(\d+)\s+m=(\d+)\s+n=(\d+)\s+p=(\d+)\s+time=([0-9.]+)")
# data[size][P] = time
data = collections.defaultdict(dict)
for line in open(sys.argv[1]):
    mobj = pat.search(line)
    if not mobj:
        continue
    P, m, n, p, t = mobj.groups()
    size = f"{m}x{n}x{p}"
    data[size][int(P)] = float(t)

# Speed-up plot
plt.figure()
for size, d in sorted(data.items()):
    Ps = sorted(d)
    t1 = d.get(1)
    if not t1:
        continue
    plt.plot(Ps, [t1 / d[P] for P in Ps], marker="o", label=size)
Pmax = max((P for d in data.values() for P in d), default=8)
plt.plot(range(1, Pmax + 1), range(1, Pmax + 1), "k--", label="ideal")
plt.xlabel("Processes P"); plt.ylabel("Speed-up S(P)=T1/TP")
plt.title("Row-Row MatMul speed-up"); plt.legend(); plt.grid(True)
plt.savefig("speedup.png", dpi=130, bbox_inches="tight")

# Efficiency plot
plt.figure()
for size, d in sorted(data.items()):
    Ps = sorted(d)
    t1 = d.get(1)
    if not t1:
        continue
    plt.plot(Ps, [(t1 / d[P]) / P for P in Ps], marker="s", label=size)
plt.axhline(1.0, ls="--", color="k", label="ideal")
plt.xlabel("Processes P"); plt.ylabel("Efficiency E(P)=S(P)/P")
plt.title("Row-Row MatMul efficiency"); plt.legend(); plt.grid(True)
plt.savefig("efficiency.png", dpi=130, bbox_inches="tight")
print("Wrote speedup.png and efficiency.png")
