#!/usr/bin/env python3
# plot_q8.py - build speedup, phase-breakdown, and communication-share plots
# from the phase-timed benchmark log.
#
# Usage:
#   grep 'time=' bench_q8.txt > /dev/null   # (bench_q8.txt is the raw log)
#   python3 plot_q8.py bench_q8.txt
#
# Each line looks like:
#   MPI  P=4 N=1000000 K=10 S=1000  time=0.190657 s  scatter=0.013370 compute=0.030968 combine=0.146319
import re, sys, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    sys.exit("Usage: python3 plot_q8.py bench_q8.txt")

pat = re.compile(r"P=(\d+)\s+N=(\d+).*?time=([0-9.]+).*?scatter=([0-9.]+)\s+compute=([0-9.]+)\s+combine=([0-9.]+)")
# data[N][P] = (total, scatter, compute, combine)
data = collections.defaultdict(dict)
for line in open(sys.argv[1]):
    m = pat.search(line)
    if not m:
        continue
    P, N, tot, sc, co, cb = m.groups()
    data[int(N)][int(P)] = (float(tot), float(sc), float(co), float(cb))

def label(n):
    return {100000: "100k", 1000000: "1M", 5000000: "5M"}.get(n, str(n))

# 1) total-time speedup
plt.figure()
for N in sorted(data):
    Ps = sorted(data[N]); t1 = data[N][1][0]
    plt.plot(Ps, [t1 / data[N][p][0] for p in Ps], marker="o", label=f"N={label(N)}")
Pmax = max(p for N in data for p in data[N])
plt.plot(range(1, Pmax + 1), range(1, Pmax + 1), "k--", label="ideal")
plt.xlabel("Processes P"); plt.ylabel("Speed-up S(P)=T1/TP")
plt.title("Q8 total-time speed-up"); plt.legend(); plt.grid(True)
plt.savefig("q8_speedup.png", dpi=130, bbox_inches="tight")

# 2) compute-only speedup (the parallelisable part)
plt.figure()
for N in sorted(data):
    Ps = sorted(data[N]); c1 = data[N][1][2]
    plt.plot(Ps, [c1 / data[N][p][2] for p in Ps], marker="s", label=f"N={label(N)}")
plt.plot(range(1, Pmax + 1), range(1, Pmax + 1), "k--", label="ideal")
plt.xlabel("Processes P"); plt.ylabel("Compute-only speed-up")
plt.title("Q8 speed-up of the computation phase alone"); plt.legend(); plt.grid(True)
plt.savefig("q8_compute_speedup.png", dpi=130, bbox_inches="tight")

# 3) stacked phase breakdown for the largest dataset (5M)
big = max(data)
Ps = sorted(data[big])
scat = [data[big][p][1] for p in Ps]
comp = [data[big][p][2] for p in Ps]
comb = [data[big][p][3] for p in Ps]
plt.figure()
x = range(len(Ps))
plt.bar(x, comp, label="compute")
plt.bar(x, scat, bottom=comp, label="scatter (comm)")
plt.bar(x, comb, bottom=[comp[i] + scat[i] for i in x], label="combine (comm)")
plt.xticks(list(x), [f"P={p}" for p in Ps])
plt.ylabel("Time (s)")
plt.title(f"Q8 phase breakdown (N={label(big)})")
plt.legend(); plt.grid(True, axis="y")
plt.savefig("q8_phase_breakdown.png", dpi=130, bbox_inches="tight")

print("Wrote q8_speedup.png, q8_compute_speedup.png, q8_phase_breakdown.png")
