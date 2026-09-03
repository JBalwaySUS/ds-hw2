"""Turn the timing log (results/bench.txt) into speed-up, efficiency and
phase-breakdown plots. Images are written next to the log file."""
import re, sys, os, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    sys.exit("Usage: python3 plot_speedup.py results/bench.txt [results/bench_physical_cores.txt]")

logpath = sys.argv[1]
cmppath = sys.argv[2] if len(sys.argv) > 2 else None   # optional control-experiment log
outdir = os.path.dirname(os.path.abspath(logpath))     # write beside the log

pat = re.compile(
    r"P=(\d+)\s+m=(\d+)\s+n=(\d+)\s+p=(\d+)\s+time=([0-9.]+)"
    r"(?:\s+s\s+bcastB=([0-9.]+)\s+scatterA=([0-9.]+)\s+compute=([0-9.]+)\s+gatherC=([0-9.]+))?"
)
data = collections.defaultdict(dict)    # data[size][P] = total time
phase = collections.defaultdict(dict)   # phase[size][P] = (bcast, scatter, compute, gather)
for line in open(logpath):
    mo = pat.search(line)
    if not mo:
        continue
    P, m, n, p, t = mo.group(1), mo.group(2), mo.group(3), mo.group(4), mo.group(5)
    size = f"{m}x{n}x{p}"
    data[size][int(P)] = float(t)
    if mo.group(6) is not None:
        phase[size][int(P)] = tuple(float(mo.group(i)) for i in (6, 7, 8, 9))

if not data:
    sys.exit(f"No timing lines found in {logpath}")

def save(name):
    path = os.path.join(outdir, name)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    return path

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
written = [save("speedup.png")]

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
written.append(save("efficiency.png"))

# Phase breakdown: how the parallel phase splits into communication vs computation.
if phase:
    def volume(sz):
        a, b, c = (int(x) for x in sz.split("x"))
        return a * b * c
    # Smallest and largest problem side by side: the same algorithm is
    # communication-dominated at one end and computation-dominated at the other.
    picks = sorted(phase, key=volume)
    picks = [picks[0], picks[-1]] if len(picks) > 1 else picks
    labels = ["Bcast B", "Scatter A", "Compute", "Gather C"]
    fig, axes = plt.subplots(1, len(picks), figsize=(5.4 * len(picks), 4.4))
    if len(picks) == 1:
        axes = [axes]
    for ax, sz in zip(axes, picks):
        Ps = sorted(phase[sz])
        bottom = [0.0] * len(Ps)
        for i, lab in enumerate(labels):
            vals = [phase[sz][P][i] for P in Ps]
            ax.bar([str(P) for P in Ps], vals, bottom=bottom, label=lab)
            bottom = [b + v for b, v in zip(bottom, vals)]
        # Annotate with the measured wall-clock total. The stacked bar is the sum
        # of four independent per-phase MPI_MAX values, so it can exceed the
        # wall-clock time when different ranks are slowest in different phases;
        # percentages therefore live in the report table, not on the bars.
        for x, P in enumerate(Ps):
            ax.text(x, sum(phase[sz][P]), f"T={data[sz][P]:.4f}s",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xlabel("Processes P"); ax.set_ylabel("Time (s)")
        ax.set_title(sz); ax.grid(True, axis="y")
        ax.margins(y=0.15)
    axes[0].legend(fontsize=8)
    fig.suptitle("Where the parallel phase goes: communication vs computation")
    fig.tight_layout()
    written.append(save("phase_breakdown.png"))
    plt.close(fig)

# Control experiment: same benchmark on full physical cores instead of SMT
# siblings. Overlaying the two shows whether a P=8 plateau is the algorithm or
# the hardware allocation.
if cmppath:
    cdata = collections.defaultdict(dict)
    for line in open(cmppath):
        mo = pat.search(line)
        if mo:
            cdata[f"{mo.group(2)}x{mo.group(3)}x{mo.group(4)}"][int(mo.group(1))] = float(mo.group(5))
    shared = [z for z in sorted(cdata) if z in data]
    if shared:
        plt.figure()
        for sz in shared:
            Ps = sorted(cdata[sz])
            plt.plot(Ps, [data[sz][1] / data[sz][P] for P in Ps], marker="o",
                     label=f"{sz}  (8 SMT threads / 4 cores)")
            plt.plot(Ps, [cdata[sz][1] / cdata[sz][P] for P in Ps], marker="^", ls="--",
                     label=f"{sz}  (8 physical cores)")
        Pm = max(P for sz in shared for P in cdata[sz])
        plt.plot(range(1, Pm + 1), range(1, Pm + 1), "k:", label="ideal")
        plt.xlabel("Processes P"); plt.ylabel("Speed-up S(P)=T1/TP")
        plt.title("Effect of CPU allocation on scaling")
        plt.legend(fontsize=8); plt.grid(True)
        written.append(save("smt_vs_cores.png"))

# Formatted benchmark table, regenerated from the same log so the report never
# has to be updated by hand-transcribing numbers.
sizes = sorted(data, key=lambda k: [int(x) for x in k.split("x")])
Pall = sorted({P for d in data.values() for P in d})
w = max(14, max(len(z) for z in sizes) + 2)

def table(title, fmt, val):
    lines = [f"{'-'*64}", title, f"{'-'*64}",
             f"  {'Size'.ljust(w)}|" + "|".join(f"  P={P}  ".center(11) for P in Pall),
             f"  {'-'*w}+" + "+".join("-" * 11 for _ in Pall)]
    for sz in sizes:
        row = f"  {sz.ljust(w)}|"
        for P in Pall:
            row += (fmt.format(val(sz, P)) if P in data[sz] else "n/a").center(11) + "|"
        lines.append(row.rstrip("|"))
    return "\n".join(lines) + "\n"

out = [
    "=" * 64,
    " Q1  Row-Row Matrix Multiplication  -  Benchmark Results",
    "=" * 64,
    "",
    "Generated by plot_speedup.py from " + os.path.basename(logpath) + ".",
    "Timing : MPI_Wtime over bcast(B) + scatter(A) + local multiply + gather(C),",
    "         between two MPI_Barriers. See report.md section 6.",
    "",
    "  P        number of MPI processes",
    "  Size     matrix dimensions m x n x p",
    "  Time(s)  wall-clock seconds for the parallel phase (lower = faster)",
    "  S(P)     speed-up   = T(1) / T(P)      ideal = P, higher is better",
    "  E(P)     efficiency = S(P) / P         1.00 = perfect scaling",
    "",
    table("RUNTIME   Time(s)   [lower is better]", "{:.6f}", lambda s_, P: data[s_][P]),
    table("SPEED-UP   S(P) = T(1)/T(P)   [ideal = P]", "{:.2f}",
          lambda s_, P: data[s_][1] / data[s_][P]),
    table("EFFICIENCY   E(P) = S(P)/P   [1.00 = perfect]", "{:.2f}",
          lambda s_, P: (data[s_][1] / data[s_][P]) / P),
]
if phase:
    out.append("-" * 64)
    out.append("PHASE BREAKDOWN   seconds, slowest rank per phase")
    out.append("-" * 64)
    out.append(f"  {'Size'.ljust(w)}| P |   BcastB |  ScatterA |   Compute |   GatherC")
    for sz in sizes:
        for P in sorted(phase.get(sz, {})):
            b, sc, cp, g = phase[sz][P]
            out.append(f"  {sz.ljust(w)}|{P:>2} | {b:8.6f} | {sc:9.6f} | {cp:9.6f} | {g:9.6f}")
    out.append("")
out.append("=" * 64)

tbl = os.path.join(outdir, "benchmark_results.txt")
with open(tbl, "w") as fh:
    fh.write("\n".join(out) + "\n")
written.append(tbl)

print("Wrote: " + ", ".join(written))
