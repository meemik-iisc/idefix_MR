#!/usr/bin/env python3
"""
Single vs double precision performance diagnostics for the tcgm runs.

Parses the idefix logs (idefix.0.log) of both runs and produces:
  diagnostics/summary.md          table: wall time, cycles, perf, speedup, memory, I/O
  diagnostics/walltime.png        cumulative wall time vs simulation time
  diagnostics/throughput.png      cell updates/s vs simulation time
  diagnostics/speedup.png         double/single speedup vs simulation time
  diagnostics/timestep.png        dt and cooling sub-cycles vs simulation time

Wall time vs sim time is reconstructed from the per-log-interval rate idefix
prints: interval wall time = cyclePeriod * Ncells / (cell updates/s).

Runs were done one after the other (run_both.sh), each with the GPU to itself.
The earlier shared-GPU runs (tcgm_*1) are only quoted in the summary for reference.

Usage (from this directory):  python3 scaling_diagnostics.py
"""
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = {
    # label: (run dir, color)
    "double": (os.path.join(HERE, "build-gpu", "tcgm_double"), "#2a78d6"),
    "single": (os.path.join(HERE, "build_gpu_single", "tcgm_single"), "#eb6834"),
}
# earlier runs that shared one GPU, quoted for reference only
SHARED = {
    "double": os.path.join(HERE, "build-gpu", "tcgm_double1"),
    "single": os.path.join(HERE, "build_gpu_single", "tcgm_single1"),
}
OUTDIR = os.path.join(HERE, "diagnostics_sequential")

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "font.size": 11, "legend.frameon": False,
})


def parse_log(path):
    txt = open(path).read()
    d = {}

    # grid size from the "Grid: full grid size" block
    sizes = re.findall(r"Direction X\d:.*?\.\.\.\.(\d+)\.\.\.\.", txt)
    d["ncells"] = int(np.prod([int(s) for s in sizes]))
    d["cycle_period"] = int(re.search(r"^\s*log\s+(\d+)", txt, re.M).group(1))

    # time-integrator table
    rows = []
    for line in txt.splitlines():
        if not line.startswith("TimeIntegrator:") or "|" not in line:
            continue
        p = [x.strip() for x in line.split(":", 1)[1].split("|")]
        try:
            t, cyc, dt = float(p[0]), int(p[1]), float(p[2])
        except ValueError:
            continue  # header line
        perf = float(p[3]) if p[3] != "N/A" else np.nan
        nsub = int(p[4]) if len(p) > 4 else 0
        rows.append((t, cyc, dt, perf, nsub))
    a = np.array(rows)
    d["t"], d["cycle"], d["dt"], d["perf"], d["nsub"] = a.T

    # per-interval wall time; perf at row i is for the interval ending at row i
    dwall = np.zeros_like(d["t"])
    ok = np.isfinite(d["perf"])
    dwall[ok] = d["cycle_period"] * d["ncells"] / d["perf"][ok]
    d["wall"] = np.cumsum(dwall)

    # final summary lines
    m = re.search(r"Completed in (.*?) and (\d+) cycles", txt)
    d["completed_str"] = m.group(1)
    d["ncycles"] = int(m.group(2))
    secs = 0
    for val, unit in re.findall(r"(\d+) (day|hour|minute|second)", m.group(1)):
        secs += int(val) * {"day": 86400, "hour": 3600, "minute": 60, "second": 1}[unit]
    d["walltime"] = secs
    d["perf_total"] = float(re.search(r"Perfs are ([\d.e+-]+)", txt).group(1))
    d["t_end"] = float(re.search(r"Reached t=([\d.e+-]+)", txt).group(1))

    def mem(space):
        mm = re.search(rf"memory usage for {space} memory space: ([\d.e+-]+) (\w+)", txt)
        if not mm:
            return np.nan
        return float(mm.group(1)) * {"GB": 1024, "MB": 1, "kB": 1 / 1024}[mm.group(2)]
    d["mem_gpu_MB"], d["mem_host_MB"] = mem("Cuda"), mem("Host")

    d["vtk_write"] = np.array([float(x) for x in
                               re.findall(r"Vtk: Write file .*?done in ([\d.e+-]+)", txt)])
    d["dmp_write"] = np.array([float(x) for x in
                               re.findall(r"Dump: Write file .*?done in ([\d.e+-]+)", txt)])
    d["precision"] = re.search(r"Precision: (\w+)", txt).group(1)
    return d


def dir_size_MB(pattern):
    return sum(os.path.getsize(f) for f in glob.glob(pattern)) / 1024**2


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    R = {}
    for name, (rundir, _) in RUNS.items():
        R[name] = parse_log(os.path.join(rundir, "idefix.0.log"))
        R[name]["vtk_MB"] = dir_size_MB(os.path.join(rundir, "data.*.vtk"))
        R[name]["dmp_MB"] = dir_size_MB(os.path.join(rundir, "dump.*.dmp"))
        R[name]["nvtk"] = len(glob.glob(os.path.join(rundir, "data.*.vtk")))
    D, S = R["double"], R["single"]

    # earlier shared-GPU runs, for reference
    SH = {k: parse_log(os.path.join(v, "idefix.0.log")) for k, v in SHARED.items()
          if os.path.isfile(os.path.join(v, "idefix.0.log"))}

    # ---------------- summary table ----------------
    def row(label, fmt, key, ratio=True):
        a, b = (key(D), key(S)) if callable(key) else (D[key], S[key])
        r = f"{a / b:.2f}" if ratio else ""
        return f"| {label} | {fmt.format(a)} | {fmt.format(b)} | {r} |"

    lines = [
        "# Single vs double precision — tcgm run (sequential, GPU not shared)",
        "",
        f"Grid {D['ncells']:,} cells, t_end = {D['t_end']:g} code units, GPU run.",
        "",
        "| Quantity | Double | Single | Double / Single |",
        "|---|---|---|---|",
        f"| Wall time | {D['completed_str']} | {S['completed_str']} | {D['walltime'] / S['walltime']:.2f} |",
        row("Wall time [s]", "{:.0f}", "walltime"),
        row("Cycles", "{:d}", "ncycles"),
        row("Wall time per cycle [ms]", "{:.1f}", lambda r: 1e3 * r["walltime"] / r["ncycles"]),
        row("Throughput [Mcell-updates/s]", "{:.1f}", lambda r: r["perf_total"] / 1e6),
        row("Median interval throughput [Mcell/s]", "{:.1f}", lambda r: np.nanmedian(r["perf"]) / 1e6),
        row("Peak GPU memory [MB]", "{:.0f}", "mem_gpu_MB"),
        row("Peak host memory [MB]", "{:.0f}", "mem_host_MB"),
        row("Mean VTK write time [s]", "{:.3f}", lambda r: r["vtk_write"].mean()),
        row("Mean dump write time [s]", "{:.3f}", lambda r: r["dmp_write"].mean()),
        row("Total VTK size [MB]", "{:.0f}", "vtk_MB"),
        row("Total dump size [MB]", "{:.0f}", "dmp_MB"),
        row("Median dt", "{:.3e}", lambda r: np.median(r["dt"][1:])),
        row("Median cooling nsub", "{:.0f}", lambda r: np.median(r["nsub"][1:])),
        "",
        f"**Speedup (wall time, double/single): {D['walltime'] / S['walltime']:.2f}x**  ",
        f"**Speedup (per cycle): {(D['walltime'] / D['ncycles']) / (S['walltime'] / S['ncycles']):.2f}x**",
        "",
        f"Throughput variability (interval perf, std/median): "
        f"double {np.nanstd(D['perf'][1:]) / np.nanmedian(D['perf'][1:]):.1%}, "
        f"single {np.nanstd(S['perf'][1:]) / np.nanmedian(S['perf'][1:]):.1%}.",
        "",
        f"Cycle count difference: {S['ncycles'] - D['ncycles']:+d} "
        f"({100 * (S['ncycles'] - D['ncycles']) / D['ncycles']:+.2f} %) single vs double.",
    ]
    if len(SH) == 2:
        lines += [
            "",
            "## Reference: earlier runs sharing one GPU (tcgm_*1)",
            "",
            "| | Double | Single | Double / Single |",
            "|---|---|---|---|",
            f"| Wall time [s] | {SH['double']['walltime']} | {SH['single']['walltime']} | "
            f"{SH['double']['walltime'] / SH['single']['walltime']:.2f} |",
            f"| Throughput [Mcell/s] | {SH['double']['perf_total'] / 1e6:.1f} | "
            f"{SH['single']['perf_total'] / 1e6:.1f} | "
            f"{SH['double']['perf_total'] / SH['single']['perf_total']:.2f} |",
            "",
            f"Sequential vs shared wall time: double {D['walltime'] / SH['double']['walltime']:.2f}x, "
            f"single {S['walltime'] / SH['single']['walltime']:.2f}x.",
        ]
    summary = "\n".join(lines)
    open(os.path.join(OUTDIR, "summary.md"), "w").write(summary + "\n")
    print(summary)

    # ---------------- 1. wall time vs sim time ----------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, (_, c) in RUNS.items():
        r = R[name]
        ax.plot(r["t"], r["wall"] / 3600, color=c, label=name)
        ax.annotate(f"{name}: {r['walltime'] / 3600:.2f} h", (r["t"][-1], r["wall"][-1] / 3600),
                    xytext=(-6, 0), textcoords="offset points", ha="right", va="center",
                    color=INK2, fontsize=10)
    ax.set_xlabel("simulation time [code units]")
    ax.set_ylabel("cumulative wall time [h]")
    ax.set_title("Wall time to reach simulation time", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "walltime.png"), dpi=150)

    # ---------------- 2. throughput vs sim time ----------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, (_, c) in RUNS.items():
        r = R[name]
        ax.plot(r["t"], r["perf"] / 1e6, color=c, label=name, marker="o", ms=3)
        ax.axhline(r["perf_total"] / 1e6, color=c, lw=1, ls="--")
    ax.set_xlabel("simulation time [code units]")
    ax.set_ylabel("Mcell-updates / s")
    ax.set_title(f"Throughput per {D['cycle_period']} cycles (dashed: run average)", loc="left")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "throughput.png"), dpi=150)

    # ---------------- 3. speedup vs sim time ----------------
    tgrid = np.linspace(0, min(D["t"][-1], S["t"][-1]), 400)[1:]
    wd, ws = np.interp(tgrid, D["t"], D["wall"]), np.interp(tgrid, S["t"], S["wall"])
    # instantaneous: ratio of wall time per unit simulation time
    dwd, dws = np.gradient(wd, tgrid), np.gradient(ws, tgrid)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tgrid, wd / ws, color=INK, label="cumulative (wall to reach t)")
    ax.plot(tgrid, dwd / dws, color=MUTED, lw=1, label="instantaneous (d wall / dt)")
    ax.axhline(1, color="#c3c2b7", lw=1)
    ax.axhline(D["walltime"] / S["walltime"], color=INK, lw=1, ls="--")
    ax.set_xlabel("simulation time [code units]")
    ax.set_ylabel("speedup  (double / single wall time)")
    ax.set_title(f"Single-precision speedup, final {D['walltime'] / S['walltime']:.2f}x", loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "speedup.png"), dpi=150)

    # ---------------- 4. dt and cooling sub-cycles ----------------
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for name, (_, c) in RUNS.items():
        r = R[name]
        a1.plot(r["t"][1:], r["dt"][1:], color=c, label=name)
        a2.plot(r["t"][1:], r["nsub"][1:], color=c, label=name)
    a1.set_ylabel("dt [code units]")
    a1.set_title("Time step and cooling sub-cycles at log points", loc="left")
    a1.legend(loc="lower right")
    a2.set_ylabel("cooling nsub")
    a2.set_xlabel("simulation time [code units]")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "timestep.png"), dpi=150)

    print(f"\nFigures and summary written to {OUTDIR}")


if __name__ == "__main__":
    main()
