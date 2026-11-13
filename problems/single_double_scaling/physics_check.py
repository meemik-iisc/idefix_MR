#!/usr/bin/env python3
"""
Single vs double precision: do the global physical quantities diverge?

Nothing is strictly conserved in this setup (inflow at z_max, outflow boundaries,
wind injection, cooling, gravity), so we
  1. compare domain totals between the two runs at every VTK snapshot:
     total mass, total energy (kinetic + thermal), wind (tracer) mass,
     cold gas mass (T < 2e4 K), hot gas mass (T > 1e6 K);
  2. check tracer-mass conservation: until wind gas reaches a boundary,
     M_tracer(t) must equal the injected mass  int_0^t Mdot(t') dt'.

VTK files are float32 for both runs; all sums are done in float64.
Writes diagnostics_sequential/physics_check.{png,md}.

Usage (from this directory):  python3 physics_check.py
"""
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from pytools.vtk_io import readVTK  # noqa: E402

RUNS = {
    "double": os.path.join(HERE, "build-gpu", "tcgm_double"),
    "single": os.path.join(HERE, "build_gpu_single", "tcgm_single"),
}
OUTDIR = os.path.join(HERE, "diagnostics_sequential")

# --- units and parameters (idefix.ini) ---
L0, V0, RHO0 = 3.08568e21, 1.0e8, 1.67262e-24
GAMMA, MU = 1.6666667, 0.6
MP, KB, MSUN, YR = 1.67262e-24, 1.380649e-16, 1.989e33, 3.15576e7
T0, M0 = L0 / V0, RHO0 * L0**3
MDOT0 = 0.1 * MSUN / YR * T0 / M0   # code units
DT_MDOT = 25.0                       # setup.cpp: Mdot(t) = Mdot0 exp(-t/dt_mdot)
TCOLD, THOT = 2.0e4, 2.0e6   # T_cgm = 1e6 K exactly: keep the hot cut clear of it
MASS_MSUN = M0 / MSUN                # code mass -> Msun


def injected_mass(t):
    return MDOT0 * DT_MDOT * (1 - np.exp(-t / DT_MDOT))


def totals(fname):
    v = readVTK(fname)
    R, Rl, zl = v.x, v.xl, v.yl
    # 2D cylindrical (R, z): dV = 2 pi R dR dz
    dV = 2 * np.pi * (R * np.diff(Rl))[:, None] * np.diff(zl)[None, :]
    d = {k: v.data[k][:, :, 0].astype(np.float64) for k in ("RHO", "PRS", "VX1", "VX2", "VX3", "TR0")}
    rho, prs = d["RHO"], d["PRS"]
    T = prs / rho * MU * MP / KB * V0**2
    m = rho * dV
    ekin = 0.5 * rho * (d["VX1"]**2 + d["VX2"]**2 + d["VX3"]**2) * dV
    eth = prs / (GAMMA - 1) * dV
    # tracer edge diagnostics: has wind gas reached a boundary?
    tr = d["TR0"]
    edge = max(tr[-1, :].max(), tr[:, 0].max(), tr[:, -1].max())
    return dict(t=float(v.t[0]), M=m.sum(), Ekin=ekin.sum(), Eth=eth.sum(),
                E=ekin.sum() + eth.sum(), Mtr=(m * tr).sum(),
                Mcold=m[T < TCOLD].sum(), Mhot=m[T > THOT].sum(), tr_edge=edge)


def load(rundir):
    rows = [totals(f) for f in sorted(glob.glob(os.path.join(rundir, "data.*.vtk")))]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def main():
    R = {name: load(d) for name, d in RUNS.items()}
    D, S = R["double"], R["single"]
    assert np.allclose(D["t"], S["t"], atol=1e-2), "snapshot times differ"
    t = D["t"]

    keys = [("M", "total mass"), ("E", "total energy"), ("Mtr", "wind (tracer) mass"),
            ("Mcold", f"cold mass (T<{TCOLD:.0e} K)"), ("Mhot", f"hot mass (T>{THOT:.0e} K)")]

    def rel(k):
        return (S[k] - D[k]) / np.where(D[k] != 0, np.abs(D[k]), np.nan)

    minj = injected_mass(t)
    # first snapshot where wind gas (tracer > 1e-3) touches a boundary
    leak = np.nonzero((D["tr_edge"] > 1e-3) | (S["tr_edge"] > 1e-3))[0]
    t_leak = t[leak[0]] if leak.size else np.inf

    # ---------------- summary ----------------
    L = ["# Physics check: single vs double precision", "",
         "Relative difference (single - double) / double of domain totals over 61 snapshots.", "",
         "| Quantity | max abs rel. diff | rel. diff at t=60 | median abs rel. diff |",
         "|---|---|---|---|"]
    for k, lab in keys:
        r = rel(k)
        L.append(f"| {lab} | {np.nanmax(np.abs(r)):.2e} (t={t[np.nanargmax(np.abs(r))]:.0f}) "
                 f"| {r[-1]:+.2e} | {np.nanmedian(np.abs(r)):.2e} |")
    L += ["", f"Wind gas first reaches a boundary at t = {t_leak:g}.", "",
          "## Tracer mass vs injected mass (before wind leaves the box)", "",
          "| t | injected [Msun] | double [Msun] | single [Msun] | double/inj - 1 | single/inj - 1 |",
          "|---|---|---|---|---|---|"]
    for i in range(1, len(t)):
        if t[i] > t_leak or (i % 5 and i != 1):
            continue
        L.append(f"| {t[i]:.0f} | {minj[i] * MASS_MSUN:.4e} | {D['Mtr'][i] * MASS_MSUN:.4e} | "
                 f"{S['Mtr'][i] * MASS_MSUN:.4e} | {D['Mtr'][i] / minj[i] - 1:+.2e} | "
                 f"{S['Mtr'][i] / minj[i] - 1:+.2e} |")
    L += ["", "## Totals at t = 60", "",
          "| Quantity | double | single |", "|---|---|---|"]
    for k, lab in keys:
        L.append(f"| {lab} | {D[k][-1]:.6e} | {S[k][-1]:.6e} |")
    summary = "\n".join(L)
    open(os.path.join(OUTDIR, "physics_check.md"), "w").write(summary + "\n")
    print(summary)

    # ---------------- figure ----------------
    fig, axs = plt.subplots(3, 1, figsize=(7.5, 9), sharex=True)
    for k, lab in keys:
        axs[0].plot(t[1:], D[k][1:] / D[k][1:].max() if D[k].max() > 0 else D[k][1:], label=lab)
    axs[0].set_ylabel("double run, normalised")
    axs[0].set_title("Domain totals (double run, each / its max)", loc="left")
    axs[0].legend(fontsize=8, ncol=2)
    for k, lab in keys:
        axs[1].plot(t[1:], np.abs(rel(k)[1:]), label=lab)
    axs[1].set_yscale("log")
    axs[1].set_ylabel("|single - double| / double")
    axs[1].set_title("Relative difference between runs", loc="left")
    axs[1].legend(fontsize=8, ncol=2)
    axs[2].plot(t[1:], D["Mtr"][1:] / minj[1:] - 1, label="double")
    axs[2].plot(t[1:], S["Mtr"][1:] / minj[1:] - 1, label="single", ls="--")
    if np.isfinite(t_leak):
        axs[2].axvline(t_leak, color="0.5", lw=1, ls=":")
        axs[2].text(t_leak, axs[2].get_ylim()[0], "  wind reaches boundary", fontsize=8, va="bottom")
    axs[2].axhline(0, color="0.7", lw=1)
    axs[2].set_ylabel("M_tracer / M_injected - 1")
    axs[2].set_xlabel("simulation time [code units]")
    axs[2].set_title("Tracer mass conservation", loc="left")
    axs[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "physics_check.png"), dpi=150)
    print(f"\nWritten to {OUTDIR}/physics_check.{{md,png}}")


if __name__ == "__main__":
    main()
