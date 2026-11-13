#!/usr/bin/env python3
"""
Total mass and total energy (kinetic + thermal) vs time for the single and
double precision runs, with the relative difference |single - double| / double (log y) below.
Single is drawn as diamonds, double as solid lines.
Uses the domain totals from physics_check.py.

Usage (from this directory):  python3 plot_mass_energy.py
Writes diagnostics_sequential/mass_energy.png
"""
import os
import matplotlib.pyplot as plt
import physics_check as pc

D = pc.load(pc.RUNS["double"])
S = pc.load(pc.RUNS["single"])
t = D["t"]

COL = {"M": "#2a78d6", "E": "#eb6834"}
LAB = {"M": "total mass", "E": "total energy"}

fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
for k in ("M", "E"):
    a1.plot(t, D[k], color=COL[k], lw=2, label=f"{LAB[k]}, double")
    a1.plot(t, S[k], color=COL[k], ls="none", marker="D", ms=4, mfc="white", mew=1.2,
            label=f"{LAB[k]}, single")
    a2.plot(t, abs(S[k] - D[k]) / D[k], color=COL[k], marker="o", ms=3, label=LAB[k])

a1.set_ylabel("domain total [code units]")
a1.set_title("Total mass and energy: single vs double precision", loc="left")
a1.legend()
a1.grid(alpha=0.3)

a2.set_yscale("log")
a2.set_ylabel("|single - double| / double")
a2.set_xlabel("simulation time [code units]")
a2.legend()
a2.grid(alpha=0.3, which="both")

fig.tight_layout()
out = os.path.join(pc.OUTDIR, "mass_energy.png")
fig.savefig(out, dpi=150)
print(f"Written to {out}")
