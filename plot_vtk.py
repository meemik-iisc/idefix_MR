#!/usr/bin/env python3
"""
Plot number density and temperature from RunawayBH VTK outputs.

Usage:
    python plot_vtk.py <data.XXXX.vtk> [more.vtk ...] [--zoom L] [--show]

Units and mu are read from idefix_run.ini next to the VTK file (written by
run_file.sh). One PNG is saved per input file, next to it.
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pytools"))
from vtk_io import readVTK  # noqa: E402

M_P = 1.67262192369e-24   # g
K_B = 1.380649e-16        # erg/K
KYR = 3.15576e10          # s


def read_ini(path):
    """Minimal reader for the [Units] and [Setup] entries we need."""
    units = {"length": 1.0, "velocity": 1.0, "density": 1.0}
    mu = 0.6
    block = None
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            if line.startswith("["):
                block = line.strip("[]")
                continue
            key, *vals = line.split()
            if block == "Units" and key in units:
                units[key] = float(vals[0])
            elif block == "Setup" and key == "mu":
                mu = float(vals[0])
    return units, mu


def mirror(R, z, f):
    """Reflect an (nR, nz) field about the axis to show the full R-z plane."""
    R_full = np.concatenate([-R[::-1], R])
    f_full = np.concatenate([f[::-1, :], f], axis=0)
    return R_full, z, f_full


def plot_file(fname, zoom=None, show=False):
    ini = os.path.join(os.path.dirname(os.path.abspath(fname)), "idefix_run.ini")
    if not os.path.isfile(ini):
        ini = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "problems", "RunawayBH", "idefix.ini")
    units, mu = read_ini(ini)
    rho0, v0, L0 = units["density"], units["velocity"], units["length"]

    d = readVTK(fname)
    R, z = d.x, d.y                               # code length (pc)
    rho = d.data["RHO"][:, :, 0]
    prs = d.data["PRS"][:, :, 0]

    # rho_code = mu m_p n / rho0  ->  n [cm^-3]
    n = rho * rho0 / (mu * M_P)
    # P_code = n k T / (rho0 v0^2)  ->  T = (P/rho) mu m_p v0^2 / k
    T = (prs / rho) * mu * M_P * v0**2 / K_B

    t_kyr = float(np.atleast_1d(d.t)[0]) * (L0 / v0) / KYR

    # size the figure to the plotted region: two panels of width 2*Rmax, height zmax-zmin
    width, height = (2*zoom, 2*zoom) if zoom else (2*R.max(), z.max() - z.min())
    panel_h = 9.0 if height > width else 5.0
    fig, axes = plt.subplots(1, 2, sharey=True, constrained_layout=True,
                             figsize=(2*panel_h*width/height + 3.0, panel_h + 1.0))
    panels = [(n, r"$n$ [cm$^{-3}$]", "viridis"),
              (T, r"$T$ [K]", "inferno")]
    for ax, (field, label, cmap) in zip(axes, panels):
        Rf, zf, ff = mirror(R, z, field)
        im = ax.pcolormesh(Rf, zf, ff.T, cmap=cmap, shading="nearest",
                           norm=LogNorm(vmin=ff.min(), vmax=ff.max()),
                           rasterized=True)
        fig.colorbar(im, ax=ax, label=label, shrink=0.8)
        ax.set_aspect("equal")
        ax.set_xlabel("R [pc]")
        ax.plot(0, 0, marker="+", color="white", ms=8, mew=1.5)
        if zoom:
            ax.set_xlim(-zoom, zoom)
            ax.set_ylim(-zoom, zoom)
    axes[0].set_ylabel("z [pc]")
    fig.suptitle(f"{os.path.basename(fname)}   t = {t_kyr:.2f} kyr")

    suffix = f"_zoom{zoom:g}" if zoom else ""
    out = os.path.splitext(fname)[0] + suffix + ".png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="VTK file(s) to plot")
    p.add_argument("--zoom", type=float, default=None,
                   help="half-width of a square window around the BH, in pc")
    p.add_argument("--show", action="store_true", help="open an interactive window")
    args = p.parse_args()
    for f in args.files:
        plot_file(f, zoom=args.zoom, show=args.show)


if __name__ == "__main__":
    main()
