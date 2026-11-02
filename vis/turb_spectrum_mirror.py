#!/usr/bin/env python3
"""
Kinetic-energy spectrum E(k) of the poloidal turbulence in turb_box_mirror VTK outputs.

Usage:
    python turb_spectrum_mirror.py [run_dir] [--tmin T] [--tmax T] [--zmin Z] [--zmax Z]
                                   [--nbins N] [--out file.png] [--show]

Every data.*.vtk in run_dir is plotted (thin lines, coloured by time), plus the
time average of E(k) over the outputs with tmin <= t <= tmax [Myr] (thick line).
If run_dir is not given, it is asked for.

The velocity perturbation (v_R, v_z + v_bh) is projected on the mirror modes the
setup uses, with k_R = pi m/R_max: v_R on s1 = sin(k_R R), and v_z on
c0 = sin(x)/x + cos(x), x = k_R R, times a Fourier series in z (Hann window over
[zmin, zmax]). The c0 are not orthogonal, so v_z is projected through the
stream function: v_z = (1/R) d(R A)/dR gives A = (1/R) int_0^R R' v_z dR', and A
goes exactly on sin(k_R R) under dR. Moving the R' integral onto the basis,

    <A sin(k_R R)>_R = <R v_z [Si(pi m) - Si(k_R R)]>_R ,

so no cumulative sum is needed. The k_R = 0 part of v_z (volume average over R
at each z) is taken out first and kept as its own mode. Each mode's energy is
its own area-averaged <v^2> (dR dz weight); the c0 overlap a little, so
sum_k E(k) dk is close to, but not exactly, <dv_R^2 + dv_z^2>.

The spectrum the setup intends (same lattice and k^{-7/3} amplitudes as BuildTurb)
is drawn for comparison, with a k^{-5/3} line. BuildTurb normalises sigv over
R < Rnorm on the grid, which has no closed form here, so the intended curve is
scaled to the measured mode energy of the earliest output. The table also prints
the setup's sigma (volume average over R < Rnorm). Parameters are read from
idefix_run.ini in run_dir. k is in 1/kpc, E in (km/s)^2 kpc.
"""
import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sici

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pytools"))
from vtk_io import readVTK  # noqa: E402

MYR = 3.15576e13          # s


def read_ini(path):
    """Minimal reader for the [Units], [Setup] and [Turb] entries we need."""
    want = {"Units": ["length", "velocity"], "Setup": ["vbh"],
            "Turb": ["sigv", "Rnorm", "lmin", "lmax", "Lz"]}
    par = {}
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
            if key in want.get(block, []):
                par[key] = float(vals[0])
    return par


def mirror_basis(R, Rmax):
    """k_R = pi m/R_max (m = 1..nR), the s1 and c0 profiles, and the dual basis for v_z."""
    nR = R.size
    kR = np.pi*np.arange(1, nR + 1)/Rmax
    x = np.outer(kR, R)                      # (nR, nR)
    s1 = np.sin(x)
    c0 = np.sin(x)/x + np.cos(x)
    G = R*(sici(np.pi*np.arange(1, nR + 1))[0][:, None] - sici(x)[0])
    norm = np.mean(s1**2, axis=1)            # 1/2, except 1 for m = nR
    return kR, s1, c0, G, norm


def measured_modes(fname, zmin, zmax, v_kms, vbh_kms, Rnorm):
    """Return k, e (energy of each (k_R, k_z) mode), sigma^2 (area), sigma^2 (setup) and time."""
    d = readVTK(fname)
    R, z = d.x, d.y
    vR = d.data["VX1"][:, :, 0] * v_kms
    vz = d.data["VX2"][:, :, 0] * v_kms

    sel = (z >= zmin) & (z <= zmax)
    z, vR, vz = z[sel], vR[:, sel], vz[:, sel]
    nR, nz = vR.shape
    dR, dz = R[1] - R[0], z[1] - z[0]
    Rmax = R[-1] + 0.5*dR

    # sigma as the setup defines it: v_z + v_bh, volume average over R < Rnorm
    w_V = R*(R < Rnorm)
    w_V = w_V / w_V.sum()
    sig2_setup = np.sum(w_V[:, None]*(vR**2 + (vz + vbh_kms)**2)) / nz

    sig2 = np.mean(vR**2 + (vz - vz.mean())**2)

    # k_R = 0 part of v_z (volume average at each z); what is left has int R v_z dR = 0,
    # so its stream function vanishes at R_max like the sin modes
    w_R = R*dR / (0.5*Rmax**2)
    bz0 = w_R @ vz
    vz = vz - bz0[None, :]
    bz0 = bz0 - bz0.mean()                   # drop the mean inflow (-v_bh)

    # radial projection: v_R = sum s1 b_R, v_z = sum c0 b_z, with b_z = k_R <A s1>/<s1^2>
    kR, s1, c0, G, norm = mirror_basis(R, Rmax)
    bR = s1 @ vR / nR / norm[:, None]        # (nR, nz)
    bz = kR[:, None] * (G @ vz) / nR / norm[:, None]

    # z Fourier series, Hann window, one-sided power
    win = np.hanning(nz)
    cw = np.mean(win**2)
    kz = 2*np.pi*np.fft.rfftfreq(nz, d=dz)
    one_sided = np.full(kz.size, 2.0)
    one_sided[0] = 1.0
    if nz % 2 == 0:
        one_sided[-1] = 1.0

    def power(b):
        return one_sided*np.abs(np.fft.rfft(b*win, axis=-1)/nz)**2 / cw

    # energy of each mode: <s1^2> |b_R|^2 + <c0^2> |b_z|^2, area averaged
    e = power(bR)*norm[:, None] + power(bz)*np.mean(c0**2, axis=1)[:, None]   # (nR, nkz)
    k = np.sqrt(kR[:, None]**2 + kz[None, :]**2)
    k = np.concatenate([k.ravel(), kz])
    e = np.concatenate([e.ravel(), power(bz0)])
    return k, e, sig2, sig2_setup, float(np.atleast_1d(d.t)[0]), R


def intended_modes(par, R, etot):
    """Mode energies of the lattice built by BuildTurb (setup.cpp), scaled to total etot."""
    Rmax = R[-1] + 0.5*(R[1] - R[0])
    kmin, kmax = 2*np.pi/par["lmax"], 2*np.pi/par["lmin"]
    M = int(kmax*Rmax/np.pi)
    kR = np.pi*np.arange(1, M + 1)/Rmax
    N = int(kmax*par["Lz"]/(2*np.pi))
    kz = 2*np.pi*np.arange(1, N + 1)/par["Lz"]
    k = np.sqrt(kR[:, None]**2 + kz[None, :]**2)
    # v_R = a kz s1 sin(th), v_z = a kR c0 cos(th): <v^2> = a^2 (kz^2 <s1^2> + kR^2 <c0^2>) / 2
    x = np.outer(kR, R)
    s1sq = np.mean(np.sin(x)**2, axis=1)[:, None]
    c0sq = np.mean((np.sin(x)/x + np.cos(x))**2, axis=1)[:, None]
    e = 0.5*k**(-14.0/3.0)*(kz[None, :]**2*s1sq + kR[:, None]**2*c0sq)
    e[(k < kmin) | (k > kmax)] = 0.0
    e *= etot / e.sum()
    return k.ravel(), e.ravel()


def bin_spectrum(k, e, edges):
    """E(k) = (energy in shell) / (shell width)."""
    E, _ = np.histogram(k, bins=edges, weights=e)
    return E / np.diff(edges)


def slope(kc, E, kmin, kmax):
    m = (kc > kmin) & (kc < kmax) & (E > 0)
    return np.polyfit(np.log(kc[m]), np.log(E[m]), 1)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", nargs="?", default=None,
                   help="run folder with data.*.vtk and idefix_run.ini (asked if omitted)")
    p.add_argument("--tmin", type=float, default=-np.inf, help="start of the time average [Myr]")
    p.add_argument("--tmax", type=float, default=np.inf, help="end of the time average [Myr]")
    p.add_argument("--zmin", type=float, default=-np.inf, help="lower z of the window [kpc]")
    p.add_argument("--zmax", type=float, default=np.inf, help="upper z of the window [kpc]")
    p.add_argument("--nbins", type=int, default=40, help="log-spaced bins in k")
    p.add_argument("--out", default=None, help="output PNG (default: plots/ next to the data)")
    p.add_argument("--show", action="store_true", help="open an interactive window")
    args = p.parse_args()

    run_dir = args.run_dir or input("Run folder: ").strip()
    run_dir = os.path.abspath(os.path.expanduser(run_dir))
    files = sorted(glob.glob(os.path.join(run_dir, "data.*.vtk")))
    if not files:
        sys.exit(f"no data.*.vtk in {run_dir}")
    par = read_ini(os.path.join(run_dir, "idefix_run.ini"))
    v_kms = par["velocity"] / 1e5
    t_myr = par["length"] / par["velocity"] / MYR
    kmin, kmax = 2*np.pi/par["lmax"], 2*np.pi/par["lmin"]

    # bins are fixed by the first file (all outputs share the grid)
    edges = None
    curves = []
    print(f"{'file':<20} {'t [Myr]':>8} {'sigma [km/s]':>13} {'sig(R<Rn)':>10} {'slope':>7}")
    for f in files:
        k, e, sig2, sig2_setup, t, R = measured_modes(f, args.zmin, args.zmax, v_kms,
                                                      par["vbh"]*v_kms, par["Rnorm"])
        if edges is None:
            edges = np.logspace(np.log10(0.5*kmin/4), np.log10(k.max()), args.nbins + 1)
            kc = np.sqrt(edges[1:]*edges[:-1])
        E = bin_spectrum(k, e, edges)
        curves.append((t*t_myr, E, sig2, sig2_setup, e.sum()))
        print(f"{os.path.basename(f):<20} {t*t_myr:8.2f} {np.sqrt(sig2):13.2f} "
              f"{np.sqrt(sig2_setup):10.2f} {slope(kc, E, kmin, kmax):7.2f}")

    times = np.array([c[0] for c in curves])
    i0 = np.argmin(times)
    ki, ei = intended_modes(par, R, curves[i0][4])
    Ei = bin_spectrum(ki, ei, edges)

    fig, (ax, axc) = plt.subplots(2, 1, sharex=True, figsize=(7, 8),
                                  gridspec_kw={"height_ratios": [2, 1]},
                                  constrained_layout=True)
    good = Ei > 0
    ax.loglog(kc[good], Ei[good], "k--", lw=1.5,
              label=f"intended, scaled to t = {times[i0]:.1f} Myr "
                    f"(slope {slope(kc, Ei, kmin, kmax):.2f})")
    axc.semilogx(kc[good], (kc**(5/3)*Ei)[good], "k--", lw=1.5)

    # k^{-5/3} through the intended spectrum at the geometric middle of the range
    kmid = np.sqrt(kmin*kmax)
    Emid = np.interp(np.log(kmid), np.log(kc[good]), np.log(Ei[good]))
    kref = np.array([kmin, kmax])
    ax.loglog(kref, np.exp(Emid)*(kref/kmid)**(-5/3), color="tab:gray", lw=3, alpha=0.4,
              label=r"$k^{-5/3}$")

    # every output, coloured by time
    cnorm = plt.Normalize(times.min(), times.max())
    cmap = plt.cm.viridis
    for t, E, *_ in curves:
        m = E > 0
        ax.loglog(kc[m], E[m], color=cmap(cnorm(t)), lw=0.6, alpha=0.5)
        axc.semilogx(kc[m], (kc**(5/3)*E)[m], color=cmap(cnorm(t)), lw=0.6, alpha=0.5)
    sm = plt.cm.ScalarMappable(norm=cnorm, cmap=cmap)
    fig.colorbar(sm, ax=[ax, axc], label="t [Myr]", shrink=0.6)

    # time average over [tmin, tmax]
    use = (times >= args.tmin) & (times <= args.tmax)
    if not use.any():
        sys.exit(f"no outputs with {args.tmin} <= t <= {args.tmax} Myr")
    Eavg = np.mean([curves[i][1] for i in np.flatnonzero(use)], axis=0)
    sig_avg = np.sqrt(np.mean([curves[i][2] for i in np.flatnonzero(use)]))
    sig_setup_avg = np.sqrt(np.mean([curves[i][3] for i in np.flatnonzero(use)]))
    s_avg = slope(kc, Eavg, kmin, kmax)
    m = Eavg > 0
    ax.loglog(kc[m], Eavg[m], color="tab:orange", lw=2.5,
              label=f"mean over {times[use].min():.1f}-{times[use].max():.1f} Myr "
                    f"({use.sum()} outputs, slope {s_avg:.2f})")
    axc.semilogx(kc[m], (kc**(5/3)*Eavg)[m], color="tab:orange", lw=2.5)

    print(f"time average: {use.sum()} outputs, {times[use].min():.2f}-{times[use].max():.2f} Myr, "
          f"sigma = {sig_avg:.2f} km/s (area), {sig_setup_avg:.2f} km/s (R < Rnorm), "
          f"slope = {s_avg:.2f}")
    print(f"setup sigv = {par['sigv']:.2f} km/s over R < {par['Rnorm']:g} kpc, slope fitted over "
          f"{kmin:.2f} < k < {kmax:.2f} 1/kpc")

    for a in (ax, axc):
        for kk, lab in ((kmin, r"$2\pi/l_{max}$"), (kmax, r"$2\pi/l_{min}$")):
            a.axvline(kk, color="tab:red", ls=":", lw=1)
        a.grid(True, which="both", alpha=0.2)
    ax.set_ylim(bottom=1e-6*np.nanmax(Ei))
    ax.set_ylabel(r"$E(k)$ [(km/s)$^2$ kpc]")
    axc.set_ylabel(r"$k^{5/3}E(k)$")
    axc.set_xlabel(r"$k = \sqrt{k_R^2 + k_z^2}$, $k_R = \pi m/R_{max}$ [kpc$^{-1}$]")
    zlo = max(args.zmin, -1e30)
    window = "full box" if np.isinf(args.zmin) and np.isinf(args.zmax) \
        else f"{zlo:g} < z < {args.zmax:g} kpc"
    ax.set_title(f"{os.path.basename(run_dir)}   ({window}, mirror modes)")
    ax.legend(fontsize=8)

    out = args.out or os.path.join(run_dir, "plots", "turb_spectrum_mirror.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
