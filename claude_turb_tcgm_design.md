# turb_tcgm: grid, injection, softening and wind-parameter design notes

Notes for `problems/turb_tcgm` (2D cylindrical, BH frame, CGM flowing in from +z at -v_bh).
Analytical reference: `/data/data_meemik/RunawayBH/AnalyticalSol/Mdot_params/Mdot_vs_L.py`.

Units: L0 = 1 kpc, v0 = 1000 km/s, so 1 code time = 0.978 Myr and L [kpc] = v_bh t = t [code].

---

## 1. Standoff radius and its time dependence

Ram-pressure balance:

    R_so^2 = Mdot v_w / (4 pi rho_cgm v_bh^2)

With exponentials `X = X0 exp(-t/tau_X)`, R_so^2 ∝ exp[(1/tau_n - 1/tau_Mdot - 1/tau_vw) t].
R_so grows only if `1/tau_n > 1/tau_Mdot + 1/tau_vw` (rates add for the product Mdot*v_w).

- tau_Mdot = 25, tau_vw = 15 → need tau_n < 9.4.
- `Mdot_vs_L.py` effectively used n = 10 e^{-L/5} (two e^{-L/10} factors), so R_so grows
  12.5 pc → ~200 pc. The sim had `dt_ncgm = 10` → R_so slowly *shrinks* (12 → 10 pc).
  **Set `dt_ncgm = 5.0` to match the script** (or adopt the new model in §5).
- Code vs ini mismatch to remember: ini comment says Mdot ∝ exp(+t/dt_mdot), code uses exp(-t/dt_mdot).

When R_so grows, **t = 0 is the tightest point**: R_so(0) ≈ 12 pc.

## 2. Injection radius

Requirement: R_so > r_inj at all times, with margin (R_so ≳ 2 r_inj so the wind is free-streaming
before the reverse shock), and r_inj resolved by ≥ 4–5 cells.

- r_inj = 5 pc (`rinj 0.005`), core cell size 1 pc.
- `Volinj = 4/3 pi r_inj^3` is analytic; the stair-stepped 5-cell sphere differs by ~10–20%,
  so injected Mdot ≠ Mdot0 exactly. Fix if needed: sum the actual cell volumes with r < r_inj at setup.

## 3. Grid

Idefix syntax: `Xn-grid  nPatches  x0  N1 type1 x1  N2 type2 x2 ...`

- `u`  uniform
- `s-` stretched, reference Δx from the uniform patch on its **left**, cells grow to the right
- `s+` stretched, reference Δx from the uniform patch on its **right**, cells grow to the left

A stretched patch must border a uniform patch; Idefix solves for the ratio q itself.

```ini
[Grid]
X1-grid    3   0.0     50 u 0.05   135 s- 0.5   448 u 4.0
X2-grid    5  -60.0  7616 u -0.5   135 s+ -0.05  100 u 0.05  135 s- 0.5  448 u 4.0
```

| Region | Range (kpc) | Cells | Δ |
|---|---|---|---|
| Core | R < 0.05, \|z\| < 0.05 | 50 × 100 | 1 pc |
| Stretch | 0.05 → 0.5 | 135 | 1 → 7.8 pc, q ≈ 1.0155 |
| Far field | rest | 448 (R), 7616 + 448 (z) | 1/128 kpc = 7.8 pc |

- 135 stretched cells chosen so the last stretched cell ≈ 7.8 pc (no jump into the uniform patch).
  If the core or far-field Δ changes, recompute N (script: Newton solve of
  `(q^{N+1}-q)/(q-1) = (x_end - x_start)/Δx0`, same as `src/gridHost.cpp`).
- Far field keeps `lmin = 0.125` kpc at 16 cells; the Bessel turbulence uses `data.x`, so it works on
  a non-uniform grid unchanged.
- Total ≈ 633 × 8434 ≈ 5.3 M cells: fine on one GPU. Any nx1 × nx2 is fine for GPU/Kokkos.
- MPI on several GPUs: decompose along z only, e.g. `-dec 1 N`.
- Timestep set by the 1 pc cells and the 5000 km/s wind: dt ≈ 4e-5 → ~1.5 M steps to t = 60.
  Test with a short `tstop` first.

**Cell aspect ratio.** ΔR depends only on R and Δz only on z, so cells are square in the core,
the far field and the stretch-zone diagonal, but not in two strips:

- |z| < 50 pc, R > 50 pc: flat cells, up to 7.8 : 1
- R < 50 pc, |z| > 50 pc (the whole wake axis): tall cells, up to 1 : 7.8

Fine for stability and GPU; mildly anisotropic dissipation. Never coarser than the original uniform
grid. Shrinking the core (e.g. 30 pc, still ≫ R_so(0)) reduces the strips.

## 4. Softening length

Bondi–Hoyle radius with M = 2e7 Msun, v_bh = 1000 km/s (c_s ≈ 150 km/s at 1e6 K, negligible):

| Definition | R_B | 0.1 R_B (kpc) |
|---|---|---|
| 2GM / v_bh^2 | 0.172 pc | 1.72e-5 |
| 2GM / (v_bh^2 + c_s^2) | 0.168 pc | 1.68e-5 |
| GM / v_bh^2 | 0.086 pc | 0.86e-5 |

- `epsilon 1.7e-5` gives 0.1 R_B. The old ini comment (10 pc = 0.1 R_B) implied v ≈ 40 km/s; outdated.
- ε ≪ Δ = 1 pc: nearest cell centre is r ≈ 0.7 pc ≈ 40 ε, so ε has no effect; the grid softens at ~1 pc.
  Bondi accretion is not resolved (would need Δ ~ 0.01 pc).
- Gravity is weak where resolved: v_esc ≈ 190 km/s at 5 pc, ≈ 120 km/s at 12 pc (vs 5000 km/s wind,
  1000 km/s inflow).
- Alternative: ε ≈ 1–2 cells (1–2 pc), the usual choice when R_B is unresolved. Practically the same physics.

## 5. Wind parameters: 1 kpc wide wake, M_tot < 1e7 Msun, cooling to 1e4 K

**Wake width.** Shell at 90° sits at √3 R0 (Wilkin 1996) → full width ≈ 2√3 R0 ≈ 3.5 R0.
1 kpc full width → R0 ≈ 0.29 kpc.

**Mass budget** depends only on Mdot; R0 depends on momentum flux Mdot·v_w.

**Cooling constraint.** Eliminating n with the R0 relation:

    t_cool,sw = 9 mu m_p v_w^4 / (128 n v_bh^2 Lambda)
              = (9 pi / 32) mu^2 m_p^2 v_w^3 R0^2 / (Mdot Lambda(T_sw))

At fixed wake size, t_cool depends only on v_w^3 / Mdot, not on n_cgm or v_bh.
→ late-time wind must be slow, Mdot reasonably large; low n_cgm is what makes the wake wide.

t_cool,sw [Myr] / n_cgm needed [cm^-3] at R0 = 0.29 kpc (μ = 1, cooltable.dat):

| v_w (km/s) | T_sw | Mdot = 0.03 | Mdot = 0.1 | Mdot = 0.3 |
|---|---|---|---|---|
| 200 | 9e5 K | 1.3 / 2e-4 | 0.4 / 8e-4 | 0.13 / 2e-3 |
| 300 | 2e6 K | 9 / 3e-4 | 2.8 / 1e-3 | 0.9 / 3e-3 |
| 500 | 6e6 K | 130 / 6e-4 | 40 / 2e-3 | 13 / 6e-3 |
| 1000 | 2e7 K | 1600 / 1e-3 | 480 / 4e-3 | 160 / 1e-2 |

Why the old models fail:

- Mdot 0.1 e^{-L/25}, v_w 5000 e^{-L/15}, n 10 e^{-L/5}: R0(60) ≈ 0.2 kpc → width only ~0.7 kpc.
- Larger `dt_vw` widens the wake for free in mass, but leaves v_w ~ 1000 km/s late → t_cool ~ 1e2–1e5 Myr.
- Pure exponential n keeps dropping (6e-5 at L = 60) while Mdot decays → v_w^3/Mdot large.

**Recommended model** (floors instead of pure exponentials):

    n_cgm(L) = max(10 e^{-L/5}, 1e-3)
    v_w(L)   = v_f + (5000 - v_f) e^{-L/10}   km/s
    Mdot     = const

Design B: Mdot = 0.15 Msun/yr, v_f = 250 km/s → M_tot = 9e6 Msun

| L | n_cgm | v_w | R0 | width | t_cool,sw |
|---|---|---|---|---|---|
| 0 | 10 | 5000 | 15 pc | 0.05 kpc | 32 Myr |
| 20 | 0.18 | 890 | 48 pc | 0.17 kpc | 5.9 Myr |
| 30 | 0.025 | 490 | 96 pc | 0.33 kpc | 2.7 Myr |
| 40 | 3e-3 | 340 | 218 pc | 0.76 kpc | 2.5 Myr |
| 45 | 1.2e-3 | 300 | 341 pc | 1.18 kpc | 2.8 Myr |
| 60 | 1e-3 | 260 | 352 pc | 1.22 kpc | 1.2 Myr |

Design A: Mdot = 0.1, v_f = 300 → M_tot = 6e6, width ≈ 1.1 kpc, t_cool,sw 4–10 Myr.

Open choices / checks:

- Which t_cool target? Cool before leaving the box (wake flow time, tens of Myr): B passes easily.
  Cool within the shell near the BH (R0 / v_bh ≈ 0.3 Myr): needs v_w ≈ 200 km/s or Mdot ≈ 0.3,
  the latter only affordable if high Mdot is restricted to late times.
- Sim must be changed to match: `N_CGM`, `Vw_t`, `Mdot_t` are pure exponentials in `setup.cpp`;
  add floors and ini parameters for them.
- μ: script uses 1, sim 0.6. t_cool ∝ μ^2 → sim ≈ 2.8× shorter. R0 in sim ≈ 1.09× script
  (μ and v_inj = v_w/√2). Set μ = 0.6 in the script for consistency.
- Box: half-width √3 R0 ≈ 0.6 kpc ≪ R_max = 4 kpc. R_so(0) ≈ 12–15 pc, so the 1 pc core and
  r_inj = 5 pc from §2–3 still hold.
