# Turbulent CGM inflow for the runaway-BH box — design notes

Saved from Claude Code discussions (2026-10-23, 2026-10-28, 2026-10-29).
Context: `problems/tcgm_subcycle`, 2D cylindrical (R, z), R = 0–4 kpc (512 cells),
z = −60–4 kpc (8192 cells), dx = 7.8 pc, v_bh = 1000 km/s, T_cgm = 1e6 K (c_s ≈ 150 km/s),
inflow through the user-defined boundary `WindZBC` at X2-end, `mask_tracer` so pure CGM does not cool.

**Goal of the turbulence:** widen the wake. At present the wake shrinks to the grid scale
(early on, and in its older parts at late times).

---

## 1. Original idea and corrections

Idea: v(x,t) = Σ_k v(k) sin(2π k·x/L + Φ(t)) − v_bh ẑ, with Φ updated by dt × random, used as
both the IC and the z boundary. Feeding a supersonic inflow from the boundary is the right
approach. What needed fixing:

- **Phases:** one Φ(t) shared by all modes is a pattern that drifts, not turbulence. Each mode
  needs its own φ_k.
- **Time update:** "+ dt × rand" adds variance ∝ dt², so the result depends on the CFL number.
  Applied to amplitudes it is an unbounded random walk. Use Ornstein–Uhlenbeck instead:
  a(t+dt) = a e^{−dt/τ} + σ √(1 − e^{−2dt/τ}) N(0,1), with τ ~ l/σ_v.
- **One advected field:** v = −v_bh ẑ + δv(R, z + v_bh t). This one field sets the IC and fills
  the ghost cells, so there is no jump at z_top. It is deterministic in t, so `WindZBC` can be
  called at every RK stage and restarts need no extra state.
- **Frozen field is enough:** gas reaches the BH about 4 Myr after it enters at z = 4 kpc, much
  less than an eddy time (~10–40 Myr). A frozen field (Taylor's hypothesis) is fine, and OU is
  unnecessary. If OU is ever used: update it once per step, not in the BC; save its state in
  the dumps; draw random numbers from one fixed seed on every rank.
- **Scales:** R_max is only 4 kpc, so λ_max ≲ 2 kpc and λ_min ≳ 16 cells ≈ 125 pc. Keep
  σ_v ≪ v_bh − c_s so the boundary stays supersonic inflow. If the k_z values lie on a grid,
  the field is periodic in z + v_bh t, and the inflow repeats over 120 Myr (≈120 kpc).

## 2. Why the field should be divergence-free

This is a modelling choice; real turbulence has both kinds of motion. A Helmholtz split gives
v = v_sol + v_comp. Random vector amplitudes put about half the energy into v_comp in 2D
(a third in 3D), and v_comp is sound waves, not eddies.

- **Consistency with uniform ρ, p:** ∂ρ/∂t = −ρ∇·v. A solenoidal field leaves ρ unchanged.
  The pressure is still out of balance, but only at the level δp/p ~ M². A compressive part
  gives δρ/ρ ~ M. Here M = σ_v/c_s ≈ 0.3–0.7, so the compressive error is the larger one, and
  it steepens into weak shocks.
- **Boundary:** a compressive part keeps varying the mass flux through z_top, so the boundary
  keeps launching waves into the box.
- **Physics:** turbulence affects Bondi–Hoyle accretion through vorticity (Krumholz, McKee &
  Klein 2005). Solenoidal driving isolates that effect. Density clumps can be added on purpose.
- The "cooling artifacts" argument does not apply, because `mask_tracer` stops CGM cooling.
- **Geometry:** in (R, z), ∇·v = ∂_R v_R + ∂_z v_z + v_R/R. A planar 2D curl misses the
  v_R/R term and leaves a compressive error of order v/(kR) near the axis. Hence the
  construction below.

## 3. Vector potential and stream function

For ∇·v = 0, write v = ∇×A. With axisymmetry, A = A_φ(R,z) φ̂ gives a purely poloidal flow:

```
v_R = −∂_z A_φ          v_z = (1/R) ∂_R (R A_φ) = ∂_R A_φ + A_φ/R
ψ = R A_φ               v_R = −(1/R) ∂_z ψ,   v_z = (1/R) ∂_R ψ
A_φ(R,z) = (1/R) ∫_0^R v_z R' dR'   = (axial flux through the disk of radius R) / 2πR   [Stokes]
```

- Any smooth A_φ gives ∇·v = 0 exactly. In practice, generate a random A_φ and take its curl.
- 2π Δψ is the volume flux between two radii, and ψ is constant along streamlines.
- Gauge: A_φ φ̂ already satisfies ∇·A = 0 (Coulomb). The only freedom left is
  A_φ → A_φ + c/R, which is singular on the axis, so regularity fixes c = 0 (ψ = 0 on the axis).
- **Axis regularity:** v_z is even in R and v_R is odd, with v_R(0) = 0. Hence
  ψ = ½ v_z(0,z) R² + O(R⁴) and **A_φ = R·h(R², z) must be odd in R**. Compare a uniform field:
  A_φ = U R/2, like the potential of a uniform B.
- **Divergence-free on the grid as well:** take ψ at cell corners. Then
  ⟨v_R⟩ = −Δ_zψ/(R Δz) on R-faces and ⟨v_z⟩ = 2Δ_Rψ/(R₂² − R₁²) on z-faces, and average the
  two faces to get cell centres. This is not critical, because Idefix stores velocities at
  cell centres.
- **Swirl:** any v_φ(R,z) with v_φ ∝ R near the axis is divergence-free on its own. It needs
  COMPONENTS = 3. Undecided.

## 4. Bessel modes

The eigenmodes come from the vector Laplacian: (∇²A)_φ = ∇²A_φ − A_φ/R² = −k² A_φ, which also
gives ω = k²A and ∇²v = −k²v. Separating A_φ = f(R) cos(k_z z + φ) gives Bessel's equation of
order 1 in x = k_R R, with k² = k_R² + k_z². The order is 1 because A_φ and v_R are components
of a vector, which behave like m = 1. J1 is odd, so it is regular on the axis; Y1 ~ 1/x is
excluded.

```
v_R = A k_z J1(k_R R) sin(k_z z + φ)     (odd)
v_z = A k_R J0(k_R R) cos(k_z z + φ)     (even, J0(0) = 1)       [d/dx(x J1) = x J0]
```

- **Shape:** each cell is a toroidal vortex ring, with ψ = 0 at the J1 zeros (k_R R = 3.83,
  7.02, …) and every π/k_z in z. The mode drives an alternating axial jet on the axis.
  Far from the axis it looks like a standing wave whose amplitude falls as (k_R R)^{−1/2}.
- **Boundary:** with k_R,n = j_{1,n}/R_max, v_R = 0 at R_max (a wall); J0 zeros would instead
  give v_z = 0 there. X1-end is `outflow`, so neither is required physically.
- **Orthogonality** (Sturm–Liouville, weight R): ∫J1(k_nR)J1(k_mR) R dR = (R_max²/2) J0(j_{1,n})² δ_nm,
  and J0 obeys the same relation. Energy therefore splits by mode as ∝ Σ|A_n|² k_n² N_n.
  Modes are spaced roughly uniformly in (k_R, k_z), so a k-shell holds ∝ k dk modes.

**Why independent random amplitudes don't give uniform turbulence:** with random phases,
⟨v_z²⟩(R) = ½ Σ|A_n|² k_n² J0(k_nR)² ≈ ½ Σ|A_n|² k_n² min(1, 1/(π k_n R)). So σ_v is flat for
R < 1/k_max and falls as **R^{−1/2}** beyond 1/k_min. A numerical check with 60 equal-weight
modes gave ⟨v_z²⟩ = 1.00, 0.52, 0.26, 0.13 at R/R_max = 0.1, 0.2, 0.4, 0.8. Every term is
non-increasing in R, so **no choice of |A_n| can flatten it**; that needs correlated
amplitudes. Equivalently, the completeness relation Σφ_n(R)φ_n(R') = δ(R−R')/R means white
noise in this basis has variance ∝ 1/R, where the R comes from the volume element R dR.

**Another reading** (not yet checked numerically): from Jacobi–Anger, the axisymmetric part of
a 3D plane wave is J0(k_⊥R), i.e. the plane wave averaged around a ring. A ring of radius
R ≫ 1/k averages over ~kR independent eddies. Independent-amplitude Bessel modes may
therefore reproduce 3D turbulence averaged around the axis: the full 3D σ on the axis and
weaker fluctuations further out. The choice is then (a) uniform σ in the (R, z) plane versus
(b) 3D turbulence averaged around the axis.

Practical note: `std::cyl_bessel_j` only runs on the host, but the R-profiles do not depend on
time, so tabulate them once per rank (n_modes × N_R) and copy to the device.

## 5. Plane-wave A_φ (for uniform σ)

**First attempt, which was wrong:** A_φ = W(R) Σ a_k sin(k_R R + k_z z + φ_k) with
W = 1 − e^{−R²/R0²}. For R0 < R < 1/k, A_φ ≈ const, so v_z ≈ A_φ/R ∝ 1/R, which peaks at
σ/(kR0) on the axis. Numerically, σ_vz was 12.6× its far-field value at R ≈ 31 pc and 2.6× at
125 pc. It also violates regularity (A_φ ∝ R² is even), giving v_R ∝ R² and v_z → 0 on the axis.

**Fix:** mirror the field so it is odd in R, A_φ = [G(R,z) − G(−R,z)]/√2, which for plane
waves is

```
A_φ = Σ 2a_k sin(k_R R) cos(k_z z + φ_k)
v_R = Σ 2a_k k_z sin(k_R R) sin(k_z z + φ_k)
v_z = Σ 2a_k [k_R cos(k_R R) + sin(k_R R)/R] cos(k_z z + φ_k)
```

This needs no R0, is regular (A_φ/R stays finite), and σ is uniform for R ≳ 0.5 kpc. Within
about 1/k_min of the axis, σ_vz rises to about 2.5× and σ_vR drops to 0. That appears to be
built into axisymmetric flow that conserves mass.

**Spectrum:** for E(k) ∝ k^{−5/3} in 2D with modes spread uniformly in k-space,
|a_k|² ∝ k^{−2} · k^{−5/3} / k. The k^{−2} is from the curl and the 1/k from shell density.
For log-uniform sampling of k (as in `turb_box`) this becomes a ∝ k^{−4/3}. Normalise σ_v
numerically, near the axis.

## 6. Will turbulence widen the wake?

- In 3D a thin wake widens mainly by **meandering** (m = 1), which needs v_x, v_y ≠ 0 on the
  axis. In axisymmetry, **v_R = 0 on the axis for any field**. Near the axis every field
  reduces to v_R ≈ s(z) R and v_z ≈ v_z0(z), with s = −½ ∂_z v_z0.
- So 2D turbulence can only **swell or pinch** the wake (a sausage mode). Wake material
  collects where s > 0 and is stretched thin where s < 0. Expect a **string of beads**
  ~λ/2 apart, not a uniformly wider wake. The e-folding time is ~1/s ~ 4–5 Myr for
  σ = 75 km/s and λ = 1 kpc.
- Near the axis, Bessel and mirror give much the same field. The R^{−1/2} falloff is in gas
  the wake never reaches, so **the Bessel recipe is acceptable for this goal**. What matters is
  normalising σ_v near the axis, not by a box average.
- **First check whether the thin wake is physical.** Only wake material cools (`mask_tracer`).
  Cooling from 10⁶ to 10⁴ K at fixed P makes it ~100× denser and ~10× narrower, which could
  genuinely be sub-grid; turbulent mixing could even make the cold wake grow (Gronke & Oh).
  It could also be the axis focusing artifact of 2D cylindrical runs. Diagnostics:
  (1) compare the width of the tracer with the width of the cold gas; (2) run at 2× resolution.

## 7. Passive-scalar test: `problems/turb_box`

The setup without the BH: no gravity, wind or cooling; uniform CGM at −v_bh; R = 0–4 kpc,
z = −28–4 kpc (512 × 4096, same dx), COMPONENTS = 3 (v_φ = 0), tstop = 40.
**Must be 3:** in Idefix CYLINDRICAL the p/R source term is only added for COMPONENTS == 3
(`src/fluid/addSourceTerms.hpp`). With 2, uniform pressure drives a spurious ~200 km/s
inflow onto the axis and the density there piles up by 10³–10⁵. A tracer is set to 1 inside
r < r_src = 20 pc at the origin, and the marked streak is the "wake" (age = −z/v_bh).

- `[Turbulence] recipe = none | bessel | mirror`, plus `sigma_v` (0.075 = 75 km/s, rms of
  (v_R, v_z) over R < `R_norm` = 0.5 kpc), `lmin`/`lmax` (0.125/2 kpc), `nmodes` (200), `seed`.
- Modes: |k| log-uniform, random direction, a ∝ k^{−4/3} N(0,1). For Bessel, k_R is snapped to
  the nearest j_{1,n}/R_max. The profiles F(n,i) and G(n,i) are tabulated per rank;
  `FillCGM` sets both the IC and the z_max ghost cells, at z + v_bh t.
- `wake_width.py run_a/data.XXXX.vtk run_b/... --label a b` plots tracer-weighted r_rms and
  r_90 against age and prints the fraction of the wake with r_rms > 4 cells.
- **Smoke test** (200 cycles, all three recipes run cleanly). σ_v in km/s at t = 0:

| R (pc)  | 4  | 66 | 254 | 504 | 1004 | 2004 | 3910 |
|---------|----|----|-----|-----|------|------|------|
| bessel  | 85 | 79 | 66  | 53  | 48   | 41   | 18   |
| mirror  | 89 | 81 | 66  | 51  | 56   | 58   | 54   |

  The discrete |∇·v|/|∇×v| is about 0.003. A full run is about 20 min on the RTX 4000
  when the GPU is shared.

**Next:** run none/bessel/mirror to t = 40 and compare wake widths. In parallel, run the
cold-gas versus tracer and 2× resolution diagnostics on `tcgm_subcycle`. Then decide between
(a) and (b) and whether to add swirl.
