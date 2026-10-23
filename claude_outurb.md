# Turbulent CGM inflow for the runaway-BH box — design notes

Saved from a Claude Code discussion (2026-10-23), for future reference.
Context: `problems/tcgm_subcycle` — 2D cylindrical (R, z), 512 x 8192, v_bh = 1000 km/s,
T_cgm = 1e6 K, user-defined boundary `WindZBC` at X2-end.

## Question

> I want to add turbulence in velocity in my box, what I was thinking was something like
> v_k(x,t) = \sum_k v(k) sin((2\pi k.x/L) + Phi(t)) + (-v_bh) \hat{z}
> such that v_k(x,t+\delta t) = v_k(x,t) + \delta t * some random number.
> This will be both the initial condition and the driving force through the z boundary
> condition, what do you think?

## Answer

Overall the plan is sound: a supersonic inflow boundary carrying turbulence, consistent with
the initial field, is a common way to put a moving object into a turbulent medium. A few
things should change before implementing it.

### 1. Fix the time update

**Use a separate phase for each mode.** A single Phi(t) shared by all modes keeps them in
lockstep, so the field is a fixed pattern that drifts rather than turbulence. Each mode needs
its own phase phi_k(t).

**`x(t+dt) = x(t) + dt*rand` is the wrong update.** Each step adds variance ∝ dt², so over a
fixed physical time the total change goes to zero as dt shrinks. That makes the decorrelation
depend on the CFL number and resolution. Applied to the amplitudes, it's also an unbounded
random walk, so the energy drifts without limit.

Use an Ornstein–Uhlenbeck process (Eswaran & Pope 1988) instead:

```
a_k(t+dt) = a_k(t)*exp(-dt/tau) + sigma_k*sqrt(1 - exp(-2 dt/tau))*N(0,1)
```

- The noise scales with sqrt(dt), so the result doesn't depend on the timestep.
- The amplitudes stay statistically stationary.
- tau is a physical correlation time, roughly l/sigma_v.

### 2. Use one advected field for both the IC and the boundary

In the BH frame the CGM moves at -v_bh ẑ, so the turbulent pattern is advected:

```
v(x,t) = -v_bh ẑ + δv(R, z + v_bh*t, t)
```

The boundary at z_top then just samples δv(R, z_top + v_bh*t). This gives three things:

- The IC and the inflow are the same field, so there's no jump at z_top.
- If δv is a deterministic function of t, it can be called from `WindZBC` at every RK stage,
  and restarts are trivial.
- Physically, the box crossing time (~64 kpc / 1000 km/s ≈ 63 Myr) is comparable to or
  shorter than the eddy time (l/sigma ≈ a few kpc / 50–100 km/s ≈ 30–100 Myr). The BH mostly
  sweeps through a nearly **frozen** field (Taylor's hypothesis), so OU evolution of the
  phases is a secondary effect. Start with a frozen field; add OU only if it matters.

If OU is used: update it once per full timestep, not inside the boundary function (which is
called at every stage). Store the OU state in the dump, and generate random numbers on one
rank with a fixed seed, then broadcast so all MPI ranks agree.

### 3. Make the field divergence-free

A random vector amplitude v(k) has a compressive part. That makes sound waves and weak shocks,
and with cooling switched on they turn into density and cooling artifacts. In Cartesian
coordinates, project the amplitude onto the plane perpendicular to k.

The setup is **axisymmetric (R, z)**, so plane-wave sin(k·x) modes don't fit: they aren't
regular at the axis, and v_R must vanish at R = 0. Use a stream function instead. With

```
psi = Σ A_k R J1(k_R R) cos(k_z z + phi_k)
```

- v_R = A_k k_z J1(k_R R) sin(k_z z + phi_k)
- v_z = A_k k_R J0(k_R R) cos(k_z z + phi_k)

This is exactly divergence-free, and v_R -> 0 on the axis automatically. Choose k_R so the
field behaves sensibly at R_max (e.g. zeros of J1).

### 4. Smaller choices

- **Spectrum:** for a Kolmogorov-like start, |A_k|² x (mode density) ∝ k^{-5/3}. Normalize
  the sum to a target sigma_v.
- **Range of k:** keep k_min at a few times the box scale and the smallest wavelength above
  ~16 cells. Keep sigma_v << v_bh so the boundary stays supersonic inflow everywhere
  (c_s ≈ 150 km/s, v_bh ≈ 1000 km/s gives a lot of room).
- **Density and pressure:** keep pressure uniform. Density can stay uniform too; turbulence
  subsonic relative to c_s doesn't need a consistent δρ.
- **It's not yet developed turbulence:** a Fourier-synthesized field has random phases, no
  intermittency and no real cascade. It relaxes over about one eddy time. If that matters,
  run a separate periodic box driven to steady state and inject that field instead.
- **2D caveat:** axisymmetric "eddies" are toroidal rings, and 2D turbulence sends energy to
  large scales (inverse cascade). Fine for a first look, but quantitative mixing/entrainment
  results need 3D. In 3D the plane-wave version works as-is, with the projection and OU
  changes above.

### Suggested form

```
v(R,z,t) = -v_bh ẑ + Σ_k A_k(t) * u_k(R, z + v_bh*t)
```

u_k are the solenoidal Bessel–Fourier modes above. A_k is complex, with fixed |A_k| following
the spectrum, either frozen or evolved with OU at correlation time tau ~ l/sigma_v. The same
function sets the initial condition and fills the ghost cells at z_top.

Next step: implement in a new `problems/` directory as a copy of `tcgm_subcycle`, starting
with a frozen field and adding OU as an option.
