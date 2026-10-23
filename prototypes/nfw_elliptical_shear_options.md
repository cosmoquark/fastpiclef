# Elliptical NFW + shear — options, status, and resume notes

**Date:** 2026-10-23 (work started 2026-10-22)
**Status:** prototype complete; options to review; no commit made.
**Scope:** the `prototypes/new_test.txt` case — an elliptical NFW halo plus
external shear, 500×500 source-plane maps of `|F(ω, β)|`.

This file is meant to be read cold after a context compaction. It records
what was done, the measured numbers, the exact reduction, the failure modes,
and the options for making fpl faster/robust on this workload.

---

## 0. TL;DR

- The case is **FIONA notebook §5.4** (`refs/FIONA/elliptical_sheared_lenses.ipynb`),
  also used by the GWLENS authors.
- fpl **can solve this model exactly** even though elliptical profiles are
  not in its native class: an affine rescaling maps the elliptical NFW to a
  circular NFW plus a diagonal macro quadratic. Validated against GWLENS to
  `≤ 6e-13` in `|F|` and against FIONA's potential to `7e-14`.
- A prototype and the 3×3 maps exist:
  `prototypes/nfw_elliptical_shear.py`,
  `prototypes/results/nfw_elliptical_shear/`.
- fpl exact handles every panel up to `ω=20`, and `c200=15` at `ω=100`
  (~104 s). The compact halos `c200=30, 60` at `ω=100` **stall** in the
  adaptive contour; those two panels use GWLENS and are labelled.
- Timings on this 6-physical-core box, 500×500:
  - fpl: 7 panels in **151.7 s**;
  - GWLENS: the 2 hard panels in **41.4 s**;
  - hybrid total **193.1 s**;
  - GWLENS alone for all 9 panels: **50.1 s**.
- fpl does **not** beat GWLENS on this dense-map benchmark with the current
  per-source-point outer quadrature. The exact engine is a precision anchor,
  not a dense-map engine.
- Several exact and approximate speed-up options are listed in §5. The
  cheapest exact win is the central-inversion symmetry `F(-β)=F(β)` (2×),
  which the current map does **not** exploit.

---

## 1. The case and its conventions

### 1.1 Parameters

| quantity | value |
|---|---|
| NFW mass | `M200 = 1e13 Msun` |
| lens/source redshift | `z_l = 0.5`, `z_s = 1.5` |
| cosmology | flat ΛCDM, `H0 = 70`, `Ωm = 0.3`, `Ob0 = 0.048` |
| concentration | `c200 = 15, 30, 60` |
| NFW potential ellipticity | `e1 = 0.2`, `e2 = 0` |
| external shear | `gamma1 = 0.03`, `gamma2 = 0.01` |
| source grid | 500×500, `y1, y2 ∈ [-1.5, 1.5]` arcsec |
| dimensionless frequency | `ω = 1, 10, 100` |

`LensCosmo.nfw_physical2angle(M=1e13, c)` gives the angular parameters
(`Rs`, `alpha_Rs` in arcsec):

| c200 | Rs [arcsec] | alpha_Rs [arcsec] |
|---|---|---|
| 15 | 4.05686713256022546 | 1.51259845796567971 |
| 30 | 2.02843356628011273 | 2.25099458385512863 |
| 60 | 1.01421678314005637 | 3.55038709413520559 |

The prototype tabulates these, so astropy/lenstronomy are not needed at run
time (`--from-lenstronomy` recomputes them).

### 1.2 The ellipticity convention (a cryptic bit of `new_test.txt`)

`e1 = 0.2` is **lenstronomy's complex ellipticity**, not the anisotropic
scaling used in the potential. The mapping is

```
c = |e_vec|,            q = (1-c)/(1+c),
e = |1-q^2|/(1+q^2),    phi = atan2(e2,e1)/2.
```

For `e1=0.2, e2=0`: `q = 2/3`, `e = 5/13 ≈ 0.3846`. The elliptical potential is

```
psi_NFW(s),   s^2 = (1-e) x1'^2 + (1+e) x2'^2,
```

with `x'` the principal (ellipticity) frame. This is the same
`transform_e1e2_square_average` convention as FIONA's
`NFW_ELLIPSE_POTENTIAL`.

### 1.3 β

`β = (β0, β1)` is the **source-plane position** (impact parameter).
In the Fermat phase

```
phi(x;β) = ½|x|² - psi(x) - β·x + ½|β|².
```

Every pixel of a map is a different β; in the figures the axes `y1,y2` are
`β0,β1`. `ω` and `c200` are fixed per panel.

---

## 2. What was delivered

| path | contents |
|---|---|
| `prototypes/nfw_elliptical_shear.py` | self-contained prototype: exact reduction, fpl/GWLENS maps, validation, plots |
| `prototypes/results/nfw_elliptical_shear/nfw_elliptical_shear_linear.png` | 3×3 `|F|` figure, linear scale |
| `.../nfw_elliptical_shear_log.png` | same, log scale |
| `.../summary.md` | per-panel solver, wall time, inner solves, validation |
| `.../maps.npz` | compact float32 maps (6 MB; opt-in via `--save-data`) |
| `Journal.typ` | appended `#opencode-entry("2026-10-22", ...)`; `Journal.pdf` recompiled |

Run (preferred: dedicated GWLENS environment):

```bash
pixi run -e gwlens nfw-elliptical
```

The `gwlens` environment installs only the solver's light runtime deps
(`finufft`, `numexpr`, `threadpoolctl`, `astropy`); the `gwlens-setup` task
clones the pinned GWLENS source into `.pixi/gwlens-src` (gitignored) and the
task puts it on `PYTHONPATH`.  This avoids GWLENS's heavy `bilby`/`lalsuite`
metadata, which does not build here.

Fallback from the comparison environment, using a scratch clone:

```bash
git clone https://github.com/hsreekanth15/GWLENS /tmp/GWLENS
pixi run -e comparison python prototypes/nfw_elliptical_shear.py \
    --gwlens-dir /tmp/GWLENS
```

Options: `--quick` (n=64, separate output dir), `--solver auto|fpl|gwlens`,
`--force-fpl`, `--save-data`, `--no-plot`, `--from-lenstronomy`.  To pass
options to the task, use the direct form:
`pixi run -e gwlens bash -c 'PYTHONPATH=.pixi/gwlens-src python prototypes/nfw_elliptical_shear.py --quick'`.

GWLENS is **not vendored** (`refs/` is read-only).

---

## 3. The exact reduction (important — do not lose this)

fpl's class is an axisymmetric profile plus a **diagonal** macro quadratic.
The elliptical NFW fits after an affine rescaling.

Let

```
Se = diag(1/sqrt(1-e), 1/sqrt(1+e)),
R  = rotation into the ellipticity principal frame,
G  = [[gamma1, gamma2], [gamma2, -gamma1]]  (external shear),
```

and change variables `x = Se u`. Then:

1. `s² = (1-e)x1'² + (1+e)x2'² = u1² + u2²`, so the NFW is **circular** in `u`.
2. The Fermat quadratic becomes `½ uᵀ A u` with

   ```
   A = Se (I - G) Se.
   ```

3. `A` is symmetric; diagonalise it:
   `A = rot(ψ) diag(μ0, μ1) rot(ψ)ᵀ`, with `μ0 ≤ μ1`.
   Then set

   ```
   kappa = 1 - (μ0 + μ1)/2,
   gamma = (μ1 - μ0)/2,
   (gamma1_fpl, gamma2_fpl) = gamma (cos 2ψ, sin 2ψ).
   ```

4. The source transforms as

   ```
   beta_u = Se R beta.
   ```

5. The Jacobian is `d²x = d²u / sqrt(1-e²)`, so

   ```
   F_ell(beta) = F_fpl(beta_u) / sqrt(1-e²).
   ```

That is exact, not an approximation. It effectively gives the P4.5
"affine quadratic form" extension for a **single common ellipticity**
(any number of co-axial components with that ellipticity). Different
ellipticities per component still need the full P4.5 contour work.

### 3.1 The `LensField.map` frame gotcha

`LensField.solve` applies the internal shear rotation to its `beta` argument.
`LensField.map` does **not**: `beta0s/beta1s` are expected **already in the
macro-diagonal frame**. The prototype keeps two helpers:

```
beta_solver_input = Se R beta                 # for .solve
beta_map          = rot(psi) Se R beta        # for .map
```

For the inverse (plotting), `map_to_physical` undoes the rotation and the
scaling. A stub/docstring note would prevent silent frame mistakes.

### 3.2 Validation of the reduction

- Against GWLENS's direct elliptical-potential scalar solver, at 5 sources × 3
  frequencies: `|ΔF|/|F| ≤ 6e-13`.
- Against FIONA's `NFW_ELLIPSE_POTENTIAL` on 200 random real points:
  max relative potential difference `7.2e-14`, median `2.6e-16`.
- In the 500×500 runs, `fpl_vs_gwlens` at six random points per feasible
  panel: `1.2e-13 … 9.4e-10`.
- Node checks (map at its own nodes vs a high-accuracy point solve):
  `7.4e-13 … 1.8e-9` for fpl; `1.5e-12`, `3.9e-13` for the two GWLENS panels.

---

## 4. Measured performance and failure modes

### 4.1 Timings (this 6-physical-core box; load-dependent)

500×500, hybrid run:

| c200 | ω=1 | ω=10 | ω=100 |
|---|---|---|---|
| 15 | fpl 2.78 s | fpl 11.36 s | fpl 104.32 s |
| 30 | fpl 3.66 s | fpl 11.65 s | GWLENS 18.26 s |
| 60 | fpl 3.33 s | fpl 14.59 s | GWLENS 23.16 s |

Totals: fpl 151.7 s (7 panels), GWLENS 41.4 s (2 panels), hybrid 193.1 s.
A separate `--solver gwlens` run over all nine panels: **50.1 s** total
(~0.4 s at ω=1,10; 10.8–19.7 s at ω=100).

The authors' 6.2 s (FIONA 130 s) is not reproduced here; it is
hardware-dependent (their notebook used 112 threads). The ratio is what they
quote; the absolute numbers on this box differ.

### 4.2 Exact-engine envelope

Empirical (see `fpl_feasible` in the prototype):

- all `c200` up to `ω ≈ 20`;
- `c200=30` up to `ω ≈ 50` (map);
- `c200=15` up to `ω=100`;
- `c200=30, 60` at `ω=100`: **stall** — even a 4×4 map probe and a single
  scalar point exceed 40 s; scalar runs exceed 5 min in several settings.

### 4.3 Why the stall (current diagnosis)

The adaptive 1D quadrature keeps refining; it is not a crash. The phase
varies by `~ω ψ0 ~ 1000–1500 rad` across the field, and the NFW `log²` cusp
makes it vary faster near the centre. The RC contour's real core passes
through/near the centre, so the integrand oscillates on an arbitrarily fine
scale there; the adaptive integrator subdivides panels without a cusp-aware
treatment. The shared map contour is sized for the whole β range, and the
vector rule accepts a panel only when **every** source in the chunk
converges, so one hard region forces refinement for the whole chunk. The
interrupt hook is checked only at orchestration boundaries, so Ctrl-C/SIGALRM
can take tens of seconds — it *looks* hung.

Important controls:

- A **smooth CIS** (`psi0=15, xc=1.0`) and even an **SIS** (`psi0=15`) also
  stall at `ω=100`; `psi0=1` is fine. So it is **not the NFW cusp alone**;
  the large potential amplitude / contour conditioning is the trigger.
- GWLENS does not need an adaptive contour: it samples a real-space grid fine
  enough for the Fresnel scale and uses one NUFFT for all β. Its cost is set
  by the grid, not by unbounded refinement.

### 4.4 Batch/chunking behaviour (already-shared contour)

`LensField.map` → `FieldBatch` → `slice_field_batch_g`. The phase is

```
phi = phi_0(z0,x2) - beta0*z0 - beta1*x2 + (beta0^2+beta1^2)/2.
```

The batch builds **one shared contour per group**:

- inner `z0` contour: union of the chunk's `β0`, the shifted centres
  `β0/λ0`, branch abscissae, plus `margin`;
- outer `x2` contour: union of `β1`, mass heights, plus `margin_outer`.

The lens potential is evaluated once per node; the β dependence is twiddles
(`e^{-iωβ0 z0}` inner, `e^{-iωβ1 x2}` outer). Uniform grids use a re-anchored
recurrence (`REANCHOR=16`). This is already a strong form of "one common
thimble for the whole grid".

`MAX_GROUP_COMPONENTS = 65536`; a 500×500 map runs as **4 chunks** of
131×500. Chunking is along `β0` (the inner variable), not `β1`.

Resolution scan at ω=10, c=15, same β range:

| n | points | groups | inner_solves | wall |
|---|---|---|---|---|
| 250 | 62 500 | 1 | 1 340 | 4.48 s |
| 256 | 65 536 | 1 | 1 340 | 4.68 s |
| 257 | 66 049 | 2 | 2 680 | 4.54 s |
| 512 | 262 144 | 4 | 5 360 | 14.20 s |

Doubling `inner_solves` at n=257 does **not** double the wall: each chunk's
`β0` range is halved, so its contour is smaller. The chunking is not
catastrophic, but it is a memory cap, not a conditioning-aware grouping.

### 4.5 Exact symmetry not currently used

For a centred lens, `F(-β) = F(β)` exactly (complex values; verified to
`≤ 5.9e-16`). Every map is 2× redundant: compute `β0 ≥ 0` and mirror.
The inner contour also shrinks from `[-βmax,βmax]` to `[0,βmax]`.
This is the cheapest exact win and is **not** implemented.

---

## 5. Options

Legend: **exact?** = preserves the exact RC result; **gain** = expected
speed/robustness; **effort** = rough implementation cost.

### A. Exact-engine options

| id | option | exact? | gain | effort | notes |
|---|---|---|---|---|---|
| A1 | central-inversion symmetry | yes | 2× on all maps | tiny | `F(-β)=F(β)`; verified 1e-16 |
| A2 | batch/chunk tuning | yes | 1.2–2× | small | raise `MAX_GROUP_COMPONENTS`; one group for 500²; try chunking along β1 + cache inner |
| A3 | conditioning-aware clustering | yes | 1.5–3× | medium | group by caustic/phase difficulty, not contiguous β0; isolate hard sources |
| A4 | cusp-aware contour/quadrature | yes | extends envelope to c=30/60 at ω=100 | medium–high | subtract `log²` cusp, graded/DE panels, contour detour around centre |
| A5 | polar-slice exact solver | yes | likely extends envelope; radial cusp is an endpoint | high | radial inner `r dr`, angular outer (periodic FFT/trapezoid) |
| A6 | complex-frequency fast β transform | yes | dense-map speed | research | global transform on the deformed contour; not a vanilla NUFFT |

**A1 — central symmetry.** `F(-β)=F(β)` for a centred lens (all components
and the macro quadratic are even). Compute only `β0 ≥ 0`, mirror. This halves
the number of points and the inner contour; the outer `β1` range stays full.
Do this first.

**A2 — batch/chunk tuning.** `MAX_GROUP_COMPONENTS=65536` forces 4 chunks for
500². The memory for one group is small (250k complex ≈ 4 MB), so raising the
cap to e.g. 262144 is safe and may save contour rebuilds. The code currently
chunks along `β0`; chunking along `β1` would keep the expensive inner
contour fixed across chunks, but requires caching the inner `G(x2)` (the
current `run_group` recomputes it per chunk).

**A3 — conditioning-aware clustering.** The vector adaptive integrator
accepts a panel only when all components converge. A single caustic/hard
source can dominate a chunk. Group by local difficulty, use smaller contours
per cluster, and relax the all-converge coupling (component-wise refinement
or separate hard-source solves).

**A4 — cusp-aware / high-amplitude contour.** The CIS/SIS controls show the
problem is broader than the NFW cusp, so a cusp-only fix may not suffice.
Options: analytic subtraction of the `log²` cusp, a double-exponential /
graded panel rule near the centre, a small contour detour around the cusp,
and per-chunk `margin`/`margin_outer`/tail-angle tuning. This is the neatest
route to exact `c=30/60, ω=100`, but needs a careful numerical study.

**A5 — polar-slice exact solver.** Slice in polar coordinates: inner radial
integral `∫ r dr`, outer angular integral (periodic). The cusp is an endpoint
of the radial integral, which a DE/tanh-sinh rule handles naturally. Bigger
change than A4, but likely the most natural exact fix.

**A6 — complex-frequency fast β transform.** A global transform on the
deformed contour. Standard NUFFT assumes real nodes; on a PL contour the
kernel is a generalized Laplace kernel. On a straight tail at angle α the
effective frequency is `ωβ(1+i tan α)`. Research-grade.

### B. Fast map modes

| id | option | exact? | gain | effort | notes |
|---|---|---|---|---|---|
| B1 | vectorized geometric + correction | asymptotic | seconds for high-ω maps | medium | 1D radial lens equation for the scaled circular NFW + diagonal macro; validate against exact |
| B2 | real-space NUFFT map mode | controlled approximation/new solver | 6.2 s-class dense maps | high | FIONA/GWLENS style; separate backend |
| B3 | phase-aware adaptive interpolation | error-controlled approximate | 2–10× on smooth maps | medium | interpolate envelope + eikonal, quadtree refine caustics |
| B4 | hybrid exact/geometric | approximate high-ω | fast figure | small–medium | exact RC at ω=1,10; geometric at 100 |
| B5 | angular-harmonic / Fourier–Bessel | semi-analytic | potentially fast for this model | research | circular NFW + diagonal quadratic; expand in angular harmonics |
| B6 | MGE / softened cusp | approximate | conditioning | medium | model approximation; CIS/SIS controls suggest limited help |

**B1 — vectorized geometric + correction.** After the affine reduction the
lens equation is 1D in radius:

```
r² = β1²/(λ1 - a(r)/r)² + β2²/(λ2 - a(r)/r)²,
a(r) = dψ_NFW/dr.
```

For a given `r`, `x_i = β_i/(λ_i - a(r)/r)`; solve the scalar equation by a
vectorized root scan for all sources at once, then form the geometric sum
plus the P4.1 correction. This is the fastest route to the high-ω panels.
It is **asymptotic**, not exact: the current per-point `geometric_corrected`
had 2–8% errors at some ω=100 points, so the implementation and its error
budget need validation against exact RC at ω≤20 and c=15 ω=100.

**B2 — real-space NUFFT map mode.** The proven FIONA/GWLENS route: sample the
lens phase on a real-space grid, window/tail-correct, and evaluate all β with
one NUFFT. This is a new backend, not an optimization of the PL contour. It
is the route to 6.2 s-class 500² maps; the accuracy comes from grid/window/
tail control.

**B3 — phase-aware adaptive interpolation.** Compute exact on a coarse β
grid, write `F = A exp(iωΦ)` with smooth envelope `A` and smooth eikonal `Φ`,
interpolate `A` and `Φ` (not `F`), estimate the error, refine only near
caustics/fringes. This exploits the output smoothness you asked about, but
needs an error certificate and careful handling of the fringe Nyquist.

**B4 — hybrid exact/geometric.** Exact RC for low ω (1,10), vectorized
geometric+correction for high ω (100), with exact anchors. This is the
pragmatic "get the figure fast" route, but with fpl's own geometric machinery
once B1 exists.

**B5 — angular-harmonic / Fourier–Bessel.** The scaled problem is a circular
NFW + diagonal quadratic. In polar coordinates the angular dependence is only
through the quadratic; expand in angular harmonics and reduce the 2D integral
to 1D radial integrals. Potentially fast and cusp-friendly, but research-grade
(the anisotropy is large, so the number of harmonics may not be small).

**B6 — MGE / softened cusp.** Approximate the NFW by a multi-Gaussian
expansion or a tiny core to make the contour well-conditioned. Model
approximation; the smooth CIS/SIS controls suggest a small core may not fix
the large-amplitude conditioning.

### C. Engineering / pragmatic

| id | option | gain | effort | notes |
|---|---|---|---|---|
| C1 | SIMD/parallel/GPU | modest | medium | high-ω problem is algorithmic, not arithmetic |
| C2 | cache/reuse contours and inner G | 1.5–2× | medium | reuse across chunks, symmetry, zero-source phase |
| C3 | plot-only coarse grid | 4× at half resolution | tiny | 250² upsampled; not an exact map |
| C4 | GWLENS/FIONA for maps + fpl anchor | immediate | tiny | what the current prototype does |

**C1.** SIMD twiddles, parallel independent chunks, better memory layout.
Secondary until the algorithmic issues are addressed.

**C2.** Cache contours/inner results across chunks; reuse the contour for the
symmetric half-grid; cache the zero-source phase.

**C3.** The dark interference nulls are already sub-pixel at 500²; a 250²
exact map would look similar for a figure. Only for plots, not for data.

**C4.** Keep fpl as the precision anchor and use GWLENS/FIONA for dense maps.
The current hybrid figure is this option.

---

## 6. Recommended order

1. **A1 central symmetry** — exact, trivial, 2× on every map.
2. **A2 chunk cap** — cheap; measure one-group vs four-group at 500² and
   ω=100. Try chunking along β1 + caching the inner G.
3. **B1 vectorized geometric mode** — prototype in Python for the scaled 1D
   radial equation; validate against exact RC. Best fast route for the
   c=30/60, ω=100 panels.
4. **A4/A5 cusp-aware exact contour** — if exactness at high ω is required;
   A5 (polar slicing) is the neatest conceptually but a bigger change.
5. **B2 real-space NUFFT** — if 6.2 s-class dense maps are the target.

The key question to settle before coding further:

- optimize for **exact c=30/60 at ω=100** → A1 + A4/A5;
- optimize for **fast high-ω panels** → B1;
- optimize for **dense-map throughput** → B2.

A1 + A2 are worth doing regardless.

---

## 7. Known issues / side findings

1. **`LensField.map` frame convention** — `solve` rotates `beta` internally,
   `map` does not. `map` expects `beta0s/beta1s` in the macro-diagonal frame.
   A stub/docstring note is warranted.
2. **NFW image-finder certificate at β=0** — `LensField.images()` raises for a
   cuspy NFW at `β=0`: the index identity treats NFW as smooth, but the
   central cusp contributes. Brute force finds 4 images (2 positive, 2
   negative `det H`, index 0) where the certificate expects 1. This affects
   `hybrid`/`geometric`, not the exact RC maps.
3. **Dark nulls are sub-pixel** — at c=60, ω=10 the interference nulls are
   narrower than the 500² spacing (~0.006 arcsec). Scalar fpl and GWLENS agree
   to ~1e-9 at the nulls; map values *between* nodes are not meaningful (fpl
   samples, GWLENS's NUFFT smooths). Compare at nodes or with scalar solves.
4. **fpl map error estimate at nulls** — the map's reported error (~1e-5)
   does not cover the cancellation at dark nulls; the scalar error estimate at
   hard points can be huge (e.g. 1.25) while the value is correct.
5. **FIONA fixed-grid attempt was unconverged** — with
   `auto_R_from_gl_nodes=False, gl=500–700, R=6`, FIONA gave |F| values far
   from GWLENS/fpl (8.65 vs 0.14, etc.); its window/grid is inadequate for this
   strong lens. Do not use that attempt as a reference. The valid FIONA check
   is at the potential level (7e-14).
6. **`geometric_corrected` at ω=100** — current per-point path gave 2e-2,
   7.7e-2, 1.1e-4, 3.5e-4, 5.6e-6 relative errors at five points; the
   reported budget did not always cover the true error. Treat it as
   asymptotic, not exact; validate any fast geometric mode against RC.
7. **No commit made** — `prototypes/` is untracked; `notebooks/` has staged
   additions from an earlier session (not ours). `Journal.typ`/`Journal.pdf`
   are not tracked by git in this checkout (`git ls-files` is empty for them);
   they were updated and compiled locally.

---

## 8. Environment / reproduction

- Environment: `pixi run -e comparison` (has `fastpiclef`, numpy, scipy,
  matplotlib, astropy/lenstronomy, finufft).
- GWLENS (preferred): the `gwlens` pixi environment (`gwlens` feature in
  `pyproject.toml`, pinned commit `e73167af...`); `gwlens-setup` clones into
  `.pixi/gwlens-src` (gitignored).  Fallback: a scratch clone at
  `/tmp/opencode/GWLENS` (symlink `/tmp/GWLENS`).  Not vendored (`refs/` is
  read-only).
- FIONA: `refs/FIONA` (for the potential-level check and the notebook case).
- Machine: Intel i7-10710U, 6 physical cores; the box was loaded during the
  measurements (load average 4–8), so timings are not quiet-window numbers.
- Journal render: `typst compile Journal.typ Journal.pdf`.
- Quick sanity check:

  ```bash
  pixi run -e comparison python prototypes/nfw_elliptical_shear.py \
      --quick --gwlens-dir /tmp/GWLENS
  ```

- Full hybrid run (~3 min):

  ```bash
  pixi run -e comparison python prototypes/nfw_elliptical_shear.py \
      --gwlens-dir /tmp/GWLENS --save-data
  ```

- GWLENS-only benchmark (all 9 panels, ~50 s):

  ```bash
  pixi run -e comparison python prototypes/nfw_elliptical_shear.py \
      --solver gwlens --gwlens-dir /tmp/GWLENS \
      --outdir /tmp/gwlens_full --no-plot
  ```

---

## 9. Open questions for Paul

1. Which objective should we optimize: exact c=30/60 at ω=100, fast high-ω
   panels, or 6.2 s-class dense-map throughput?
2. Is a prototype-level exact-symmetry/chunk optimization enough, or should
   the 2× symmetry and chunk cap go into the Rust core?
3. Should GWLENS be integrated as a comparator in `comparison/` (it needs a
   scratch clone; licensing/deps to confirm)?
4. A4 (cusp-aware contour) vs A5 (polar-slice exact solver): which is the
   preferred research direction for the exact engine?
5. Should the prototype get a `pixi` task, or stay a standalone script?
