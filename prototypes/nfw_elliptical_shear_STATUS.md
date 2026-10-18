# Elliptical NFW + shear — current status / resume note

**Date:** 2026-10-23
**Read this first**, then `prototypes/nfw_elliptical_shear_options.md` for the
full options list and derivations.

---

## Where we are

- The `prototypes/new_test.txt` case is **FIONA notebook §5.4**: an
  elliptical NFW + external shear, 500×500 source maps, `c200 = 15, 30, 60`,
  `ω = 1, 10, 100`. Parameters and conventions are in the options file §1.
- **fpl can solve this model exactly** via an affine rescaling of the
  elliptical potential to a circular NFW + diagonal macro. The derivation,
  the source transformation, and the `LensField.map` frame gotcha are in
  options §3. Validated against GWLENS to `≤ 6e-13` and FIONA's potential to
  `7e-14`.
- The prototype is complete:
  - `prototypes/nfw_elliptical_shear.py`
  - `prototypes/results/nfw_elliptical_shear/nfw_elliptical_shear_linear.png`
  - `prototypes/results/nfw_elliptical_shear/nfw_elliptical_shear_log.png`
  - `prototypes/results/nfw_elliptical_shear/summary.md`
  - `prototypes/results/nfw_elliptical_shear/maps.npz` (6 MB, optional;
    regenerated with `--save-data`)
- The Journal has a `#opencode-entry("2026-10-22", ...)` recording the
  milestone; `Journal.pdf` was recompiled. No commit was made.

## Headline numbers (this 6-core box, load-dependent)

500×500 hybrid run:

| c200 | ω=1 | ω=10 | ω=100 |
|---|---|---|---|
| 15 | fpl 2.78 s | fpl 11.36 s | fpl 104.32 s |
| 30 | fpl 3.66 s | fpl 11.65 s | GWLENS 18.26 s |
| 60 | fpl 3.33 s | fpl 14.59 s | GWLENS 23.16 s |

- fpl 7 panels: **151.7 s**; GWLENS 2 hard panels: **41.4 s**;
  hybrid total **193.1 s**.
- GWLENS alone, all 9 panels: **50.1 s** total.
- The authors' 6.2 s (FIONA 130 s) is hardware-dependent; not reproduced on
  this 6-core box.

## Current blocker

fpl's exact RC engine **stalls** for `c200=30, 60` at `ω=100`: even a 4×4
map probe and a single scalar point exceed 40 s; scalar runs exceed 5 min in
several settings. A smooth CIS (`psi0=15, xc=1.0`) and an SIS (`psi0=15`)
also stall at `ω=100`, so this is **not the NFW cusp alone** — it is the
large potential amplitude / contour conditioning (see options §4.3). The two
hard panels in the figure are GWLENS and are labelled.

## What is *not* done / next actions

1. **A1 — central symmetry (do first).** `F(-β) = F(β)` exactly for a centred
   lens (verified 1e-16). Compute only `β0 ≥ 0`, mirror. This is a free 2×
   on every map and also shrinks the inner contour. Not implemented.
2. **A2 — chunk cap.** `MAX_GROUP_COMPONENTS = 65536` makes a 500×500 map
   4 chunks of 131×500. Try raising it so the map is one group; measure
   ω=10/100. Optionally chunk along `β1` and cache the inner `G(x2)`.
3. **B1 — vectorized geometric mode (best fast route for ω=100).** After the
   affine reduction the lens equation is 1D in radius:
   `r² = β1²/(λ1 - a(r)/r)² + β2²/(λ2 - a(r)/r)²`. Solve by a vectorized
   root scan for all sources, then geometric sum + P4.1 correction. It is
   asymptotic; validate against exact RC at ω≤20 and c=15 ω=100. The current
   per-point `geometric_corrected` had 2–8% errors at some ω=100 points, so
   the error budget needs work.
4. **A4/A5 — exact high-ω contour.** If exact `c=30/60, ω=100` is required:
   cusp-aware quadrature / contour detour (A4), or a polar-slice exact solver
   (A5, bigger change).
5. **B2 — real-space NUFFT map mode.** If the target is 6.2 s-class dense
   maps, this is the proven route (FIONA/GWLENS style), but it is a new
   backend, not an optimization of the PL contour.

A1 + A2 are low-risk and worth doing regardless. The main strategic question
is whether to optimize for exactness at high ω (A1+A4/A5), fast high-ω panels
(B1), or dense-map throughput (B2).

## Commands

```bash
# quick smoke test (n=64, separate output dir)
pixi run -e comparison python prototypes/nfw_elliptical_shear.py \
    --quick --gwlens-dir /tmp/GWLENS

# full hybrid run (~3 min), with compact data
pixi run -e comparison python prototypes/nfw_elliptical_shear.py \
    --gwlens-dir /tmp/GWLENS --save-data

# GWLENS-only benchmark (all 9 panels, ~50 s)
pixi run -e comparison python prototypes/nfw_elliptical_shear.py \
    --solver gwlens --gwlens-dir /tmp/GWLENS \
    --outdir /tmp/gwlens_full --no-plot
```

GWLENS is a scratch clone at `/tmp/opencode/GWLENS` (symlink `/tmp/GWLENS`);
it is not vendored because `refs/` is read-only.

## Cautions

- `LensField.map` expects `beta0s/beta1s` **already in the macro-diagonal
  frame**; `solve` rotates internally. Use the prototype's helpers.
- The dark interference nulls (c=60, ω=10) are narrower than the 500² pixel
  spacing. Compare maps at nodes or with scalar solves, not at random
  between-node points.
- fpl's map error estimate does not cover cancellation at the nulls; the
  scalar error estimate can be huge while the value is correct.
- `LensField.images()` fails its certificate at `β=0` for a cuspy NFW (the
  index identity treats NFW as smooth; the true set has 4 images, index 0).
  This affects `hybrid`/`geometric`, not the RC maps.
- The FIONA fixed-grid attempt was unconverged; the valid FIONA check is at
  the potential level.
- No commit made. `prototypes/` is untracked; `notebooks/` has staged
  additions from an earlier session (not ours).
