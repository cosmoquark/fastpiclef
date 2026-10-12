# wolensing integration status (P4, 2026-10-12)

## Status: RESOLVED — the comparison is meaningful

The 2026-10-12 "frequency stretch" mystery was a **unit conversion on our
side**, not a wolensing problem.  wolensing's pipeline returns the *physical*
frequency `fs` (Hz) — its FFT conjugates the *physical* time `ts*Tscale` — and
its shipped test files store the *physical angular* frequency `ws = 2 pi f`
(rad/s; the maximum 1.26e4 = 2 pi * 2000 Hz = 2 pi * `freq_end`).  The
dimensionless Fresnel frequency is

    w = 2 pi f Tscale = ws Tscale          (Tscale = 4(1+z_L) M_ref G/c^3),

which is exactly our `omega`; comparing their raw physical array against our
omega produced the fake `1/Tscale` stretch (1/Tscale = 33.8 for the fixture's
Tscale = 0.0296 s; the earlier "sqrt(m_ref)" hypothesis was a numerical
coincidence).

**Decisive check (no live pipeline run)**: `wolensing_driver._validate_stored_vs_ours()`
compares their *stored* SIS reference (y = 0.1, the conftest case) with our
exact RC solver at w = ws*Tscale:

| ws (rad/s) | w = ws*Tscale | wolensing |F| | ours |F| | rel |
|---|---|---|---|---|
| 10    | 0.2955 | 1.629293 | 1.612532 | 1.04e-2 |
| 30    | 0.8866 | 2.246543 | 2.254385 | 3.48e-3 |
| 100   | 2.9553 | 3.965135 | 3.979327 | 3.57e-3 |
| 300   | 8.8659 | 6.162325 | 6.170206 | 1.28e-3 |
| 1000  | 29.5529 | 3.324948 | 3.338613 | 4.09e-3 |
| 3000  | 88.6588 | 1.372840 | 1.375200 | 1.72e-3 |
| 10000 | 295.5295 | 5.550733 | 5.576007 | 4.53e-3 |

They agree across w = 0.3..296 to 1.3e-3..1.0e-2 — wolensing's own
FFT/time-window accuracy, which is the honest figure of merit for their
method (their time-domain integration + FFT is not a precision solver; the
scatter across w is their apodization/window artifact).

## Done

- `refs/wolensing` cloned at upstream commit **7b9101f** ("updated constants"),
  the same convention as the other refs (git clone in `refs/`).
- Runtime deps installed in the **comparison** env: `lenstronomy`, `astropy`,
  `tqdm` (conda-forge) and `fast_histogram` (PyPI-only; declared under
  `[tool.pixi.feature.comparison.pypi-dependencies]`).
- `comparison/wolensing_driver.py` implements their pipeline for a
  *dimensionless* lens setup, mirroring their own `test/conftest.py` fixture
  exactly (`thetaE = Einstein_radius(zL, zS, m_ref)`, `Tscale = 4(1+zL) m_ref G/c^3`,
  `T0 = thetaE^-2 T(0,0)`, `TimeStep = 1e-5/Tscale`, `TimeLength = 0.13/Tscale`,
  `TExtend = 10/Tscale`, `WindowSize = 210 thetaE(m=10)`), and provides
  `run()` (returns the dimensionless `w`) / `F_at()`.
- **Reproducibility validated**: a live run reproduces wolensing's shipped SIS
  reference to <= 2e-11 relative (with the same Tscale conversion applied to
  the stored ws).  wolensing is deterministic (unlike glworia, which fails its
  own shipped reference under jax 0.10.2).
- **Convention validated**: the stored/live curves match the exact truth at
  their accuracy (table above); the point mass matches the Nakamura-Deguchi
  closed form at the same level (`wolensing_driver` validation (b)).
- **A `POINT_MASS` landmine**: lenstronomy's `fermat_potential` at the lens
  centre returns a numerical-floor constant (57.69, independent of theta_E)
  instead of diverging; their own `point_mass.ipynb` anchors the time window at
  exactly that value, which then lies far past the image delays and silently
  drops the image signals (78% wrong).  The driver detects it (centre value
  more than 5 above the potential's grid minimum) and anchors at the grid
  minimum instead; the point mass then agrees with ND to 1.2e-3..1.4e-3.
- **Head-to-head** (`comparison/results/wolensing_compare.csv`; |F|, ours at
  tol 1e-14, Hankel/ND referee as external truth; `rel_*` are relative
  differences):

  | case | ours vs referee | wolensing vs referee | wolensing vs ours |
  |---|---|---|---|
  | point mass, y = 0.5 | 7.8e-16..1.8e-15 | 1.2e-3..1.4e-3 | 1.2e-3..1.4e-3 |
  | NFW, y = 0.3 | 6.8e-16..7.9e-16 | 5.0e-4..2.8e-3 | 5.0e-4..2.8e-3 |
  | SIS, y = 0.1 | 5.8e-16..7.5e-16 | 1.1e-3..1.1e-2 | 1.1e-3..1.1e-2 |
  | SIS, y = 0.5 | 6.1e-16..6.8e-16 | 1.8e-2..2.9e-2 | 1.6e-2..6.7e-2 |

  (referee rows at omega = 1, 10; the last column is the max over
  omega = 1, 3, 10, 30).  Our solver is exact to machine precision where their
  shipped pipeline is accurate to 1e-3..1e-2 (configuration-dependent), at
  minutes per configuration for them against milliseconds for us.

## Environment note

While the user's Julia jobs run, wall-clock measurements are unreliable
(load average 15-25).  The P4.1 cost run was therefore parked; its
load-independent step counts are saved at
`comparison/results/high_omega_loadpolluted.txt`
(omega = 10/30/100/300 -> 2766/4366/11406/29966 inner-solve steps, accuracy vs
ND 7.5e-16 .. 2.5e-14).  Re-run `python -m comparison.high_omega` on a quiet
machine for the wall times.

