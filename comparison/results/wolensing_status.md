# wolensing integration status (P4, 2026-10-12)

## Done

- `refs/wolensing` cloned at upstream commit **7b9101f** ("updated constants"),
  the same convention as the other refs (git clone in `refs/`).
- Runtime deps installed in the **comparison** env: `lenstronomy`, `astropy`,
  `tqdm` (conda-forge) and `fast_histogram` (PyPI-only; declared under
  `[tool.pixi.feature.comparison.pypi-dependencies]`).
- `comparison/wolensing_driver.py` implements their pipeline for a
  *dimensionless* lens setup, mirroring their own `test/conftest.py` fixture
  (`thetaE = Einstein_radius(zL, zS, m_ref)`, `Tscale = 4(1+zL) m_ref G/c³`,
  `T0 = thetaE^-2 T(0,0)`, `WindowSize = 210 thetaE(m_window)`), and provides
  `run()` / `F_at()`.
- **Validation (a): reproducibility.** A live run reproduces wolensing's own
  shipped SIS reference (`test/test_sis_Fws.txt`, 20260 points, w up to
  1.26e4) to **<= 2.0e-11** relative at w = 0.5, 1, 3, 10.  wolensing is
  deterministic and reproducible (unlike glworia, which fails its own
  reference under jax 0.10.2).

## Open question (blocking the comparison)

Their shipped SIS reference does **not** agree with the exact values.  With
their conftest settings, for an SIS (theta_E = thetaE_ref, source y = 0.5):

| w | wolensing (live & stored) | ours (exact RC) | ND/Hankel referee |
|---|---|---|---|
| 1  | 1.0198 | 2.1755 | 2.1756 (Hankel, |I|/(2pi/w)) |
| 3  | 1.2033 | 2.1300 | — |
| 10 | 1.6111 | 1.6414 | — |
| 30 | 2.1184 | 0.7106 | — |

Their curve rises from ~1 (unlensed) toward the geometric plateau ~2.2 as w
grows, while ours falls from the low-w plateau ~2.2 toward the geometric
limit: the shapes are those of the *same* physics but *stretched in frequency*
(the strong-lensing transition appears at w_their ~ 30-100 vs omega ~ 0.5-1;
a factor ~30 ~ sqrt(1e3) = the sqrt of the reference mass ratio is
suspicious).  Their point-mass case (lenstronomy `POINT_MASS`) likewise
disagrees (1.025 vs 1.701 at w = 1, y = 0.5).

Tried: their production time window (`TimeLength = 4/Tscale`,
`TExtend = 10/Tscale`, from `examples/point_mass.ipynb`) -- changes the values
(w = 30: 2.118 -> 2.058) but does not fix the stretch.

## Next diagnostics (in order)

1. Fit the frequency stretch: compute the *geometric* |F| for their SIS case
   (2 images at u = 1 +- v; |F_geo| ~ 2.2) and locate the w where their curve
   approaches it; compare with the *time* normalisation actually used
   (`Tscale`, `TimeStep = 1e-5/Tscale`, `ts*Tscale` in `fourier()`), i.e.
   verify the phase = w * T_tilde with w = 2*pi*f*Tscale.
2. Compare their notebook parameter path with the conftest path: the notebooks
   use `param_processing(zL, zS, mL_ref)` (in `wolensing.utils.utils`) for the
   window scale, while the conftest uses `Einstein_radius`; check whether
   `param_processing` applies an extra normalisation (and whether the notebooks
   set `Tscale` differently).
3. Inspect `wolensing.utils.utils.Morse_indices` / `compute_geometrical` usage
   (their geometric-optics machinery) and the `kwargs_macro` semantics
   (`theta_E`, `mu` feed `F_tilde_extend`'s tail fit only).
4. Only then re-run `comparison/wolensing_compare.py` (already patched: Hankel
   returns the raw integral -- converted with `* w/(2 pi)`; the point-mass
   referee is the ND form) and record `results/wolensing_compare.csv`.

## Environment note

While the user's Julia jobs run, wall-clock measurements are unreliable
(load average 15-25).  The P4.1 cost run was therefore parked; its
load-independent step counts are saved at
`comparison/results/high_omega_loadpolluted.txt`
(omega = 10/30/100/300 -> 2766/4366/11406/29966 inner-solve steps, accuracy vs
ND 7.5e-16 .. 2.5e-14).  Re-run `python -m comparison.high_omega` on a quiet
machine for the wall times.
