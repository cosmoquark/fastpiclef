# P5 logistic-bridging comparator -- results

The heuristic competitor (Paul's end-stage item): bridge the wave limit `F -> 1` and the geometric sum `F_geo(omega)` with a logistic window `s(omega) = 1/(1 + (omega_c/omega)^p)` in log omega, instead of evaluating the Fresnel integral.  V1 blends the complex values; V2 blends the moduli and takes the phase from `F_geo`.  `tuned` = `(omega_c, p)` fitted per case (sees the exact answer); `fixed` = one pair for the three primaries (P1, M1, C1 at y = 0.2).

Reference: the exact RC engine (`sweep`); referees: the Nakamura-Deguchi closed form (moduli) for the point-mass cases and a second contour margin for M1.

## tuned per case (maximally fair)

| case | variant | omega_c | p | max rel err | RMS | at omega |
|------|---------|---------|---|-------------|-----|----------|
| P1_point_y1 | v1 | 0.2383 | 1.341 | 2.426e-02 | 1.353e-02 | 0.3 |
| P1_point_y1 | v2 | 0.1787 | 0.4484 | 1.346e-01 | 6.140e-02 | 0.3 |
| M1_macro_point | v1 | 0.3178 | 0.7477 | 2.250e-01 | 1.332e-01 | 0.3 |
| M1_macro_point | v2 | 0.2383 | 3.106 | 3.310e-01 | 1.975e-01 | 0.3 |
| C1_y0.2 | v1 | 0.5036 | 0.8974 | 4.087e-01 | 1.588e-01 | 11.94 |
| C1_y0.2 | v2 | 0.7982 | 13.88 | 4.683e-01 | 2.415e-01 | 11.94 |
| C1_y0.1 | v1 | 5.036 | 2.004 | 8.620e-01 | 3.767e-01 | 23.83 |
| C1_y0.1 | v2 | 0.7536 | 10.75 | 8.888e-01 | 4.116e-01 | 23.83 |
| C1_y0.05 | v1 | 2.674 | 1.391 | 1.384e-01 | 8.804e-02 | 2.778 |
| C1_y0.05 | v2 | 3.566 | 1.293 | 4.001e-01 | 1.721e-01 | 0.3 |
| C1_y0.02 | v1 | 4.755 | 1.293 | 1.902e-01 | 1.119e-01 | 3.239 |
| C1_y0.02 | v2 | 4.238 | 1.443 | 4.134e-01 | 2.085e-01 | 0.3 |
| C1_y0 | v1 | 0.03 | 7.194 | 1.000e+00 | 1.000e+00 | 3.635 |
| C1_y0 | v2 | 0.03 | 7.194 | 1.000e+00 | 1.000e+00 | 3.635 |

## fixed prescription (one pair, no per-case knowledge)

`omega_c = 0.5036`, `p = 0.8974` (minimax over the three primaries; worst-case max error 4.087e-01).

| case | max rel err | RMS | at omega |
|------|-------------|-----|----------|
| P1_point_y1 | 1.819e-01 | 1.031e-01 | 2.573 |
| M1_macro_point | 2.321e-01 | 1.322e-01 | 0.3 |
| C1_y0.2 | 4.087e-01 | 1.588e-01 | 11.94 |

### blind canonical pairs (written down without data)

| case | (1,1) | (1,2) | (1,4) |
|---|---|---|---|
| P1_point_y1 | 2.91e-01 | 2.24e-01 | 2.80e-01 |
| M1_macro_point | 2.66e-01 | 3.17e-01 | 3.51e-01 |
| C1_y0.2 | 4.19e-01 | 4.55e-01 | 4.68e-01 |

## ingredient floors (context)

| case | floor | max rel err |
|------|-------|-------------|
| P1_point_y1 | geo anchor | 1.893e-01 |
| P1_point_y1 | F=1 only | 2.374e+00 |
| M1_macro_point | geo anchor | 3.600e-01 |
| M1_macro_point | F=1 only | 1.357e+00 |
| C1_y0.2 | geo anchor | 1.034e+00 |
| C1_y0.2 | F=1 only | 3.937e+00 |
| C1_y0.1 | geo anchor | 1.741e+00 |
| C1_y0.1 | F=1 only | 4.964e+00 |
| C1_y0.05 | geo anchor | 2.767e+00 |
| C1_y0.05 | F=1 only | 1.211e+00 |
| C1_y0.02 | geo anchor | 4.832e+00 |
| C1_y0.02 | F=1 only | 1.205e+00 |
| C1_y0 | geo anchor | 1.000e+00 |
| C1_y0 | F=1 only | 1.204e+00 |

## referees

| case | check | value | detail |
|------|-------|-------|--------|
| P1_point_y1 | ND modulus referee | 4.93e-11 | max |d|F|| over 102/121 omegas with nd.err < 1e-8 (nd err at w=30: 8.4e-02) |
| P1_point_y1 | sweep vs solve | 9.14e-17 | omega = 0.3 |
| P1_point_y1 | sweep vs solve | 4.67e-16 | omega = 3 |
| P1_point_y1 | sweep vs solve | 9.27e-16 | omega = 30 |
| M1_macro_point | contour margin (2.0 vs 1.5) | 2.66e-16 | omega = 1 |
| M1_macro_point | contour margin (2.0 vs 1.5) | 2.49e-16 | omega = 3 |
| M1_macro_point | contour margin (2.0 vs 1.5) | 4.52e-16 | omega = 10 |
| M1_macro_point | contour margin (2.0 vs 1.5) | 2.13e-16 | omega = 30 |
| M1_macro_point | sweep vs solve | 1.09e-16 | omega = 0.3 |
| M1_macro_point | sweep vs solve | 5.32e-16 | omega = 3 |
| M1_macro_point | sweep vs solve | 1.33e-15 | omega = 30 |
| C1_y0.2 | ND modulus referee | 4.17e-14 | max |d|F|| over 121/121 omegas with nd.err < 1e-8 (nd err at w=30: 5.6e-13) |
| C1_y0.2 | sweep vs solve | 1.91e-16 | omega = 0.3 |
| C1_y0.2 | sweep vs solve | 3.17e-16 | omega = 3 |
| C1_y0.2 | sweep vs solve | 3.14e-15 | omega = 30 |
| C1_y0.1 | ND modulus referee | 4.44e-14 | max |d|F|| over 121/121 omegas with nd.err < 1e-8 (nd err at w=30: 5.6e-14) |
| C1_y0.1 | sweep vs solve | 3.74e-16 | omega = 0.3 |
| C1_y0.1 | sweep vs solve | 0.00e+00 | omega = 3 |
| C1_y0.1 | sweep vs solve | 1.42e-15 | omega = 30 |
| C1_y0.05 | ND modulus referee | 8.70e-14 | max |d|F|| over 121/121 omegas with nd.err < 1e-8 (nd err at w=30: 3.3e-14) |
| C1_y0.05 | sweep vs solve | 8.93e-17 | omega = 0.3 |
| C1_y0.05 | sweep vs solve | 0.00e+00 | omega = 3 |
| C1_y0.05 | sweep vs solve | 1.10e-15 | omega = 30 |
| C1_y0.02 | ND modulus referee | 1.06e-13 | max |d|F|| over 121/121 omegas with nd.err < 1e-8 (nd err at w=30: 4.2e-14) |
| C1_y0.02 | sweep vs solve | 4.47e-17 | omega = 0.3 |
| C1_y0.02 | sweep vs solve | 1.45e-16 | omega = 3 |
| C1_y0.02 | sweep vs solve | 2.01e-16 | omega = 30 |
| C1_y0 | ND modulus referee | 1.10e-13 | max |d|F|| over 121/121 omegas with nd.err < 1e-8 (nd err at w=30: 3.7e-14) |
| C1_y0 | sweep vs solve | 2.00e-16 | omega = 0.3 |
| C1_y0 | sweep vs solve | 0.00e+00 | omega = 3 |
| C1_y0 | sweep vs solve | 0.00e+00 | omega = 30 |

## the caustic family (structural failure)

At `y = 0` the image finder returns no images (the two solutions sit on the critical curve and are discarded), so `|F_geo| = 0`; the exact `|F|` grows as a power law over the band with fitted slope **0.468** (`sqrt(omega)` for the Einstein-ring stationary manifold; `|F| = 1.243 .. 9.708`).

| y | tuned V1 max err | geo-anchor max err | F=1 max err |
|---|------------------|--------------------|-------------|
| 0.2 | 4.087e-01 | 1.034e+00 | 3.937e+00 |
| 0.1 | 8.620e-01 | 1.741e+00 | 4.964e+00 |
| 0.05 | 1.384e-01 | 2.767e+00 | 1.211e+00 |
| 0.02 | 1.902e-01 | 4.832e+00 | 1.205e+00 |
| 0 | 1.000e+00 | 1.000e+00 | 1.204e+00 |

The largest tuned-bridge error in the family sits at a *fringe minimum* (C1_y0.1, omega = 23.83: |F| = 0.244, |F_geo| = 0.256 — a near-cancellation of the two images), where any additive bridge inherits the geometric cancellation; the absolute error there is small and the RMS column averages this out.  The fit metric is the max relative error, so the floors above are conservative at such nulls.

