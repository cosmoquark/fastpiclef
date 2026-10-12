# Difficult-case taxonomy (P3)

| case | definition | params | our rel err | cost [s] | budget | referee | note |
|---|---|---|---|---|---|---|---|
| near_caustic_airy | Airy cubic phase at/near the caustic (y->0) | y = 0.05 (thimble) and y = 0.0 (degenerate saddle: the thimble refuses by design, the mesh runs) | 3.2e-15 | 0.01 | 1.1x | Airy closed form (external) | y=0.05 via thimble: 1.7e-15; thimble at y=0.0: RuntimeError (degenerate saddle (caustic); use the mesh-flow solver); y=0.0 via mesh: 3.2e-15 |
| near_caustic_2d | point mass, source on/near the caustic (y_crit = 1) | y in {0.9, 0.99, 1.0, 1.01}, w = 10 | 2.4e-15 | 0.80 | 17.4x | Nakamura-Deguchi (mpmath dps 30) | FIONA at the caustic: rel 3.0e-03 (3.0 s) |
| near_pinch_slice | inner RC slice at the pinch height x2 = b_j | x2 = b + {1e-6, 1e-9, 1e-12, 0}, w = 10 | 4.7e-15 | 0.00 | 1.0x | Python RC referee at margin 4 (different contour) | kernel-vs-referee; worst Cauchy margin spread 2.1e-15 |
| large_omega | high frequency (cost grows ~w^1.5; see S3/P4.1) | w in {100, 300}, y = 1 | 2.5e-14 | 77.58 | 232.5x | Nakamura-Deguchi (mpmath dps 30) | FIONA's node count also grows with w (see C1 sweep) |
| large_offset | source far outside the lens (weak field) | y in {5, 10}, w = 10 | 1.9e-14 | 1.94 | 48.0x | Nakamura-Deguchi (mpmath dps 30) | FIONA at y=5: rel 1.6e-01 (2.6 s) |
| low_omega | low frequency (F - 1 is the physical quantity) | w in {1e-3, 1e-4}, y = 0.5 | 6.9e-16 | 0.08 | 42.0x | Nakamura-Deguchi (mpmath dps 30) | worst rel error on F - 1: 7.0e-16; the optional free-space subtraction is not used (P4) |
| many_masses | many point masses (per-node cost O(N)) | N in {25, 100}, w = 1, m_j = 0.02 | 7.9e-16 | 294.06 | 92.1x | margin-family Cauchy spread | cost scaling: N=25: 16.4 s, N=100: 294.1 s |
| near_critical_macro | macro model near critical (lambda0 = 0.05: flat Gaussian) | kappa = 0.45, gamma1 = 0.5, w = 3 | 8.5e-16 | 0.43 | 34.6x | margin-family Cauchy spread | long tails / slow decay: the cost is the price of exactness |
| saddle_macro | type-II saddle macro image (lambda0 = -0.2) | kappa = 0.5, gamma1 = 0.7, w = 10 | 5.5e-15 | 0.89 | 15.6x | margin-family Cauchy spread | FIONA errs 29-49% here (P0 finding of record) - never use it as truth |
| small_core_cis | CIS with a tiny core (xc = 0.01; stiff potential) | psi0 = 1, xc = 0.01, w = 3 | 5.1e-16 | 0.09 | 44.8x | margin-family Cauchy spread | the P2 polar-referee anchor was at xc = 0.2 (P0 E5) |
| nfw | NFW halo (log^2 potential, u ~ 1 cancellation handled) | psi0 = 1, xs = 1, w in {1, 10}, y = 0.5 | 1.3e-15 | 0.13 | 22.0x | margin-family Cauchy spread (P2 tests: <= 8e-15 vs the Hankel referee) | glworia would be the third-party comparator (blocked by JAX drift, see notes) |
| mixed_field | stars + halo + macro (the science configuration) | point + NFW + (kappa=0.3, gamma1=0.2), w = 3 | 6.2e-16 | 0.52 | 29.6x | margin-family Cauchy spread |  |
| maximum_macro | type-III maximum macro image (both lambdas < 0) | kappa = 1.5, gamma1 = 0.2, w = 10 | 1.3e-10 | 1.11 | 43.9x | margin-family Cauchy spread | |I| = 1.4e-05: near-total cancellation, so the spread 1.3e-10 relative is ~1.8e-15 absolute (the roundoff floor of the integrand scale); FIONA cannot represent it (P0) |
