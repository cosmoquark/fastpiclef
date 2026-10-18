# Elliptical NFW + external shear -- prototype run

Case: M200=1e13 Msun, z_l=0.5, z_s=1.5, e1=0.2, e2=0, gamma=(0.03,0.01), y in [-1.5,1.5] arcsec, n=500.

| c200 | omega | solver | wall [s] | inner_solves | validation |
|---|---|---|---|---|---|
| 15 | 1 | fpl | 2.78 | 1520 | node_max_rel=7.4e-13, fpl_vs_gwlens=1.2e-13 |
| 15 | 10 | fpl | 11.36 | 5360 | node_max_rel=9.0e-12, fpl_vs_gwlens=2.7e-14 |
| 15 | 100 | fpl | 104.32 | 30320 | node_max_rel=3.6e-07, fpl_vs_gwlens=2.2e-12 |
| 30 | 1 | fpl | 3.66 | 1520 | node_max_rel=2.8e-12, fpl_vs_gwlens=4.5e-13 |
| 30 | 10 | fpl | 11.65 | 5200 | node_max_rel=4.8e-13, fpl_vs_gwlens=1.3e-13 |
| 30 | 100 | gwlens | 18.26 | 0 | node_max_rel=1.5e-12 |
| 60 | 1 | fpl | 3.33 | 1520 | node_max_rel=1.1e-12, fpl_vs_gwlens=5.5e-14 |
| 60 | 10 | fpl | 14.59 | 5360 | node_max_rel=1.8e-09, fpl_vs_gwlens=9.4e-10 |
| 60 | 100 | gwlens | 23.16 | 0 | node_max_rel=3.9e-13 |

fpl total for the 7 fpl panels: 151.7 s.  GWLENS total for the 2 GWLENS-only panels: 41.4 s (wall includes the adaptive grid build).  Hybrid total: 193.1 s.  This is not like-for-like: fpl's exact RC engine stalls for the compact NFW at omega=100 (c200=30, 60), so those two panels are GWLENS and are labelled as such in the figure.

Validation: `node_max_rel` compares each map at four of its own nodes with a high-accuracy point solve of the same solver (the maps are exact at their nodes); `fpl_vs_gwlens` compares six random source positions between the two independent solvers.  The dark interference nulls are narrower than the pixel spacing, so a *value between nodes* (or a nearest-pixel comparison) is not meaningful; at the node the two scalar solvers agree to ~1e-9 even at the nulls.

The exact reduction (elliptical NFW -> fpl's circular NFW + diagonal macro via x = Se u) was also checked against GWLENS directly at 5 sources x 3 frequencies: |dF|/|F| <= 6e-13.

Reference claims: FIONA 130 s, GWLENS 6.2 s for this figure (from the GWLENS authors; not reproduced here).  On this 6-core box GWLENS measures ~0.4 s map + a few seconds of grid build at omega=1,10, and ~11 s map + ~8 s build at omega=100, i.e. 18-21 s per omega=100 panel end-to-end.  A separate `--solver gwlens` run over all nine panels measured 50.1 s in total on this box.  The absolute numbers are hardware-dependent; the ratio is what the authors quote.
