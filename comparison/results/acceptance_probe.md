# The acceptance-probe experiment (item 6): GL10 probe — no-go

Dataset: `acceptance_probe.csv` (example `examples/acceptance_probe.rs`,
2026-10-18; load-independent node counts — `inner_solves` for the 2D RC
engine, `evals` for the kernels that expose them).  **Verdict: no-go — do not
adopt.**  The prototype stays behind `SliceConfig::gl10_probe` (default
`false`) for reproducibility; the value/budget semantics of the classic path
are untouched (bitwise baselines green).

## What was tried

Variant A (as specified in the plan): evaluate the GL20 whole panel (20
nodes) and probe it with GL10 halves (2 × 10 nodes); on agreement accept the
GL20 value (40 nodes per accepted panel instead of the classic 60); on
disagreement evaluate the GL20 halves and continue exactly as before (80
nodes for that attempt).

## Why it cannot win (the telescoping argument)

Let `A` be the number of accepted panels and `R = A − 1` the rejected ones
(a binary refinement tree).  Classic cost = 20 + 40·(A + R) = 80A − 20.
Probe cost = 20 + 20·(A′ + R′) + 40·R′ = 80A′ − 40.  The difference is
`80(A′ − A) − 20`: the probe wins only if it accepts *strictly fewer* panels
than the classic rule (A′ ≤ A, best case a 20-node saving, ~1%).  A cheaper
proxy is necessarily weaker — the probe compares GL10 halves against the GL20
whole (≈ the GL10 error) while the classic compares GL20 halves against the
GL20 whole (≈ the much smaller GL20 error) — so the probe's tree is always at
least as fine (A′ ≥ A) and the typical outcome is a loss.

## Measured (ratio = classic/probe; < 1 means the probe costs more)

| case | ω | classic | probe | ratio | rel diff |
|---|---|---|---|---|---|
| slice_field two-mass | 1 | 1846 | 1926 | 0.96 | 8.9e-17 |
| slice_field two-mass | 3 | 2566 | 2646 | 0.97 | 9.3e-17 |
| slice_field two-mass | 10 | 3846 | 4006 | 0.96 | 1.1e-15 |
| slice_field two-mass | 30 | 5046 | 5846 | 0.86 | 1.2e-15 |
| slice_field single mass | 1 | 1326 | 1366 | 0.97 | 1.0e-16 |
| slice_field single mass | 10 | 2606 | 2726 | 0.96 | 1.6e-15 |
| slice_field single mass | 30 | 3886 | 4806 | 0.81 | 3.0e-15 |
| slice_field single mass | 100 | 10286 | 12486 | 0.82 | 8.5e-15 |
| slice_field single mass | 300 | 27886 | 34486 | 0.81 | 2.6e-14 |
| lens_field macro type I | 3 | 3086 | 3206 | 0.96 | 3.2e-17 |
| lens_field macro type II | 3 | 3166 | 3286 | 0.96 | 2.1e-16 |
| lens_field macro type III | 3 | 3086 | 3206 | 0.96 | 1.5e-15 |
| lens_field sis | 3 | 526 | 566 | 0.93 | 0 |
| lens_field cis | 3 | 526 | 486 | 1.08 | 0 |
| lens_field nfw | 3 | 526 | 486 | 1.08 | 2.1e-16 |
| slice_field_batch sweep (5 ω) | 10 | 14680 | 17600 | 0.83 | 8.9e-17 |
| slice_field_batch map (3×3) | 3 | 2740 | 2780 | 0.99 | 8.7e-16 |
| rc_1d_batch P-LPI grid (512) | 10 | 3400 | 4640 | 0.73 | 0 |
| slice_inner_rc x2=0.2 (real) | 10 | 880 | 880 | 1.00 | 1.5e-15 |
| slice_inner_rc x2=1.5+e4·0.5 | 10 | 800 | 800 | 1.00 | 1.0e-15 |

The probe is 0–37% *more* expensive where it matters (single mass ω = 300:
0.81; P-LPI grid: 0.73; batch sweep: 0.83); the only wins are the small
profile cases (CIS/NFW at ω = 3, 8%, absolute counts ~500), and the inner
kernels are neutral (1.00).  Values agree at the quadrature level
(`rel_diff ≤ 2.6e-14`): the implementation is sound, the strategy is not.

## Variant B projection (not implemented)

B would spend 30 nodes per popped panel (GL20 whole + GL10 whole on the same
panel) → cost = 60A′ − 30.  Using the measured tree inflation
`A′ = (probe_count + 40)/80` per row, B projects to 0.77–1.02 of the classic
count (best ≈ 23% at the low-ω two-mass case, typical ≈ 10%, a *loss* on the
P-LPI grid) — below the 1.2× gate, with a substantially larger
implementation (the stack would carry both rules' data).  Recorded as a
projection rather than measured, per the plan's kill criterion.
