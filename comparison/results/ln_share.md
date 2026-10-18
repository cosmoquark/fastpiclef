# Item 8, Step 0 — the `ln` share of the many-mass per-node phase sum

Harness: `examples/ln_share.rs` → `comparison/results/ln_share.csv`
(measured 2026-10-19; the box ran at loadavg ~9.7, so only the *ratios* to
variant A are reportable).

## Method

A faithful replication of the production real-core phase sum
(`src/slice.rs::PointMassEval::phi_core_real`, with the per-slice `d_k`
precomputed once per slice, exactly as the engine does) over N point
components: 8 slice positions × 1024 core nodes each, four variants —

- **A** — libm `ln` (the production path);
- **B** — the log replaced by a bare multiply (an absolute ceiling);
- **C** — a scalar bit-hack log (exponent extraction + atanh series);
- **D** — a 4-wide AVX2 version of C over a structure-of-arrays layout (a
  *timing proxy* for the vector log item 8 would build).

Five interleaved repetitions per (N, variant); the minimum wall per variant
and the ratio to A.  The proxies' accuracy on a log-uniform sample: max
absolute error 1.24e-5 (max relative 9.8e-3 away from `x ≈ 1`); the
accumulated totals agree with A to 3e-6 (N = 10) … 5e-4 (N = 1000) relative,
so the speedups are not an artefact of a broken log.

## Numbers

| N | A ns/comp | B ns/comp | C ns/comp | D ns/comp | A/B | A/C | A/D |
|---|---|---|---|---|---|---|---|
| 10 | 18.8 | 2.8 | 13.3 | 5.9 | 6.6 | 1.4 | 3.2 |
| 100 | 20.0 | 2.9 | 16.8 | 4.4 | 6.8 | 1.2 | 4.5 |
| 1000 | 19.7 | 3.0 | 16.1 | 4.2 | 6.6 | 1.2 | 4.7 |

## Findings

- **The log is ~85% of the per-node cost** (`1 − 1/(A/B)`) at every N — the
  budget arithmetic is confirmed: the per-node O(N) real logs are ~the whole
  cost.
- **A 4-wide vector log already runs the phase sum 3.2–4.7× faster** (D).
  A shipped ≤1-ulp log would be somewhat slower than this ~1e-5 proxy, so
  treat 3–4.5× as the optimistic end of the phase-sum gain.
- The scalar bit-hack (C) gains only 1.2–1.4×: the win is the *vector* path,
  not a cheaper scalar log.
- **No dependency is required**: variant D uses `std::arch` AVX2 only
  (runtime-detected, scalar fallback).  This is what D1 should now be read
  against.

## Caveats

- These are per-node ratios.  The solve-level gain additionally needs the
  phase-sum share of the whole solve; the plan's Step 3 (N = 100 wall in a
  quiet window, ≥ 1.7× gate) is the decisive solve-level measurement.  The
  P4.2 reconnaissance (O(N²) node slots with an O(N) per-node cost) makes a
  phase-sum-dominated solve plausible at N = 100.
- Only the real-core path is timed (the tails' complex log is a small node
  fraction; the cores are the bulk of the RC node slots).
