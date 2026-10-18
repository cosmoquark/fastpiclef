# Error budgets of the fastpiclef solvers

This is the P5 write-up of *every* error source and budget in the solver
family, with the mechanism that keeps each bound honest and the dataset that
pinned it.  It is the technical backbone for the later, readable report.

**Reading rule (project discipline).**  Every path returns `(value, error)`
with the contract: `error` is a *conservative* bound on the true error, and no
claim may be quoted without naming the dataset that supports it.  Self-agreement
is not truth: the mesh "oracle" once agreed with the slice kernels to 11% error
because they shared code.

---

## 1. The 2D RC engine (`slice_field`, `lens_field_solve`)

Geometry: a real core in `x2` anchored at every pinch height `b_j` (each mass
contributes one), plus two rotated tails `x2 = end ± e^{iπ/4}s`; for each `x2`,
an inner integral in `z0` with its own real core (anchored at every branch-point
abscissa `a_j`) and rotated tails.  The construction is exact by Cauchy for
`margin, margin_outer > 0` and `tail_angle ∈ (0, π/4]` (`> π/4` can cross a cut
in the mixed-sign saddle case and is rejected).

Budget composition (all terms conservative):
- **Per-panel quadrature differences** — GL20 with the embedded differences;
  `tol_outer = 1e-13` per unit length in `x2`, `tol_inner = 1e-14` per unit
  length in `z0` (defaults).  The accepted panel's budget is the *observed*
  disagreement `|l + r − whole|` plus both halves' propagated errors.  (A
  GL10-probe acceptance alternative was measured and rejected — item 6,
  `results/acceptance_probe.{csv,md}`: a cheaper acceptance proxy is
  necessarily weaker, and the resulting tree inflation outweighs the
  per-panel node saving.)
- **Nested propagation** — each outer node's value is an inner solve; its
  reported error is propagated through the Jacobian into the outer budget.
- **Tail truncation** — the outer tails probe their integrand until it decays
  below `1e-17` of the initial scale (doubling up to 5×); if it decayed the
  truncation is budgeted as `4·|G(s_end)|`, else `s_end·|G(s_end)|`.
- **Roundoff floors** — panels are accepted when the disagreement falls below
  `4e-16·(1 + ω·φ_scale)` (`φ_scale` = the quadratic span plus
  `potential_scale()` for the log terms), i.e. at the ulp level of the phase
  `ω·φ`; `min_len = 1e-12` (outer) / `1e-15` / `1e-14` (inner core / tails).
- **Panel caps** — `max_panels = 200_000`; the unrefined remainder is budgeted
  by its `|f|` mass and `capped` is reported.
- **Measure-zero pinches** — a node exactly on a pinched branch point is
  non-finite; it contributes `0` and its neighbours' `|f|` mass carries the
  budget.  The anchored panels make this a measure-zero event.

Independent arbiters (validation, not budget): the **α-family** (π/4 vs π/8:
a different contour, the same integral), the **margin** sweeps, and
**transposed slicing** (`tests/field.rs`, `tests/slice.rs`,
`tests/macro.rs::macro_alpha_family_and_margins_agree`).

**Low-ω free-space subtraction (`cfg.subtract_free`, O2).**  At ω ≤ 1e-2 the
direct path's `F − 1` is limited by the quadrature's accumulated `ε·|I_free|`
(the raw value is ~2π/ω while `F − 1` is ~ω).  The subtracted mode integrates
`e^{iωφ_free}(e^{−iωΨ} − 1)` (Ψ = the profile/mass potential; `e^{−iθ} − 1`
evaluated as `−2i e^{−iθ/2} sin(θ/2)` to avoid the `ε/|θ|` cancellation) on
the *same* contour and adds the exact closed-form free/macro integral, whose
`8ε·|I_free|` roundoff joins the budget.  Measured `F − 1` relative errors
against the 30-digit ND referee (`results/low_omega_rust.csv`, 2026-10-18):
direct 0.9–1.9e-12 → subtracted 2.9e-15–4.7e-13 at ω = 1e-4; 1.2–2.5e-13 →
4.8e-16–6.2e-14 at 1e-3; ~8.9e-15 → 4–10e-16 at 1e-2.  The budgets cover the
true integral error by 95–10⁴× (subtracted) / 30–133× (direct); the subtracted
path's own floor is the final `free + J` addition (`~ε/|F − 1|`).  Batched
entry points reject the flag (single-field only).

**Vector log (item 8; `cfg.simd_log`, opt-in).**  The many-mass real-core
phase sums can replace the scalar libm `ln` with a 4-wide AVX2 log
(`src/simd_ln.rs`, hand-rolled `std::arch`, runtime-detected, scalar
fallback).  Accuracy: ≤ 1 ulp vs libm on the argument domain (`w = t² + d²`) —
measured 0.994 ulp relative for `|ln w| ≥ 1e-3`, 0.996 ulp near `w = 1`
(absolute ≤ 1.1e-19), pinned by the module tests.  The end-to-end agreement is
at the roundoff level (N = 1 vs the 30-digit ND referee < 1e-12; N = 3/10
scalar vs vector < 1e-13, with identical refinement trees), so the budget
class is unchanged.  Default `false` until the Step-3 wall gate (N = 100 in a
quiet window, at least ~1.7×) passes; the bitwise baselines are regenerated
deliberately if the default flips.

## 2. The 1D slice kernels (`slice_inner_rc`, `rc_1d_batch`)

Same structure in one variable, plus the **swept-branch-point indicator**
(`branch_point_indicator`): the homology-incompleteness risk (a branch point
swept between adjacent slices) is bounded by the indicator with a **100×
safety factor** (`src/slice.rs`); measured: the true error stayed within
0.05–6× of the indicator, so the 100× bound is honest but the indicator itself
must not be used alone.

## 3. The thimble (1D, `ThimbleSolver`, `solve_1d_hybrid`)

- Saddle set completeness is guaranteed for `RationalPhase` (the complete
  source) and approximate otherwise (multi-start Newton).
- `extra_err` accumulates: `e^{ω h}` for saddles skipped by the Morse
  threshold or whose ascent is inconclusive (window exit / step cap), and
  `e^{ω h(end)}` for traces truncated before the integrand died.
- The error also carries the contour-gap norms `|hi − lo|` and, in the hybrid,
  `100×` the branch-point indicator.
- `solve_1d_hybrid` falls back to the mesh-flow solver on
  `TraceStalled`/`DegenerateSaddle`; the mesh path's own budget is the
  `Solver` family's (not covered here).
- Validated against the Airy closed form and the quadratic phase
  (`tests/airy.rs`, `tests/thimble.rs`).

## 4. The polar solver (`solve_point_mass_polar`)

Axisymmetric point-mass potentials via `r = e^u` (removes the log branch
point); validated against the external high-precision **Hankel referee**
(`comparison/referee.py::hankel_referee_hp`) and the ND closed form.  Used as
a *referee* for the 2D engine (independent method, same physics).

## 5. The batched paths (`slice_field_batch`, `sweep`, `map`, `grid`)

- Frequencies are grouped into bands of ratio `band_ratio = 10` (each band gets
  one shared contour sized for its hardest member).
- `parallel` evaluation is **bitwise identical** to serial (fixed node order
  and reduction order; pinned in `tests/bitwise_baseline.rs` and
  `tests/batch.rs`).
- Accuracy is validated against per-point solves and the ND/series references
  (`tests/batch.rs::batch_entries_match_scalar_on_uniform_and_jittered_grids`,
  `sweep_uniform_omega_uses_series_and_matches_scalar`).

## 6. The high-ω hybrid (P4.1, `lens_field_hybrid`, `point_mass_field_hybrid`)

- Value: geometric sum with the stationary-phase next-order correction,
  `F = Σ_j G0_j (1 + C_j/(iω))`.
- Budget: `3 × Σ_j |G0_j| |C_j| / ω`; the *raw* sum covered the corrected
  residual by **≥ 3.5×** in every measured family (point-mass family
  17–4700×, 3-mass field 3.5×, macro exact), so the reported budget keeps a
  ≥ 10× margin on the measured minimum (`comparison/results/hybrid_check.csv`,
  `tests/hybrid.rs`).
- Switch: the geometric branch is taken only when the budget fits `tol`
  (default `tol = 1e-9` stays on the exact RC); a **caustic** (degenerate
  image) or a failed certificate returns `None`/falls back.
- Caveat: the budget covers the *asymptotic* error at the found images, not a
  **missing** image.  The finder is completeness-certified for the general
  model too (O1, 2026-10-19): non-degeneracy (`|det H| ≥ 1e-8`), the index
  identity, and per-cusp-group companion checks — a violated certificate is a
  loud error, never a silently short list (see §9; the certification was added
  after the old dense grid was found to silently miss the faint image of the
  unit point mass at y = 1: 9.6% error with a 1.5e-3 budget).

## 7. The geometric-optics path (`geometric`, `images`)

`.error = 0` by definition: it is an asymptotic form, not a quadrature.  The
signed-magnification sum rule was tested as a completeness certificate and
**rejected** (it fails for multi-point-mass fields — 0.877 for the 3-mass case
— and for cuspy/infinite-mass profiles: SIS 2.000, NFW 2.01).  The certificate
actually used is the topological **index identity** plus per-cusp-group
companion checks (§9): for the general model the identity's right-hand side is
`sign(λ0λ2) − #(active cusp groups)` (a cusp group is active when its dominant
class is a point, or an SIS/CIS{xc=0} cusp with the source inside its Einstein
ring; cored/smooth profiles never; coincident cusps grouped).

## 8. Validation evidence (external referees)

| reference | what it pins | dataset |
|---|---|---|
| Nakamura–Deguchi closed form (mpmath, dps 30) | isolated point mass, `|F|` | `tests/point_mass.rs`, `tests/field.rs` |
| mass-sheet ND reduction | macro + point mass (complex) | `tests/macro.rs` |
| high-precision polar/Hankel referee (dps 45) | SIS/CIS/NFW profiles | `tests/macro.rs`, `comparison/results/p0_profiles.csv` |
| wolensing (`refs/wolensing` @7b9101f) | third-party arbiter (1e-3…1e-2) | `comparison/results/wolensing_compare.csv` |
| the α-family / margins / transposed slicing | internal Cauchy arbiters | `comparison/results/p0_macro.csv` |
| bitwise baselines (pinned bits) | refactor safety | `tests/bitwise_baseline.rs` |

Measured floors: our own values agree with the referees at 5.8e-16…1.8e-15
across the comparison set; the competitors sit at ~1e-3 (FIONA, geometry
limited), ~6e-9 (P-LPI floor), 1e-3…1e-2 (wolensing).

## 9. Known limitations (not budgeted, by design or pending)

- **Image completeness**: **certified** for the general model as well (O1,
  2026-10-19).  `images` / `geometric` enforce non-degeneracy, the index
  identity `#(det H>0) − #(det H<0) = sign(λ0λ2) − #(active cusp groups)` and a
  per-cusp-group companion check with the *local* scale (`√(|m|/λ_loc)` /
  `|psi0|/λ_loc`, `λ_loc` the smooth Hessian eigenvalue at the centre).  A
  cusp group is active when its dominant class is a point (always) or an
  SIS/CIS{xc=0} cusp with the source inside its Einstein ring; the sign of the
  weight does not matter (the deflection is anti-holomorphic, `~1/d̄`; a
  negative weight is a π-rotation).  Coincident cusps form one group with the
  most singular class dominating; a locally defocusing cusp (no positive
  smooth-Hessian eigenvalue) or a repulsive one has no pinned companion.
  Profile components carry
  the index identity only (no pinned companion scale).  A violated certificate
  (or a caustic) is a loud error; `geometric_corrected` maps caustics to `None`
  and `hybrid` falls back to the exact RC.  Battery:
  `src/lens_field.rs::tests::finder_certificate_battery` (31 families with
  hand-derived RHS values) + the 40-mass cross-check against the point-mass
  finder.
- **Low-ω free-space subtraction**: landed in the Rust engine (O2,
  2026-10-18) — `SliceConfig::subtract_free`, honoured by `slice_field` /
  `lens_field_solve` and the Python `solve(..., subtract_free=True)`; the
  batched entry points reject it rather than ignoring it.  Measured floors and
  the mechanism are in §1; dataset `comparison/results/low_omega_rust.csv`.
- **Many-mass cost** (N-scaling, P4.2): the exact engine is ~O(N²) node slots;
  a scalable mode needs an FMM-with-winding or an explicit approximate mode
  (see the plan's deferred tiers).  Accuracy is not affected.
- **Potential-class boundary**: exactness holds for functions of a
  (complex) quadratic form per component + macro; SIE/EPL and arbitrary ψ are
  deferred tiers.
- **Contour-budget caveat**: `error` covers the *integral*, not a wrong model
  (e.g. a wrong parameter mapping).

## 10. Quote templates

- "Exact to machine precision: 5.8e-16…1.8e-15 against the referees
  (`comparison/results/wolensing_compare.csv`)."
- "High-ω branch: corrected residual ≤ budget/10, budget = 3× raw
  Σ|G0||C|/ω (`comparison/results/hybrid_check.csv`)."
- "RC: `error` includes the outer disagreements, the propagated inner budgets,
  the tail truncations and the ulp floor; α-family and margin arbiters agree
  to ~1e-15 (`tests/field.rs`, `tests/macro.rs`)."
