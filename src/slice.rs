//! Cartesian-slice reduction of the 2D point-mass field (Path 2, milestone
//! M3, option A).
//!
//! The 2D Fresnel integral
//! $I(\beta) = \iint e^{i\omega\phi(x_0,x_1)}\,dx_0\,dx_1$ is evaluated as an
//! outer integral over $x_1$ of an **inner 1D thimble** in $x_0$:
//! $$ I = \int_{\mathbb R} G(x_1)\,dx_1, \qquad
//!    G(x_1) = \int_{\mathbb R} e^{i\omega\phi(z_0, x_1)}\,dz_0. $$
//! This module provides two inner kernels. The *exact* one (A0', since
//! 2026-10-11) is [`slice_inner_rc`]: a real-axis core spanning every
//! branch-point abscissa plus tails rotated by $\pi/4$ into the Gaussian
//! valleys — no branch point is ever swept, so the deformation is exact by
//! Cauchy with no sheet/Stokes bookkeeping (see the Journal entry of
//! 2026-10-11 for the measurements that motivated the switch). The thimble
//! kernel [`slice_inner`] (A0) remains as the high-$\omega$ path — its
//! $O(1)$-in-$\omega$ cost is not needed below $\omega \sim$ a few hundred,
//! where [`slice_inner_rc`] is both faster and exact — with the
//! swept-branch-point indicator [`branch_point_indicator`] wired into its
//! error budget so it can no longer report $10^{-13}$ while being 80% wrong.
//!
//! At a fixed $x_1 = c$ the slice phase in $z_0$ is
//! $\phi_c(z_0) = \tfrac12(z_0-\beta_0)^2 + \tfrac12(c-\beta_1)^2
//!  - \sum_j \tfrac{m_j}{2}\ln w_j$, with $w_j(z_0) = (z_0-a_j)^2 + (c-b_j)^2$.
//! Its derivative is rational, $\phi_c' = N/D$ with $D = \prod_j w_j$ and
//! $\deg N = 2N+1$ ($N$ = number of masses), so every stationary point is a
//! root of a single polynomial — the complete saddle source
//! [`ThimbleSolver::new_rational`] consumes. The measure is plain $dz_0$, so
//! the inner amplitude is $1$ (exactly what `ThimbleSolver` uses), unlike the
//! polar wrapper's $r\,dr$.
//!
//! The log branch points sit at $z_0 = a_j \pm i(c - b_j)$, off the real axis
//! for a generic slice ($c \ne b_j$) and pinching onto it as the slice passes
//! through a mass — the outer integrator (milestone A1) treats those $c = b_j$
//! crossings as panel breakpoints.

use crate::adaptive::{AdaptiveConfig, integrate};
use crate::phase::{Phase, RationalPhase};
use crate::phases::PointMassField;
use crate::roots::{poly_add, poly_mul, poly_scale};
use crate::thimble::{ThimbleConfig, ThimbleSolver};
use crate::{Error, Output, Result};
use nalgebra::SVector;
use num_complex::Complex64;
use std::sync::atomic::{AtomicUsize, Ordering};

fn poly_sub(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    poly_add(a, &poly_scale(b, Complex64::new(-1.0, 0.0)))
}

/// The 2D point-mass field phase restricted to the line $x_1 = $ `x2`, viewed
/// as a 1D phase in the (complexified) coordinate $z_0$. The outer coordinate
/// is stored as a [`Complex64`] so the same slice serves a real-line outer
/// quadrature (A1) and, later, a deformed complex outer contour (A2).
pub struct X1SlicePhase<'a> {
    pub field: &'a PointMassField,
    pub x2: Complex64,
}

impl<'a> X1SlicePhase<'a> {
    fn embed(&self, z0: Complex64) -> SVector<Complex64, 2> {
        SVector::<Complex64, 2>::new(z0, self.x2)
    }

    /// Ascending coefficients of the denominator $D(z_0) = \prod_j w_j$ and of
    /// the cleared numerator $N(z_0)$ of $\phi_c'(z_0) = N/D$. Exposed for the
    /// $N = \phi'\!\cdot D$ identity test.
    pub fn stationary_fraction(&self) -> (Vec<Complex64>, Vec<Complex64>) {
        let one = Complex64::new(1.0, 0.0);
        let b0 = Complex64::from(self.field.beta[0]);

        // w_j(z0) = z0^2 - 2 a_j z0 + (a_j^2 + (c - b_j)^2), ascending.
        let wj: Vec<Vec<Complex64>> = self
            .field
            .lenses
            .iter()
            .map(|&(a, b, _)| {
                let a = Complex64::from(a);
                let dv = self.x2 - b;
                vec![a * a + dv * dv, -2.0 * a, one]
            })
            .collect();

        // D = prod_j w_j.
        let mut d = vec![one];
        for w in &wj {
            d = poly_mul(&d, w);
        }

        // N = (z0 - beta0) D - sum_j m_j (z0 - a_j) prod_{k != j} w_k.
        let mut n = poly_mul(&[-b0, one], &d);
        for (j, &(a, _, m)) in self.field.lenses.iter().enumerate() {
            let mut without = vec![one];
            for (k, w) in wj.iter().enumerate() {
                if k != j {
                    without = poly_mul(&without, w);
                }
            }
            let term = poly_mul(&[Complex64::from(-a), one], &without);
            n = poly_sub(&n, &poly_scale(&term, Complex64::from(m)));
        }
        (n, d)
    }
}

impl<'a> Phase<1> for X1SlicePhase<'a> {
    fn value(&self, z: &SVector<Complex64, 1>) -> Complex64 {
        // Same phase as `field.value` on the slice, but the log of each mass is
        // evaluated in FACTORED form ln(z0 - p+) + ln(z0 - p-) with
        // p± = a ± i(c - b) the two branch points, instead of the composite
        // ln((z0-a)^2 + (c-b)^2). Both agree on the real axis (where the two
        // factors are conjugate and the sum is real), but the composite's
        // principal cut runs *vertically* from a ± i(c-b) — straight across the
        // diagonal descent thimbles, whose crossing makes Im(phi) jump by m*pi
        // and stalls the tracer. The factored cuts run *horizontally* (west)
        // from each branch point, clear of those thimbles. The gradient is
        // rational (single-valued), so it is unaffected by this choice.
        let z0 = z[0];
        let u = z0 - self.field.beta[0];
        let v = self.x2 - self.field.beta[1];
        let mut acc = 0.5 * (u * u + v * v);
        for &(a, b, m) in &self.field.lenses {
            let d = Complex64::i() * (self.x2 - b);
            let p_plus = a + d;
            let p_minus = a - d;
            acc -= 0.5 * m * ((z0 - p_plus).ln() + (z0 - p_minus).ln());
        }
        acc
    }

    fn gradient(&self, z: &SVector<Complex64, 1>) -> SVector<Complex64, 1> {
        // x_1 is held fixed, so the 1D derivative is just the (single-valued,
        // branch-independent) x_0 component of the 2D gradient.
        let g = self.field.gradient(&self.embed(z[0]));
        SVector::<Complex64, 1>::new(g[0])
    }
}

impl<'a> RationalPhase for X1SlicePhase<'a> {
    fn stationary_polynomial(&self) -> Vec<Complex64> {
        self.stationary_fraction().0
    }

    /// The $2N$ log branch points $z_0 = a_j \pm i(c - b_j)$ (zeros of $D$).
    fn poles(&self) -> Vec<Complex64> {
        let mut v = Vec::with_capacity(2 * self.field.lenses.len());
        for &(a, b, _) in &self.field.lenses {
            let d = Complex64::i() * (self.x2 - b);
            v.push(a + d);
            v.push(a - d);
        }
        v
    }
}

/// Inner integral $G(x_1) = \int_{\mathbb R} e^{i\omega\phi(z_0, x_1)}\,dz_0$
/// for a real slice position `x2`, as a full-line thimble sum with the
/// complete polynomial saddle source. Returns the **raw** integral (no
/// `free_space` prefactor).
pub fn slice_inner(field: &PointMassField, x2: f64, cfg: &ThimbleConfig) -> Result<Output> {
    let slice = X1SlicePhase {
        field,
        x2: Complex64::from(x2),
    };
    // The log potential is multivalued, so h and the quadrature must track phi
    // by continuation of its single-valued gradient.
    let omega = cfg.omega;
    let cfg = ThimbleConfig {
        continuous_phase: true,
        ..cfg.clone()
    };
    let mut out = ThimbleSolver::new_rational(&slice, cfg).run()?;
    // The saddle sum is blind to swept-branch-point (Hankel) contributions by
    // construction (homology incompleteness, not quadrature error); the
    // indicator bounds them, with a safety factor of 100 over the measured
    // 0.05-6x spread. This keeps the thimble kernel usable as the high-omega
    // path: it is trustworthy exactly where this budget is small.
    out.error += 100.0 * branch_point_indicator(field, x2, omega);
    Ok(out)
}

/// Swept-branch-point indicator of the *thimble* inner kernel
/// ([`slice_inner`]): an upper bound (up to a measured factor ~0.05-6x) on the
/// branch-cut (Hankel) contribution the disconnected saddle sum drops when
/// the deformation of `RR` onto the thimbles sweeps a log branch point.
///
/// For each branch point `p = a_j ± i(x2 - b_j)` it is `e^{omega h(p)}` with
/// `h = -Im phi` taken on the sheet reached *vertically* from `RR`: the
/// vertical segment from `(a_j, 0)` to `p` crosses no cut of the composite log
/// (whose cuts run vertically away from `RR`), and along it each `w_k` stays
/// in one half plane, so `field.value` at `p` is the continued value. Only
/// valley-side points (`h < 0`) can be swept — the descent cannot reach a
/// hill. Returns 1.0 at an exact pinch or for a non-positive `omega`
/// (conservative).
///
/// Measured on the two-mass sweep (2026-10-06, Journal entry 2026-10-11): the
/// thimble's true error stays within 0.05-6x of the indicator, and routing on
/// `100 * indicator < 1e-12 * |G|` never admitted a wrong slice.
pub fn branch_point_indicator(field: &PointMassField, x2: f64, omega: f64) -> f64 {
    if !(omega > 0.0) {
        return 1.0;
    }
    let mut worst = 0.0_f64;
    for &(a, b, _) in &field.lenses {
        let d = x2 - b;
        if d == 0.0 {
            return 1.0; // exact pinch: the branch point sits on RR
        }
        for s in [1.0_f64, -1.0] {
            let z = SVector::<Complex64, 2>::new(
                Complex64::new(a, s * d * (1.0 - 1e-9)),
                Complex64::from(x2),
            );
            let h = -field.value(&z).im;
            if h < 0.0 {
                worst = worst.max((omega * h).exp());
            }
        }
    }
    worst
}

/// Configuration of the RC (real core + rotated tails) slice kernels —
/// [`slice_inner_rc`] here, and the outer integral of the 2D field solver.
#[derive(Debug, Clone)]
pub struct SliceConfig {
    /// Frequency in `e^{i omega phi}`.
    pub omega: f64,
    /// The inner real-axis core spans every branch-point abscissa (and
    /// `beta_0`) plus this margin, widened by `|Im x2|` for complex slice
    /// positions so the rotated-tail wedges stay branch-point-free.
    pub margin: f64,
    /// The outer core of the 2D field integral spans the mass heights `b_j`
    /// (and `beta_1`) plus this margin.
    pub margin_outer: f64,
    /// Inner acceptance tolerance per unit length of `z_0`.
    pub tol_inner: f64,
    /// Outer acceptance tolerance per unit length of `x_2`.
    pub tol_outer: f64,
    /// Evaluate outer panels' nodes in parallel (bitwise identical to serial).
    pub parallel: bool,
}

impl Default for SliceConfig {
    fn default() -> Self {
        Self {
            omega: 1.0,
            margin: 1.5,
            margin_outer: 1.5,
            tol_inner: 1e-14,
            tol_outer: 1e-13,
            parallel: true,
        }
    }
}

/// Evaluation context of one inner solve. The slice position is carried as
/// `x2 = base + off` (`base` real) so that every `x2 - b_k` is formed as the
/// exact constant `(base - b_k) + off`: on the outer core (`off = ±rho^2`
/// real) this is what keeps `|x2 - b_j|` cancellation-free next to the pinch
/// heights, where a naive difference loses ~10 orders of relative precision
/// and stalls adaptive refinement.
struct InnerCtx<'a> {
    field: &'a PointMassField,
    /// `d_k = (base - b_k) + off` per mass, and `dbeta = (base - beta_1) + off`.
    dk: Vec<Complex64>,
    dbeta: Complex64,
    mass_sum: f64,
}

impl<'a> InnerCtx<'a> {
    fn new(field: &'a PointMassField, base: f64, off: Complex64) -> Self {
        let dk = field
            .lenses
            .iter()
            .map(|&(_, b, _)| Complex64::from(base - b) + off)
            .collect();
        let dbeta = Complex64::from(base - field.beta[1]) + off;
        let mass_sum = field.lenses.iter().map(|&(_, _, m)| m).sum();
        Self {
            field,
            dk,
            dbeta,
            mass_sum,
        }
    }

    /// Phase on the real-axis core at `x = x0 + sgn*r` — real arithmetic,
    /// valid when `x2` is real (`off.im == 0`). Every `x - a_k` is formed as
    /// the exact constant `(x0 - a_k) + sgn*r`.
    #[inline]
    fn phi_real(&self, r: f64, x0: f64, sgn: f64) -> f64 {
        let u = (x0 - self.field.beta[0]) + sgn * r;
        let d1 = self.dbeta.re;
        let mut v = 0.5 * (u * u + d1 * d1);
        for (k, &(a, _, m)) in self.field.lenses.iter().enumerate() {
            let t = (x0 - a) + sgn * r;
            let d = self.dk[k].re;
            v -= 0.5 * m * (t * t + d * d).ln();
        }
        v
    }

    /// Phase on the real-axis core at `x = x0 + sgn*r` — complex slice
    /// position (composite principal log; each `w_k` stays in the upper half
    /// plane, checked by the caller).
    #[inline]
    fn phi_complex(&self, r: f64, x0: f64, sgn: f64) -> Complex64 {
        let u = Complex64::from((x0 - self.field.beta[0]) + sgn * r);
        let mut v = 0.5 * (u * u + self.dbeta * self.dbeta);
        for (k, &(a, _, m)) in self.field.lenses.iter().enumerate() {
            let t = Complex64::from((x0 - a) + sgn * r);
            let w = t * t + self.dk[k] * self.dk[k];
            v -= 0.5 * m * w.ln();
        }
        v
    }

    /// Phase at complex `z_0` (rotated-tail segments), composite principal log.
    #[inline]
    fn phi_tail(&self, z: Complex64) -> Complex64 {
        let u = z - self.field.beta[0];
        let mut v = 0.5 * (u * u + self.dbeta * self.dbeta);
        for (k, &(a, _, m)) in self.field.lenses.iter().enumerate() {
            let t = z - a;
            let w = t * t + self.dk[k] * self.dk[k];
            v -= 0.5 * m * w.ln();
        }
        v
    }
}

/// The RC inner solve at `x2 = base + off` (the workhorse behind
/// [`slice_inner_rc`] and the outer integral's anchored slice positions):
///
/// * **core** — the real segment `[A, B]`, `A <= min(beta_0, a_j) - margin`,
///   `B >= max(beta_0, a_j) + margin` (widened by `|Im x2|`), each
///   half-interval parametrised from its own breakpoint;
/// * **tails** — `B + e^{i pi/4} t` and `A - e^{i pi/4} t`, `t >= 0`, into the
///   Gaussian valleys, with a decay probe so a large total mass cannot leave a
///   non-negligible truncation unnoticed.
///
/// No branch point lies in the swept wedges (`Re p_± = a_j ∓ Im(x2 - b_j)`),
/// and along the tails every `w_k` stays in the open upper half plane, so the
/// principal log is the analytic continuation of the physical value: the
/// deformation never leaves the physical sheet. Returns the solve and its
/// node-evaluation count.
fn inner_rc(
    field: &PointMassField,
    base: f64,
    off: Complex64,
    cfg: &SliceConfig,
) -> Result<(Output, usize)> {
    let omega = cfg.omega;
    if !(omega > 0.0) || !omega.is_finite() {
        return Err(Error::InvalidInput("omega must be > 0 and finite"));
    }
    if off.im != 0.0 {
        // Sector validity: Im(d_k^2) > 0 for every mass is what keeps each
        // w_k in the open upper half plane along the whole inner contour.
        // (Satisfied on the outer rotated tails by construction; a plain
        // complex x2 off those rays may violate it — reject it.)
        for &(_, b, m) in &field.lenses {
            if m == 0.0 {
                continue; // a zero-coefficient log has no branch point
            }
            let d = Complex64::from(base - b) + off;
            if (d * d).im <= 0.0 {
                return Err(Error::InvalidInput(
                    "complex slice position outside the validated sector: Im((x2-b)^2) <= 0",
                ));
            }
        }
    }
    Ok(inner_rc_unchecked(field, base, off, cfg))
}

/// [`inner_rc`] without the input validation — for callers that have already
/// established validity (`slice_field`: omega checked once, sector satisfied
/// by construction on its anchored positions).
fn inner_rc_unchecked(
    field: &PointMassField,
    base: f64,
    off: Complex64,
    cfg: &SliceConfig,
) -> (Output, usize) {
    let omega = cfg.omega;
    let ctx = InnerCtx::new(field, base, off);
    let real_slice = off.im == 0.0;

    // Core breakpoints: every branch-point abscissa (and beta_0), bracketed.
    let shift = off.im.abs();
    let (mut amin, mut amax) = (field.beta[0], field.beta[0]);
    for &(a, _, _) in &field.lenses {
        amin = amin.min(a);
        amax = amax.max(a);
    }
    let (aa, bb) = (amin - cfg.margin - shift, amax + cfg.margin + shift);
    let mut pts = vec![aa, bb];
    for &(a, _, _) in &field.lenses {
        pts.push(a);
    }
    pts.sort_by(|p, q| p.partial_cmp(q).unwrap());
    pts.dedup();

    // Roundoff floor at the ulp level of omega*phi (see `adaptive`'s docs);
    // `+ mass_sum` covers the log terms of large fields.
    let span = (aa - field.beta[0]).abs().max((bb - field.beta[0]).abs());
    let phimax = 0.5 * span * span + 0.5 * ctx.dbeta.norm_sqr() + 5.0 + ctx.mass_sum;
    let floor = 4e-16 * (1.0 + omega * phimax);
    let core_cfg = AdaptiveConfig {
        tol: cfg.tol_inner,
        floor,
        min_len: 1e-15,
        max_panels: 200_000,
    };
    let tail_cfg = AdaptiveConfig {
        tol: cfg.tol_inner,
        floor,
        min_len: 1e-14,
        max_panels: 200_000,
    };

    let mut value = Complex64::new(0.0, 0.0);
    let mut error = 0.0;
    let mut evals = 0usize;

    // Core: each half-interval is parametrised from its own breakpoint (both
    // halves map to dr in [0, half] with dx = +dr after orientation).
    for win in pts.windows(2) {
        let (lo, hi) = (win[0], win[1]);
        let half = 0.5 * (hi - lo);
        for (x0, sgn) in [(lo, 1.0), (hi, -1.0)] {
            let f = |r: f64| -> (Complex64, f64) {
                if real_slice {
                    let phi = ctx.phi_real(r, x0, sgn);
                    (Complex64::from_polar(1.0, omega * phi), 0.0)
                } else {
                    let phi = ctx.phi_complex(r, x0, sgn);
                    ((Complex64::i() * omega * phi).exp(), 0.0)
                }
            };
            let res = integrate(&f, 0.0, half, &core_cfg, false);
            value += res.value;
            error += res.error;
            evals += res.evals;
        }
    }

    // Rotated tails, pi/4 into the Gaussian valleys.
    let e4 = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
    let iw = Complex64::i() * omega;
    for (end_z, sgn) in [(bb, 1.0), (aa, -1.0)] {
        // z = end_z + sgn * e4 * t. On the right tail (sgn = +1) this is the
        // plain change of variables dz = e4 dt; on the left (sgn = -1) both
        // the Jacobian (-e4 dt) and the flipped t-orientation (integral over
        // t in [0, inf) runs z from A toward -infinite) contribute a sign, so
        // both tails enter as +e4 * integrand.
        let f = |t: f64| (e4 * (iw * ctx.phi_tail(end_z + sgn * e4 * t)).exp(), 0.0);
        // Decay probe: the Gaussian part gives e^{-45} at the default length,
        // but large total mass slows the decay — double until negligible.
        let mut t_end = (2.0 * 45.0 / omega).sqrt() + 2.0;
        let scale = f(0.0).0.norm().max(1e-30);
        let mut decayed = false;
        for _ in 0..5 {
            if f(t_end).0.norm() <= 1e-17 * scale {
                decayed = true;
                break;
            }
            t_end *= 2.0;
        }
        let res = integrate(&f, 0.0, t_end, &tail_cfg, false);
        value += res.value;
        error += res.error;
        evals += res.evals;
        // Truncation budget: with the probe satisfied the remaining tail is a
        // geometric continuation of the last resolved value; otherwise (a
        // pathological field) budget it linearly so the caller sees it.
        let end = f(t_end).0.norm();
        error += if decayed { 4.0 * end } else { t_end * end };
    }

    (Output { value, error }, evals)
}

/// Inner integral `G(x2) = integral_RR e^{i omega phi(z_0, x_2)} dif z_0` for
/// the point-mass field by the RC construction — the *exact* kernel of the
/// Cartesian-slice reduction (A0'; see the module docs). `x2` may be complex
/// (the outer rotated tails of the 2D field solver need that), provided
/// `Im((x2 - b_j)^2) > 0` for every mass. Returns the raw integral.
pub fn slice_inner_rc(field: &PointMassField, x2: Complex64, cfg: &SliceConfig) -> Result<Output> {
    Ok(slice_inner_rc_evals(field, x2, cfg)?.0)
}

/// [`slice_inner_rc`] with the node-evaluation count (diagnostics/benchmarks).
pub fn slice_inner_rc_evals(
    field: &PointMassField,
    x2: Complex64,
    cfg: &SliceConfig,
) -> Result<(Output, usize)> {
    inner_rc(field, x2.re, Complex64::new(0.0, x2.im), cfg)
}

/// Result of the full 2D field integral [`slice_field`].
#[derive(Debug, Clone, Copy)]
pub struct SliceFieldOutput {
    /// The raw integral $I = integral.double_{RR^2} e^{i omega phi} dif x_0 dif x_1$.
    pub value: Complex64,
    /// Conservative error estimate: outer panel disagreements plus every
    /// propagated inner error plus the tail truncation budgets.
    pub error: f64,
    /// Number of inner solves performed (the cost driver; one per outer node).
    pub inner_solves: usize,
}

/// The full 2D point-mass-field integral — the RC construction in *both*
/// variables (option A, milestone A1):
///
/// * **outer core** — the real segment in $x_2$ spanning every pinch height
///   $b_j$ (and $beta_1$), each half-interval parametrised from its breakpoint
///   as $x_2 = x_0 ± rho^2$ (`dx_2 = 2 rho dif rho`), which both softens the
///   $|x_2 - b_j|^{1 - i omega m_j}$ cusp into a quadrature-smooth integrand
///   and hands the inner solve its `(base, offset)` pair so $x_2 - b_j$ is
///   formed exactly;
/// * **outer tails** — $x_2 = B_2 + e^{i pi/4} s$ and $A_2 - e^{i pi/4} s$,
///   on which the inner integral runs at *complex* $x_2$: there
///   $"Im"(x_2 - b_j)^2 > 0$, so every $w_k$ stays in the upper half plane
///   along the whole inner contour and the principal log remains the analytic
///   continuation — the inner core widens by $|"Im"\,x_2|$ to keep the inner
///   tail wedges branch-point-free.
///
/// The outer integrand reports the inner solve's error through the Jacobian,
/// so the returned error budget is the full nested estimate. `cfg.parallel`
/// evaluates outer panels' nodes (i.e. inner solves) with rayon, bitwise
/// identical to the serial evaluation. Returns the **raw** integral —
/// normalisation to the free-space amplification happens in the caller
/// (the Python layer, as for the other solvers).
pub fn slice_field(field: &PointMassField, cfg: &SliceConfig) -> Result<SliceFieldOutput> {
    let omega = cfg.omega;
    if !(omega > 0.0) || !omega.is_finite() {
        return Err(Error::InvalidInput("omega must be > 0 and finite"));
    }
    if !(cfg.margin > 0.0) || !cfg.margin.is_finite()
        || !(cfg.margin_outer > 0.0) || !cfg.margin_outer.is_finite()
    {
        return Err(Error::InvalidInput("margins must be > 0 and finite"));
    }
    for &(a, b, m) in &field.lenses {
        if !a.is_finite() || !b.is_finite() || !m.is_finite() {
            return Err(Error::InvalidInput("lens positions and masses must be finite"));
        }
    }
    // Zero-mass lenses contribute nothing (a zero-coefficient log has no
    // branch point); dropping them once here keeps their abscissae out of
    // every breakpoint list below.
    let eff_field = PointMassField {
        beta: field.beta,
        lenses: field
            .lenses
            .iter()
            .copied()
            .filter(|&(_, _, m)| m != 0.0)
            .collect(),
    };

    // Outer breakpoints: every pinch height b_j (and beta_1), bracketed.
    let (mut bmin, mut bmax) = (field.beta[1], field.beta[1]);
    for &(_, b, _) in &eff_field.lenses {
        bmin = bmin.min(b);
        bmax = bmax.max(b);
    }
    let (a2, b2) = (bmin - cfg.margin_outer, bmax + cfg.margin_outer);
    let mut pts = vec![a2, b2];
    for &(_, b, _) in &eff_field.lenses {
        pts.push(b);
    }
    pts.sort_by(|p, q| p.partial_cmp(q).unwrap());
    pts.dedup();

    let mass_sum: f64 = eff_field.lenses.iter().map(|&(_, _, m)| m).sum();
    let span = (a2 - field.beta[1]).abs().max((b2 - field.beta[1]).abs());
    let floor = 4e-16 * (1.0 + omega * (0.5 * span * span + 10.0 + mass_sum));
    let outer_cfg = AdaptiveConfig {
        tol: cfg.tol_outer,
        floor,
        min_len: 1e-12,
        max_panels: 200_000,
    };
    let solves = AtomicUsize::new(0);

    let mut value = Complex64::new(0.0, 0.0);
    let mut error = 0.0;

    // Outer core: anchored halves x2 = x0 + sgn*rho^2. Both halves map to
    // drho in [0, rmax] with a + Jacobian 2*rho after orientation (the
    // right-anchored half traverses [mid, hi] backwards in rho).
    for win in pts.windows(2) {
        let (lo, hi) = (win[0], win[1]);
        let rmax = (0.5 * (hi - lo)).sqrt();
        for (x0, sgn) in [(lo, 1.0), (hi, -1.0)] {
            let f = |rho: f64| -> (Complex64, f64) {
                solves.fetch_add(1, Ordering::Relaxed);
                let off = Complex64::new(sgn * rho * rho, 0.0);
                let (o, _) = inner_rc_unchecked(&eff_field, x0, off, cfg);
                let jac = 2.0 * rho;
                (o.value * jac, o.error * jac)
            };
            let res = integrate(&f, 0.0, rmax, &outer_cfg, cfg.parallel);
            value += res.value;
            error += res.error;
        }
    }

    // Outer tails: x2 = end + sgn*e4*s. Same sign bookkeeping as the inner
    // tails (Jacobian and orientation flips cancel), so both contribute
    // + e4 * G.
    let e4 = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
    for (end_x2, sgn) in [(b2, 1.0), (a2, -1.0)] {
        let f = |s: f64| -> (Complex64, f64) {
            solves.fetch_add(1, Ordering::Relaxed);
            let off = e4 * Complex64::from(sgn * s);
            let (o, _) = inner_rc_unchecked(&eff_field, end_x2, off, cfg);
            (o.value * e4, o.error) // |dz2/ds| = |e4| = 1
        };
        // Decay probe (the s^2/2 self-term of the rotated ray gives e^{-45}
        // at the default length; a huge total mass can slow it — double until
        // negligible, then budget the truncation).
        let mut s_end = (2.0 * 45.0 / omega).sqrt() + 2.0;
        let scale = f(0.0).0.norm().max(1e-30);
        let mut decayed = false;
        for _ in 0..5 {
            if f(s_end).0.norm() <= 1e-17 * scale {
                decayed = true;
                break;
            }
            s_end *= 2.0;
        }
        let res = integrate(&f, 0.0, s_end, &outer_cfg, cfg.parallel);
        value += res.value;
        error += res.error;
        let end = f(s_end).0.norm();
        error += if decayed { 4.0 * end } else { s_end * end };
    }

    Ok(SliceFieldOutput {
        value,
        error,
        inner_solves: solves.into_inner(),
    })
}
