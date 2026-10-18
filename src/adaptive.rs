//! Adaptive 1D Gauss-Legendre quadrature over panels, for the RC
//! (real core + rotated tails) kernels of [`crate::slice`].
//!
//! A panel `[a, b]` is evaluated with a 20-node Gauss-Legendre rule and
//! compared with the same rule on its two halves; it is accepted when the two
//! agree to `tol` per unit length, or to `floor` times the panel's `|f|` mass
//! (the roundoff floor — `e^{i omega phi}` carries an absolute phase error of
//! a few ulps of `omega*phi`, so panels must be accepted at that level, not at
//! the integrand's), or when it is shorter than `min_len` (its whole `|f|`
//! mass is then budgeted into the error).
//!
//! The integrand reports, next to its value, an *absolute error* of that value
//! (0 for an exactly-evaluated integrand; the outer field integral passes the
//! inner solve's error estimate through). Panel errors are summed
//! conservatively: an accepted panel contributes `|whole - halves|` plus the
//! propagated node errors.
//!
//! `parallel = true` evaluates a panel's nodes with rayon, collecting into a
//! fixed-order array before summing, so results are bitwise identical to the
//! serial evaluation (panels themselves are always processed sequentially).

use num_complex::Complex64;
use rayon::prelude::*;
use std::sync::OnceLock;

const GL_N: usize = 20;
/// Node count of the acceptance-probe rule (variant A of the acceptance
/// experiment: GL10 halves against the GL20 whole).
const GL10_N: usize = 10;

fn legendre_pair(n: usize, z: f64) -> (f64, f64) {
    // (P_n(z), P_n'(z)) by the three-term recurrence.
    let mut p0 = 1.0;
    let mut p1 = z;
    for k in 2..=n {
        let p2 = ((2 * k - 1) as f64 * z * p1 - (k - 1) as f64 * p0) / k as f64;
        p0 = p1;
        p1 = p2;
    }
    let dp = n as f64 * (z * p1 - p0) / (z * z - 1.0);
    (p1, dp)
}

/// Gauss-Legendre nodes/weights on `[-1, 1]` for `n` points (Newton from the
/// Chebyshev-like initial guesses). Exactness is unit-tested on monomials.
fn build_rule(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        loop {
            let (p, dp) = legendre_pair(n, z);
            let dz = p / dp;
            z -= dz;
            if dz.abs() < 1e-16 {
                break;
            }
        }
        let (_, dp) = legendre_pair(n, z);
        x[i] = z;
        w[i] = 2.0 / ((1.0 - z * z) * dp * dp);
    }
    (x, w)
}

/// The 20-node rule (the workhorse), computed once.
fn gl_rule() -> &'static (Vec<f64>, Vec<f64>) {
    static RULE: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    RULE.get_or_init(|| build_rule(GL_N))
}

/// The 10-node probe rule, computed once.
fn gl10_rule() -> &'static (Vec<f64>, Vec<f64>) {
    static RULE: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    RULE.get_or_init(|| build_rule(GL10_N))
}

/// Knobs of one adaptive integration.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveConfig {
    /// Acceptance tolerance per unit length of the integration variable.
    pub tol: f64,
    /// Roundoff floor: panels agreeing to this fraction of their `|f|` mass
    /// are accepted (phase-noise level, see the module docs).
    pub floor: f64,
    /// Panels shorter than this are accepted with their `|f|` mass budgeted.
    pub min_len: f64,
    /// Stop refining after this many panels have been evaluated; the
    /// unrefined remainder is budgeted into the error and `capped` is set.
    pub max_panels: usize,
    /// Acceptance-probe strategy (the item-6 experiment, **measured no-go** —
    /// see `comparison/results/acceptance_probe.md`): when set, a panel's
    /// GL10 halves are compared against the GL20 whole first, and the GL20
    /// value is accepted on agreement (40 nodes per accepted panel instead of
    /// the classic 60); on disagreement the validated GL20-halves recursion
    /// runs (80 nodes for that panel).  The weaker probe proxy inflates the
    /// refinement tree, which cancels the per-panel saving (the telescoping
    /// argument in the note); the flag exists to reproduce the measurement.
    pub gl10_probe: bool,
}

/// Result of one adaptive integration.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveResult {
    pub value: Complex64,
    /// Conservative error estimate (see the module docs).
    pub error: f64,
    /// Number of integrand-node evaluations.
    pub evals: usize,
    /// Whether `max_panels` stopped the refinement.
    pub capped: bool,
}

/// One GL-`n` evaluation of `f` on `[a, b]`: (integral, |f| mass, node
/// errors).  `n` is [`GL_N`] (the workhorse rule) or [`GL10_N`] (the
/// acceptance-probe rule).  Nodes are evaluated in a fixed order (rayon-
/// parallel when requested, bitwise identical to serial) and reduced
/// sequentially.
fn gl_eval<F>(f: &F, a: f64, b: f64, n: usize, parallel: bool) -> (Complex64, f64, f64)
where
    F: Fn(f64) -> (Complex64, f64) + Sync,
{
    let (x, w) = if n == GL_N {
        gl_rule()
    } else {
        debug_assert_eq!(n, GL10_N);
        gl10_rule()
    };
    let c = 0.5 * (a + b);
    let h = 0.5 * (b - a);
    let nodes: [f64; GL_N] = std::array::from_fn(|i| if i < n { c + h * x[i] } else { 0.0 });
    let mut vals = [(Complex64::new(0.0, 0.0), 0.0); GL_N];
    if parallel {
        vals[..n]
            .par_iter_mut()
            .zip(nodes[..n].par_iter())
            .for_each(|(slot, &t)| *slot = f(t));
    } else {
        for (slot, &t) in vals[..n].iter_mut().zip(nodes[..n].iter()) {
            *slot = f(t);
        }
    }
    let mut s = Complex64::new(0.0, 0.0);
    let mut mass = 0.0;
    let mut err = 0.0;
    for (v, &wi) in vals[..n].iter().zip(w[..n].iter()) {
        // A non-finite value (a node exactly on a pinched branch point)
        // contributes nothing.
        let v = if v.0.re.is_finite() && v.0.im.is_finite() {
            *v
        } else {
            (Complex64::new(0.0, 0.0), 0.0)
        };
        s += v.0 * wi;
        mass += v.0.norm() * wi;
        err += v.1 * wi;
    }
    (s * h, mass * h.abs(), (err * h).abs())
}

/// Adaptive GL20 on `[a, b]` (the integration variable need not be `x`; the
/// RC kernels use arc length / rotated-tail parameter / softened `rho`).
pub fn integrate<F>(f: &F, a: f64, b: f64, cfg: &AdaptiveConfig, parallel: bool) -> AdaptiveResult
where
    F: Fn(f64) -> (Complex64, f64) + Sync,
{
    let mut value = Complex64::new(0.0, 0.0);
    let mut error = 0.0;
    let mut evals = 0usize;
    let mut capped = false;

    let (whole, whole_mass, whole_err) = gl_eval(f, a, b, GL_N, parallel);
    evals += GL_N;
    // (lo, hi, whole value, whole |f| mass, whole propagated error)
    let mut stack = vec![(a, b, whole, whole_mass, whole_err)];
    while let Some((lo, hi, whole, whole_mass, whole_err)) = stack.pop() {
        if evals >= cfg.max_panels {
            // Budget the unrefined remainder by its |f| mass.
            value += whole;
            error += whole_err + whole_mass;
            capped = true;
            continue;
        }
        let mid = 0.5 * (lo + hi);
        let len = hi - lo;
        if cfg.gl10_probe {
            // Probe first: GL10 halves against the GL20 whole.  On agreement
            // accept the GL20 value (40 nodes per accepted panel instead of
            // 60); on disagreement fall through to the validated GL20-halves
            // recursion (the probe's 20 nodes are then wasted: 80 total).
            let (l10, lm10, le10) = gl_eval(f, lo, mid, GL10_N, parallel);
            let (r10, rm10, re10) = gl_eval(f, mid, hi, GL10_N, parallel);
            evals += 2 * GL10_N;
            let diff10 = (l10 + r10 - whole).norm();
            if diff10 <= cfg.tol * len
                || diff10 <= cfg.floor * (lm10 + rm10)
                || len < cfg.min_len
            {
                // The accepted value is the GL20 `whole`; its propagated node
                // errors are `whole_err` (the probe's are not part of it).
                value += whole;
                error += diff10 + whole_err + le10 + re10;
                continue;
            }
        }
        let (l, lm, le) = gl_eval(f, lo, mid, GL_N, parallel);
        let (r, rm, re) = gl_eval(f, mid, hi, GL_N, parallel);
        evals += 2 * GL_N;
        let diff = (l + r - whole).norm();
        if diff <= cfg.tol * len || diff <= cfg.floor * (lm + rm) || len < cfg.min_len {
            value += l + r;
            error += diff + le + re;
        } else {
            stack.push((lo, mid, l, lm, le));
            stack.push((mid, hi, r, rm, re));
        }
    }
    // Whatever is still on the stack when the panel budget ran out.
    while let Some((_, _, whole, whole_mass, whole_err)) = stack.pop() {
        value += whole;
        error += whole_err + whole_mass;
        capped = true;
    }
    AdaptiveResult { value, error, evals, capped }
}

/// Knobs of one *vector-valued* adaptive integration: like [`AdaptiveConfig`]
/// but with one roundoff `floor` per component (the RC batches run many
/// frequencies and slice positions at once, each carrying its own phase scale).
#[derive(Debug, Clone)]
pub struct AdaptiveVecConfig {
    /// Acceptance tolerance per unit length (all components).
    pub tol: f64,
    /// Per-component roundoff floors (accept a panel when every component
    /// agrees to its floor times its own `|f|` mass).
    pub floors: Vec<f64>,
    /// Panels shorter than this are accepted with their `|f|` mass budgeted.
    pub min_len: f64,
    /// Stop refining after this many node evaluations; the unrefined remainder
    /// is budgeted into the errors and `capped` is set.
    pub max_panels: usize,
    /// Acceptance-probe strategy; see [`AdaptiveConfig::gl10_probe`].
    pub gl10_probe: bool,
}

/// Result of one vector-valued adaptive integration.
#[derive(Debug, Clone)]
pub struct AdaptiveVecResult {
    pub values: Vec<Complex64>,
    /// Conservative per-component error estimate (same rule as [`Integrate`]).
    pub errors: Vec<f64>,
    /// Node evaluations (20 per panel call, as in the scalar version).
    pub evals: usize,
    pub capped: bool,
}

/// One GL-`n` evaluation of a *vector* integrand on `[a, b]`: `f(t, values,
/// errors)` fills one value and absolute error per component.  `n` is
/// [`GL_N`] or [`GL10_N`] (the probe rule).  Nodes are evaluated in a fixed
/// order (rayon-parallel when requested, bitwise identical to serial), then
/// reduced sequentially.  `valbuf` and `errbuf` are caller-owned scratch
/// slices of length `GL_N * m`, reused across panels.
#[allow(clippy::too_many_arguments)]
fn gl_eval_vec<F>(
    f: &F,
    a: f64,
    b: f64,
    m: usize,
    n: usize,
    parallel: bool,
    valbuf: &mut [Complex64],
    errbuf: &mut [f64],
) -> (Vec<Complex64>, Vec<f64>, Vec<f64>)
where
    F: Fn(f64, &mut [Complex64], &mut [f64]) + Sync,
{
    let (x, w) = if n == GL_N {
        gl_rule()
    } else {
        debug_assert_eq!(n, GL10_N);
        gl10_rule()
    };
    let c = 0.5 * (a + b);
    let h = 0.5 * (b - a);
    let nodes: [f64; GL_N] = std::array::from_fn(|i| if i < n { c + h * x[i] } else { 0.0 });
    if parallel {
        valbuf[..n * m]
            .par_chunks_mut(m)
            .zip(errbuf[..n * m].par_chunks_mut(m))
            .zip(nodes[..n].par_iter())
            .for_each(|((vr, er), &t)| f(t, vr, er));
    } else {
        for (i, &t) in nodes[..n].iter().enumerate() {
            f(t, &mut valbuf[i * m..(i + 1) * m], &mut errbuf[i * m..(i + 1) * m]);
        }
    }
    let mut values = vec![Complex64::new(0.0, 0.0); m];
    let mut masses = vec![0.0f64; m];
    let mut errs = vec![0.0f64; m];
    for (i, &wi) in w[..n].iter().enumerate() {
        let vr = &valbuf[i * m..(i + 1) * m];
        let er = &errbuf[i * m..(i + 1) * m];
        for k in 0..m {
            // A non-finite component (a node exactly on a pinch) counts as 0.
            let v = if vr[k].re.is_finite() && vr[k].im.is_finite() {
                vr[k]
            } else {
                Complex64::new(0.0, 0.0)
            };
            values[k] += v * wi;
            masses[k] += v.norm() * wi;
            errs[k] += er[k] * wi;
        }
    }
    for k in 0..m {
        values[k] *= h;
        masses[k] *= h.abs();
        errs[k] = (errs[k] * h).abs();
    }
    (values, masses, errs)
}

/// Vector-valued adaptive GL20 on `[a, b]`: `f(t, values, errors)` fills the
/// component values and their absolute errors at one node.  A panel is
/// accepted only when `every` component passes the rule (tol per unit length,
/// or its own roundoff floor times its `|f|` mass, or `min_len`); components
/// are otherwise tracked independently.  `parallel` evaluates a panel's nodes
/// with rayon in fixed order, so results are bitwise identical to serial.
///
/// With `floors = &[floor]` and an integrand writing a single component this
/// reproduces [`integrate`] bitwise (unit-tested) — the refactor guard for the
/// batched RC kernels.
pub fn integrate_vec<F>(
    f: &F,
    a: f64,
    b: f64,
    cfg: &AdaptiveVecConfig,
    parallel: bool,
) -> AdaptiveVecResult
where
    F: Fn(f64, &mut [Complex64], &mut [f64]) + Sync,
{
    let m = cfg.floors.len();
    assert!(m > 0, "integrate_vec needs at least one component");
    let mut values = vec![Complex64::new(0.0, 0.0); m];
    let mut errors = vec![0.0f64; m];
    let mut valbuf = vec![Complex64::new(0.0, 0.0); GL_N * m];
    let mut errbuf = vec![0.0f64; GL_N * m];
    let mut evals = 0usize;
    let mut capped = false;

    let (whole, whole_mass, whole_err) = gl_eval_vec(f, a, b, m, GL_N, parallel, &mut valbuf, &mut errbuf);
    evals += GL_N;
    // (lo, hi, whole values, whole |f| masses, whole propagated errors)
    let mut stack = vec![(a, b, whole, whole_mass, whole_err)];
    while let Some((lo, hi, whole, whole_mass, whole_err)) = stack.pop() {
        if evals >= cfg.max_panels {
            for k in 0..m {
                values[k] += whole[k];
                errors[k] += whole_err[k] + whole_mass[k];
            }
            capped = true;
            continue;
        }
        let mid = 0.5 * (lo + hi);
        let len = hi - lo;
        if cfg.gl10_probe {
            // Probe: GL10 halves against the GL20 whole; accept the GL20
            // values when every component agrees (see `integrate`).
            let (l10, lm10, le10) =
                gl_eval_vec(f, lo, mid, m, GL10_N, parallel, &mut valbuf, &mut errbuf);
            let (r10, rm10, re10) =
                gl_eval_vec(f, mid, hi, m, GL10_N, parallel, &mut valbuf, &mut errbuf);
            evals += 2 * GL10_N;
            let mut accept = true;
            for k in 0..m {
                let diff = (l10[k] + r10[k] - whole[k]).norm();
                if !(diff <= cfg.tol * len
                    || diff <= cfg.floors[k] * (lm10[k] + rm10[k])
                    || len < cfg.min_len)
                {
                    accept = false;
                    break;
                }
            }
            if accept {
                for k in 0..m {
                    let diff = (l10[k] + r10[k] - whole[k]).norm();
                    values[k] += whole[k];
                    errors[k] += diff + whole_err[k] + le10[k] + re10[k];
                }
                continue;
            }
        }
        let (l, lm, le) = gl_eval_vec(f, lo, mid, m, GL_N, parallel, &mut valbuf, &mut errbuf);
        let (r, rm, re) = gl_eval_vec(f, mid, hi, m, GL_N, parallel, &mut valbuf, &mut errbuf);
        evals += 2 * GL_N;
        let mut accept = true;
        for k in 0..m {
            let diff = (l[k] + r[k] - whole[k]).norm();
            if !(diff <= cfg.tol * len || diff <= cfg.floors[k] * (lm[k] + rm[k]) || len < cfg.min_len)
            {
                accept = false;
                break;
            }
        }
        if accept {
            for k in 0..m {
                let diff = (l[k] + r[k] - whole[k]).norm();
                values[k] += l[k] + r[k];
                errors[k] += diff + le[k] + re[k];
            }
        } else {
            stack.push((lo, mid, l, lm, le));
            stack.push((mid, hi, r, rm, re));
        }
    }
    // Whatever is still on the stack when the panel budget ran out.
    while let Some((_, _, whole, whole_mass, whole_err)) = stack.pop() {
        for k in 0..m {
            values[k] += whole[k];
            errors[k] += whole_err[k] + whole_mass[k];
        }
        capped = true;
    }
    AdaptiveVecResult { values, errors, evals, capped }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Wrap a scalar integrand as a single-component vector integrand.
    fn as_vec<F>(f: &F) -> impl Fn(f64, &mut [Complex64], &mut [f64]) + Sync + '_
    where
        F: Fn(f64) -> (Complex64, f64) + Sync,
    {
        move |t, v, e| {
            let (a, b) = f(t);
            v[0] = a;
            e[0] = b;
        }
    }

    #[test]
    fn gl20_exact_on_polynomials() {
        // Degree 39 = exactness of the 20-node rule.
        for deg in 0..=39 {
            let f = |x: f64| (Complex64::from(x.powi(deg)), 0.0);
            let r = integrate(&f, -1.0, 1.0, &AdaptiveConfig { tol: 1e-16, floor: 1e-16, min_len: 1e-15, max_panels: 1000, gl10_probe: false }, false);
            let exact = if deg % 2 == 0 { 2.0 / (deg as f64 + 1.0) } else { 0.0 };
            assert!(
                (r.value - Complex64::from(exact)).norm() < 1e-13,
                "degree {deg}: got {:.17e}, exact {exact:.17e}",
                r.value
            );
        }
    }

    #[test]
    fn gl10_exact_on_polynomials() {
        // Degree 19 = exactness of the 10-node probe rule.
        for deg in 0..=19 {
            let f = |x: f64| (Complex64::from(x.powi(deg)), 0.0);
            let (v, _, _) = gl_eval(&f, -1.0, 1.0, GL10_N, false);
            let exact = if deg % 2 == 0 { 2.0 / (deg as f64 + 1.0) } else { 0.0 };
            assert!(
                (v - Complex64::from(exact)).norm() < 1e-13,
                "degree {deg}: got {:.17e}, exact {exact:.17e}",
                v
            );
        }
    }

    #[test]
    fn gl10_probe_agrees_and_budgets_honestly() {
        // The probe strategy is GL20-quality on accepted panels: it must agree
        // with the classic strategy within the two budgets, and on the
        // non-smooth sqrt case (panels refine to `min_len`) its own budget
        // must cover the true error.
        let base = AdaptiveConfig {
            tol: 1e-14,
            floor: 1e-15,
            min_len: 1e-15,
            max_panels: 200_000,
            gl10_probe: false,
        };
        let probe = AdaptiveConfig { gl10_probe: true, ..base };
        for f in [
            (|x: f64| (Complex64::from_polar(1.0, 30.0 * (x * x - 0.4)), 0.0))
                as fn(f64) -> (Complex64, f64),
            |x: f64| (Complex64::new(1.0 / (1.0 + x * x), 0.0), 0.0),
            |x: f64| (Complex64::new((x * x).exp() * 0.3, (x + 1.0).ln()), x.abs() * 1e-16),
        ] {
            let s = integrate(&f, 0.0, 2.0, &base, false);
            let p = integrate(&f, 0.0, 2.0, &probe, false);
            let allow = 8.0 * f64::EPSILON * (s.value.norm() + p.value.norm());
            assert!(
                s.error + p.error + allow >= (s.value - p.value).norm(),
                "probe vs classic outside budgets: {:.3e} vs {:.3e}+{:.3e}",
                (s.value - p.value).norm(),
                s.error,
                p.error
            );
        }
        // Non-smooth: the true error must be covered by the probe's budget.
        let sing = |x: f64| (Complex64::from((x - 0.11).abs().sqrt()), 0.0);
        let exact = (0.11_f64.powf(1.5) + 0.89_f64.powf(1.5)) / 1.5;
        let p = integrate(&sing, 0.0, 1.0, &probe, false);
        assert!(
            p.error >= (p.value - Complex64::from(exact)).norm(),
            "probe budget {:+.3e} < true error {:+.3e}",
            p.error,
            (p.value - Complex64::from(exact)).norm()
        );
        // And the parallel evaluation stays bitwise identical to serial.
        let pp = integrate(&sing, 0.0, 1.0, &probe, true);
        assert_eq!(p.value, pp.value);
        assert_eq!(p.error, pp.error);
        assert_eq!(p.evals, pp.evals);
    }

    #[test]
    fn gl10_probe_vec_matches_classic_within_budgets() {
        // The vector probe path: same agreement contract, and the vector
        // result must stay bitwise parallel==serial.
        let f = |t: f64, v: &mut [Complex64], e: &mut [f64]| {
            v[0] = Complex64::new(1.0 / (1.0 + t * t), 0.0);
            v[1] = Complex64::from_polar(1.0, 40.0 * t);
            e[0] = 0.0;
            e[1] = 0.0;
        };
        let base = AdaptiveVecConfig {
            tol: 1e-14,
            floors: vec![1e-16, 1e-16],
            min_len: 1e-12,
            max_panels: 200_000,
            gl10_probe: false,
        };
        let probe = AdaptiveVecConfig { gl10_probe: true, ..base.clone() };
        let s = integrate_vec(&f, -3.0, 3.0, &base, false);
        let p = integrate_vec(&f, -3.0, 3.0, &probe, false);
        for k in 0..2 {
            let allow = 8.0 * f64::EPSILON * (s.values[k].norm() + p.values[k].norm());
            assert!(
                s.errors[k] + p.errors[k] + allow >= (s.values[k] - p.values[k]).norm(),
                "component {k} outside budgets"
            );
        }
        let pp = integrate_vec(&f, -3.0, 3.0, &probe, true);
        assert_eq!(p.values, pp.values);
        assert_eq!(p.errors, pp.errors);
        assert_eq!(p.evals, pp.evals);
    }

    #[test]
    fn adaptive_resolves_oscillation() {
        // exp(i k x) on [-6, 6]: ~10 and ~48 full oscillations — a non-adaptive
        // GL20 on the whole interval is garbage; the adaptive rule must reach
        // the exact finite-interval value 2 sin(6k)/k. The accuracy target is
        // floor-aware: the integrand carries an absolute phase-roundoff error
        // of a few ulps of |k x| (see `AdaptiveConfig::floor`), a legitimate
        // worst case of ~3e-11 rel at k=50 here — the bound keeps ~30x margin.
        for k in [10.0_f64, 50.0] {
            let f = |x: f64| (Complex64::from_polar(1.0, k * x), 0.0);
            let r = integrate(
                &f,
                -6.0,
                6.0,
                &AdaptiveConfig {
                    tol: 1e-14,
                    floor: 4e-16 * (1.0 + 6.0 * k),
                    min_len: 1e-15,
                    max_panels: 200_000,
                    gl10_probe: false,
                },
                false,
            );
            let exact = Complex64::from(2.0 * (6.0 * k).sin() / k);
            let rel = (r.value - exact).norm() / exact.norm();
            assert!(rel < 1e-9, "k={k}: rel err {rel:.3e}");
            assert!(r.error >= (r.value - exact).norm(), "error must cover the true error");
        }
    }

    #[test]
    fn error_budgets_min_len_and_cap() {
        // An integrable sqrt singularity (non-smooth, so panels refine all
        // the way to min_len): the budget must cover the true error against
        // the closed-form value int_0^1 |x - 0.11|^{1/2} dx.
        let f = |x: f64| (Complex64::from((x - 0.11).abs().sqrt()), 0.0);
        let exact = (0.11_f64.powf(1.5) + 0.89_f64.powf(1.5)) / 1.5;
        let r = integrate(
            &f,
            0.0,
            1.0,
            &AdaptiveConfig { tol: 1e-14, floor: 1e-15, min_len: 1e-9, max_panels: 200_000, gl10_probe: false },
            false,
        );
        let true_err = (r.value - Complex64::from(exact)).norm();
        assert!(r.error >= true_err, "budget {:+.3e} < true error {:+.3e}", r.error, true_err);
        assert!(true_err < 1e-4);

        // A tight panel cap must flag itself and budget the unrefined
        // remainder by its |f| mass (clearly larger than the refined error).
        let capped = integrate(
            &f,
            0.0,
            1.0,
            &AdaptiveConfig { tol: 1e-14, floor: 1e-15, min_len: 1e-9, max_panels: 10, gl10_probe: false },
            false,
        );
        assert!(capped.capped);
        assert!(capped.error > r.error);
    }

    #[test]
    fn parallel_is_bitwise_identical() {
        let f = |x: f64| (Complex64::from_polar(1.0, 30.0 * (x * x + (x - 0.3).exp())), 0.0);
        let cfg = AdaptiveConfig { tol: 1e-14, floor: 1e-15, min_len: 1e-15, max_panels: 200_000, gl10_probe: false };
        let s = integrate(&f, -3.0, 3.0, &cfg, false);
        let p = integrate(&f, -3.0, 3.0, &cfg, true);
        assert_eq!(s.value, p.value);
        assert_eq!(s.evals, p.evals);
    }

    #[test]
    #[allow(clippy::type_complexity)]
    fn vec_matches_scalar_for_one_component_bitwise() {
        // The refactor guard: integrate_vec with a single component and floor
        // must reproduce `integrate` bit for bit (values, errors, evals).
        let cases: Vec<Box<dyn Fn(f64) -> (Complex64, f64) + Sync>> = vec![
            Box::new(|x: f64| (Complex64::from_polar(1.0, 30.0 * (x * x - 0.4)), 0.0)),
            Box::new(|x: f64| (Complex64::from((x - 0.11).abs().sqrt()), 0.0)),
            Box::new(|x: f64| (Complex64::new((x * x).exp() * 0.3, (x + 1.0).ln()), x.abs() * 1e-16)),
        ];
        for f in &cases {
            let scfg = AdaptiveConfig { tol: 1e-13, floor: 1e-15, min_len: 1e-10, max_panels: 50_000, gl10_probe: false };
            let s = integrate(f, 0.0, 2.0, &scfg, false);
            let vcfg = AdaptiveVecConfig { tol: 1e-13, floors: vec![1e-15], min_len: 1e-10, max_panels: 50_000, gl10_probe: false };
            let fv = as_vec(f);
            let v = integrate_vec(&fv, 0.0, 2.0, &vcfg, false);
            assert_eq!(v.values[0], s.value, "value bits differ");
            assert_eq!(v.errors[0], s.error, "error bits differ");
            assert_eq!(v.evals, s.evals, "evals differ");
            let vp = integrate_vec(&fv, 0.0, 2.0, &vcfg, true);
            assert_eq!(vp.values[0], s.value, "parallel value bits differ");
            assert_eq!(vp.errors[0], s.error, "parallel error bits differ");
        }
    }

    #[test]
    fn vec_components_are_independent_and_error_honest() {
        // One smooth component, one oscillatory: the vector acceptance must
        // refine for the *worst* component while each error covers its own
        // deviation from a tightly refined scalar reference.
        let f = |t: f64, v: &mut [Complex64], e: &mut [f64]| {
            v[0] = Complex64::new(1.0 / (1.0 + t * t), 0.0);
            v[1] = Complex64::from_polar(1.0, 40.0 * t);
            e[0] = 0.0;
            e[1] = 0.0;
        };
        let vcfg = AdaptiveVecConfig { tol: 1e-14, floors: vec![1e-16, 1e-16], min_len: 1e-12, max_panels: 200_000, gl10_probe: false };
        let r = integrate_vec(&f, -3.0, 3.0, &vcfg, false);
        let rp = integrate_vec(&f, -3.0, 3.0, &vcfg, true);
        assert_eq!(r.values[0], rp.values[0]);
        assert_eq!(r.values[1], rp.values[1]);
        // Exact references: int 1/(1+t^2) = 2 atan(3); int e^{i 40 t} = sin(120)/20.
        let exact0 = Complex64::new(2.0 * 3.0_f64.atan(), 0.0);
        let exact1 = Complex64::new(120.0_f64.sin() / 20.0, 0.0);
        assert!(
            (r.values[0] - exact0).norm() < 1e-12,
            "smooth component off: {:.3e}",
            (r.values[0] - exact0).norm()
        );
        assert!(
            (r.values[1] - exact1).norm() < 1e-12,
            "oscillatory component off: {:.3e}",
            (r.values[1] - exact1).norm()
        );
        // Honesty at the roundoff floor: the estimator sums accepted-panel
        // disagreements, so it can under-cover by a couple of ulps of the
        // *value* when the true error itself is at that level.
        let allowance = |v: Complex64| 8.0 * f64::EPSILON * v.norm();
        for (k, exact) in [(0, exact0), (1, exact1)] {
            let true_err = (r.values[k] - exact).norm();
            assert!(
                r.errors[k] + allowance(r.values[k]) >= true_err,
                "component {k}: estimate {:.3e} does not cover true error {:.3e}",
                r.errors[k],
                true_err
            );
            assert!(r.errors[k] < 1e-12, "component {k}: estimate suspiciously loose");
        }
    }
}
