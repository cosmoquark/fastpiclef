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

/// Gauss-Legendre nodes/weights on `[-1, 1]` (Newton from the Chebyshev-like
/// initial guesses), computed once. Exactness is unit-tested on monomials.
fn gl_rule() -> &'static (Vec<f64>, Vec<f64>) {
    static RULE: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    RULE.get_or_init(|| {
        let n = GL_N;
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
    })
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

/// One GL20 evaluation of `f` on `[a, b]`: (integral, |f| mass, node errors).
fn gl20<F>(f: &F, a: f64, b: f64, parallel: bool) -> (Complex64, f64, f64)
where
    F: Fn(f64) -> (Complex64, f64) + Sync,
{
    let (x, w) = gl_rule();
    let c = 0.5 * (a + b);
    let h = 0.5 * (b - a);
    let nodes: Vec<f64> = x.iter().map(|&xi| c + h * xi).collect();
    let vals: Vec<(Complex64, f64)> = if parallel {
        nodes.par_iter().map(|&t| f(t)).collect()
    } else {
        nodes.iter().map(|&t| f(t)).collect()
    };
    let mut s = Complex64::new(0.0, 0.0);
    let mut mass = 0.0;
    let mut err = 0.0;
    for (v, &wi) in vals.iter().zip(w) {
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

    let (whole, whole_mass, whole_err) = gl20(f, a, b, parallel);
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
        let (l, lm, le) = gl20(f, lo, mid, parallel);
        let (r, rm, re) = gl20(f, mid, hi, parallel);
        evals += 2 * GL_N;
        let diff = (l + r - whole).norm();
        if diff <= cfg.tol * (hi - lo)
            || diff <= cfg.floor * (lm + rm)
            || hi - lo < cfg.min_len
        {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gl20_exact_on_polynomials() {
        // Degree 39 = exactness of the 20-node rule.
        for deg in 0..=39 {
            let f = |x: f64| (Complex64::from(x.powi(deg)), 0.0);
            let r = integrate(&f, -1.0, 1.0, &AdaptiveConfig { tol: 1e-16, floor: 1e-16, min_len: 1e-15, max_panels: 1000 }, false);
            let exact = if deg % 2 == 0 { 2.0 / (deg as f64 + 1.0) } else { 0.0 };
            assert!(
                (r.value - Complex64::from(exact)).norm() < 1e-13,
                "degree {deg}: got {:.17e}, exact {exact:.17e}",
                r.value
            );
        }
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
            &AdaptiveConfig { tol: 1e-14, floor: 1e-15, min_len: 1e-9, max_panels: 200_000 },
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
            &AdaptiveConfig { tol: 1e-14, floor: 1e-15, min_len: 1e-9, max_panels: 10 },
            false,
        );
        assert!(capped.capped);
        assert!(capped.error > r.error);
    }

    #[test]
    fn parallel_is_bitwise_identical() {
        let f = |x: f64| (Complex64::from_polar(1.0, 30.0 * (x * x + (x - 0.3).exp())), 0.0);
        let cfg = AdaptiveConfig { tol: 1e-14, floor: 1e-15, min_len: 1e-15, max_panels: 200_000 };
        let s = integrate(&f, -3.0, 3.0, &cfg, false);
        let p = integrate(&f, -3.0, 3.0, &cfg, true);
        assert_eq!(s.value, p.value);
        assert_eq!(s.evals, p.evals);
    }
}
