//! Nakamura–Deguchi closed form for the isolated point-mass lens — the
//! **µs-class evaluator** inside its band, where the 2D RC engine costs
//! 70–270 ms at ω ≤ 3 (0.6 s at 10, 6.8 s at 100) and the polar 1D path
//! 40 ms–0.4 s at ω = 10–100, but 0.9–2.3 s and loose at ω ≤ 1
//! (single-thread quiet-window walls: `comparison/results/fast_paths.csv`).
//!
//! $$ F(\omega, y) = e^{\pi\omega/4}\,
//!   e^{i\frac{\omega}{2}\left(\ln\frac{\omega}{2} - 2\phi_m\right)}\,
//!   \Gamma\!\left(1 - \tfrac{i\omega}{2}\right)\,
//!   {}_1F_1\!\left(\tfrac{i\omega}{2}, 1; \tfrac{i\omega y^2}{2}\right) $$
//! with $x_m = \tfrac12(y + \sqrt{y^2+4})$, $\phi_m = \tfrac12(x_m-y)^2 - \ln x_m$
//! (Takahashi & Nakamura 2003), the same convention as
//! `comparison/referee.py::point_mass_exact` and `tests/point_mass.rs`.
//!
//! Numerics: everything in *log space* (the $e^{\pm\pi\omega/4}$ scales exceed
//! `f64` for $\omega \gtrsim 900$), $\ln\Gamma$ by the Stirling series with a
//! recurrence upshift, and the confluent hypergeometric by its series with a
//! cancellation-based error estimate.  The series' worst term grows like
//! $e^{\omega y}$ in the *compound* parameter $\omega y$ (measured 2026-10-18
//! against the exact RC, `comparison/results/nd_band.csv`): rel errors run
//! $10^(-15)$..$10^(-12)$ for $\omega y lt.eq 10$, $tilde 10^(-8)$ at 20,
//! $tilde 10^(-3)$ at 30, failing outright beyond $tilde 40$; the returned
//! estimate stays honest throughout (coverage $gt.eq 9 times$).  Above the
//! band, use the polar path (`PointMass.solve(method="polar")`, fast for
//! $\omega \gtrsim 10$) or the high-$\omega$ hybrid.

use num_complex::Complex64;

/// `ln Gamma(z)` by the Stirling series, with a recurrence upshift so the
/// asymptotic series is evaluated where it converges (Re z >= 16 with eight
/// terms, i.e. ~1e-17 relative).  Log space: the caller's arguments reach
/// `1 - i w/2` with `w ~ 1e4`.
fn ln_gamma(z: Complex64) -> Complex64 {
    let mut z = z;
    let mut acc = Complex64::new(0.0, 0.0);
    while z.re < 16.0 {
        acc -= z.ln();
        z += Complex64::new(1.0, 0.0);
    }
    // Stirling coefficients c_k for z^{-(2k-1)}: B_{2k}/(2k(2k-1)).
    const C: [f64; 8] = [
        1.0 / 12.0,
        -1.0 / 360.0,
        1.0 / 1260.0,
        -1.0 / 1680.0,
        1.0 / 1188.0,
        -691.0 / 360360.0,
        1.0 / 156.0,
        -3617.0 / 122400.0,
    ];
    let mut s = (z - Complex64::new(0.5, 0.0)) * z.ln() - z
        + Complex64::new(0.5 * (2.0 * std::f64::consts::PI).ln(), 0.0);
    let zinv2 = 1.0 / (z * z);
    let mut p = 1.0 / z;
    for &c in C.iter() {
        s += c * p;
        p = p * zinv2;
    }
    s + acc
}

/// The isolated unit point mass at the origin with source `(y, 0)`:
/// `(F, error estimate)` in the amplification convention (1 for free space).
///
/// The estimate is conservative: it carries a few ulps of the series' largest
/// term (the cancellation bound), the series truncation and the `ln Gamma`
/// roundoff.  Validated against the mpmath referee to ~1e-13 for
/// `|z| = omega*y^2/2 <= 10` and ~1e-8 at `|z| = 20` (`tests/nd.rs`).
pub fn point_mass_nd(omega: f64, y: f64) -> (Complex64, f64) {
    let xm = 0.5 * (y + (y * y + 4.0).sqrt());
    let phi_m = 0.5 * (xm - y) * (xm - y) - xm.ln();
    let a = Complex64::new(0.0, 0.5 * omega);
    let z = Complex64::new(0.0, 0.5 * omega * y * y);

    // 1F1(a, 1, z) = sum_n (a)_n z^n / (n!)^2  (the (1)_n = n! denominator).
    let mut term = Complex64::new(1.0, 0.0);
    let mut sum = term;
    let mut max_term = 1.0f64;
    let mut n_used = 0usize;
    for n in 1..=20_000usize {
        term *= a + Complex64::new((n - 1) as f64, 0.0);
        term *= z;
        term /= Complex64::new((n * n) as f64, 0.0);
        sum += term;
        let tn = term.norm();
        max_term = max_term.max(tn);
        n_used = n;
        if tn <= 1e-18 * sum.norm() {
            break;
        }
    }

    // Log prefactor: pi w/4 + i (w/2)(ln(w/2) - 2 phi_m) + ln Gamma(1 - i w/2).
    let lnf = Complex64::new(0.25 * std::f64::consts::PI * omega, 0.0)
        + Complex64::new(0.0, 0.5 * omega) * Complex64::new((0.5 * omega).ln() - 2.0 * phi_m, 0.0)
        + ln_gamma(Complex64::new(1.0, -0.5 * omega));
    let scale = lnf.exp();
    let f = scale * sum;

    // Budget: the series' cancellation (n_used ulps of its largest term, scaled
    // by |e^{lnf}|), the truncation (<= 1e-18 * |sum| * |scale| plus the same
    // for the last term's neighbours), and ~32 ulps of the result for the
    // ln Gamma / exp roundoff.
    let err = 2.0
        * (scale.norm() * (n_used as f64 + 8.0) * f64::EPSILON * max_term
            + scale.norm() * 1e-17 * sum.norm() * 4.0
            + 32.0 * f64::EPSILON * f.norm());
    (f, err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_gamma_matches_the_magnitude_identity() {
        // |Gamma(1 + i t)|^2 = pi t / sinh(pi t): an external closed form.
        for &t in &[0.3f64, 1.0, 4.0, 20.0, 200.0] {
            let l = ln_gamma(Complex64::new(1.0, t));
            let want = (std::f64::consts::PI * t / (std::f64::consts::PI * t).sinh()).ln() / 2.0;
            assert!((l.re - want).abs() < 1e-13, "t={t}: {} vs {want}", l.re);
        }
        // Gamma(1) = 1 exactly in log space.
        let l = ln_gamma(Complex64::new(1.0, 0.0));
        assert!(l.norm() < 1e-14, "ln Gamma(1) = {l}");
    }

    #[test]
    fn nd_matches_the_mpmath_referee() {
        // mpmath dps 40, comparison/referee.py::point_mass_exact, 2026-10-13.
        const CASES: &[(f64, f64, f64, f64)] = &[
            (0.1, 0.1, 1.06546084792284246e+00, -1.73279323012136027e-01),
            (0.3, 0.1, 1.17928186973131344e+00, -3.90880104783098481e-01),
            (1.0, 0.1, 1.58190834496196042e+00, -8.74450823275391032e-01),
            (3.0, 0.1, 2.72021736153357585e+00, -1.26842825601871145e+00),
            (10.0, 0.1, 4.20110227822937876e+00, 8.64533448093296464e-01),
            (0.3, 0.5, 1.20680698688551646e+00, -2.65531586118585461e-01),
            (1.0, 0.5, 1.67840615998213827e+00, -2.74118567456421769e-01),
            (3.0, 0.5, 1.38167984601493465e+00, 7.82641158493647793e-01),
            (10.0, 0.5, 7.96078113705452894e-01, 6.29005213288373222e-01),
            (1.0, 1.0, 1.37744791808669720e+00, 2.09211704719407671e-01),
            (3.0, 1.0, 1.05026960762060595e+00, -3.89822956330972303e-01),
            (10.0, 1.0, 1.46569803520333042e+00, 1.54978479861528906e-01),
            (1.0, 2.0, 8.49497039642139296e-01, 8.39835783155317220e-04),
            (3.0, 2.0, 1.17793862536949079e+00, -5.50246606861003601e-02),
            (10.0, 2.0, 1.17748271809338134e+00, 6.28628881089321745e-02),
        ];
        for &(w, y, re, im) in CASES {
            let want = Complex64::new(re, im);
            let (got, err) = point_mass_nd(w, y);
            let d = (got - want).norm();
            let zabs = 0.5 * w * y * y;
            // Cancellation-degraded tolerance: the series loses ~eps*e^{|z|}
            // per accumulation (measured 2026-10-13: rel = 1e-15..5.6e-13 up to
            // |z| = 6, 3.8e-7 at |z| = 20).  Mirrors the docs.
            let tol = 1e-13 + 64.0 * f64::EPSILON * zabs.exp();
            assert!(d / want.norm() < tol, "w={w} y={y}: rel {} (|z|={zabs})", d / want.norm());
            // the estimate must cover the true error
            assert!(err >= d, "w={w} y={y}: estimate {err:e} < true {d:e}");
        }
    }

    #[test]
    fn nd_agrees_with_the_polar_path_at_high_omega() {
        // The polar (1D thimble) path is the fast path above |z| ~ 10; the two
        // must agree in *magnitude* (the ND form carries the standard constant
        // Fermat-offset phase, so complex values are not comparable across
        // conventions — AGENTS note).
        use crate::polar::solve_point_mass_polar;
        use crate::thimble::ThimbleConfig;
        for &(w, y) in &[(10.0f64, 0.5f64), (30.0, 0.5), (100.0, 0.5)] {
            let (f, err) = point_mass_nd(w, y);
            let cfg = crate::polar::PolarConfig {
                radial: ThimbleConfig { omega: w, ..Default::default() },
                ..Default::default()
            };
            let (out, _) = solve_point_mass_polar(y, &cfg).unwrap();
            let amp = out.value * w / (2.0 * std::f64::consts::PI * Complex64::new(0.0, 1.0));
            let d = (amp.norm() - f.norm()).abs();
            assert!(
                d <= err + out.error + 1e-12,
                "w={w} y={y}: |diff| {d:e} vs {err:e} + {:e}",
                out.error
            );
        }
    }
}



