//! Polynomial utilities and a complete complex root finder, backing the
//! rational-phase saddle source (see [`crate::saddles::saddles_from_polynomial`]).
//!
//! Coefficient convention throughout: ascending order, `c[k]` multiplies
//! $z^k$.
//!
//! The root finder is Durand-Kerner (Weierstrass) iteration: all roots
//! simultaneously, complex coefficients natively, no linear-algebra
//! dependencies. Its contract is *completeness or an error*: on success it
//! returns exactly `degree` roots, each with a residual small relative to
//! the polynomial's own term sizes — it never silently returns a short
//! list. (This is the property grid-Newton saddle searching cannot offer,
//! and the reason multi-image lens configurations become trustworthy.)

use crate::{Error, Result};
use num_complex::Complex64;

/// Evaluate the polynomial at `z` (Horner).
pub fn poly_eval(c: &[Complex64], z: Complex64) -> Complex64 {
    let mut acc = Complex64::new(0.0, 0.0);
    for &ck in c.iter().rev() {
        acc = acc * z + ck;
    }
    acc
}

/// Product of two polynomials (convolution of coefficients).
pub fn poly_mul(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![Complex64::new(0.0, 0.0); a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] += ai * bj;
        }
    }
    out
}

/// Sum of two polynomials.
pub fn poly_add(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    let n = a.len().max(b.len());
    let mut out = vec![Complex64::new(0.0, 0.0); n];
    for (k, &ak) in a.iter().enumerate() {
        out[k] += ak;
    }
    for (k, &bk) in b.iter().enumerate() {
        out[k] += bk;
    }
    out
}

/// The polynomial scaled by a constant.
pub fn poly_scale(a: &[Complex64], s: Complex64) -> Vec<Complex64> {
    a.iter().map(|&ak| ak * s).collect()
}

/// Coefficients of the derivative polynomial.
pub fn poly_derivative(a: &[Complex64]) -> Vec<Complex64> {
    if a.len() <= 1 {
        return Vec::new();
    }
    a.iter()
        .enumerate()
        .skip(1)
        .map(|(k, &ak)| ak * Complex64::from(k as f64))
        .collect()
}

/// All complex roots of the polynomial, by Durand-Kerner iteration.
///
/// Trailing coefficients below `1e-14 * max|c|` are trimmed (they are
/// numerical zeros of the leading terms); the remaining degree must be at
/// least 1. Returns exactly `degree` roots (with multiplicity — a double
/// root appears as a converged cluster) or an error; residuals are checked
/// against the polynomial's own term magnitudes at each root, so an
/// ill-converged iteration cannot pass silently.
pub fn poly_roots(coeffs: &[Complex64]) -> Result<Vec<Complex64>> {
    let max_c = coeffs.iter().map(|c| c.norm()).fold(0.0, f64::max);
    if max_c == 0.0 {
        return Err(Error::InvalidInput("zero polynomial has no roots"));
    }
    let mut len = coeffs.len();
    while len > 0 && coeffs[len - 1].norm() < 1e-14 * max_c {
        len -= 1;
    }
    if len < 2 {
        return Err(Error::InvalidInput(
            "polynomial must have degree >= 1 after trimming",
        ));
    }
    let lead = coeffs[len - 1];
    let monic: Vec<Complex64> = coeffs[..len].iter().map(|&c| c / lead).collect();
    let n = len - 1;

    // Cauchy bound on root magnitudes, and the classic symmetry-breaking
    // start (powers of 0.4 + 0.9i spiral inside the bound).
    let radius = 1.0 + monic[..n].iter().map(|c| c.norm()).fold(0.0, f64::max);
    let seed = Complex64::new(0.4, 0.9);
    let mut roots: Vec<Complex64> = Vec::with_capacity(n);
    let mut w = Complex64::new(1.0, 0.0);
    for _ in 0..n {
        roots.push(w * Complex64::from(radius));
        w *= seed;
    }

    for _ in 0..500 {
        let mut max_step = 0.0_f64;
        for k in 0..n {
            let zk = roots[k];
            let mut denom = Complex64::new(1.0, 0.0);
            for (j, &rj) in roots.iter().enumerate() {
                if j != k {
                    denom *= zk - rj;
                }
            }
            if denom.norm() == 0.0 {
                // Coincident iterates: nudge apart and keep sweeping.
                roots[k] += Complex64::new(1e-8 * (1.0 + zk.norm()), 1e-8);
                max_step = f64::INFINITY;
                continue;
            }
            let step = poly_eval(&monic, zk) / denom;
            roots[k] = zk - step;
            max_step = max_step.max(step.norm() / (1.0 + roots[k].norm()));
        }
        if max_step < 1e-13 {
            break;
        }
    }

    // Completeness contract: every root must actually satisfy p(z) ≈ 0,
    // measured against the size of the terms that were summed.
    for &r in &roots {
        let scale = monic
            .iter()
            .enumerate()
            .map(|(k, c)| c.norm() * r.norm().powi(k as i32))
            .sum::<f64>()
            .max(1.0);
        if !r.is_finite() || !(poly_eval(&monic, r).norm() < 1e-10 * scale) {
            return Err(Error::InvalidInput(
                "polynomial root finding did not converge",
            ));
        }
    }
    Ok(roots)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    /// Build the monic polynomial with the given roots.
    fn from_roots(roots: &[Complex64]) -> Vec<Complex64> {
        let mut p = vec![c(1.0, 0.0)];
        for &r in roots {
            p = poly_mul(&p, &[-r, c(1.0, 0.0)]);
        }
        p
    }

    fn assert_recovers(expected: &[Complex64]) {
        let got = poly_roots(&from_roots(expected)).unwrap();
        assert_eq!(got.len(), expected.len());
        for &e in expected {
            let hit = got
                .iter()
                .any(|&g| (g - e).norm() < 1e-8 * (1.0 + e.norm()));
            assert!(hit, "root {e} not recovered in {got:?}");
        }
    }

    #[test]
    fn recovers_prescribed_complex_roots() {
        assert_recovers(&[
            c(1.0, 0.0),
            c(-2.0, 0.0),
            c(0.0, 1.0),
            c(0.0, -1.0),
            c(0.3, -0.7),
        ]);
    }

    #[test]
    fn handles_double_root_as_cluster() {
        // (z - 1)^2 (z + 2): the double root converges as a cluster; both
        // copies must sit near 1 within the (looser) linear-convergence
        // neighbourhood consistent with the residual contract.
        let p = from_roots(&[c(1.0, 0.0), c(1.0, 0.0), c(-2.0, 0.0)]);
        let got = poly_roots(&p).unwrap();
        assert_eq!(got.len(), 3);
        let near_one = got
            .iter()
            .filter(|g| (*g - c(1.0, 0.0)).norm() < 1e-4)
            .count();
        let near_m2 = got
            .iter()
            .filter(|g| (*g - c(-2.0, 0.0)).norm() < 1e-8)
            .count();
        assert_eq!((near_one, near_m2), (2, 1), "got {got:?}");
    }

    #[test]
    fn seventeenth_degree_random_polynomial() {
        // Deterministic pseudo-random roots; count + residual contract.
        let expected: Vec<Complex64> = (0..17)
            .map(|k| {
                let a = 2.399963 * k as f64; // golden-angle spread
                Complex64::from_polar(0.3 + 0.15 * k as f64, a)
            })
            .collect();
        assert_recovers(&expected);
    }

    #[test]
    fn trims_trailing_zeros_and_rejects_constants() {
        // z - 3 padded with numerically-zero cubic terms.
        let p = vec![c(-3.0, 0.0), c(1.0, 0.0), c(1e-17, 0.0), c(0.0, 1e-16)];
        let got = poly_roots(&p).unwrap();
        assert_eq!(got.len(), 1);
        assert!((got[0] - c(3.0, 0.0)).norm() < 1e-10);

        assert!(poly_roots(&[c(5.0, 0.0)]).is_err());
        assert!(poly_roots(&[]).is_err());
    }

    #[test]
    fn derivative_and_eval_agree() {
        // p = 1 + 2z + 3z^2 -> p' = 2 + 6z, checked at a point.
        let p = [c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)];
        let dp = poly_derivative(&p);
        let z = c(0.7, -0.2);
        assert!((poly_eval(&dp, z) - (c(2.0, 0.0) + c(6.0, 0.0) * z)).norm() < 1e-14);
    }
}
