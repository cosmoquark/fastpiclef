//! Saddle-point (image) finding for 1D phases — the prerequisite step of
//! the constant-phase contour method (Shi 2024, `refs/2409.12991v1.pdf`).
//!
//! Two sources with different guarantees:
//! - [`find_saddles`]: Newton iteration from a grid of seeds. Works for any
//!   analytic phase, but is *best-effort*: roots whose Newton basins are
//!   smaller than the seed spacing are missed (observed for saddles hugging
//!   the poles of rational lens potentials).
//! - [`saddles_from_polynomial`]: complete by construction for phases whose
//!   $\phi'(z) = 0$ clears to a polynomial (see [`crate::RationalPhase`]).
//!   All roots come from Durand-Kerner ([`crate::roots::poly_roots`]) and
//!   are then polished against the *true* $\phi'$.

use crate::Phase;
use nalgebra::SVector;
use num_complex::Complex64;

/// A stationary point of the phase, $\phi'(z) = 0$.
#[derive(Debug, Clone, Copy)]
pub struct Saddle {
    pub z: Complex64,
    /// $\phi(z)$; the Morse height is $h = -\mathrm{Im}\,\phi$ and the
    /// thimble's constant phase is $H = \mathrm{Re}\,\phi$.
    pub value: Complex64,
    /// $\phi''(z)$ (numerical). Vanishes at a caustic.
    pub second: Complex64,
}

impl Saddle {
    pub fn h(&self) -> f64 {
        -self.value.im
    }
    pub fn big_h(&self) -> f64 {
        self.value.re
    }
}

pub(crate) fn val(phase: &dyn Phase<1>, z: Complex64) -> Complex64 {
    phase.value(&SVector::<Complex64, 1>::new(z))
}

pub(crate) fn grad(phase: &dyn Phase<1>, z: Complex64) -> Complex64 {
    phase.gradient(&SVector::<Complex64, 1>::new(z))[0]
}

/// $\phi''(z)$ by central difference of the user-supplied gradient. The
/// integrand is analytic, so a real step of $10^{-5}(1+|z|)$ keeps the
/// truncation error at $O(10^{-10})$ — ample for Newton steps and for
/// thimble take-off directions.
pub fn second_derivative(phase: &dyn Phase<1>, z: Complex64) -> Complex64 {
    let step = 1e-5 * (1.0 + z.norm());
    (grad(phase, z + step) - grad(phase, z - step)) / (2.0 * step)
}

/// Find the saddles of `phase` inside the window `re_range` x `im_range`
/// by Newton iteration from a `seeds_per_axis` x `seeds_per_axis` grid.
/// Roots are accepted when $|\phi'| <$ `tol`, deduplicated, and returned
/// sorted by real part (then imaginary part).
pub fn find_saddles(
    phase: &dyn Phase<1>,
    re_range: (f64, f64),
    im_range: (f64, f64),
    seeds_per_axis: usize,
    tol: f64,
) -> Vec<Saddle> {
    let n = seeds_per_axis.max(2);
    let mut roots: Vec<Complex64> = Vec::new();
    let pad = 1e-9 * (1.0 + re_range.1.abs().max(im_range.1.abs()));
    let in_window = |z: Complex64| {
        z.re >= re_range.0 - pad
            && z.re <= re_range.1 + pad
            && z.im >= im_range.0 - pad
            && z.im <= im_range.1 + pad
    };

    for i in 0..n {
        for j in 0..n {
            let mut z = Complex64::new(
                re_range.0 + (re_range.1 - re_range.0) * (i as f64 + 0.5) / n as f64,
                im_range.0 + (im_range.1 - im_range.0) * (j as f64 + 0.5) / n as f64,
            );
            let mut converged = false;
            for _ in 0..50 {
                let g = grad(phase, z);
                if g.norm() < tol {
                    converged = true;
                    break;
                }
                let g2 = second_derivative(phase, z);
                if g2.norm() < 1e-300 || !z.is_finite() {
                    break;
                }
                z -= g / g2;
            }
            if !(converged && z.is_finite() && in_window(z)) {
                continue;
            }
            let dup = roots
                .iter()
                .any(|&r| (r - z).norm() < 1e-7 * (1.0 + z.norm()));
            if !dup {
                roots.push(z);
            }
        }
    }

    let mut saddles: Vec<Saddle> = roots
        .into_iter()
        .map(|z| Saddle {
            z,
            value: val(phase, z),
            second: second_derivative(phase, z),
        })
        .collect();
    saddles.sort_by(|a, b| {
        (a.z.re, a.z.im)
            .partial_cmp(&(b.z.re, b.z.im))
            .expect("saddle coordinates are finite")
    });
    saddles
}

/// The *complete* saddle set of a phase whose $\phi'(z) = N(z)/D(z)$ with
/// polynomial $N$ (coefficients supplied ascending; see
/// [`crate::RationalPhase::stationary_polynomial`]).
///
/// The polynomial coefficients only need to be accurate enough to land each
/// Durand-Kerner root inside its Newton basin — machine precision comes
/// from the polish against the phase's own gradient. A root that fails to
/// polish is an error (`InvalidInput`), never silently dropped: it almost
/// always means the coefficients are inconsistent with the phase. Multiple
/// roots collapse in the dedup and surface downstream as
/// [`crate::Error::DegenerateSaddle`] (caustic) via the $|\phi''|$ cut.
pub fn saddles_from_polynomial(
    phase: &dyn Phase<1>,
    coeffs: &[Complex64],
    newton_tol: f64,
) -> crate::Result<Vec<Saddle>> {
    let roots = crate::roots::poly_roots(coeffs)?;
    let mut polished: Vec<Complex64> = Vec::with_capacity(roots.len());
    for mut z in roots {
        let mut converged = false;
        for _ in 0..10 {
            let g = grad(phase, z);
            if g.norm() < newton_tol {
                converged = true;
                break;
            }
            let g2 = second_derivative(phase, z);
            if g2.norm() < 1e-300 || !z.is_finite() {
                break;
            }
            z -= g / g2;
        }
        // Near-multiple roots stall Newton at |phi'| ~ sqrt-scale; accept
        // them at the same threshold the degenerate-saddle cut uses, so the
        // caustic path (not a hard error) handles them.
        let near_degenerate_ok = grad(phase, z).norm() < newton_tol.sqrt();
        if !((converged || near_degenerate_ok) && z.is_finite()) {
            return Err(crate::Error::InvalidInput(
                "polynomial saddle polishing failed (are the stationary-polynomial \
                 coefficients consistent with the phase gradient?)",
            ));
        }
        if !polished
            .iter()
            .any(|&r| (r - z).norm() < 1e-7 * (1.0 + z.norm()))
        {
            polished.push(z);
        }
    }

    let mut saddles: Vec<Saddle> = polished
        .into_iter()
        .map(|z| Saddle {
            z,
            value: val(phase, z),
            second: second_derivative(phase, z),
        })
        .collect();
    saddles.sort_by(|a, b| {
        (a.z.re, a.z.im)
            .partial_cmp(&(b.z.re, b.z.im))
            .expect("saddle coordinates are finite")
    });
    Ok(saddles)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phases::AiryPhase;

    #[test]
    fn airy_two_real_saddles() {
        let saddles = find_saddles(&AiryPhase { y: 1.0 }, (-8.0, 8.0), (-8.0, 8.0), 12, 1e-11);
        assert_eq!(saddles.len(), 2);
        assert!((saddles[0].z - Complex64::new(-1.0, 0.0)).norm() < 1e-9);
        assert!((saddles[1].z - Complex64::new(1.0, 0.0)).norm() < 1e-9);
        // phi'' = 2z at the saddles.
        assert!((saddles[1].second - Complex64::new(2.0, 0.0)).norm() < 1e-5);
    }

    #[test]
    fn airy_complex_pair() {
        let saddles = find_saddles(&AiryPhase { y: -1.0 }, (-8.0, 8.0), (-8.0, 8.0), 12, 1e-11);
        assert_eq!(saddles.len(), 2);
        assert!(
            (saddles[0].z - Complex64::new(0.0, -1.0)).norm() < 1e-9
                || (saddles[0].z - Complex64::new(0.0, 1.0)).norm() < 1e-9
        );
        // h = -Im phi = ∓2/3 at z = ±i.
        let h_of_upper = saddles
            .iter()
            .find(|s| s.z.im > 0.0)
            .expect("upper saddle present")
            .h();
        assert!((h_of_upper - (-2.0 / 3.0)).abs() < 1e-9);
    }
}
