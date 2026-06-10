//! Library of ready-made phase functions with known reference solutions,
//! used for validation, convergence studies, and cross-code comparisons.

use crate::Phase;
use nalgebra::SVector;
use num_complex::Complex64;

/// Cubic ("Airy") phase in one dimension:
/// $\phi(z; y) = z^3/3 - y z$.
///
/// Exact Fresnel integral:
/// $\int_\mathbb{R} e^{i\omega(x^3/3 - y x)}\,dx
///   = 2\pi\,\omega^{-1/3}\,\mathrm{Ai}(-\omega^{2/3} y)$.
///
/// Saddle structure: $\phi'(z) = z^2 - y$, so
/// - $y > 0$: two real images at $z = \pm\sqrt{y}$ (oscillatory regime);
/// - $y = 0$: a degenerate saddle at the origin — the caustic;
/// - $y < 0$: a complex-conjugate pair $z = \pm i\sqrt{-y}$ of which only
///   one lies on the deformed contour (exponentially damped regime).
///
/// Since $\phi(-x) = -\phi(x)$ on the real axis, the integral is real for
/// every real $y$ — the imaginary part of a computed value is a free
/// accuracy check.
pub struct AiryPhase {
    pub y: f64,
}

impl Phase<1> for AiryPhase {
    fn value(&self, z: &SVector<Complex64, 1>) -> Complex64 {
        let z = z[0];
        z * z * z / 3.0 - self.y * z
    }

    fn gradient(&self, z: &SVector<Complex64, 1>) -> SVector<Complex64, 1> {
        let z = z[0];
        SVector::<Complex64, 1>::new(z * z - self.y)
    }
}
