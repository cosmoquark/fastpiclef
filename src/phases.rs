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

/// One-dimensional lens phase with a rational potential (Shi 2024's example):
/// $\phi(z) = (z - \beta)^2 / 2 - \kappa / (1 + z^2)$.
///
/// Convention follows the paper: `kappa > 0` is a diverging lens, `kappa < 0`
/// a converging one. The potential $\psi(z) = 1/(1+z^2)$ has double **poles**
/// at $z = \pm i$ — this is the phase that exercises the contour machinery
/// against singularities in the complex plane (thimble pieces may terminate
/// near a pole; mesh-flow trajectories may approach one).
///
/// $\phi'(z) = 0$ clears to a degree-5 polynomial, so there are five complex
/// stationary points; how many are real (geometric images) depends on
/// $(\beta, \kappa)$. At $\beta = 0$ a converging lens with $\kappa \le -1/2$
/// opens a pair of real images — the multi-image regime.
///
/// No closed form is known; validated by agreement between the independent
/// thimble and mesh-flow solvers.
pub struct RationalLens {
    pub beta: f64,
    pub kappa: f64,
}

impl Phase<1> for RationalLens {
    fn value(&self, z: &SVector<Complex64, 1>) -> Complex64 {
        let z = z[0];
        let u = z - self.beta;
        0.5 * u * u - self.kappa / (1.0 + z * z)
    }

    fn gradient(&self, z: &SVector<Complex64, 1>) -> SVector<Complex64, 1> {
        let z = z[0];
        let d = 1.0 + z * z;
        // d/dz [-kappa/(1+z^2)] = +2 kappa z / (1+z^2)^2
        SVector::<Complex64, 1>::new((z - self.beta) + 2.0 * self.kappa * z / (d * d))
    }
}

/// Axisymmetric 2D lens with the same rational potential, holomorphically
/// continued: with $s = z_0^2 + z_1^2$,
/// $\phi(\mathbf{z}) = \tfrac12[(z_0-\beta_0)^2 + (z_1-\beta_1)^2]
///  - \kappa / (1 + s)$.
///
/// The radius-squared enters as the holomorphic quadratic form $z_0^2+z_1^2$
/// (not the non-analytic $|\mathbf{z}|^2$), so the phase is entire away from
/// the polar locus $s = -1$. Along any ray $\mathbf{z} = r\,\hat{n}$ the
/// restriction is analytic in the complex radius $r$ (poles at $r = \pm i$),
/// which is exactly what [`crate::PolarSolver`] needs — making this the
/// natural multi-image stress test for the polar wrapper, cross-checked
/// against the 2D mesh-flow solver.
pub struct RationalLens2D {
    pub beta: [f64; 2],
    pub kappa: f64,
}

impl Phase<2> for RationalLens2D {
    fn value(&self, z: &SVector<Complex64, 2>) -> Complex64 {
        let u = z[0] - self.beta[0];
        let v = z[1] - self.beta[1];
        let s = z[0] * z[0] + z[1] * z[1];
        0.5 * (u * u + v * v) - self.kappa / (1.0 + s)
    }

    fn gradient(&self, z: &SVector<Complex64, 2>) -> SVector<Complex64, 2> {
        let s = z[0] * z[0] + z[1] * z[1];
        let d = 1.0 + s;
        let c = 2.0 * self.kappa / (d * d);
        SVector::<Complex64, 2>::new(
            (z[0] - self.beta[0]) + c * z[0],
            (z[1] - self.beta[1]) + c * z[1],
        )
    }
}

/// One-dimensional lens with the 'simG' rational-Gaussian potential
/// (Shi 2024, fig. 1): $\psi(z) = 1 / P(z)$ with
/// $P(z) = \sum_{k=0}^{k_\max} z^{2k} / k!$, a partial sum of $e^{z^2}$.
/// As $k_\max \to \infty$ the lens approaches a true Gaussian.
///
/// $P$ has $2 k_\max$ complex zeros, i.e. that many poles of $\psi$, so this
/// is the deliberate stress test for the saddle finder: $\phi'(z) = 0$ has
/// many roots scattered through the plane. Validated by saddle count and by
/// thimble/mesh agreement.
pub struct SimGLens {
    pub beta: f64,
    pub kappa: f64,
    pub kmax: u32,
}

impl SimGLens {
    /// $P(z) = \sum_{k=0}^{k_\max} z^{2k}/k!$ and
    /// $P'(z) = \sum_{k=1}^{k_\max} 2k\,z^{2k-1}/k!$. The two sums are built
    /// from running even/odd powers of $z$, never dividing by $z$, so both
    /// are exact at $z = 0$ (where $P = 1$, $P' = 0$).
    fn poly(&self, z: Complex64) -> (Complex64, Complex64) {
        let z2 = z * z;
        let mut even = Complex64::new(1.0, 0.0); // z^{2k}
        let mut fact = 1.0_f64; // k!
        let mut p = Complex64::new(0.0, 0.0);
        for k in 0..=self.kmax {
            if k > 0 {
                fact *= k as f64;
            }
            p += even / fact;
            even *= z2;
        }
        let mut odd = z; // z^{2k-1} at k = 1
        let mut fact = 1.0_f64;
        let mut dp = Complex64::new(0.0, 0.0);
        for k in 1..=self.kmax {
            fact *= k as f64;
            dp += Complex64::from(2.0 * k as f64) * odd / fact;
            odd *= z2;
        }
        (p, dp)
    }
}

impl Phase<1> for SimGLens {
    fn value(&self, z: &SVector<Complex64, 1>) -> Complex64 {
        let z = z[0];
        let (p, _) = self.poly(z);
        let u = z - self.beta;
        0.5 * u * u - self.kappa / p
    }

    fn gradient(&self, z: &SVector<Complex64, 1>) -> SVector<Complex64, 1> {
        let z = z[0];
        let (p, dp) = self.poly(z);
        // d/dz [-kappa/P] = +kappa P'/P^2
        SVector::<Complex64, 1>::new((z - self.beta) + self.kappa * dp / (p * p))
    }
}
