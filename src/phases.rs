//! Library of ready-made phase functions with known reference solutions,
//! used for validation, convergence studies, and cross-code comparisons.
//!
//! Every phase here has a rational derivative, so all of them also declare
//! the [`RationalPhase`] / [`RationalPhase2D`] capability: the complete
//! (polynomial) saddle source applies. The coefficient expansions are
//! verified in this module's tests against the identity
//! $N(z) = \phi'(z) \cdot D(z)$ at sample points, so no hand algebra is
//! trusted.

use crate::Phase;
use crate::phase::{RationalPhase, RationalPhase2D};
use crate::roots::{poly_add, poly_derivative, poly_eval, poly_mul, poly_roots, poly_scale};
use nalgebra::SVector;
use num_complex::Complex64;

/// The bare-quadratic + rational-potential lens with *user-supplied*
/// coefficients: $\phi(z) = \tfrac12 (z-\beta)^2 - \kappa\,A(z)/B(z)$, where
/// $A$ (`num`) and $B$ (`den`) are polynomials given in ascending order.
///
/// This is the escape hatch for defining new 1D rational lenses without a
/// per-evaluation callback — value, gradient, the complete stationary
/// polynomial, and the poles are all assembled from `A, B` with the
/// polynomial utilities in [`crate::roots`], so the full [`RationalPhase`]
/// machinery (Durand-Kerner saddles, thimble tracer) applies unchanged. A
/// constant `den = [1]` gives a polynomial potential (no poles).
#[derive(Clone)]
pub struct CustomRationalLens {
    pub beta: f64,
    pub kappa: f64,
    /// Potential numerator $A$, ascending coefficients.
    pub num: Vec<Complex64>,
    /// Potential denominator $B$, ascending coefficients (`[1]` ⇒ polynomial).
    pub den: Vec<Complex64>,
}

impl Phase<1> for CustomRationalLens {
    fn value(&self, z: &SVector<Complex64, 1>) -> Complex64 {
        let z = z[0];
        let u = z - self.beta;
        0.5 * u * u - self.kappa * poly_eval(&self.num, z) / poly_eval(&self.den, z)
    }

    fn gradient(&self, z: &SVector<Complex64, 1>) -> SVector<Complex64, 1> {
        let z = z[0];
        let a = poly_eval(&self.num, z);
        let b = poly_eval(&self.den, z);
        let da = poly_eval(&poly_derivative(&self.num), z);
        let db = poly_eval(&poly_derivative(&self.den), z);
        // d/dz [-kappa A/B] = -kappa (A'B - A B') / B^2
        SVector::<Complex64, 1>::new((z - self.beta) - self.kappa * (da * b - a * db) / (b * b))
    }
}

impl RationalPhase for CustomRationalLens {
    /// $N = (z-\beta) B^2 - \kappa (A'B - A B')$, from clearing $B^2$ out of
    /// $\phi'(z) = 0$.
    fn stationary_polynomial(&self) -> Vec<Complex64> {
        let a = &self.num;
        let b = &self.den;
        let da = poly_derivative(a);
        let db = poly_derivative(b);
        let b2 = poly_mul(b, b);
        let zmb = [Complex64::from(-self.beta), Complex64::from(1.0)];
        let cross = poly_add(&poly_mul(&da, b), &poly_scale(&poly_mul(a, &db), Complex64::from(-1.0)));
        poly_add(
            &poly_mul(&zmb, &b2),
            &poly_scale(&cross, Complex64::from(-self.kappa)),
        )
    }

    fn poles(&self) -> Vec<Complex64> {
        if self.den.len() <= 1 {
            Vec::new()
        } else {
            poly_roots(&self.den).unwrap_or_default()
        }
    }
}

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

impl RationalPhase for AiryPhase {
    /// $\phi'(z) = z^2 - y$ is already a polynomial: $N = z^2 - y$, $D = 1$.
    fn stationary_polynomial(&self) -> Vec<Complex64> {
        vec![
            Complex64::from(-self.y),
            Complex64::new(0.0, 0.0),
            Complex64::from(1.0),
        ]
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

/// $N(z) = (z-\beta)(1+z^2)^2 + 2\kappa z$ (denominator $D = (1+z^2)^2$),
/// shared by the 1D lens and the 2D lens's radial restriction.
fn rational_lens_polynomial(beta: f64, kappa: f64) -> Vec<Complex64> {
    // Expanded: z^5 - b z^4 + 2 z^3 - 2 b z^2 + (1 + 2 k) z - b.
    [-beta, 1.0 + 2.0 * kappa, -2.0 * beta, 2.0, -beta, 1.0]
        .into_iter()
        .map(Complex64::from)
        .collect()
}

impl RationalPhase for RationalLens {
    fn stationary_polynomial(&self) -> Vec<Complex64> {
        rational_lens_polynomial(self.beta, self.kappa)
    }

    fn poles(&self) -> Vec<Complex64> {
        vec![Complex64::i(), -Complex64::i()]
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

impl RationalPhase2D for RationalLens2D {
    /// Along $\mathbf{z} = r \hat{n}$: $\phi_\theta(r) = (r^2 - 2 r b +
    /// |\beta|^2)/2 - \kappa/(1+r^2)$ with $b = \beta \cdot \hat{n}$, so the
    /// radial derivative clears to the same quintic as the 1D lens with
    /// $\beta \to b$.
    fn radial_stationary_polynomial(&self, cos_t: f64, sin_t: f64) -> Vec<Complex64> {
        let b = self.beta[0] * cos_t + self.beta[1] * sin_t;
        rational_lens_polynomial(b, self.kappa)
    }

    fn radial_poles(&self, _cos_t: f64, _sin_t: f64) -> Vec<Complex64> {
        vec![Complex64::i(), -Complex64::i()]
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

impl SimGLens {
    /// Ascending coefficients of $P(z) = \sum_{k=0}^{k_\max} z^{2k}/k!$.
    fn p_coeffs(&self) -> Vec<Complex64> {
        let mut c = vec![Complex64::new(0.0, 0.0); 2 * self.kmax as usize + 1];
        let mut fact = 1.0_f64;
        for k in 0..=self.kmax as usize {
            if k > 0 {
                fact *= k as f64;
            }
            c[2 * k] = Complex64::from(1.0 / fact);
        }
        c
    }
}

impl RationalPhase for SimGLens {
    /// $N = (z - \beta) P^2 + \kappa P'$, degree $4 k_\max + 1$
    /// (17 stationary points at $k_\max = 4$, matching Shi 2024, appendix A).
    fn stationary_polynomial(&self) -> Vec<Complex64> {
        let p = self.p_coeffs();
        let p2 = poly_mul(&p, &p);
        let z_minus_beta = [Complex64::from(-self.beta), Complex64::from(1.0)];
        poly_add(
            &poly_mul(&z_minus_beta, &p2),
            &poly_scale(&poly_derivative(&p), Complex64::from(self.kappa)),
        )
    }

    /// The $2 k_\max$ zeros of $P$, computed on call via the polynomial
    /// root finder (an empty list is returned in the — unexpected —
    /// event that the root finder fails on $P$).
    fn poles(&self) -> Vec<Complex64> {
        poly_roots(&self.p_coeffs()).unwrap_or_default()
    }
}

/// A field of point-mass lenses in the 2D lensing plane — the microlensing
/// / stellar-field problem.
///
/// The Fermat phase is the holomorphic continuation of
/// $\phi(\mathbf{x}) = \tfrac12 |\mathbf{x} - \boldsymbol\beta|^2 - \sum_j m_j
/// \ln|\mathbf{x} - \mathbf{x}_j|$ (each point mass $j$ contributes a
/// logarithmic lensing potential, with $m_j$ its mass in Einstein-radius$^2$
/// units), obtained by writing $|\mathbf{x} - \mathbf{x}_j|^2$ as the
/// *holomorphic* quadratic form $w_j = (z_0 - a_j)^2 + (z_1 - b_j)^2$ and
/// $\ln|\cdot| = \tfrac12 \ln w_j$:
/// $$ \phi(\mathbf{z}) = \tfrac12\big[(z_0-\beta_0)^2 + (z_1-\beta_1)^2\big] -
///    \sum_j \tfrac{m_j}{2}\,\ln w_j. $$
///
/// The field is generally **not axisymmetric**, so only the general
/// mesh-flow [`crate::Solver`]`<2>` applies — the polar wrapper does not.
/// The logarithm's branch points lie on the complex variety $w_j = 0$, off
/// the real plane (where $\phi$ is real and $h = 0$); the mesh flow's
/// $h$-monotonicity guard keeps the deforming contour from tunnelling across
/// them, exactly as it does for the rational lens's poles.
pub struct PointMassField {
    /// Source position $\boldsymbol\beta$.
    pub beta: [f64; 2],
    /// Each lens as `(x, y, mass)`.
    pub lenses: Vec<(f64, f64, f64)>,
}

impl Phase<2> for PointMassField {
    fn value(&self, z: &SVector<Complex64, 2>) -> Complex64 {
        let u = z[0] - self.beta[0];
        let v = z[1] - self.beta[1];
        let mut acc = 0.5 * (u * u + v * v);
        for &(a, b, m) in &self.lenses {
            let du = z[0] - a;
            let dv = z[1] - b;
            let w = du * du + dv * dv;
            acc -= 0.5 * m * w.ln();
        }
        acc
    }

    fn gradient(&self, z: &SVector<Complex64, 2>) -> SVector<Complex64, 2> {
        let mut g0 = z[0] - self.beta[0];
        let mut g1 = z[1] - self.beta[1];
        for &(a, b, m) in &self.lenses {
            let du = z[0] - a;
            let dv = z[1] - b;
            let w = du * du + dv * dv;
            g0 -= m * du / w;
            g1 -= m * dv / w;
        }
        SVector::<Complex64, 2>::new(g0, g1)
    }
}

/// The isolated point-mass lens, axisymmetric and holomorphically continued:
/// with $s = z_0^2 + z_1^2$ (so $s = r^2$ along any ray from the origin),
/// $\phi(\mathbf{z}) = \tfrac12[(z_0 - y)^2 + z_1^2] - \tfrac12 \ln s$ — the
/// Fermat potential $\tfrac12|\mathbf{x} - \mathbf{y}|^2 - \ln|\mathbf{x}|$ of a
/// unit point mass ($\psi = \ln r$) with the source at $(y, 0)$.
///
/// Unlike the rational lenses, this potential has a **logarithmic branch point**
/// at the origin (not a pole): along a ray the radial restriction is
/// $\phi_\theta(r) = \tfrac12 r^2 - b r + \tfrac12 y^2 - \ln r$ with
/// $b = y\cos\theta$, and the integrand carries an $r^{-i\omega}$ factor whose
/// cut is the imaginary $r$-axis — which stalls a direct half-line trace at low
/// $\omega$. The fast path ([`crate::solve_point_mass_polar`]) instead solves
/// each ray in $u = \ln r$, where the phase is entire (no cut) and the radial
/// saddles are $u = \ln\rho$ with $\rho$ the roots of the point-mass lens
/// equation $\rho^2 - b\rho - 1 = 0$. This is the axisymmetric counterpart to
/// [`PointMassField`] that the fast solver can handle (the field itself is
/// non-axisymmetric — mesh only), validated against the Nakamura closed form.
/// The complex-`ln` value/gradient here is the reusable kernel for a future
/// general 2D log-thimble.
pub struct PointMass {
    /// Source impact parameter $y$ (source at $(y, 0)$, lens at the origin).
    pub y: f64,
}

impl Phase<2> for PointMass {
    fn value(&self, z: &SVector<Complex64, 2>) -> Complex64 {
        let u = z[0] - self.y;
        let v = z[1];
        let s = z[0] * z[0] + z[1] * z[1];
        0.5 * (u * u + v * v) - 0.5 * s.ln()
    }

    fn gradient(&self, z: &SVector<Complex64, 2>) -> SVector<Complex64, 2> {
        let s = z[0] * z[0] + z[1] * z[1];
        SVector::<Complex64, 2>::new((z[0] - self.y) - z[0] / s, z[1] - z[1] / s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roots::poly_eval;
    use crate::saddles::grad;

    /// Sample points for the identity checks, including one near the
    /// rational lens's poles at ±i.
    fn samples() -> Vec<Complex64> {
        vec![
            Complex64::new(0.3, 0.4),
            Complex64::new(-1.2, 0.7),
            Complex64::new(2.0, -0.3),
            Complex64::new(0.05, 1.2),
            Complex64::new(-0.8, -1.1),
        ]
    }

    /// Assert N(z) = phi'(z) * D(z) at the sample points.
    fn check_identity(phase: &dyn Phase<1>, n: &[Complex64], d: &[Complex64]) {
        for z in samples() {
            let lhs = poly_eval(n, z);
            let rhs = grad(phase, z) * poly_eval(d, z);
            assert!(
                (lhs - rhs).norm() < 1e-10 * (1.0 + lhs.norm() + rhs.norm()),
                "identity fails at {z}: N = {lhs}, phi' D = {rhs}"
            );
        }
    }

    #[test]
    fn airy_polynomial_identity() {
        let phase = AiryPhase { y: 0.8 };
        check_identity(
            &phase,
            &phase.stationary_polynomial(),
            &[Complex64::from(1.0)],
        );
    }

    #[test]
    fn rational_lens_polynomial_identity() {
        let phase = RationalLens {
            beta: 0.37,
            kappa: -0.42,
        };
        // D = (1 + z^2)^2, built independently of the hand expansion.
        let one_plus_z2 = [
            Complex64::from(1.0),
            Complex64::new(0.0, 0.0),
            Complex64::from(1.0),
        ];
        let d = poly_mul(&one_plus_z2, &one_plus_z2);
        check_identity(&phase, &phase.stationary_polynomial(), &d);
    }

    #[test]
    fn rational_lens_2d_radial_identity() {
        let lens = RationalLens2D {
            beta: [0.4, -0.25],
            kappa: 0.3,
        };
        let one_plus_z2 = [
            Complex64::from(1.0),
            Complex64::new(0.0, 0.0),
            Complex64::from(1.0),
        ];
        let d = poly_mul(&one_plus_z2, &one_plus_z2);
        for theta in [0.0, 0.9, 2.3, 4.1] {
            let (c, s) = (f64::cos(theta), f64::sin(theta));
            let n = lens.radial_stationary_polynomial(c, s);
            for r in samples() {
                // Radial derivative by the chain rule from the 2D gradient.
                let g2 = lens.gradient(&SVector::<Complex64, 2>::new(r * c, r * s));
                let radial_grad = g2[0] * c + g2[1] * s;
                let lhs = poly_eval(&n, r);
                let rhs = radial_grad * poly_eval(&d, r);
                assert!(
                    (lhs - rhs).norm() < 1e-10 * (1.0 + lhs.norm() + rhs.norm()),
                    "radial identity fails at theta={theta}, r={r}"
                );
            }
        }
    }

    #[test]
    fn simg_polynomial_identity_and_degree() {
        let phase = SimGLens {
            beta: 0.4,
            kappa: 0.15,
            kmax: 4,
        };
        let n = phase.stationary_polynomial();
        assert_eq!(n.len(), 4 * 4 + 2, "degree must be 4 kmax + 1");
        let p = phase.p_coeffs();
        check_identity(&phase, &n, &poly_mul(&p, &p));
        // P has 2 kmax zeros = poles of the potential.
        assert_eq!(phase.poles().len(), 8);
    }

    #[test]
    fn custom_rational_lens_reproduces_builtin() {
        // psi = 1/(1+z^2): num = [1], den = [1, 0, 1] — must match RationalLens.
        let beta = 0.37;
        let kappa = -0.42;
        let custom = CustomRationalLens {
            beta,
            kappa,
            num: vec![Complex64::from(1.0)],
            den: vec![Complex64::from(1.0), Complex64::new(0.0, 0.0), Complex64::from(1.0)],
        };
        let builtin = RationalLens { beta, kappa };
        for z in samples() {
            let zc = SVector::<Complex64, 1>::new(z);
            assert!((custom.value(&zc) - builtin.value(&zc)).norm() < 1e-12);
            assert!((custom.gradient(&zc)[0] - builtin.gradient(&zc)[0]).norm() < 1e-12);
        }
        // Same complete saddle set.
        let sc = crate::saddles::saddles_from_polynomial(&custom, &custom.stationary_polynomial(), 1e-11).unwrap();
        let sb = crate::saddles::saddles_from_polynomial(&builtin, &builtin.stationary_polynomial(), 1e-11).unwrap();
        assert_eq!(sc.len(), sb.len());
        for (a, b) in sc.iter().zip(&sb) {
            assert!((a.z - b.z).norm() < 1e-9);
        }
        assert_eq!(custom.poles().len(), 2);
    }

    #[test]
    fn point_mass_field_is_real_on_the_plane_and_reduces_to_free() {
        let field = PointMassField {
            beta: [0.3, -0.2],
            lenses: vec![(1.0, 0.5, 0.4), (-0.8, 0.6, 0.3), (0.2, -1.1, 0.5)],
        };
        // Real on the real plane (h = 0 initially), away from the masses.
        for (x, y) in [(2.0, 1.3), (-1.5, 0.4), (0.0, 2.2)] {
            let v = field.value(&SVector::<Complex64, 2>::new(
                Complex64::from(x),
                Complex64::from(y),
            ));
            assert!(v.im.abs() < 1e-12, "phi not real at ({x},{y}): {v}");
        }
        // Gradient matches a finite difference of the value (guards the hand
        // derivative of the log term).
        let z = SVector::<Complex64, 2>::new(Complex64::new(1.7, 0.3), Complex64::new(-0.4, 0.9));
        let g = field.gradient(&z);
        let eps = 1e-6;
        for k in 0..2 {
            let mut zp = z;
            let mut zm = z;
            zp[k] += Complex64::from(eps);
            zm[k] += Complex64::from(-eps);
            let num = (field.value(&zp) - field.value(&zm)) / (2.0 * eps);
            assert!((g[k] - num).norm() < 1e-5, "grad[{k}] off: {} vs {num}", g[k]);
        }
        // Zero masses ⇒ the bare quadratic phase.
        let free = PointMassField {
            beta: [0.3, -0.2],
            lenses: vec![],
        };
        let u = Complex64::new(1.4, 0.0);
        let v = Complex64::new(0.5, 0.0);
        let got = free.value(&SVector::<Complex64, 2>::new(u, v));
        let want = 0.5 * ((u - 0.3) * (u - 0.3) + (v + 0.2) * (v + 0.2));
        assert!((got - want).norm() < 1e-14);
    }
}
