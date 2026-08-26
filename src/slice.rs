//! Cartesian-slice reduction of the 2D point-mass field (Path 2, milestone
//! M3, option A).
//!
//! The 2D Fresnel integral
//! $I(\beta) = \iint e^{i\omega\phi(x_0,x_1)}\,dx_0\,dx_1$ is evaluated as an
//! outer integral over $x_1$ of an **inner 1D thimble** in $x_0$:
//! $$ I = \int_{\mathbb R} G(x_1)\,dx_1, \qquad
//!    G(x_1) = \int_{\mathbb R} e^{i\omega\phi(z_0, x_1)}\,dz_0. $$
//! This module provides the inner kernel (milestone A0): the slice phase and
//! its complete polynomial saddle source, so each inner integral runs on the
//! proven full-line [`ThimbleSolver`] rather than the (log-unstable) mesh.
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

use crate::phase::{Phase, RationalPhase};
use crate::phases::PointMassField;
use crate::roots::{poly_add, poly_mul, poly_scale};
use crate::saddles::{Saddle, saddles_from_polynomial, second_derivative};
use crate::thimble::{ThimbleConfig, ThimbleSolver, integrate_thimbles, saddle_thimble};
use crate::{Output, Result};
use nalgebra::SVector;
use num_complex::Complex64;

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
    let cfg = ThimbleConfig {
        continuous_phase: true,
        ..cfg.clone()
    };
    ThimbleSolver::new_rational(&slice, cfg).run()
}

/// The slice phase around a single log branch point `p`, in the coordinate
/// `u` with `z0 = p + e^u`. The substitution turns the `p`-log into the *linear*
/// (entire) term `-(m/2) u` — the C1 `r = e^u` trick applied to an interior
/// branch point — so the branch point maps to the regular endpoint `u -> -oo`,
/// where the `e^u` amplitude kills the integrand. The remaining logs (the other
/// masses and `p`'s partner) stay, tracked by the continuous-phase machinery.
struct HankelUPhase<'a> {
    field: &'a PointMassField,
    x2: Complex64,
    p: Complex64,
}

impl<'a> HankelUPhase<'a> {
    fn z_of(&self, u: Complex64) -> Complex64 {
        self.p + u.exp()
    }
}

impl<'a> Phase<1> for HankelUPhase<'a> {
    fn value(&self, u: &SVector<Complex64, 1>) -> Complex64 {
        let uu = u[0];
        let z0 = self.z_of(uu);
        let du = z0 - self.field.beta[0];
        let dv = self.x2 - self.field.beta[1];
        let mut acc = 0.5 * (du * du + dv * dv);
        for &(a, b, m) in &self.field.lenses {
            let d = Complex64::i() * (self.x2 - b);
            for q in [a + d, a - d] {
                if (q - self.p).norm() < 1e-9 * (1.0 + self.p.norm()) {
                    // ln(z0 - p) = u, the linear entire term.
                    acc -= 0.5 * m * uu;
                } else {
                    acc -= 0.5 * m * (z0 - q).ln();
                }
            }
        }
        acc
    }

    fn gradient(&self, u: &SVector<Complex64, 1>) -> SVector<Complex64, 1> {
        // dPhi/du = phi'(z0) * dz0/du = phi'(z0) * e^u; phi' is single-valued.
        let eu = u[0].exp();
        let z0 = self.p + eu;
        let g = self
            .field
            .gradient(&SVector::<Complex64, 2>::new(z0, self.x2));
        SVector::<Complex64, 1>::new(g[0] * eu)
    }
}

/// Per-saddle diagnostic: for each slice saddle, its (position, Morse height
/// `h`, unit-multiplicity thimble integral). Lets us check whether the missing
/// swept-branch-point contribution is just a sub-dominant saddle the standard
/// (single-valued) effectiveness test drops.
pub fn slice_saddle_contributions(
    field: &PointMassField,
    x2: f64,
    cfg: &ThimbleConfig,
) -> Result<Vec<(Complex64, f64, Complex64)>> {
    let slice = X1SlicePhase {
        field,
        x2: Complex64::from(x2),
    };
    let cfg = ThimbleConfig {
        continuous_phase: true,
        ..cfg.clone()
    };
    let saddles = saddles_from_polynomial(&slice, &slice.stationary_polynomial(), cfg.newton_tol)?;
    let amp = |_: Complex64| Complex64::new(1.0, 0.0);
    let mut out = Vec::new();
    for s in &saddles {
        let contrib = saddle_thimble(&slice, &cfg, s, &amp)?;
        out.push((s.z, -s.value.im, contrib));
    }
    Ok(out)
}

/// The branch-cut *ray* integral for one log branch point `p`: the thimble sum
/// of the `u = ln(z0 - p)` phase [`HankelUPhase`] over the full `u`-line, with
/// the `dz0 = e^u du` amplitude. Its saddles are exactly the slice saddles
/// mapped `u = ln(z0_saddle - p)` (since `phi'(z0)=0` iff `dPhi/du=0`), so no
/// new saddle search is needed. Returns the raw ray integral `I_ray(p)`; the
/// caller applies the monodromy prefactor and decides whether `p` is swept.
pub fn hankel_ray(field: &PointMassField, x2: f64, p: Complex64, cfg: &ThimbleConfig) -> Result<Complex64> {
    let x2c = Complex64::from(x2);
    let phase = HankelUPhase { field, x2: x2c, p };

    let slice = X1SlicePhase { field, x2: x2c };
    let z_saddles = saddles_from_polynomial(&slice, &slice.stationary_polynomial(), cfg.newton_tol)?;
    let u_saddles: Vec<Saddle> = z_saddles
        .iter()
        .map(|s| {
            let u = (s.z - p).ln();
            Saddle {
                z: u,
                value: phase.value(&SVector::<Complex64, 1>::new(u)),
                second: second_derivative(&phase, u),
            }
        })
        .collect();

    let cfg = ThimbleConfig {
        continuous_phase: true,
        ..cfg.clone()
    };
    let amplitude = |u: Complex64| u.exp();
    integrate_thimbles(&phase, &cfg, &amplitude, &u_saddles).map(|o| o.value)
}
