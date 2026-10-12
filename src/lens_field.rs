//! The general lens model of the plan's P2: a superposition of axisymmetric
//! star / halo profiles plus the external macro model (convergence `kappa` and
//! shear `gamma`), solved by the *same* RC slice engine as the point-mass field
//! ([`crate::slice`]) — the engine only needs a [`SliceModel`] implementation.
//!
//! Phase (in the frame where the shear is diagonal, `lambda0 = 1 - kappa - gamma`,
//! `lambda2 = 1 - kappa + gamma`; component centres and the source rotated by
//! `-atan2(g2, g1)/2`, an exact measure-preserving reduction since every
//! component is axisymmetric):
//!
//! ```text
//! phi = 1/2 lambda0 z0^2 + 1/2 lambda2 x2^2 - beta0 z0 - beta1 x2
//!       + 1/2 (beta0^2 + beta1^2) - sum_j psi_j(w_j),
//! w_j = (z0 - x_j)^2 + (x2 - y_j)^2.
//! ```
//!
//! At `kappa = gamma = 0` with point components this is exactly
//! [`crate::phases::PointMassField`]'s phase.  The exactness argument is the
//! P0-validated one: the tails rotate by `sign(lambda)*angle` into the Gaussian
//! valleys (`angle <= pi/4` is what makes every `Re w_j > 0` on the contour,
//! including the mixed-sign saddle macro images), so no branch point is ever
//! swept and the principal-branch potentials are the analytic continuation.
//!
//! Profile potentials (P0-validated forms, principal branches, the cut on
//! `(-inf, 0]`): see [`Profile`].

use crate::error::{Error, Result};
use crate::phase::Phase;
use crate::slice::{SliceConfig, SliceEval, SliceFieldOutput, SliceModel, slice_field_g};
use nalgebra::SVector;
use num_complex::Complex64;

/// Complex `log1p` with a series for tiny `|z|` (the NFW branch factor as
/// `u -> 0`): numpy's complex `log1p` analogue of the Python referee's fix —
/// `log(1 + z)` loses all relative accuracy for `|z| ~ 1e-16`.
fn log1p_c(z: Complex64) -> Complex64 {
    if z.norm() < 0.1 {
        let mut acc = Complex64::new(1.0 / 30.0, 0.0);
        for k in (1..30).rev() {
            acc = acc * (-z) + Complex64::new(1.0 / k as f64, 0.0);
        }
        z * acc
    } else {
        (Complex64::new(1.0, 0.0) + z).ln()
    }
}

/// Axisymmetric radial profile `psi_j(w)`, `w` the *squared* holomorphic
/// radius.  Every variant is analytic in the complex plane minus the cut
/// `(-inf, 0]` with principal branches — exactly the class the RC contour
/// keeps (`Re w > 0` along the whole contour).
#[derive(Debug, Clone, Copy)]
pub enum Profile {
    /// `(m/2) ln(w + xc^2)`; `xc = 0` is the pure point mass.
    Point { mass: f64, xc: f64 },
    /// `psi0 sqrt(w)`.
    Sis { psi0: f64 },
    /// `psi0 [ s + xc ln(2 xc/(s + xc)) ]`, `s = sqrt(w + xc^2)` (cored
    /// isothermal, the FIONA/GLoW normalisation; `xc = 0` reduces to SIS).
    Cis { psi0: f64, xc: f64 },
    /// `(psi0/2) ln((1-s)/2) ln((1+s)/2)`, `s = sqrt(1 - w/xs^2)` (NFW in the
    /// factored, cancellation-free form; on the real axis identical to
    /// FIONA/GLoW's `(psi0/2)[ln^2(u/2) + (u^2-1)F(u)^2]`).
    Nfw { psi0: f64, xs: f64 },
}

impl Profile {
    /// `psi(w)` at complex `w` (principal branches).
    pub fn psi(&self, w: Complex64) -> Complex64 {
        match *self {
            Profile::Point { mass, xc } => {
                Complex64::new(0.5 * mass, 0.0) * (w + Complex64::from(xc * xc)).ln()
            }
            Profile::Sis { psi0 } => Complex64::new(psi0, 0.0) * w.sqrt(),
            Profile::Cis { psi0, xc } => {
                if xc == 0.0 {
                    Complex64::new(psi0, 0.0) * w.sqrt()
                } else {
                    let s = (w + Complex64::from(xc * xc)).sqrt();
                    Complex64::new(psi0, 0.0)
                        * (s + Complex64::from(xc) * (Complex64::from(2.0 * xc) / (s + Complex64::from(xc))).ln())
                }
            }
            Profile::Nfw { psi0, xs } => {
                let u2 = w / Complex64::from(xs * xs);
                let s = (Complex64::new(1.0, 0.0) - u2).sqrt();
                let first = (u2 / (Complex64::new(2.0, 0.0) * (Complex64::new(1.0, 0.0) + s))).ln();
                let half = log1p_c(-u2 / (Complex64::new(2.0, 0.0) * (Complex64::new(1.0, 0.0) + s)));
                Complex64::new(0.5 * psi0, 0.0) * first * half
            }
        }
    }

    /// A representative `|psi|` for the roundoff-floor bound (a deliberate
    /// over-estimate; anything too small only costs extra refinement).
    fn reference_magnitude(&self) -> f64 {
        match *self {
            Profile::Point { mass, xc } => 0.5 * mass.abs() * (4.0 + xc * xc).ln().abs() + 1.0,
            Profile::Sis { psi0 } => psi0.abs() * 8.0 + 1.0,
            Profile::Cis { psi0, xc } => psi0.abs() * (8.0 + xc.abs() + 1.0),
            Profile::Nfw { psi0, xs } => {
                0.5 * psi0.abs() * (2.0 + (8.0 / xs.abs().max(1e-12)).max(1.0).ln()).powi(2) + 1.0
            }
        }
    }

    fn validate(&self) -> Result<()> {
        match *self {
            Profile::Point { mass, xc } => {
                if !mass.is_finite() || !xc.is_finite() || xc < 0.0 {
                    return Err(Error::InvalidInput("point profile: mass/xc must be finite, xc >= 0"));
                }
            }
            Profile::Sis { psi0 } => {
                if !psi0.is_finite() {
                    return Err(Error::InvalidInput("SIS profile: psi0 must be finite"));
                }
            }
            Profile::Cis { psi0, xc } => {
                if !psi0.is_finite() || !xc.is_finite() || xc < 0.0 {
                    return Err(Error::InvalidInput("CIS profile: psi0/xc must be finite, xc >= 0"));
                }
            }
            Profile::Nfw { psi0, xs } => {
                if !psi0.is_finite() || !xs.is_finite() || xs <= 0.0 {
                    return Err(Error::InvalidInput("NFW profile: psi0 finite, xs > 0"));
                }
            }
        }
        Ok(())
    }

    /// Zero-weight component (invisible to the integral).
    fn is_zero(&self) -> bool {
        match *self {
            Profile::Point { mass, .. } => mass == 0.0,
            Profile::Sis { psi0 } => psi0 == 0.0,
            Profile::Cis { psi0, .. } => psi0 == 0.0,
            Profile::Nfw { psi0, .. } => psi0 == 0.0,
        }
    }
}

/// One lens component at `(x, y)` with an axisymmetric profile.
#[derive(Debug, Clone, Copy)]
pub struct Component {
    pub x: f64,
    pub y: f64,
    pub profile: Profile,
}

/// Stars + halos + macro model, in the shear-diagonal frame.  Construct with
/// [`LensField::new`], which performs the rotation and the `lambda != 0` checks;
/// all fields are then in that frame (and `with_beta` expects source positions
/// in the same frame — the batched entry points only use the zero source).
#[derive(Debug, Clone)]
pub struct LensField {
    pub beta: [f64; 2],
    /// `1 - kappa - gamma` (`gamma = |(gamma1, gamma2)|`).
    pub lambda0: f64,
    /// `1 - kappa + gamma`.
    pub lambda2: f64,
    /// `1/2 (beta0^2 + beta1^2)` (the phase's source constant).
    pub kc: f64,
    pub components: Vec<Component>,
}

impl LensField {
    /// Rotate into the shear-diagonal frame and validate.
    pub fn new(
        beta: [f64; 2],
        kappa: f64,
        gamma1: f64,
        gamma2: f64,
        components: Vec<Component>,
    ) -> Result<Self> {
        for c in &components {
            if !c.x.is_finite() || !c.y.is_finite() {
                return Err(Error::InvalidInput("component centres must be finite"));
            }
            c.profile.validate()?;
        }
        if !beta[0].is_finite() || !beta[1].is_finite() || !kappa.is_finite() {
            return Err(Error::InvalidInput("source/kappa must be finite"));
        }
        if !gamma1.is_finite() || !gamma2.is_finite() {
            return Err(Error::InvalidInput("shear must be finite"));
        }
        let gamma = gamma1.hypot(gamma2);
        let th = if gamma != 0.0 { 0.5 * gamma2.atan2(gamma1) } else { 0.0 };
        let (ct, st) = (th.cos(), th.sin());
        let rot = |x: f64, y: f64| (ct * x + st * y, -st * x + ct * y);
        let lambda0 = 1.0 - kappa - gamma;
        let lambda2 = 1.0 - kappa + gamma;
        if lambda0 == 0.0 || lambda2 == 0.0 {
            return Err(Error::InvalidInput(
                "critical macro model: lambda0 or lambda2 = 0 is rejected",
            ));
        }
        let (b0, b1) = rot(beta[0], beta[1]);
        let components = components
            .into_iter()
            .map(|c| {
                let (x, y) = rot(c.x, c.y);
                Component { x, y, profile: c.profile }
            })
            .collect();
        Ok(LensField {
            beta: [b0, b1],
            lambda0,
            lambda2,
            kc: 0.5 * (b0 * b0 + b1 * b1),
            components,
        })
    }
}

/// Per-slice evaluation state of a [`LensField`].
pub(crate) struct LensEval<'a> {
    field: &'a LensField,
    /// `x2 - y_k` per component (the anchored slice offset).
    dk: Vec<Complex64>,
    /// The slice position itself (the `lambda2 x2^2/2 - beta1 x2` terms).
    x2: Complex64,
}

impl SliceModel for LensField {
    type Eval<'a> = LensEval<'a>;

    fn beta(&self) -> [f64; 2] {
        self.beta
    }

    fn abscissae(&self) -> Vec<f64> {
        self.components.iter().map(|c| c.x).collect()
    }

    fn outer_abscissae(&self) -> Vec<f64> {
        self.components.iter().map(|c| c.y).collect()
    }

    fn potential_scale(&self) -> f64 {
        self.components.iter().map(|c| c.profile.reference_magnitude()).sum()
    }

    fn validate_model(&self) -> Result<()> {
        for c in &self.components {
            if !c.x.is_finite() || !c.y.is_finite() {
                return Err(Error::InvalidInput("component centres must be finite"));
            }
            c.profile.validate()?;
        }
        Ok(())
    }

    fn inner_tail(&self) -> (f64, f64, f64) {
        (self.lambda0.signum(), self.lambda0.abs(), self.beta[0].abs())
    }

    fn outer_tail(&self) -> (f64, f64, f64) {
        (self.lambda2.signum(), self.lambda2.abs(), self.beta[1].abs())
    }

    fn centres(&self) -> (f64, f64) {
        (self.beta[0] / self.lambda0, self.beta[1] / self.lambda2)
    }

    fn validate_slice(&self, base: f64, off: Complex64) -> Result<()> {
        // Sufficient condition for the whole inner contour to stay off the
        // cut: Re w_j > 0 there whenever Re(d_j^2) > 0 (see the module docs).
        for c in &self.components {
            if c.profile.is_zero() {
                continue;
            }
            let d = Complex64::from(base - c.y) + off;
            if (d * d).re <= 0.0 {
                return Err(Error::InvalidInput(
                    "complex slice position outside the validated sector: Re((x2-y)^2) <= 0",
                ));
            }
        }
        Ok(())
    }

    fn evaluate(&self, base: f64, off: Complex64) -> LensEval<'_> {
        LensEval {
            field: self,
            dk: self.components.iter().map(|c| Complex64::from(base - c.y) + off).collect(),
            x2: Complex64::from(base) + off,
        }
    }

    fn with_beta(&self, beta: [f64; 2]) -> Self {
        LensField { beta, kc: 0.5 * (beta[0] * beta[0] + beta[1] * beta[1]), ..self.clone() }
    }

    fn pruned(&self) -> Self {
        LensField {
            components: self.components.iter().copied().filter(|c| !c.profile.is_zero()).collect(),
            ..self.clone()
        }
    }
}

impl<'a> LensEval<'a> {
    /// The `beta`-independent, `z0`-linear part of the phase on a real node:
    /// `1/2 lambda0 z0^2 - beta0 z0` with `z0 = x0 + sgn*r`.
    #[inline]
    fn quad_real(&self, z0: f64) -> f64 {
        0.5 * self.field.lambda0 * z0 * z0 - self.field.beta[0] * z0
    }
    /// The `x2`-part (constant in `z0`): `1/2 lambda2 x2^2 - beta1 x2 + kc`.
    #[inline]
    fn x2_part_real(&self) -> f64 {
        let x2 = self.x2.re;
        0.5 * self.field.lambda2 * x2 * x2 - self.field.beta[1] * x2 + self.field.kc
    }
    #[inline]
    fn x2_part_complex(&self) -> Complex64 {
        0.5 * Complex64::from(self.field.lambda2) * self.x2 * self.x2
            - Complex64::from(self.field.beta[1]) * self.x2
            + Complex64::from(self.field.kc)
    }
}

impl<'a> SliceEval for LensEval<'a> {
    #[inline]
    fn phi_core_real(&self, r: f64, x0: f64, sgn: f64) -> f64 {
        let z0 = x0 + sgn * r;
        let mut v = self.quad_real(z0) + self.x2_part_real();
        for (k, c) in self.field.components.iter().enumerate() {
            let t = (x0 - c.x) + sgn * r;
            let d = self.dk[k].re;
            let w = t * t + d * d;
            v -= match c.profile {
                Profile::Point { mass, xc } => 0.5 * mass * (w + xc * xc).ln(),
                _ => c.profile.psi(Complex64::from(w)).re,
            };
        }
        v
    }

    #[inline]
    fn phi_core_complex(&self, r: f64, x0: f64, sgn: f64) -> Complex64 {
        let z0 = Complex64::from(x0 + sgn * r);
        let mut v = 0.5 * Complex64::from(self.field.lambda0) * z0 * z0
            - Complex64::from(self.field.beta[0]) * z0
            + self.x2_part_complex();
        for (k, c) in self.field.components.iter().enumerate() {
            let t = z0 - c.x;
            let w = t * t + self.dk[k] * self.dk[k];
            v -= c.profile.psi(w);
        }
        v
    }

    #[inline]
    fn phi_tail(&self, z: Complex64) -> Complex64 {
        let mut v = 0.5 * Complex64::from(self.field.lambda0) * z * z
            - Complex64::from(self.field.beta[0]) * z
            + self.x2_part_complex();
        for (k, c) in self.field.components.iter().enumerate() {
            let t = z - c.x;
            let w = t * t + self.dk[k] * self.dk[k];
            v -= c.profile.psi(w);
        }
        v
    }

    #[inline]
    fn slice_quad_sq(&self) -> f64 {
        // Twice a bound on the x2-part of the phase, for the floor:
        // |1/2 lambda2 x2^2 - beta1 x2 + kc| <= (that) / 2.
        let x2 = self.x2.norm();
        self.field.lambda2.abs() * x2 * x2
            + 2.0 * self.field.beta[1].abs() * x2
            + 2.0 * self.field.kc.abs()
    }
}

impl Phase<2> for LensField {
    fn value(&self, z: &SVector<Complex64, 2>) -> Complex64 {
        let (z0, x2) = (z[0], z[1]);
        let mut v = 0.5 * Complex64::from(self.lambda0) * z0 * z0
            + 0.5 * Complex64::from(self.lambda2) * x2 * x2
            - Complex64::from(self.beta[0]) * z0
            - Complex64::from(self.beta[1]) * x2
            + Complex64::from(self.kc);
        for c in &self.components {
            let d0 = z0 - c.x;
            let d1 = x2 - c.y;
            v -= c.profile.psi(d0 * d0 + d1 * d1);
        }
        v
    }

    fn gradient(&self, z: &SVector<Complex64, 2>) -> SVector<Complex64, 2> {
        let (z0, x2) = (z[0], z[1]);
        let mut g0 = Complex64::from(self.lambda0) * z0 - Complex64::from(self.beta[0]);
        let mut g1 = Complex64::from(self.lambda2) * x2 - Complex64::from(self.beta[1]);
        for c in &self.components {
            let d0 = z0 - c.x;
            let d1 = x2 - c.y;
            let w = d0 * d0 + d1 * d1;
            let dpsi = c.profile.dpsi_dw(w);
            g0 -= dpsi * 2.0 * d0;
            g1 -= dpsi * 2.0 * d1;
        }
        SVector::<Complex64, 2>::new(g0, g1)
    }
}

impl Profile {
    /// `d psi / d w` (principal branch): used by the real-space geometry
    /// helpers (images / geometric optics).  For the NFW branch use a small
    /// complex step so no closed form has to be trusted unchecked.
    pub fn dpsi_dw(&self, w: Complex64) -> Complex64 {
        match *self {
            Profile::Point { mass, xc } => Complex64::new(0.5 * mass, 0.0) / (w + Complex64::from(xc * xc)),
            Profile::Sis { psi0 } => Complex64::new(0.5 * psi0, 0.0) / w.sqrt(),
            Profile::Cis { psi0, xc } => {
                if xc == 0.0 {
                    Complex64::new(0.5 * psi0, 0.0) / w.sqrt()
                } else {
                    let s = (w + Complex64::from(xc * xc)).sqrt();
                    Complex64::new(psi0, 0.0)
                        * (Complex64::new(1.0, 0.0) - Complex64::from(xc) / (s + Complex64::from(xc)))
                        / (Complex64::new(2.0, 0.0) * s)
                }
            }
            Profile::Nfw { .. } => {
                // central difference in the complex plane (the branch is
                // analytic in the neighbourhood of the real axis)
                let h = 1e-6 * (1.0 + w.norm());
                (self.psi(w + h) - self.psi(w - h)) / (2.0 * h)
            }
        }
    }
}

/// Exact RC field integral of a [`LensField`]: `solve` equivalent of
/// [`crate::slice::slice_field`] for the general model.
pub fn lens_field_solve(field: &LensField, cfg: &SliceConfig) -> Result<SliceFieldOutput> {
    slice_field_g(field, cfg)
}

/// Batched [`LensField`] integral (see [`crate::slice_batch::slice_field_batch`]).
pub fn lens_field_batch(
    field: &LensField,
    batch: &crate::slice_batch::FieldBatch,
    cfg: &SliceConfig,
) -> Result<crate::slice_batch::FieldBatchOutput> {
    crate::slice_batch::slice_field_batch_g(field, batch, cfg)
}

/// Real-space geometry of a [`LensField`]: the images (solutions of the lens
/// equation `grad phi = 0`) found by multi-start Newton, and the geometric
/// (stationary-phase) amplification.  The Jacobian/Hessian come from
/// complex-step derivatives of the holomorphic phase (machine-accurate, no
/// hand algebra beyond the gradient), so the κ/γ + profile cases reuse exactly
/// the same potential as the integral solver.
pub fn lens_field_images(field: &LensField) -> Vec<[f64; 2]> {
    let g = |x: [f64; 2]| -> [f64; 2] {
        let gr = field.gradient(&SVector::<Complex64, 2>::new(
            Complex64::from(x[0]),
            Complex64::from(x[1]),
        ));
        // phi is holomorphic in each variable and real at real arguments, so
        // the real gradient is just the holomorphic derivative there.
        [gr[0].re, gr[1].re]
    };
    let h = 1e-8;
    // Jacobian of the (real) gradient by complex steps (phi is holomorphic in
    // each variable, so dF/dz = Im(F(z + i h))/h exactly to machine order).
    let jac = |x: [f64; 2]| -> [[f64; 2]; 2] {
        let mut out = [[0.0; 2]; 2];
        for (j, slot) in out.iter_mut().enumerate() {
            for k in 0..2 {
                let mut z = [Complex64::from(x[0]), Complex64::from(x[1])];
                z[k] += Complex64::i() * h;
                let gr = field.gradient(&SVector::<Complex64, 2>::new(z[0], z[1]));
                slot[k] = if j == 0 { gr[0].im } else { gr[1].im } / h;
            }
        }
        out
    };
    let mut images: Vec<[f64; 2]> = Vec::new();
    let span = field
        .components
        .iter()
        .fold(2.0f64, |acc, c| acc.max(c.x.abs()).max(c.y.abs()));
    let span = span + 2.0;
    let n = 8;
    for i in 0..=n {
        for j in 0..=n {
            let mut x = [
                -span + 2.0 * span * i as f64 / n as f64,
                -span + 2.0 * span * j as f64 / n as f64,
            ];
            let mut ok = false;
            for _ in 0..60 {
                let gx = g(x);
                let jx = jac(x);
                let det = jx[0][0] * jx[1][1] - jx[0][1] * jx[1][0];
                if det.abs() < 1e-14 {
                    break;
                }
                let dx = [
                    (jx[1][1] * gx[0] - jx[0][1] * gx[1]) / det,
                    (-jx[1][0] * gx[0] + jx[0][0] * gx[1]) / det,
                ];
                x[0] -= dx[0];
                x[1] -= dx[1];
                if dx[0].abs() + dx[1].abs() < 1e-13 {
                    ok = true;
                    break;
                }
            }
            if ok && !images.iter().any(|p| (p[0] - x[0]).abs() + (p[1] - x[1]).abs() < 1e-7) {
                images.push(x);
            }
        }
    }
    images
}

/// Stationary-phase (geometric-optics) amplification of a [`LensField`] —
/// exact as `omega -> infinity` away from caustics; O(#images).
pub fn lens_field_geometric(field: &LensField, omega: f64) -> Complex64 {
    let h = 1e-8;
    let mut f = Complex64::new(0.0, 0.0);
    for x in lens_field_images(field) {
        // Hessian of the real potential by complex-step second derivatives.
        let mut hess = [[0.0f64; 2]; 2];
        for (j, slot) in hess.iter_mut().enumerate() {
            for k in 0..2 {
                let mut z = [Complex64::from(x[0]), Complex64::from(x[1])];
                z[k] += Complex64::i() * h;
                let gr = field.gradient(&SVector::<Complex64, 2>::new(z[0], z[1]));
                slot[k] = if j == 0 { gr[0].im } else { gr[1].im } / h;
            }
        }
        let det = hess[0][0] * hess[1][1] - hess[0][1] * hess[1][0];
        let tr = hess[0][0] + hess[1][1];
        if det.abs() < 1e-12 {
            continue; // at a caustic: the asymptotic form breaks down
        }
        let mu = 1.0 / det;
        let n = if det < 0.0 {
            0.5
        } else if tr > 0.0 {
            0.0
        } else {
            1.0
        };
        let xz = SVector::<Complex64, 2>::new(Complex64::from(x[0]), Complex64::from(x[1]));
        let phi = field.value(&xz).re;
        f += Complex64::from(mu.abs().sqrt())
            * (Complex64::i() * (omega * phi) - Complex64::i() * std::f64::consts::PI * n).exp();
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phases::PointMassField;
    use crate::slice::slice_field;
    use std::f64::consts::PI;

    fn point(x: f64, y: f64, m: f64) -> Component {
        Component { x, y, profile: Profile::Point { mass: m, xc: 0.0 } }
    }

    /// Macro-only (no components): the exact Gaussian, for every macro-image
    /// type including the saddle cases (`lambda < 0`) that a real-box method
    /// cannot represent (P0 finding of record).
    #[test]
    fn macro_only_matches_closed_form() {
        for (kappa, g1, g2) in [(0.3, 0.2, 0.0), (0.5, 0.7, 0.0), (1.5, 0.2, 0.0), (0.45, 0.5, 0.0), (0.1, 0.35, 0.1)] {
            let field = LensField::new([0.4, -0.3], kappa, g1, g2, vec![]).unwrap();
            let omega = 3.0;
            let out = lens_field_solve(&field, &SliceConfig { omega, ..Default::default() }).unwrap();
            let (l0, l2) = (field.lambda0, field.lambda2);
            // Gaussian completions of the two beta-linear terms (the Python
            // referee's Kc = kc - b0^2/(2 l0) - b1^2/(2 l2)).
            let kin = field.kc
                - field.beta[0] * field.beta[0] / (2.0 * l0)
                - field.beta[1] * field.beta[1] / (2.0 * l2);
            let refr = (2.0 * PI / (omega * (l0 * l2).abs().sqrt()))
                * Complex64::from_polar(1.0, (l0.signum() + l2.signum()) * PI / 4.0)
                * (Complex64::i() * (omega * kin)).exp();
            let d = (out.value - refr).norm() / refr.norm();
            assert!(d < 1e-13, "k={kappa} g=({g1},{g2}): {d:.3e}");
        }
    }

    /// At `kappa = gamma = 0` with point components the general field must
    /// coincide with the validated point-mass solver (same contour geometry,
    /// different phase arithmetic — a cross-implementation check).
    #[test]
    fn kappa_gamma_zero_matches_point_mass_field() {
        let lenses = vec![(-0.8, 0.5, 0.6), (0.9, -0.4, 0.4)];
        let beta = [0.2, 0.1];
        let lf = LensField::new(
            beta,
            0.0,
            0.0,
            0.0,
            lenses.iter().map(|&(x, y, m)| point(x, y, m)).collect(),
        )
        .unwrap();
        let pm = PointMassField { beta, lenses };
        for omega in [1.0, 3.0, 10.0] {
            let a = lens_field_solve(&lf, &SliceConfig { omega, ..Default::default() }).unwrap();
            let b = slice_field(&pm, &SliceConfig { omega, ..Default::default() }).unwrap();
            let d = (a.value - b.value).norm() / b.value.norm();
            assert!(d < 1e-12, "omega={omega}: {d:.3e}");
        }
    }

    /// Profile potentials on the real axis against the closed forms
    /// (FIONA/GLoW normalisation).
    #[test]
    fn profile_values_real_axis() {
        let cases: Vec<(Profile, Box<dyn Fn(f64) -> f64>)> = vec![
            (Profile::Point { mass: 0.7, xc: 0.2 }, Box::new(|r| 0.5 * 0.7 * (r * r + 0.04).ln())),
            (Profile::Sis { psi0: 1.3 }, Box::new(|r| 1.3 * r)),
            (
                Profile::Cis { psi0: 1.0, xc: 0.2 },
                Box::new(|r| {
                    let s = (r * r + 0.04).sqrt();
                    s + 0.2 * (2.0 * 0.2 / (s + 0.2)).ln()
                }),
            ),
            (
                Profile::Nfw { psi0: 1.0, xs: 0.5 },
                Box::new(|r| {
                    let u = r / 0.5;
                    let (log2, f2) = if u > 1.0 {
                        let s = (u * u - 1.0).sqrt();
                        ((u / 2.0).ln().powi(2), (s.atan() / s).powi(2))
                    } else if u < 1.0 {
                        let s = (1.0 - u * u).sqrt();
                        ((u / 2.0).ln().powi(2), (s.atanh() / s).powi(2))
                    } else {
                        ((u / 2.0).ln().powi(2), 1.0)
                    };
                    0.5 * (log2 + (u * u - 1.0) * f2)
                }),
            ),
        ];
        for (profile, reference) in cases {
            for r in [0.05f64, 0.3, 0.5, 0.9, 1.0, 1.7, 4.0] {
                let ours = profile.psi(Complex64::from(r * r)).re;
                let want = reference(r);
                let d = (ours - want).abs() / want.abs().max(1e-30);
                assert!(d < 1e-11, "r={r}: ours {ours:e} vs {want:e} (rel {d:e})");
            }
        }
    }

    /// Geometric optics at kappa = gamma = 0 must coincide with the
    /// polynomial (Witt-Mao) point-mass implementation: same images, same
    /// stationary-phase sum.
    #[test]
    fn geometric_matches_point_mass_at_kappa_gamma_zero() {
        let lenses = vec![(-0.8, 0.5, 0.6), (0.9, -0.4, 0.4)];
        let beta = [0.2, 0.1];
        let lf = LensField::new(
            beta,
            0.0,
            0.0,
            0.0,
            lenses.iter().map(|&(x, y, m)| point(x, y, m)).collect(),
        )
        .unwrap();
        let ims = lens_field_images(&lf);
        let poly = crate::images::point_mass_field_images(beta, &lenses);
        assert_eq!(ims.len(), poly.len(), "image count: {ims:?} vs {poly:?}");
        for p in &poly {
            let best = ims
                .iter()
                .map(|q| (q[0] - p[0]).abs() + (q[1] - p[1]).abs())
                .fold(f64::INFINITY, f64::min);
            assert!(best < 1e-6, "image {p:?} not matched (closest {best:e})");
        }
        let omega = 100.0;
        let a = lens_field_geometric(&lf, omega);
        let b = crate::images::geometric_amplification(beta, &lenses, omega);
        assert!((a - b).norm() / b.norm() < 1e-5, "{a} vs {b}");
    }

    /// Macro-only: a single image at the quadratic centre with
    /// `mu = 1/(lambda0 lambda2)`.
    #[test]
    fn geometric_macro_only_one_image() {
        let lf = LensField::new([0.4, -0.3], 0.3, 0.2, 0.0, vec![]).unwrap();
        let ims = lens_field_images(&lf);
        assert_eq!(ims.len(), 1, "{ims:?}");
        let (x0, x2) = (
            lf.beta[0] / lf.lambda0,
            lf.beta[1] / lf.lambda2,
        );
        assert!((ims[0][0] - x0).abs() < 1e-9 && (ims[0][1] - x2).abs() < 1e-9);
        let f = lens_field_geometric(&lf, 1e6);
        let want = (1.0 / (lf.lambda0 * lf.lambda2)).abs().sqrt();
        assert!((f.norm() - want).abs() < 1e-6, "|F| {} vs {want}", f.norm());
    }

    /// The rotation into the shear-diagonal frame: the physical phase at a
    /// point must be invariant under expressing it in either frame.
    #[test]
    fn shear_rotation_identity() {
        let (kappa, g1, g2) = (0.3f64, 0.2f64, 0.1f64);
        let gamma = g1.hypot(g2);
        let th = 0.5 * g2.atan2(g1);
        let beta = [0.4, -0.3];
        let comps = vec![point(0.3, -0.6, 0.5)];
        let lf = LensField::new(beta, kappa, g1, g2, comps).unwrap();
        // physical point -> rotated frame
        let rot = |x: f64, y: f64| {
            (th.cos() * x + th.sin() * y, -th.sin() * x + th.cos() * y)
        };
        for (x0, x2) in [(0.5f64, 0.2f64), (-0.7, 1.1), (1.3, -0.9)] {
            let (z0, z2) = rot(x0, x2);
            let ours = lf.value(&SVector::<Complex64, 2>::new(Complex64::from(z0), Complex64::from(z2)));
            // direct formula in the original frame
            let psi = 0.5 * 0.5 * ((x0 - 0.3) * (x0 - 0.3) + (x2 + 0.6) * (x2 + 0.6)).ln();
            let want = 0.5 * (1.0 - kappa) * (x0 * x0 + x2 * x2)
                - 0.5 * g1 * (x0 * x0 - x2 * x2)
                - g2 * x0 * x2
                - beta[0] * x0
                - beta[1] * x2
                + 0.5 * (beta[0] * beta[0] + beta[1] * beta[1])
                - psi;
            assert!(
                (ours - Complex64::from(want)).norm() < 1e-12,
                "rotation identity: {ours} vs {want}"
            );
            let _ = gamma;
        }
    }
}
