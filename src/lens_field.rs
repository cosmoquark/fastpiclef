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
use crate::slice::{InterruptHook, SliceConfig, SliceEval, SliceFieldOutput, SliceModel, slice_field_g};
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
        // Reject exact and near-critical macro models (including float-rounded
        // critical pairs like kappa = 0.3, gamma1 = 0.7 where 1 - k - g is
        // ~1e-16): the Gaussian centres beta/lambda and the tail length scale
        // like 1/|lambda| and degenerate as lambda -> 0.
        const CRITICAL_LAMBDA_FLOOR: f64 = 1e-8;
        if lambda0.abs() < CRITICAL_LAMBDA_FLOOR || lambda2.abs() < CRITICAL_LAMBDA_FLOOR {
            return Err(Error::InvalidInput(
                "critical or near-critical macro model: |lambda0| and |lambda2| must be >= 1e-8",
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

    fn free_integral(&self, omega: f64) -> (Complex64, f64) {
        // The macro-only closed form (`referee.py::macro_only_integral`):
        //   I = (2 pi / (omega sqrt|L0 L2|)) e^{i (sgn L0 + sgn L2) pi/4}
        //       e^{i omega Kc},   Kc = 1/2|beta|^2 - b0^2/(2 L0) - b1^2/(2 L2),
        // with b the source in the shear-diagonal frame.  The stored `kc` is
        // the 1/2|beta|^2 term; the rest completes each quadratic's square.
        let (l0, l2) = (self.lambda0, self.lambda2);
        let kc = self.kc
            - self.beta[0] * self.beta[0] / (2.0 * l0)
            - self.beta[1] * self.beta[1] / (2.0 * l2);
        let mag = std::f64::consts::TAU / (omega * (l0 * l2).abs().sqrt());
        let phase = (l0.signum() + l2.signum()) * std::f64::consts::FRAC_PI_4 + omega * kc;
        let v = Complex64::from_polar(mag, phase);
        (v, 8.0 * f64::EPSILON * v.norm() * (1.0 + omega * kc.abs()))
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
    /// The free/macro quadratic part `1/2 L0 z0^2 - b0 z0 + x2part` on the
    /// real core; `phi_core_real` subtracts the profile potentials from it.
    #[inline]
    fn phi_free_core_real(&self, r: f64, x0: f64, sgn: f64) -> f64 {
        self.quad_real(x0 + sgn * r) + self.x2_part_real()
    }

    #[inline]
    fn phi_free_core_complex(&self, r: f64, x0: f64, sgn: f64) -> Complex64 {
        let z0 = Complex64::from(x0 + sgn * r);
        0.5 * Complex64::from(self.field.lambda0) * z0 * z0
            - Complex64::from(self.field.beta[0]) * z0
            + self.x2_part_complex()
    }

    #[inline]
    fn phi_free_tail(&self, z: Complex64) -> Complex64 {
        0.5 * Complex64::from(self.field.lambda0) * z * z
            - Complex64::from(self.field.beta[0]) * z
            + self.x2_part_complex()
    }

    #[inline]
    fn phi_core_real(&self, r: f64, x0: f64, sgn: f64) -> f64 {
        let mut v = self.phi_free_core_real(r, x0, sgn);
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
        let mut v = self.phi_free_core_complex(r, x0, sgn);
        for (k, c) in self.field.components.iter().enumerate() {
            let t = z0 - c.x;
            let w = t * t + self.dk[k] * self.dk[k];
            v -= c.profile.psi(w);
        }
        v
    }

    #[inline]
    fn phi_tail(&self, z: Complex64) -> Complex64 {
        let mut v = self.phi_free_tail(z);
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

    /// `d^2 psi / d w^2` (principal branch).  For the analytic profiles the
    /// closed form is used (a generic complex step of [`Self::dpsi_dw`] loses
    /// the *imaginary* part to the `eps/|psi'| / h` cancellation, which the
    /// nested stationary-phase tensors would amplify); the closed forms are
    /// pinned against that generic step by a unit test at real `w`.  NFW has
    /// no closed form here, so it keeps the complex step — its contribution to
    /// the `O(1/omega)` correction stays far below the correction itself.
    pub fn dpsi_dw2(&self, w: Complex64) -> Complex64 {
        match *self {
            Profile::Point { mass, xc } => {
                let d = w + Complex64::from(xc * xc);
                Complex64::new(-0.5 * mass, 0.0) / (d * d)
            }
            Profile::Sis { psi0 } => Complex64::new(-0.25 * psi0, 0.0) / (w * w.sqrt()),
            Profile::Cis { psi0, xc } => {
                let u = (w + Complex64::from(xc * xc)).sqrt();
                let x = Complex64::from(xc);
                let t = x * (Complex64::new(2.0, 0.0) * u + x) / ((u + x) * (u + x));
                Complex64::new(psi0, 0.0) / (Complex64::new(4.0, 0.0) * u * u * u)
                    * (t - Complex64::new(1.0, 0.0))
            }
            Profile::Nfw { .. } => {
                let h = 1e-5 * w.norm();
                (self.dpsi_dw(w + Complex64::i() * h) - self.dpsi_dw(w - Complex64::i() * h))
                    / (Complex64::i() * (2.0 * h))
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

/// Real-space gradient and Hessian of the phase at a real point `x` in a
/// single pass over `field.components` (the chain rule with `psi'` and
/// `psi''`; matches [`hess_analytic`] at real `z` and the complex-step
/// Hessian to machine precision), skipping a set of component indices — the
/// local *smooth part* at a cusp centre, used by [`cusp_groups`].
fn grad_hess_excluding(
    field: &LensField,
    x: [f64; 2],
    skip: &[usize],
) -> ([f64; 2], [[f64; 2]; 2]) {
    let mut g = [
        field.lambda0 * x[0] - field.beta[0],
        field.lambda2 * x[1] - field.beta[1],
    ];
    let mut h = [[field.lambda0, 0.0], [0.0, field.lambda2]];
    for (k, c) in field.components.iter().enumerate() {
        if skip.contains(&k) {
            continue;
        }
        let d = [x[0] - c.x, x[1] - c.y];
        let wc = Complex64::from(d[0] * d[0] + d[1] * d[1]);
        let p1 = c.profile.dpsi_dw(wc).re;
        let p2 = c.profile.dpsi_dw2(wc).re;
        g[0] -= 2.0 * p1 * d[0];
        g[1] -= 2.0 * p1 * d[1];
        for a in 0..2 {
            for b in 0..2 {
                h[a][b] -= 4.0 * p2 * d[a] * d[b];
                if a == b {
                    h[a][b] -= 2.0 * p1;
                }
            }
        }
    }
    (g, h)
}

/// [`grad_hess_excluding`] over all components.
fn real_grad_and_hess(field: &LensField, x: [f64; 2]) -> ([f64; 2], [[f64; 2]; 2]) {
    grad_hess_excluding(field, x, &[])
}

/// Smallest `|eigenvalue|` of a symmetric `2x2` matrix.
fn min_abs_eigenvalue(h: [[f64; 2]; 2]) -> f64 {
    let (a, b) = eig_range(h);
    a.abs().min(b.abs())
}

/// Eigenvalue range `(min, max)` of a symmetric `2x2` matrix.
fn eig_range(h: [[f64; 2]; 2]) -> (f64, f64) {
    let tr = h[0][0] + h[1][1];
    let det = h[0][0] * h[1][1] - h[0][1] * h[1][0];
    let disc = (tr * tr - 4.0 * det).max(0.0).sqrt();
    ((tr - disc) / 2.0, (tr + disc) / 2.0)
}

/// Near-singular local smooth Hessian floor: below it the local cusp model is
/// near-critical, the companion scale is not certified, and the per-group
/// companion check is skipped (the index-identity term still applies).
const LAMBDA_FLOOR: f64 = 1e-8;

/// One *effective cusp* of a lens field: a group of coincident components
/// whose deflection winds around their common centre.  The completeness
/// certificate of [`lens_field_images`] uses the group's winding `W_j` in the
/// index identity and its local scale for the companion check; the finder
/// seeds its Newton ladders from the group.
struct CuspGroup {
    /// Group centre.
    x: f64,
    y: f64,
    /// Signed weight: the summed mass of the group's unsoftened points
    /// (point-dominated) or the summed `psi0` of its SIS-type cusps.  The
    /// sign does not affect the winding (`W_j` is `+1` per active group) — it
    /// only gates the companion check (repulsive groups are skipped).
    weight: f64,
    /// The group's points dominate at the centre (`1/|d|` against a bounded
    /// deflection).
    is_point: bool,
    /// `W_j`, the winding of `grad phi` around the centre (counterclockwise):
    /// `1` for an active group, `0` otherwise.  The winding is `+1` whatever
    /// the weight's sign — the deflection is anti-holomorphic (`~ 1/d̄`) and a
    /// negative weight is a `pi`-rotation, which leaves the winding unchanged.
    /// A group is active when its dominant class is a point (always) or an
    /// SIS-type cusp with `|g_s| < |weight|` (the source inside the cusp's
    /// Einstein ring).  The index identity subtracts `W_j` (see
    /// [`index_identity_rhs`]).
    winding: i64,
    /// Smooth gradient at the centre (macro + non-singular components): the
    /// weak-regime companion direction and scale.
    g_s: [f64; 2],
    /// Smallest `|eigenvalue|` of the smooth Hessian at the centre (the
    /// strong-regime seed scale).
    lambda_loc: f64,
    /// Local Einstein scale: `sqrt(|weight|/lambda_loc)` for point-dominated
    /// groups, `|weight|/lambda_loc` for SIS-type groups.
    scale: f64,
    /// Companion-check radius (`2 * r + 0.1`, see [`cusp_groups`]); `0` = no
    /// check (inactive winding, a repulsive group, no positive eigenvalue, or
    /// a local Hessian below [`LAMBDA_FLOOR`]).
    companion_radius: f64,
}

/// The field's cusp groups, in component order.  Components with zero weight
/// are invisible to `grad phi` and skipped; cored profiles (CIS with
/// `xc > 0`, NFW, softened points) are smooth at their centre and contribute
/// no winding.  Coincident cusps form one group whose most singular class
/// dominates: points (summed masses) over SIS-type cusps (summed `psi0`); an
/// exact cancellation leaves a smooth centre.
fn cusp_groups(field: &LensField) -> Vec<CuspGroup> {
    /// Coincident centres within this L1 distance form one group.
    const COINCIDENT_TOL: f64 = 1e-9;
    let cusp = |c: &Component| -> Option<(f64, bool)> {
        match c.profile {
            Profile::Point { mass, xc } if xc == 0.0 && mass != 0.0 => Some((mass, true)),
            Profile::Sis { psi0 } if psi0 != 0.0 => Some((psi0, false)),
            Profile::Cis { psi0, xc } if xc == 0.0 && psi0 != 0.0 => Some((psi0, false)),
            _ => None,
        }
    };
    let mut groups = Vec::new();
    let mut assigned = vec![false; field.components.len()];
    for j in 0..field.components.len() {
        if assigned[j] {
            continue;
        }
        let cj = &field.components[j];
        let Some((w0, p0)) = cusp(cj) else { continue };
        assigned[j] = true;
        let mut members = vec![j];
        let mut weights = vec![(w0, p0)];
        for (k, ck) in field.components.iter().enumerate().skip(j + 1) {
            if assigned[k] {
                continue;
            }
            if (ck.x - cj.x).abs() + (ck.y - cj.y).abs() <= COINCIDENT_TOL
                && let Some((w, p)) = cusp(ck)
            {
                members.push(k);
                weights.push((w, p));
                assigned[k] = true;
            }
        }
        let m_sum: f64 = weights.iter().filter(|(_, p)| *p).map(|(w, _)| w).sum();
        let p_sum: f64 = weights.iter().filter(|(_, p)| !*p).map(|(w, _)| w).sum();
        let (weight, is_point) = if m_sum != 0.0 {
            (m_sum, true)
        } else if p_sum != 0.0 {
            (p_sum, false)
        } else {
            continue; // exact cancellation: a smooth centre
        };
        let (g_s, h_s) = grad_hess_excluding(field, [cj.x, cj.y], &members);
        let (lam_min, lam_max) = eig_range(h_s);
        let lambda_loc = lam_min.abs().min(lam_max.abs());
        let lambda_pos_min = if lam_min > 0.0 {
            lam_min
        } else if lam_max > 0.0 {
            lam_max
        } else {
            0.0
        };
        let winding = if is_point || g_s[0].hypot(g_s[1]) < weight.abs() {
            1
        } else {
            0
        };
        let lam = lambda_loc.max(LAMBDA_FLOOR);
        let scale = if is_point {
            (weight.abs() / lam).sqrt()
        } else {
            weight.abs() / lam
        };
        // Companion radius.  Along a positive-curvature direction the local
        // companion sits within `sqrt(|w|/lambda_pos)` (the Einstein scale;
        // the weak regime is smaller), and along a negative-curvature
        // direction only the weak regime admits a solution, within
        // `0.5 sqrt(|w|/|lambda|)`.  A negative semi-definite local model has
        // no pinned companion at all (e.g. a star under a strongly defocusing
        // macro can have no images anywhere: its winding is absorbed by the
        // `1/|d|^2` pole).  A *repulsive* group (`weight < 0`) is skipped
        // too: its companions (where they exist at all, `|g_s| > 2
        // sqrt(lambda |w|)`) lie along `-g_s` at `|g_s|/lambda`, and the
        // ladders already seed every direction.
        let companion_radius =
            if winding != 0 && weight > 0.0 && lambda_pos_min > 0.0 && lambda_loc >= LAMBDA_FLOOR {
                let r = if is_point {
                    (weight / lambda_pos_min)
                        .sqrt()
                        .max(0.5 * (weight / lam).sqrt())
                } else {
                    weight / lambda_pos_min
                };
                2.0 * r + 0.1
            } else {
                0.0
            };
        groups.push(CuspGroup {
            x: cj.x,
            y: cj.y,
            weight,
            is_point,
            winding,
            g_s,
            lambda_loc,
            scale,
            companion_radius,
        });
    }
    groups
}

/// Index-identity right-hand side: `sign(lambda0 lambda2) - sum_j W_j` over
/// the field's cusp groups, i.e. `sign(lambda0 lambda2) - #(active groups)`
/// (`W_j` is `1` for each active group whatever the weight's sign; see
/// [`CuspGroup::winding`]).  For a generic source the sum of the Hessian
/// signs over all images equals this (excision on the sphere: the large-circle
/// winding of `grad phi` is `sign(lambda0 lambda2)`, and each cusp group
/// contributes its own winding `W_j`; smooth profiles wind trivially).  A
/// missed image *pair* that leaves the sum unchanged is caught by the
/// per-group companion check instead.
fn index_identity_rhs(field: &LensField, groups: &[CuspGroup]) -> i64 {
    let mut rhs = if field.lambda0 * field.lambda2 > 0.0 { 1 } else { -1 };
    for g in groups {
        rhs -= g.winding;
    }
    rhs
}

/// Real-space geometry of a [`LensField`]: the images (solutions of the lens
/// equation `grad phi = 0`) found by multi-start Newton, and the geometric
/// (stationary-phase) amplification.  Each Newton step evaluates the gradient
/// and Hessian in one analytic pass ([`real_grad_and_hess`], pinned against
/// complex-step derivatives in the unit tests), so the κ/γ + profile cases
/// reuse the exact potential of the integral solver.
///
/// The finder is completeness-certified, with the same contract as the
/// point-mass-only finder ([`crate::images::point_mass_field_images`]):
///
/// 1. Seeds: a 24x24 golden-ratio-jittered global grid over the component and
///    macro-image extent (an *aligned* grid silently missed the faint image
///    of the unit point mass at y = 1 — its Newton basin lay between the
///    starts and the start at -1 sat exactly on the critical curve), the macro
///    image centre, and Newton ladders per cusp group (the weak-field
///    companion scale `|w|/|g_s|` and the local Einstein scale; see
///    [`CuspGroup`]) and per profile (its deflection scale).
/// 2. Every candidate is residual-verified (`|grad phi|^2 <= 1e-16 (1+|x|^2)`)
///    and deduplicated.
/// 3. Certificate 1: non-degeneracy (`|det H|` at or above the caustic floor
///    `1e-8`; at an exact caustic the image set is not discrete — e.g. the
///    whole Einstein ring at `beta = 0` — and the stationary-phase sum is
///    undefined) and the index identity ([`index_identity_rhs`]).
/// 4. Certificate 2: every cusp group with an active winding and a positive
///    local smooth-Hessian eigenvalue owns an image within its local scale
///    ([`CuspGroup::companion_radius`]) — this catches a missed image *pair*,
///    which the index sum alone cannot see.  A locally defocusing cusp (no
///    positive eigenvalue) has no pinned companion scale; the index identity
///    still applies, and an empty image set is legal exactly when the
///    identity allows it (see the note at the end of the search).
///
/// A violated certificate is an error, never a silently short list.
pub fn lens_field_images(field: &LensField) -> Result<Vec<[f64; 2]>> {
    let groups = cusp_groups(field);
    // Start-grid extent: component centres, the macro image position, and the
    // cusp image scales.
    let mut span = 2.0f64;
    for c in &field.components {
        span = span.max(c.x.abs()).max(c.y.abs());
    }
    span = span
        .max((field.beta[0] / field.lambda0).abs())
        .max((field.beta[1] / field.lambda2).abs());
    for g in &groups {
        span = span.max(g.x.abs() + g.scale).max(g.y.abs() + g.scale);
    }
    let span = span + 2.0;
    // (a) Global grid, deterministically golden-ratio jittered so no start
    // sits exactly on a critical curve or a singular centre.
    const GRID: usize = 24;
    let mut seeds: Vec<[f64; 2]> = Vec::new();
    for i in 0..=GRID {
        for j in 0..=GRID {
            let k = i * (GRID + 1) + j;
            let jit = |s: usize| ((s as f64 + 1.0) * 0.618_033_988_749_895).fract() - 0.5;
            seeds.push([
                -span + 2.0 * span * (i as f64 + 0.6 * jit(k)) / GRID as f64,
                -span + 2.0 * span * (j as f64 + 0.6 * jit(k + 7)) / GRID as f64,
            ]);
        }
    }
    // (b) The macro image centre.
    seeds.push([field.beta[0] / field.lambda0, field.beta[1] / field.lambda2]);
    // (c) Newton ladders.  Angle 0 of each ring points along the local smooth
    // gradient (the weak-field companion direction).
    const RADII: [f64; 6] = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0];
    let mut ring = |cx: f64, cy: f64, r: f64, dir: [f64; 2]| {
        for k in 0..8 {
            let (s, c) = (k as f64 * std::f64::consts::PI / 4.0).sin_cos();
            seeds.push([
                cx + r * (c * dir[0] - s * dir[1]),
                cy + r * (s * dir[0] + c * dir[1]),
            ]);
        }
    };
    let direction = |v: [f64; 2]| {
        let n = v[0].hypot(v[1]);
        if n > 1e-12 { [v[0] / n, v[1] / n] } else { [1.0, 0.0] }
    };
    for g in &groups {
        let dir = direction(g.g_s);
        let lam = g.lambda_loc.max(LAMBDA_FLOOR);
        let weak = if g.is_point {
            g.weight.abs() / g.g_s[0].hypot(g.g_s[1]).max((lam * g.weight.abs()).sqrt())
        } else {
            g.scale
        };
        for base in [weak, g.scale] {
            if base.is_finite() && base > 0.0 {
                for s in RADII {
                    ring(g.x, g.y, base * s, dir);
                }
            }
        }
    }
    for (k, c) in field.components.iter().enumerate() {
        if c.profile.is_zero() {
            continue;
        }
        // Deflection scale of a smooth profile (cusp components are covered by
        // the group ladders above).
        let (g_s, h_s) = {
            let mut skip = vec![k];
            skip.extend((0..field.components.len()).filter(|&q| {
                q != k
                    && (field.components[q].x - c.x).abs()
                        + (field.components[q].y - c.y).abs()
                        <= 1e-9
            }));
            grad_hess_excluding(field, [c.x, c.y], &skip)
        };
        let lam = min_abs_eigenvalue(h_s).max(LAMBDA_FLOOR);
        let base = match c.profile {
            Profile::Point { mass, xc } if xc > 0.0 => mass.abs() / (xc * lam),
            Profile::Point { .. } | Profile::Sis { .. } => continue, // cusp
            Profile::Cis { xc: 0.0, .. } => continue,                // cusp (== SIS)
            Profile::Cis { psi0, .. } => psi0.abs() / lam,
            Profile::Nfw { psi0, xs } => (psi0.abs() / lam).max(xs),
        };
        let dir = direction(g_s);
        if base.is_finite() && base > 0.0 {
            for s in RADII {
                ring(c.x, c.y, base * s, dir);
            }
        }
    }
    // Newton on the real lens equation, residual-verified, deduplicated.
    let mut images: Vec<[f64; 2]> = Vec::new();
    for seed in seeds {
        let mut x = seed;
        let mut ok = false;
        for _ in 0..60 {
            let (gx, jx) = real_grad_and_hess(field, x);
            let det = jx[0][0] * jx[1][1] - jx[0][1] * jx[1][0];
            if !det.is_finite() || det.abs() < 1e-14 {
                break;
            }
            let dx = [
                (jx[1][1] * gx[0] - jx[0][1] * gx[1]) / det,
                (-jx[1][0] * gx[0] + jx[0][0] * gx[1]) / det,
            ];
            x[0] -= dx[0];
            x[1] -= dx[1];
            if !(x[0].is_finite() && x[1].is_finite()) {
                break;
            }
            if dx[0].abs() + dx[1].abs() < 1e-13 {
                ok = true;
                break;
            }
        }
        // Verify: a small Newton step is not enough near a degenerate
        // Jacobian; the point must actually solve the lens equation.
        if ok {
            let (gx, _) = real_grad_and_hess(field, x);
            if gx[0] * gx[0] + gx[1] * gx[1] > 1e-16 * (1.0 + x[0] * x[0] + x[1] * x[1]) {
                ok = false;
            }
        }
        if ok && !images.iter().any(|p| (p[0] - x[0]).abs() + (p[1] - x[1]).abs() < 1e-7) {
            images.push(x);
        }
    }
    // No hard empty-set error: a strongly defocusing macro with a point
    // component can have *no* images at all (the cusp's winding is absorbed
    // by its `1/|d|^2` pole; e.g. kappa = 1.5, gamma = 0.2, a unit point at
    // the centre), and an empty set is then the correct answer.  The
    // certificate below still rejects a wrong empty set: the index identity
    // fails unless `rhs == 0`, and every winding-active cusp with a positive
    // local eigenvalue must own a nearby image.
    // Certificate 1: non-degeneracy and the index identity.
    const CAUSTIC_DET: f64 = 1e-8;
    let mut n_pos = 0i64;
    let mut n_neg = 0i64;
    for x in &images {
        let (_, h) = real_grad_and_hess(field, *x);
        let det = h[0][0] * h[1][1] - h[0][1] * h[1][0];
        if !det.is_finite() || det.abs() < CAUSTIC_DET {
            return Err(Error::DegenerateImage);
        }
        if det > 0.0 {
            n_pos += 1;
        } else {
            n_neg += 1;
        }
    }
    let rhs = index_identity_rhs(field, &groups);
    if n_pos - n_neg != rhs {
        return Err(Error::ImageSearch(
            "index identity violated (an image was likely missed)",
        ));
    }
    // Certificate 2: every winding-active cusp group owns a nearby image.
    for g in &groups {
        if g.companion_radius > 0.0
            && !images
                .iter()
                .any(|x| (x[0] - g.x).hypot(x[1] - g.y) <= g.companion_radius)
        {
            return Err(Error::ImageSearch(
                "a cusp component has no image nearby (a companion image was likely missed)",
            ));
        }
    }
    Ok(images)
}

/// Stationary-phase (geometric-optics) amplification of a [`LensField`] —
/// exact as `omega -> infinity` away from caustics; O(#images).
///
/// Images come from the completeness-certified [`lens_field_images`]; a
/// degenerate image (an exact caustic) is reported as
/// [`Error::DegenerateImage`], never silently dropped.  For point-mass-only
/// fields [`crate::images::geometric_amplification`] is equivalent.
pub fn lens_field_geometric(field: &LensField, omega: f64) -> Result<Complex64> {
    let mut f = Complex64::new(0.0, 0.0);
    for x in lens_field_images(field)? {
        let (_, hess) = real_grad_and_hess(field, x);
        let det = hess[0][0] * hess[1][1] - hess[0][1] * hess[1][0];
        let tr = hess[0][0] + hess[1][1];
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
    Ok(f)
}

/// Analytic continuation of the phase Hessian `d_a d_b phi` as a function of
/// the complex point `z` (the chain rule with `psi'` and `psi''`).  At real
/// arguments it reproduces [`hess_cs`] to machine precision (unit-tested); it
/// exists to build the third/fourth derivatives of the stationary-phase
/// correction by *central complex steps*, which do not amplify roundoff the
/// way nested one-sided complex steps would.
fn hess_analytic(field: &LensField, z: [Complex64; 2]) -> [[Complex64; 2]; 2] {
    let mut h = [
        [Complex64::from(field.lambda0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::from(field.lambda2)],
    ];
    for c in &field.components {
        let d = [z[0] - Complex64::from(c.x), z[1] - Complex64::from(c.y)];
        let w = d[0] * d[0] + d[1] * d[1];
        let p1 = c.profile.dpsi_dw(w);
        let p2 = c.profile.dpsi_dw2(w);
        for a in 0..2 {
            for b in 0..2 {
                h[a][b] -= 4.0 * p2 * d[a] * d[b];
                if a == b {
                    h[a][b] -= 2.0 * p1;
                }
            }
        }
    }
    h
}

/// Third-derivative tensor `d_c d_b d_a phi` at a real point: a central
/// complex step of the analytic Hessian.
fn phi3_cs(field: &LensField, x: [f64; 2], h: f64) -> [[[f64; 2]; 2]; 2] {
    let mut out = [[[0.0f64; 2]; 2]; 2];
    for c in 0..2 {
        let mut zp = [Complex64::from(x[0]), Complex64::from(x[1])];
        let mut zm = zp;
        zp[c] += Complex64::i() * h;
        zm[c] -= Complex64::i() * h;
        let hp = hess_analytic(field, zp);
        let hm = hess_analytic(field, zm);
        for a in 0..2 {
            for b in 0..2 {
                out[a][b][c] = ((hp[a][b] - hm[a][b]) / (Complex64::i() * (2.0 * h))).re;
            }
        }
    }
    out
}

/// Fourth-derivative tensor `d_d d_c d_b d_a phi` at a real point: the
/// two-dimensional central stencil on the analytic Hessian.
fn phi4_cs(field: &LensField, x: [f64; 2], h: f64) -> [[[[f64; 2]; 2]; 2]; 2] {
    let base = [Complex64::from(x[0]), Complex64::from(x[1])];
    let shift = |c: usize, d: usize, sc: f64, sd: f64| -> [Complex64; 2] {
        let mut z = base;
        z[c] += Complex64::i() * (sc * h);
        if d != c {
            z[d] += Complex64::i() * (sd * h);
        }
        z
    };
    let mut out = [[[[0.0f64; 2]; 2]; 2]; 2];
    for c in 0..2 {
        for d in 0..2 {
            let mut out_cd = [[0.0f64; 2]; 2];
            if c == d {
                let hp = hess_analytic(field, shift(c, d, 2.0, 0.0));
                let hm = hess_analytic(field, shift(c, d, -2.0, 0.0));
                let h0 = hess_analytic(field, base);
                for a in 0..2 {
                    for b in 0..2 {
                        out_cd[a][b] = ((hp[a][b] - 2.0 * h0[a][b] + hm[a][b])
                            / (Complex64::i() * (2.0 * h)).powi(2))
                        .re;
                    }
                }
            } else {
                let hpp = hess_analytic(field, shift(c, d, 1.0, 1.0));
                let hpm = hess_analytic(field, shift(c, d, 1.0, -1.0));
                let hmp = hess_analytic(field, shift(c, d, -1.0, 1.0));
                let hmm = hess_analytic(field, shift(c, d, -1.0, -1.0));
                for a in 0..2 {
                    for b in 0..2 {
                        out_cd[a][b] = ((hpp[a][b] - hpm[a][b] - hmp[a][b] + hmm[a][b])
                            / (Complex64::i() * (2.0 * h)).powi(2))
                        .re;
                    }
                }
            }
            for (a, row_out) in out.iter_mut().enumerate() {
                for (b, slot) in row_out.iter_mut().enumerate() {
                    slot[c][d] = out_cd[a][b];
                }
            }
        }
    }
    out
}

/// The 15-pairing sum `Sum_15 = sum phi3_abc phi3_def H^-1 H^-1 H^-1` of the
/// Isserlis expansion (the `phi3^2` contribution to the `O(1/omega)`
/// coefficient), by direct enumeration: 15 matchings x 2^6 index choices.
fn sum15(phi3: &[[[f64; 2]; 2]; 2], hinv: &[[f64; 2]; 2]) -> f64 {
    // Slot 0 is paired with each of 1..5; the remaining four slots are then
    // matched in their three possible ways.
    const MATCHINGS: [[(usize, usize); 3]; 15] = [
        [(0, 1), (2, 3), (4, 5)],
        [(0, 1), (2, 4), (3, 5)],
        [(0, 1), (2, 5), (3, 4)],
        [(0, 2), (1, 3), (4, 5)],
        [(0, 2), (1, 4), (3, 5)],
        [(0, 2), (1, 5), (3, 4)],
        [(0, 3), (1, 2), (4, 5)],
        [(0, 3), (1, 4), (2, 5)],
        [(0, 3), (1, 5), (2, 4)],
        [(0, 4), (1, 2), (3, 5)],
        [(0, 4), (1, 3), (2, 5)],
        [(0, 4), (1, 5), (2, 3)],
        [(0, 5), (1, 2), (3, 4)],
        [(0, 5), (1, 3), (2, 4)],
        [(0, 5), (1, 4), (2, 3)],
    ];
    let mut total = 0.0;
    for m in MATCHINGS {
        for bits in 0..64usize {
            let i = [
                bits & 1,
                (bits >> 1) & 1,
                (bits >> 2) & 1,
                (bits >> 3) & 1,
                (bits >> 4) & 1,
                (bits >> 5) & 1,
            ];
            let mut t = phi3[i[0]][i[1]][i[2]] * phi3[i[3]][i[4]][i[5]];
            for &(p, q) in &m {
                t *= hinv[i[p]][i[q]];
            }
            total += t;
        }
    }
    total
}

/// Per-image stationary-phase data for the P4.1 correction.
struct SpImageData {
    /// The next-order coefficient `C`.
    c: f64,
    /// `det H` (its sign carries the Morse index).
    det: f64,
    /// `tr H`.
    tr: f64,
}

/// `C = (1/8) H^-1_ab H^-1_cd phi4_abcd - (1/72) Sum_15(phi3, H^-1)` at one
/// image (derived and validated in `comparison/sp_correction.py`); `None` at a
/// numerically singular Hessian (caustic), where the asymptotic form breaks
/// down and the caller must fall back to the exact RC engine.
fn sp_image_data(field: &LensField, x: [f64; 2]) -> Option<SpImageData> {
    let (_, hess) = real_grad_and_hess(field, x);
    let det = hess[0][0] * hess[1][1] - hess[0][1] * hess[1][0];
    if !det.is_finite() || det.abs() < 1e-12 {
        return None;
    }
    let tr = hess[0][0] + hess[1][1];
    let hinv = [
        [hess[1][1] / det, -hess[0][1] / det],
        [-hess[1][0] / det, hess[0][0] / det],
    ];
    let p3 = phi3_cs(field, x, 1e-5);
    let p4 = phi4_cs(field, x, 3e-5);
    let mut c4 = 0.0;
    for a in 0..2 {
        for b in 0..2 {
            for c in 0..2 {
                for d in 0..2 {
                    c4 += hinv[a][b] * hinv[c][d] * p4[a][b][c][d];
                }
            }
        }
    }
    Some(SpImageData {
        c: 0.125 * c4 - sum15(&p3, &hinv) / 72.0,
        det,
        tr,
    })
}

/// The corrected stationary-phase sum over a given image list,
/// `sum_j G0_j (1 + C_j/(i omega))`, plus the error estimate
/// `3 * sum_j |G0_j| |C_j| / omega`; `None` at a caustic image or with no
/// images.
///
/// The raw sum `sum_j |G0_j| |C_j| / omega` was measured (2026-10-12,
/// `comparison/hybrid_check.py` headless + the prototype `sp_correction.py`)
/// to cover the corrected residual by >= 3.5x over the point-mass family, the
/// 3-mass field and macro cases; the factor 3 is folded in so the *reported*
/// budget keeps a >= 10x margin on the measured minimum (error estimates must
/// cover the true error, per the project's validation discipline).
fn sp_corrected_sum(
    field: &LensField,
    images: &[[f64; 2]],
    omega: f64,
) -> Option<(Complex64, f64)> {
    const SAFETY: f64 = 3.0;
    if images.is_empty() {
        return None;
    }
    let mut f = Complex64::new(0.0, 0.0);
    let mut est = 0.0f64;
    for &x in images {
        let data = sp_image_data(field, x)?;
        let n = if data.det < 0.0 {
            0.5
        } else if data.tr > 0.0 {
            0.0
        } else {
            1.0
        };
        let xz = SVector::<Complex64, 2>::new(Complex64::from(x[0]), Complex64::from(x[1]));
        let phi = field.value(&xz).re;
        let g0 = Complex64::from(1.0 / data.det.abs().sqrt())
            * (Complex64::i() * (omega * phi) - Complex64::i() * std::f64::consts::PI * n).exp();
        f += g0 * (Complex64::new(1.0, 0.0) - Complex64::i() * (data.c / omega));
        est += g0.norm() * data.c.abs() / omega;
    }
    Some((f, SAFETY * est))
}

/// Result of [`lens_field_geometric_corrected`].
#[derive(Debug, Clone, Copy)]
pub struct GeometricCorrected {
    /// Corrected stationary-phase amplification (same convention as
    /// [`lens_field_geometric`]).
    pub value: Complex64,
    /// `3 * sum_j |G0_j| |C_j| / omega`: the asymptotic error estimate with
    /// the documented safety factor of [`sp_corrected_sum`].  The *raw* sum
    /// covered the corrected residual by >= 3.5x in every measured family
    /// (point masses, the 3-mass field, macro cases;
    /// `comparison/results/hybrid_check.csv`), so this budget is conservative.
    pub error: f64,
    /// Number of images used.
    pub images: usize,
}

/// Geometric-optics amplification **with the stationary-phase next-order
/// correction** (P4.1).  Expanding `int d^2u e^{i omega phi(u)}` to quartic
/// order around each stationary point gives `G0 [1 + C/(i omega) +
/// O(omega^-2)]`, with `G0` the geometric term and `C` the coefficient of
/// [`sp_image_data`]; the corrected residual falls like `omega^-2` instead of
/// `omega^-1` (validated on the Airy integral and on the point mass against
/// the Nakamura-Deguchi closed form).
///
/// `Ok(None)` when the asymptotic form cannot be trusted (a caustic, i.e.
/// [`Error::DegenerateImage`]); any other finder error (a violated
/// completeness certificate) propagates — it is a bug, not a caustic.  The
/// caller falls back to the exact RC engine.
pub fn lens_field_geometric_corrected(
    field: &LensField,
    omega: f64,
) -> Result<Option<GeometricCorrected>> {
    let images = match lens_field_images(field) {
        Ok(images) => images,
        // A caustic: `None` is this function's documented "asymptotic form
        // invalid" answer (mirrors `point_mass_field_geometric_corrected`).
        Err(Error::DegenerateImage) => return Ok(None),
        Err(e) => return Err(e),
    };
    let n = images.len();
    Ok(sp_corrected_sum(field, &images, omega).map(|(value, error)| GeometricCorrected {
        value,
        error,
        images: n,
    }))
}

/// Which engine a [`lens_field_hybrid`] call used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridMethod {
    /// Geometric optics + the stationary-phase next-order correction (P4.1).
    GeometricCorrected,
    /// The exact RC slice engine (below the threshold, or at a caustic).
    Rc,
}

/// Result of [`lens_field_hybrid`] / [`point_mass_field_hybrid`].
#[derive(Debug, Clone, Copy)]
pub struct HybridOutput {
    /// Raw integral in the same convention as [`lens_field_solve`].
    pub value: Complex64,
    /// Error estimate: the asymptotic estimate on the geometric branch, the
    /// RC budget on the fallback.
    pub error: f64,
    pub method: HybridMethod,
    /// Images used on the geometric branch (0 on the fallback).
    pub images: usize,
    /// Inner RC solves on the fallback (0 on the geometric branch).
    pub inner_solves: usize,
}

/// 2D free-space integral of the bare quadratic phase, `2 pi i / omega` — the
/// factor between the stationary-phase *amplification* ([`lens_field_geometric`])
/// and the raw integral ([`lens_field_solve`]).
fn free_space_2d(omega: f64) -> Complex64 {
    Complex64::from_polar(
        2.0 * std::f64::consts::PI / omega,
        std::f64::consts::FRAC_PI_2,
    )
}

/// High-omega hybrid (P4.1): geometric optics with the validated
/// stationary-phase correction when its conservative error estimate is within
/// `tol`, and the exact RC engine otherwise.  The switch is honest: the raw
/// estimate covered the corrected residual by >= 3.5x in every measured family
/// and the reported budget carries a further factor 3, so `tol` bounds the
/// true error on the geometric branch (validated against the exact RC and the
/// external ND closed form in `tests/hybrid.rs` and
/// `comparison/hybrid_check.py`).
///
/// `value`/`error` are the raw integral and its budget, in the same convention
/// as [`lens_field_solve`] (the geometric branch's amplification estimate is
/// scaled by the free-space factor `2 pi / omega`); `tol` bounds the returned
/// integral error, which is the same *relative* bound on the amplification.
/// `interrupt` is passed to the exact RC engine on the fallback branch (the
/// geometric branch is already microseconds); see [`InterruptHook`].
pub fn lens_field_hybrid(
    field: &LensField,
    omega: f64,
    tol: f64,
    interrupt: Option<InterruptHook>,
) -> Result<HybridOutput> {
    let fs = free_space_2d(omega);
    // The geometric branch needs a trustworthy image set.  A degenerate
    // (caustic) or certificate-failed set falls through to the exact RC
    // engine below — caustics are its home turf — so any finder error maps to
    // the empty set (= no geometric branch) rather than propagating (mirrors
    // `point_mass_field_hybrid`).
    if let Ok(Some(g)) = lens_field_geometric_corrected(field, omega)
        && g.error * fs.norm() <= tol
    {
        return Ok(HybridOutput {
            value: g.value * fs,
            error: g.error * fs.norm(),
            method: HybridMethod::GeometricCorrected,
            images: g.images,
            inner_solves: 0,
        });
    }
    let out = lens_field_solve(field, &SliceConfig { omega, interrupt, ..Default::default() })?;
    Ok(HybridOutput {
        value: out.value,
        error: out.error,
        method: HybridMethod::Rc,
        images: 0,
        inner_solves: out.inner_solves,
    })
}

/// The `LensField` equivalent of a point-mass-only field: `lambda = 1` and
/// `kc = |beta|^2/2` reproduce the [`crate::phases::PointMassField`] phase
/// exactly.
fn point_mass_equivalent(
    source: [f64; 2],
    lenses: &[(f64, f64, f64)],
) -> Result<LensField> {
    let components: Vec<Component> = lenses
        .iter()
        .map(|&(x, y, mass)| Component {
            x,
            y,
            profile: Profile::Point { mass, xc: 0.0 },
        })
        .collect();
    LensField::new(source, 0.0, 0.0, 0.0, components)
}

/// Geometric-optics amplification **with the stationary-phase correction** for
/// a field of point masses, with the verified (certificate-checked) image set
/// from [`crate::images::point_mass_field_images`].  A failed completeness
/// certificate is an error; `Ok(None)` at a caustic, where the asymptotic form
/// is invalid.
pub fn point_mass_field_geometric_corrected(
    source: [f64; 2],
    lenses: &[(f64, f64, f64)],
    omega: f64,
) -> Result<Option<GeometricCorrected>> {
    let field = point_mass_equivalent(source, lenses)?;
    let images = match crate::images::point_mass_field_images(source, lenses) {
        Ok(images) => images,
        // A caustic: `None` is this function's documented "asymptotic form
        // invalid" answer.  Any other finder error (a violated certificate)
        // is a bug worth surfacing, so it propagates.
        Err(Error::DegenerateImage) => return Ok(None),
        Err(e) => return Err(e),
    };
    let n = images.len();
    Ok(sp_corrected_sum(&field, &images, omega)
        .map(|(value, error)| GeometricCorrected {
            value,
            error,
            images: n,
        }))
}

/// High-omega hybrid for a field of point masses ([`crate::images`]): the same
/// correction as [`lens_field_hybrid`], but the images come from the verified
/// certificate-checked finder rather than the general multi-start Newton grid
/// — a failed completeness certificate is an error, so the geometric branch
/// can no longer silently miss an image.
/// `lenses` is a list of `(x, y, mass)`; the phase convention matches
/// [`crate::phases::PointMassField`] and
/// [`crate::images::geometric_amplification`].  `interrupt` is passed to the
/// exact RC engine on the fallback branch; see [`InterruptHook`].
pub fn point_mass_field_hybrid(
    source: [f64; 2],
    lenses: &[(f64, f64, f64)],
    omega: f64,
    tol: f64,
    interrupt: Option<InterruptHook>,
) -> Result<HybridOutput> {
    let field = point_mass_equivalent(source, lenses)?;
    let fs = free_space_2d(omega);
    // The geometric branch needs a trustworthy image set.  A degenerate
    // (caustic) or certificate-failed set falls through to the exact RC
    // engine below — caustics are its home turf — so any finder error maps to
    // the empty set (= no geometric branch) rather than propagating.
    let images = crate::images::point_mass_field_images(source, lenses).unwrap_or_default();
    if let Some((value, error)) = sp_corrected_sum(&field, &images, omega)
        && error * fs.norm() <= tol
    {
        return Ok(HybridOutput {
            value: value * fs,
            error: error * fs.norm(),
            method: HybridMethod::GeometricCorrected,
            images: images.len(),
            inner_solves: 0,
        });
    }
    let out = lens_field_solve(&field, &SliceConfig { omega, interrupt, ..Default::default() })?;
    Ok(HybridOutput {
        value: out.value,
        error: out.error,
        method: HybridMethod::Rc,
        images: 0,
        inner_solves: out.inner_solves,
    })
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

    /// O1, the completeness-certificate battery (2026-10-19): the finder must
    /// pass its own certificate on every family, and the certificate's RHS —
    /// the index identity `#(det H>0) - #(det H<0) = sign(l0*l2) - sum W_j`
    /// with the per-cusp-group windings of [`cusp_groups`] — must match the
    /// hand-derived table (pinned here *independently* of the implementation):
    /// points wind always, SIS/CIS{xc=0} only when the source is inside their
    /// Einstein ring (`|g_s| < |psi0|`), cored/smooth profiles never, and
    /// coincident cusps group with the most singular class dominating.
    ///
    /// The 5-star stock and the close pair are the cases the old grid silently
    /// missed (their faint companions); the certificate now demands them.
    #[test]
    fn finder_certificate_battery() {
        // (name, field, hand-derived RHS, pinned image count)
        let cases: Vec<(&str, LensField, i64, usize)> = vec![
            ("macro + k=0.3 g=0.2", LensField::new([0.4, -0.3], 0.3, 0.2, 0.0, vec![]).unwrap(), 1, 1),
            ("macro - saddle k=0.5 g=0.7", LensField::new([0.4, -0.3], 0.5, 0.7, 0.0, vec![]).unwrap(), -1, 1),
            ("macro + k=1.5 g=0.2", LensField::new([0.4, -0.3], 1.5, 0.2, 0.0, vec![]).unwrap(), 1, 1),
            ("macro + k=0.45 g=0.5", LensField::new([0.4, -0.3], 0.45, 0.5, 0.0, vec![]).unwrap(), 1, 1),
            ("1 point k=0", LensField::new([0.5, 0.0], 0.0, 0.0, 0.0, vec![point(0.0, 0.0, 1.0)]).unwrap(), 0, 2),
            ("1 point k=0.3", LensField::new([0.5, 0.0], 0.3, 0.0, 0.0, vec![point(0.0, 0.0, 1.0)]).unwrap(), 0, 2),
            ("1 point k=0.9", LensField::new([0.5, 0.0], 0.9, 0.0, 0.0, vec![point(0.0, 0.0, 1.0)]).unwrap(), 0, 2),
            ("1 point k=1.5 defocusing (no images)", LensField::new([0.3, 0.1], 1.5, 0.2, 0.0, vec![point(0.0, 0.0, 1.0)]).unwrap(), 0, 0),
            ("1 point k=1.5 defocusing, light star", LensField::new([0.3, 0.1], 1.5, 0.2, 0.0, vec![point(0.0, 0.0, 0.01)]).unwrap(), 0, 2),
            ("2 points k=0", LensField::new([0.2, 0.1], 0.0, 0.0, 0.0, vec![point(-0.8, 0.5, 0.6), point(0.9, -0.4, 0.4)]).unwrap(), -1, 3),
            ("2 points k=0.3 g=0.2", LensField::new([0.2, 0.1], 0.3, 0.2, 0.0, vec![point(-0.8, 0.5, 0.6), point(0.9, -0.4, 0.4)]).unwrap(), -1, 3),
            ("SIS psi0=1 (inside)", LensField::new([0.5, 0.0], 0.0, 0.0, 0.0, vec![Component { x: 0.0, y: 0.0, profile: Profile::Sis { psi0: 1.0 } }]).unwrap(), 0, 2),
            ("SIS psi0=0.1 (outside)", LensField::new([0.5, 0.0], 0.0, 0.0, 0.0, vec![Component { x: 0.0, y: 0.0, profile: Profile::Sis { psi0: 0.1 } }]).unwrap(), 1, 1),
            ("SIS psi0=0.1 (inside)", LensField::new([0.05, 0.0], 0.0, 0.0, 0.0, vec![Component { x: 0.0, y: 0.0, profile: Profile::Sis { psi0: 0.1 } }]).unwrap(), 0, 2),
            ("SIS psi0=5 (strong)", LensField::new([0.5, 0.0], 0.0, 0.0, 0.0, vec![Component { x: 0.0, y: 0.0, profile: Profile::Sis { psi0: 5.0 } }]).unwrap(), 0, 2),
            ("SIS + saddle macro", LensField::new([0.3, 0.1], 0.5, 0.7, 0.0, vec![Component { x: 0.0, y: 0.0, profile: Profile::Sis { psi0: 1.0 } }]).unwrap(), -2, 2),
            ("SIS + defocusing macro", LensField::new([0.3, 0.1], 1.5, 0.2, 0.0, vec![Component { x: 0.0, y: 0.0, profile: Profile::Sis { psi0: 1.0 } }]).unwrap(), 0, 0),
            ("CIS xc=0 psi0=0.1 (outside)", LensField::new([0.5, 0.0], 0.0, 0.0, 0.0, vec![Component { x: 0.0, y: 0.0, profile: Profile::Cis { psi0: 0.1, xc: 0.0 } }]).unwrap(), 1, 1),
            ("CIS xc=0.2", LensField::new([0.5, 0.0], 0.0, 0.0, 0.0, vec![Component { x: 0.0, y: 0.0, profile: Profile::Cis { psi0: 1.0, xc: 0.2 } }]).unwrap(), 1, 1),
            ("NFW psi0=1 xs=0.5", LensField::new([0.5, 0.0], 0.0, 0.0, 0.0, vec![Component { x: 0.0, y: 0.0, profile: Profile::Nfw { psi0: 1.0, xs: 0.5 } }]).unwrap(), 1, 1),
            ("softened point xc=0.3", LensField::new([0.5, 0.0], 0.0, 0.0, 0.0, vec![Component { x: 0.0, y: 0.0, profile: Profile::Point { mass: 1.0, xc: 0.3 } }]).unwrap(), 1, 3),
            ("weak softened point beta=0", LensField::new([0.0, 0.0], 0.0, 0.0, 0.0, vec![Component { x: 0.0, y: 0.0, profile: Profile::Point { mass: 0.01, xc: 0.3 } }]).unwrap(), 1, 1),
            ("2 close points m=1,1e-3", LensField::new([0.5, 0.0], 0.0, 0.0, 0.0, vec![point(0.0, 0.0, 1.0), point(0.01, 0.0, 0.001)]).unwrap(), -1, 3),
            ("coincident points m=1,0.5", LensField::new([0.5, 0.0], 0.0, 0.0, 0.0, vec![point(0.0, 0.0, 1.0), point(0.0, 0.0, 0.5)]).unwrap(), 0, 2),
            ("coincident point(1)+SIS(0.5)", LensField::new([0.5, 0.0], 0.0, 0.0, 0.0, vec![point(0.0, 0.0, 1.0), Component { x: 0.0, y: 0.0, profile: Profile::Sis { psi0: 0.5 } }]).unwrap(), 0, 2),
            ("coincident SIS+SIS 0.3,0.3", LensField::new([0.5, 0.0], 0.0, 0.0, 0.0, vec![Component { x: 0.0, y: 0.0, profile: Profile::Sis { psi0: 0.3 } }, Component { x: 0.0, y: 0.0, profile: Profile::Sis { psi0: 0.3 } }]).unwrap(), 0, 2),
            ("SIS + point", LensField::new([0.5, 0.0], 0.0, 0.0, 0.0, vec![Component { x: 0.0, y: 0.0, profile: Profile::Sis { psi0: 0.5 } }, point(1.0, 0.3, 0.2)]).unwrap(), -1, 3),
            ("negative point m=-1 (no images)", LensField::new([0.5, 0.0], 0.0, 0.0, 0.0, vec![point(0.0, 0.0, -1.0)]).unwrap(), 0, 0),
            ("negative SIS psi0=-1 (no images)", LensField::new([0.5, 0.0], 0.0, 0.0, 0.0, vec![Component { x: 0.0, y: 0.0, profile: Profile::Sis { psi0: -1.0 } }]).unwrap(), 0, 0),
            ("negative SIS psi0=-0.1 (outside)", LensField::new([0.5, 0.0], 0.0, 0.0, 0.0, vec![Component { x: 0.0, y: 0.0, profile: Profile::Sis { psi0: -0.1 } }]).unwrap(), 1, 1),
            ("negative + positive point", LensField::new([0.5, 0.2], 0.0, 0.0, 0.0, vec![point(0.0, 0.0, -0.5), point(1.0, 0.0, 0.5)]).unwrap(), -1, 1),
            ("5-star stock k=0", LensField::new([0.5, 0.2], 0.0, 0.0, 0.0, vec![
                point(0.75057279962800205, 2.3832828058174531, 0.05),
                point(1.6541141414711609, -1.6487568600564488, 0.05),
                point(-1.1990022905326474, 2.2413206723775714, 0.05),
                point(-2.9684081726065514, 1.9273705102965977, 0.05),
                point(1.7824165725122771, -0.19239028293767557, 0.05),
            ]).unwrap(), -4, 6),
        ];
        for (name, f, want_rhs, want_n) in cases {
            let groups = cusp_groups(&f);
            let rhs = index_identity_rhs(&f, &groups);
            assert_eq!(rhs, want_rhs, "{name}: hand-derived RHS");
            let ims = lens_field_images(&f)
                .unwrap_or_else(|e| panic!("{name}: finder failed its certificate: {e}"));
            assert_eq!(ims.len(), want_n, "{name}: image count");
            let (mut np, mut nn) = (0i64, 0i64);
            for x in &ims {
                let (_, h) = real_grad_and_hess(&f, *x);
                let det = h[0][0] * h[1][1] - h[0][1] * h[1][0];
                assert!(det.abs() >= 1e-8, "{name}: degenerate image");
                if det > 0.0 {
                    np += 1;
                } else {
                    nn += 1;
                }
            }
            assert_eq!(np - nn, rhs, "{name}: index sum vs RHS");
        }
    }

    /// The close-pair light star's companion is the case the old grid missed;
    /// its local scale (set by the heavy star's Hessian, not the macro) must
    /// pin it: the companion sits at `~m/|g_s|` and the group's companion
    /// radius must cover it.
    #[test]
    fn close_pair_companion_radius_covers_the_faint_image() {
        let f = LensField::new(
            [0.5, 0.0],
            0.0,
            0.0,
            0.0,
            vec![point(0.0, 0.0, 1.0), point(0.01, 0.0, 0.001)],
        )
        .unwrap();
        let groups = cusp_groups(&f);
        let light = groups.iter().find(|g| g.x == 0.01).expect("light group");
        assert!(light.is_point && light.companion_radius > 0.0);
        let ims = lens_field_images(&f).unwrap();
        let d = ims
            .iter()
            .map(|x| (x[0] - light.x).hypot(x[1] - light.y))
            .fold(f64::INFINITY, f64::min);
        assert!(d <= light.companion_radius, "companion at {d:e} > radius {:e}", light.companion_radius);
    }

    /// The general finder on the getting-started 40-mass stock must reproduce
    /// the point-mass-only finder (independent seed logic and certificate) and
    /// the pinned external references of `crate::images`: 41 images, 40 of
    /// them saddles, and the pinned F_geo(2).
    #[test]
    fn lens_field_finder_matches_point_mass_finder_on_40_masses() {
        use crate::images::tests::{NOTEBOOK_BETA, NOTEBOOK_FIELD};
        let lenses: Vec<(f64, f64, f64)> = NOTEBOOK_FIELD.to_vec();
        let lf = LensField::new(
            NOTEBOOK_BETA,
            0.0,
            0.0,
            0.0,
            lenses.iter().map(|&(x, y, m)| point(x, y, m)).collect(),
        )
        .unwrap();
        let ims = lens_field_images(&lf).unwrap();
        let poly = crate::images::point_mass_field_images(NOTEBOOK_BETA, &lenses).unwrap();
        assert_eq!(ims.len(), 41);
        assert_eq!(poly.len(), 41);
        for p in &poly {
            let best = ims
                .iter()
                .map(|q| (q[0] - p[0]).abs() + (q[1] - p[1]).abs())
                .fold(f64::INFINITY, f64::min);
            assert!(best < 1e-6, "image {p:?} not matched (closest {best:e})");
        }
        let f = lens_field_geometric(&lf, 2.0).unwrap();
        let want = Complex64::new(-1.458_607_255_602_454_0, 0.459_269_484_456_860_0);
        assert!((f - want).norm() < 1e-8, "F_geo(2) = {f} vs {want}");
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
    #[allow(clippy::type_complexity)]
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
    /// point-mass implementation: same images, same stationary-phase sum.
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
        let ims = lens_field_images(&lf).unwrap();
        let poly = crate::images::point_mass_field_images(beta, &lenses).unwrap();
        assert_eq!(ims.len(), poly.len(), "image count: {ims:?} vs {poly:?}");
        for p in &poly {
            let best = ims
                .iter()
                .map(|q| (q[0] - p[0]).abs() + (q[1] - p[1]).abs())
                .fold(f64::INFINITY, f64::min);
            assert!(best < 1e-6, "image {p:?} not matched (closest {best:e})");
        }
        let omega = 100.0;
        let a = lens_field_geometric(&lf, omega).unwrap();
        let b = crate::images::geometric_amplification(beta, &lenses, omega).unwrap();
        assert!((a - b).norm() / b.norm() < 1e-5, "{a} vs {b}");
    }

    /// A caustic (beta = 0 for an isolated point mass) has no discrete
    /// geometric sum: the finder reports `DegenerateImage`, `geometric`
    /// propagates it (never a silent zero), `geometric_corrected` answers
    /// `Ok(None)` (its documented "asymptotic form invalid"), and the hybrid
    /// falls back to the exact RC — the point-mass-sibling contract, now for
    /// general fields.
    #[test]
    fn lens_field_caustic_reported_not_silent() {
        let lf = LensField::new([0.0, 0.0], 0.0, 0.0, 0.0, vec![point(0.0, 0.0, 1.0)]).unwrap();
        let omega = 2.0;
        assert!(matches!(
            lens_field_images(&lf).unwrap_err(),
            Error::DegenerateImage
        ));
        assert!(matches!(
            lens_field_geometric(&lf, omega).unwrap_err(),
            Error::DegenerateImage
        ));
        assert!(lens_field_geometric_corrected(&lf, omega).unwrap().is_none());
        let h = lens_field_hybrid(&lf, omega, 1e-3, None).unwrap();
        assert_eq!(h.method, HybridMethod::Rc);
        let direct = crate::phases::PointMassField { beta: [0.0, 0.0], lenses: vec![(0.0, 0.0, 1.0)] };
        let exact =
            crate::slice::slice_field(&direct, &SliceConfig { omega, ..Default::default() })
                .unwrap();
        assert!((h.value - exact.value).norm() / exact.value.norm() < 1e-13);
    }

    /// A caustic (beta = 0 for an isolated point mass) has no discrete
    /// geometric sum: `geometric_corrected` answers `None` (its documented
    /// "asymptotic form invalid") and the hybrid falls back to the exact RC
    /// instead of poisoning the fast branch with a degenerate image set.
    #[test]
    fn point_mass_caustic_has_no_geometric_branch() {
        let lenses = [(0.0, 0.0, 1.0)];
        let src = [0.0, 0.0];
        assert!(
            point_mass_field_geometric_corrected(src, &lenses, 2.0)
                .unwrap()
                .is_none()
        );
        let h = point_mass_field_hybrid(src, &lenses, 2.0, 1e-3, None).unwrap();
        assert_eq!(h.method, HybridMethod::Rc);
        let direct = crate::phases::PointMassField { beta: src, lenses: lenses.to_vec() };
        let exact =
            crate::slice::slice_field(&direct, &SliceConfig { omega: 2.0, ..Default::default() })
                .unwrap();
        assert!((h.value - exact.value).norm() / exact.value.norm() < 1e-13);
    }

    /// Macro-only: a single image at the quadratic centre with
    /// `mu = 1/(lambda0 lambda2)`.
    #[test]
    fn geometric_macro_only_one_image() {
        let lf = LensField::new([0.4, -0.3], 0.3, 0.2, 0.0, vec![]).unwrap();
        let ims = lens_field_images(&lf).unwrap();
        assert_eq!(ims.len(), 1, "{ims:?}");
        let (x0, x2) = (
            lf.beta[0] / lf.lambda0,
            lf.beta[1] / lf.lambda2,
        );
        assert!((ims[0][0] - x0).abs() < 1e-9 && (ims[0][1] - x2).abs() < 1e-9);
        let f = lens_field_geometric(&lf, 1e6).unwrap();
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


    /// Complex-step Hessian oracle at a real point, used to verify the
    /// analytic Hessian in [`sp_derivative_tensors_match_analytic_point_mass`].
    fn hess_cs(field: &LensField, x: [f64; 2], h: f64) -> [[f64; 2]; 2] {
        let mut hess = [[0.0f64; 2]; 2];
        for (j, slot) in hess.iter_mut().enumerate() {
            for k in 0..2 {
                let mut z = [Complex64::from(x[0]), Complex64::from(x[1])];
                z[k] += Complex64::i() * h;
                let gr = field.gradient(&SVector::<Complex64, 2>::new(z[0], z[1]));
                slot[k] = if j == 0 { gr[0].im } else { gr[1].im } / h;
            }
        }
        hess
    }

    /// P4.1: the analytic-Hessian chain rule must reproduce the complex-step
    /// Hessian at real points, and the central complex steps must reproduce
    /// the exact point-mass tensors (which `comparison/sp_correction.py`
    /// FD-verified: `phi4_2222 = 6/x^4`, not the `4/x^4` a hand slip gave).
    #[test]
    fn sp_derivative_tensors_match_analytic_point_mass() {
        let field =
            LensField::new([0.5, 0.0], 0.0, 0.0, 0.0, vec![point(0.0, 0.0, 1.0)]).unwrap();
        for y in [0.5f64, 2.0] {
            let root = (y * y + 4.0).sqrt();
            for x0 in [0.5 * (y + root), 0.5 * (y - root)] {
                let x = [x0, 0.0];
                // (a) analytic Hessian vs the complex-step Hessian
                let h_cs = hess_cs(&field, x, 1e-8);
                let h_an =
                    hess_analytic(&field, [Complex64::from(x[0]), Complex64::from(x[1])]);
                let (_, h_real) = real_grad_and_hess(&field, x);
                for a in 0..2 {
                    for b in 0..2 {
                        let d = (h_an[a][b] - Complex64::from(h_cs[a][b])).norm();
                        assert!(d < 1e-8, "H[{a}][{b}] analytic vs complex step: {d:e}");
                        let dr = (h_real[a][b] - h_cs[a][b]).abs();
                        assert!(dr < 1e-8, "H[{a}][{b}] real_grad_and_hess vs complex step: {dr:e}");
                    }
                }
                // (b) exact tensors of phi = (x1-y)^2/2 + x2^2/2 - ln|x| at (x0, 0)
                let x3 = x0.powi(3);
                let x4 = x0.powi(4);
                let mut p3 = [[[0.0f64; 2]; 2]; 2];
                p3[0][0][0] = -2.0 / x3;
                p3[0][1][1] = 2.0 / x3;
                p3[1][0][1] = 2.0 / x3;
                p3[1][1][0] = 2.0 / x3;
                let got3 = phi3_cs(&field, x, 1e-5);
                for a in 0..2 {
                    for b in 0..2 {
                        for c in 0..2 {
                            let d = (got3[a][b][c] - p3[a][b][c]).abs();
                            assert!(
                                d < 1e-6,
                                "phi3[{a}][{b}][{c}] at x={x0}: {d:e} (want {:e}, got {:e})",
                                p3[a][b][c],
                                got3[a][b][c]
                            );
                        }
                    }
                }
                let got4 = phi4_cs(&field, x, 3e-5);
                for (a, g_a) in got4.iter().enumerate() {
                    for (b, g_ab) in g_a.iter().enumerate() {
                        for (c, g_abc) in g_ab.iter().enumerate() {
                            for (d, &got_val) in g_abc.iter().enumerate() {
                                let want = if (a + b + c + d) % 2 == 1 {
                                    0.0 // odd in x2: the phase is even in x2
                                } else if (a, b, c, d) == (0, 0, 0, 0)
                                    || (a, b, c, d) == (1, 1, 1, 1)
                                {
                                    6.0 / x4
                                } else {
                                    -6.0 / x4
                                };
                                let err = (got_val - want).abs();
                                // scale-aware: the stencil's truncation grows
                                // with the higher derivatives near the centre
                                assert!(
                                    err < 1e-5 * want.abs().max(1.0),
                                    "phi4[{a}][{b}][{c}][{d}] at x={x0}: {err:e} \
                                     (want {want:e}, got {got_val:e})",
                                );
                            }
                        }
                    }
                }
                // (c) the assembled coefficient C against the prototype's
                // analytic value (`comparison/sp_correction.py`, numpy)
                let data = sp_image_data(&field, x).unwrap();
                let want_c = if y == 0.5 {
                    if x0 > 0.0 {
                        -0.1572635723378
                    } else {
                        0.2945184742985
                    }
                } else if x0 > 0.0 {
                    -0.004838188479867
                } else {
                    -0.07849514485346
                };
                assert!(
                    (data.c - want_c).abs() < 1e-5,
                    "C at x={x0}, y={y}: {} vs prototype {}",
                    data.c,
                    want_c
                );
            }
        }
    }

    /// The 15-pairing sum must collapse to the textbook 1D count when only
    /// `phi3[000]` is nonzero: `Sum_15 = 15 p^2 g^3`.
    #[test]
    fn sum15_reduces_to_the_1d_pairing_count() {
        let mut p3 = [[[0.0f64; 2]; 2]; 2];
        p3[0][0][0] = 2.0;
        let hinv = [[0.5, 0.0], [0.0, 0.25]];
        let s = sum15(&p3, &hinv);
        assert!((s - 15.0 * 4.0 * 0.125).abs() < 1e-12, "sum15 = {s}");
    }

    /// The closed-form `psi''` must match a generic complex step of `psi'` at
    /// real arguments, so the hand algebra is *checked*, not trusted.
    #[test]
    fn dpsi_dw2_closed_forms_match_the_generic_step() {
        let profiles = [
            Profile::Point { mass: 1.3, xc: 0.0 },
            Profile::Point { mass: 0.7, xc: 0.25 },
            Profile::Sis { psi0: 1.0 },
            Profile::Cis { psi0: 1.0, xc: 0.0 },
            Profile::Cis { psi0: 1.0, xc: 0.2 },
            Profile::Nfw { psi0: 1.0, xs: 0.5 },
        ];
        for p in profiles {
            for w in [0.2f64, 1.0, 3.7] {
                let wc = Complex64::from(w);
                let generic = {
                    let h = 1e-5 * w;
                    (p.dpsi_dw(wc + Complex64::i() * h) - p.dpsi_dw(wc - Complex64::i() * h))
                        / (Complex64::i() * (2.0 * h))
                };
                let got = p.dpsi_dw2(wc);
                let d = (got - generic).norm() / generic.norm().max(1e-12);
                assert!(d < 1e-8, "{p:?} at w={w}: {d:e} (got {got}, generic {generic})");
            }
        }
    }
}
