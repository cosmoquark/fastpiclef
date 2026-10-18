//! Batched RC (real core + rotated tails) for 1D rational lenses (P1, plan
//! §3.4): one shared contour serves a whole source grid.
//!
//! `phi(z; beta) = 1/2 (z - beta)^2 - pot(z)` with `pot` rational; the exactness
//! argument is the simplest of the family — the potential's singularities are
//! *poles*, so Cauchy needs no sheet bookkeeping at all: keep the real core
//! spanning every pole's real part (times the margin), rotate the tails by
//! `cfg.tail_angle ∈ (0, pi/4]` into the Gaussian valleys (where the quadratic
//! dominates and every pole stays outside the swept wedges), and the
//! deformation is exact.  Rejected up front: an unbounded potential
//! (`deg A > deg B`, the quadratic no longer wins) and real-axis poles (the
//! original real-line integral diverges).
//!
//! The batch factors the source dependence out exactly as the 2D batched inner
//! solve does: `e^{i omega phi} = e^{i omega (z^2/2 - pot)} e^{-i omega beta z}
//! e^{i omega beta^2/2}`, with a re-anchored recurrence for uniform source
//! grids.  Graded against the production thimble in `tests/batch.rs`.

use crate::adaptive::{AdaptiveVecConfig, integrate_vec};
use crate::error::{Error, Result};
use crate::phase::RationalPhase;
use crate::phases::{CustomRationalLens, RationalLens, SimGLens};
use crate::roots::poly_eval;
use crate::slice::SliceConfig;
use crate::slice_batch::{fill_series, uniform_step};
use nalgebra::SVector;
use num_complex::Complex64;

/// Internal extension point of the 1D batched kernel: a rational lens phase
/// `1/2 (z - beta)^2 - pot(z)` whose poles (via [`RationalPhase::poles`]) and
/// asymptotic behaviour are known.
pub(crate) trait Rational1DModel: RationalPhase + Sync {
    fn beta(&self) -> f64;
    /// Same lens at a different source position (the per-point thimble grid).
    #[allow(dead_code)] // used by the Python bindings (feature-gated)
    fn with_beta(&self, beta: f64) -> Self
    where
        Self: Sized;
    /// `deg A <= deg B`: the quadratic leads at infinity and the tails decay.
    fn bounded_at_infinity(&self) -> bool;
    /// `1/2 z^2 - pot(z)` (the beta-independent part of the phase), recovered
    /// from the phase itself: `value(z) + beta z - beta^2/2`.
    fn phase_no_source(&self, z: Complex64) -> Complex64 {
        let b = Complex64::from(self.beta());
        self.value(&SVector::<Complex64, 1>::new(z)) + b * z - 0.5 * b * b
    }
}

impl Rational1DModel for RationalLens {
    fn beta(&self) -> f64 {
        self.beta
    }
    fn with_beta(&self, beta: f64) -> Self {
        RationalLens { beta, kappa: self.kappa }
    }
    fn bounded_at_infinity(&self) -> bool {
        true
    }
}

impl Rational1DModel for CustomRationalLens {
    fn beta(&self) -> f64 {
        self.beta
    }
    fn with_beta(&self, beta: f64) -> Self {
        CustomRationalLens {
            beta,
            kappa: self.kappa,
            num: self.num.clone(),
            den: self.den.clone(),
        }
    }
    fn bounded_at_infinity(&self) -> bool {
        let trim = |v: &[Complex64]| {
            let mut n = v.len();
            while n > 1 && v[n - 1].norm() == 0.0 {
                n -= 1;
            }
            n
        };
        trim(&self.num) <= trim(&self.den)
    }
    fn phase_no_source(&self, z: Complex64) -> Complex64 {
        // Direct evaluation (avoids the value() round trip for the escape
        // hatch; numerically identical expression).
        0.5 * z * z - Complex64::from(self.kappa) * poly_eval(&self.num, z) / poly_eval(&self.den, z)
    }
}

impl Rational1DModel for SimGLens {
    fn beta(&self) -> f64 {
        self.beta
    }
    fn with_beta(&self, beta: f64) -> Self {
        SimGLens { beta, kappa: self.kappa, kmax: self.kmax }
    }
    fn bounded_at_infinity(&self) -> bool {
        true
    }
}

/// Batched raw integral `int_R e^{i omega phi} dz` for a rational lens over a
/// grid of source positions.
///
/// `cfg.omega` is overridden by the `omega` argument; `margin`, `tol_inner`,
/// `tail_angle` and `parallel` are honoured (`margin_outer`, `tol_outer` and
/// `band_ratio` are ignored — there is no outer variable here).  Returns
/// `(values, errors)` in the caller's beta order; raw integrals, as everywhere.
#[allow(private_bounds)]
pub fn rc_1d_batch<M: Rational1DModel>(
    lens: &M,
    betas: &[f64],
    omega: f64,
    cfg: &SliceConfig,
) -> Result<(Vec<Complex64>, Vec<f64>)> {
    if !(omega > 0.0) || !omega.is_finite() {
        return Err(Error::InvalidInput("omega must be > 0 and finite"));
    }
    if betas.is_empty() {
        return Err(Error::InvalidInput("empty source grid"));
    }
    for &b in betas {
        if !b.is_finite() {
            return Err(Error::InvalidInput("source positions must be finite"));
        }
    }
    if !(cfg.margin > 0.0) || !cfg.margin.is_finite() {
        return Err(Error::InvalidInput("margin must be > 0 and finite"));
    }
    if !(cfg.tail_angle > 0.0) || cfg.tail_angle > std::f64::consts::FRAC_PI_4 {
        return Err(Error::InvalidInput("tail angle must be in (0, pi/4]"));
    }
    if !lens.bounded_at_infinity() {
        return Err(Error::InvalidInput(
            "potential unbounded at infinity (deg A > deg B): the RC construction requires deg A <= deg B",
        ));
    }
    let poles = lens.poles();
    for p in &poles {
        if !p.re.is_finite() || !p.im.is_finite() {
            return Err(Error::InvalidInput("lens poles must be finite"));
        }
        if p.im.abs() < 1e-12 {
            return Err(Error::InvalidInput(
                "pole on the real axis: the real-line integral diverges",
            ));
        }
    }

    let m = betas.len();
    let bmax = betas.iter().fold(0.0f64, |acc, &b| acc.max(b.abs()));

    // contour: real core spanning every pole's real part (and the sources)
    let (mut amin, mut amax) = (betas[0], betas[0]);
    for &b in betas {
        amin = amin.min(b);
        amax = amax.max(b);
    }
    for p in &poles {
        amin = amin.min(p.re);
        amax = amax.max(p.re);
    }
    let (aa, bb) = (amin - cfg.margin, amax + cfg.margin);
    let mut pts = vec![aa, bb];
    for p in &poles {
        pts.push(p.re);
    }
    pts.sort_by(|p, q| p.partial_cmp(q).unwrap());
    pts.dedup();

    // roundoff floor: sampled |pot| bound (an underestimate only costs panels)
    let zmax = aa.abs().max(bb.abs());
    let mut potmax = 1.0f64;
    for &z0 in &[aa, 0.5 * (aa + bb), bb] {
        let pot = 0.5 * z0 * z0 - lens.phase_no_source(Complex64::from(z0));
        potmax = potmax.max(pot.norm());
    }
    let qbound = 0.5 * zmax * zmax + bmax * zmax + 0.5 * bmax * bmax + potmax + 5.0;
    let floor = 4e-16 * (1.0 + omega * qbound);
    let floors = vec![floor; m];

    // twiddle constants e^{i omega beta^2 / 2} (once for the whole batch)
    let c: Vec<Complex64> = betas
        .iter()
        .map(|&b| (Complex64::i() * (omega * b * b * 0.5)).exp())
        .collect();
    let beta_uniform = uniform_step(betas);
    let iw = Complex64::i() * omega;

    // integrand: e4fac * e^{i omega (z^2/2 - pot)} * e^{-i omega beta z} * c
    let eval = |z: Complex64, e4fac: Complex64, vals: &mut [Complex64], errs: &mut [f64]| {
        let base = e4fac * (iw * lens.phase_no_source(z)).exp();
        if beta_uniform.is_some() {
            let mut t = vec![Complex64::new(0.0, 0.0); m];
            fill_series(&mut t, betas, -z * omega, beta_uniform);
            for i in 0..m {
                vals[i] = base * t[i] * c[i];
                errs[i] = 0.0;
            }
        } else {
            for (i, &b) in betas.iter().enumerate() {
                vals[i] = base * (iw * (-(b * z))).exp() * c[i];
                errs[i] = 0.0;
            }
        }
    };

    let one = Complex64::new(1.0, 0.0);
    let core_cfg = AdaptiveVecConfig {
        tol: cfg.tol_inner,
        floors: floors.clone(),
        min_len: 1e-15,
        max_panels: 200_000,
    };
    let tail_cfg = AdaptiveVecConfig {
        tol: cfg.tol_inner,
        floors: floors.clone(),
        min_len: 1e-14,
        max_panels: 200_000,
    };
    let mut total = vec![Complex64::new(0.0, 0.0); m];
    let mut terr = vec![0.0f64; m];

    for win in pts.windows(2) {
        let (lo, hi) = (win[0], win[1]);
        let half = 0.5 * (hi - lo);
        for (x0, sgn) in [(lo, 1.0), (hi, -1.0)] {
            let f = |t: f64, vals: &mut [Complex64], errs: &mut [f64]| {
                eval(Complex64::from(x0 + sgn * t), one, vals, errs)
            };
            let res = integrate_vec(&f, 0.0, half, &core_cfg, cfg.parallel);
            for k in 0..m {
                total[k] += res.values[k];
                terr[k] += res.errors[k];
            }
        }
    }

    // rotated tails: z = end + sgn*e4*t, both contributing +e4 * integrand
    let e4 = Complex64::from_polar(1.0, cfg.tail_angle);
    let mut t_end = (2.0 * 45.0 / omega).sqrt() + bmax / cfg.tail_angle.cos() + 2.0;
    for (end_z, sgn) in [(bb, 1.0), (aa, -1.0)] {
        let eval_tail = |t: f64, vals: &mut [Complex64], errs: &mut [f64]| {
            eval(end_z + sgn * e4 * t, e4, vals, errs)
        };
        let mut probed = false;
        for _ in 0..5 {
            let mut peak = vec![0.0f64; m];
            let mut vals = vec![Complex64::new(0.0, 0.0); m];
            let mut errs = vec![0.0f64; m];
            for s in 0..16 {
                let tt = t_end * (s as f64) / 16.0;
                eval_tail(tt, &mut vals, &mut errs);
                for k in 0..m {
                    peak[k] = peak[k].max(vals[k].norm());
                }
            }
            eval_tail(t_end, &mut vals, &mut errs);
            probed = (0..m).all(|k| vals[k].norm() <= 1e-17 * peak[k].max(1e-30));
            if probed {
                break;
            }
            t_end *= 2.0;
        }
        let res = integrate_vec(&eval_tail, 0.0, t_end, &tail_cfg, cfg.parallel);
        let mut end_vals = vec![Complex64::new(0.0, 0.0); m];
        let mut end_errs = vec![0.0f64; m];
        eval_tail(t_end, &mut end_vals, &mut end_errs);
        for k in 0..m {
            total[k] += res.values[k];
            terr[k] += res.errors[k];
            let end = end_vals[k].norm();
            terr[k] += if probed { 4.0 * end } else { t_end * end };
        }
    }

    Ok((total, terr))
}
