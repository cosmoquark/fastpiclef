//! Polar-coordinate 2D wrapper around the 1D constant-phase contour solver
//! (Shi 2024, section 6):
//!
//! $F = \int_0^{2\pi} d\theta \int_0^\infty r\,e^{i\omega\phi(r\hat{n}(\theta))}\,dr$.
//!
//! Each radial integral is a *half-line* 1D problem solved on its Lefshetz
//! thimbles plus the flow line of the fixed endpoint $r = 0$
//! ([`crate::thimble`]); the analytic continuation in $r$ and the chain-rule
//! gradient come from a thin [`Phase<1>`] adapter, and the $r$ measure
//! factor enters as the quadrature amplitude. The $\theta$ integral is
//! periodic and analytic in $\theta$ for generic configurations, so the
//! trapezoidal rule on a uniform grid converges spectrally; the grid is
//! doubled (reusing all previous evaluations) until the value stabilises.
//!
//! Caveat: the radial saddles must stay inside the search window for every
//! $\theta$. Lensing phases dominated by $|x - \beta|^2/2$ confine them;
//! phases with runaway directions (e.g. a separable cubic, whose radial
//! saddle escapes to infinity as $\cos\theta \to 0$) are not suitable for
//! the polar decomposition.

use crate::saddles::saddles_from_polynomial;
use crate::thimble::{ThimbleConfig, solve_half_line};
use crate::{Error, Output, Phase, RationalPhase2D, Result};
use nalgebra::SVector;
use num_complex::Complex64;
use std::f64::consts::TAU;

/// Knobs for the polar pipeline. The frequency $\omega$ lives in
/// `radial.omega`.
#[derive(Debug, Clone)]
pub struct PolarConfig {
    /// Configuration of the per-angle radial thimble solves.
    pub radial: ThimbleConfig,
    /// Initial number of uniform theta points.
    pub theta_init: usize,
    /// Cap on the number of theta points.
    pub theta_max: usize,
    /// Stop doubling once |F_2N - F_N| < theta_tol * |F_2N|.
    pub theta_tol: f64,
}

impl Default for PolarConfig {
    fn default() -> Self {
        Self {
            radial: ThimbleConfig::default(),
            theta_init: 16,
            theta_max: 1024,
            theta_tol: 1e-10,
        }
    }
}

/// The 2D phase restricted to the ray $z \mapsto z\,\hat{n}(\theta)$,
/// viewed as a 1D phase in the (complexified) radius.
struct RadialPhase<'a> {
    phase: &'a dyn Phase<2>,
    cos_t: f64,
    sin_t: f64,
}

impl<'a> RadialPhase<'a> {
    fn embed(&self, r: Complex64) -> SVector<Complex64, 2> {
        SVector::<Complex64, 2>::new(r * self.cos_t, r * self.sin_t)
    }
}

impl<'a> Phase<1> for RadialPhase<'a> {
    fn value(&self, z: &SVector<Complex64, 1>) -> Complex64 {
        self.phase.value(&self.embed(z[0]))
    }

    fn gradient(&self, z: &SVector<Complex64, 1>) -> SVector<Complex64, 1> {
        let g = self.phase.gradient(&self.embed(z[0]));
        SVector::<Complex64, 1>::new(g[0] * self.cos_t + g[1] * self.sin_t)
    }
}

/// 2D Fresnel-integral solver: polar decomposition into radial
/// constant-phase contour solves.
///
/// Construct with [`PolarSolver::new`] for a general analytic phase, or
/// with [`PolarSolver::new_rational`] for a [`RationalPhase2D`] — then each
/// radial solve uses the complete per-angle polynomial saddle source
/// (required for multi-image lenses, and much cheaper than the per-angle
/// grid-Newton search).
pub struct PolarSolver<'a> {
    phase: &'a dyn Phase<2>,
    cfg: PolarConfig,
    rational: Option<&'a dyn RationalPhase2D>,
    thetas_used: usize,
}

impl<'a> PolarSolver<'a> {
    pub fn new(phase: &'a dyn Phase<2>, cfg: PolarConfig) -> Self {
        Self {
            phase,
            cfg,
            rational: None,
            thetas_used: 0,
        }
    }

    /// Like [`PolarSolver::new`], but the radial saddles at each angle come
    /// from the phase's declared radial polynomial (complete by
    /// construction).
    pub fn new_rational<P: RationalPhase2D>(phase: &'a P, cfg: PolarConfig) -> Self {
        Self {
            phase,
            cfg,
            rational: Some(phase),
            thetas_used: 0,
        }
    }

    /// Number of theta points at which the last [`PolarSolver::run`]
    /// stopped doubling.
    pub fn thetas_used(&self) -> usize {
        self.thetas_used
    }

    pub fn run(&mut self) -> Result<Output> {
        if self.cfg.theta_init < 2 {
            return Err(Error::InvalidInput("theta_init must be >= 2"));
        }
        let phase = self.phase;
        let rational = self.rational;
        let cfg = &self.cfg;
        let eval = |theta: f64| -> Result<Output> {
            let (cos_t, sin_t) = (theta.cos(), theta.sin());
            let radial = RadialPhase {
                phase,
                cos_t,
                sin_t,
            };
            let injected = match rational {
                Some(rp) => Some(saddles_from_polynomial(
                    &radial,
                    &rp.radial_stationary_polynomial(cos_t, sin_t),
                    cfg.radial.newton_tol,
                )?),
                None => None,
            };
            solve_half_line(&radial, &cfg.radial, &|r| r, injected.as_deref())
        };

        // Periodic trapezoid = mean over a uniform grid; doubling adds the
        // midpoints, so every radial solve is reused.
        let mut n = cfg.theta_init;
        let mut sum = Complex64::new(0.0, 0.0);
        let mut radial_err_sum = 0.0;
        for k in 0..n {
            let out = eval(TAU * k as f64 / n as f64)?;
            sum += out.value;
            radial_err_sum += out.error;
        }
        let mut value = sum * Complex64::from(TAU / n as f64);
        let mut delta = f64::INFINITY;
        while n < cfg.theta_max {
            for k in 0..n {
                let out = eval(TAU * (k as f64 + 0.5) / n as f64)?;
                sum += out.value;
                radial_err_sum += out.error;
            }
            n *= 2;
            let refined = sum * Complex64::from(TAU / n as f64);
            delta = (refined - value).norm();
            value = refined;
            if delta <= cfg.theta_tol * value.norm() {
                break;
            }
        }
        self.thetas_used = n;

        Ok(Output {
            value,
            error: delta + radial_err_sum * TAU / n as f64,
        })
    }
}
