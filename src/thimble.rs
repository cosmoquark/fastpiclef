//! Constant-phase contour solver (1D): the "decisive speedup" path of
//! Shi 2024 (`refs/2409.12991v1.pdf`).
//!
//! Instead of flowing a mesh of the whole real line, the Lefshetz thimbles
//! are constructed *directly*. For an analytic phase write
//! $i\phi = h + iH$ (so $h = -\mathrm{Im}\,\phi$, $H = \mathrm{Re}\,\phi$).
//! By Cauchy-Riemann, the level curves of $H$ are exactly the steepest
//! descent/ascent paths of $h$ — so each thimble $\mathcal{J}_i$ is the
//! piece of the level set $H = H(z_i)$ that descends from the saddle $z_i$.
//!
//! Pipeline:
//! 1. find the saddles $\phi'(z) = 0$ in a complex window ([`find_saddles`]);
//! 2. select the *effective* saddles — those that are real, or whose
//!    ascending constant-phase contour crosses the real axis (Shi's
//!    criterion; the ascent can never touch $\mathbb{R}$ when the saddle has
//!    $h > 0$, which is what silences the wrong member of a conjugate pair);
//! 3. trace each effective thimble by a predictor-corrector that follows
//!    the level curve $H = H_i$ (midpoint predictor along $\mp\nabla h$,
//!    Newton corrector along $\nabla H$), stopping once
//!    $\omega h <$ `morse_threshold` (integrand dead);
//! 4. integrate $e^{i\omega\phi}$ along the traced polylines with embedded
//!    Gauss-Legendre rules (8- vs 4-node) for the error estimate.
//!
//! Tracing the level curve from the saddle realises Shi's
//! "constant-phase contour method" without a global marching-squares grid:
//! the accuracy is set by the corrector tolerance (~1e-11) rather than by a
//! grid resolution, and the cost scales with the thimble length only.
//!
//! Known limitations (deliberate, v1): degenerate saddles (caustics,
//! $\phi'' = 0$) are rejected — use the mesh-flow [`crate::Solver`] there;
//! configurations on a Stokes line may stall the tracer (reported as an
//! error rather than silently mis-assembled).

use crate::saddles::{Saddle, find_saddles, grad, second_derivative, val};
use crate::{Error, Output, Phase, Result};
use num_complex::Complex64;

/// Knobs for the constant-phase contour pipeline.
#[derive(Debug, Clone)]
pub struct ThimbleConfig {
    /// Complex window (real extent) for saddle search and contour tracing.
    pub re_range: (f64, f64),
    /// Complex window (imaginary extent).
    pub im_range: (f64, f64),
    /// Newton seeds per axis for the saddle search.
    pub seeds_per_axis: usize,
    /// Acceptance tolerance on |phi'| for a saddle.
    pub newton_tol: f64,
    /// Maximum arc-length step of the contour tracer.
    pub ds_max: f64,
    /// Maximum change of omega*h per tracer step (controls sampling density
    /// relative to the integrand's decay).
    pub max_dh: f64,
    /// Stop tracing a descending branch once omega*h falls below this
    /// (same semantics as the mesh solver's pruning threshold).
    pub morse_threshold: f64,
    /// Frequency in the integrand e^{i omega phi}.
    pub omega: f64,
}

impl Default for ThimbleConfig {
    fn default() -> Self {
        Self {
            re_range: (-8.0, 8.0),
            im_range: (-8.0, 8.0),
            seeds_per_axis: 16,
            newton_tol: 1e-11,
            ds_max: 0.2,
            max_dh: 1.0,
            morse_threshold: -30.0,
            omega: 1.0,
        }
    }
}

/// Why a level-curve trace ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stop {
    /// omega*h fell below the Morse threshold (descending; the good exit).
    MorseFloor,
    /// omega*h rose above -morse_threshold (ascending; saddle not reachable
    /// from the real axis through this branch).
    Escaped,
    /// The trace crossed the real axis (ascending; marks the saddle
    /// effective).
    CrossedReal,
    /// Left the search window before any other criterion fired.
    WindowExit,
    /// Step budget exhausted.
    StepCap,
}

struct Trace {
    points: Vec<Complex64>,
    stop: Stop,
    /// omega*h at the last accepted point.
    end_omega_h: f64,
    /// Where the real axis was crossed (interpolated), for ascending traces
    /// that ended with [`Stop::CrossedReal`].
    crossing_re: Option<f64>,
}

/// Unit take-off direction of the *descending* branch pair at a saddle:
/// $i\phi'' d^2 \in \mathbb{R}_-$, normalised to Re(d) > 0 (ties: Im > 0)
/// so that assembled thimbles run left-to-right like the real axis.
fn take_off_direction(phi2: Complex64) -> Complex64 {
    let theta = (std::f64::consts::PI - (Complex64::i() * phi2).arg()) / 2.0;
    let mut d = Complex64::from_polar(1.0, theta);
    if d.re < 0.0 || (d.re.abs() < 1e-12 && d.im < 0.0) {
        d = -d;
    }
    d
}

/// Follow the level curve $H = $ `target` from `start` (offset by a small
/// kick when `start` is a stationary point, where the tangent formula is
/// undefined). Descending branches move along $-\nabla h$ (h decreases),
/// ascending along $+\nabla h$; both keep $H$ fixed with a Newton corrector.
fn trace_level_curve(
    phase: &dyn Phase<1>,
    cfg: &ThimbleConfig,
    start: Complex64,
    target: f64,
    kick: Option<Complex64>,
    ascending: bool,
) -> Result<Trace> {
    let omega = cfg.omega;
    let tol_target = 1e-11 * (1.0 + target.abs());
    let sign = if ascending { -1.0 } else { 1.0 };
    let stalled = || {
        Err(Error::InvalidInput(
            "constant-phase trace stalled (Stokes line or pole?)",
        ))
    };

    // Unit tangent of the level curve: descending = -grad h = i conj(phi').
    let tangent = |z: Complex64| -> Option<Complex64> {
        let g = grad(phase, z);
        let n = g.norm();
        if n < 1e-13 {
            return None;
        }
        Some(Complex64::new(0.0, sign) * g.conj() / n)
    };

    let eps = 1e-4 * (1.0 + start.norm());
    let mut z = match kick {
        Some(dir) => start + dir * Complex64::from(eps),
        None => start,
    };
    let mut omega_h = omega * (-val(phase, z).im);
    let mut ds = (10.0 * eps).min(cfg.ds_max);
    let mut points = vec![z];

    for _ in 0..20_000 {
        // Termination on the current point.
        if !ascending && omega_h < cfg.morse_threshold {
            return Ok(Trace {
                points,
                stop: Stop::MorseFloor,
                end_omega_h: omega_h,
                crossing_re: None,
            });
        }
        if ascending && omega_h > -cfg.morse_threshold {
            return Ok(Trace {
                points,
                stop: Stop::Escaped,
                end_omega_h: omega_h,
                crossing_re: None,
            });
        }
        if z.re < cfg.re_range.0
            || z.re > cfg.re_range.1
            || z.im < cfg.im_range.0
            || z.im > cfg.im_range.1
        {
            return Ok(Trace {
                points,
                stop: Stop::WindowExit,
                end_omega_h: omega_h,
                crossing_re: None,
            });
        }

        // One adaptive predictor-corrector step.
        let (zn, omega_h_new) = loop {
            if ds < 1e-9 * (1.0 + z.norm()) {
                return stalled();
            }
            let Some(t0) = tangent(z) else {
                return stalled();
            };
            let Some(t1) = tangent(z + t0 * Complex64::from(0.5 * ds)) else {
                return stalled();
            };
            let mut zn = z + t1 * Complex64::from(ds);

            // Newton corrector back onto H = target, along grad H.
            let mut corrected = 0.0;
            let mut on_level = false;
            for _ in 0..5 {
                let residual = val(phase, zn).re - target;
                if residual.abs() < tol_target {
                    on_level = true;
                    break;
                }
                let g = grad(phase, zn);
                if g.norm() < 1e-13 {
                    return stalled();
                }
                let step = residual * g.conj() / g.norm_sqr();
                zn -= step;
                corrected += step.norm();
            }
            let omega_h_new = omega * (-val(phase, zn).im);
            let rough =
                !on_level || corrected > 0.2 * ds || (omega_h_new - omega_h).abs() > cfg.max_dh;
            if rough {
                ds *= 0.5;
                continue;
            }
            break (zn, omega_h_new);
        };

        let crossing_re = if ascending && (z.im > 0.0) != (zn.im > 0.0) {
            // Interpolate the real-axis crossing (the signs differ, so the
            // denominator cannot vanish).
            let t = z.im / (z.im - zn.im);
            Some(z.re + t * (zn.re - z.re))
        } else {
            None
        };
        z = zn;
        omega_h = omega_h_new;
        points.push(z);
        if crossing_re.is_some() {
            return Ok(Trace {
                points,
                stop: Stop::CrossedReal,
                end_omega_h: omega_h,
                crossing_re,
            });
        }
        ds = (1.5 * ds).min(cfg.ds_max);
    }
    Ok(Trace {
        points,
        stop: Stop::StepCap,
        end_omega_h: omega_h,
        crossing_re: None,
    })
}

// Gauss-Legendre nodes/weights on [-1, 1]; the 8-node rule (degree 15)
// carries the value, the 4-node rule (degree 7) the embedded error estimate.
// (Both verified against monomial integrals in the unit tests.)
const GL4_X: [f64; 4] = [
    -0.8611363115940526,
    -0.3399810435848563,
    0.3399810435848563,
    0.8611363115940526,
];
const GL4_W: [f64; 4] = [
    0.3478548451374538,
    0.6521451548625461,
    0.6521451548625461,
    0.3478548451374538,
];
const GL8_X: [f64; 8] = [
    -0.9602898564975363,
    -0.7966664774136267,
    -0.5255324099163290,
    -0.1834346424956498,
    0.1834346424956498,
    0.5255324099163290,
    0.7966664774136267,
    0.9602898564975363,
];
const GL8_W: [f64; 8] = [
    0.1012285362903763,
    0.2223810344533745,
    0.3137066458778873,
    0.3626837833783620,
    0.3626837833783620,
    0.3137066458778873,
    0.2223810344533745,
    0.1012285362903763,
];

/// Integrate amplitude(z) * e^{i omega phi} along a polyline with both
/// embedded rules.
fn integrate_polyline(
    phase: &dyn Phase<1>,
    omega: f64,
    points: &[Complex64],
    amplitude: &dyn Fn(Complex64) -> Complex64,
) -> (Complex64, Complex64) {
    let iw = Complex64::new(0.0, omega);
    let f = |z: Complex64| amplitude(z) * (iw * val(phase, z)).exp();
    let mut hi = Complex64::new(0.0, 0.0);
    let mut lo = Complex64::new(0.0, 0.0);
    for seg in points.windows(2) {
        let mid = (seg[0] + seg[1]) * Complex64::from(0.5);
        let half = (seg[1] - seg[0]) * Complex64::from(0.5);
        let mut s8 = Complex64::new(0.0, 0.0);
        for (x, w) in GL8_X.iter().zip(GL8_W) {
            s8 += f(mid + half * Complex64::from(*x)) * Complex64::from(w);
        }
        let mut s4 = Complex64::new(0.0, 0.0);
        for (x, w) in GL4_X.iter().zip(GL4_W) {
            s4 += f(mid + half * Complex64::from(*x)) * Complex64::from(w);
        }
        hi += half * s8;
        lo += half * s4;
    }
    (hi, lo)
}

/// 1D Fresnel-integral solver via constant-phase contours.
///
/// After [`ThimbleSolver::run`], the located saddles and the traced thimble
/// polylines remain available through [`ThimbleSolver::saddles`] and
/// [`ThimbleSolver::pieces`] (e.g. for plotting).
pub struct ThimbleSolver<'a> {
    phase: &'a dyn Phase<1>,
    cfg: ThimbleConfig,
    saddles: Vec<Saddle>,
    pieces: Vec<Vec<Complex64>>,
}

impl<'a> ThimbleSolver<'a> {
    pub fn new(phase: &'a dyn Phase<1>, cfg: ThimbleConfig) -> Self {
        Self {
            phase,
            cfg,
            saddles: Vec::new(),
            pieces: Vec::new(),
        }
    }

    pub fn saddles(&self) -> &[Saddle] {
        &self.saddles
    }

    /// Traced thimble polylines (one per effective saddle), each ordered
    /// "left to right" so that their sum is homotopic to the real axis.
    pub fn pieces(&self) -> &[Vec<Complex64>] {
        &self.pieces
    }

    pub fn run(&mut self) -> Result<Output> {
        if !(self.cfg.omega > 0.0) {
            return Err(Error::InvalidInput("omega must be > 0"));
        }
        let phase = self.phase;
        let cfg = &self.cfg;
        self.pieces.clear();
        self.saddles = find_saddles(
            phase,
            cfg.re_range,
            cfg.im_range,
            cfg.seeds_per_axis,
            cfg.newton_tol,
        );
        if self.saddles.is_empty() {
            return Err(Error::InvalidInput(
                "no saddle point found in the search window",
            ));
        }

        let mut hi = Complex64::new(0.0, 0.0);
        let mut lo = Complex64::new(0.0, 0.0);
        let mut extra_err = 0.0;
        for saddle in &self.saddles {
            // Near a double root Newton stalls at |z - z*| ~ sqrt(newton_tol),
            // where the numerical phi'' is ~ sqrt(newton_tol) * phi''' — so
            // sqrt(newton_tol) is the scale below which a "small" phi'' is
            // indistinguishable from a caustic. (Phases with a genuinely tiny
            // |phi''| at a saddle need a tighter `newton_tol`.)
            if saddle.second.norm() < 10.0 * cfg.newton_tol.sqrt() {
                return Err(Error::NotImplemented(
                    "degenerate saddle (caustic) in ThimbleSolver — use the \
                     mesh-flow Solver",
                ));
            }

            // Effectiveness: real saddles lie on the original contour;
            // complex saddles qualify iff an ascending branch reaches R.
            let is_real = saddle.z.im.abs() < 1e-9 * (1.0 + saddle.z.norm());
            let d_desc = take_off_direction(saddle.second);
            if !is_real {
                let d_asc = d_desc * Complex64::i();
                let mut effective = false;
                for dir in [d_asc, -d_asc] {
                    let trace =
                        trace_level_curve(phase, cfg, saddle.z, saddle.big_h(), Some(dir), true)?;
                    match trace.stop {
                        Stop::CrossedReal => {
                            effective = true;
                            break;
                        }
                        Stop::Escaped => {}
                        // Inconclusive ascent: count the saddle out but
                        // budget its largest possible contribution.
                        Stop::WindowExit | Stop::StepCap => {
                            extra_err += (cfg.omega * saddle.h()).exp();
                        }
                        Stop::MorseFloor => unreachable!("ascending trace"),
                    }
                }
                if !effective {
                    continue;
                }
            }

            // A thimble whose saddle already sits below the Morse floor
            // contributes less than e^{threshold}: skip, but budget it.
            if cfg.omega * saddle.h() < cfg.morse_threshold {
                extra_err += (cfg.omega * saddle.h()).exp();
                continue;
            }

            let plus =
                trace_level_curve(phase, cfg, saddle.z, saddle.big_h(), Some(d_desc), false)?;
            let minus =
                trace_level_curve(phase, cfg, saddle.z, saddle.big_h(), Some(-d_desc), false)?;
            for t in [&plus, &minus] {
                if t.stop != Stop::MorseFloor {
                    // Truncated before the integrand died: budget the tail.
                    extra_err += t.end_omega_h.exp();
                }
            }

            let mut piece: Vec<Complex64> = minus.points.iter().rev().copied().collect();
            piece.push(saddle.z);
            piece.extend(&plus.points);

            let (p_hi, p_lo) =
                integrate_polyline(phase, cfg.omega, &piece, &|_| Complex64::new(1.0, 0.0));
            hi += p_hi;
            lo += p_lo;
            self.pieces.push(piece);
        }

        Ok(Output {
            value: hi,
            error: (hi - lo).norm() + extra_err,
        })
    }
}

/// Half-line variant for the polar 2D wrapper: deform
/// $\int_0^\infty A(r)\,e^{i\omega\phi(r)}\,dr$ onto
/// (i) the flow line of the *fixed* endpoint $r = 0$ — which is itself the
/// level curve $H = \mathrm{Re}\,\phi(0)$, traced descending (when the
/// endpoint happens to be a stationary point, the single descending branch
/// with $\mathrm{Re} > 0$ take-off plays that role) — plus
/// (ii) the thimbles of the effective saddles, where effectiveness now
/// refers to the *positive* axis: real saddles need $\mathrm{Re}\,z > 0$
/// (the negative axis belongs to the $\theta + \pi$ radial problem), and
/// complex saddles need their ascending contour to cross $\mathbb{R}_{>0}$.
pub(crate) fn solve_half_line(
    phase: &dyn Phase<1>,
    cfg: &ThimbleConfig,
    amplitude: &dyn Fn(Complex64) -> Complex64,
) -> Result<Output> {
    if !(cfg.omega > 0.0) {
        return Err(Error::InvalidInput("omega must be > 0"));
    }
    let origin = Complex64::new(0.0, 0.0);
    let mut hi = Complex64::new(0.0, 0.0);
    let mut lo = Complex64::new(0.0, 0.0);
    let mut extra_err = 0.0;

    // (i) The endpoint piece, traversed from 0 outwards/downwards.
    let target0 = val(phase, origin).re;
    let g0 = grad(phase, origin);
    let endpoint_is_saddle = g0.norm() < 1e-7;
    let endpoint = if endpoint_is_saddle {
        let phi2 = second_derivative(phase, origin);
        if phi2.norm() < 10.0 * cfg.newton_tol.sqrt() {
            return Err(Error::NotImplemented(
                "degenerate stationary point at the radial endpoint r = 0 — \
                 use the mesh-flow Solver",
            ));
        }
        trace_level_curve(
            phase,
            cfg,
            origin,
            target0,
            Some(take_off_direction(phi2)),
            false,
        )?
    } else {
        trace_level_curve(phase, cfg, origin, target0, None, false)?
    };
    if endpoint.stop != Stop::MorseFloor {
        extra_err += endpoint.end_omega_h.exp();
    }
    let mut piece: Vec<Complex64> = if endpoint_is_saddle {
        vec![origin]
    } else {
        Vec::new()
    };
    piece.extend(&endpoint.points);
    let (p_hi, p_lo) = integrate_polyline(phase, cfg.omega, &piece, amplitude);
    hi += p_hi;
    lo += p_lo;

    // (ii) Thimbles of the effective saddles on the positive half-plane.
    let saddles = find_saddles(
        phase,
        cfg.re_range,
        cfg.im_range,
        cfg.seeds_per_axis,
        cfg.newton_tol,
    );
    for saddle in &saddles {
        if saddle.z.norm() < 1e-6 {
            continue; // the endpoint logic above owns the origin
        }
        if saddle.second.norm() < 10.0 * cfg.newton_tol.sqrt() {
            return Err(Error::NotImplemented(
                "degenerate saddle (caustic) in the radial problem — use the \
                 mesh-flow Solver",
            ));
        }
        let is_real = saddle.z.im.abs() < 1e-9 * (1.0 + saddle.z.norm());
        if is_real && saddle.z.re <= 0.0 {
            continue;
        }
        let d_desc = take_off_direction(saddle.second);
        if !is_real {
            let d_asc = d_desc * Complex64::i();
            let mut effective = false;
            for dir in [d_asc, -d_asc] {
                let trace =
                    trace_level_curve(phase, cfg, saddle.z, saddle.big_h(), Some(dir), true)?;
                match trace.stop {
                    Stop::CrossedReal => {
                        if trace.crossing_re.unwrap_or(f64::NEG_INFINITY) > 0.0 {
                            effective = true;
                            break;
                        }
                    }
                    Stop::Escaped => {}
                    Stop::WindowExit | Stop::StepCap => {
                        extra_err += (cfg.omega * saddle.h()).exp();
                    }
                    Stop::MorseFloor => unreachable!("ascending trace"),
                }
            }
            if !effective {
                continue;
            }
        }
        if cfg.omega * saddle.h() < cfg.morse_threshold {
            extra_err += (cfg.omega * saddle.h()).exp();
            continue;
        }

        let plus = trace_level_curve(phase, cfg, saddle.z, saddle.big_h(), Some(d_desc), false)?;
        let minus = trace_level_curve(phase, cfg, saddle.z, saddle.big_h(), Some(-d_desc), false)?;
        for t in [&plus, &minus] {
            if t.stop != Stop::MorseFloor {
                extra_err += t.end_omega_h.exp();
            }
        }
        let mut piece: Vec<Complex64> = minus.points.iter().rev().copied().collect();
        piece.push(saddle.z);
        piece.extend(&plus.points);
        let (p_hi, p_lo) = integrate_polyline(phase, cfg.omega, &piece, amplitude);
        hi += p_hi;
        lo += p_lo;
    }

    Ok(Output {
        value: hi,
        error: (hi - lo).norm() + extra_err,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_legendre_rules_are_correct() {
        // Sum of weights = length of [-1, 1].
        assert!((GL4_W.iter().sum::<f64>() - 2.0).abs() < 1e-14);
        assert!((GL8_W.iter().sum::<f64>() - 2.0).abs() < 1e-14);
        // Exactness on monomials up to the design degree, and a clean miss
        // one degree past it (odd degrees vanish by symmetry, so probe the
        // even degree above).
        let quad = |xs: &[f64], ws: &[f64], k: i32| -> f64 {
            xs.iter().zip(ws).map(|(x, w)| w * x.powi(k)).sum()
        };
        for k in 0..=7 {
            let exact = if k % 2 == 0 {
                2.0 / (k as f64 + 1.0)
            } else {
                0.0
            };
            assert!((quad(&GL4_X, &GL4_W, k) - exact).abs() < 1e-14, "GL4 x^{k}");
        }
        for k in 0..=15 {
            let exact = if k % 2 == 0 {
                2.0 / (k as f64 + 1.0)
            } else {
                0.0
            };
            assert!((quad(&GL8_X, &GL8_W, k) - exact).abs() < 1e-14, "GL8 x^{k}");
        }
        assert!((quad(&GL4_X, &GL4_W, 8) - 2.0 / 9.0).abs() > 1e-6);
        assert!((quad(&GL8_X, &GL8_W, 16) - 2.0 / 17.0).abs() > 1e-8);
    }

    #[test]
    fn take_off_directions() {
        // Quadratic phase phi'' = 1: thimble at +45 degrees.
        let d = take_off_direction(Complex64::new(1.0, 0.0));
        assert!((d - Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4)).norm() < 1e-12);
        // Airy upper saddle (phi'' = 2i): horizontal thimble.
        let d = take_off_direction(Complex64::new(0.0, 2.0));
        assert!((d - Complex64::new(1.0, 0.0)).norm() < 1e-12);
    }
}
