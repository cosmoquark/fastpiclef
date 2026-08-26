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
//! 1. find the saddles $\phi'(z) = 0$ — completely, from the phase's
//!    declared polynomial when it is a [`RationalPhase`]
//!    ([`saddles_from_polynomial`]); best-effort grid-Newton otherwise
//!    ([`find_saddles`]);
//! 2. weigh each saddle by its *signed traversal multiplicity*: real
//!    saddles count once (left-to-right); a complex saddle counts once per
//!    ascent crossing of $\mathbb{R}$ (Shi's effectiveness criterion), with
//!    the *orientation inherited through that crossing* — see
//!    `crossing_orientation`; the ascent can never touch $\mathbb{R}$ from
//!    $h > 0$, which silences the wrong member of a conjugate pair;
//! 3. trace each contributing thimble by a predictor-corrector that follows
//!    the level curve $H = H_i$ (midpoint predictor along $\mp\nabla h$,
//!    Newton corrector along $\nabla H$), stopping once
//!    $\omega h <$ `morse_threshold` (integrand dead);
//! 4. integrate $e^{i\omega\phi}$ along the traced polylines with embedded
//!    Gauss-Legendre rules (8- vs 4-node) for the error estimate, weighted
//!    by the multiplicities.
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

use crate::saddles::{Saddle, find_saddles, grad, saddles_from_polynomial, second_derivative, val};
use crate::{Error, Output, Phase, RationalPhase, Result, Solver, SolverConfig};
use nalgebra::SVector;
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
    /// Track `phi` by continuation of its (single-valued) gradient along each
    /// traced contour, instead of evaluating `phase.value` pointwise. Required
    /// for **multivalued** phases (e.g. the log potential of a point-mass
    /// slice): there `Re phi` is continuous but `Im phi` jumps by `m*pi` across
    /// the log branch cuts, which would otherwise stall the tracer (the jump
    /// trips `max_dh`) and corrupt the quadrature. A no-op for single-valued
    /// (rational/entire) phases, so it defaults to `false`.
    pub continuous_phase: bool,
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
            continuous_phase: false,
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
    /// Unit direction of the last accepted step. For [`Stop::CrossedReal`]
    /// this is the tangent with which the trace crossed the real axis —
    /// the datum that fixes the thimble's traversal orientation (see
    /// `crossing_orientation`).
    end_tangent: Complex64,
}

/// Signed multiplicity contributed by one ascent crossing of the original
/// contour, for a *complex* saddle.
///
/// The deformed contour inherits its traversal of the thimble by continuity
/// through the crossing point $a_i$: the ascent curve is the *stable*
/// manifold of the descent flow, so the $x > a_i$ side of $\mathbb{R}$
/// peels onto one specific descending branch. In the local conformal
/// coordinate $v$ (stable = imaginary axis, unstable = real axis,
/// orientation preserved), the $+\mathrm{Re}\,v$ branch is the *right*
/// side of the ascent traversed saddle$\to\mathbb{R}$; mapping back, the
/// $x > a_i$ exit branch is $-i\,a$ when the crossing tangent has
/// $\mathrm{Im} > 0$ and $+i\,a$ otherwise ($a$ = launch direction of the
/// traced ascent). The thimble piece is assembled with its "plus" branch
/// along `d_desc`, so the crossing contributes $+1$ if the exit branch
/// aligns with `d_desc` and $-1$ if it is the reversed traversal.
///
/// (For a *real* saddle the same continuity argument reduces to the
/// left-to-right rule already built into `take_off_direction`.)
fn crossing_orientation(ascent_dir: Complex64, end_tangent: Complex64, d_desc: Complex64) -> i32 {
    let plus_dir = if end_tangent.im > 0.0 {
        -Complex64::i() * ascent_dir
    } else {
        Complex64::i() * ascent_dir
    };
    if (plus_dir.conj() * d_desc).re > 0.0 {
        1
    } else {
        -1
    }
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
    let stalled = || Err(Error::TraceStalled);

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
    let mut last_dir = kick.unwrap_or(Complex64::new(0.0, 0.0));

    for _ in 0..20_000 {
        // Gradient at the current accepted point, for continuous-phase h
        // tracking (a single evaluation reused across ds-halving retries).
        let g_z = if cfg.continuous_phase {
            grad(phase, z)
        } else {
            Complex64::new(0.0, 0.0)
        };
        // Termination on the current point.
        if !ascending && omega_h < cfg.morse_threshold {
            return Ok(Trace {
                points,
                stop: Stop::MorseFloor,
                end_omega_h: omega_h,
                crossing_re: None,
                end_tangent: last_dir,
            });
        }
        if ascending && omega_h > -cfg.morse_threshold {
            return Ok(Trace {
                points,
                stop: Stop::Escaped,
                end_omega_h: omega_h,
                crossing_re: None,
                end_tangent: last_dir,
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
                end_tangent: last_dir,
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
            // h = -Im phi. For a multivalued phase, evaluating `val` pointwise
            // makes h jump by `omega*m*pi` across a log cut; instead advance it
            // by the (single-valued) gradient, dphi = trapezoid(phi') * dz.
            let omega_h_new = if cfg.continuous_phase {
                let dphi = (g_z + grad(phase, zn)) * Complex64::from(0.5) * (zn - z);
                omega_h - omega * dphi.im
            } else {
                omega * (-val(phase, zn).im)
            };
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
        let step = zn - z;
        if step.norm() > 0.0 {
            last_dir = step / Complex64::from(step.norm());
        }
        z = zn;
        omega_h = omega_h_new;
        points.push(z);
        if crossing_re.is_some() {
            return Ok(Trace {
                points,
                stop: Stop::CrossedReal,
                end_omega_h: omega_h,
                crossing_re,
                end_tangent: last_dir,
            });
        }
        ds = (1.5 * ds).min(cfg.ds_max);
    }
    Ok(Trace {
        points,
        stop: Stop::StepCap,
        end_omega_h: omega_h,
        crossing_re: None,
        end_tangent: last_dir,
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
///
/// With `continuous = false`, `phi` is `phase.value` evaluated pointwise. With
/// `continuous = true` (multivalued phases), `phi` is instead continued along
/// the polyline from the `anchor` vertex — where `phase.value` is on the
/// physical branch (the saddle, which sits on the real axis for a real saddle)
/// — by integrating the single-valued gradient. This keeps `e^{i omega phi}`
/// on one sheet across the log branch cuts the polyline may cross.
fn integrate_polyline(
    phase: &dyn Phase<1>,
    omega: f64,
    points: &[Complex64],
    amplitude: &dyn Fn(Complex64) -> Complex64,
    continuous: bool,
    anchor: usize,
) -> (Complex64, Complex64) {
    let iw = Complex64::new(0.0, omega);
    let half_c = Complex64::from(0.5);

    // Line integral of the (single-valued) gradient along the straight segment
    // [a, b], by the embedded 8-node rule — exact enough that continued `phi`
    // matches the pointwise value away from cuts.
    let grad_integral = |a: Complex64, b: Complex64| -> Complex64 {
        let mid = (a + b) * half_c;
        let half = (b - a) * half_c;
        let mut s = Complex64::new(0.0, 0.0);
        for (x, w) in GL8_X.iter().zip(GL8_W) {
            s += grad(phase, mid + half * Complex64::from(*x)) * Complex64::from(w);
        }
        s * half
    };

    // phi at each vertex, continued outward from the anchor (only used in
    // continuous mode).
    let phi_vert: Vec<Complex64> = if continuous {
        let n = points.len();
        let mut v = vec![Complex64::new(0.0, 0.0); n];
        let a = anchor.min(n.saturating_sub(1));
        v[a] = val(phase, points[a]);
        for i in (0..a).rev() {
            v[i] = v[i + 1] - grad_integral(points[i], points[i + 1]);
        }
        for i in (a + 1)..n {
            v[i] = v[i - 1] + grad_integral(points[i - 1], points[i]);
        }
        v
    } else {
        Vec::new()
    };

    let mut hi = Complex64::new(0.0, 0.0);
    let mut lo = Complex64::new(0.0, 0.0);
    for (k, seg) in points.windows(2).enumerate() {
        let mid = (seg[0] + seg[1]) * half_c;
        let half = (seg[1] - seg[0]) * half_c;
        let f = |z: Complex64| -> Complex64 {
            let phi = if continuous {
                phi_vert[k] + grad_integral(seg[0], z)
            } else {
                val(phase, z)
            };
            amplitude(z) * (iw * phi).exp()
        };
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

/// Widen the tracing window to cover every found saddle (with margin).
/// The complete polynomial source may legitimately return saddles outside
/// the configured window — grid-Newton never did, it filtered — and a trace
/// must never *start* out of bounds, or its error would be mis-budgeted.
/// The saddle-search semantics are unchanged; only the trace bounds grow.
fn extend_window(cfg: &ThimbleConfig, saddles: &[Saddle]) -> ThimbleConfig {
    let mut out = cfg.clone();
    let margin = 2.0;
    for s in saddles {
        out.re_range.0 = out.re_range.0.min(s.z.re - margin);
        out.re_range.1 = out.re_range.1.max(s.z.re + margin);
        out.im_range.0 = out.im_range.0.min(s.z.im - margin);
        out.im_range.1 = out.im_range.1.max(s.z.im + margin);
    }
    out
}

/// 1D Fresnel-integral solver via constant-phase contours.
///
/// Construct with [`ThimbleSolver::new`] for a general analytic phase
/// (grid-Newton saddle search, best-effort), or with
/// [`ThimbleSolver::new_rational`] for a [`RationalPhase`] (complete
/// polynomial saddle source — required for multi-image configurations,
/// where a missed saddle silently corrupts the result).
///
/// After [`ThimbleSolver::run`], the located saddles and the traced thimble
/// polylines remain available through [`ThimbleSolver::saddles`] and
/// [`ThimbleSolver::pieces`] (e.g. for plotting).
pub struct ThimbleSolver<'a> {
    phase: &'a dyn Phase<1>,
    cfg: ThimbleConfig,
    /// `Some(N)` = complete saddle source from the roots of N (rational
    /// phases); `None` = grid-Newton search.
    polynomial: Option<Vec<Complex64>>,
    saddles: Vec<Saddle>,
    pieces: Vec<Vec<Complex64>>,
}

impl<'a> ThimbleSolver<'a> {
    pub fn new(phase: &'a dyn Phase<1>, cfg: ThimbleConfig) -> Self {
        Self {
            phase,
            cfg,
            polynomial: None,
            saddles: Vec::new(),
            pieces: Vec::new(),
        }
    }

    /// Like [`ThimbleSolver::new`], but with the complete saddle source a
    /// [`RationalPhase`] enables: every stationary point is a root of the
    /// declared polynomial, found by Durand-Kerner and polished against the
    /// phase's own gradient.
    pub fn new_rational<P: RationalPhase>(phase: &'a P, cfg: ThimbleConfig) -> Self {
        let polynomial = Some(phase.stationary_polynomial());
        Self {
            phase,
            cfg,
            polynomial,
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
        self.pieces.clear();
        self.saddles = match &self.polynomial {
            Some(coeffs) => saddles_from_polynomial(phase, coeffs, self.cfg.newton_tol)?,
            None => find_saddles(
                phase,
                self.cfg.re_range,
                self.cfg.im_range,
                self.cfg.seeds_per_axis,
                self.cfg.newton_tol,
            ),
        };
        if self.saddles.is_empty() {
            return Err(Error::InvalidInput(
                "no saddle point found in the search window",
            ));
        }
        let cfg = &extend_window(&self.cfg, &self.saddles);

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
                return Err(Error::DegenerateSaddle);
            }

            // Effectiveness and traversal multiplicity. A real saddle lies
            // on the original contour and is crossed once, left-to-right. A
            // complex saddle enters once per ascent crossing of R, each with
            // the orientation inherited through that crossing (see
            // `crossing_orientation`); the signed sum is its multiplicity
            // (0 = not effective, and crossings of opposite orientation
            // cancel, as they must).
            let is_real = saddle.z.im.abs() < 1e-9 * (1.0 + saddle.z.norm());
            let d_desc = take_off_direction(saddle.second);
            let mut multiplicity: i32 = if is_real { 1 } else { 0 };
            if !is_real {
                let d_asc = d_desc * Complex64::i();
                for dir in [d_asc, -d_asc] {
                    let trace =
                        trace_level_curve(phase, cfg, saddle.z, saddle.big_h(), Some(dir), true)?;
                    match trace.stop {
                        Stop::CrossedReal => {
                            multiplicity += crossing_orientation(dir, trace.end_tangent, d_desc);
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
                if multiplicity == 0 {
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

            let (p_hi, p_lo) = integrate_polyline(
                phase,
                cfg.omega,
                &piece,
                &|_| Complex64::new(1.0, 0.0),
                cfg.continuous_phase,
                minus.points.len(),
            );
            let m = Complex64::from(multiplicity as f64);
            hi += p_hi * m;
            lo += p_lo * m;
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
    given_saddles: Option<&[Saddle]>,
) -> Result<Output> {
    if !(cfg.omega > 0.0) {
        return Err(Error::InvalidInput("omega must be > 0"));
    }
    let origin = Complex64::new(0.0, 0.0);
    let mut hi = Complex64::new(0.0, 0.0);
    let mut lo = Complex64::new(0.0, 0.0);
    let mut extra_err = 0.0;

    // Saddles first (injected by the caller — e.g. the polar wrapper's
    // per-theta polynomial source — or grid-Newton), so the trace window
    // can be extended around them before anything is traced.
    let saddles: Vec<Saddle> = match given_saddles {
        Some(s) => s.to_vec(),
        None => find_saddles(
            phase,
            cfg.re_range,
            cfg.im_range,
            cfg.seeds_per_axis,
            cfg.newton_tol,
        ),
    };
    let cfg = &extend_window(cfg, &saddles);

    // (i) The endpoint piece, traversed from 0 outwards/downwards. Skipped when
    // the origin is a singularity of the phase (e.g. the point-mass log branch
    // point psi = ln r, where phi -> +inf): there the polar integrand
    // ~ r^{1-i*omega} vanishes, so the lower limit contributes nothing and the
    // saddle thimble(s) below carry the whole half-line integral. (For a regular
    // phase the origin is an ordinary endpoint whose contour must be included.)
    let v0 = val(phase, origin);
    let g0 = grad(phase, origin);
    let origin_regular =
        v0.re.is_finite() && v0.im.is_finite() && g0.re.is_finite() && g0.im.is_finite();
    if origin_regular {
        let target0 = v0.re;
        let endpoint_is_saddle = g0.norm() < 1e-7;
        let endpoint = if endpoint_is_saddle {
            let phi2 = second_derivative(phase, origin);
            if phi2.norm() < 10.0 * cfg.newton_tol.sqrt() {
                return Err(Error::DegenerateSaddle);
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
        let (p_hi, p_lo) =
            integrate_polyline(phase, cfg.omega, &piece, amplitude, cfg.continuous_phase, 0);
        hi += p_hi;
        lo += p_lo;
    }

    // (ii) Thimbles of the effective saddles on the positive half-plane.
    for saddle in &saddles {
        if saddle.z.norm() < 1e-6 {
            continue; // the endpoint logic above owns the origin
        }
        if saddle.second.norm() < 10.0 * cfg.newton_tol.sqrt() {
            return Err(Error::DegenerateSaddle);
        }
        let is_real = saddle.z.im.abs() < 1e-9 * (1.0 + saddle.z.norm());
        if is_real && saddle.z.re <= 0.0 {
            continue;
        }
        let d_desc = take_off_direction(saddle.second);
        // Signed multiplicity, as in the full-line solver, but counting
        // only ascent crossings of the *positive* axis.
        let mut multiplicity: i32 = if is_real { 1 } else { 0 };
        if !is_real {
            let d_asc = d_desc * Complex64::i();
            for dir in [d_asc, -d_asc] {
                let trace =
                    trace_level_curve(phase, cfg, saddle.z, saddle.big_h(), Some(dir), true)?;
                match trace.stop {
                    Stop::CrossedReal => {
                        if trace.crossing_re.unwrap_or(f64::NEG_INFINITY) > 0.0 {
                            multiplicity += crossing_orientation(dir, trace.end_tangent, d_desc);
                        }
                    }
                    Stop::Escaped => {}
                    Stop::WindowExit | Stop::StepCap => {
                        extra_err += (cfg.omega * saddle.h()).exp();
                    }
                    Stop::MorseFloor => unreachable!("ascending trace"),
                }
            }
            if multiplicity == 0 {
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
        let (p_hi, p_lo) = integrate_polyline(
            phase,
            cfg.omega,
            &piece,
            amplitude,
            cfg.continuous_phase,
            minus.points.len(),
        );
        let m = Complex64::from(multiplicity as f64);
        hi += p_hi * m;
        lo += p_lo * m;
    }

    Ok(Output {
        value: hi,
        error: (hi - lo).norm() + extra_err,
    })
}

/// Radial phase of the isolated point mass after the substitution $r = e^u$,
/// which maps the half-line $(0, \infty)$ to the full line and unwraps the
/// logarithmic branch point at $r = 0$ (to $u \to -\infty$), leaving an
/// *entire* phase — no branch cut for the trace to stall on:
/// $\Psi(u) = \tfrac12 e^{2u} - b\,e^u + \tfrac12 y^2 - u$, with
/// $b = y\cos\theta$. The radial integral
/// $\int_0^\infty e^{i\omega\phi_\theta(r)}\,r\,dr$ then equals
/// $\int_{\mathbb R} e^{i\omega\Psi(u)}\,e^{2u}\,du$ (the $e^{2u}$ is the
/// $r\,dr$ Jacobian, carried as the quadrature amplitude).
struct PointMassRadialU {
    b: f64,
    half_y2: f64,
}

impl Phase<1> for PointMassRadialU {
    fn value(&self, u: &SVector<Complex64, 1>) -> Complex64 {
        let e = u[0].exp();
        0.5 * e * e - self.b * e + self.half_y2 - u[0]
    }

    fn gradient(&self, u: &SVector<Complex64, 1>) -> SVector<Complex64, 1> {
        let e = u[0].exp();
        SVector::<Complex64, 1>::new(e * e - self.b * e - Complex64::from(1.0))
    }
}

/// Radial integral $\int_0^\infty e^{i\omega\phi_\theta(r)}\,r\,dr$ for the
/// isolated point mass at impact parameter `y` and angle with `cos_t`, via the
/// branch-cut-free $r = e^u$ substitution ([`PointMassRadialU`]). Robust at all
/// $\omega$, unlike the direct [`solve_half_line`] trace, which crosses the
/// $\ln$ cut on the imaginary $r$-axis (and stalls on the $\mathrm{Im}\,\phi$
/// jump there) at low $\omega$. The single real $u$-saddle is $u_s = \ln r_+$
/// with $r_+ = (b + \sqrt{b^2 + 4})/2 > 0$ the positive image; the negative
/// image belongs to the $\theta + \pi$ radial problem. This is the reusable
/// log-thimble kernel earmarked for a future general 2D solver.
pub(crate) fn point_mass_radial(y: f64, cos_t: f64, cfg: &ThimbleConfig) -> Result<Output> {
    let b = y * cos_t;
    let phase = PointMassRadialU {
        b,
        half_y2: 0.5 * y * y,
    };
    let amplitude = |u: Complex64| (2.0 * u).exp();

    // Psi'(u) = e^{2u} - b e^u - 1 vanishes where e^u = rho, rho = (b +- sqrt(b^2+4))/2.
    // rho_+ > 0 gives the real saddle u_+ = ln(rho_+) (the bright image); rho_- < 0
    // gives the conjugate pair u_- = ln|rho_-| +- i*pi. The u_- saddles are
    // sub-dominant (larger Im Psi) and exponentially suppressed at large omega
    // -- negligible in the geometric-optics regime, but their contribution is
    // needed at low omega. (Nearest copies only; the e^u periodicity puts the
    // rest at Im = +-2*pi, past intervening valleys.)
    let disc = (b * b + 4.0).sqrt();
    let ln_abs_rho_minus = (0.5 * (disc - b)).ln(); // ln|rho_-|
    let mk = |u: Complex64| -> Saddle {
        Saddle {
            z: u,
            value: val(&phase, u),
            second: second_derivative(&phase, u),
        }
    };
    let saddles = [
        mk(Complex64::from((0.5 * (b + disc)).ln())),
        mk(Complex64::new(ln_abs_rho_minus, std::f64::consts::PI)),
        mk(Complex64::new(ln_abs_rho_minus, -std::f64::consts::PI)),
    ];
    integrate_thimbles(&phase, cfg, &amplitude, &saddles)
}

/// Integrate `int_R amplitude(u) e^{i omega phi(u)} du` over the thimbles of the
/// given (analytically known) saddles, with the standard effectiveness /
/// multiplicity bookkeeping: a real saddle sits on the contour and is crossed
/// once; a complex saddle enters once per signed ascent crossing of R (opposite
/// orientations cancel), and one already below the Morse floor is budgeted and
/// skipped. The full-line counterpart of [`solve_half_line`], used where the
/// complete saddle set is supplied rather than found (the point-mass u-phase,
/// and the slice solver's Hankel branch-cut rays).
pub(crate) fn integrate_thimbles(
    phase: &dyn Phase<1>,
    cfg: &ThimbleConfig,
    amplitude: &dyn Fn(Complex64) -> Complex64,
    saddles: &[Saddle],
) -> Result<Output> {
    let cfg = &extend_window(cfg, saddles);
    let mut hi = Complex64::new(0.0, 0.0);
    let mut lo = Complex64::new(0.0, 0.0);
    let mut extra_err = 0.0;
    for saddle in saddles {
        if saddle.second.norm() < 10.0 * cfg.newton_tol.sqrt() {
            return Err(Error::DegenerateSaddle);
        }
        let is_real = saddle.z.im.abs() < 1e-9 * (1.0 + saddle.z.norm());
        let d_desc = take_off_direction(saddle.second);
        let mut multiplicity: i32 = if is_real { 1 } else { 0 };
        if !is_real {
            let d_asc = d_desc * Complex64::i();
            for dir in [d_asc, -d_asc] {
                let trace =
                    trace_level_curve(phase, cfg, saddle.z, saddle.big_h(), Some(dir), true)?;
                match trace.stop {
                    Stop::CrossedReal => {
                        multiplicity += crossing_orientation(dir, trace.end_tangent, d_desc);
                    }
                    Stop::Escaped => {}
                    Stop::WindowExit | Stop::StepCap => {
                        extra_err += (cfg.omega * saddle.h()).exp();
                    }
                    Stop::MorseFloor => unreachable!("ascending trace"),
                }
            }
            if multiplicity == 0 {
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
        let (p_hi, p_lo) = integrate_polyline(
            phase,
            cfg.omega,
            &piece,
            amplitude,
            cfg.continuous_phase,
            minus.points.len(),
        );
        let m = Complex64::from(multiplicity as f64);
        hi += p_hi * m;
        lo += p_lo * m;
    }
    Ok(Output {
        value: hi,
        error: (hi - lo).norm() + extra_err,
    })
}

/// The thimble integral of a *single* saddle — both descending branches, unit
/// multiplicity, no effectiveness gating. For diagnosing which sub-dominant
/// saddles a multivalued integrand makes effective (the slice solver's
/// swept-branch-point contribution shows up as one of these).
pub(crate) fn saddle_thimble(
    phase: &dyn Phase<1>,
    cfg: &ThimbleConfig,
    saddle: &Saddle,
    amplitude: &dyn Fn(Complex64) -> Complex64,
) -> Result<Complex64> {
    let cfg = &extend_window(cfg, std::slice::from_ref(saddle));
    let d_desc = take_off_direction(saddle.second);
    let plus = trace_level_curve(phase, cfg, saddle.z, saddle.big_h(), Some(d_desc), false)?;
    let minus = trace_level_curve(phase, cfg, saddle.z, saddle.big_h(), Some(-d_desc), false)?;
    let mut piece: Vec<Complex64> = minus.points.iter().rev().copied().collect();
    piece.push(saddle.z);
    piece.extend(&plus.points);
    let (hi, _) = integrate_polyline(
        phase,
        cfg.omega,
        &piece,
        amplitude,
        cfg.continuous_phase,
        minus.points.len(),
    );
    Ok(hi)
}

/// Which solver produced a [`solve_1d_hybrid`] result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveMethod {
    /// The fast constant-phase contour solver succeeded.
    Thimble,
    /// The thimble solver hit a configuration it cannot treat uniformly
    /// (a caustic or a Stokes-line/pole stall) and the robust mesh-flow
    /// solver was used instead.
    MeshFallback,
}

/// Solve the 1D Fresnel integral with the fast thimble solver, falling back
/// to the mesh-flow [`Solver`] when the thimble solver reports a structural
/// difficulty it has no uniform treatment for — a degenerate saddle
/// ([`Error::DegenerateSaddle`], i.e. a caustic) or a stalled trace
/// ([`Error::TraceStalled`], i.e. a Stokes-line configuration). This is the
/// recommended way to "route around" those two failure modes rather than
/// attempt their (research-grade) uniform treatment in the thimble solver.
///
/// Note the fallback's scope: it is the right move for caustics and Stokes
/// lines, where the thimble solver fails *and the mesh solver is robust*. It
/// is **not** a remedy for the strong-lens/near-pole regime, where the roles
/// reverse — there the mesh solver is the fragile one and the thimble result
/// should be preferred. The thimble solver does not raise an error in that
/// regime, so this fallback is never triggered by it.
pub fn solve_1d_hybrid(
    phase: &dyn Phase<1>,
    thimble_cfg: ThimbleConfig,
    mesh_cfg: SolverConfig,
) -> Result<(Output, SolveMethod)> {
    match ThimbleSolver::new(phase, thimble_cfg).run() {
        Ok(out) => Ok((out, SolveMethod::Thimble)),
        Err(Error::DegenerateSaddle | Error::TraceStalled) => {
            let out = Solver::<1>::new(phase, mesh_cfg).run()?;
            Ok((out, SolveMethod::MeshFallback))
        }
        Err(e) => Err(e),
    }
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
