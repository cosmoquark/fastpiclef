//! Mesh-free connected descent-deformation of a 1D contour — the correct inner
//! kernel for the multivalued log potential (Path 2 / M3A).
//!
//! This is Feldbrugge's "Algorithm 1" (arXiv:1909.04632, reference C++ in
//! `refs/P-LPI/code_1D/`) specialised to our slice phase: deform the real line as
//! a connected polyline under the descent flow, and each step (1) **flow** every
//! active vertex down $\dot z = i\,\overline{\phi'(z)}$, (2) **prune** any vertex
//! whose integrand has died ($\omega h <$ `morse_threshold`), and (3) **insert**
//! midpoints on over-long segments. A fixed number of steps is run; the prune is
//! what keeps an already-converged contour stable (no early-stop needed).
//!
//! Where the disconnected saddle-sum ([`crate::ThimbleSolver`]) misses the
//! branch-cut contribution on a swept slice, this connected contour **wraps** the
//! branch point automatically: the h-monotonicity guard freezes a vertex at the
//! pinch rather than letting it tunnel across the cut.
//!
//! We keep three things that improve on the P-LPI reference for the multivalued
//! `ln`: RK4 + h-guard (vs plain Euler), the **analytic single-valued gradient**
//! (vs finite differences that straddle cuts), and **continuous-phase** GL
//! integration (vs principal-branch Romberg). The flow conserves $\mathrm{Re}\,
//! \phi$ and drives $\mathrm{Im}\,\phi\to+\infty$, so the ends sink into the
//! valleys and are pruned.

use crate::phase::Phase;
use crate::saddles::{grad, val};
use crate::thimble::integrate_polyline;
use num_complex::Complex64;

const H_RISE_TOL: f64 = 1e-9;
const MIN_SUBSTEP_FRAC: f64 = 6e-5;

/// Knobs for the connected descent tracer.
#[derive(Debug, Clone)]
pub struct FlowContourConfig {
    /// Initial real-line window `[lo, hi]`; its ends flow into the valleys and
    /// are pruned. Wide enough that the truncated tail is exponentially small.
    pub domain: (f64, f64),
    /// Descent step (an upper bound; the guard sub-steps below it).
    pub dt: f64,
    /// Number of descent steps (a fixed count, P-LPI style — prune keeps an
    /// already-converged contour from over-flowing, so this only needs to be
    /// large enough for the slowest slice).
    pub steps: usize,
    /// Prune a vertex once `omega * h < morse_threshold` (integrand negligible).
    pub morse_threshold: f64,
    /// Speed cap for the descent field (as in the mesh flow).
    pub max_speed: f64,
    /// Split any active segment longer than this (adaptive density).
    pub max_seg: f64,
    /// Frequency in `e^{i omega phi}`.
    pub omega: f64,
}

impl Default for FlowContourConfig {
    fn default() -> Self {
        Self {
            domain: (-10.0, 10.0),
            dt: 0.05,
            steps: 100,
            morse_threshold: -30.0,
            max_speed: 10.0,
            max_seg: 0.25,
            omega: 1.0,
        }
    }
}

#[derive(Clone, Copy)]
struct Vertex {
    z: Complex64,
    active: bool,
}

/// Descent field `v = i * conj(phi')`, componentwise `i*conj(a+ib) = b+ia`,
/// saturated at `cap` (positive rescaling preserves `h`-monotonicity).
fn velocity(phase: &dyn Phase<1>, z: Complex64, cap: f64) -> Complex64 {
    let g = grad(phase, z);
    let mut v = Complex64::new(g.im, g.re);
    let s = v.norm();
    if s > cap {
        v *= cap / s;
    }
    v
}

fn rk4_dz(phase: &dyn Phase<1>, z: Complex64, step: f64, cap: f64) -> Complex64 {
    let h = 0.5 * step;
    let f = step;
    let k1 = velocity(phase, z, cap);
    let k2 = velocity(phase, z + k1 * h, cap);
    let k3 = velocity(phase, z + k2 * h, cap);
    let k4 = velocity(phase, z + k3 * f, cap);
    (k1 + (k2 + k3) * 2.0 + k4) * (step / 6.0)
}

fn h_of(phase: &dyn Phase<1>, z: Complex64) -> f64 {
    -val(phase, z).im
}

/// Advance one vertex over the full step `dt` with h-monotone halving sub-steps;
/// freeze once a sub-step underflows (a pole/branch-point pinch → the wrap).
fn guarded_advance(phase: &dyn Phase<1>, z0: Complex64, dt: f64, cap: f64, min_substep: f64) -> Complex64 {
    let mut z = z0;
    let mut remaining = dt;
    let mut sub = dt;
    let mut iters = 0;
    while remaining > 1e-9 * dt && iters < 128 {
        iters += 1;
        let step = sub.min(remaining);
        let h_old = h_of(phase, z);
        let dz = rk4_dz(phase, z, step, cap);
        let cand = z + dz;
        let ok = cand.re.is_finite()
            && cand.im.is_finite()
            && h_of(phase, cand) <= h_old + H_RISE_TOL * (1.0 + h_old.abs());
        if ok {
            z = cand;
            remaining -= step;
            sub = (sub * 2.0).min(remaining);
        } else if step <= min_substep {
            break;
        } else {
            sub *= 0.5;
        }
    }
    z
}

/// Insert an active midpoint on every active–active segment longer than
/// `max_seg` (one pass). Inactive vertices break the polyline, so a segment
/// touching an inactive vertex is never split. A midpoint that lands close to a
/// singularity (`|phi'|` huge) is **not** inserted — otherwise the tight wrap of
/// a near-axis branch point spawns an unbounded cascade of near-singular points.
fn split_segments(phase: &dyn Phase<1>, verts: &mut Vec<Vertex>, max_seg: f64, grad_cap: f64) {
    let mut out = Vec::with_capacity(verts.len() + 8);
    for w in verts.windows(2) {
        out.push(w[0]);
        if w[0].active && w[1].active && (w[1].z - w[0].z).norm() > max_seg {
            let mid = (w[0].z + w[1].z) * Complex64::from(0.5);
            if grad(phase, mid).norm() < grad_cap {
                out.push(Vertex { z: mid, active: true });
            }
        }
    }
    if let Some(&last) = verts.last() {
        out.push(last);
    }
    *verts = out;
}

/// Integrate over each maximal run of consecutive active vertices (the contour
/// may fragment where it dips below the floor between saddles), summing the
/// runs. Returns `(value, error estimate)`.
fn integrate_runs(phase: &dyn Phase<1>, omega: f64, verts: &[Vertex]) -> (Complex64, f64) {
    let amp = |_: Complex64| Complex64::new(1.0, 0.0);
    let mut hi = Complex64::new(0.0, 0.0);
    let mut lo = Complex64::new(0.0, 0.0);
    let mut i = 0;
    while i < verts.len() {
        if !verts[i].active {
            i += 1;
            continue;
        }
        let start = i;
        while i < verts.len() && verts[i].active {
            i += 1;
        }
        let run: Vec<Complex64> = verts[start..i].iter().map(|v| v.z).collect();
        if run.len() < 2 {
            continue;
        }
        // Anchor the continuous-phase continuation at the least-sunk vertex
        // (min |Im phi|, nearest the real axis / a saddle) where `phase.value`
        // is reliably on the physical branch.
        let anchor = (0..run.len())
            .min_by(|&a, &b| {
                val(phase, run[a])
                    .im
                    .abs()
                    .partial_cmp(&val(phase, run[b]).im.abs())
                    .unwrap()
            })
            .unwrap_or(0);
        let (rhi, rlo) = integrate_polyline(phase, omega, &run, &amp, true, anchor);
        hi += rhi;
        lo += rlo;
    }
    (hi, (hi - lo).norm())
}

/// Integrate `int_R e^{i omega phi} dz` by deforming the real line to its
/// connected descent contour (Feldbrugge Algorithm 1). Returns
/// `(value, error_estimate)`; the value is the **raw** integral (no `free_space`).
pub fn flow_contour_integral(phase: &dyn Phase<1>, cfg: &FlowContourConfig) -> (Complex64, f64) {
    let (lo, hi) = cfg.domain;
    let mut verts = vec![
        Vertex { z: Complex64::new(lo, 0.0), active: true },
        Vertex { z: Complex64::new(hi, 0.0), active: true },
    ];
    let grad_cap = 100.0;
    let max_pts = 20_000;
    // Initial resolution: split until every segment is within max_seg.
    while verts.len() < max_pts
        && verts
            .windows(2)
            .any(|w| (w[1].z - w[0].z).norm() > cfg.max_seg)
    {
        split_segments(phase, &mut verts, cfg.max_seg, grad_cap);
    }

    let cap = cfg.max_speed;
    let min_sub = cfg.dt * MIN_SUBSTEP_FRAC;
    for _ in 0..cfg.steps {
        for v in verts.iter_mut() {
            if !v.active {
                continue;
            }
            // Prune: integrand negligible here (or sunk into a pole valley).
            if cfg.omega * h_of(phase, v.z) < cfg.morse_threshold {
                v.active = false;
                continue;
            }
            v.z = guarded_advance(phase, v.z, cfg.dt, cap, min_sub);
        }
        if verts.len() < max_pts {
            split_segments(phase, &mut verts, cfg.max_seg, grad_cap);
        }
    }
    if std::env::var("FLOW_DEBUG").is_ok() {
        let n = verts.iter().filter(|v| v.active).count();
        eprintln!("flow_contour: {n} active / {} total vertices", verts.len());
    }
    integrate_runs(phase, cfg.omega, &verts)
}
