use crate::{Error, Mesh, Phase, Result};
use nalgebra::SVector;
use num_complex::Complex64;
use rayon::prelude::*;

/// Per-step diagnostics returned by [`Flow::step`].
#[derive(Debug, Clone, Copy)]
pub struct StepReport {
    /// Largest vertex displacement |Δz| over the active vertices.
    pub max_step: f64,
    /// Mean of the Morse function $h = -\mathrm{Im}\,\phi$ over the active
    /// vertices after the step (ω-free; the solver folds in ω where needed).
    /// `-∞` when no vertex is active.
    pub mean_morse: f64,
}

/// Descent integrator that moves every mesh point one step along the
/// Picard-Lefshetz downward flow of $h = \mathrm{Re}(i\phi) = -\mathrm{Im}\,\phi$.
///
/// Writing $i\phi = h + iH$, the steepest-descent field of $h$ in each
/// complex coordinate is $\dot z_j = -\overline{\partial_j (i\phi)}
/// = i\,\overline{\partial_j \phi}$. Along the flow $h$ decreases
/// monotonically ($\dot h = -|\nabla\phi|^2$) while $H = \mathrm{Re}\,\phi$
/// is exactly conserved, so the contour sinks towards the Lefshetz thimbles
/// without changing the value of the integral (Cauchy).
pub trait Flow<const D: usize> {
    fn step<P: Phase<D> + ?Sized>(&mut self, mesh: &mut Mesh<D>, phase: &P) -> Result<StepReport>;
}

/// Fixed-step classical RK4 integrator on the speed-capped descent field,
/// with a monotonicity guard against pole overshoot.
///
/// *Speed cap.* The raw field $i\,\overline{\nabla\phi}$ can grow
/// superlinearly in $|z|$ (already $|z|^2$ for a cubic phase), in which case
/// outer trajectories reach infinity in *finite* flow time and any
/// fixed-step integrator overflows. We saturate the speed at `max_speed`:
/// rescaling the field by a positive scalar preserves its integral curves —
/// hence the deformed contour and the value of the integral — and only
/// slows traversal; the slowed far-out simplices are exactly the ones
/// pruning removes. (Unlike a unit normalisation $v/|v|$, saturation leaves
/// the field untouched near saddles, where the natural slowdown prevents
/// overshooting the thimble.)
///
/// *Monotonicity guard (near-pole robustness).* Along the true flow
/// $\dot h = -|\nabla\phi|^2 \le 0$, and the positive speed rescaling
/// preserves that sign, so `h` is monotone non-increasing on every
/// trajectory. A fixed-step move that *raises* `h` has therefore tunnelled
/// numerically across a pole onto the $h\to+\infty$ side, where the
/// integrand $e^{\omega h}$ blows up. We refuse such moves: each vertex is
/// advanced over the step interval by RK4 sub-steps that are halved
/// whenever the candidate raises `h`, and frozen at its last good position
/// if even the smallest sub-step still would (it has reached the pole-pinch,
/// where the thimble genuinely terminates). On phases without poles `h`
/// strictly decreases, the full step is always accepted, and the result is
/// bit-identical to a plain fixed-step RK4.
#[derive(Debug, Clone, Copy)]
pub struct Rk4 {
    pub dt: f64,
    pub max_speed: f64,
}

/// Tolerance on an `h` increase before a sub-step is rejected. A pole
/// overshoot raises `h` by O(1) or more, while a vertex settling onto a
/// saddle can jitter `h` up by rounding — this margin separates the two.
const H_RISE_TOL: f64 = 1e-9;
/// Smallest sub-step (as a fraction of `dt`) before a vertex is declared
/// frozen at a pole-pinch (~14 halvings).
const MIN_SUBSTEP_FRAC: f64 = 6e-5;

impl<const D: usize> Flow<D> for Rk4 {
    fn step<P: Phase<D> + ?Sized>(&mut self, mesh: &mut Mesh<D>, phase: &P) -> Result<StepReport> {
        let mask = mesh.active_vertex_mask();
        let cap = self.max_speed;
        let dt = self.dt;

        // i * conj(grad phi), componentwise: i * conj(a + ib) = b + ia,
        // saturated at |v| = cap.
        let velocity = |z: &SVector<Complex64, D>| {
            let mut v = phase.gradient(z).map(|g| Complex64::new(g.im, g.re));
            let speed = v.norm();
            if speed > cap {
                v *= Complex64::from(cap / speed);
            }
            v
        };

        // One RK4 increment over a sub-interval of length `step`.
        let rk4_dz = |z: &SVector<Complex64, D>, step: f64| {
            let half = Complex64::from(0.5 * step);
            let full = Complex64::from(step);
            let sixth = Complex64::from(step / 6.0);
            let k1 = velocity(z);
            let k2 = velocity(&(*z + k1 * half));
            let k3 = velocity(&(*z + k2 * half));
            let k4 = velocity(&(*z + k3 * full));
            (k1 + (k2 + k3) * Complex64::from(2.0) + k4) * sixth
        };

        let h_of = |z: &SVector<Complex64, D>| -phase.value(z).im;
        let min_substep = dt * MIN_SUBSTEP_FRAC;

        // (max |Δz|, Σ h, active count, all finite)
        let (max_step, h_sum, count, finite) = mesh
            .points
            .par_iter_mut()
            .enumerate()
            .filter(|(i, _)| mask[*i])
            .map(|(_, z)| {
                let start = *z;
                // Advance over the full interval `dt` with h-monotone,
                // halving sub-steps; freeze once a sub-step underflows.
                let mut remaining = dt;
                let mut sub = dt;
                let mut iters = 0;
                while remaining > 1e-9 * dt && iters < 128 {
                    iters += 1;
                    let step = sub.min(remaining);
                    let h_old = h_of(z);
                    let dz = rk4_dz(z, step);
                    let cand = *z + dz;
                    let ok = cand.iter().all(|c| c.re.is_finite() && c.im.is_finite())
                        && h_of(&cand) <= h_old + H_RISE_TOL * (1.0 + h_old.abs());
                    if ok {
                        *z = cand;
                        remaining -= step;
                        sub = (sub * 2.0).min(remaining);
                    } else if step <= min_substep {
                        break; // pole-pinch: freeze here
                    } else {
                        sub *= 0.5;
                    }
                }
                let moved = (*z - start).norm();
                let finite = z.iter().all(|c| c.re.is_finite() && c.im.is_finite());
                (moved, h_of(z), 1usize, finite)
            })
            .reduce(
                || (0.0, 0.0, 0, true),
                |a, b| (a.0.max(b.0), a.1 + b.1, a.2 + b.2, a.3 && b.3),
            );

        if !finite {
            return Err(Error::FlowDiverged);
        }
        Ok(StepReport {
            max_step,
            mean_morse: if count > 0 {
                h_sum / count as f64
            } else {
                f64::NEG_INFINITY
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1D quadratic phase, exact flow known: z(λ) = x₀ (cosh λ + i sinh λ).
    struct Quad1D;
    impl Phase<1> for Quad1D {
        fn value(&self, z: &SVector<Complex64, 1>) -> Complex64 {
            0.5 * z[0] * z[0]
        }
        fn gradient(&self, z: &SVector<Complex64, 1>) -> SVector<Complex64, 1> {
            *z
        }
    }

    #[test]
    fn rk4_descends_h_and_conserves_re_phi() {
        let mut mesh = Mesh::<1>::rectangular(&[(-2.0, 2.0)], 0.5).unwrap();
        let re_phi_before: Vec<f64> = mesh.points.iter().map(|z| Quad1D.value(z).re).collect();

        // max_speed chosen high enough that this test's trajectories
        // (|v| <= 4) never saturate, so the exact-trajectory check holds.
        let mut flow = Rk4 {
            dt: 0.05,
            max_speed: 10.0,
        };
        let mut last_mean = 0.0;
        for _ in 0..20 {
            let report = Flow::<1>::step(&mut flow, &mut mesh, &Quad1D).unwrap();
            assert!(report.mean_morse <= last_mean + 1e-12, "h must descend");
            last_mean = report.mean_morse;
        }
        assert!(last_mean < -0.1);

        // H = Re phi is an exact invariant of the continuous flow; RK4 keeps
        // it to its local truncation error.
        for (z, before) in mesh.points.iter().zip(re_phi_before) {
            assert!((Quad1D.value(z).re - before).abs() < 1e-7);
        }

        // Exact trajectory check at x0 = 2 (the last lattice point):
        // λ = 20 * 0.05 = 1.
        let lambda: f64 = 1.0;
        let exact = Complex64::new(2.0 * lambda.cosh(), 2.0 * lambda.sinh());
        let got = mesh.points.last().unwrap()[0];
        assert!((got - exact).norm() < 1e-6, "got {got}, expected {exact}");
    }

    #[test]
    fn guard_keeps_vertices_off_the_pole_side() {
        // A converging rational lens (poles at ±i) whose unguarded flow used
        // to tunnel vertices onto the h > 0 side. The guard's invariant:
        // since the lens is real on ℝ (h = 0 initially) and ḣ ≤ 0, no
        // trajectory may reach h > 0. After many steps every vertex must
        // still satisfy h ≤ (rounding) — a direct check of the guard.
        let lens = crate::phases::RationalLens {
            beta: 0.5,
            kappa: -0.3,
        };
        let mut mesh = Mesh::<1>::rectangular(&[(-8.0, 8.0)], 0.02).unwrap();
        let mut flow = Rk4 {
            dt: 0.05,
            max_speed: 10.0,
        };
        for _ in 0..60 {
            Flow::<1>::step(&mut flow, &mut mesh, &lens).unwrap();
        }
        let worst = mesh
            .points
            .iter()
            .map(|z| -lens.value(z).im)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(worst < 1e-6, "a vertex reached h = {worst} > 0 (pole side)");
    }
}
