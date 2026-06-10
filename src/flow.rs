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

/// Fixed-step classical RK4 integrator on the raw descent field.
#[derive(Debug, Clone, Copy)]
pub struct Rk4 {
    pub dt: f64,
}

impl<const D: usize> Flow<D> for Rk4 {
    fn step<P: Phase<D> + ?Sized>(&mut self, mesh: &mut Mesh<D>, phase: &P) -> Result<StepReport> {
        let mask = mesh.active_vertex_mask();
        let dt = Complex64::from(self.dt);
        let half = Complex64::from(0.5 * self.dt);
        let sixth = Complex64::from(self.dt / 6.0);

        // i * conj(grad phi), componentwise: i * conj(a + ib) = b + ia.
        let velocity =
            |z: &SVector<Complex64, D>| phase.gradient(z).map(|g| Complex64::new(g.im, g.re));

        // (max |Δz|, Σ h, active count, all finite)
        let (max_step, h_sum, count, finite) = mesh
            .points
            .par_iter_mut()
            .enumerate()
            .filter(|(i, _)| mask[*i])
            .map(|(_, z)| {
                let k1 = velocity(z);
                let k2 = velocity(&(*z + k1 * half));
                let k3 = velocity(&(*z + k2 * half));
                let k4 = velocity(&(*z + k3 * dt));
                let dz = (k1 + (k2 + k3) * Complex64::from(2.0) + k4) * sixth;
                *z += dz;
                let finite = z.iter().all(|c| c.re.is_finite() && c.im.is_finite());
                (dz.norm(), -phase.value(z).im, 1usize, finite)
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

        let mut flow = Rk4 { dt: 0.05 };
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
}
