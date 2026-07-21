use crate::{
    Error, Mesh, Phase, Result,
    flow::{Flow, Rk4},
    quadrature::{GrundmannMoeller, Quadrature},
};
use num_complex::Complex64;

/// Knobs that govern the descent + quadrature pipeline.
///
/// The defaults are tuned for $\omega = O(1)$ phases whose stationary points
/// sit within a few units of the origin (the quadratic test family). For
/// larger $\omega$ shrink `delta` (the thimble width scales like
/// $1/\sqrt{\omega}$); for smaller $\omega$ widen `domain`.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Integration domain: one `(lo, hi)` interval per dimension. An empty
    /// vector means the default box $(-8, 8)^D$.
    pub domain: Vec<(f64, f64)>,
    /// Initial mesh spacing along each axis.
    pub delta: f64,
    /// Drop simplices once $\omega h < $ `morse_threshold` at every vertex,
    /// i.e. once the integrand magnitude $e^{\omega h}$ falls below
    /// $e^{\text{threshold}}$ everywhere on the simplex.
    pub morse_threshold: f64,
    /// Maximum number of descent steps. The flow stops early once the
    /// integral value has plateaued (see `flow_tol`), so this is a budget,
    /// not a fixed count.
    pub flow_steps: usize,
    /// Relative-change threshold for adaptive flow stopping: descent halts
    /// once the integral value changes by less than this between successive
    /// checks (two in a row). This lands the contour in the exact window
    /// *before* pole interference degrades an over-flowed mesh — the plateau
    /// and the degradation are separated by ~12 orders of magnitude, so the
    /// exact value is unambiguous. Set to `0.0` to force the full
    /// `flow_steps` budget (the old fixed-step behaviour).
    pub flow_tol: f64,
    /// Step size for the descent integrator.
    pub dt_init: f64,
    /// Speed cap for the descent field (see [`Rk4`]): keeps fixed-step
    /// integration stable when $|\nabla\phi|$ grows superlinearly, without
    /// changing the pathlines (hence the integral). Raise it if the
    /// stationary points of interest live where $|\nabla\phi|$ exceeds it.
    pub max_speed: f64,
    /// Grundmann-Möller order $s$ (polynomial degree $2s+1$); must be >= 1.
    pub quad_order: u32,
    /// Frequency $\omega$ entering the integrand $e^{i \omega \phi}$.
    pub omega: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            domain: Vec::new(),
            delta: 0.1,
            morse_threshold: -30.0,
            flow_steps: 60,
            flow_tol: 1e-6,
            dt_init: 0.05,
            max_speed: 10.0,
            quad_order: 3,
            omega: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Output {
    pub value: Complex64,
    /// Embedded quadrature error estimate (difference between the order-$s$
    /// and order-$(s-1)$ Grundmann-Möller results on the same contour).
    pub error: f64,
}

/// Orchestrates the Picard-Lefshetz pipeline at a fixed compile-time
/// dimension `D`. The struct owns the working mesh and integrators; it
/// borrows the phase (so users can keep their phase struct on the stack
/// and reuse it across calls). After [`Solver::run`] the deformed mesh
/// remains available on the solver for inspection.
pub struct Solver<'a, const D: usize> {
    phase: &'a dyn Phase<D>,
    cfg: SolverConfig,
    mesh: Mesh<D>,
    flow: Rk4,
    quad: GrundmannMoeller,
    steps_used: usize,
}

impl<'a, const D: usize> Solver<'a, D> {
    pub fn new(phase: &'a dyn Phase<D>, cfg: SolverConfig) -> Self {
        let flow = Rk4 {
            dt: cfg.dt_init,
            max_speed: cfg.max_speed,
        };
        let quad = GrundmannMoeller {
            order: cfg.quad_order,
        };
        Self {
            phase,
            cfg,
            mesh: Mesh::empty(),
            flow,
            quad,
            steps_used: 0,
        }
    }

    /// The working mesh (the deformed contour after [`Solver::run`]).
    pub fn mesh(&self) -> &Mesh<D> {
        &self.mesh
    }

    /// Number of descent steps actually taken by the last [`Solver::run`]
    /// (≤ `flow_steps`; adaptive stopping usually halts well before the
    /// budget).
    pub fn steps_used(&self) -> usize {
        self.steps_used
    }

    /// Run the full pipeline: build the initial real mesh, sink it towards
    /// the Lefshetz thimbles with the descent flow (pruning simplices whose
    /// integrand has become negligible), then integrate $e^{i\omega\phi}$
    /// over the deformed simplicial chain.
    ///
    /// By Cauchy's theorem the deformation does not change the value of the
    /// integral, so the flow only needs to run long enough that (i) the
    /// integrand is negligible on the domain boundary (controlled truncation)
    /// and (ii) the contour is well-conditioned for quadrature. It must also
    /// not run *too* long: past that point, a mesh threading complex poles
    /// slowly accumulates near-pole distortion. The value therefore plateaus
    /// (to machine precision) over a window and then drifts, so we stop
    /// adaptively at the plateau — integrating periodically and halting once
    /// the value stops changing (see `flow_tol`). The `h`-monotonicity guard
    /// in [`Rk4`] keeps the drift bounded even if the budget is exhausted.
    pub fn run(&mut self) -> Result<Output> {
        if !(self.cfg.omega > 0.0) {
            return Err(Error::InvalidInput("omega must be > 0"));
        }
        if !(self.cfg.max_speed > 0.0) {
            return Err(Error::InvalidInput("max_speed must be > 0"));
        }
        let domain: [(f64, f64); D] = if self.cfg.domain.is_empty() {
            [(-8.0, 8.0); D]
        } else if self.cfg.domain.len() == D {
            std::array::from_fn(|k| self.cfg.domain[k])
        } else {
            return Err(Error::InvalidInput(
                "cfg.domain must be empty or have exactly D intervals",
            ));
        };

        self.mesh = Mesh::rectangular(&domain, self.cfg.delta)?;

        let phase = self.phase;
        let omega = self.cfg.omega;
        let threshold = self.cfg.morse_threshold;
        // Integrate every `CHECK_EVERY` steps; halt on two consecutive
        // sub-`flow_tol` relative changes (a confirmed plateau).
        const CHECK_EVERY: usize = 2;
        let mut out: Option<Output> = None;
        let mut prev_value = Complex64::new(f64::NAN, f64::NAN);
        let mut small_streak = 0;
        self.steps_used = 0;
        for step in 1..=self.cfg.flow_steps {
            self.flow.step(&mut self.mesh, phase)?;
            self.mesh
                .prune(|z| omega * (-phase.value(z).im) > threshold);
            self.steps_used = step;
            if step % CHECK_EVERY == 0 || step == self.cfg.flow_steps {
                let (value, error) = self.quad.integrate(&self.mesh, phase, omega)?;
                let denom = value.norm().max(1e-300);
                if (value - prev_value).norm() / denom < self.cfg.flow_tol {
                    small_streak += 1;
                } else {
                    small_streak = 0;
                }
                out = Some(Output { value, error });
                if small_streak >= 2 {
                    break;
                }
                prev_value = value;
            }
        }
        // flow_steps == 0 leaves the undeformed mesh; integrate it directly.
        match out {
            Some(o) => Ok(o),
            None => {
                let (value, error) = self.quad.integrate(&self.mesh, phase, omega)?;
                Ok(Output { value, error })
            }
        }
    }
}
