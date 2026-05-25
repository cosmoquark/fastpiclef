use crate::{
    Error, Mesh, Phase, Result,
    flow::Rk4,
    quadrature::GrundmannMoeller,
};
use num_complex::Complex64;

/// Knobs that govern the descent + quadrature pipeline.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Maximum edge length before a simplex is refined.
    pub max_edge: f64,
    /// Drop simplices whose Morse value $h = -\mathrm{Im}\,\phi$ stays below this.
    pub morse_threshold: f64,
    /// Number of descent steps to perform.
    pub flow_steps: usize,
    /// Initial step size for the descent integrator.
    pub dt_init: f64,
    /// Grundmann-Moeller polynomial order for the final quadrature.
    pub quad_order: u32,
    /// Frequency $\omega$ entering the integrand $e^{i \omega \phi}$.
    pub omega: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_edge: 0.25,
            morse_threshold: -30.0,
            flow_steps: 100,
            dt_init: 0.05,
            quad_order: 3,
            omega: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Output {
    pub value: Complex64,
    pub error: f64,
}

/// Orchestrates the Picard-Lefshetz pipeline at a fixed compile-time
/// dimension `D`. The struct owns the working mesh and integrators; it
/// borrows the phase (so users can keep their phase struct on the stack
/// and reuse it across calls).
pub struct Solver<'a, const D: usize> {
    phase: &'a dyn Phase<D>,
    cfg: SolverConfig,
    mesh: Mesh<D>,
    flow: Rk4,
    quad: GrundmannMoeller,
}

impl<'a, const D: usize> Solver<'a, D> {
    pub fn new(phase: &'a dyn Phase<D>, cfg: SolverConfig) -> Self {
        let flow = Rk4 { dt: cfg.dt_init };
        let quad = GrundmannMoeller { order: cfg.quad_order };
        Self {
            phase,
            cfg,
            mesh: Mesh::empty(),
            flow,
            quad,
        }
    }

    /// Run the full pipeline: build mesh -> flow + refine + prune loop -> quadrature.
    /// Currently every step beyond construction is a `todo!()` placeholder
    /// returned as `Error::NotImplemented`.
    pub fn run(&mut self) -> Result<Output> {
        // 1. Build an initial mesh.
        //    Skeleton: leave the domain unspecified for now; this is where a
        //    future caller-supplied domain or critical-point seed will land.
        //    Returning NotImplemented preserves the dispatch surface for tests.
        let _ = (&self.mesh, &self.flow, &self.quad, &self.cfg, self.phase);
        Err(Error::NotImplemented("Solver::run"))
    }
}
