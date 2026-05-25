use crate::{Mesh, Phase, Result, Error};

#[derive(Debug, Clone, Copy)]
pub struct StepReport {
    pub max_step: f64,
    pub mean_morse: f64,
}

/// Descent integrator that moves every mesh point one step along
/// the Picard-Lefshetz downward flow of $h = -\mathrm{Im}\,\phi$.
pub trait Flow<const D: usize> {
    fn step<P: Phase<D>>(&mut self, mesh: &mut Mesh<D>, phase: &P) -> Result<StepReport>;
}

/// Fixed-step classical RK4 integrator (skeleton).
#[derive(Debug, Clone, Copy)]
pub struct Rk4 {
    pub dt: f64,
}

impl<const D: usize> Flow<D> for Rk4 {
    fn step<P: Phase<D>>(&mut self, _mesh: &mut Mesh<D>, _phase: &P) -> Result<StepReport> {
        Err(Error::NotImplemented("Rk4::step"))
    }
}
