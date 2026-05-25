use crate::{Error, Mesh, Phase, Result};
use num_complex::Complex64;

/// Numerical integration of $e^{i \omega \phi}$ over a deformed contour
/// represented by `mesh`. Implementations return both a value and an
/// error estimate (e.g. obtained by comparing two quadrature orders).
pub trait Quadrature<const D: usize> {
    fn integrate<P: Phase<D>>(
        &self,
        mesh: &Mesh<D>,
        phase: &P,
        omega: f64,
    ) -> Result<(Complex64, f64)>;
}

/// Grundmann-Moeller quadrature on D-simplices.
#[derive(Debug, Clone, Copy)]
pub struct GrundmannMoeller {
    pub order: u32,
}

impl<const D: usize> Quadrature<D> for GrundmannMoeller {
    fn integrate<P: Phase<D>>(
        &self,
        _mesh: &Mesh<D>,
        _phase: &P,
        _omega: f64,
    ) -> Result<(Complex64, f64)> {
        Err(Error::NotImplemented("GrundmannMoeller::integrate"))
    }
}
