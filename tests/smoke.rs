//! Skeleton smoke tests. They verify that
//!  - the typed `Solver::<D>` constructor compiles and runs (returning
//!    `NotImplemented` while the algorithm is a stub);
//!  - the runtime `solve(d, ...)` entry point dispatches into the right
//!    specialisation when `d` is supported;
//!  - the dispatcher reports `UnsupportedDimension` for out-of-range `d`;
//!  - the dispatcher reports `DimensionMismatch` when the phase's own
//!    dimension does not match the requested `d`.

use fast_picard_lefshetz_integral_code::{
    Error, Phase, Solver, SolverConfig, as_dyn, solve,
};
use nalgebra::SVector;
use num_complex::Complex64;

struct QuadPhase1D;
impl Phase<1> for QuadPhase1D {
    fn value(&self, z: &SVector<Complex64, 1>) -> Complex64 {
        0.5 * z[0] * z[0]
    }
    fn gradient(&self, z: &SVector<Complex64, 1>) -> SVector<Complex64, 1> {
        SVector::<Complex64, 1>::from_column_slice(&[z[0]])
    }
}

#[test]
fn typed_solver_constructs_and_dispatches() {
    let phase = QuadPhase1D;
    let cfg = SolverConfig::default();
    let mut solver = Solver::<1>::new(&phase, cfg);
    // The algorithm is unimplemented, so we expect NotImplemented — not a
    // panic and not any other variant.
    match solver.run() {
        Err(Error::NotImplemented(_)) => {}
        other => panic!("expected NotImplemented, got {other:?}"),
    }
}

#[test]
fn runtime_dispatch_routes_supported_dimensions() {
    let phase = as_dyn(QuadPhase1D);
    let cfg = SolverConfig::default();
    match solve(1, &phase, cfg) {
        Err(Error::NotImplemented(_)) => {}
        other => panic!("expected NotImplemented from supported dim, got {other:?}"),
    }
}

#[test]
fn unsupported_dimension_is_rejected() {
    let phase = as_dyn(QuadPhase1D);
    let cfg = SolverConfig::default();
    match solve(5, &phase, cfg) {
        Err(Error::UnsupportedDimension(5)) => {}
        other => panic!("expected UnsupportedDimension(5), got {other:?}"),
    }
}

#[test]
fn dimension_mismatch_is_rejected() {
    // Phase claims dim=1 but caller asks for d=2; the dispatcher should
    // refuse before invoking the solver.
    let phase = as_dyn(QuadPhase1D);
    let cfg = SolverConfig::default();
    match solve(2, &phase, cfg) {
        Err(Error::DimensionMismatch { phase: 1, requested: 2 }) => {}
        other => panic!("expected DimensionMismatch, got {other:?}"),
    }
}
