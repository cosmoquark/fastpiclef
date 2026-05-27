//! Smoke-level demo for the skeleton solver.
//!
//! Defines two quadratic Fresnel phases (1D and 2D) and invokes the solver
//! through both entry points (typed and runtime-dispatched). The numerical
//! pipeline is unimplemented, so the calls are expected to surface
//! `Error::NotImplemented` — the binary's job for now is to prove the API
//! shapes compile and the dispatch boundary is wired correctly.

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

struct QuadPhase2D;

impl Phase<2> for QuadPhase2D {
    fn value(&self, z: &SVector<Complex64, 2>) -> Complex64 {
        0.5 * (z[0] * z[0] + z[1] * z[1])
    }
    fn gradient(&self, z: &SVector<Complex64, 2>) -> SVector<Complex64, 2> {
        SVector::<Complex64, 2>::from_column_slice(&[z[0], z[1]])
    }
}

fn report<T: std::fmt::Debug>(label: &str, result: Result<T, Error>) {
    match result {
        Ok(v) => println!("{label}: Ok({v:?})"),
        Err(e) => println!("{label}: Err({e})"),
    }
}

fn main() {
    let cfg = SolverConfig::default();

    // Typed entry point (zero-overhead, dimension known at compile time).
    let phase1 = QuadPhase1D;
    let mut solver1 = Solver::<1>::new(&phase1, cfg.clone());
    report("1D typed Solver::run", solver1.run());

    let phase2 = QuadPhase2D;
    let mut solver2 = Solver::<2>::new(&phase2, cfg.clone());
    report("2D typed Solver::run", solver2.run());

    // Runtime entry point exercising the seq-macro dispatch.
    let dyn1 = as_dyn(QuadPhase1D);
    report("1D solve(d=1, ..)", solve(1, &dyn1, cfg.clone()));

    let dyn2 = as_dyn(QuadPhase2D);
    report("2D solve(d=2, ..)", solve(2, &dyn2, cfg.clone()));

    // Boundary cases of the dispatcher.
    report("unsupported solve(d=99, ..)", solve(99, &dyn2, cfg.clone()));
    report("mismatched solve(d=3, ..)", solve(3, &dyn2, cfg));
}
