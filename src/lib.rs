//! Fast Picard-Lefshetz Fresnel-integral solver — first skeleton.
//!
//! The crate is generic in the integration dimension via `const D: usize`.
//! Callers who know the dimension at compile time should reach for
//! [`Solver`] directly; callers with a runtime dimension can use [`solve`],
//! which dispatches through a `seq-macro`-generated match.

pub mod error;
pub mod flow;
pub mod mesh;
pub mod phase;
pub mod quadrature;
pub mod solver;

pub use error::{Error, Result};
pub use flow::{Flow, Rk4, StepReport};
pub use mesh::Mesh;
pub use phase::{DynPhase, Phase, PhaseAsDyn, as_dyn};
pub use quadrature::{GrundmannMoeller, Quadrature};
pub use solver::{Output, Solver, SolverConfig};

/// Range of compile-time dimensions for which the runtime dispatcher
/// has a specialisation. Widen by editing this range and the macro below.
pub const SUPPORTED_DIMS: std::ops::RangeInclusive<usize> = 1..=4;

/// Runtime entry point. Dispatches on `d` to the right const-generic
/// `Solver::<D>` specialisation.
pub fn solve(d: usize, phase: &dyn DynPhase, cfg: SolverConfig) -> Result<Output> {
    seq_macro::seq!(D in 1..=4 {
        match d {
            #(
                D => {
                    let adapter = phase::DynAsPhase::<D>::new(phase)?;
                    Solver::<D>::new(&adapter, cfg).run()
                }
            )*
            _ => Err(Error::UnsupportedDimension(d)),
        }
    })
}
