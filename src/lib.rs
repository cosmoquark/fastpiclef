//! Fast Picard-Lefshetz Fresnel-integral solver.
//!
//! Pipeline: a real rectangular simplicial mesh ([`Mesh::rectangular`]) is
//! sunk towards the Lefshetz thimbles by the descent flow of
//! $h = -\mathrm{Im}\,\phi$ ([`Rk4`]), pruning simplices whose integrand has
//! become negligible, and the resulting complex simplicial chain is
//! integrated with Grundmann-Möller quadrature ([`GrundmannMoeller`]).
//!
//! The crate is generic in the integration dimension via `const D: usize`.
//! Callers who know the dimension at compile time should reach for
//! [`Solver`] directly; callers with a runtime dimension can use [`solve`],
//! which dispatches through a `seq-macro`-generated match.

pub mod error;
pub mod flow;
pub mod mesh;
pub mod phase;
pub mod phases;
pub mod quadrature;
pub mod solver;

pub use error::{Error, Result};
pub use flow::{Flow, Rk4, StepReport};
pub use mesh::Mesh;
pub use phase::{DynPhase, Phase, PhaseAsDyn, as_dyn};
pub use quadrature::{GrundmannMoeller, Quadrature};
pub use solver::{Output, Solver, SolverConfig};

/// Runtime entry point. Dispatches on `d` to the right const-generic
/// `Solver::<D>` specialisation. The supported range lives only in the
/// `seq!` bound below — widen it there.
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
