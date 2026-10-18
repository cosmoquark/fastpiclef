//! Fast Picard-Lefschetz and Cauchy-contour Fresnel-integral solver for
//! wave-optics gravitational lensing.
//!
//! Solver map:
//! - **2D exact RC slice engine** ([`slice_field`], [`lens_field_solve`]) —
//!   Cartesian-slice reduction with a breakpoint-anchored real core and
//!   $\pi/4$-rotated Gaussian tails in both variables, for point-mass fields
//!   ([`phases::PointMassField`]) and the general macro ($\kappa, \gamma$) +
//!   axisymmetric star/halo profile model ([`LensField`]).  Exact by Cauchy
//!   with no branch-point sweeping.
//! - **Shared-contour batching** ([`slice_field_batch`], [`lens_field_batch`],
//!   [`rc_1d_batch`], [`rc_1d_sweep`]) — one contour serves a tensor source
//!   grid and/or frequency band via [`adaptive::integrate_vec`].
//! - **High-$\omega$ stationary-phase hybrid** ([`lens_field_hybrid`],
//!   [`point_mass_field_hybrid`]) — geometric optics plus the validated
//!   $O(1/\omega)$ correction ([`lens_field_geometric_corrected`],
//!   [`point_mass_field_geometric_corrected`]) when its conservative budget
//!   fits `tol`, falling back to the exact RC engine otherwise.
//! - **1D constant-phase contour solver** ([`ThimbleSolver`],
//!   [`solve_1d_hybrid`]) — traces steepest-descent level curves directly from
//!   the saddles (complete polynomial root source for [`RationalPhase`]).
//! - **Axisymmetric 2D fast paths** ([`PolarSolver`],
//!   [`solve_point_mass_polar`], and the $\mu$s-class Nakamura-Deguchi closed
//!   form [`point_mass_nd`]).
//! - **General-dimension mesh-flow solver** ([`Solver`]`<D>`, [`solve`]) — a
//!   real rectangular simplicial mesh ([`Mesh::rectangular`]) is sunk towards
//!   the Lefschetz thimbles by the descent flow of $h = -\mathrm{Im}\,\phi$
//!   ([`Rk4`]), pruning negligible simplices, and integrated with
//!   Grundmann-Möller quadrature ([`GrundmannMoeller`]).

pub mod adaptive;
pub mod error;
pub mod flow;
pub mod images;
pub mod lens_field;
pub mod nd;
pub mod mesh;
pub mod phase;
pub mod phases;
pub mod polar;
pub mod rc1d;
pub mod quadrature;
pub mod roots;
pub mod saddles;
pub mod slice;
pub mod slice_batch;
pub mod solver;
pub mod thimble;

mod simd_ln;

#[cfg(feature = "python")]
mod python;

pub use adaptive::{AdaptiveConfig, AdaptiveResult, integrate};
pub use error::{Error, Result};
pub use flow::{Flow, Rk4, StepReport};
pub use lens_field::{
    Component, GeometricCorrected, HybridMethod, HybridOutput, LensField, Profile,
    lens_field_batch, lens_field_geometric, lens_field_geometric_corrected, lens_field_hybrid,
    lens_field_images, lens_field_solve, point_mass_field_geometric_corrected,
    point_mass_field_hybrid,
};
pub use mesh::Mesh;
pub use nd::point_mass_nd;
pub use phase::{DynPhase, Phase, PhaseAsDyn, RationalPhase, RationalPhase2D, as_dyn};
pub use polar::{PolarConfig, PolarSolver, solve_point_mass_polar};
pub use quadrature::{GrundmannMoeller, Quadrature};
pub use rc1d::{rc_1d_batch, rc_1d_batch_evals, rc_1d_sweep};
pub use roots::poly_roots;
pub use saddles::{Saddle, find_saddles, saddles_from_polynomial, second_derivative};
pub use slice::{
    InterruptHook, SliceConfig, SliceFieldOutput, X1SlicePhase, branch_point_indicator, slice_field,
    slice_inner, slice_inner_rc, slice_inner_rc_evals,
};
pub use slice_batch::{FieldBatch, FieldBatchOutput, slice_field_batch};
pub use solver::{Output, Solver, SolverConfig};
pub use thimble::{SolveMethod, ThimbleConfig, ThimbleSolver, solve_1d_hybrid};

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
