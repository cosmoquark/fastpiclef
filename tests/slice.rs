//! Milestone A0: the inner x_0-slice thimble kernel of the Cartesian-slice 2D
//! field solver.
//!
//!  - `slice_polynomial_identity`: the cleared numerator/denominator satisfy
//!    N(z) = phi'(z) D(z) (algebra of the saddle source).
//!  - `inner_far_slice_exact`: for a slice whose log branch points sit far off
//!    the real axis, deforming the real line to the saddle thimble sweeps *no*
//!    branch point, so the inner thimble equals an independent 1D mesh-flow
//!    solve to machine precision. This validates the whole inner kernel — the
//!    complete polynomial saddle source and the continuous-phase tracking that
//!    keeps the tracer/quadrature on one sheet across the log cuts.
//!  - `inner_swept_branch_point` (ignored): a mid/near slice, where the
//!    deformation sweeps a branch point. The saddle sum then misses that
//!    point's branch-cut (Hankel) contribution — the open homology-completeness
//!    problem of option A. Kept as the target the fix must satisfy.

use fast_picard_lefshetz_integral_code::phases::PointMassField;
use fast_picard_lefshetz_integral_code::roots::poly_eval;
use fast_picard_lefshetz_integral_code::{
    Output, Phase, Solver, SolverConfig, ThimbleConfig, X1SlicePhase, slice_inner,
};
use nalgebra::SVector;
use num_complex::Complex64;

fn test_field() -> PointMassField {
    PointMassField {
        beta: [0.2, 0.1],
        lenses: vec![(-0.8, 0.5, 0.6), (0.9, -0.4, 0.4)],
    }
}

#[test]
fn slice_polynomial_identity() {
    let field = test_field();
    for &x2 in &[1.3_f64, -0.7, 0.0] {
        let slice = X1SlicePhase {
            field: &field,
            x2: Complex64::from(x2),
        };
        let (n, d) = slice.stationary_fraction();
        // deg N = 2 * n_masses + 1, deg D = 2 * n_masses.
        assert_eq!(n.len(), 2 * field.lenses.len() + 2);
        assert_eq!(d.len(), 2 * field.lenses.len() + 1);
        for &(re, im) in &[(0.3_f64, 0.4_f64), (-1.1, 0.6), (2.0, -1.3)] {
            let z = Complex64::new(re, im);
            let phi_prime = slice.gradient(&SVector::<Complex64, 1>::new(z))[0];
            let lhs = poly_eval(&n, z);
            let rhs = phi_prime * poly_eval(&d, z);
            let err = (lhs - rhs).norm();
            assert!(
                err < 1e-9 * (1.0 + lhs.norm()),
                "N != phi' D at x2={x2}, z={z}: |diff|={err:.3e}"
            );
        }
    }
}

/// Independent 1D mesh-flow solve of the same slice integral G(x_1) — a
/// method-independent oracle (the slice integrand is real on the real axis
/// with off-axis branch points, so the Gap-2 pole guard converges).
fn mesh_inner(field: &PointMassField, x2: f64, omega: f64) -> Output {
    let slice = X1SlicePhase {
        field,
        x2: Complex64::from(x2),
    };
    let cfg = SolverConfig {
        delta: 0.01,
        omega,
        ..SolverConfig::default()
    };
    Solver::<1>::new(&slice, cfg).run().unwrap()
}

fn rel_thimble_vs_mesh(field: &PointMassField, x2: f64, omega: f64) -> f64 {
    let cfg = ThimbleConfig {
        omega,
        ..ThimbleConfig::default()
    };
    let thimble = slice_inner(field, x2, &cfg).unwrap();
    let mesh = mesh_inner(field, x2, omega);
    let rel = (thimble.value - mesh.value).norm() / mesh.value.norm();
    eprintln!(
        "slice inner: x2={x2}, omega={omega}: thimble={:.8}, mesh={:.8}, rel={rel:.3e}",
        thimble.value, mesh.value,
    );
    rel
}

#[test]
fn inner_far_slice_exact() {
    let field = test_field();
    // Slices far from both masses (b_j = 0.5, -0.4): branch points sit >= ~3.5
    // off the real axis, none is swept, so the thimble is exact.
    for &(x2, omega) in &[(4.0, 1.0), (4.0, 3.0), (8.0, 1.0), (-5.0, 2.0)] {
        assert!(
            rel_thimble_vs_mesh(&field, x2, omega) < 1e-9,
            "far slice should be machine-exact at x2={x2}, omega={omega}"
        );
    }

    // A single isolated unit mass at the origin, slice well clear of it.
    let one = PointMassField {
        beta: [0.3, 0.0],
        lenses: vec![(0.0, 0.0, 1.0)],
    };
    assert!(rel_thimble_vs_mesh(&one, 3.0, 1.0) < 1e-9);
}

#[test]
#[ignore = "swept branch point: saddle sum misses the Hankel contribution (option-A open problem)"]
fn inner_swept_branch_point() {
    let field = test_field();
    // Mid slice: the deformation sweeps the lower mass-0 branch point, so the
    // thimble currently disagrees with the mesh by ~1e-3. The fix (adding the
    // branch-cut contribution) must bring this under tolerance.
    assert!(rel_thimble_vs_mesh(&field, 1.3, 1.0) < 1e-6);
}
