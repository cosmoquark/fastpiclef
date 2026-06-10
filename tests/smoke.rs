//! End-to-end tests of the descent + quadrature pipeline on quadratic
//! phases, where the Fresnel integral is known in closed form:
//!
//!   ∫_R^d exp(i ω |x|²/2) d^d x = (2π/ω)^{d/2} e^{i d π/4}.
//!
//! Also pins the runtime-dispatch boundary: rejection of unsupported
//! dimensions and of phase/caller dimension mismatches.

use fast_picard_lefshetz_integral_code::{Error, Phase, Solver, SolverConfig, as_dyn, solve};
use nalgebra::SVector;
use num_complex::Complex64;
use std::f64::consts::{FRAC_PI_4, PI};

struct QuadPhase1D;

impl Phase<1> for QuadPhase1D {
    fn value(&self, z: &SVector<Complex64, 1>) -> Complex64 {
        0.5 * z[0] * z[0]
    }
    fn gradient(&self, z: &SVector<Complex64, 1>) -> SVector<Complex64, 1> {
        *z
    }
}

struct QuadPhase2D;

impl Phase<2> for QuadPhase2D {
    fn value(&self, z: &SVector<Complex64, 2>) -> Complex64 {
        0.5 * (z[0] * z[0] + z[1] * z[1])
    }
    fn gradient(&self, z: &SVector<Complex64, 2>) -> SVector<Complex64, 2> {
        *z
    }
}

/// (2π/ω)^{d/2} e^{i d π/4}
fn analytic(d: u32, omega: f64) -> Complex64 {
    Complex64::from_polar(
        (2.0 * PI / omega).powf(d as f64 / 2.0),
        d as f64 * FRAC_PI_4,
    )
}

#[test]
fn typed_1d_quadratic_matches_analytic() {
    let cfg = SolverConfig {
        delta: 0.05,
        flow_steps: 25,
        ..SolverConfig::default()
    };
    let out = Solver::<1>::new(&QuadPhase1D, cfg).run().unwrap();
    let exact = analytic(1, 1.0);
    let rel = (out.value - exact).norm() / exact.norm();
    assert!(
        rel < 1e-6,
        "got {}, expected {exact}, rel err {rel:.3e}",
        out.value
    );
    assert!(out.error.is_finite());
}

#[test]
fn typed_2d_quadratic_matches_analytic() {
    let cfg = SolverConfig {
        domain: vec![(-3.5, 3.5); 2],
        delta: 0.1,
        flow_steps: 20,
        dt_init: 0.075,
        ..SolverConfig::default()
    };
    let out = Solver::<2>::new(&QuadPhase2D, cfg).run().unwrap();
    let exact = analytic(2, 1.0);
    let rel = (out.value - exact).norm() / exact.norm();
    assert!(
        rel < 2e-3,
        "got {}, expected {exact}, rel err {rel:.3e}",
        out.value
    );
}

#[test]
fn runtime_dispatch_routes_supported_dimensions() {
    let dyn_phase = as_dyn(QuadPhase1D);
    let out = solve(1, &dyn_phase, SolverConfig::default()).unwrap();
    let exact = analytic(1, 1.0);
    let rel = (out.value - exact).norm() / exact.norm();
    assert!(
        rel < 2e-3,
        "got {}, expected {exact}, rel err {rel:.3e}",
        out.value
    );
}

#[test]
fn unsupported_dimension_is_rejected() {
    let dyn_phase = as_dyn(QuadPhase1D);
    match solve(5, &dyn_phase, SolverConfig::default()) {
        Err(Error::UnsupportedDimension(5)) => {}
        other => panic!("expected UnsupportedDimension(5), got {other:?}"),
    }
}

#[test]
fn dimension_mismatch_is_rejected() {
    let dyn_phase = as_dyn(QuadPhase1D);
    match solve(2, &dyn_phase, SolverConfig::default()) {
        Err(Error::DimensionMismatch {
            phase: 1,
            requested: 2,
        }) => {}
        other => panic!("expected DimensionMismatch, got {other:?}"),
    }
}
