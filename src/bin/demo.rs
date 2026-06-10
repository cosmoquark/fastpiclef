//! Demo of the descent + quadrature pipeline on quadratic Fresnel phases,
//! where the analytic answer is (2π/ω)^{d/2} e^{i d π/4}. Exercises both the
//! typed (`Solver::<D>`) and runtime-dispatched (`solve(d, ...)`) entry
//! points, plus the two rejection paths of the dispatcher.
//!
//! Run with `cargo run --release --bin demo` (the 2D mesh is sluggish in
//! debug builds).

use fast_picard_lefshetz_integral_code::{Phase, Solver, SolverConfig, as_dyn, solve};
use nalgebra::SVector;
use num_complex::Complex64;
use std::f64::consts::{FRAC_PI_4, PI};
use std::time::Instant;

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

fn analytic(d: u32, omega: f64) -> Complex64 {
    Complex64::from_polar(
        (2.0 * PI / omega).powf(d as f64 / 2.0),
        d as f64 * FRAC_PI_4,
    )
}

fn report(label: &str, value: Complex64, est_err: f64, exact: Complex64, t: Instant) {
    let rel = (value - exact).norm() / exact.norm();
    println!("{label}");
    println!("  value     = {value:.10}");
    println!("  analytic  = {exact:.10}");
    println!("  rel error = {rel:.3e}   (estimate {est_err:.3e})");
    println!("  wall time = {:?}\n", t.elapsed());
}

fn main() {
    let omega = 1.0;

    // --- 1D quadratic phase, typed entry point, default configuration.
    let t = Instant::now();
    let out = Solver::<1>::new(&QuadPhase1D, SolverConfig::default())
        .run()
        .unwrap();
    report(
        "1D quadratic (defaults)",
        out.value,
        out.error,
        analytic(1, omega),
        t,
    );

    // --- Same, finer mesh.
    let cfg = SolverConfig {
        delta: 0.02,
        ..SolverConfig::default()
    };
    let t = Instant::now();
    let out = Solver::<1>::new(&QuadPhase1D, cfg).run().unwrap();
    report(
        "1D quadratic (delta = 0.02)",
        out.value,
        out.error,
        analytic(1, omega),
        t,
    );

    // --- 2D quadratic phase, typed entry point.
    let cfg = SolverConfig {
        domain: vec![(-6.0, 6.0); 2],
        delta: 0.05,
        ..SolverConfig::default()
    };
    let t = Instant::now();
    let out = Solver::<2>::new(&QuadPhase2D, cfg).run().unwrap();
    report(
        "2D quadratic (delta = 0.05)",
        out.value,
        out.error,
        analytic(2, omega),
        t,
    );

    // --- Runtime dispatch: same 2D solve through `solve(d, ...)`.
    let cfg = SolverConfig {
        domain: vec![(-3.5, 3.5); 2],
        delta: 0.1,
        flow_steps: 20,
        dt_init: 0.075,
        ..SolverConfig::default()
    };
    let dyn_2d = as_dyn(QuadPhase2D);
    let t = Instant::now();
    let out = solve(2, &dyn_2d, cfg).unwrap();
    report(
        "2D quadratic (runtime dispatch, coarse)",
        out.value,
        out.error,
        analytic(2, omega),
        t,
    );

    // --- Dispatcher boundary cases.
    let dyn_1d = as_dyn(QuadPhase1D);
    println!(
        "solve(99, 1D phase): {:?}",
        solve(99, &dyn_1d, SolverConfig::default()).unwrap_err()
    );
    println!(
        "solve(3,  2D phase): {:?}",
        solve(3, &dyn_2d, SolverConfig::default()).unwrap_err()
    );
}
