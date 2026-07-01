//! The constant-phase contour solver (Shi 2024) against exact answers —
//! the same reference cases as `tests/airy.rs` plus the quadratic phase,
//! so the two solver paths are directly comparable.
//!
//! Reference values: quadratic = sqrt(2*pi/omega) e^{i pi/4}; Airy values
//! generated 2026-06-10 with scipy (see tests/airy.rs).

use fast_picard_lefshetz_integral_code::{
    Error, Phase, SolveMethod, SolverConfig, ThimbleConfig, ThimbleSolver, phases::AiryPhase,
    solve_1d_hybrid,
};
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

fn check(phase: &dyn Phase<1>, cfg: ThimbleConfig, expected: Complex64, tol: f64) {
    let mut solver = ThimbleSolver::new(phase, cfg);
    let out = solver.run().unwrap();
    let rel = (out.value - expected).norm() / expected.norm();
    eprintln!(
        "thimble: got {}, expected {expected}, rel err {rel:.3e} (estimate {:.3e}), {} piece(s), {} saddle(s)",
        out.value,
        out.error / expected.norm(),
        solver.pieces().len(),
        solver.saddles().len(),
    );
    assert!(
        rel < tol,
        "got {}, expected {expected}, rel err {rel:.3e}",
        out.value
    );
}

#[test]
fn quadratic_matches_analytic() {
    let expected = Complex64::from_polar((2.0 * PI).sqrt(), FRAC_PI_4);
    check(&QuadPhase1D, ThimbleConfig::default(), expected, 1e-10);
}

#[test]
fn airy_two_real_images() {
    check(
        &AiryPhase { y: 1.0 },
        ThimbleConfig::default(),
        Complex64::new(3.365_028_273_002_628_2, 0.0),
        1e-9,
    );
}

#[test]
fn airy_no_real_images() {
    // Only the saddle at +i is effective; its conjugate at -i has h > 0 and
    // must be rejected by the ascending-contour criterion.
    check(
        &AiryPhase { y: -1.0 },
        ThimbleConfig::default(),
        Complex64::new(0.850_067_322_349_920_66, 0.0),
        1e-9,
    );
}

#[test]
fn airy_higher_frequency() {
    let cfg = ThimbleConfig {
        omega: 4.0,
        ..ThimbleConfig::default()
    };
    check(
        &AiryPhase { y: 1.0 },
        cfg,
        Complex64::new(-0.497_688_158_651_127_14, 0.0),
        1e-9,
    );
}

#[test]
fn caustic_is_rejected_with_clear_error() {
    let phase = AiryPhase { y: 0.0 };
    let mut solver = ThimbleSolver::new(&phase, ThimbleConfig::default());
    match solver.run() {
        Err(Error::DegenerateSaddle) => {}
        other => panic!("expected DegenerateSaddle for the caustic, got {other:?}"),
    }
}

#[test]
fn hybrid_falls_back_to_mesh_at_the_caustic() {
    // The thimble solver cannot treat the y=0 caustic; the hybrid must route
    // to the mesh solver and recover the exact answer 2π ω^{-1/3} Ai(0).
    let mesh_cfg = SolverConfig {
        domain: vec![(-6.0, 6.0)],
        delta: 0.025,
        flow_steps: 20,
        ..SolverConfig::default()
    };
    // A regular case stays on the fast path...
    let (_out, how) = solve_1d_hybrid(
        &AiryPhase { y: 1.0 },
        ThimbleConfig::default(),
        mesh_cfg.clone(),
    )
    .unwrap();
    assert_eq!(how, SolveMethod::Thimble);

    // ...the caustic falls back and is still correct.
    let (out, how) =
        solve_1d_hybrid(&AiryPhase { y: 0.0 }, ThimbleConfig::default(), mesh_cfg).unwrap();
    assert_eq!(how, SolveMethod::MeshFallback);
    let exact = 2.230_707_051_824_495_6;
    let rel = (out.value - Complex64::new(exact, 0.0)).norm() / exact;
    assert!(rel < 1e-6, "hybrid caustic value off: rel={rel:.3e}");
}

#[test]
fn only_one_piece_for_single_effective_saddle() {
    let phase = AiryPhase { y: -1.0 };
    let mut solver = ThimbleSolver::new(&phase, ThimbleConfig::default());
    solver.run().unwrap();
    assert_eq!(solver.saddles().len(), 2);
    assert_eq!(solver.pieces().len(), 1);
}
