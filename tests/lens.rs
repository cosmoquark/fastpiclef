//! Lens-phase validation by *cross-method agreement* (no closed form known):
//! the independent thimble and mesh-flow solvers must agree where both are
//! reliable. The exploration that produced these cases (2026-06-14) also
//! mapped where they are *not* both reliable — recorded here so the gaps stay
//! visible (see also the journal's "to do" flags):
//!
//!  * Grid-Newton saddle finding misses complex saddles that sit very close
//!    to the potential's poles (their Newton basins are tiny). Harmless for a
//!    gentle *diverging* lens, where those saddles are non-effective (the
//!    thimble answer is still machine-exact), but it makes multi-image and
//!    simG configurations untrustworthy until a polynomial/companion root
//!    finder lands. -> TODO: rational-phase root finder.
//!  * The mesh-flow solver overshoots near a pole for a *converging* lens
//!    (e.g. beta=0.5, kappa=-0.3 returns ~270), so it cannot be used as the
//!    reference there. -> TODO: mesh near-pole overshoot guard.
//!
//! Hence the asserting cases below are all gentle diverging lenses.

use fast_picard_lefshetz_integral_code::{
    PolarConfig, PolarSolver, Solver, SolverConfig, ThimbleConfig, ThimbleSolver, find_saddles,
    phases::{RationalLens, RationalLens2D, SimGLens},
};

fn mesh_cfg_1d() -> SolverConfig {
    SolverConfig {
        domain: vec![(-8.0, 8.0)],
        delta: 0.01,
        flow_steps: 30,
        omega: 1.0,
        ..SolverConfig::default()
    }
}

fn real_saddles(s: &[fast_picard_lefshetz_integral_code::Saddle]) -> usize {
    s.iter().filter(|sd| sd.z.im.abs() < 1e-6).count()
}

#[test]
fn rational_lens_thimble_vs_mesh_gentle() {
    // Gentle diverging lenses: one real image, complex saddles non-effective.
    for (beta, kappa) in [(0.3, 0.2), (0.0, 0.25)] {
        let lens = RationalLens { beta, kappa };
        let s = find_saddles(&lens, (-8.0, 8.0), (-8.0, 8.0), 16, 1e-11);
        assert_eq!(
            real_saddles(&s),
            1,
            "beta={beta} kappa={kappa}: expected one real image"
        );

        let thim = ThimbleSolver::new(&lens, ThimbleConfig::default())
            .run()
            .unwrap();
        let mesh = Solver::<1>::new(&lens, mesh_cfg_1d()).run().unwrap();
        let rel = (thim.value - mesh.value).norm() / mesh.value.norm();
        eprintln!(
            "rational beta={beta} kappa={kappa}: thimble={:.10} mesh={:.10} rel={:.2e}",
            thim.value, mesh.value, rel
        );
        assert!(rel < 1e-9, "thimble/mesh disagree: rel={rel:.3e}");
    }
}

#[test]
fn rational_lens_2d_polar_vs_mesh_gentle() {
    let lens = RationalLens2D {
        beta: [0.4, 0.0],
        kappa: 0.2,
    };
    let polar = PolarSolver::new(&lens, PolarConfig::default())
        .run()
        .unwrap();
    let mesh_cfg = SolverConfig {
        domain: vec![(-6.0, 6.0); 2],
        delta: 0.04,
        flow_steps: 30,
        omega: 1.0,
        ..SolverConfig::default()
    };
    let mesh = Solver::<2>::new(&lens, mesh_cfg).run().unwrap();
    let rel = (polar.value - mesh.value).norm() / mesh.value.norm();
    eprintln!(
        "rational 2D: polar={:.10} mesh={:.10} rel={:.2e}",
        polar.value, mesh.value, rel
    );
    assert!(rel < 1e-7, "polar/mesh 2D disagree: rel={rel:.3e}");
}

#[test]
fn simg_lens_evaluates_and_finds_its_image() {
    // Smoke-level only: the simG potential evaluates and its real image is
    // found. Full multi-saddle validation waits on the rational-phase root
    // finder (see module header) — the thimble and mesh answers do NOT yet
    // agree for simG, so no agreement is asserted here.
    let lens = SimGLens {
        beta: 0.4,
        kappa: 0.15,
        kmax: 4,
    };
    let s = find_saddles(&lens, (-8.0, 8.0), (-8.0, 8.0), 24, 1e-10);
    assert!(real_saddles(&s) >= 1, "simG: real image not found");
    let thim = ThimbleSolver::new(&lens, ThimbleConfig::default())
        .run()
        .unwrap();
    assert!(thim.value.is_finite(), "simG thimble value not finite");
}
