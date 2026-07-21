//! Lens-phase validation by *cross-method agreement* (no closed form known):
//! the independent thimble and mesh-flow solvers must agree where both are
//! reliable. The 2026-06-24 exploration mapped where they are not, and the
//! rational (complete) saddle source has since closed the first gap:
//!
//!  * CLOSED — grid-Newton missed complex saddles hugging the potential's
//!    poles (found 1 of the rational lens's 5). `ThimbleSolver::new_rational`
//!    now takes every stationary point from the phase's declared polynomial
//!    (Durand-Kerner + polish), verified complete below.
//!  * OPEN — the mesh-flow solver overshoots near a pole for a *converging*
//!    lens (e.g. beta=0.5, kappa=-0.3 returns ~270), so it still cannot be
//!    the reference there; the multi-image value below is validated by
//!    internal consistency + the geometric-optics asymptote instead, and the
//!    mesh cross-check is deferred to the near-pole guard (Gap 2).

use fast_picard_lefshetz_integral_code::{
    PolarConfig, PolarSolver, RationalPhase, Solver, SolverConfig, ThimbleConfig, ThimbleSolver,
    find_saddles,
    phases::{RationalLens, RationalLens2D, SimGLens},
    saddles_from_polynomial,
};
use num_complex::Complex64;

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
fn rational_lens_complete_saddles() {
    // The case that exposed Gap 1: grid-Newton found 1 of these 5.
    let lens = RationalLens {
        beta: 0.3,
        kappa: 0.2,
    };
    let s = saddles_from_polynomial(&lens, &lens.stationary_polynomial(), 1e-11).unwrap();
    eprintln!("complete rational saddles:");
    for sd in &s {
        eprintln!("  z = {:.6}  h = {:+.4}", sd.z, sd.h());
    }
    assert_eq!(s.len(), 5, "quintic must yield five stationary points");
    assert_eq!(real_saddles(&s), 1);
    // The four complex ones hug the poles at +-i.
    let near_poles = s
        .iter()
        .filter(|sd| (sd.z - Complex64::i()).norm() < 0.6 || (sd.z + Complex64::i()).norm() < 0.6)
        .count();
    assert_eq!(near_poles, 4);
}

#[test]
fn rational_lens_thimble_vs_mesh_gentle() {
    // Gentle diverging lenses: one real image, complex saddles non-effective.
    // Run BOTH thimble constructors — the rational source must reproduce the
    // (already machine-exact) grid-Newton result.
    for (beta, kappa) in [(0.3, 0.2), (0.0, 0.25)] {
        let lens = RationalLens { beta, kappa };
        let thim = ThimbleSolver::new(&lens, ThimbleConfig::default())
            .run()
            .unwrap();
        let thim_rat = ThimbleSolver::new_rational(&lens, ThimbleConfig::default())
            .run()
            .unwrap();
        let mesh = Solver::<1>::new(&lens, mesh_cfg_1d()).run().unwrap();
        let rel = (thim_rat.value - mesh.value).norm() / mesh.value.norm();
        let rel_paths = (thim_rat.value - thim.value).norm() / thim.value.norm();
        eprintln!(
            "rational beta={beta} kappa={kappa}: rational-thimble={:.10} mesh={:.10} \
             rel(mesh)={rel:.2e} rel(grid-thimble)={rel_paths:.2e}",
            thim_rat.value, mesh.value
        );
        assert!(rel < 1e-9, "thimble/mesh disagree: rel={rel:.3e}");
        assert!(
            rel_paths < 1e-11,
            "saddle sources disagree: {rel_paths:.3e}"
        );
    }
}

#[test]
fn rational_lens_2d_polar_vs_mesh_gentle() {
    let lens = RationalLens2D {
        beta: [0.4, 0.0],
        kappa: 0.2,
    };
    let polar = PolarSolver::new_rational(&lens, PolarConfig::default())
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
        "rational 2D (rational polar): polar={:.10} mesh={:.10} rel={:.2e}",
        polar.value, mesh.value, rel
    );
    assert!(rel < 1e-7, "polar/mesh 2D disagree: rel={rel:.3e}");
}

#[test]
fn simg_lens_complete_saddles_and_agreement() {
    // Formerly smoke-level (Gap 1): with the complete source the 17
    // stationary points of the kmax=4 simG lens (Shi 2024, appendix A) are
    // all found. This case also exposed (and now regression-tests) the
    // complex-saddle orientation rule: the thimble traversal is inherited
    // through each ascent crossing, not fixed by the local Re(d) > 0
    // convention.
    //
    // Reference at omega=1: method-independent brute-force quadrature of the
    // raw oscillatory integrand on [-L, L] + exact Fresnel tails
    // (scratch script `brute_check.py`, 2026-07: L-independent to 1e-6).
    // The mesh solver is NOT used as the omega=1 oracle — simG's eight poles
    // sit at |z| ~ 1.2-1.7 and the mesh value scatters by O(10%) with
    // configuration (Gap 2). At omega=30 pruning shields the poles, the
    // mesh is healthy again, and the two solvers must agree tightly.
    let lens = SimGLens {
        beta: 0.4,
        kappa: 0.15,
        kmax: 4,
    };
    let s = saddles_from_polynomial(&lens, &lens.stationary_polynomial(), 1e-10).unwrap();
    assert_eq!(s.len(), 17, "simG kmax=4 must have 17 stationary points");
    let grid = find_saddles(&lens, (-8.0, 8.0), (-8.0, 8.0), 24, 1e-10);
    eprintln!("simG: grid-Newton finds {} of 17 saddles", grid.len());

    // omega = 1 against the brute-force referee.
    let thim = ThimbleSolver::new_rational(&lens, ThimbleConfig::default())
        .run()
        .unwrap();
    let referee = Complex64::new(1.831_975, 1.537_818);
    let rel = (thim.value - referee).norm() / referee.norm();
    eprintln!(
        "simG omega=1: thimble={:.10} referee={referee:.6} rel={rel:.2e}",
        thim.value
    );
    assert!(rel < 1e-5, "simG vs brute-force referee: rel={rel:.3e}");

    // omega = 30: mesh back in its safe regime; the two independent solvers
    // must coincide.
    let thim30 = ThimbleSolver::new_rational(
        &lens,
        ThimbleConfig {
            omega: 30.0,
            ..ThimbleConfig::default()
        },
    )
    .run()
    .unwrap();
    let mesh30 = Solver::<1>::new(
        &lens,
        SolverConfig {
            domain: vec![(-6.0, 6.0)],
            delta: 0.0025,
            flow_steps: 30,
            omega: 30.0,
            ..SolverConfig::default()
        },
    )
    .run()
    .unwrap();
    let rel30 = (thim30.value - mesh30.value).norm() / mesh30.value.norm();
    eprintln!(
        "simG omega=30: thimble={:.10} mesh={:.10} rel={rel30:.2e}",
        thim30.value, mesh30.value
    );
    assert!(rel30 < 1e-6, "simG omega=30 thimble/mesh: rel={rel30:.3e}");
}

#[test]
fn multi_image_converging_lens() {
    // beta=0, kappa=-0.72: three real images at 0 and +-sqrt(sqrt(-2k)-1)
    // = +-sqrt(0.2), plus a complex-conjugate pair at +-i sqrt(1+sqrt(-2k))
    // = +-i sqrt(2.2) — all analytically derivable from the quintic.
    // The mesh solver is NOT reliable here (Gap 2: near-pole overshoot for
    // converging lenses), so the value is validated by (i) internal
    // consistency under trace refinement and (ii) the method-independent
    // brute-force referee at omega = 1 and 30; the mesh cross-check is
    // deferred to the pole guard.
    let lens = RationalLens {
        beta: 0.0,
        kappa: -0.72,
    };
    let s = saddles_from_polynomial(&lens, &lens.stationary_polynomial(), 1e-11).unwrap();
    assert_eq!(s.len(), 5);
    assert_eq!(real_saddles(&s), 3, "expected the three-image regime");
    // Set-matching: the sort key (re, im) is unstable under the +-1e-30
    // sign noise of the origin saddle's real part, so no positional order
    // is assumed.
    let expect = [
        Complex64::new(-(0.2_f64.sqrt()), 0.0),
        Complex64::new(0.0, -(2.2_f64.sqrt())),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 2.2_f64.sqrt()),
        Complex64::new(0.2_f64.sqrt(), 0.0),
    ];
    for want in expect {
        assert!(
            s.iter().any(|got| (got.z - want).norm() < 1e-9),
            "analytic saddle {want} not found in {:?}",
            s.iter().map(|sd| sd.z).collect::<Vec<_>>()
        );
    }

    // (i) Internal consistency: refine the tracer by 4x.
    let coarse = ThimbleSolver::new_rational(&lens, ThimbleConfig::default())
        .run()
        .unwrap();
    let fine_cfg = ThimbleConfig {
        ds_max: 0.05,
        max_dh: 0.25,
        ..ThimbleConfig::default()
    };
    let fine = ThimbleSolver::new_rational(&lens, fine_cfg).run().unwrap();
    let drift = (coarse.value - fine.value).norm() / fine.value.norm();
    eprintln!(
        "multi-image omega=1: F = {:.12} (refinement drift {drift:.2e}, estimate {:.2e})",
        fine.value,
        fine.error / fine.value.norm()
    );
    assert!(drift < 1e-9, "trace-refinement drift {drift:.3e}");

    // (ii) Brute-force referee values (scratch script `brute_check.py`:
    // adaptive quadrature of the raw integrand + exact Fresnel tails;
    // budget ~5e-4 from the O(kappa/L^2) tail truncation at L = 40).
    // A geometric-optics oracle is NOT usable here: the origin image has
    // |phi''| = 0.44, so its stationary-phase corrections are
    // ~phi''''/(8 omega phi''^2) ~ 40%/omega-unit — far too slow.
    let referee_1 = Complex64::new(0.918_432, 2.829_687);
    let rel = (fine.value - referee_1).norm() / referee_1.norm();
    eprintln!("multi-image omega=1 vs referee: rel={rel:.2e}");
    assert!(rel < 1e-3, "omega=1 referee missed: rel={rel:.3e}");

    let thim30 = ThimbleSolver::new_rational(
        &lens,
        ThimbleConfig {
            omega: 30.0,
            ..ThimbleConfig::default()
        },
    )
    .run()
    .unwrap();
    let referee_30 = Complex64::new(-1.145_750_33, 0.683_663_36);
    let rel30 = (thim30.value - referee_30).norm() / referee_30.norm();
    eprintln!(
        "multi-image omega=30: thimble={:.8} referee={referee_30:.8} rel={rel30:.2e}",
        thim30.value
    );
    assert!(rel30 < 1e-3, "omega=30 referee missed: rel={rel30:.3e}");
}
