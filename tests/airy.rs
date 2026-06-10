//! Solver vs the exact answer on the cubic (Airy) phase,
//! F(ω, y) = 2π ω^{-1/3} Ai(-ω^{2/3} y), across its three saddle regimes
//! plus a higher-frequency run. The cubic's descent field grows like |z|²,
//! so these tests also exercise the speed cap (without it, boundary
//! trajectories blow up in finite flow time).
//!
//! Reference values generated 2026-06-10 in the pixi env:
//!   pixi run python -c "from scipy.special import airy; import numpy as np;
//!     print(2*np.pi * omega**(-1/3) * airy(-omega**(2/3)*y)[0])"

use fast_picard_lefshetz_integral_code::{Solver, SolverConfig, phases::AiryPhase};

/// y in [-1, 1.5] at omega = 1: saddles within |z| <= 1.2, pruning rim at
/// |x| ~ (3·30/ω)^{1/3} ≈ 4.5, hence the (-6, 6) box. The saddle Hessian
/// |φ''| = 2√y stretches the mesh like e^{2λ}, so keep the flow time at
/// λ = 1 and the spacing fine (0.04 leaves the y = 1 case at 3e-8).
fn airy_cfg() -> SolverConfig {
    SolverConfig {
        domain: vec![(-6.0, 6.0)],
        delta: 0.025,
        flow_steps: 20,
        ..SolverConfig::default()
    }
}

fn check(y: f64, cfg: SolverConfig, expected: f64, tol: f64) {
    let omega = cfg.omega;
    let phase = AiryPhase { y };
    let out = Solver::<1>::new(&phase, cfg).run().unwrap();
    let rel = (out.value - expected).norm() / expected.abs();
    eprintln!(
        "airy y={y}, omega={omega}: rel err {rel:.3e} (estimate {:.3e})",
        out.error / expected.abs()
    );
    assert!(
        rel < tol,
        "y={y}, omega={omega}: got {}, expected {expected}, rel err {rel:.3e}",
        out.value
    );
    // φ is odd on the real axis, so the exact integral is real.
    assert!(
        out.value.im.abs() < tol * expected.abs(),
        "y={y}, omega={omega}: spurious imaginary part {:.3e}",
        out.value.im
    );
}

#[test]
fn two_real_images() {
    check(1.0, airy_cfg(), 3.365_028_273_002_628_2, 1e-8);
}

#[test]
fn at_the_caustic() {
    check(0.0, airy_cfg(), 2.230_707_051_824_495_6, 1e-8);
}

#[test]
fn no_real_images() {
    check(-1.0, airy_cfg(), 0.850_067_322_349_920_66, 1e-8);
}

#[test]
fn higher_frequency() {
    let cfg = SolverConfig {
        domain: vec![(-5.0, 5.0)],
        delta: 0.02,
        flow_steps: 20,
        omega: 4.0,
        ..SolverConfig::default()
    };
    // Past the first Airy zero: the exact value is negative.
    check(1.0, cfg, -0.497_688_158_651_127_14, 1e-6);
}
