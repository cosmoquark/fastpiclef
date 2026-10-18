//! The polar 2D wrapper against exact answers. The three cases cover the
//! three structural paths of the half-line radial solver:
//!  - centred quadratic: the radial stationary point sits exactly at the
//!    endpoint r = 0 for every theta (single descending branch);
//!  - offset quadratic: ordinary endpoint flow line, plus a real radial
//!    saddle at r = beta.n(theta) that is effective only on the half of the
//!    circle where it is positive (exact answer is translation-invariant:
//!    2 pi i / omega);
//!  - anisotropic quadratic: endpoint-saddle again but with a
//!    theta-dependent radial scale, so the theta quadrature does real work
//!    (exact answer 2 pi i / (omega sqrt(a b))).

use fast_picard_lefshetz_integral_code::{Phase, PolarConfig, PolarSolver};
use nalgebra::SVector;
use num_complex::Complex64;
use std::f64::consts::TAU;

struct OffsetQuad {
    bx: f64,
    by: f64,
}

impl Phase<2> for OffsetQuad {
    fn value(&self, z: &SVector<Complex64, 2>) -> Complex64 {
        let u = z[0] - self.bx;
        let v = z[1] - self.by;
        0.5 * (u * u + v * v)
    }
    fn gradient(&self, z: &SVector<Complex64, 2>) -> SVector<Complex64, 2> {
        SVector::<Complex64, 2>::new(z[0] - self.bx, z[1] - self.by)
    }
}

struct AnisoQuad {
    a: f64,
    b: f64,
}

impl Phase<2> for AnisoQuad {
    fn value(&self, z: &SVector<Complex64, 2>) -> Complex64 {
        0.5 * (self.a * z[0] * z[0] + self.b * z[1] * z[1])
    }
    fn gradient(&self, z: &SVector<Complex64, 2>) -> SVector<Complex64, 2> {
        SVector::<Complex64, 2>::new(self.a * z[0], self.b * z[1])
    }
}

fn check(phase: &dyn Phase<2>, cfg: PolarConfig, expected: Complex64, tol: f64) {
    let mut solver = PolarSolver::new(phase, cfg);
    let out = solver.run().unwrap();
    let rel = (out.value - expected).norm() / expected.norm();
    eprintln!(
        "polar: got {}, expected {expected}, rel err {rel:.3e} (estimate {:.3e}), {} thetas",
        out.value,
        out.error / expected.norm(),
        solver.thetas_used(),
    );
    assert!(
        rel < tol,
        "got {}, expected {expected}, rel err {rel:.3e}",
        out.value
    );
}

#[test]
fn centred_quadratic() {
    let cfg = PolarConfig {
        theta_init: 4,
        theta_max: 64,
        ..PolarConfig::default()
    };
    check(
        &OffsetQuad { bx: 0.0, by: 0.0 },
        cfg,
        Complex64::new(0.0, TAU),
        1e-10,
    );
}

#[test]
fn offset_quadratic() {
    check(
        &OffsetQuad { bx: 0.7, by: 0.3 },
        PolarConfig::default(),
        Complex64::new(0.0, TAU),
        1e-8,
    );
}

#[test]
fn anisotropic_quadratic() {
    let cfg = PolarConfig {
        theta_init: 8,
        theta_max: 256,
        ..PolarConfig::default()
    };
    check(
        &AnisoQuad { a: 1.0, b: 2.0 },
        cfg,
        Complex64::new(0.0, TAU / 2.0_f64.sqrt()),
        1e-9,
    );
}

#[test]
fn parallel_equals_serial_bitwise() {
    use fast_picard_lefshetz_integral_code::{ThimbleConfig, solve_point_mass_polar};

    let phase = AnisoQuad { a: 1.0, b: 2.0 };
    let ser_cfg = PolarConfig {
        theta_init: 8,
        theta_max: 256,
        parallel: false,
        ..PolarConfig::default()
    };
    let par_cfg = PolarConfig {
        parallel: true,
        ..ser_cfg.clone()
    };
    let mut s_ser = PolarSolver::new(&phase, ser_cfg);
    let mut s_par = PolarSolver::new(&phase, par_cfg);
    let o_ser = s_ser.run().unwrap();
    let o_par = s_par.run().unwrap();
    assert_eq!(o_ser.value.re.to_bits(), o_par.value.re.to_bits());
    assert_eq!(o_ser.value.im.to_bits(), o_par.value.im.to_bits());
    assert_eq!(o_ser.error.to_bits(), o_par.error.to_bits());
    assert_eq!(s_ser.thetas_used(), s_par.thetas_used());

    let pm_ser_cfg = PolarConfig {
        radial: ThimbleConfig { omega: 10.0, ..Default::default() },
        parallel: false,
        ..PolarConfig::default()
    };
    let pm_par_cfg = PolarConfig {
        parallel: true,
        ..pm_ser_cfg.clone()
    };
    let (pm_ser, n_ser) = solve_point_mass_polar(0.5, &pm_ser_cfg).unwrap();
    let (pm_par, n_par) = solve_point_mass_polar(0.5, &pm_par_cfg).unwrap();
    assert_eq!(pm_ser.value.re.to_bits(), pm_par.value.re.to_bits());
    assert_eq!(pm_ser.value.im.to_bits(), pm_par.value.im.to_bits());
    assert_eq!(pm_ser.error.to_bits(), pm_par.error.to_bits());
    assert_eq!(n_ser, n_par);
}
