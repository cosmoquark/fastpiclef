//! P4.1: the high-omega hybrid — geometric optics with the stationary-phase
//! next-order correction, plus the exact RC fallback.
//!
//! Graded against: the external Nakamura-Deguchi closed form (point mass, |F|;
//! emitted 2026-10-12 with `comparison/referee.py::point_mass_exact`, mpmath
//! dps 30), the exact Gaussian closed form for quadratic phases, and the exact
//! RC engine of this crate (the correction must improve on the bare geometric
//! sum, and the reported estimate must cover the corrected residual — the
//! coverage claim of `comparison/sp_correction.py`, re-checked on the Rust
//! tensors).
//!
//! Convention: [`lens_field_geometric`] and [`lens_field_geometric_corrected`]
//! return the *amplification* `F`; [`lens_field_solve`] and the hybrids return
//! the raw integral `I = F * (2 pi i / omega)`.

use fast_picard_lefshetz_integral_code::{
    Component, HybridMethod, LensField, Profile, SliceConfig, lens_field_geometric,
    lens_field_geometric_corrected, lens_field_hybrid, lens_field_solve, point_mass_field_hybrid,
};
use num_complex::Complex64;
use std::f64::consts::PI;

fn point(x: f64, y: f64, m: f64) -> Component {
    Component {
        x,
        y,
        profile: Profile::Point { mass: m, xc: 0.0 },
    }
}

fn unit_field(y: f64) -> LensField {
    LensField::new([y, 0.0], 0.0, 0.0, 0.0, vec![point(0.0, 0.0, 1.0)]).unwrap()
}

/// `2 pi i / omega`: integral -> amplification.
fn free2(omega: f64) -> Complex64 {
    Complex64::new(0.0, 2.0 * PI / omega)
}

/// Raw integral -> amplification.
fn amp(i: Complex64, omega: f64) -> Complex64 {
    i / free2(omega)
}

#[test]
fn macro_only_correction_is_exact() {
    // A quadratic phase has C = 0: the corrected sum must equal the geometric
    // sum and the exact Gaussian closed form, with a zero estimate.  Includes
    // a negative-lambda saddle macro image (det < 0).
    for (kappa, g1, g2) in [
        (0.3, 0.2, 0.0),
        (1.5, 0.2, 0.0),
        (0.45, 0.5, 0.0),
        (1.1, 0.3, 0.0),
    ] {
        let field = LensField::new([0.4, -0.3], kappa, g1, g2, vec![]).unwrap();
        for omega in [3.0, 100.0] {
            let g = lens_field_geometric_corrected(&field, omega).unwrap();
            let geo = lens_field_geometric(&field, omega);
            assert!(
                (g.value - geo).norm() < 1e-12,
                "kappa={kappa}: corrected != geometric"
            );
            assert!(g.error < 1e-12, "kappa={kappa}: estimate {:.2e}", g.error);
            let (l0, l2) = (field.lambda0, field.lambda2);
            let kin = field.kc
                - field.beta[0] * field.beta[0] / (2.0 * l0)
                - field.beta[1] * field.beta[1] / (2.0 * l2);
            let refr = (2.0 * PI / (omega * (l0 * l2).abs().sqrt()))
                * Complex64::from_polar(1.0, (l0.signum() + l2.signum()) * PI / 4.0)
                * (Complex64::i() * (omega * kin)).exp();
            let d = (g.value - amp(refr, omega)).norm() / amp(refr, omega).norm();
            assert!(d < 1e-12, "kappa={kappa} omega={omega}: {d:e}");
        }
    }
}

#[test]
fn point_mass_corrected_beats_geometric_vs_nd() {
    // |F| from the external ND closed form (mpmath dps 30).
    const CASES: &[(f64, f64, f64)] = &[
        (30.0, 0.5, 0.659758515563),
        (100.0, 0.5, 1.762705414788),
        (300.0, 0.5, 2.029404153457),
        (30.0, 1.0, 0.987092283349),
        (100.0, 1.0, 1.384645249664),
        (100.0, 2.0, 1.101386272686),
    ];
    for &(omega, y, fnd) in CASES {
        let field = unit_field(y);
        let eg = (lens_field_geometric(&field, omega).norm() - fnd).abs() / fnd;
        let g = lens_field_geometric_corrected(&field, omega).unwrap();
        let ec = (g.value.norm() - fnd).abs() / fnd;
        assert!(
            ec < eg,
            "omega={omega} y={y}: corrected {ec:e} not better than geometric {eg:e}"
        );
        // the estimate covers the corrected residual
        assert!(
            g.error >= ec * fnd,
            "omega={omega} y={y}: estimate {:e} < residual {:e}",
            g.error,
            ec * fnd
        );
    }
}

#[test]
fn corrected_residual_falls_with_omega_and_covers_against_rc() {
    // Against the exact RC engine (complex values, our convention): the
    // correction improves and the estimate covers, at both a mild (y = 0.1)
    // and a strong (y = 0.5) image pair.
    for y in [0.1f64, 0.5] {
        let field = unit_field(y);
        for omega in [30.0f64, 100.0] {
            if y == 0.1 && omega == 100.0 {
                continue; // keep the release suite fast (RC cost ~ omega^1.5)
            }
            let rc =
                lens_field_solve(&field, &SliceConfig { omega, ..Default::default() }).unwrap();
            let truth = amp(rc.value, omega);
            let geo = lens_field_geometric(&field, omega);
            let g = lens_field_geometric_corrected(&field, omega).unwrap();
            let r_geo = (geo - truth).norm();
            let r_corr = (g.value - truth).norm();
            assert!(
                g.error >= r_corr,
                "y={y} omega={omega}: estimate {:e} < residual {r_corr:e}",
                g.error
            );
            assert!(
                r_corr < r_geo,
                "y={y} omega={omega}: corrected {r_corr:e} not better than geometric {r_geo:e}"
            );
        }
    }
}

#[test]
fn hybrid_switches_and_respects_its_budget() {
    let field = unit_field(0.5);
    let omega = 100.0;
    let rc = lens_field_solve(&field, &SliceConfig { omega, ..Default::default() }).unwrap();
    // Tight tolerance: the asymptotic estimate cannot meet it -> exact RC.
    let tight = lens_field_hybrid(&field, omega, 1e-12, None).unwrap();
    assert_eq!(tight.method, HybridMethod::Rc);
    assert!((tight.value - rc.value).norm() < 1e-9 * rc.value.norm());
    // Loose tolerance: the geometric branch, and its budget covers the truth.
    let loose = lens_field_hybrid(&field, omega, 0.05, None).unwrap();
    assert_eq!(loose.method, HybridMethod::GeometricCorrected);
    assert!(loose.images >= 2);
    let resid = (loose.value - rc.value).norm();
    assert!(
        resid <= loose.error,
        "residual {resid:e} > budget {:e}",
        loose.error
    );
    let geo_resid = (lens_field_geometric(&field, omega) * free2(omega) - rc.value).norm();
    assert!(
        resid < geo_resid,
        "correction must improve: {resid:e} vs {geo_resid:e}"
    );
}

#[test]
fn point_mass_field_hybrid_matches_the_lens_field_path() {
    // The point-mass verified image set must give the same corrected sum as
    // the general Newton-image path, and cover against the exact RC.
    let lenses = [(0.6, 0.0, 0.4), (-0.5, 0.4, 0.3), (0.1, -0.6, 0.35)];
    let source = [0.2, 0.1];
    let omega = 100.0;
    let h = point_mass_field_hybrid(source, &lenses, omega, 1.0, None).unwrap();
    assert_eq!(h.method, HybridMethod::GeometricCorrected);
    assert_eq!(h.images, 6);
    let comps: Vec<Component> = lenses.iter().map(|&(x, y, m)| point(x, y, m)).collect();
    let field = LensField::new(source, 0.0, 0.0, 0.0, comps).unwrap();
    let g = lens_field_geometric_corrected(&field, omega).unwrap();
    assert!(
        (h.value - g.value * free2(omega)).norm() < 1e-9,
        "paths disagree"
    );
    let rc = lens_field_solve(&field, &SliceConfig { omega, ..Default::default() }).unwrap();
    let resid = (h.value - rc.value).norm();
    assert!(
        resid <= h.error,
        "residual {resid:e} > budget {:e}",
        h.error
    );
    let geo_resid = (lens_field_geometric(&field, omega) * free2(omega) - rc.value).norm();
    assert!(
        resid < geo_resid,
        "correction must improve: {resid:e} vs {geo_resid:e}"
    );
}

#[test]
fn hybrid_falls_back_at_the_point_mass_caustic() {
    // Source exactly on the point-mass caustic (beta = 0): both images sit on
    // the critical curve (det H = 0) and the asymptotic form is invalid; the
    // hybrid must take the exact RC branch (and if the finder still returns a
    // near-caustic image, the estimate must be huge).
    let field = unit_field(0.0);
    let omega = 50.0;
    if let Some(g) = lens_field_geometric_corrected(&field, omega) {
        assert!(
            g.error > 1.0,
            "caustic estimate must be huge, got {:.2e}",
            g.error
        );
    }
    let h = lens_field_hybrid(&field, omega, 1e-3, None).unwrap();
    assert_eq!(h.method, HybridMethod::Rc);
    let rc = lens_field_solve(&field, &SliceConfig { omega, ..Default::default() }).unwrap();
    assert!((h.value - rc.value).norm() < 1e-9 * rc.value.norm());
}
