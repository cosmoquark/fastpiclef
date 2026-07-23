//! The fast axisymmetric point-mass solver (psi = ln r) against the
//! Nakamura-Deguchi closed form. We compare the amplification magnitude |F|,
//! the physical, convention-independent quantity: the complex phase differs by
//! the standard Fermat-potential reference convention, but |F| is unambiguous.
//!
//! Reference |F| values generated 2026-07-23 with mpmath (Takahashi & Nakamura
//! 2003): F = exp[pi w/4 + i(w/2)(ln(w/2) - 2 phi_m)] * Gamma(1 - i w/2)
//! * 1F1(i w/2, 1; i w y^2/2), with x_m = (y + sqrt(y^2+4))/2 and
//! phi_m = (x_m - y)^2/2 - ln(x_m). See comparison/referee.py::point_mass_exact.

use fast_picard_lefshetz_integral_code::{PolarConfig, ThimbleConfig, solve_point_mass_polar};
use std::f64::consts::PI;

/// Amplification magnitude |F(w, y)| from the polar point-mass solver. The raw
/// polar integral divided by the 2D free-space value 2*pi*i/omega gives F, so
/// |F| = |raw| * omega / (2*pi).
fn amp_abs(y: f64, w: f64) -> f64 {
    let cfg = PolarConfig {
        radial: ThimbleConfig {
            omega: w,
            ..Default::default()
        },
        ..Default::default()
    };
    let (out, _n) = solve_point_mass_polar(y, &cfg).unwrap();
    out.value.norm() * w / (2.0 * PI)
}

#[test]
fn point_mass_matches_nakamura_deguchi_high_omega() {
    // Geometric-optics regime: the thimble is essentially exact.
    let cases = [
        (10.0_f64, 1.0_f64, 1.473868738938_f64),
        (20.0, 0.5, 2.015790792103),
        (5.0, 2.0, 0.876108825593),
        (50.0, 0.3, 0.501604439353),
    ];
    for (w, y, fnd) in cases {
        let rel = (amp_abs(y, w) - fnd).abs() / fnd;
        assert!(rel < 1e-6, "w={w} y={y}: rel err vs ND = {rel:e}");
    }
}

#[test]
fn point_mass_robust_and_accurate_at_low_omega() {
    // The r = e^u substitution removes the log branch point, so the trace does
    // not stall in the deep-wave regime where the direct half-line trace does;
    // the sub-dominant (Im = +-pi) saddles keep it accurate there too.
    let cases = [(1.0_f64, 1.0_f64, 1.393245242028_f64), (0.3, 1.0, 1.214885790560)];
    for (w, y, fnd) in cases {
        let rel = (amp_abs(y, w) - fnd).abs() / fnd;
        assert!(rel < 3e-3, "w={w} y={y}: rel err vs ND = {rel:e}");
    }
}
