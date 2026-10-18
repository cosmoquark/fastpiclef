//! The full 2D point-mass-field solver (`slice_field`, option A milestone A1):
//! the RC construction in both variables — outer core with breakpoints at the
//! pinch heights, rotated outer tails with the inner solved at complex x2.
//!
//! Validation, in order of strength:
//! * the external Nakamura-Deguchi closed form for the isolated unit mass
//!   (the only external truth for the field integral) — full complex value;
//! * regression anchors and two-contour (Cauchy) agreement on the two-mass
//!   test field;
//! * transposed slicing: reflecting the field across the diagonal maps it to
//!   the transposed lens list, so the integral is unchanged — while the slice
//!   decomposition (which variable is inner) is entirely different;
//! * free space, including zero-mass dummy lenses (a zero-coefficient log has
//!   no branch point and must be invisible).

use fast_picard_lefshetz_integral_code::phases::PointMassField;
use fast_picard_lefshetz_integral_code::{SliceConfig, slice_field};
use num_complex::Complex64;
use std::f64::consts::PI;

fn two_mass_field() -> PointMassField {
    PointMassField {
        beta: [0.2, 0.1],
        lenses: vec![(-0.8, 0.5, 0.6), (0.9, -0.4, 0.4)],
    }
}

fn rel(got: Complex64, want: Complex64) -> f64 {
    (got - want).norm() / want.norm()
}

/// Raw 2D integral references for the isolated unit point mass at the origin
/// with source `(y, 0)`, from the Nakamura-Deguchi closed form
/// `I = (2 pi i / w) F e^{i w phi_m}` at 30 digits
/// (comparison/referee.py::slice_raw_from_nd, emitted 2026-10-11 with
/// `--emit-rust-cases`; the phase relation was measured against the
/// brute-force RC field integral on 2026-10-06).
const REF_ND: &[(f64, f64, f64, f64)] = &[
    (0.3, 1.0, 5.05948805778166566e+00, 2.49364109938757785e+01),
    (1.0, 0.5, 1.11515184895840158e+00, 1.06271092478315978e+01),
    (3.0, 0.1, -4.38075403947874253e+00, 4.50828847266479293e+00),
    (10.0, 2.0, 7.40014699628212980e-01, -3.59581721741785224e-02),
    (30.0, 1.0, 6.65314702567186000e-02, -1.95738061172352179e-01),
];

/// Low-omega raw-integral references (the E3 grid), from the same 30-digit
/// Nakamura-Deguchi conversion (`comparison/referee.py::slice_raw_from_nd`,
/// emitted 2026-10-18 for the O2 free-space-subtraction tests).  At
/// omega = 1e-4 the raw value is ~6.3e4 while `F - 1` is ~4.7e-4: these pin
/// the *absolute* integral accuracy at the 1e-13 level that the direct path
/// cannot reach (its quadrature floor is the accumulated `eps |I_free|`).
const REF_LOW_OMEGA: &[(f64, f64, f64, f64)] = &[
    (1.0e-04, 0.1, 2.93016473835276585e+01, 6.28367811051384415e+04),
    (1.0e-04, 1.0, 2.93016473129695356e+01, 6.28367809496174086e+04),
    (1.0e-04, 3.0, 2.93016468840285498e+01, 6.28367796928817334e+04),
    (1.0e-03, 1.0, 2.20828449536341473e+01, 6.28808040701938899e+03),
    (1.0e-02, 3.0, 1.49453014245563089e+01, 6.32940859163893037e+02),
];

fn single_mass(y: f64) -> PointMassField {
    PointMassField {
        beta: [y, 0.0],
        lenses: vec![(0.0, 0.0, 1.0)],
    }
}

#[test]
fn free_space() {
    // No lenses (and zero-mass dummies, which must be invisible: a
    // zero-coefficient log has no branch point): the integral is the bare
    // quadratic one, 2 pi i / omega.
    let empty = PointMassField { beta: [0.3, -0.2], lenses: vec![] };
    let dummies = PointMassField {
        beta: [0.3, -0.2],
        lenses: vec![(0.5, 0.7, 0.0), (-0.4, 0.2, 0.0), (1.1, -0.9, 0.0)],
    };
    for omega in [1.0_f64, 10.0] {
        let want = Complex64::i() * (2.0 * PI / omega);
        let o1 = slice_field(&empty, &SliceConfig { omega, ..Default::default() }).unwrap();
        let o2 = slice_field(&dummies, &SliceConfig { omega, ..Default::default() }).unwrap();
        assert!(rel(o1.value, want) < 1e-13, "omega={omega}: {}", rel(o1.value, want));
        assert_eq!(o1.value, o2.value, "zero-mass lenses must be invisible");
    }
}

#[test]
fn single_mass_matches_nakamura_deguchi() {
    for &(omega, y, re, im) in REF_ND {
        let out = slice_field(&single_mass(y), &SliceConfig { omega, ..Default::default() })
            .unwrap();
        let want = Complex64::new(re, im);
        assert!(
            rel(out.value, want) < 1e-12,
            "ND w={omega}, y={y}: rel={:.3e}",
            rel(out.value, want)
        );
    }
}

#[test]
#[ignore = "slow (about 10^4 inner solves)"]
fn single_mass_matches_nakamura_deguchi_high_omega() {
    let (omega, y) = (100.0, 0.5);
    let out = slice_field(&single_mass(y), &SliceConfig { omega, ..Default::default() }).unwrap();
    let want = Complex64::new(8.95259962335362286e-02, 6.52039497653385536e-02);
    assert!(rel(out.value, want) < 1e-12, "rel={:.3e}", rel(out.value, want));
}

/// O2: the low-omega free-space subtraction.  At `omega <= 1e-2` the direct
/// path's `F - 1` is limited by the quadrature's accumulated `eps |I_free|`;
/// the subtracted path integrates the small
/// `e^{i w phi_free}(e^{-i w Psi} - 1)` on the same contour and adds the
/// exact `2 pi i / w`.  Both routes are graded against the pinned 30-digit
/// values, `F - 1 = |I - I_ref| / (|F_ref - 1| * 2 pi / w)`, and the
/// subtracted budget must cover the true integral error.
#[test]
fn subtract_free_low_omega_matches_the_nd_referee() {
    for &(omega, y, re, im) in REF_LOW_OMEGA {
        let want = Complex64::new(re, im);
        let fref = want * Complex64::new(0.0, -omega / (2.0 * PI));
        let i_scale = (fref - 1.0).norm() * 2.0 * PI / omega;
        let cfg = SliceConfig { omega, ..Default::default() };
        let d = slice_field(&single_mass(y), &cfg).unwrap();
        let s = slice_field(&single_mass(y), &SliceConfig { subtract_free: true, ..cfg }).unwrap();
        let e_d = (d.value - want).norm() / i_scale;
        let e_s = (s.value - want).norm() / i_scale;
        assert!(e_d < 5e-12, "direct w={omega}, y={y}: F-1 rel {e_d:.3e}");
        assert!(e_s < 1e-12, "subtracted w={omega}, y={y}: F-1 rel {e_s:.3e}");
        assert!(
            e_s < e_d,
            "subtraction must beat the direct path at w={omega}, y={y}: {e_s:.3e} vs {e_d:.3e}"
        );
        for o in [&d, &s] {
            let allow = 8.0 * f64::EPSILON * o.value.norm();
            assert!(
                o.error + allow >= (o.value - want).norm(),
                "w={omega}, y={y}: budget {:.3e} does not cover the true error {:.3e}",
                o.error,
                (o.value - want).norm()
            );
        }
    }
}

/// With no lenses the subtracted integrand is identically zero
/// (`Psi == 0`), so the solve returns the closed form `2 pi i / w` exactly.
#[test]
fn subtract_free_empty_field_is_exact() {
    let empty = PointMassField { beta: [0.3, -0.2], lenses: vec![] };
    for omega in [1e-4_f64, 1.0] {
        let o = slice_field(&empty, &SliceConfig { omega, subtract_free: true, ..Default::default() })
            .unwrap();
        let want = Complex64::i() * (2.0 * PI / omega);
        assert_eq!(o.value, want, "omega={omega}");
    }
}

/// Both routes are exact at moderate omega; the subtraction must not disturb
/// them, and its budget must still cover the (quadrature-level) difference.
#[test]
fn subtract_free_matches_direct_at_moderate_omega() {
    let field = two_mass_field();
    for omega in [0.3_f64, 1.0, 3.0] {
        let d = slice_field(&field, &SliceConfig { omega, ..Default::default() }).unwrap();
        let s = slice_field(&field, &SliceConfig { omega, subtract_free: true, ..Default::default() })
            .unwrap();
        let diff = (d.value - s.value).norm();
        assert!(diff / d.value.norm() < 1e-11, "omega={omega}: {:.3e}", diff / d.value.norm());
        let allow = 8.0 * f64::EPSILON * (d.value.norm() + s.value.norm());
        assert!(s.error + allow >= diff, "omega={omega}: budget {:.3e} < {diff:.3e}", s.error);
    }
}

#[test]
fn two_mass_two_contours_agree() {
    // Two outer margins are two contours: Cauchy. Plus the regression
    // anchors measured on 2026-10-06 (Python RC x RC, cross-validated
    // against Nakamura-Deguchi on the single mass).
    let field = two_mass_field();
    let anchors = [
        (3.0, Complex64::new(-0.437797336628663, 2.968066928420613)),
        (10.0, Complex64::new(0.620555704800363, 1.035804262604527)),
    ];
    for (omega, anchor) in anchors {
        let o1 = slice_field(&field, &SliceConfig { omega, ..Default::default() }).unwrap();
        let o2 = slice_field(
            &field,
            &SliceConfig { omega, margin_outer: 3.0, ..Default::default() },
        )
        .unwrap();
        assert!(
            rel(o1.value, o2.value) < 1e-12,
            "two-contour at omega={omega}: {:.3e}",
            rel(o1.value, o2.value)
        );
        assert!(
            rel(o1.value, anchor) < 1e-12,
            "anchor at omega={omega}: {:.3e}",
            rel(o1.value, anchor)
        );
    }
}

#[test]
fn transposed_slicing_agrees() {
    // Reflecting the field across the diagonal (x0, x1) -> (x1, x0) maps it
    // to the transposed lens list and source, and preserves the measure, so
    // the integral is exactly the same — while the slice decomposition (which
    // variable is the inner one, where the breakpoints fall) is entirely
    // different. A construction-level cross-check no single-slicing bug can pass.
    let field = two_mass_field();
    let transposed = PointMassField {
        beta: [0.1, 0.2],
        lenses: vec![(0.5, -0.8, 0.6), (-0.4, 0.9, 0.4)],
    };
    for omega in [3.0_f64, 10.0] {
        let o = slice_field(&field, &SliceConfig { omega, ..Default::default() }).unwrap();
        let ot = slice_field(&transposed, &SliceConfig { omega, ..Default::default() }).unwrap();
        assert!(
            rel(o.value, ot.value) < 1e-12,
            "transposed at omega={omega}: {:.3e}",
            rel(o.value, ot.value)
        );
    }
}

#[test]
fn field_error_is_honest() {
    // Against the exact closed form (no cross-implementation roundoff floor
    // here, unlike the slice referee comparisons): the estimate must cover
    // the true error and not be absurdly loose (measured margins 12-180x).
    for &(omega, y, re, im) in REF_ND {
        let out = slice_field(&single_mass(y), &SliceConfig { omega, ..Default::default() })
            .unwrap();
        let want = Complex64::new(re, im);
        let true_err = (out.value - want).norm();
        assert!(
            out.error >= true_err,
            "estimate {:+.3e} below true error {:+.3e} at omega={omega}",
            out.error,
            true_err
        );
        assert!(
            out.error < 1e-10 * want.norm().max(1.0),
            "estimate absurdly loose ({:.3e}) at omega={omega}",
            out.error
        );
    }
}

#[test]
fn parallel_equals_serial() {
    let field = two_mass_field();
    let cfg = SliceConfig { omega: 3.0, ..Default::default() };
    let serial = slice_field(&field, &SliceConfig { parallel: false, ..cfg.clone() }).unwrap();
    let parallel = slice_field(&field, &cfg).unwrap();
    assert_eq!(serial.value, parallel.value);
    assert_eq!(serial.error, parallel.error);
    assert_eq!(serial.inner_solves, parallel.inner_solves);
}

#[test]
fn large_total_mass_two_contours() {
    // Total mass 3 at omega = 1: the log-argument growth eats into the tail
    // decay margins, exactly the regime the decay probe guards. Two contours
    // must still agree (and the default tail length must have sufficed).
    let field = PointMassField {
        beta: [0.2, 0.1],
        lenses: vec![(-0.5, 0.3, 1.5), (0.5, -0.3, 1.5)],
    };
    let o1 = slice_field(&field, &SliceConfig::default()).unwrap();
    let o2 = slice_field(&field, &SliceConfig { margin_outer: 3.0, ..Default::default() })
        .unwrap();
    assert!(
        rel(o1.value, o2.value) < 1e-12,
        "two-contour under large total mass: {:.3e}",
        rel(o1.value, o2.value)
    );
}

#[test]
fn slice_field_rejects_bad_input() {
    let field = two_mass_field();
    for omega in [0.0, -1.0, f64::NAN] {
        let err = slice_field(&field, &SliceConfig { omega, ..Default::default() }).unwrap_err();
        assert!(matches!(err, fast_picard_lefshetz_integral_code::Error::InvalidInput(_)));
    }
    let bad_margin = SliceConfig { margin: 0.0, ..Default::default() };
    assert!(matches!(
        slice_field(&field, &bad_margin).unwrap_err(),
        fast_picard_lefshetz_integral_code::Error::InvalidInput(_)
    ));
    let nan_lens = PointMassField { beta: [0.0, 0.0], lenses: vec![(f64::NAN, 0.0, 1.0)] };
    assert!(matches!(
        slice_field(&nan_lens, &SliceConfig::default()).unwrap_err(),
        fast_picard_lefshetz_integral_code::Error::InvalidInput(_)
    ));
}
