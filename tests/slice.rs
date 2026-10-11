//! The inner x_0-slice kernels of the Cartesian-slice 2D field solver.
//!
//! Two kernels live side by side (see `src/slice.rs` module docs):
//! * `slice_inner_rc` — the *exact* RC kernel (real core + rotated tails;
//!   A0'), validated here against hard-coded referee values;
//! * `slice_inner` — the thimble kernel (A0), kept as the high-omega path,
//!   with the swept-branch-point indicator wired into its error budget.
//!
//! Reference values were generated 2026-10-11 with
//! `pixi run -e comparison python -m comparison.referee --emit-rust-cases`
//! (margin-4.0 contour; the referee's two-contour self-check agrees to
//! <= 9e-14 on all of them, exact pinches included). Method and provenance:
//! `comparison/referee.py`, Journal entry of 2026-10-11.

use fast_picard_lefshetz_integral_code::phases::PointMassField;
use fast_picard_lefshetz_integral_code::roots::poly_eval;
use fast_picard_lefshetz_integral_code::{
    Error, Output, Phase, SliceConfig, ThimbleConfig, X1SlicePhase, branch_point_indicator,
    slice_inner, slice_inner_rc,
};
use nalgebra::SVector;
use num_complex::Complex64;

fn test_field() -> PointMassField {
    PointMassField {
        beta: [0.2, 0.1],
        lenses: vec![(-0.8, 0.5, 0.6), (0.9, -0.4, 0.4)],
    }
}

fn rc(field: &PointMassField, x2: f64, omega: f64) -> Output {
    slice_inner_rc(field, Complex64::from(x2), &SliceConfig { omega, ..Default::default() })
        .unwrap()
}

fn rel(got: Complex64, want: Complex64) -> f64 {
    (got - want).norm() / want.norm()
}

/// Reference values of the two-mass test field: `(omega, x2, Re G, Im G)`.
/// Far, swept, near-pinch (0.49 / -0.41 / 0.5+1e-6) and caustic-band slices.
const REF_SLICES: &[(f64, f64, f64, f64)] = &[
    (1.0, 2.5, -2.45184952459356431e+00, 1.04663752792442688e+00),
    (1.0, 1.3, 1.21630908527338688e+00, 2.56849199464597877e+00),
    (1.0, -1.1, 1.34744642012360560e+00, 2.62672458721258373e+00),
    (1.0, 0.0, 2.17647837018821821e+00, 2.41528549017134697e+00),
    (1.0, 0.49, 1.52532063469217993e+00, 2.45611935011879501e+00),
    (1.0, -0.41, 1.94144077470878740e+00, 2.62702569039639755e+00),
    (1.0, 0.500001, 1.52051758998006603e+00, 2.44100974471162724e+00),
    (10.0, 2.5, 6.57733248123213987e-02, 8.34763223673227972e-01),
    (10.0, 1.3, -6.82279612639092314e-01, -4.00959290979350236e-01),
    (10.0, -1.1, -3.85652571801710031e-01, -6.76067121313286723e-01),
    (10.0, 0.0, -6.65319240031733705e-02, 5.05175517221173376e-01),
    (10.0, 0.49, 9.24841607010154276e-01, 6.95194362554367062e-01),
    (10.0, -0.41, 4.18355685122539789e-01, 8.38350936653473933e-01),
    (10.0, 0.500001, 9.12591490799860683e-01, 7.14062407427104362e-01),
    (100.0, 2.5, 2.17788760488760082e-03, 2.64743302601447961e-01),
    (100.0, 1.3, -6.29734857923873181e-02, -2.42644473601647759e-01),
    (100.0, -1.1, -2.51500862452140039e-01, 4.80857390304946578e-02),
    (100.0, 0.0, -1.96212848670135942e-01, 1.08721209764971547e-02),
    (100.0, 0.49, -1.15308961014625289e-01, -1.13720924933420064e-01),
    (100.0, -0.41, -1.17146736381274541e-01, 4.82761892165415943e-02),
    (100.0, 0.500001, -1.48897047600222127e-01, -1.35794476407948422e-01),
];

/// Reference values at complex slice positions on the rotated rays (where the
/// outer integral's tails evaluate the inner): `(omega, Re x2, Im x2, Re G, Im G)`.
/// The omega=10, s=1.0 entries sit deep in the tail (|G| ~ 1e-8) and still
/// reproduce to 1e-12 relative.
const REF_COMPLEX: &[(f64, f64, f64, f64, f64)] = &[
    (1.0, 2.21213203435596428e+00, 2.12132034355964227e-01, -1.04933074411841698e+00, 1.55280700138499772e+00),
    (1.0, -2.11213203435596419e+00, -2.12132034355964227e-01, -1.24964291487570867e+00, 1.34277929745437574e+00),
    (1.0, 2.70710678118654746e+00, 7.07106781186547462e-01, -5.13270815996709984e-01, 1.33275963325121455e-01),
    (1.0, -2.60710678118654737e+00, -7.07106781186547462e-01, -4.88999139166471031e-01, 2.01470589304175857e-02),
    (10.0, 2.21213203435596428e+00, 2.12132034355964227e-01, -1.13958127422021712e-02, 1.86281307007316009e-02),
    (10.0, -2.11213203435596419e+00, -2.12132034355964227e-01, -1.46011729532669055e-02, -1.10394104827856616e-02),
    (10.0, 2.70710678118654746e+00, 7.07106781186547462e-01, -9.15074979559037493e-08, -4.59044737180429689e-09),
    (10.0, -2.60710678118654737e+00, -7.07106781186547462e-01, 2.51964787634845882e-08, -3.50878398800075110e-08),
];

#[test]
fn slice_polynomial_identity() {
    let field = test_field();
    for &x2 in &[1.3_f64, -0.7, 0.0] {
        let slice = X1SlicePhase {
            field: &field,
            x2: Complex64::from(x2),
        };
        let (n, d) = slice.stationary_fraction();
        // deg N = 2 * n_masses + 1, deg D = 2 * n_masses.
        assert_eq!(n.len(), 2 * field.lenses.len() + 2);
        assert_eq!(d.len(), 2 * field.lenses.len() + 1);
        for &(re, im) in &[(0.3_f64, 0.4_f64), (-1.1, 0.6), (2.0, -1.3)] {
            let z = Complex64::new(re, im);
            let phi_prime = slice.gradient(&SVector::<Complex64, 1>::new(z))[0];
            let lhs = poly_eval(&n, z);
            let rhs = phi_prime * poly_eval(&d, z);
            let err = (lhs - rhs).norm();
            assert!(
                err < 1e-9 * (1.0 + lhs.norm()),
                "N != phi' D at x2={x2}, z={z}: |diff|={err:.3e}"
            );
        }
    }
}

#[test]
fn inner_rc_matches_referee() {
    let field = test_field();
    for &(omega, x2, re, im) in REF_SLICES {
        let g = rc(&field, x2, omega).value;
        let want = Complex64::new(re, im);
        assert!(
            rel(g, want) < 1e-12,
            "RC vs referee at x2={x2}, omega={omega}: rel={:.3e}",
            rel(g, want)
        );
    }
}

#[test]
fn inner_rc_complex_x2_matches_referee() {
    let field = test_field();
    for &(omega, x2re, x2im, re, im) in REF_COMPLEX {
        let out = slice_inner_rc(
            &field,
            Complex64::new(x2re, x2im),
            &SliceConfig { omega, ..Default::default() },
        )
        .unwrap();
        let want = Complex64::new(re, im);
        assert!(
            rel(out.value, want) < 1e-12,
            "RC vs referee at x2={x2re}+{x2im}i, omega={omega}: rel={:.3e}",
            rel(out.value, want)
        );
    }
}

#[test]
fn inner_rc_two_contours_agree() {
    // Two different core margins are two different contours; their agreement
    // is the Cauchy self-check, and it also catches any wrong-sheet log on
    // the rotated tails (the failure mode would be margin-dependent).
    let field = test_field();
    for &omega in &[1.0, 10.0, 100.0] {
        for k in 0..=24i32 {
            let x2 = -3.0 + 0.25 * k as f64;
            // exact pinches: the branch point sits on RR (measure-zero for
            // the outer integral; the near-pinch 0.5+1e-6 is covered above)
            if (x2 - 0.5).abs() < 1e-9 || (x2 + 0.4).abs() < 1e-9 {
                continue;
            }
            let g1 = slice_inner_rc(
                &field,
                Complex64::from(x2),
                &SliceConfig { omega, margin: 1.5, ..Default::default() },
            )
            .unwrap();
            let g2 = slice_inner_rc(
                &field,
                Complex64::from(x2),
                &SliceConfig { omega, margin: 4.0, ..Default::default() },
            )
            .unwrap();
            assert!(
                rel(g1.value, g2.value) < 1e-12,
                "two-contour disagreement at x2={x2}, omega={omega}: {:.3e}",
                rel(g1.value, g2.value)
            );
        }
    }
}

#[test]
fn inner_rc_error_is_honest() {
    // The estimate must cover the kernel's true error against the referee —
    // up to the referee's own roundoff floor: the two implementations differ
    // by a few ulps of |G| (~1e-15 x |G|, measured 2026-10-11), which the
    // 1e-14*|G| slack below absorbs. And it must not be absurdly loose.
    let field = test_field();
    for &(omega, x2, re, im) in REF_SLICES {
        let out = rc(&field, x2, omega);
        let want = Complex64::new(re, im);
        let true_err = (out.value - want).norm();
        assert!(
            out.error + 1e-14 * want.norm() >= true_err,
            "error estimate {:+.3e} below true error {:+.3e} at x2={x2}, omega={omega}",
            out.error,
            true_err
        );
        assert!(
            out.error < 1e-10 * want.norm().max(1.0),
            "error estimate absurdly loose ({:.3e}) at x2={x2}, omega={omega}",
            out.error
        );
    }
}

#[test]
fn inner_far_slice_exact() {
    // Where no branch point is swept, the thimble kernel is still exact — and
    // now checked against the RC kernel (the mesh oracle was retired: it
    // shares X1SlicePhase with the thimble, so it was never independent).
    let field = test_field();
    for &(x2, omega) in &[(4.0_f64, 1.0), (4.0, 3.0), (8.0, 1.0), (-5.0, 2.0)] {
        let thimble = slice_inner(
            &field,
            x2,
            &ThimbleConfig { omega, ..Default::default() },
        )
        .unwrap();
        let want = rc(&field, x2, omega).value;
        assert!(
            rel(thimble.value, want) < 1e-11,
            "thimble vs RC on far slice x2={x2}, omega={omega}: {:.3e}",
            rel(thimble.value, want)
        );
    }

    // A single isolated unit mass at the origin, slice well clear of it.
    let one = PointMassField {
        beta: [0.3, 0.0],
        lenses: vec![(0.0, 0.0, 1.0)],
    };
    let thimble = slice_inner(&one, 3.0, &ThimbleConfig::default()).unwrap();
    let want = rc(&one, 3.0, 1.0).value;
    assert!(rel(thimble.value, want) < 1e-11);
}

#[test]
fn thimble_error_now_covers_swept_slices() {
    // The thimble kernel's error budget (quadrature estimate + 100x the
    // swept-branch-point indicator) must cover its true discrepancy against
    // the exact RC kernel everywhere on the sweep. The 1e-13 slack covers the
    // ulp-level edge where the indicator is ~0 (far slices) and both kernels
    // are at their quadrature floors.
    let field = test_field();
    for &omega in &[1.0, 3.0, 10.0, 30.0] {
        for k in 0..=60i32 {
            let x2 = -3.0 + 0.1 * k as f64;
            if (x2 - 0.5).abs() < 1e-9 || (x2 + 0.4).abs() < 1e-9 {
                continue;
            }
            let thimble = slice_inner(
                &field,
                x2,
                &ThimbleConfig { omega, ..Default::default() },
            )
            .unwrap();
            let want = rc(&field, x2, omega).value;
            let diff = (thimble.value - want).norm();
            assert!(
                thimble.error + 1e-13 >= diff,
                "thimble error {:+.3e} does not cover |thimble - RC| = {:+.3e} \
                 at x2={x2}, omega={omega}",
                thimble.error,
                diff
            );
        }
    }
}

#[test]
fn thimble_matches_rc_where_admitted() {
    // The routing rule of the future high-omega path: where
    // 100 * indicator < 1e-12 * |G|, the thimble must agree with the exact
    // kernel (measured 2026-10-06: the rule never admitted a wrong slice).
    let field = test_field();
    let omega = 100.0;
    let mut admitted = 0;
    for k in 0..=24i32 {
        let x2 = -3.0 + 0.25 * k as f64;
        if (x2 - 0.5).abs() < 1e-9 {
            continue;
        }
        let want = rc(&field, x2, omega).value;
        let ind = branch_point_indicator(&field, x2, omega);
        if 100.0 * ind >= 1e-12 * want.norm() {
            continue; // not admitted: the thimble is not trusted here
        }
        admitted += 1;
        let thimble = slice_inner(
            &field,
            x2,
            &ThimbleConfig { omega, ..Default::default() },
        )
        .unwrap();
        assert!(
            rel(thimble.value, want) < 1e-11,
            "admitted slice wrong: x2={x2}, rel={:.3e}, indicator={ind:.3e}",
            rel(thimble.value, want)
        );
    }
    assert!(admitted >= 5, "routing rule admitted only {admitted} slices — vacuous test");
}

#[test]
fn inner_swept_branch_point() {
    // The historical option-A target, un-ignored: a mid slice where the
    // deformation to the thimbles sweeps a branch point. The thimble kernel
    // misses the branch-cut (Hankel) contribution there (~1e-1 at omega=1);
    // the RC kernel never sweeps the branch point, so it is exact.
    let field = test_field();
    let out = rc(&field, 1.3, 1.0);
    let want = Complex64::new(1.21630908527338688e+00, 2.56849199464597877e+00);
    assert!(rel(out.value, want) < 1e-12, "rel={:.3e}", rel(out.value, want));
}

#[test]
fn inner_rc_rejects_bad_input() {
    let field = test_field();
    // Non-positive / non-finite omega.
    for omega in [0.0, -1.0, f64::NAN] {
        let err = slice_inner_rc(
            &field,
            Complex64::from(1.3),
            &SliceConfig { omega, ..Default::default() },
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)), "omega={omega}: {err:?}");
    }
    // Complex x2 outside the validated sector: for b=0.5,
    // Im((0.2+0.5i-0.5)^2) = -0.3 <= 0 — the principal log would not be the
    // analytic continuation along the contour.
    let err = slice_inner_rc(
        &field,
        Complex64::new(0.2, 0.5),
        &SliceConfig::default(),
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
    // ... and a valid complex position goes through.
    slice_inner_rc(&field, Complex64::new(2.2, 0.2), &SliceConfig::default()).unwrap();
}
