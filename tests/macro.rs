//! P2a: the general lens model (`LensField`: macro kappa/gamma + star/halo
//! profiles) solved by the RC engine.
//!
//! Graded against: the external mass-sheet reduction of the Nakamura-Deguchi
//! closed form (kappa = 0.3, unit point mass — emitted from
//! `comparison.referee.mass_sheet_nd`, mpmath dps 30, 2026-10-12); the
//! independent polar (Hankel) referee for the axisymmetric profiles; and the
//! contour-family / margin Cauchy checks for the macro-image types, including
//! the negative-lambda saddle cases.

use fast_picard_lefshetz_integral_code::{
    Component, FieldBatch, LensField, Profile, SliceConfig, lens_field_batch,
    lens_field_geometric, lens_field_solve,
};
use num_complex::Complex64;
use std::f64::consts::FRAC_PI_8;

fn point(x: f64, y: f64, m: f64) -> Component {
    Component {
        x,
        y,
        profile: Profile::Point { mass: m, xc: 0.0 },
    }
}

fn rel(a: Complex64, b: Complex64) -> f64 {
    (a - b).norm() / b.norm()
}

#[test]
fn mass_sheet_matches_nd_external() {
    // A unit point mass under a convergence sheet (gamma = 0): the exact
    // mass-sheet reduction of the ND closed form (external truth).
    const CASE: &[(f64, f64, f64)] = &[
        (1.00000000000000000e+00, 4.93258324657320291e+00, 1.40040264864224469e+01),
        (3.00000000000000000e+00, 4.93236935596103043e-01, 3.29533014326382201e+00),
        (1.00000000000000000e+01, 7.74873674236865317e-02, -9.61549134939347239e-01),
    ];
    for &(omega, re, im) in CASE {
        let field =
            LensField::new([0.5, 0.0], 0.3, 0.0, 0.0, vec![point(0.0, 0.0, 1.0)]).unwrap();
        let out =
            lens_field_solve(&field, &SliceConfig { omega, ..Default::default() }).unwrap();
        let want = Complex64::new(re, im);
        let d = rel(out.value, want);
        assert!(d < 1e-11, "omega={omega}: {d:.3e}");
        // honest budget (few ulps of slack at the roundoff floor)
        let allow = 8.0 * f64::EPSILON * out.value.norm();
        assert!(out.error + allow >= (out.value - want).norm());
    }
}

#[test]
#[allow(clippy::type_complexity)]
fn profiles_match_polar_referee() {
    // Single centred profiles vs the high-precision polar (Hankel) referee —
    // two independent methods (emitted 2026-10-12, dps 45).
    let cases: &[(&str, Profile, &[(f64, f64, f64)])] = &[
        (
            "sis",
            Profile::Sis { psi0: 1.0 },
            &[
                (1.00000000000000000e+00, 1.26188000730648717e+01, 5.25436978178415348e+00),
                (3.00000000000000000e+00, 2.21863952037528067e+00, -3.87023745646334216e+00),
                (1.00000000000000000e+01, -8.63989430613479042e-06, -1.03129618473873852e+00),
            ],
        ),
        (
            "cis",
            Profile::Cis { psi0: 1.0, xc: 0.2 },
            &[
                (1.00000000000000000e+00, 9.96933479082813534e+00, 7.68986381206810599e+00),
                (3.00000000000000000e+00, 4.09430458297971978e+00, -1.97359398345652215e+00),
                (1.00000000000000000e+01, 1.13983719668221517e+00, 6.23555853631252432e-01),
            ],
        ),
        (
            "nfw",
            Profile::Nfw { psi0: 1.0, xs: 0.5 },
            &[
                (1.00000000000000000e+00, 5.66280188650743987e+00, 8.05452665194169271e+00),
                (3.00000000000000000e+00, 3.58312030459172171e+00, 1.36148298130112599e+00),
                (1.00000000000000000e+01, -8.55397375311740871e-01, -1.65036861938967716e-01),
            ],
        ),
    ];
    for (name, profile, table) in cases {
        for &(omega, re, im) in *table {
            let field = LensField::new(
                [0.5, 0.0],
                0.0,
                0.0,
                0.0,
                vec![Component { x: 0.0, y: 0.0, profile: *profile }],
            )
            .unwrap();
            let out =
                lens_field_solve(&field, &SliceConfig { omega, ..Default::default() }).unwrap();
            let want = Complex64::new(re, im);
            let d = rel(out.value, want);
            assert!(d < 1e-11, "{name} omega={omega}: {d:.3e}");
        }
    }
}

#[test]
fn macro_alpha_family_and_margins_agree() {
    // All four macro-image types on the two-mass test field: the construction
    // is exact for any tail angle in (0, pi/4] and any core margin, so the
    // family/margin variations are genuine two-contour Cauchy checks.
    let comps = vec![point(-0.8, 0.5, 0.6), point(0.9, -0.4, 0.4)];
    for (kappa, g1) in [(0.3, 0.2), (0.5, 0.7), (1.5, 0.2), (0.45, 0.5)] {
        let solve = |cfg: SliceConfig| {
            let f = LensField::new([0.2, 0.1], kappa, g1, 0.0, comps.clone()).unwrap();
            lens_field_solve(&f, &cfg).unwrap().value
        };
        let base = SliceConfig { omega: 3.0, ..Default::default() };
        let v0 = solve(base.clone());
        let va = solve(SliceConfig { tail_angle: FRAC_PI_8, ..base.clone() });
        let vo = solve(SliceConfig { margin_outer: 3.0, ..base.clone() });
        let vi = solve(SliceConfig { margin: 2.5, ..base.clone() });
        let scale = v0.norm().max(1.0);
        for (name, v) in [("alpha", va), ("margin_outer", vo), ("margin", vi)] {
            let d = (v0 - v).norm() / scale;
            assert!(d < 1e-11, "k={kappa} g={g1} {name}: {d:.3e}");
        }
    }
}

#[test]
fn macro_transposed_agrees() {
    // The reflection (x0, x2) -> (x2, x0) maps gamma1 -> -gamma1 with swapped
    // component centres and source; the integral is invariant.
    let comps = vec![point(-0.8, 0.5, 0.6), point(0.9, -0.4, 0.4)];
    let swapped: Vec<Component> = comps
        .iter()
        .map(|c| Component { x: c.y, y: c.x, profile: c.profile })
        .collect();
    let cfg = SliceConfig { omega: 3.0, ..Default::default() };
    for (kappa, g1) in [(0.3, 0.2), (0.5, 0.7)] {
        let a = lens_field_solve(
            &LensField::new([0.2, 0.1], kappa, g1, 0.0, comps.clone()).unwrap(),
            &cfg,
        )
        .unwrap()
        .value;
        let b = lens_field_solve(
            &LensField::new([0.1, 0.2], kappa, -g1, 0.0, swapped.clone()).unwrap(),
            &cfg,
        )
        .unwrap()
        .value;
        let d = (a - b).norm() / a.norm().max(1.0);
        assert!(d < 1e-11, "k={kappa} g={g1} transposed: {d:.3e}");
    }
}

#[test]
fn macro_batch_matches_scalar() {
    // The batched LensField entry point against the scalar solver, on a small
    // source grid with a macro model + two point components.
    let comps = vec![point(-0.8, 0.5, 0.6), point(0.9, -0.4, 0.4)];
    let field = LensField::new([0.2, 0.1], 0.3, 0.2, 0.0, comps).unwrap();
    let b0s: Vec<f64> = (0..9).map(|i| -1.0 + 0.25 * i as f64).collect();
    let b1s: Vec<f64> = (0..5).map(|i| -1.0 + 0.5 * i as f64).collect();
    let cfg = SliceConfig { omega: 1.0, ..Default::default() };
    let out = lens_field_batch(
        &field,
        &FieldBatch {
            omegas: vec![1.0, 10.0],
            beta0s: b0s.clone(),
            beta1s: b1s.clone(),
        },
        &cfg,
    )
    .unwrap();
    let mut scale = 0.0f64;
    for v in &out.values {
        scale = scale.max(v.norm());
    }
    let mut worst = 0.0f64;
    for (j, &w) in [1.0f64, 10.0].iter().enumerate() {
        for (i0, &b0) in b0s.iter().enumerate() {
            for (i1, &b1) in b1s.iter().enumerate() {
                let f = LensField::new(
                    [b0, b1],
                    0.3,
                    0.2,
                    0.0,
                    vec![point(-0.8, 0.5, 0.6), point(0.9, -0.4, 0.4)],
                )
                .unwrap();
                let sc = lens_field_solve(&f, &SliceConfig { omega: w, ..Default::default() })
                    .unwrap()
                    .value;
                let idx = (j * b0s.len() + i0) * b1s.len() + i1;
                worst = worst.max((out.values[idx] - sc).norm() / scale);
            }
        }
    }
    assert!(worst < 1e-11, "macro batch vs scalar worst {worst:.3e}");
}

#[test]
fn geometric_limit_sharpens_with_omega() {
    // Away from caustics the exact integral converges to the stationary-phase
    // sum; the deviation must shrink roughly like 1/omega between two
    // frequencies (an order check, not a precision claim).
    let comps = vec![point(-0.8, 0.5, 0.6), point(0.9, -0.4, 0.4)];
    let field = LensField::new([0.2, 0.1], 0.3, 0.2, 0.0, comps).unwrap();
    let d = |omega: f64| {
        let exact = lens_field_solve(&field, &SliceConfig { omega, ..Default::default() })
            .unwrap()
            .value;
        let f_exact = exact * omega / (2.0 * std::f64::consts::PI * Complex64::i());
        let f_geo = lens_field_geometric(&field, omega).unwrap();
        (f_exact - f_geo).norm()
    };
    let (d50, d100) = (d(50.0), d(100.0));
    assert!(d100 < d50, "geometric deviation did not shrink: {d50:.3e} -> {d100:.3e}");
    // order check with O(1) prefactors: measured 9.6e-2 at omega=100 (and
    // it shrinks with omega, asserted above)
    assert!(d100 < 0.2, "geometric deviation at omega=100 too large: {d100:.3e}");
}

#[test]
fn macro_batch_saddle_types_match_scalar() {
    // Negative lambda (saddle macro images) exercise the batch's sign-aware
    // tail directions — the case that was silently wrong before the fix.
    let comps = vec![point(-0.8, 0.5, 0.6), point(0.9, -0.4, 0.4)];
    for (kappa, g1) in [(0.5, 0.7), (1.5, 0.2)] {
        let field = LensField::new([0.2, 0.1], kappa, g1, 0.0, comps.clone()).unwrap();
        let b0s: Vec<f64> = (0..9).map(|i| -1.0 + 0.25 * i as f64).collect();
        let b1s: Vec<f64> = (0..5).map(|i| -1.0 + 0.5 * i as f64).collect();
        let out = lens_field_batch(
            &field,
            &FieldBatch { omegas: vec![3.0], beta0s: b0s.clone(), beta1s: b1s.clone() },
            &SliceConfig { omega: 3.0, ..Default::default() },
        )
        .unwrap();
        let mut scale = 0.0f64;
        for v in &out.values {
            scale = scale.max(v.norm());
        }
        let mut worst = 0.0f64;
        for (i0, &b0) in b0s.iter().enumerate() {
            for (i1, &b1) in b1s.iter().enumerate() {
                let f = LensField::new([b0, b1], kappa, g1, 0.0, comps.clone()).unwrap();
                let sc = lens_field_solve(&f, &SliceConfig { omega: 3.0, ..Default::default() })
                    .unwrap()
                    .value;
                worst = worst.max((out.values[i0 * b1s.len() + i1] - sc).norm() / scale);
            }
        }
        assert!(worst < 1e-11, "k={kappa} g={g1}: batch vs scalar {worst:.3e}");
    }
}

#[test]
fn rejects_critical_and_invalid() {
    // lambda = 0 (critical macro model, including float-rounded critical pairs
    // like kappa = 0.3, gamma1 = 0.7 where 1 - k - g = -1.11e-16) and
    // near-critical |lambda| < 1e-8 are rejected, as are bad profiles.
    assert!(LensField::new([0.0, 0.0], 1.0, 0.0, 0.0, vec![]).is_err());
    assert!(LensField::new([0.0, 0.0], 0.5, 0.5, 0.0, vec![]).is_err());
    assert!(LensField::new([0.4, -0.3], 0.3, 0.7, 0.0, vec![]).is_err());
    assert!(LensField::new([0.4, -0.3], 1.0 - 1e-10, 0.0, 0.0, vec![]).is_err());
    assert!(
        LensField::new([0.0, 0.0], 0.3, 0.0, 0.0, vec![point(f64::NAN, 0.0, 1.0)]).is_err()
    );
    assert!(
        LensField::new(
            [0.0, 0.0],
            0.3,
            0.0,
            0.0,
            vec![Component { x: 0.0, y: 0.0, profile: Profile::Nfw { psi0: 1.0, xs: 0.0 } }],
        )
        .is_err()
    );
}

/// O2: the low-omega free-space subtraction for `LensField`.
///
/// (a) A macro-only field has `Psi = 0`, so the subtracted solve returns the
/// exact free/macro closed form; pinned against the 30-digit referee
/// (`comparison/referee.py::macro_only_integral`, mpmath dps 30, emitted
/// 2026-10-18) — this pins the `free_integral` formula including the
/// shear-rotation handling.
#[test]
fn lens_field_subtract_free_free_limit_matches_referee() {
    let cases: &[(f64, f64, f64, f64)] = &[
        // gamma = (0.2, 0.0): no rotation.
        (0.2, 0.0, 7.96145669514098042e-01, 9.36641964104927465e+04),
        // gamma = (0.2, 0.1): rotated frame.
        (0.2, 0.1, 5.92016140689123005e-01, 9.47225825080982177e+04),
    ];
    for &(g1, g2, re, im) in cases {
        let lf = LensField::new([0.4, -0.3], 0.3, g1, g2, vec![]).unwrap();
        let o = lens_field_solve(
            &lf,
            &SliceConfig { omega: 1e-4, subtract_free: true, ..Default::default() },
        )
        .unwrap();
        let d = rel(o.value, Complex64::new(re, im));
        assert!(d < 1e-14, "gamma=({g1},{g2}): {d:.3e}");
    }
}

/// (b) `LensField` with `kappa = gamma = 0` and one point component is the
/// same physical model as `PointMassField`; their free/potential splits
/// differ in arithmetic, so they must agree at quadrature level — at low
/// omega, where the subtraction is what makes the comparison meaningful.
#[test]
fn lens_field_subtract_free_point_equals_point_mass_field() {
    use fast_picard_lefshetz_integral_code::phases::PointMassField;
    use fast_picard_lefshetz_integral_code::slice_field;
    let lf = LensField::new([1.0, 0.0], 0.0, 0.0, 0.0, vec![point(0.0, 0.0, 1.0)]).unwrap();
    let pm = PointMassField { beta: [1.0, 0.0], lenses: vec![(0.0, 0.0, 1.0)] };
    for omega in [1e-4_f64, 1e-2] {
        let a = lens_field_solve(
            &lf,
            &SliceConfig { omega, subtract_free: true, ..Default::default() },
        )
        .unwrap();
        let b = slice_field(
            &pm,
            &SliceConfig { omega, subtract_free: true, ..Default::default() },
        )
        .unwrap();
        let d = (a.value - b.value).norm() / b.value.norm();
        assert!(d < 1e-11, "omega={omega}: {d:.3e}");
    }
}
