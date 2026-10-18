//! The batched RC field integral (`slice_field_batch`): accuracy against the
//! scalar production solver (a *different contour* — the batch shares one
//! core spanning the whole grid — so this is a Cauchy check), against the
//! external Nakamura-Deguchi references on a frequency sweep, transposed-grid
//! invariance, error honesty, parallel determinism and input validation.
//!
//! The 17-digit ND references were emitted by `comparison.referee.slice_raw_from_nd`
//! (mpmath, dps 30) on 2026-10-12; provenance noted per block.

use fast_picard_lefshetz_integral_code::phases::{CustomRationalLens, PointMassField, RationalLens, SimGLens};
use fast_picard_lefshetz_integral_code::{
    Error, FieldBatch, InterruptHook, RationalPhase, SliceConfig, ThimbleConfig, ThimbleSolver,
    rc_1d_batch, rc_1d_sweep, slice_field, slice_field_batch,
};
use num_complex::Complex64;

fn two_mass() -> PointMassField {
    PointMassField {
        beta: [0.2, 0.1],
        lenses: vec![(-0.8, 0.5, 0.6), (0.9, -0.4, 0.4)],
    }
}

fn single_mass() -> PointMassField {
    PointMassField { beta: [0.5, 0.0], lenses: vec![(0.0, 0.0, 1.0)] }
}

fn rel(a: Complex64, b: Complex64, scale: f64) -> f64 {
    (a - b).norm() / scale
}

#[test]
fn batch_entries_match_scalar_on_uniform_and_jittered_grids() {
    let field = two_mass();
    let omegas = [1.0, 10.0];
    for (name, b0s, b1s) in [
        (
            "uniform-beta0",
            (0..9).map(|i| -1.0 + 0.25 * i as f64).collect::<Vec<f64>>(),
            (0..5).map(|i| -1.0 + 0.5 * i as f64).collect::<Vec<f64>>(),
        ),
        (
            "jittered (direct path)",
            (0..9).map(|i| -1.0 + 0.25 * i as f64 + 1e-14 * ((i * 7919) % 13) as f64).collect(),
            (0..5).map(|i| -1.0 + 0.5 * i as f64 - 1e-14 * ((i * 104729) % 11) as f64).collect(),
        ),
    ] {
        let out = slice_field_batch(
            &field,
            &FieldBatch { omegas: omegas.to_vec(), beta0s: b0s.clone(), beta1s: b1s.clone() },
            &SliceConfig { omega: 1.0, ..Default::default() },
        )
        .unwrap();
        let mut scale = 0.0f64;
        for v in &out.values {
            scale = scale.max(v.norm());
        }
        let mut worst = 0.0f64;
        for (j, &w) in omegas.iter().enumerate() {
            for (i0, &b0) in b0s.iter().enumerate() {
                for (i1, &b1) in b1s.iter().enumerate() {
                    let f = PointMassField { beta: [b0, b1], lenses: field.lenses.clone() };
                    let sc = slice_field(&f, &SliceConfig { omega: w, ..Default::default() })
                        .unwrap()
                        .value;
                    let idx = (j * b0s.len() + i0) * b1s.len() + i1;
                    worst = worst.max(rel(out.values[idx], sc, scale));
                }
            }
        }
        assert!(worst < 1e-12, "{name}: batch vs scalar worst {worst:.3e}");
    }
}

#[test]
fn sweep_matches_nakamura_deguchi() {
    // One band (ratio 10) and three bands (ratio 1000), single unit mass at
    // the origin, source (0.5, 0); raw-integer references from the ND closed
    // form (mpmath dps 30).
    let field = single_mass();
    const ONE_BAND: &[(f64, f64, f64)] = &[
        (1.00000000000000000e+00, 1.11515184895840158e+00, 1.06271092478315978e+01),
        (1.29154966501488389e+00, 3.46632574923839212e-02, 8.88265016316615963e+00),
        (1.66810053720005880e+00, -1.03865387510097884e+00, 7.13478559515883681e+00),
        (2.15443469003188381e+00, -1.89378023185492528e+00, 5.22076474596427609e+00),
        (2.78255940220712450e+00, -2.18601391791826005e+00, 3.15929782108150770e+00),
        (3.59381366380462763e+00, -1.58862639678163364e+00, 1.35205054424142057e+00),
        (4.64158883361277841e+00, -3.08252507932861686e-01, 6.17304763929644729e-01),
        (5.99484250318940859e+00, 3.20050668749341516e-01, 1.28480823094669439e+00),
        (7.74263682681126930e+00, -6.84159138846451187e-01, 1.48939610057555782e+00),
        (1.00000000000000000e+01, -6.03353332588194791e-01, 2.05793144830923674e-01),
    ];
    const THREE_BANDS: &[(f64, f64, f64)] = &[
        (1.00000000000000006e-01, 8.18034041100685805e+00, 6.72883199498779732e+01),
        (2.15443469003188337e-01, 6.05266953590797119e+00, 3.35735966614998418e+01),
        (4.64158883361277863e-01, 3.83966878630654973e+00, 1.80053915474419242e+01),
        (1.00000000000000000e+00, 1.11515184895840158e+00, 1.06271092478315978e+01),
        (2.15443469003188337e+00, -1.89378023185492483e+00, 5.22076474596427786e+00),
        (4.64158883361277752e+00, -3.08252507932862629e-01, 6.17304763929644729e-01),
        (1.00000000000000000e+01, -6.03353332588194791e-01, 2.05793144830923674e-01),
        (2.15443469003188213e+01, -4.69704518585327691e-01, -6.96680699257343217e-02),
        (4.64158883361277717e+01, 2.04470263855007124e-04, -2.19980902091682418e-01),
        (1.00000000000000000e+02, 8.95259962335362286e-02, 6.52039497653385536e-02),
    ];
    for (name, table) in [("one band", ONE_BAND), ("three bands", THREE_BANDS)] {
        let omegas: Vec<f64> = table.iter().map(|&(w, _, _)| w).collect();
        let out = slice_field_batch(
            &field,
            &FieldBatch { omegas, beta0s: vec![0.5], beta1s: vec![0.0] },
            &SliceConfig { omega: 1.0, ..Default::default() },
        )
        .unwrap();
        assert_eq!(out.values.len(), table.len());
        let mut worst = 0.0f64;
        for (i, &(_, re, im)) in table.iter().enumerate() {
            let refr = Complex64::new(re, im);
            let d = (out.values[i] - refr).norm() / refr.norm();
            worst = worst.max(d);
            // error honesty (with a couple of ulps of slack at the roundoff floor)
            let allow = 8.0 * f64::EPSILON * out.values[i].norm();
            assert!(
                out.errors[i] + allow >= (out.values[i] - refr).norm(),
                "{name} [{}]: budget {:.3e} does not cover true error {:.3e}",
                i,
                out.errors[i],
                (out.values[i] - refr).norm()
            );
            assert!(out.errors[i] < 1e-9 * refr.norm(), "{name} [{}]: loose budget", i);
        }
        assert!(worst < 1e-11, "{name}: worst ND rel {worst:.3e}");
    }
}

#[test]
fn transposed_grid_matches() {
    let field = two_mass();
    let swapped = PointMassField {
        beta: [field.beta[1], field.beta[0]],
        lenses: field.lenses.iter().map(|&(a, b, m)| (b, a, m)).collect(),
    };
    let b0s: Vec<f64> = (0..9).map(|i| -1.0 + 0.25 * i as f64).collect();
    let b1s: Vec<f64> = (0..5).map(|i| -1.0 + 0.5 * i as f64).collect();
    let cfg = SliceConfig { omega: 1.0, ..Default::default() };
    let a = slice_field_batch(
        &field,
        &FieldBatch { omegas: vec![3.0], beta0s: b0s.clone(), beta1s: b1s.clone() },
        &cfg,
    )
    .unwrap();
    let b = slice_field_batch(
        &swapped,
        &FieldBatch { omegas: vec![3.0], beta0s: b1s.clone(), beta1s: b0s.clone() },
        &cfg,
    )
    .unwrap();
    let mut scale = 0.0f64;
    for v in &a.values {
        scale = scale.max(v.norm());
    }
    let mut worst = 0.0f64;
    for i0 in 0..b0s.len() {
        for i1 in 0..b1s.len() {
            let ia = i0 * b1s.len() + i1;
            let ib = i1 * b0s.len() + i0;
            worst = worst.max((a.values[ia] - b.values[ib]).norm() / scale);
        }
    }
    assert!(worst < 1e-12, "transposed grid worst {worst:.3e}");
}

#[test]
fn parallel_equals_serial_bitwise() {
    let field = two_mass();
    let batch = FieldBatch {
        omegas: vec![1.0, 3.0, 10.0],
        beta0s: (0..12).map(|i| -1.0 + i as f64 / 5.5).collect(),
        beta1s: (0..12).map(|i| -1.0 + i as f64 / 5.5).collect(),
    };
    let cfg = SliceConfig { omega: 1.0, parallel: true, ..Default::default() };
    let par = slice_field_batch(&field, &batch, &cfg).unwrap();
    let ser = slice_field_batch(&field, &batch, &SliceConfig { parallel: false, ..cfg }).unwrap();
    assert_eq!(par.values, ser.values);
    assert_eq!(par.errors, ser.errors);
    assert_eq!(par.inner_solves, ser.inner_solves);
    assert_eq!(par.groups, ser.groups);
}

#[test]
fn batch_rejects_bad_input() {
    let field = two_mass();
    let good = FieldBatch { omegas: vec![1.0], beta0s: vec![0.0], beta1s: vec![0.0] };
    let cfg = SliceConfig::default();
    //omega <= 0
    let bad = FieldBatch { omegas: vec![0.0], ..good.clone() };
    assert!(slice_field_batch(&field, &bad, &cfg).is_err());
    // empty grids (regression: previously panicked in step_by(0) / div-by-0)
    let bad = FieldBatch { omegas: vec![], ..good.clone() };
    assert!(slice_field_batch(&field, &bad, &cfg).is_err());
    let bad = FieldBatch { beta0s: vec![], ..good.clone() };
    assert!(slice_field_batch(&field, &bad, &cfg).is_err());
    let bad = FieldBatch { beta1s: vec![], ..good.clone() };
    assert!(slice_field_batch(&field, &bad, &cfg).is_err());
    // non-finite source
    let bad = FieldBatch { beta0s: vec![f64::NAN], ..good.clone() };
    assert!(slice_field_batch(&field, &bad, &cfg).is_err());
    // band ratio
    let bad_cfg = SliceConfig { band_ratio: 1.0, ..cfg.clone() };
    assert!(slice_field_batch(&field, &good, &bad_cfg).is_err());
    // tail angle out of (0, pi/4]
    let bad_cfg = SliceConfig { tail_angle: 0.9, ..cfg };
    assert!(slice_field_batch(&field, &good, &bad_cfg).is_err());
}

#[test]
fn sweep_uniform_omega_uses_series_and_matches_scalar() {
    // A uniform omega grid (n >= 8) triggers the re-anchored frequency series;
    // every entry must still match the scalar solver (two-mass field).
    let field = two_mass();
    let omegas: Vec<f64> = (0..9).map(|i| 1.0 + 0.5 * i as f64).collect();
    let cfg = SliceConfig { omega: 1.0, ..Default::default() };
    let out = slice_field_batch(
        &field,
        &FieldBatch { omegas: omegas.clone(), beta0s: vec![0.2], beta1s: vec![0.1] },
        &cfg,
    )
    .unwrap();
    let mut scale = 0.0f64;
    for v in &out.values {
        scale = scale.max(v.norm());
    }
    for (i, &w) in omegas.iter().enumerate() {
        let sc = slice_field(&field, &SliceConfig { omega: w, ..Default::default() })
            .unwrap()
            .value;
        let d = (out.values[i] - sc).norm() / scale;
        assert!(d < 1e-12, "uniform-omega sweep [{i}] w={w}: {d:.3e}");
    }
}

#[test]
fn sweep_low_omega_matches_nd() {
    // The low-omega regime (P0's E3): the batch's direct path stays accurate
    // there (a scalar free-space subtraction is optional and not used here).
    let field = single_mass();
    let table: &[(f64, f64, f64)] = &[
        (1.00000000000000002e-02, 1.49469695267153693e+01, 6.33079359910506014e+02),
        (1.00000000000000002e-03, 2.20828489099470424e+01, 6.28808158603529318e+03),
    ];
    let omegas: Vec<f64> = table.iter().map(|&(w, _, _)| w).collect();
    let out = slice_field_batch(
        &field,
        &FieldBatch { omegas, beta0s: vec![0.5], beta1s: vec![0.0] },
        &SliceConfig { omega: 1.0, ..Default::default() },
    )
    .unwrap();
    for (i, &(_, re, im)) in table.iter().enumerate() {
        let refr = Complex64::new(re, im);
        let d = (out.values[i] - refr).norm() / refr.norm();
        // graded on F - 1 below omega = 1 (the physical quantity there)
        assert!(d < 1e-11, "low-omega [{i}]: raw rel {d:.3e}");
    }
}

// ---------------------------------------------------------------------------
// 1D rational batch (`rc_1d_batch`)
// ---------------------------------------------------------------------------

#[test]
fn rc1d_matches_thimble_on_the_plpi_grid() {
    // The P-LPI contest grid: 512 sources, omega = 10, RationalLens(kappa=-2).
    // The thimble is the machine-precision production reference; the batch
    // shares one contour over the whole grid (a grid that is *uniform*, so
    // this also exercises the re-anchored twiddle recurrence).
    let omega = 10.0;
    let n = 512;
    let betas: Vec<f64> = (0..n).map(|i| -4.0 + 8.0 * i as f64 / (n as f64 - 1.0)).collect();
    let cfg = SliceConfig { omega, ..Default::default() };
    let (vals, errs) = rc_1d_batch(&RationalLens { beta: 0.0, kappa: -2.0 }, &betas, omega, &cfg)
        .unwrap();
    let mut scale = 0.0f64;
    let mut thim = Vec::with_capacity(n);
    for &b in &betas {
        let t = ThimbleSolver::new_rational(
            &RationalLens { beta: b, kappa: -2.0 },
            ThimbleConfig { omega, ..Default::default() },
        )
        .run()
        .unwrap()
        .value;
        scale = scale.max(t.norm());
        thim.push(t);
    }
    let mut worst = 0.0f64;
    for i in 0..n {
        let d = (vals[i] - thim[i]).norm();
        worst = worst.max(d / scale);
        let allow = 8.0 * f64::EPSILON * vals[i].norm();
        assert!(
            errs[i] + allow >= d,
            "rc1d [{i}]: budget {:.3e} does not cover true error {:.3e}",
            errs[i],
            d
        );
        assert!(errs[i] < 1e-9 * scale, "rc1d [{i}]: loose budget {:.3e}", errs[i]);
    }
    assert!(worst < 1e-12, "rc1d vs thimble worst {worst:.3e}");
}

#[test]
fn rc1d_custom_lens_matches_rational_lens() {
    // CustomRationalLens(kappa=1, A=[-2], B=[1,0,1]) == RationalLens(kappa=-2).
    let beta = 0.3;
    let omega = 3.0;
    let betas: Vec<f64> = (0..16).map(|i| -1.0 + 2.0 * i as f64 / 15.0).collect();
    let cfg = SliceConfig { omega, ..Default::default() };
    let (a, _) = rc_1d_batch(&RationalLens { beta, kappa: -2.0 }, &betas, omega, &cfg).unwrap();
    let custom = CustomRationalLens {
        beta,
        kappa: 1.0,
        num: vec![Complex64::new(-2.0, 0.0)],
        den: vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    };
    let (b, _) = rc_1d_batch(&custom, &betas, omega, &cfg).unwrap();
    let mut scale = 0.0f64;
    for v in &a {
        scale = scale.max(v.norm());
    }
    for i in 0..betas.len() {
        let d = (a[i] - b[i]).norm() / scale;
        assert!(d < 1e-12, "custom vs rational [{i}]: {d:.3e}");
    }
}

#[test]
fn rc1d_simg_matches_thimble() {
    let omega = 1.0;
    let betas: Vec<f64> = (0..8).map(|i| -0.7 + 1.4 * i as f64 / 7.0).collect();
    let cfg = SliceConfig { omega, ..Default::default() };
    let lens0 = SimGLens { beta: 0.0, kappa: 1.0, kmax: 2 };
    let (vals, _) = rc_1d_batch(&lens0, &betas, omega, &cfg).unwrap();
    let mut worst = 0.0f64;
    let mut scale = 0.0f64;
    let mut thim = Vec::new();
    for &b in &betas {
        let t = ThimbleSolver::new_rational(
            &SimGLens { beta: b, kappa: 1.0, kmax: 2 },
            ThimbleConfig { omega, ..Default::default() },
        )
        .run()
        .unwrap()
        .value;
        scale = scale.max(t.norm());
        thim.push(t);
    }
    for i in 0..betas.len() {
        worst = worst.max((vals[i] - thim[i]).norm() / scale);
    }
    assert!(worst < 1e-11, "simg rc1d vs thimble worst {worst:.3e}");
}

#[test]
fn rc1d_rejects_bad_lenses() {
    let cfg = SliceConfig { omega: 1.0, ..Default::default() };
    // real-axis pole: den = [-1, 1] -> z = 1
    let bad = CustomRationalLens {
        beta: 0.0,
        kappa: 1.0,
        num: vec![Complex64::new(1.0, 0.0)],
        den: vec![Complex64::new(-1.0, 0.0), Complex64::new(1.0, 0.0)],
    };
    assert!(rc_1d_batch(&bad, &[0.0], 1.0, &cfg).is_err());
    // unbounded at infinity: num = z^2, den = 1
    let bad = CustomRationalLens {
        beta: 0.0,
        kappa: 1.0,
        num: vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        den: vec![Complex64::new(1.0, 0.0)],
    };
    assert!(rc_1d_batch(&bad, &[0.0], 1.0, &cfg).is_err());
    // omega <= 0
    let good = RationalLens { beta: 0.0, kappa: -2.0 };
    assert!(rc_1d_batch(&good, &[0.0], 0.0, &cfg).is_err());
}

// ---------------------------------------------------------------------------
// 1D frequency sweep (`rc_1d_sweep`)
// ---------------------------------------------------------------------------

/// The per-frequency thimble for one lens and grid — the production
/// machine-precision reference `(value, error)`.
fn thimble_sweep<M: RationalPhase>(lens: &M, omegas: &[f64]) -> Vec<(Complex64, f64)> {
    omegas
        .iter()
        .map(|&w| {
            let out =
                ThimbleSolver::new_rational(lens, ThimbleConfig { omega: w, ..Default::default() })
                    .run()
                    .unwrap();
            (out.value, out.error)
        })
        .collect()
}

/// One sweep against the thimble, entry by entry: agreement, RC-budget
/// tightness, and the sum-of-budgets cover check.  Each method's reported
/// error bounds its own deviation — the RC's is tight (see
/// [`rc1d_sweep_budget_covers_contour_family`]) while the thimble's is
/// deliberately loose at low omega (1e-9..1e-10 at omega <= 1), so their sum
/// bounds the difference; where both are tight the check bites.
fn assert_sweep_matches_thimble(
    name: &str,
    vals: &[Complex64],
    errs: &[f64],
    thim: &[(Complex64, f64)],
) {
    let mut scale = 0.0f64;
    for (t, _) in thim {
        scale = scale.max(t.norm());
    }
    for i in 0..vals.len() {
        let d = (vals[i] - thim[i].0).norm();
        let allow = 8.0 * f64::EPSILON * vals[i].norm();
        assert!(
            errs[i] + thim[i].1 + allow >= d,
            "{name} [{i}]: budgets {:.3e} (rc) + {:.3e} (thimble) do not cover \
             the difference {:.3e}",
            errs[i],
            thim[i].1,
            d
        );
        assert!(errs[i] < 1e-9 * scale, "{name} [{i}]: loose budget {:.3e}", errs[i]);
    }
    let mut worst = 0.0f64;
    for i in 0..vals.len() {
        worst = worst.max((vals[i] - thim[i].0).norm() / scale);
    }
    assert!(worst < 1e-11, "{name}: sweep vs thimble worst {worst:.3e}");
}

#[test]
fn rc1d_sweep_matches_thimble_on_uniform_and_jittered_omegas() {
    // A single lens across a *uniform* frequency grid (n >= 8: the re-anchored
    // twiddle recurrence, one band) and a jittered log-spaced grid up to
    // omega ~ 270 (several bands, exact sincos); every entry against the
    // per-frequency thimble.
    let lens = RationalLens { beta: -0.72, kappa: -2.0 };
    let cfg = SliceConfig { omega: 1.0, ..Default::default() };
    let uniform: Vec<f64> = (0..12).map(|i| 1.0 + 0.5 * i as f64).collect();
    let jittered: Vec<f64> = (0..12)
        .map(|i| 0.7 * 1.7f64.powi(i) * (1.0 + 0.13 * (2.3 * i as f64).sin()))
        .collect();
    for (name, omegas) in [("uniform", uniform), ("jittered log", jittered)] {
        let (vals, errs) = rc_1d_sweep(&lens, &omegas, &cfg).unwrap();
        assert_eq!(vals.len(), omegas.len());
        let thim = thimble_sweep(&lens, &omegas);
        assert_sweep_matches_thimble(name, &vals, &errs, &thim);
    }
}

#[test]
fn rc1d_sweep_matches_batch_across_bands() {
    // The sweep's banded shared contour must reproduce the single-frequency
    // batch (the already-validated construction) on both a uniform grid
    // (recurrence path, one band) and a jittered multi-band grid, for two
    // lens families.
    let cfg = SliceConfig { omega: 1.0, ..Default::default() };
    let uniform: Vec<f64> = (0..12).map(|i| 1.0 + 0.5 * i as f64).collect();
    let jittered: Vec<f64> = (0..12)
        .map(|i| 0.7 * 1.7f64.powi(i) * (1.0 + 0.13 * (2.3 * i as f64).sin()))
        .collect();
    let rl = RationalLens { beta: -0.72, kappa: -2.0 };
    let sg = SimGLens { beta: 0.4, kappa: 1.0, kmax: 2 };
    for omegas in [&uniform, &jittered] {
        let (sv, _) = rc_1d_sweep(&rl, omegas, &cfg).unwrap();
        let (tv, _) = rc_1d_sweep(&sg, omegas, &cfg).unwrap();
        let mut scale = 0.0f64;
        for v in sv.iter().chain(tv.iter()) {
            scale = scale.max(v.norm());
        }
        for (i, &w) in omegas.iter().enumerate() {
            let (bv, _) = rc_1d_batch(&rl, &[rl.beta], w, &cfg).unwrap();
            let (cv, _) = rc_1d_batch(&sg, &[sg.beta], w, &cfg).unwrap();
            let dr = (sv[i] - bv[0]).norm() / scale;
            let ds = (tv[i] - cv[0]).norm() / scale;
            assert!(dr < 1e-12, "rational sweep vs batch [{i}] w={w}: {dr:.3e}");
            assert!(ds < 1e-12, "simg sweep vs batch [{i}] w={w}: {ds:.3e}");
        }
    }
}

#[test]
fn rc1d_sweep_simg_and_custom_match_thimble() {
    let cfg = SliceConfig { omega: 1.0, ..Default::default() };
    let omegas: Vec<f64> = (0..8).map(|i| 0.5 * 2.0f64.powi(i)).collect();
    // SimGLens vs the thimble
    let simg = SimGLens { beta: 0.4, kappa: 1.0, kmax: 2 };
    let (vals, errs) = rc_1d_sweep(&simg, &omegas, &cfg).unwrap();
    assert_sweep_matches_thimble("simg", &vals, &errs, &thimble_sweep(&simg, &omegas));
    // CustomRationalLens(kappa=1, A=[-2], B=[1,0,1]) == RationalLens(kappa=-2)
    let custom = CustomRationalLens {
        beta: 0.4,
        kappa: 1.0,
        num: vec![Complex64::new(-2.0, 0.0)],
        den: vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    };
    let (cvals, _) = rc_1d_sweep(&custom, &omegas, &cfg).unwrap();
    let rational = RationalLens { beta: 0.4, kappa: -2.0 };
    let (rvals, _) = rc_1d_sweep(&rational, &omegas, &cfg).unwrap();
    let mut scale = 0.0f64;
    for v in &rvals {
        scale = scale.max(v.norm());
    }
    for i in 0..omegas.len() {
        let d = (cvals[i] - rvals[i]).norm() / scale;
        assert!(d < 1e-12, "custom vs rational [{i}]: {d:.3e}");
    }
}

#[test]
fn rc1d_sweep_budget_covers_contour_family() {
    // The RC tail angle is a free contour parameter: the sweep's value must be
    // invariant across the family, and the reported budget must cover the
    // residual spread (the house "estimates cover the true error" rule, with
    // the family as the arbiter).  Measured spread <= 5e-16 over omega
    // 0.5..40 for both lenses; the budgets carry >= 3x margin everywhere.
    let cfg_a = SliceConfig { omega: 1.0, ..Default::default() };
    let cfg_b = SliceConfig {
        omega: 1.0,
        tail_angle: std::f64::consts::FRAC_PI_4 / 1.5,
        ..Default::default()
    };
    macro_rules! family {
        ($name:literal, $lens:expr) => {{
            let lens = $lens;
            for w in [0.5f64, 1.0, 4.0, 40.0] {
                let (va, ea) = rc_1d_sweep(&lens, &[w], &cfg_a).unwrap();
                let (vb, eb) = rc_1d_sweep(&lens, &[w], &cfg_b).unwrap();
                let d = (va[0] - vb[0]).norm();
                let allow = 8.0 * f64::EPSILON * (va[0].norm() + vb[0].norm());
                assert!(
                    d <= ea[0] + eb[0] + allow,
                    "{} w={w}: family spread {d:.3e} exceeds the budgets {:.3e}",
                    $name,
                    ea[0] + eb[0]
                );
                let scale = va[0].norm().max(vb[0].norm()).max(1e-30);
                assert!(
                    d / scale < 1e-13,
                    "{} w={w}: family spread {:.3e} relative",
                    $name,
                    d / scale
                );
            }
        }};
    }
    family!("RationalLens", RationalLens { beta: -0.72, kappa: -2.0 });
    family!("SimGLens", SimGLens { beta: 0.4, kappa: 1.0, kmax: 2 });
}

#[test]
fn rc1d_sweep_rejects_bad_input() {
    let good = RationalLens { beta: 0.0, kappa: -2.0 };
    let cfg = SliceConfig { omega: 1.0, ..Default::default() };
    // empty grid, omega <= 0, non-finite omega
    assert!(rc_1d_sweep(&good, &[], &cfg).is_err());
    assert!(rc_1d_sweep(&good, &[1.0, 0.0], &cfg).is_err());
    assert!(rc_1d_sweep(&good, &[f64::NAN], &cfg).is_err());
    // band_ratio <= 1 (the bands are load-bearing here)
    let bad_cfg = SliceConfig { omega: 1.0, band_ratio: 1.0, ..Default::default() };
    assert!(rc_1d_sweep(&good, &[1.0, 2.0], &bad_cfg).is_err());
    // real-axis pole and unbounded potential share the rc_1d_batch checks
    let bad = CustomRationalLens {
        beta: 0.0,
        kappa: 1.0,
        num: vec![Complex64::new(1.0, 0.0)],
        den: vec![Complex64::new(-1.0, 0.0), Complex64::new(1.0, 0.0)],
    };
    assert!(rc_1d_sweep(&bad, &[1.0], &cfg).is_err());
    let bad = CustomRationalLens {
        beta: 0.0,
        kappa: 1.0,
        num: vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        den: vec![Complex64::new(1.0, 0.0)],
    };
    assert!(rc_1d_sweep(&bad, &[1.0], &cfg).is_err());
}

#[test]
fn rc1d_sweep_parallel_equals_serial_bitwise() {
    let lens = RationalLens { beta: -0.72, kappa: -2.0 };
    let omegas: Vec<f64> = (0..12).map(|i| 1.0 + 0.5 * i as f64).collect();
    let ser = SliceConfig { omega: 1.0, parallel: false, ..Default::default() };
    let par = SliceConfig { parallel: true, ..ser.clone() };
    let (vs, es) = rc_1d_sweep(&lens, &omegas, &ser).unwrap();
    let (vp, ep) = rc_1d_sweep(&lens, &omegas, &par).unwrap();
    for i in 0..omegas.len() {
        assert_eq!(vs[i].re.to_bits(), vp[i].re.to_bits(), "value {i} differs");
        assert_eq!(vs[i].im.to_bits(), vp[i].im.to_bits(), "value {i} differs");
        assert_eq!(es[i].to_bits(), ep[i].to_bits(), "error {i} differs");
    }
}

/// The cooperative interrupt hook on the 1D sweep: a firing probe aborts with
/// [`Error::Interrupted`] at the first band boundary.
#[test]
fn interrupt_hook_aborts_1d_sweep() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let lens = RationalLens { beta: 0.5, kappa: -2.0 };
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let firing = InterruptHook::new(move || {
        counter.fetch_add(1, Ordering::Relaxed);
        true
    });
    let cfg = SliceConfig { omega: 1.0, interrupt: Some(firing), ..Default::default() };
    let omegas: Vec<f64> = (0..8).map(|i| 1.0 + i as f64).collect();
    let err = rc_1d_sweep(&lens, &omegas, &cfg).unwrap_err();
    assert!(matches!(err, Error::Interrupted), "{err:?}");
    assert!(calls.load(Ordering::Relaxed) >= 1);
}

/// The cooperative interrupt hook on the batch path: a firing probe aborts
/// with [`Error::Interrupted`] at the first band boundary.
#[test]
fn interrupt_hook_aborts_batch() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let field = two_mass();
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let firing = InterruptHook::new(move || {
        counter.fetch_add(1, Ordering::Relaxed);
        true
    });
    let cfg = SliceConfig { omega: 10.0, interrupt: Some(firing), ..Default::default() };
    let batch = FieldBatch { omegas: vec![1.0, 10.0], beta0s: vec![0.2], beta1s: vec![0.1] };
    let err = slice_field_batch(&field, &batch, &cfg).unwrap_err();
    assert!(matches!(err, Error::Interrupted), "{err:?}");
    assert!(calls.load(Ordering::Relaxed) >= 1);
}
