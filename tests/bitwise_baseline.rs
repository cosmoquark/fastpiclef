//! Bitwise regression baseline for the RC slice kernels.
//!
//! History: captured *before* the `SliceModel` refactor (commit `c70b128`) and
//! **regenerated 2026-10-12 after the P2a tail-geometry change** — the tail
//! length now scales with the model's `|lambda|` and includes the
//! `|beta|/(|lambda| cos alpha)` term (both required by the macro model);
//! point-mass results shifted by 1-2 ulps and the tolerance-based suites were
//! unchanged.  This is deliberately not a numerical tolerance test — any
//! *intentional* arithmetic change must regenerate the tables via the ignored
//! `print_current_bits` helper below and say so in the commit message /
//! journal.
//!
//! Values are IEEE-754 bit patterns produced by the public API with default
//! configs.

use fast_picard_lefshetz_integral_code::phases::PointMassField;
use fast_picard_lefshetz_integral_code::{SliceConfig, slice_field, slice_inner_rc};
use num_complex::Complex64;

fn chk(name: &str, got: Complex64, re: u64, im: u64) {
    assert_eq!(
        got.re.to_bits(),
        re,
        "{name}: real part bits changed (refactor altered the arithmetic; if intentional, \
         regenerate with the ignored `print_current_bits` and record it)"
    );
    assert_eq!(
        got.im.to_bits(),
        im,
        "{name}: imaginary part bits changed (refactor altered the arithmetic; if intentional, \
         regenerate with the ignored `print_current_bits` and record it)"
    );
}

fn two_mass() -> PointMassField {
    PointMassField {
        beta: [0.2, 0.1],
        lenses: vec![(-0.8, 0.5, 0.6), (0.9, -0.4, 0.4)],
    }
}

// (a) slice_inner_rc on the two-mass field: (omega, x2, re_bits, im_bits)
const INNER_CASES: &[(f64, f64, u64, u64)] = &[
    (1.0, -3.0, 0xbfe8a6b3a4cff4be, 0xc00408d3d632f22f),
    (1.0, -1.1, 0x3ff58f23fa38b0d7, 0x400503882e2d6ac9),
    (1.0, -0.41, 0x3fff102433a7ec29, 0x400504260b9008e4),
    (1.0, 0.0, 0x4001696d7de3473e, 0x4003528132f65066),
    (1.0, 0.49, 0x3ff867b69c1eab89, 0x4003a621e6dea9f6),
    (1.0, 0.500001, 0x3ff8540a409f5060, 0x400387301df607a7),
    (3.0, -3.0, 0x3feecbf2c9afd364, 0xbff2b2c226fc8884),
    (3.0, -1.1, 0xbfe034f381c9fb22, 0x3ff9dc56dc66bf11),
    (3.0, -0.41, 0xbfa4e6b4128f4e54, 0x3ff846fe379cb2ec),
    (3.0, 0.0, 0x3fe291b4c330da5a, 0x3ff81b07037ff492),
    (3.0, 0.49, 0x3ff035915af8de45, 0x3fc89630d03b1bc6),
    (3.0, 0.500001, 0x3ff057b302d06527, 0x3fc65ead0a8e4b21),
    (10.0, -3.0, 0x3fe637f43be6a062, 0xbfdd03a3ba7ba6af),
    (10.0, -1.1, 0xbfd8ae881fe06cb6, 0xbfe5a25783fe2022),
    (10.0, -0.41, 0x3fdac656ec6c9b0d, 0x3fead3c557efec9e),
    (10.0, 0.0, 0xbfb1083c75bbcdec, 0x3fe02a65d8a68de8),
    (10.0, 0.49, 0x3fed984d6d02d69a, 0x3fe63f083f711b37),
    (10.0, 0.500001, 0x3fed33f311f2fc3f, 0x3fe6d99967e67a63),
    (30.0, -3.0, 0xbfde3baa935640f9, 0x3fb3ea65526cdf9d),
    (30.0, -1.1, 0xbfae671b054fbc32, 0xbfdda48cfe95c4af),
    (30.0, -0.41, 0xbfbd3cf599f25936, 0x3fa1e87567d604a3),
    (30.0, 0.0, 0xbfb33899aadf39dd, 0x3fd3a41f401e4f25),
    (30.0, 0.49, 0x3fe11a3f266bb7e3, 0x3fca4ca2e22d2841),
    (30.0, 0.500001, 0x3fe092d654627b68, 0x3fcc289738a9bd7a),
];

// (b) slice_field, single unit mass at the origin, source (y, 0)
const FIELD_SINGLE_CASES: &[(f64, f64, u64, u64)] = &[
    (0.3, 1.0, 0x40143cea6ffab306, 0x4038efb8a1824fb6),
    (1.0, 0.5, 0x3ff1d7a977159ab0, 0x40254114769ce96c),
    (3.0, 0.1, 0xc01185e4630d8641, 0x4012087cc5fc1dca),
    (10.0, 2.0, 0x3fe7ae334eaecc4c, 0xbfa2691c0b062944),
    (30.0, 1.0, 0x3fb10834d8e84b76, 0xbfc90df1dda8aa6d),
];

// (c) slice_field, two-mass field: (omega, re_bits, im_bits)
const FIELD_TWO_MASS_CASES: &[(f64, u64, u64)] = &[
    (3.0, 0xbfdc04df1ec624e9, 0x4007be99dfaf3ede),
    (10.0, 0x3fe3db97a32ed75a, 0x3ff092a77d8f19ed),
];

#[test]
fn inner_rc_baseline_is_bitwise() {
    let field = two_mass();
    for &(omega, x2, re, im) in INNER_CASES {
        let cfg = SliceConfig { omega, ..Default::default() };
        let g = slice_inner_rc(&field, Complex64::new(x2, 0.0), &cfg).unwrap();
        chk(&format!("inner w={omega} x2={x2}"), g.value, re, im);
    }
}

#[test]
fn field_baseline_is_bitwise() {
    for &(omega, y, re, im) in FIELD_SINGLE_CASES {
        let field = PointMassField { beta: [y, 0.0], lenses: vec![(0.0, 0.0, 1.0)] };
        let cfg = SliceConfig { omega, ..Default::default() };
        let o = slice_field(&field, &cfg).unwrap();
        chk(&format!("field single w={omega} y={y}"), o.value, re, im);
    }
    let field = two_mass();
    for &(omega, re, im) in FIELD_TWO_MASS_CASES {
        let cfg = SliceConfig { omega, ..Default::default() };
        let o = slice_field(&field, &cfg).unwrap();
        chk(&format!("field two-mass w={omega}"), o.value, re, im);
    }
}

/// Regeneration helper (prints the current tables; compare against the consts).
#[test]
#[ignore = "regeneration helper: run with --ignored --nocapture"]
fn print_current_bits() {
    let field = two_mass();
    println!("// (a) slice_inner_rc: (omega, x2, re_bits, im_bits)");
    for &(omega, x2, _, _) in INNER_CASES {
        let cfg = SliceConfig { omega, ..Default::default() };
        let g = slice_inner_rc(&field, Complex64::new(x2, 0.0), &cfg).unwrap();
        println!("    ({omega:?}, {x2:?}, 0x{:016x}, 0x{:016x}),", g.value.re.to_bits(), g.value.im.to_bits());
    }
    println!("// (b) slice_field single mass");
    for &(omega, y, _, _) in FIELD_SINGLE_CASES {
        let f = PointMassField { beta: [y, 0.0], lenses: vec![(0.0, 0.0, 1.0)] };
        let cfg = SliceConfig { omega, ..Default::default() };
        let o = slice_field(&f, &cfg).unwrap();
        println!("    ({omega:?}, {y:?}, 0x{:016x}, 0x{:016x}),", o.value.re.to_bits(), o.value.im.to_bits());
    }
    println!("// (c) slice_field two-mass field");
    for &(omega, _, _) in FIELD_TWO_MASS_CASES {
        let cfg = SliceConfig { omega, ..Default::default() };
        let o = slice_field(&field, &cfg).unwrap();
        println!("    ({omega:?}, 0x{:016x}, 0x{:016x}),", o.value.re.to_bits(), o.value.im.to_bits());
    }
}
