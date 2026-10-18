//! Item-6 measurement: the GL10-probe acceptance strategy against the classic
//! GL20-halves rule, in load-independent node counts.
//!
//! Every matrix row runs both strategies (`SliceConfig::gl10_probe`) and
//! prints the counts `classic -> probe` with the ratio; the table is written
//! to `comparison/results/acceptance_probe.csv`.  `rel_diff` is the relative
//! difference of the two strategies' values (both are GL20-quality on
//! accepted panels, so it must sit at the quadrature level — a large value
//! would mean the probe is not value-safe).
//!
//!     cargo run --release --example acceptance_probe

use fast_picard_lefshetz_integral_code::phases::{PointMassField, RationalLens};
use fast_picard_lefshetz_integral_code::{
    Component, FieldBatch, LensField, Profile, SliceConfig, lens_field_solve, rc_1d_batch_evals,
    slice_field, slice_field_batch, slice_inner_rc_evals,
};
use num_complex::Complex64;
use std::io::Write;

fn cfg(omega: f64, probe: bool) -> SliceConfig {
    SliceConfig { omega, gl10_probe: probe, ..Default::default() }
}

struct Row {
    case: String,
    omega: f64,
    classic: usize,
    probe: usize,
    rel: f64,
}

/// Run one case under both strategies and record the row.
fn run(
    rows: &mut Vec<Row>,
    case: &str,
    omega: f64,
    f: &dyn Fn(&SliceConfig) -> (Complex64, usize),
) {
    let (vc, nc) = f(&cfg(omega, false));
    let (vp, np) = f(&cfg(omega, true));
    let rel = (vc - vp).norm() / vc.norm().max(1e-300);
    println!(
        "{case:44} w={omega:7.2}  {nc:8} -> {np:8}   ratio {:5.2}   rel {rel:.2e}",
        nc as f64 / np as f64
    );
    rows.push(Row { case: case.to_string(), omega, classic: nc, probe: np, rel });
}

fn two_mass() -> PointMassField {
    PointMassField { beta: [0.2, 0.1], lenses: vec![(-0.8, 0.5, 0.6), (0.9, -0.4, 0.4)] }
}

fn point(x: f64, y: f64, m: f64) -> Component {
    Component { x, y, profile: Profile::Point { mass: m, xc: 0.0 } }
}

fn main() {
    let mut rows: Vec<Row> = Vec::new();

    // 1. two-mass field (scalar 2D RC engine; count = inner solves).
    let field = two_mass();
    for omega in [1.0, 3.0, 10.0, 30.0] {
        run(&mut rows, "slice_field two-mass", omega, &|c| {
            let o = slice_field(&field, c).unwrap();
            (o.value, o.inner_solves)
        });
    }

    // 2. single point mass (the P-LPI-style isolated lens).
    for omega in [1.0, 10.0, 30.0, 100.0, 300.0] {
        let pm = PointMassField { beta: [0.5, 0.0], lenses: vec![(0.0, 0.0, 1.0)] };
        run(&mut rows, "slice_field single mass", omega, &|c| {
            let o = slice_field(&pm, c).unwrap();
            (o.value, o.inner_solves)
        });
    }

    // 3. macro types I-III + one profile each (general LensField engine).
    let stock = [point(-0.6, 0.0, 0.4), point(0.5, 0.4, 0.3), point(0.1, -0.6, 0.35)];
    for (name, kappa, g1) in [("I", 0.3, 0.2), ("II", 0.5, 0.7), ("III", 1.5, 0.2)] {
        let lf = LensField::new([0.2, 0.1], kappa, g1, 0.0, stock.to_vec()).unwrap();
        run(&mut rows, &format!("lens_field macro type {name}"), 3.0, &|c| {
            let o = lens_field_solve(&lf, c).unwrap();
            (o.value, o.inner_solves)
        });
    }
    for (name, profile) in [
        ("sis", Profile::Sis { psi0: 1.0 }),
        ("cis", Profile::Cis { psi0: 1.0, xc: 0.2 }),
        ("nfw", Profile::Nfw { psi0: 1.0, xs: 0.2 }),
    ] {
        let lf =
            LensField::new([0.2, 0.1], 0.3, 0.2, 0.0, vec![Component { x: 0.0, y: 0.0, profile }])
                .unwrap();
        run(&mut rows, &format!("lens_field {name}"), 3.0, &|c| {
            let o = lens_field_solve(&lf, c).unwrap();
            (o.value, o.inner_solves)
        });
    }

    // 4. the batched vector path (shared contour over a frequency sweep and a
    //    source map; count = inner solves).
    let sweep_batch = FieldBatch {
        omegas: vec![1.0, 3.0, 10.0, 30.0, 100.0],
        beta0s: vec![0.2],
        beta1s: vec![0.1],
    };
    run(&mut rows, "slice_field_batch sweep (5 omegas)", 10.0, &|c| {
        let o = slice_field_batch(&field, &sweep_batch, c).unwrap();
        (o.values[0], o.inner_solves)
    });
    let map_batch = FieldBatch {
        omegas: vec![3.0],
        beta0s: (0..3).map(|i| -0.4 + 0.4 * i as f64).collect(),
        beta1s: (0..3).map(|i| -0.4 + 0.4 * i as f64).collect(),
    };
    run(&mut rows, "slice_field_batch map (3x3)", 3.0, &|c| {
        let o = slice_field_batch(&field, &map_batch, c).unwrap();
        (o.values[0], o.inner_solves)
    });

    // 5. 1D rational batch on the P-LPI contest grid (512 sources).
    let lens = RationalLens { beta: 0.0, kappa: -2.0 };
    let betas: Vec<f64> = (0..512).map(|i| -4.0 + 8.0 * i as f64 / 511.0).collect();
    run(&mut rows, "rc_1d_batch P-LPI grid (512)", 10.0, &|c| {
        let (v, _, n) = rc_1d_batch_evals(&lens, &betas, 10.0, c).unwrap();
        (v[0], n)
    });

    // 6. the 1D inner kernel at a real and a complex slice position.
    run(&mut rows, "slice_inner_rc x2=0.2 (real)", 10.0, &|c| {
        let (o, n) = slice_inner_rc_evals(&field, Complex64::new(0.2, 0.0), c).unwrap();
        (o.value, n)
    });
    let e4 = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
    let x2c = Complex64::new(1.5, 0.0) + e4 * 0.5;
    run(&mut rows, "slice_inner_rc x2=1.5+e4*0.5 (complex)", 10.0, &|c| {
        let (o, n) = slice_inner_rc_evals(&field, x2c, c).unwrap();
        (o.value, n)
    });

    let path = "comparison/results/acceptance_probe.csv";
    let mut fh = std::fs::File::create(path).unwrap();
    writeln!(fh, "case,omega,classic_nodes,probe_nodes,ratio,rel_diff").unwrap();
    for r in &rows {
        writeln!(
            fh,
            "{},{},{},{},{:.4},{:.3e}",
            r.case,
            r.omega,
            r.classic,
            r.probe,
            r.classic as f64 / r.probe as f64,
            r.rel
        )
        .unwrap();
    }
    println!("wrote {path} ({} rows)", rows.len());
}
