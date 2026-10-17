//! P1 acceptance benchmark: batched vs per-point RC (plan §3.7).  Writes
//! `comparison/results/batch_bench.csv` (append; one set of rows per run) and
//! prints a summary.  Thread count comes from `RAYON_NUM_THREADS` (the CSV
//! records `rayon::current_num_threads()`), so run it twice:
//!
//!     cargo run --release --example batch_bench -- 1     # RAYON_NUM_THREADS=1
//!     cargo run --release --example batch_bench          # all threads
//!
//! Cases:
//!   * 1D P-LPI grid (512 sources, omega = 10, kappa = -2): batch vs the
//!     production per-point thimble vs per-point RC;
//!   * 2D source maps: N = 1, 3, 40 lenses, grids 16x16 and 64x64, omega = 1/3:
//!     batch vs a per-point subsample (extrapolated, labelled);
//!   * the N = 100 4x4 source map at omega = 3 (the report's amortisation
//!     rounding case): batch vs a single-anchor extrapolation;
//!   * frequency sweeps: 200 log-spaced omega in [0.01, 10], N = 1 and 40:
//!     batch vs a per-point subsample.

use fast_picard_lefshetz_integral_code::phases::{PointMassField, RationalLens};
use fast_picard_lefshetz_integral_code::{
    FieldBatch, SliceConfig, ThimbleConfig, ThimbleSolver, rc_1d_batch, slice_field,
    slice_field_batch,
};
use std::fs::OpenOptions;
use std::io::Write;
use std::time::Instant;

fn deterministic_lenses(n: usize, seed: u64) -> Vec<(f64, f64, f64)> {
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n).map(|_| (next() * 6.0 - 3.0, next() * 6.0 - 3.0, 0.05)).collect()
}

fn threads() -> usize {
    rayon::current_num_threads()
}

fn loadavg() -> f64 {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next().and_then(|t| t.parse::<f64>().ok()))
        .unwrap_or(f64::NAN)
}

fn log_row(case: &str, n_lenses: usize, grid: &str, n_omega: usize, mode: &str, wall: f64, note: &str) {
    let path = "comparison/results/batch_bench.csv";
    let new = !std::path::Path::new(path).exists();
    let mut fh = OpenOptions::new().create(true).append(true).open(path).unwrap();
    if new {
        writeln!(fh, "case,n_lenses,grid,n_omega,threads,mode,wall_s,loadavg,note").unwrap();
    }
    writeln!(
        fh,
        "{case},{n_lenses},{grid},{n_omega},{},{mode},{wall:.6e},{:.2},{note}",
        threads(),
        loadavg()
    )
    .unwrap();
}

fn main() {
    let t_all = Instant::now();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let want = |case: &str| {
        args.is_empty() || args.iter().any(|a| a == "all" || a == case)
    };

    // ---- 1D: the P-LPI grid (512 sources, omega = 10, kappa = -2) ----
    if want("1d") {
        let omega = 10.0;
        let n = 512;
        let betas: Vec<f64> = (0..n).map(|i| -4.0 + 8.0 * i as f64 / (n as f64 - 1.0)).collect();
        let cfg = SliceConfig { omega, ..Default::default() };
        let lens = RationalLens { beta: 0.0, kappa: -2.0 };

        let t0 = Instant::now();
        let (vals, _) = rc_1d_batch(&lens, &betas, omega, &cfg).unwrap();
        let t_batch = t0.elapsed().as_secs_f64();
        log_row("1d-plpi-grid", 1, "512-betas", 1, "batch-rc", t_batch, "shared contour");
        println!("1D P-LPI grid (512): batch RC {:.4} s", t_batch);

        // per-point production thimble
        let t0 = Instant::now();
        for &b in &betas {
            let _ = ThimbleSolver::new_rational(
                &RationalLens { beta: b, kappa: -2.0 },
                ThimbleConfig { omega, ..Default::default() },
            )
            .run()
            .unwrap()
            .value;
        }
        let t_thim = t0.elapsed().as_secs_f64();
        log_row("1d-plpi-grid", 1, "512-betas", 1, "thimble-loop", t_thim, "per-point production");
        println!("1D P-LPI grid (512): thimble loop {:.4} s  (speedup {:.1}x)", t_thim, t_thim / t_batch);

        // per-point independent RC (own contour each)
        let t0 = Instant::now();
        for &b in &betas {
            let _ = rc_1d_batch(&lens, &[b], omega, &cfg).unwrap();
        }
        let t_ind = t0.elapsed().as_secs_f64();
        log_row("1d-plpi-grid", 1, "512-betas", 1, "rc-loop", t_ind, "independent contours");
        println!("1D P-LPI grid (512): independent RC loop {:.4} s  (speedup {:.1}x)", t_ind, t_ind / t_batch);
        let _ = vals;
    }

    // ---- 2D source maps ----
    if want("maps") {
    for &n_lenses in &[1usize, 3, 40] {
        let lenses = deterministic_lenses(n_lenses, 0x9E3779B97F4A7C15);
        for &side in &[16usize, 64] {
            for &omega in &[1.0f64, 3.0] {
                let g: Vec<f64> = (0..side).map(|i| -1.0 + 2.0 * i as f64 / (side as f64 - 1.0)).collect();
                let field = PointMassField { beta: [0.0, 0.0], lenses: lenses.clone() };
                let cfg = SliceConfig { omega, ..Default::default() };
                let batch = FieldBatch { omegas: vec![omega], beta0s: g.clone(), beta1s: g.clone() };
                let t0 = Instant::now();
                let out = slice_field_batch(&field, &batch, &cfg).unwrap();
                let t_batch = t0.elapsed().as_secs_f64();
                log_row(
                    "2d-map",
                    n_lenses,
                    &format!("{side}x{side}"),
                    1,
                    "batch",
                    t_batch,
                    &format!("groups={}", out.groups),
                );
                // per-point subsample (8 points, corner/centre spread)
                let sub: Vec<(f64, f64)> = (0..8)
                    .map(|k| {
                        let i = (k * (side - 1) / 7).min(side - 1);
                        (g[i], g[(i * 5) % side])
                    })
                    .collect();
                let t0 = Instant::now();
                for &(b0, b1) in &sub {
                    let f = PointMassField { beta: [b0, b1], lenses: lenses.clone() };
                    let _ = slice_field(&f, &cfg).unwrap();
                }
                let t_sub = t0.elapsed().as_secs_f64();
                let per = t_sub / sub.len() as f64;
                let est = per * (side * side) as f64;
                log_row(
                    "2d-map",
                    n_lenses,
                    &format!("{side}x{side}"),
                    1,
                    "scalar-sub8",
                    est,
                    &format!("extrapolated from 8 points ({per:.4}s/pt); measured {t_sub:.4}s"),
                );
                println!(
                    "2D map {side}x{side} N={n_lenses} w={omega}: batch {t_batch:.4} s vs est scalar {est:.4} s  (speedup {:.1}x)",
                    est / t_batch
                );
            }
        }
    }

    }

    // ---- N = 100 masses, 4x4 source map (the amortisation rounding case) ----
    if want("map-100") {
        let n_lenses = 100usize;
        let side = 4usize;
        let omega = 3.0;
        let lenses = deterministic_lenses(n_lenses, 0x9E3779B97F4A7C15);
        let g: Vec<f64> = (0..side).map(|i| -1.0 + 2.0 * i as f64 / (side as f64 - 1.0)).collect();
        let field = PointMassField { beta: [0.0, 0.0], lenses: lenses.clone() };
        let cfg = SliceConfig { omega, ..Default::default() };
        let batch = FieldBatch { omegas: vec![omega], beta0s: g.clone(), beta1s: g.clone() };
        let t0 = Instant::now();
        let out = slice_field_batch(&field, &batch, &cfg).unwrap();
        let t_batch = t0.elapsed().as_secs_f64();
        log_row(
            "map-100",
            n_lenses,
            "4x4",
            1,
            "batch",
            t_batch,
            &format!("groups={}; {:.3} s/pt-equiv", out.groups, t_batch / 16.0),
        );
        // one scalar anchor at the nearest-to-centre grid point, extrapolated x16
        let bc = g[side / 2];
        let f = PointMassField { beta: [bc, bc], lenses: lenses.clone() };
        let t0 = Instant::now();
        let _ = slice_field(&f, &cfg).unwrap();
        let t_1 = t0.elapsed().as_secs_f64();
        let est = 16.0 * t_1;
        log_row(
            "map-100",
            n_lenses,
            "4x4",
            1,
            "scalar-anchor",
            est,
            &format!("1 point ({t_1:.3} s at beta=({bc:.3},{bc:.3})), x16"),
        );
        println!(
            "map-100 N=100 4x4 w=3: batch {t_batch:.3} s vs scalar anchor est {est:.3} s  (speedup {:.1}x)",
            est / t_batch
        );
    }

    // ---- frequency sweeps ----
    if want("sweeps") {
    for &n_lenses in &[1usize, 40] {
        let lenses = deterministic_lenses(n_lenses, 0xD1B54A32D192ED03);
        let nw = 200;
        let omegas: Vec<f64> = (0..nw)
            .map(|i| 0.01 * (1000.0f64).powf(i as f64 / (nw as f64 - 1.0)))
            .collect();
        let field = PointMassField { beta: [0.0, 0.0], lenses: lenses.clone() };
        let cfg = SliceConfig { omega: 1.0, ..Default::default() };
        let batch = FieldBatch {
            omegas: omegas.clone(),
            beta0s: vec![0.0],
            beta1s: vec![0.0],
        };
        let t0 = Instant::now();
        let out = slice_field_batch(&field, &batch, &cfg).unwrap();
        let t_batch = t0.elapsed().as_secs_f64();
        log_row("sweep", n_lenses, "point", nw, "batch", t_batch, &format!("groups={}", out.groups));
        // per-omega subsample (6 frequencies)
        let sub: Vec<f64> = (0..6).map(|k| omegas[k * (nw - 1) / 5]).collect();
        let t0 = Instant::now();
        for &w in &sub {
            let _ = slice_field(&field, &SliceConfig { omega: w, ..Default::default() }).unwrap();
        }
        let t_sub = t0.elapsed().as_secs_f64();
        let per = t_sub / sub.len() as f64;
        let est = per * nw as f64;
        log_row(
            "sweep",
            n_lenses,
            "point",
            nw,
            "scalar-sub6",
            est,
            &format!("extrapolated from 6 omegas ({per:.4}s/freq); measured {t_sub:.4}s"),
        );
        println!(
            "sweep 200 omegas N={n_lenses}: batch {t_batch:.4} s vs est scalar {est:.4} s  (speedup {:.1}x)",
            est / t_batch
        );
    }

    }

    println!("total {:.1} s on {} threads", t_all.elapsed().as_secs_f64(), threads());
}
