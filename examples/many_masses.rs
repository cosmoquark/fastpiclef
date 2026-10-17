//! P4.2: single-point RC cost vs the number of point masses (10 .. 1e4).
//!
//! Records, per (N, omega): wall time (with the machine load recorded for
//! honesty — see the timing discipline in AGENTS.md), the load-independent
//! inner-solve count, the value and its error budget, and for small N a
//! two-contour (Cauchy family, alpha = pi/8 vs pi/4) agreement check.
//! Appends to `comparison/results/many_masses.csv`.
//!
//!     cargo run --release --example many_masses -- [max_N] [omegas]
//!
//! e.g. `cargo run --release --example many_masses -- 3000 3,30`.

use fast_picard_lefshetz_integral_code::phases::PointMassField;
use fast_picard_lefshetz_integral_code::{SliceConfig, slice_field};
use std::fs::OpenOptions;
use std::io::Write;
use std::time::Instant;

/// Uniform star field in [-3, 3]^2 with equal masses (0.05); the same
/// generator as `examples/batch_bench.rs`.
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

fn loadavg() -> f64 {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next().and_then(|t| t.parse::<f64>().ok()))
        .unwrap_or(f64::NAN)
}

fn threads() -> usize {
    rayon::current_num_threads()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let max_n: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(1000);
    let omegas: Vec<f64> = args
        .get(1)
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![3.0, 30.0]);

    let path = "comparison/results/many_masses.csv";
    let new = !std::path::Path::new(path).exists();
    let mut fh = OpenOptions::new().create(true).append(true).open(path).unwrap();
    if new {
        writeln!(
            fh,
            "n_lenses,omega,wall_s,inner_solves,abs_value,error,rel_alpha,tail_angle,threads,loadavg"
        )
        .unwrap();
        println!(
            "n_lenses,omega,wall_s,inner_solves,abs_value,error,rel_alpha,tail_angle,threads,loadavg"
        );
    }

    for n in [10usize, 30, 100, 300, 1000, 3000, 10000] {
        if n > max_n {
            continue;
        }
        let lenses = deterministic_lenses(n, 12345);
        for &omega in &omegas {
            let field = PointMassField { beta: [0.5, 0.0], lenses: lenses.clone() };
            let t0 = Instant::now();
            let out =
                slice_field(&field, &SliceConfig { omega, ..Default::default() }).unwrap();
            let wall = t0.elapsed().as_secs_f64();
            // Cauchy-family arbiter for the tractable sizes only.
            let rel_alpha = if n <= 300 {
                let alt = slice_field(
                    &field,
                    &SliceConfig { omega, tail_angle: std::f64::consts::FRAC_PI_8, ..Default::default() },
                )
                .unwrap();
                (out.value - alt.value).norm() / out.value.norm()
            } else {
                f64::NAN
            };
            let la = loadavg();
            let th = threads();
            println!(
                "{n:6},{omega:6.3},{wall:10.4e},{:10},{:.8e},{:.3e},{rel_alpha:.3e},pi/4,{th},{la:.2}",
                out.inner_solves,
                out.value.norm(),
                out.error
            );
            writeln!(
                fh,
                "{n},{omega},{wall:.6e},{},{:.12e},{:.6e},{rel_alpha:.6e},pi/4,{th},{la:.2}",
                out.inner_solves,
                out.value.norm(),
                out.error
            )
            .unwrap();
            fh.flush().unwrap();
        }
    }
}
