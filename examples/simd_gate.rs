//! Item 8, Step 3 — the wall gate for the opt-in vector log
//! (`SliceConfig::simd_log`): solve the same many-mass field twice per
//! repetition (scalar libm vs the 4-wide AVX2 log) in one process and report
//! the ratio, the value agreement and the load.
//!
//! Landing criterion (plan): >= ~1.7x at N = 100 in a **quiet window**; the
//! bitwise baselines are regenerated deliberately if the default flips.
//!
//!     cargo run --release --example simd_gate -- [N] [omega] [reps]
//!
//! e.g. `cargo run --release --example simd_gate -- 100 3 3`.
//! Absolute walls are only reportable at low load; the *ratio* is the
//! load-robust quantity (timing discipline in AGENTS.md).  The reported
//! `rel` is the scalar/vector value agreement (the vector log is <= 1 ulp per
//! term, so this sits at the roundoff level).

use fast_picard_lefshetz_integral_code::phases::PointMassField;
use fast_picard_lefshetz_integral_code::{SliceConfig, slice_field};
use std::time::Instant;

/// Uniform star field in [-3, 3]^2 with equal masses (0.05); the same
/// generator as `examples/many_masses.rs`.
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(100);
    let omega: f64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3.0);
    let reps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);

    let field = PointMassField { beta: [0.5, 0.0], lenses: deterministic_lenses(n, 12345) };
    let base = SliceConfig { omega, ..Default::default() };

    println!("n_lenses={n} omega={omega} reps={reps} loadavg={:.2}", loadavg());
    println!("{:<6} {:>12} {:>12} {:>9} {:>10}", "rep", "scalar_s", "vector_s", "ratio", "rel");

    let mut ratios = Vec::new();
    let mut rels = Vec::new();
    for rep in 0..reps {
        let t0 = Instant::now();
        let scalar = slice_field(&field, &base).unwrap();
        let t_scalar = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let vector = slice_field(&field, &SliceConfig { simd_log: true, ..base.clone() }).unwrap();
        let t_vector = t0.elapsed().as_secs_f64();
        let rel = (scalar.value - vector.value).norm() / scalar.value.norm();
        ratios.push(t_scalar / t_vector);
        rels.push(rel);
        println!(
            "{rep:<6} {t_scalar:>12.4} {t_vector:>12.4} {:>9.3} {rel:>10.2e}",
            t_scalar / t_vector
        );
        println!(
            "       (steps: scalar {} vs vector {}; budgets {:.2e} vs {:.2e}; loadavg {:.2})",
            scalar.inner_solves,
            vector.inner_solves,
            scalar.error,
            vector.error,
            loadavg()
        );
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    rels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!();
    println!(
        "ratio: median {:.3}, min {:.3}, max {:.3} (gate >= 1.7); worst value rel {:.2e}",
        ratios[ratios.len() / 2],
        ratios[0],
        ratios[ratios.len() - 1],
        rels[rels.len() - 1]
    );
}
