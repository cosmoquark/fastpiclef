//! Cost model for P0: ns/op of the elementary operations the slice kernels are
//! made of (complex mul/div, real ln, complex exp = exp+sincos, complex ln =
//! hypot+ln+atan2). Run: `cargo run --release --example opcost`.
//!
//! These numbers feed the batched-solver speedup projection: a shared-contour
//! batch pays `N` logs once per node and then one complex multiply per grid
//! point, versus one `exp(i w phi)` per (node, point) for independent solves.

use num_complex::Complex64;
use std::hint::black_box;
use std::time::Instant;

fn bench<F: FnMut(usize) -> f64>(n: usize, reps: usize, mut f: F) -> f64 {
    // warmup
    black_box(f(n));
    let t0 = Instant::now();
    let mut acc = 0.0;
    for _ in 0..reps {
        acc += black_box(f(n));
    }
    let dt = t0.elapsed().as_secs_f64() / (reps as f64);
    black_box(acc);
    dt / (n as f64) * 1e9
}

fn main() {
    let n = 2_000_000usize;
    let reps = 8usize;

    let z0 = Complex64::new(0.37, -1.21);
    let z1 = Complex64::new(-2.04, 0.55);
    let x = 0.731_f64;

    let t_mul = bench(n, reps, |n| {
        let mut a = z0;
        for i in 0..n {
            a = a * z1 + Complex64::new(i as f64 * 1e-9, 0.0);
        }
        a.norm()
    });
    let t_div = bench(n, reps, |n| {
        let mut a = z0;
        for i in 0..n {
            a = a / z1 + Complex64::new(i as f64 * 1e-9, 0.0);
        }
        a.norm()
    });
    let t_ln = bench(n, reps, |n| {
        let mut s = 0.0;
        for i in 0..n {
            s += (x + i as f64 * 1e-9).ln();
        }
        s
    });
    let t_sincos = bench(n, reps, |n| {
        let mut s = 0.0;
        for i in 0..n {
            let (si, co) = (x + i as f64 * 1e-9).sin_cos();
            s += si + co;
        }
        s
    });
    let t_cexp = bench(n, reps, |n| {
        let mut a = z0;
        for i in 0..n {
            a += (z1 * (i as f64) * 1e-9).exp();
        }
        a.norm()
    });
    let t_clog = bench(n, reps, |n| {
        let mut a = Complex64::new(1.0, 1e-9);
        for i in 0..n {
            a += (z1 + Complex64::new(i as f64 * 1e-9, 0.0)).ln();
        }
        a.norm()
    });
    let e = |v: f64| format!("{v:8.2}");
    println!("ns/op (release, n={n}, reps={reps}):");
    println!("  complex mul        {}", e(t_mul));
    println!("  complex div        {}", e(t_div));
    println!("  real ln            {}", e(t_ln));
    println!("  real sin_cos       {}", e(t_sincos));
    println!("  complex exp        {}", e(t_cexp));
    println!("  complex ln         {}", e(t_clog));
    println!(
        "  ratios: mul/exp {:.3}  ln/exp {:.3}  mul/ln {:.3}",
        t_mul / t_cexp,
        t_ln / t_cexp,
        t_mul / t_ln
    );
}
