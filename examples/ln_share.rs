//! Step 0 of item 8 (SIMD vector log): the `ln` share of the many-mass
//! per-node phase sum — a load-robust A/B/C/D micro-benchmark.
//!
//! The production real-core phase sum
//! (`src/slice.rs::PointMassEval::phi_core_real`) is
//! `0.5[(x-beta0)^2 + (x2-beta1)^2] - sum_k (m_k/2) ln((x-a_k)^2 + (x2-b_k)^2)`
//! per node, with the per-slice `d_k = x2 - b_k` precomputed once per slice.
//! This example replicates that structure exactly over N point components and
//! times four variants of the same arithmetic:
//!
//! - **A** — libm `ln` (the production path);
//! - **B** — the log replaced by a bare multiply (an absolute ceiling: no log
//!   at all);
//! - **C** — a scalar bit-hack log (exponent extraction + atanh series);
//! - **D** — a 4-wide AVX2 version of C over a structure-of-arrays layout
//!   (the timing proxy for the vector log item 8 would build; x86_64 only,
//!   runtime-detected).
//!
//! Reported: ns per component-eval per variant and the A/B, A/C, A/D ratios
//! at N = 10/100/1000, from interleaved repetitions — the *ratio* is
//! load-robust, absolute walls at load are not reportable (timing discipline
//! in AGENTS.md).  The proxy's accuracy (log-uniform sample) is printed for
//! context: C and D are *timing* proxies, not the ≤1-ulp log item 8 needs.
//! Appends `comparison/results/ln_share.csv`.
//!
//!     cargo run --release --example ln_share

use std::hint::black_box;
use std::time::Instant;

/// Uniform star field in [-3, 3]^2 with equal masses (0.05); the same
/// generator as `examples/many_masses.rs` / `batch_bench.rs`.
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

/// Scalar bit-hack log: exponent extraction plus the atanh series on the
/// mantissa (`ln m = 2(y + y^3/3 + y^5/5 + y^7/7)`, `y = (m-1)/(m+1)`).
#[inline(always)]
fn fast_ln(x: f64) -> f64 {
    let bits = x.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    let y = (m - 1.0) / (m + 1.0);
    let y2 = y * y;
    2.0 * y * (1.0 + y2 * (1.0 / 3.0 + y2 * (1.0 / 5.0 + y2 * (1.0 / 7.0))))
        + e as f64 * std::f64::consts::LN_2
}

/// Per-slice evaluation data: the production layout (per-slice `d_k`), plus
/// the padded SoA copy variant D needs.
struct SliceData {
    base: f64,
    nodes: Vec<(f64, f64, f64)>,
    dk: Vec<f64>,
    soa_a: Vec<f64>,
    soa_d: Vec<f64>,
    soa_m: Vec<f64>,
}

fn prepare(lenses: &[(f64, f64, f64)]) -> Vec<SliceData> {
    const NBASES: usize = 8;
    const NNODES: usize = 1024;
    let mut out = Vec::with_capacity(NBASES);
    for bi in 0..NBASES {
        let base = -3.0 + 6.0 * (bi as f64 + 0.5) / NBASES as f64;
        let mut nodes = Vec::with_capacity(NNODES);
        for i in 0..NNODES {
            let r = 20.0 * (i as f64 + 0.5) / NNODES as f64;
            let x0 = -3.0 + 6.0 * ((i as f64 * 0.618_033_988_749_895).fract());
            let sgn = if i % 2 == 0 { 1.0 } else { -1.0 };
            nodes.push((r, x0, sgn));
        }
        let dk: Vec<f64> = lenses.iter().map(|&(_, b, _)| base - b).collect();
        let mut soa_a: Vec<f64> = lenses.iter().map(|&(a, _, _)| a).collect();
        let mut soa_d: Vec<f64> = lenses.iter().map(|&(_, b, _)| base - b).collect();
        let mut soa_m: Vec<f64> = lenses.iter().map(|&(_, _, m)| m).collect();
        while !soa_a.len().is_multiple_of(4) {
            soa_a.push(0.0);
            soa_d.push(1.0);
            soa_m.push(0.0);
        }
        out.push(SliceData { base, nodes, dk, soa_a, soa_d, soa_m });
    }
    out
}

/// The free quadratic part at a node (production `phi_free_core_real`).
#[inline(always)]
fn free_part(beta: [f64; 2], base: f64, r: f64, x0: f64, sgn: f64) -> f64 {
    let u = (x0 - beta[0]) + sgn * r;
    let d1 = base - beta[1];
    0.5 * (u * u + d1 * d1)
}

#[inline(always)]
fn sum_a(s: &SliceData, beta: [f64; 2], r: f64, x0: f64, sgn: f64, lenses: &[(f64, f64, f64)]) -> f64 {
    let mut v = free_part(beta, s.base, r, x0, sgn);
    for (k, &(a, _, m)) in lenses.iter().enumerate() {
        let t = (x0 - a) + sgn * r;
        let d = s.dk[k];
        v -= 0.5 * m * (t * t + d * d).ln();
    }
    v
}

#[inline(always)]
fn sum_b(s: &SliceData, beta: [f64; 2], r: f64, x0: f64, sgn: f64, lenses: &[(f64, f64, f64)]) -> f64 {
    let mut v = free_part(beta, s.base, r, x0, sgn);
    for (k, &(a, _, m)) in lenses.iter().enumerate() {
        let t = (x0 - a) + sgn * r;
        let d = s.dk[k];
        v -= 0.5 * m * (t * t + d * d);
    }
    v
}

#[inline(always)]
fn sum_c(s: &SliceData, beta: [f64; 2], r: f64, x0: f64, sgn: f64, lenses: &[(f64, f64, f64)]) -> f64 {
    let mut v = free_part(beta, s.base, r, x0, sgn);
    for (k, &(a, _, m)) in lenses.iter().enumerate() {
        let t = (x0 - a) + sgn * r;
        let d = s.dk[k];
        v -= 0.5 * m * fast_ln(t * t + d * d);
    }
    v
}

/// Variant D: 4-wide AVX2 log over the SoA layout (x86_64 only).  The body is
/// a single unsafe region (all intrinsics; the caller has verified `avx2`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn sum_d(s: &SliceData, beta: [f64; 2], r: f64, x0: f64, sgn: f64) -> f64 {
    use std::arch::x86_64::*;
    let n = s.soa_a.len();
    let mut acc = _mm256_setzero_pd();
    let b0 = _mm256_set1_pd(x0 - beta[0]);
    let sr = _mm256_set1_pd(sgn * r);
    let half = _mm256_set1_pd(0.5);
    let one = _mm256_set1_pd(1.0);
    let two = _mm256_set1_pd(2.0);
    let c3 = _mm256_set1_pd(1.0 / 3.0);
    let c5 = _mm256_set1_pd(1.0 / 5.0);
    let c7 = _mm256_set1_pd(1.0 / 7.0);
    let ln2 = _mm256_set1_pd(std::f64::consts::LN_2);
    let magic = _mm256_set1_pd(4503599627370496.0);
    let exp_mask = _mm256_set1_epi64x(0x7ff);
    let mant_mask = _mm256_set1_epi64x(0x000f_ffff_ffff_ffff);
    let one_bits = _mm256_set1_epi64x(0x3ff0_0000_0000_0000);
    let conv_or = _mm256_set1_epi64x(0x4330_0000_0000_0000);
    let bias = _mm256_set1_pd(1023.0);
    let mut k = 0;
    while k < n {
        let a = _mm256_loadu_pd(s.soa_a.as_ptr().add(k));
        let d = _mm256_loadu_pd(s.soa_d.as_ptr().add(k));
        let m = _mm256_loadu_pd(s.soa_m.as_ptr().add(k));
        let t = _mm256_add_pd(_mm256_sub_pd(b0, a), sr);
        let w = _mm256_add_pd(_mm256_mul_pd(t, t), _mm256_mul_pd(d, d));
        let bits = _mm256_castpd_si256(w);
        let eb = _mm256_and_si256(_mm256_srli_epi64(bits, 52), exp_mask);
        let mb = _mm256_or_si256(_mm256_and_si256(bits, mant_mask), one_bits);
        let mnt = _mm256_castsi256_pd(mb);
        let y = _mm256_div_pd(_mm256_sub_pd(mnt, one), _mm256_add_pd(mnt, one));
        let y2 = _mm256_mul_pd(y, y);
        let ser = _mm256_add_pd(
            one,
            _mm256_mul_pd(
                y2,
                _mm256_add_pd(c3, _mm256_mul_pd(y2, _mm256_add_pd(c5, _mm256_mul_pd(y2, c7)))),
            ),
        );
        let lnm = _mm256_mul_pd(two, _mm256_mul_pd(y, ser));
        let e_d = _mm256_sub_pd(
            _mm256_sub_pd(
                _mm256_castsi256_pd(_mm256_or_si256(eb, conv_or)),
                magic,
            ),
            bias,
        );
        let lnw = _mm256_add_pd(lnm, _mm256_mul_pd(e_d, ln2));
        acc = _mm256_sub_pd(acc, _mm256_mul_pd(_mm256_mul_pd(half, m), lnw));
        k += 4;
    }
    let mut lane = [0.0f64; 4];
    _mm256_storeu_pd(lane.as_mut_ptr(), acc);
    let mut v = free_part(beta, s.base, r, x0, sgn);
    for l in lane {
        v += l;
    }
    v
}

fn run(variant: u8, slices: &[SliceData], beta: [f64; 2], lenses: &[(f64, f64, f64)]) -> f64 {
    let mut total = 0.0f64;
    for s in slices {
        for &(r, x0, sgn) in &s.nodes {
            total += match variant {
                0 => sum_a(s, beta, r, x0, sgn, lenses),
                1 => sum_b(s, beta, r, x0, sgn, lenses),
                2 => sum_c(s, beta, r, x0, sgn, lenses),
                _ => {
                    #[cfg(target_arch = "x86_64")]
                    {
                        // SAFETY: `avx2` is checked at runtime in `main`.
                        unsafe { sum_d(s, beta, r, x0, sgn) }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        0.0
                    }
                }
            };
        }
    }
    total
}

fn loadavg() -> f64 {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next().and_then(|t| t.parse::<f64>().ok()))
        .unwrap_or(f64::NAN)
}

fn main() {
    let beta = [0.5, 0.0];
    let have_avx2 = {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    };
    let n_variants = if have_avx2 { 4 } else { 3 };
    let names = ["A ln", "B none", "C scalar", "D avx2"];

    // Accuracy of the proxies on a log-uniform sample (context: the shipped
    // vector log must be <= 1 ulp; these are timing proxies).
    let mut worst_abs = 0.0f64;
    let mut worst_rel = 0.0f64;
    for i in 0..100_000 {
        let x = 10f64.powf(-6.0 + 12.0 * (i as f64 + 0.5) / 100_000.0);
        let d = (fast_ln(x) - x.ln()).abs();
        worst_abs = worst_abs.max(d);
        if x.ln().abs() > 1e-3 {
            worst_rel = worst_rel.max(d / x.ln().abs());
        }
    }
    println!(
        "proxies (C/D): max abs err {worst_abs:.2e}, max rel err {worst_rel:.2e} (|ln x| > 1e-3)"
    );
    println!("avx2 available: {have_avx2}");
    println!();
    println!("{:<8} {:<9} {:>12} {:>14} {:>9}", "n_lenses", "var", "ns/node", "ns/component", "ratio/A");

    let path = "comparison/results/ln_share.csv";
    let new = !std::path::Path::new(path).exists();
    let mut fh = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
    use std::io::Write;
    if new {
        writeln!(fh, "n_lenses,variant,ns_per_node,ns_per_component,ratio_vs_a,loadavg").unwrap();
    }

    const REPS: usize = 5;
    for n in [10usize, 100, 1000] {
        let lenses = deterministic_lenses(n, 12345);
        let slices = prepare(&lenses);
        let n_nodes = slices.iter().map(|s| s.nodes.len()).sum::<usize>();
        // Value sanity: the proxies must reproduce A up to their own error
        // (a broken vector log would otherwise look "fast" for free).
        let ta = run(0, &slices, beta, &lenses);
        let tc = run(2, &slices, beta, &lenses);
        let td = if have_avx2 { run(3, &slices, beta, &lenses) } else { ta };
        println!(
            "n={n:<5} value check: |C-A|/|A| = {:.2e}  |D-A|/|A| = {:.2e}",
            (tc - ta).abs() / ta.abs().max(1.0),
            (td - ta).abs() / ta.abs().max(1.0)
        );
        let mut times = [[0.0f64; REPS]; 4];
        for rep in 0..REPS {
            for (v, row) in times.iter_mut().enumerate().take(n_variants) {
                let t0 = Instant::now();
                let total = run(v as u8, &slices, beta, &lenses);
                black_box(total);
                row[rep] = t0.elapsed().as_secs_f64() / n_nodes as f64;
            }
        }
        let best: Vec<f64> = times
            .iter()
            .take(n_variants)
            .map(|t| t.iter().copied().fold(f64::INFINITY, f64::min))
            .collect();
        let la = loadavg();
        for (v, name) in names.iter().enumerate().take(n_variants) {
            let ns_node = best[v] * 1e9;
            let ns_comp = ns_node / n as f64;
            let ratio = best[0] / best[v];
            println!("{n:<8} {name:<9} {ns_node:>12.1} {ns_comp:>14.2} {ratio:>9.2}");
            writeln!(fh, "{n},{name},{ns_node:.3},{ns_comp:.4},{ratio:.4},{la:.2}").unwrap();
        }
        println!();
    }
    println!("(loadavg {:.2}; ratios are the load-robust quantity)", loadavg());
}
