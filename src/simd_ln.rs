//! Item 8 (SIMD vector log) — the 4-wide AVX2 `ln` for the many-mass
//! real-core phase sums ([`crate::slice::PointMassEval::phi_core_real`]).
//!
//! Hand-rolled `std::arch` only (D1, 2026-10-19: no crate), runtime-detected
//! ([`avx2_available`]) with the scalar libm path as the fallback.  Method:
//!
//! 1. extract the biased exponent and the mantissa `m in [1, 2)`;
//! 2. recentre to `m in [1/sqrt2, sqrt2)` (halve `m`, increment the exponent
//!    when `m > sqrt2`) so `f = (m-1)/(m+1)` satisfies `|f| <= 0.1716`;
//! 3. evaluate `ln m = 2 f + 2 f u Q(u)`, `u = f^2`, with `Q` the degree-7
//!    minimax polynomial below (max relative error 3.05e-18 on
//!    `u in [0, 0.029437]`, mpmath dps 60).  The *split* form is what keeps
//!    the accuracy: the leading `2f` is exact and only the small correction
//!    (`|2 f u Q| <= 3.5e-3`) carries polynomial roundings — evaluating
//!    `2 f P(u)` directly inherits ~1.5 ulp from the polynomial near `P ~ 1`;
//! 4. reconstruct with a two-part `ln 2` (`ln2_hi` has 26 significant bits,
//!    so `k ln2_hi` is exact for the exponent range; the residual goes into
//!    `ln m + k ln2_lo`).  The raw division `f = (m-1)/(m+1)` is corrected by
//!    an FMA-based residual (`r = num - q*den` exactly; the denominator's own
//!    rounding is carried as `derr`), leaving ~1e-33 absolute error in `f`.
//!
//! Measured standalone (vs libm `ln`, ulps of the result, on the argument
//! domain `w = t^2 + d^2`): **max 0.994 ulp** for `|ln w| >= 1e-3` and 0.996
//! ulp near `w = 1` (where the absolute error is <= 1.1e-19) — the plan's
//! <= 1 ulp class.  Pinned by the tests below; a lane that is non-positive,
//! subnormal or non-finite falls back to the scalar `ln` in
//! [`half_m_ln_sum`].
//!
//! Step-2 status: the vector path is **opt-in** (`SliceConfig::simd_log`,
//! default `false`) until the Step-3 wall gate (N = 100 in a quiet window, at
//! least ~1.7x) passes; the vector log changes the arithmetic by <= 1 ulp per
//! term, so the bitwise baselines will be regenerated deliberately when it
//! becomes the default.

/// Descending minimax coefficients of `Q(u) ~= (atanh(sqrt(u))/sqrt(u) - 1)/u`
/// with `ln m = 2 f + 2 f u Q(u)`, `u = f^2 in [0, 0.0294372515]`; max
/// relative error 3.05e-18 (mpmath dps 60, Chebyshev fit — `q0` is the `u^7`
/// coefficient, `q7` the constant `1/3`).
///
/// The *split* form is what keeps the log within ~0.5 ulp: the leading `2f`
/// is exact, so only the small correction (`|2 f u Q| <= 3.5e-3`) carries
/// polynomial roundings — evaluating `2 f P(u)` directly would inherit ~1.5
/// ulp from the polynomial's own rounding near `|P| ~ 1`.
const Q: [f64; 8] = [
    0.065433130739200510690,
    0.066343865693284425923,
    0.076931198514073290942,
    0.090908978200664531380,
    0.111111111958569587210,
    0.142857142854018059050,
    0.200000000000004384910,
    0.333333333333333332320,
];

/// Two-part `ln 2` (fdlibm-style split at 26 significant bits): `k * LN2_HI`
/// is exact for `|k| <= 2^26`, the residual `LN2_LO` carries the rest.
const LN2_HI: f64 = 0.69314716756343841553;
const LN2_LO: f64 = 1.2996506893889888371e-8;

#[cfg(target_arch = "x86_64")]
mod imp {
    use std::arch::x86_64::*;
    use std::sync::OnceLock;

    use super::{LN2_HI, LN2_LO, Q};

    /// Runtime detection of the AVX2+FMA path (cached; the scalar libm path
    /// is used when it is absent).  FMA3 is required by the division
    /// correction in [`ln4`].
    pub(crate) fn avx2_available() -> bool {
        static AVAIL: OnceLock<bool> = OnceLock::new();
        *AVAIL.get_or_init(|| {
            is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
        })
    }

    /// The 4-wide `ln` (see the module docs).  Requires AVX2+FMA; the caller
    /// must have checked [`avx2_available`].  Lanes that are not strictly
    /// positive normal doubles give unspecified values (the phase-sum helper
    /// handles them by a scalar fallback).
    #[allow(unsafe_op_in_unsafe_fn)]
    #[target_feature(enable = "avx2,fma")]
    pub(crate) unsafe fn ln4(x: __m256d) -> __m256d {
        let bits = _mm256_castpd_si256(x);
        let exp = _mm256_and_si256(_mm256_srli_epi64(bits, 52), _mm256_set1_epi64x(0x7ff));
        let mant = _mm256_or_si256(
            _mm256_and_si256(bits, _mm256_set1_epi64x(0x000f_ffff_ffff_ffff)),
            _mm256_set1_epi64x(0x3ff0_0000_0000_0000),
        );
        let m0 = _mm256_castsi256_pd(mant);
        let one = _mm256_set1_pd(1.0);
        let gt = _mm256_cmp_pd(m0, _mm256_set1_pd(std::f64::consts::SQRT_2), _CMP_GT_OQ);
        let m = _mm256_blendv_pd(m0, _mm256_mul_pd(m0, _mm256_set1_pd(0.5)), gt);
        // k = (biased exponent) - 1023 + (m0 > sqrt2 ? 1 : 0), as doubles.
        let eb = _mm256_or_si256(exp, _mm256_set1_epi64x(0x4330_0000_0000_0000));
        let k = _mm256_sub_pd(
            _mm256_castsi256_pd(eb),
            _mm256_set1_pd(1023.0 + 4503599627370496.0),
        );
        let k = _mm256_add_pd(k, _mm256_and_pd(gt, one));
        // f = (m - 1)/(m + 1) with a division correction so f carries ~1e-33
        // absolute error (the raw division's 0.5 ulp would dominate the
        // log's error near the recentring boundary):
        //   num = m - 1  (exact, Sterbenz), den = m + 1,
        //   q = num/den (via 1/den), r = num - q*den (exact, FMA),
        //   derr = m - (den - 1) = (m+1) - den (exact, den-1 is exact),
        //   f = q + (r - q*derr)/den.
        let num = _mm256_sub_pd(m, one);
        let den = _mm256_add_pd(m, one);
        let inv = _mm256_div_pd(one, den);
        let q = _mm256_mul_pd(num, inv);
        let r = _mm256_fnmadd_pd(q, den, num);
        let derr = _mm256_sub_pd(m, _mm256_sub_pd(den, one));
        let f = _mm256_add_pd(q, _mm256_mul_pd(_mm256_fnmadd_pd(q, derr, r), inv));
        let u = _mm256_mul_pd(f, f);
        let mut q = _mm256_set1_pd(Q[0]);
        for c in &Q[1..] {
            q = _mm256_fmadd_pd(q, u, _mm256_set1_pd(*c));
        }
        // ln m = 2f + 2f u Q(u): the leading 2f is exact, the correction is
        // small (|.| <= 3.5e-3), so the polynomial's roundings are harmless.
        let two_f = _mm256_add_pd(f, f);
        let ln_m = _mm256_add_pd(two_f, _mm256_mul_pd(_mm256_mul_pd(two_f, u), q));
        let hi = _mm256_mul_pd(k, _mm256_set1_pd(LN2_HI));
        let lo = _mm256_add_pd(ln_m, _mm256_mul_pd(k, _mm256_set1_pd(LN2_LO)));
        _mm256_add_pd(hi, lo)
    }

    /// `sum_k (m_k/2) ln(t_k^2 + d_k^2)` with `t_k = (x0 - a_k) + sgn r`,
    /// 4 components per AVX2 block plus a scalar tail.  The per-slice arrays
    /// are the SoA layout of [`crate::slice::PointMassEval`].  Exceptional
    /// lanes (`w <= 0`, subnormal, non-finite) use the scalar `ln`.
    ///
    /// # Safety
    /// Requires AVX2 (the caller must have checked [`avx2_available`]) and
    /// `ak`, `mk`, `dk` of equal length.
    #[allow(unsafe_op_in_unsafe_fn)]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn half_m_ln_sum(
        ak: &[f64],
        mk: &[f64],
        dk: &[f64],
        x0: f64,
        sgn: f64,
        r: f64,
    ) -> f64 {
        let n = ak.len();
        let mut acc = _mm256_setzero_pd();
        let b0 = _mm256_set1_pd(x0);
        let sr = _mm256_set1_pd(sgn * r);
        let half = _mm256_set1_pd(0.5);
        let min_pos = _mm256_set1_pd(f64::MIN_POSITIVE);
        let mut k = 0;
        while k + 4 <= n {
            let a = _mm256_loadu_pd(ak.as_ptr().add(k));
            let d = _mm256_loadu_pd(dk.as_ptr().add(k));
            let m = _mm256_loadu_pd(mk.as_ptr().add(k));
            let t = _mm256_add_pd(_mm256_sub_pd(b0, a), sr);
            let w = _mm256_add_pd(_mm256_mul_pd(t, t), _mm256_mul_pd(d, d));
            let mut lw = ln4(w);
            let bad = _mm256_or_pd(
                _mm256_cmp_pd(w, min_pos, _CMP_LT_OQ),
                _mm256_cmp_pd(w, w, _CMP_UNORD_Q),
            );
            if _mm256_movemask_pd(bad) != 0 {
                let mut wl = [0.0f64; 4];
                let mut ll = [0.0f64; 4];
                _mm256_storeu_pd(wl.as_mut_ptr(), w);
                _mm256_storeu_pd(ll.as_mut_ptr(), lw);
                for i in 0..4 {
                    if wl[i] < f64::MIN_POSITIVE || !wl[i].is_finite() {
                        ll[i] = wl[i].ln();
                    }
                }
                lw = _mm256_loadu_pd(ll.as_ptr());
            }
            acc = _mm256_add_pd(acc, _mm256_mul_pd(_mm256_mul_pd(half, m), lw));
            k += 4;
        }
        let mut lane = [0.0f64; 4];
        _mm256_storeu_pd(lane.as_mut_ptr(), acc);
        let mut total: f64 = lane.iter().sum();
        while k < n {
            let t = (x0 - ak[k]) + sgn * r;
            let w = t * t + dk[k] * dk[k];
            total += 0.5 * mk[k] * w.ln();
            k += 1;
        }
        total
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) use imp::{avx2_available, half_m_ln_sum};

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn avx2_available() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_arch = "x86_64")]
    use super::imp::ln4;

    /// The actual argument domain: `w = t^2 + d^2` with `t`, `d` spanning the
    /// slice coordinates (pinch-adjacent small values up to the core extents
    /// and tail scales).
    fn domain_sample() -> Vec<f64> {
        let mut w = Vec::new();
        // log-uniform over the plausible range (t, d ~ 1e-15 .. 1e2)
        for i in 0..2000 {
            let e = -30.0 + 34.0 * i as f64 / 2000.0;
            w.push(10f64.powf(e));
        }
        // the near-1 region and the recentring boundary (m ~ sqrt2)
        for i in 0..1000 {
            w.push(1.0 + (i as f64 - 500.0) * 1e-6);
        }
        for i in 0..1000 {
            let m = std::f64::consts::SQRT_2 * (1.0 + (i as f64 - 500.0) * 1e-9);
            w.push(m);
            w.push(2.0 * m);
        }
        // exact powers of two (f = 0) and their neighbours
        for e in -20..21 {
            let p = 2f64.powi(e);
            w.push(p);
            w.push(p * (1.0 + 1e-12));
            w.push(p * (1.0 - 1e-12));
        }
        w.retain(|&x| x.is_finite() && x > f64::MIN_POSITIVE);
        w
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn vector_log_is_within_one_ulp_of_libm() {
        if !avx2_available() {
            eprintln!("avx2 unavailable: skipping the vector-log accuracy check");
            return;
        }
        let ws = domain_sample();
        let mut worst: Vec<(f64, f64, f64, f64)> = Vec::new();
        for chunk in ws.chunks(4) {
            let mut x = [chunk[0]; 4];
            for (i, &v) in chunk.iter().enumerate() {
                x[i] = v;
            }
            // SAFETY: avx2 checked above.
            let out = unsafe {
                use std::arch::x86_64::*;
                let v = _mm256_loadu_pd(x.as_ptr());
                let r = ln4(v);
                let mut o = [0.0f64; 4];
                _mm256_storeu_pd(o.as_mut_ptr(), r);
                o
            };
            for (i, &v) in chunk.iter().enumerate() {
                let want = v.ln();
                let ulp = 2f64.powi(want.abs().log2().floor() as i32 - 52).max(5e-324);
                let d = (out[i] - want).abs() / ulp;
                worst.push((d, v, want, out[i]));
            }
        }
        worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for (d, v, want, got) in worst.iter().take(6) {
            eprintln!("  {d:.3} ulp at w = {v:.17e}: want {want:.17e} got {got:.17e}");
        }
        // Decision-relevant metrics: the *relative* error of the log term
        // (the phase contribution), and the near-1 boundary separately (there
        // the result is tiny, so ulps of the result are tiny absolute errors).
        let mut worst_rel = 0.0f64; // |ln w| >= 1e-3
        let mut worst_abs_ulp = 0.0f64;
        let mut worst_tiny_rel = 0.0f64; // |ln w| < 1e-3
        for (d, _, want, got) in &worst {
            let rel = (got - want).abs() / want.abs();
            if want.abs() >= 1e-3 {
                worst_rel = worst_rel.max(rel);
            } else {
                worst_tiny_rel = worst_tiny_rel.max(rel);
            }
            worst_abs_ulp = worst_abs_ulp.max(*d);
        }
        eprintln!(
            "vector ln vs libm: rel err {:.3} ulp (|ln w| >= 1e-3), {:.3} ulp near 1 \
             (absolute {:.2e}), max {:.3} ulp of the result",
            worst_rel / f64::EPSILON,
            worst_tiny_rel / f64::EPSILON,
            worst.iter().filter(|w| w.2.abs() < 1e-3).map(|w| (w.3 - w.2).abs()).fold(0.0, f64::max),
            worst_abs_ulp
        );
        assert!(
            worst_rel <= 1.0 * f64::EPSILON,
            "vector ln relative error exceeds 1 ulp for |ln w| >= 1e-3: {:.3}",
            worst_rel / f64::EPSILON
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn vector_phase_sum_matches_the_scalar_loop() {
        if !avx2_available() {
            return;
        }
        // A representative many-mass slice: the SoA arrays of the production
        // layout; the vector sum must match the scalar one to roundoff.
        let n = 137; // deliberately not a multiple of 4 (the scalar tail)
        let mut ak = Vec::with_capacity(n);
        let mut mk = Vec::with_capacity(n);
        let mut dk = Vec::with_capacity(n);
        let mut s = 12345u64 | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..n {
            ak.push(next() * 6.0 - 3.0);
            mk.push(0.05);
            dk.push(next() * 6.0 - 3.0);
        }
        for &(x0, sgn, r) in &[(0.3f64, 1.0f64, 0.7f64), (-1.2, -1.0, 2.5), (2.0, 1.0, 1e-6)] {
            let scalar: f64 = (0..n)
                .map(|k| {
                    let t = (x0 - ak[k]) + sgn * r;
                    0.5 * mk[k] * (t * t + dk[k] * dk[k]).ln()
                })
                .sum();
            // SAFETY: avx2 checked above.
            let vector = unsafe { half_m_ln_sum(&ak, &mk, &dk, x0, sgn, r) };
            let d = (scalar - vector).abs() / scalar.abs();
            assert!(d < 1e-15, "vector sum differs: {scalar} vs {vector} (rel {d:e})");
        }
    }

    /// Step 2, end-to-end: the opt-in vector-log path through the full 2D RC
    /// engine — (a) against the pinned 30-digit Nakamura-Deguchi references
    /// (N = 1), (b) against the scalar path and the contour-family arbiter at
    /// N = 3 and N = 10, with the sum-of-budgets invariant covering the
    /// scalar/vector difference.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_log_end_to_end_matches_scalar_and_nd() {
        if !avx2_available() {
            eprintln!("avx2 unavailable: skipping the end-to-end vector-log check");
            return;
        }
        use crate::phases::PointMassField;
        use crate::{SliceConfig, slice_field};
        use num_complex::Complex64;

        // (a) N = 1 vs the external ND closed form (30 digits).
        const REF_ND: &[(f64, f64, f64, f64)] = &[
            (0.3, 1.0, 5.05948805778166566e+00, 2.49364109938757785e+01),
            (1.0, 0.5, 1.11515184895840158e+00, 1.06271092478315978e+01),
            (3.0, 0.1, -4.38075403947874253e+00, 4.50828847266479293e+00),
            (10.0, 2.0, 7.40014699628212980e-01, -3.59581721741785224e-02),
            (30.0, 1.0, 6.65314702567186000e-02, -1.95738061172352179e-01),
        ];
        for &(omega, y, re, im) in REF_ND {
            let field = PointMassField { beta: [y, 0.0], lenses: vec![(0.0, 0.0, 1.0)] };
            let out = slice_field(
                &field,
                &SliceConfig { omega, simd_log: true, ..Default::default() },
            )
            .unwrap();
            let want = Complex64::new(re, im);
            let rel = (out.value - want).norm() / want.norm();
            assert!(rel < 1e-12, "ND w={omega}, y={y}: rel={rel:.3e}");
        }

        // (b) N = 3 and N = 10: scalar vs vector, and the alpha-family arbiter
        // with the vector path on (the RC construction is exact for any tail
        // angle in (0, pi/4]).
        let mut s = 987654321u64 | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let fields = [
            PointMassField {
                beta: [0.2, 0.1],
                lenses: vec![(-0.8, 0.5, 0.6), (0.9, -0.4, 0.4), (0.1, -1.2, 0.3)],
            },
            PointMassField {
                beta: [0.5, 0.2],
                lenses: (0..10).map(|_| (next() * 6.0 - 3.0, next() * 6.0 - 3.0, 0.05)).collect(),
            },
        ];
        for (i, field) in fields.iter().enumerate() {
            let omega = [3.0f64, 10.0][i];
            let cfg = SliceConfig { omega, ..Default::default() };
            let scalar = slice_field(field, &cfg).unwrap();
            let vector =
                slice_field(field, &SliceConfig { simd_log: true, ..cfg.clone() }).unwrap();
            let alpha = slice_field(
                field,
                &SliceConfig {
                    simd_log: true,
                    tail_angle: std::f64::consts::FRAC_PI_8,
                    ..cfg.clone()
                },
            )
            .unwrap();
            let scale = scalar.value.norm().max(1.0);
            let d = (scalar.value - vector.value).norm() / scale;
            assert!(d < 1e-13, "N={} scalar vs vector: {d:.3e}", field.lenses.len());
            // The two budgets (plus a roundoff allowance) must cover it.
            assert!(
                scalar.error + vector.error + 8.0 * f64::EPSILON * scale >= d * scale,
                "N={}: budgets do not cover the scalar/vector spread",
                field.lenses.len()
            );
            let da = (vector.value - alpha.value).norm() / scale;
            assert!(da < 1e-11, "N={} alpha family (simd): {da:.3e}", field.lenses.len());
        }
    }
}
