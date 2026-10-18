//! Batched RC field integrals: one shared contour serves a whole tensor grid
//! of sources and/or frequencies — the P0-measured win
//! (`comparison/p0_experiments.py`, E1/E2; journal entry of 2026-10-11).
//!
//! The batch is exact by the same Cauchy construction as [`crate::slice::slice_field`];
//! the vector-valued adaptive integrator ([`crate::adaptive::integrate_vec`])
//! accepts a panel only when *every* component has converged, and the source
//! and frequency dependence is factored out of the expensive part:
//!
//! ```text
//! phi(z0, x2; beta) = phi_0(z0, x2) - beta0*z0 - beta1*x2 + (beta0^2 + beta1^2)/2,
//! ```
//!
//! where `phi_0` is the model's phase at zero source (the source term is
//! linear in beta by construction — pinned by a unit test on `PointMassField`).
//! Per quadrature node the shared work (`phi_0`, the lens-potential logs) is
//! done once and each grid entry costs a few complex multiplications; on
//! uniform source grids the twiddle series use a re-anchored recurrence
//! (`REANCHOR` steps between exact anchors, detection at `UNIFORM_TOL`), and
//! the accuracy of the recurrence path against the direct one is asserted in
//! the tests.
//!
//! Bands and chunks: sorted frequencies are grouped greedily while
//! `omega_max/omega_min <= cfg.band_ratio` (each band gets its own contour,
//! sized for its hardest frequency), and groups are further limited to
//! `MAX_GROUP_COMPONENTS` vector entries so memory stays bounded (large maps
//! are chunked automatically).

use crate::adaptive::{AdaptiveVecConfig, integrate_vec};
use crate::error::{Error, Result};
use crate::phases::PointMassField;
use crate::slice::{SliceConfig, SliceEval, SliceModel, check_interrupted};
use num_complex::Complex64;
use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

thread_local! {
    static SERIES_SCRATCH: RefCell<Vec<Complex64>> = const { RefCell::new(Vec::new()) };
}

#[inline]
fn with_series_scratch<R>(len: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
    SERIES_SCRATCH.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < len {
            buf.resize(len, Complex64::new(0.0, 0.0));
        }
        f(&mut buf[..len])
    })
}

/// A batch: frequencies and a tensor source grid `beta0s × beta1s`.
#[derive(Debug, Clone)]
pub struct FieldBatch {
    pub omegas: Vec<f64>,
    pub beta0s: Vec<f64>,
    pub beta1s: Vec<f64>,
}

/// Output of [`slice_field_batch`], row-major
/// `[(i_omega * beta0s.len() + i_beta0) * beta1s.len() + i_beta1]` with the
/// caller's original orders.  Raw integrals (normalisation stays in the
/// Python layer, as everywhere).
#[derive(Debug, Clone)]
pub struct FieldBatchOutput {
    pub values: Vec<Complex64>,
    pub errors: Vec<f64>,
    /// Inner solves performed, counted like [`crate::slice::SliceFieldOutput::inner_solves`]
    /// (one per vector inner evaluation).
    pub inner_solves: usize,
    /// Number of (frequency band × grid chunk) groups actually run.
    pub groups: usize,
}

/// Memory cap per vector group (complex entries).  Larger batches are chunked
/// automatically along the beta grids (and the omega bands, in the 1D sweep).
/// Unit tests can lower it through [`MAX_GROUP_COMPONENTS`] to exercise the
/// chunking path cheaply.
pub(crate) const DEFAULT_MAX_GROUP_COMPONENTS: usize = 1 << 16;
#[cfg(test)]
pub(crate) static MAX_GROUP_COMPONENTS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(DEFAULT_MAX_GROUP_COMPONENTS);

pub(crate) fn max_group_components() -> usize {
    #[cfg(test)]
    {
        MAX_GROUP_COMPONENTS.load(Ordering::Relaxed)
    }
    #[cfg(not(test))]
    {
        DEFAULT_MAX_GROUP_COMPONENTS
    }
}
/// Twiddle recurrence: exact re-anchor every this many steps.
const REANCHOR: usize = 16;
/// Uniform-grid detection: every value within `UNIFORM_TOL * max(1, scale)` of
/// the straight line through the endpoints — a couple of ulps, so standard
/// linspace grids qualify and anything looser falls back to exact sincos.
const UNIFORM_TOL: f64 = 5e-16;

/// Detect an (almost exactly) uniform grid; returns the step.
pub(crate) fn uniform_step(vals: &[f64]) -> Option<f64> {
    let n = vals.len();
    if n < 8 {
        return None;
    }
    let step = (vals[n - 1] - vals[0]) / (n as f64 - 1.0);
    if step == 0.0 || !step.is_finite() {
        return None;
    }
    let scale = vals.iter().fold(0.0f64, |m, &v| m.max(v.abs())).max(1.0);
    let tol = UNIFORM_TOL * scale;
    for (i, &v) in vals.iter().enumerate() {
        if (v - (vals[0] + step * i as f64)).abs() > tol {
            return None;
        }
    }
    Some(step)
}

/// Fill `dst[k] = exp(i * vals[k] * phi)`: exact sincos, or a complex
/// recurrence when `uniform` is the detected grid step (re-anchored every
/// `REANCHOR` steps, so the drift stays at rounding level).
pub(crate) fn fill_series(dst: &mut [Complex64], vals: &[f64], phi: Complex64, uniform: Option<f64>) {
    debug_assert_eq!(dst.len(), vals.len());
    match uniform {
        None => {
            for (d, &v) in dst.iter_mut().zip(vals) {
                *d = (Complex64::i() * (v * phi)).exp();
            }
        }
        Some(step) => {
            let iw = Complex64::i() * phi;
            let rate = (iw * step).exp();
            let mut k = 0;
            while k < dst.len() {
                let end = (k + REANCHOR).min(dst.len());
                let mut acc = (iw * vals[k]).exp();
                for d in dst[k..end].iter_mut() {
                    *d = acc;
                    acc *= rate;
                }
                k = end;
            }
        }
    }
}

/// The batched exact field integral for a point-mass field.
///
/// `cfg.omega` is ignored (the batch carries its own frequencies); margins,
/// tolerances, the tail angle and `band_ratio` are honoured.
pub fn slice_field_batch(
    field: &PointMassField,
    batch: &FieldBatch,
    cfg: &SliceConfig,
) -> Result<FieldBatchOutput> {
    slice_field_batch_g(field, batch, cfg)
}

pub(crate) fn slice_field_batch_g<M: SliceModel>(
    model: &M,
    batch: &FieldBatch,
    cfg: &SliceConfig,
) -> Result<FieldBatchOutput> {
    // ---- validation (mirrors `slice::slice_field_g`) ----
    if batch.omegas.is_empty() {
        return Err(Error::InvalidInput("empty frequency grid"));
    }
    if batch.beta0s.is_empty() || batch.beta1s.is_empty() {
        return Err(Error::InvalidInput("empty source grid"));
    }
    for &w in &batch.omegas {
        if !(w > 0.0) || !w.is_finite() {
            return Err(Error::InvalidInput("omega must be > 0 and finite"));
        }
    }
    for &b in batch.beta0s.iter().chain(batch.beta1s.iter()) {
        if !b.is_finite() {
            return Err(Error::InvalidInput("source positions must be finite"));
        }
    }
    if !(cfg.margin > 0.0) || !cfg.margin.is_finite()
        || !(cfg.margin_outer > 0.0) || !cfg.margin_outer.is_finite()
    {
        return Err(Error::InvalidInput("margins must be > 0 and finite"));
    }
    if !(cfg.tail_angle > 0.0) || cfg.tail_angle > std::f64::consts::FRAC_PI_4 {
        return Err(Error::InvalidInput("tail angle must be in (0, pi/4]"));
    }
    if !(cfg.band_ratio > 1.0) || !cfg.band_ratio.is_finite() {
        return Err(Error::InvalidInput("band_ratio must be > 1 and finite"));
    }
    if cfg.subtract_free {
        return Err(Error::InvalidInput(
            "subtract_free is not supported by the batched entry points (single-field solve only)",
        ));
    }
    model.validate_model()?;

    let m0 = batch.beta0s.len();
    let m1 = batch.beta1s.len();
    let eff = model.pruned();
    let zero = eff.with_beta([0.0, 0.0]);
    let a_vals = eff.abscissae();
    let b_vals = eff.outer_abscissae();
    let mass_sum = eff.potential_scale();

    // ---- greedy frequency bands over sorted omega ----
    let mut order: Vec<usize> = (0..batch.omegas.len()).collect();
    order.sort_by(|&i, &j| batch.omegas[i].partial_cmp(&batch.omegas[j]).unwrap());
    let mut bands: Vec<Vec<usize>> = Vec::new();
    for &i in &order {
        match bands.last_mut() {
            Some(band) if batch.omegas[i] <= batch.omegas[band[0]] * cfg.band_ratio => band.push(i),
            _ => bands.push(vec![i]),
        }
    }

    let mut values = vec![Complex64::new(0.0, 0.0); batch.omegas.len() * m0 * m1];
    let mut errors = vec![0.0f64; batch.omegas.len() * m0 * m1];
    let mut groups = 0usize;
    let solves = AtomicUsize::new(0);

    for band in &bands {
        check_interrupted(cfg)?;
        let nw = band.len();
        let omegas: Vec<f64> = band.iter().map(|&i| batch.omegas[i]).collect();
        let omega_uniform = uniform_step(&omegas);
        // vector chunks along the beta grids (and the band, for huge bands)
        let cap = max_group_components();
        let cw = nw.min(cap);
        let rem = (cap / cw).max(1);
        let c1 = m1.min(rem);
        let c0 = m0.min((rem / c1).max(1));
        for w0 in (0..nw).step_by(cw) {
            let w1 = (w0 + cw).min(nw);
            let gomegas = &omegas[w0..w1];
            for i0s in (0..m0).step_by(c0) {
                let i0e = (i0s + c0).min(m0);
                let gb0 = &batch.beta0s[i0s..i0e];
                for i1s in (0..m1).step_by(c1) {
                    let i1e = (i1s + c1).min(m1);
                    let gb1 = &batch.beta1s[i1s..i1e];
                    check_interrupted(cfg)?;
                    let (vals, errs) = run_group(
                        &zero,
                        cfg,
                        gomegas,
                        omega_uniform,
                        gb0,
                        &a_vals,
                        gb1,
                        &b_vals,
                        mass_sum,
                        &solves,
                    )?;
                    // scatter back to the caller's layout
                    for (jj, &kj) in band[w0..w1].iter().enumerate() {
                        for ii in i0s..i0e {
                            for (kk, gi1) in (i1s..i1e).enumerate() {
                                let g = (jj * gb0.len() + (ii - i0s)) * gb1.len() + kk;
                                let t = (kj * m0 + ii) * m1 + gi1;
                                values[t] = vals[g];
                                errors[t] = errs[g];
                            }
                        }
                    }
                    groups += 1;
                }
            }
        }
    }
    Ok(FieldBatchOutput {
        values,
        errors,
        inner_solves: solves.into_inner(),
        groups,
    })
}

/// One band × grid chunk: the vectorized outer RC integral on a shared
/// contour.  Returns values/errors in the group's flat `(nw, m0, m1)` layout,
/// or [`Error::Interrupted`] if the config's probe fired between outer legs
/// (only ever checked on the calling thread, never inside a rayon task).
#[allow(clippy::too_many_arguments)]
fn run_group<M: SliceModel>(
    zero: &M,
    cfg: &SliceConfig,
    omegas: &[f64],
    omega_uniform: Option<f64>,
    beta0s: &[f64],
    a_vals: &[f64],
    beta1s: &[f64],
    b_vals: &[f64],
    mass_sum: f64,
    solves: &AtomicUsize,
) -> Result<(Vec<Complex64>, Vec<f64>)> {
    let nw = omegas.len();
    let m0 = beta0s.len();
    let m1 = beta1s.len();
    let m_in = nw * m0;
    let m_out = m_in * m1;
    let wmin = omegas[0]; // sorted ascending by the caller
    let beta0_uniform = uniform_step(beta0s);
    let beta1_uniform = uniform_step(beta1s);

    // Precomputed constant factors e^{i omega_j beta^2 / 2} (shared by all
    // nodes): one sincos per (omega, grid point) for the whole group.
    let mut c0 = vec![Complex64::new(0.0, 0.0); m_in];
    for (j, &w) in omegas.iter().enumerate() {
        for (i, &b0) in beta0s.iter().enumerate() {
            c0[j * m0 + i] = (Complex64::i() * (w * b0 * b0 * 0.5)).exp();
        }
    }
    let mut c1 = vec![Complex64::new(0.0, 0.0); nw * m1];
    for (j, &w) in omegas.iter().enumerate() {
        for (i, &b1) in beta1s.iter().enumerate() {
            c1[j * m1 + i] = (Complex64::i() * (w * b1 * b1 * 0.5)).exp();
        }
    }

    // ---- inner batched solve at one slice position (base, off) ----
    let inner = |base: f64, off: Complex64| -> (Vec<Complex64>, Vec<f64>) {
        solves.fetch_add(1, Ordering::Relaxed);
        let ev = zero.evaluate(base, off, cfg.simd_log);
        let real_slice = off.im == 0.0;
        let shift = off.im.abs();
        // Inner-tail geometry from the model (lambda signs/magnitudes); the
        // beta magnitudes come from the *batch's* grid (the zero-source copy
        // has beta = 0 by construction).
        let (tail_sign, l_abs, _b_abs) = zero.inner_tail();
        let lam0 = tail_sign * l_abs;
        let (mut amin, mut amax) = (beta0s[0], beta0s[0]);
        let mut c_lo = f64::INFINITY;
        let mut c_hi = f64::NEG_INFINITY;
        for &b in beta0s {
            amin = amin.min(b);
            amax = amax.max(b);
            let c = b / lam0; // Gaussian centre beta0/lambda0
            c_lo = c_lo.min(c);
            c_hi = c_hi.max(c);
        }
        amin = amin.min(c_lo);
        amax = amax.max(c_hi);
        for &a in a_vals {
            amin = amin.min(a);
            amax = amax.max(a);
        }
        let (aa, bb) = (amin - cfg.margin - shift, amax + cfg.margin + shift);
        let mut pts = vec![aa, bb];
        pts.extend_from_slice(a_vals);
        pts.sort_by(|p, q| p.partial_cmp(q).unwrap());
        pts.dedup();

        // Conservative shared phase bound for the roundoff floors (each
        // component scales by its own omega).
        let zmax = aa.abs().max(bb.abs());
        let b0max = beta0s.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        let x2abs = base.abs() + off.norm();
        let qbound = 0.5 * zmax * zmax
            + b0max * zmax
            + 0.5 * b0max * b0max
            + 0.5 * x2abs * x2abs
            + 5.0
            + mass_sum;
        let mut floors = vec![0.0f64; m_in];
        for (j, &w) in omegas.iter().enumerate() {
            let f = 4e-16 * (1.0 + w * qbound);
            floors[j * m0..(j + 1) * m0].fill(f);
        }

        let mut total = vec![Complex64::new(0.0, 0.0); m_in];
        let mut terr = vec![0.0f64; m_in];

        let eval_core = |t: f64,
                             x0: f64,
                             sgn: f64,
                             vals: &mut [Complex64],
                             errs: &mut [f64]| {
            let phi0 = if real_slice {
                Complex64::from(ev.phi_core_real(t, x0, sgn))
            } else {
                ev.phi_core_complex(t, x0, sgn)
            };
            let z0 = Complex64::from(x0 + sgn * t);
            // Two equivalent factorizations; pick the cheaper by size (the
            // costs per node are ~ nw*(2 + m0) vs m0*(2 + nw) comparisons):
            // whichever of the two grids is larger drives the series.
            if beta0_uniform.is_some() && m0 >= nw {
                // per omega: one beta0 series (2 exps + m0 muls), base e^{i w A}
                with_series_scratch(nw, |sj| {
                    fill_series(sj, omegas, phi0, omega_uniform);
                    for (j, &w) in omegas.iter().enumerate() {
                        let row = &mut vals[j * m0..(j + 1) * m0];
                        fill_series(row, beta0s, -z0 * w, beta0_uniform);
                        let s = sj[j];
                        for i in 0..m0 {
                            row[i] = s * row[i] * c0[j * m0 + i];
                            errs[j * m0 + i] = 0.0;
                        }
                    }
                });
            } else {
                // per beta0: one series over the omegas of the *merged* phase
                // A - beta0 z0 (one exp per (node, omega) instead of two)
                with_series_scratch(nw, |t_w| {
                    for (i, &b0) in beta0s.iter().enumerate() {
                        fill_series(
                            t_w,
                            omegas,
                            phi0 - z0 * Complex64::from(b0),
                            omega_uniform,
                        );
                        for j in 0..nw {
                            vals[j * m0 + i] = t_w[j] * c0[j * m0 + i];
                            errs[j * m0 + i] = 0.0;
                        }
                    }
                });
            }
        };

        let cfg_core = AdaptiveVecConfig {
            tol: cfg.tol_inner,
            floors: floors.clone(),
            min_len: 1e-15,
            max_panels: 200_000,
            gl10_probe: cfg.gl10_probe,
        };
        for win in pts.windows(2) {
            let (lo, hi) = (win[0], win[1]);
            let half = 0.5 * (hi - lo);
            for (x0, sgn) in [(lo, 1.0), (hi, -1.0)] {
                let f = |t: f64, vals: &mut [Complex64], errs: &mut [f64]| {
                    eval_core(t, x0, sgn, vals, errs)
                };
                let res = integrate_vec(&f, 0.0, half, &cfg_core, false);
                for k in 0..m_in {
                    total[k] += res.values[k];
                    terr[k] += res.errors[k];
                }
            }
        }

        // Rotated inner tails: direction sign(lambda0)*angle, length scaled by
        // |lambda0| and including the beta-linear term (both enter as +e4).
        let e4 = Complex64::from_polar(1.0, tail_sign * cfg.tail_angle);
        let tail_cfg = AdaptiveVecConfig {
            tol: cfg.tol_inner,
            floors,
            min_len: 1e-14,
            max_panels: 200_000,
            gl10_probe: cfg.gl10_probe,
        };
        let mut t_end = (2.0 * 45.0 / (wmin * l_abs)).sqrt()
            + b0max / (l_abs * cfg.tail_angle.cos())
            + 2.0;
        for (end_z, sgn) in [(bb, 1.0), (aa, -1.0)] {
            let eval_tail = |t: f64, vals: &mut [Complex64], errs: &mut [f64]| {
                let z = end_z + sgn * e4 * t;
                let phi0 = ev.phi_tail(z);
                if beta0_uniform.is_some() && m0 >= nw {
                    with_series_scratch(nw, |sj| {
                        fill_series(sj, omegas, phi0, omega_uniform);
                        for (j, &w) in omegas.iter().enumerate() {
                            let row = &mut vals[j * m0..(j + 1) * m0];
                            fill_series(row, beta0s, -z * w, beta0_uniform);
                            let s = sj[j] * e4;
                            for i in 0..m0 {
                                row[i] = s * row[i] * c0[j * m0 + i];
                                errs[j * m0 + i] = 0.0;
                            }
                        }
                    });
                } else {
                    with_series_scratch(nw, |t_w| {
                        for (i, &b0) in beta0s.iter().enumerate() {
                            fill_series(
                                t_w,
                                omegas,
                                phi0 - z * Complex64::from(b0),
                                omega_uniform,
                            );
                            for j in 0..nw {
                                vals[j * m0 + i] = e4 * t_w[j] * c0[j * m0 + i];
                                errs[j * m0 + i] = 0.0;
                            }
                        }
                    });
                }
            };
            // decay probe: componentwise, against the sampled magnitudes
            let mut probed = false;
            for _ in 0..5 {
                let mut peak = vec![0.0f64; m_in];
                let mut vals = vec![Complex64::new(0.0, 0.0); m_in];
                let mut errs = vec![0.0f64; m_in];
                for s in 0..16 {
                    let tt = t_end * (s as f64) / 16.0;
                    eval_tail(tt, &mut vals, &mut errs);
                    for k in 0..m_in {
                        peak[k] = peak[k].max(vals[k].norm());
                    }
                }
                eval_tail(t_end, &mut vals, &mut errs);
                probed = (0..m_in).all(|k| vals[k].norm() <= 1e-17 * peak[k].max(1e-30));
                if probed {
                    break;
                }
                t_end *= 2.0;
            }
            let res = integrate_vec(&eval_tail, 0.0, t_end, &tail_cfg, false);
            let mut end_vals = vec![Complex64::new(0.0, 0.0); m_in];
            let mut end_errs = vec![0.0f64; m_in];
            eval_tail(t_end, &mut end_vals, &mut end_errs);
            for k in 0..m_in {
                total[k] += res.values[k];
                terr[k] += res.errors[k];
                let end = end_vals[k].norm();
                terr[k] += if probed { 4.0 * end } else { t_end * end };
            }
        }
        (total, terr)
    };

    // ---- outer contour (shared) ----
    let (tail_sign2, l2_abs, _b1_abs) = zero.outer_tail();
    let lam2 = tail_sign2 * l2_abs;
    let (mut bmin, mut bmax) = (beta1s[0], beta1s[0]);
    let mut c2_lo = f64::INFINITY;
    let mut c2_hi = f64::NEG_INFINITY;
    for &b in beta1s {
        bmin = bmin.min(b);
        bmax = bmax.max(b);
        let c = b / lam2; // Gaussian centre beta1/lambda2
        c2_lo = c2_lo.min(c);
        c2_hi = c2_hi.max(c);
    }
    bmin = bmin.min(c2_lo);
    bmax = bmax.max(c2_hi);
    for &b in b_vals {
        bmin = bmin.min(b);
        bmax = bmax.max(b);
    }
    let (a2, b2) = (bmin - cfg.margin_outer, bmax + cfg.margin_outer);
    let mut pts = vec![a2, b2];
    pts.extend_from_slice(b_vals);
    pts.sort_by(|p, q| p.partial_cmp(q).unwrap());
    pts.dedup();
    let b1max = beta1s.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    let mut span2 = 0.0f64;
    for &b in beta1s {
        span2 = span2.max((a2 - b).abs()).max((b2 - b).abs());
    }
    let mut floors = vec![0.0f64; m_out];
    for (j, &w) in omegas.iter().enumerate() {
        let f = 4e-16 * (1.0 + w * (0.5 * span2 * span2 + 10.0 + mass_sum));
        floors[j * m0 * m1..(j + 1) * m0 * m1].fill(f);
    }

    let mut total = vec![Complex64::new(0.0, 0.0); m_out];
    let mut terr = vec![0.0f64; m_out];

    // outer scatter: given the inner vector and x2, apply the beta1 twiddles
    // outer twiddles: e^{i w (beta1^2/2 - beta1 x2)}, again with the cheaper of
    // the two series factorizations.
    let use_beta1_series = beta1_uniform.is_some() && m1 >= nw;
    let scatter = |x2: Complex64,
                   jac: f64,
                   inner_vals: &[Complex64],
                   inner_errs: &[f64],
                   vals: &mut [Complex64],
                   errs: &mut [f64]| {
        if use_beta1_series {
            with_series_scratch(m1, |t1| {
                for (j, &w) in omegas.iter().enumerate() {
                    fill_series(t1, beta1s, -x2 * w, beta1_uniform);
                    for i0 in 0..m0 {
                        let base = inner_vals[j * m0 + i0] * jac;
                        let base_e = inner_errs[j * m0 + i0] * jac.abs();
                        for (i1, &t) in t1.iter().enumerate() {
                            let k = (j * m0 + i0) * m1 + i1;
                            vals[k] = base * t * c1[j * m1 + i1];
                            errs[k] = base_e;
                        }
                    }
                }
            });
        } else {
            // per (i0, i1): one series over the omegas of the merged phase
            // (the beta1^2/2 constant folded in, so no separate c1 factor)
            with_series_scratch(nw, |t_w| {
                for i0 in 0..m0 {
                    for (i1, &b1) in beta1s.iter().enumerate() {
                        let phi =
                            Complex64::from(0.5 * b1 * b1) - Complex64::from(b1) * x2;
                        fill_series(t_w, omegas, phi, omega_uniform);
                        for j in 0..nw {
                            let k = (j * m0 + i0) * m1 + i1;
                            vals[k] = inner_vals[j * m0 + i0] * jac * t_w[j];
                            errs[k] = inner_errs[j * m0 + i0] * jac.abs();
                        }
                    }
                }
            });
        }
    };

    // outer core: anchored halves x2 = x0 + sgn*rho^2 (dx2 = 2 rho d rho)
    let outer_cfg = AdaptiveVecConfig {
        tol: cfg.tol_outer,
        floors,
        min_len: 1e-12,
        max_panels: 200_000,
        gl10_probe: cfg.gl10_probe,
    };
    for win in pts.windows(2) {
        let (lo, hi) = (win[0], win[1]);
        let rmax = (0.5 * (hi - lo)).sqrt();
        for (x0, sgn) in [(lo, 1.0), (hi, -1.0)] {
            check_interrupted(cfg)?;
            let f = |rho: f64, vals: &mut [Complex64], errs: &mut [f64]| {
                let off = Complex64::new(sgn * rho * rho, 0.0);
                let (iv, ie) = inner(x0, off);
                scatter(Complex64::from(x0 + sgn * rho * rho), 2.0 * rho, &iv, &ie, vals, errs);
            };
            let res = integrate_vec(&f, 0.0, rmax, &outer_cfg, cfg.parallel);
            for k in 0..m_out {
                total[k] += res.values[k];
                terr[k] += res.errors[k];
            }
        }
    }

    // outer tails: x2 = end + sgn*e4*s (complex slice positions), with the
    // model's direction sign(lambda2) and |lambda2|-scaled length.
    let e4 = Complex64::from_polar(1.0, tail_sign2 * cfg.tail_angle);
    let mut s_end = (2.0 * 45.0 / (wmin * l2_abs)).sqrt()
        + b1max / (l2_abs * cfg.tail_angle.cos())
        + 2.0;
    for (end_x2, sgn) in [(b2, 1.0), (a2, -1.0)] {
        check_interrupted(cfg)?;
        let eval_outer = |s: f64, vals: &mut [Complex64], errs: &mut [f64]| {
            let off = e4 * Complex64::from(sgn * s);
            let (iv, ie) = inner(end_x2, off);
            // both outer tails contribute +e4 * G (Jacobian and orientation)
            let iv_e4: Vec<Complex64> = iv.iter().map(|v| *v * e4).collect();
            scatter(end_x2 + sgn * e4 * s, 1.0, &iv_e4, &ie, vals, errs);
        };
        // decay probe (componentwise)
        let mut probed = false;
        for _ in 0..5 {
            let mut peak = vec![0.0f64; m_out];
            let mut vals = vec![Complex64::new(0.0, 0.0); m_out];
            let mut errs = vec![0.0f64; m_out];
            for si in 0..8 {
                let ss = s_end * (si as f64) / 8.0;
                eval_outer(ss, &mut vals, &mut errs);
                for k in 0..m_out {
                    peak[k] = peak[k].max(vals[k].norm());
                }
            }
            eval_outer(s_end, &mut vals, &mut errs);
            probed = (0..m_out).all(|k| vals[k].norm() <= 1e-17 * peak[k].max(1e-30));
            if probed {
                break;
            }
            s_end *= 2.0;
        }
        let res = integrate_vec(&eval_outer, 0.0, s_end, &outer_cfg, cfg.parallel);
        let mut end_vals = vec![Complex64::new(0.0, 0.0); m_out];
        let mut end_errs = vec![0.0f64; m_out];
        eval_outer(s_end, &mut end_vals, &mut end_errs);
        for k in 0..m_out {
            total[k] += res.values[k];
            terr[k] += res.errors[k];
            let end = end_vals[k].norm();
            terr[k] += if probed { 4.0 * end } else { s_end * end };
        }
    }

    Ok((total, terr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slice::slice_field;

    /// The series helper: the re-anchored recurrence must reproduce exact
    /// sincos on a uniform grid at rounding level.
    #[test]
    fn series_recurrence_matches_direct() {
        let vals: Vec<f64> = (0..40).map(|i| -1.0 + 0.05 * i as f64).collect();
        for phi in [Complex64::new(-0.37, 0.0), Complex64::new(0.0, -2.1), Complex64::new(1.3, -0.4)] {
            let step = uniform_step(&vals);
            assert!(step.is_some());
            let mut a = vec![Complex64::new(0.0, 0.0); vals.len()];
            let mut b = vec![Complex64::new(0.0, 0.0); vals.len()];
            fill_series(&mut a, &vals, phi, step);
            fill_series(&mut b, &vals, phi, None);
            for i in 0..vals.len() {
                assert!(
                    (a[i] - b[i]).norm() < 1e-14,
                    "phi={phi} i={i}: recurrence {} vs direct {}",
                    a[i],
                    b[i]
                );
            }
        }
    }

    /// The batching factorization: phi(z0, x2; beta) =
    /// phi_0(z0, x2; 0) - beta0 z0 - beta1 x2 + (beta0^2 + beta1^2)/2.
    #[test]
    fn source_factorization_identity() {
        let field = PointMassField {
            beta: [0.37, -0.22],
            lenses: vec![(-0.8, 0.5, 0.6), (0.9, -0.4, 0.4)],
        };
        let zero = field.with_beta([0.0, 0.0]);
        let b0 = field.beta[0];
        let b1 = field.beta[1];
        let kc = 0.5 * (b0 * b0 + b1 * b1);
        for (base, off) in [
            (0.3f64, Complex64::new(0.0, 0.0)),
            (1.1, Complex64::new(0.49, 0.0)),
            (0.9, Complex64::new(0.5, 0.5)), // complex slice (an outer tail)
        ] {
            let ev = field.evaluate(base, off, false);
            let zv = zero.evaluate(base, off, false);
            let x2 = Complex64::from(base) + off;
            for (r, x0, sgn) in [(0.11f64, base, 1.0f64), (0.23, base, -1.0), (0.07, base + 0.4, 1.0)] {
                let z0 = Complex64::from(x0 + sgn * r);
                let lhs = if off.im == 0.0 {
                    Complex64::from(ev.phi_core_real(r, x0, sgn))
                } else {
                    ev.phi_core_complex(r, x0, sgn)
                };
                let rhs = if off.im == 0.0 {
                    Complex64::from(zv.phi_core_real(r, x0, sgn))
                } else {
                    zv.phi_core_complex(r, x0, sgn)
                } - (Complex64::from(b0) * z0 + Complex64::from(b1) * x2)
                    + Complex64::from(kc);
                assert!(
                    (lhs - rhs).norm() < 1e-13,
                    "factorization mismatch: {lhs} vs {rhs}"
                );
            }
        }
    }

    /// A lowered memory cap must engage the chunking path and stay correct.
    #[test]
    fn tiny_cap_chunks_and_stays_correct() {
        let field = PointMassField {
            beta: [0.2, 0.1],
            lenses: vec![(-0.8, 0.5, 0.6), (0.9, -0.4, 0.4)],
        };
        MAX_GROUP_COMPONENTS.store(8, Ordering::Relaxed);
        let batch = FieldBatch {
            omegas: vec![1.0, 3.0],
            beta0s: (0..4).map(|i| -0.6 + 0.4 * i as f64).collect(),
            beta1s: (0..3).map(|i| -0.5 + 0.5 * i as f64).collect(),
        };
        let out = slice_field_batch(
            &field,
            &batch,
            &SliceConfig { omega: 1.0, parallel: false, ..Default::default() },
        )
        .unwrap();
        MAX_GROUP_COMPONENTS.store(DEFAULT_MAX_GROUP_COMPONENTS, Ordering::Relaxed);
        assert!(out.groups > 1, "chunking did not trigger ({} groups)", out.groups);
        let mut scale = 0.0f64;
        for v in &out.values {
            scale = scale.max(v.norm());
        }
        let mut worst = 0.0f64;
        for (j, &w) in batch.omegas.iter().enumerate() {
            for (i0, &b0) in batch.beta0s.iter().enumerate() {
                for (i1, &b1) in batch.beta1s.iter().enumerate() {
                    let f = PointMassField { beta: [b0, b1], lenses: field.lenses.clone() };
                    let sc = slice_field(&f, &SliceConfig { omega: w, ..Default::default() })
                        .unwrap()
                        .value;
                    let idx = (j * batch.beta0s.len() + i0) * batch.beta1s.len() + i1;
                    worst = worst.max((out.values[idx] - sc).norm() / scale);
                }
            }
        }
        assert!(worst < 1e-11, "chunked batch worst {worst:.3e}");
    }
}
