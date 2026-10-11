//! Data generator for the slice-kernel visualization: sweep the slice position
//! x2 at two frequencies and record the thimble kernel (value, its own error
//! estimate, the swept-branch-point indicator) and the exact RC kernel (value,
//! error estimate, evaluation count), with per-call timings. Prints CSV to
//! stdout; `comparison/plot_slice_sweep.py` grades both against the
//! independent Python referee and renders the figure.
//!
//! (The old mesh "oracle" was retired: it shares `X1SlicePhase` with the
//! thimble, so its agreement/disagreement was never independent evidence.)

use fast_picard_lefshetz_integral_code::phases::PointMassField;
use fast_picard_lefshetz_integral_code::{
    SliceConfig, ThimbleConfig, branch_point_indicator, slice_inner, slice_inner_rc_evals,
};
use num_complex::Complex64;
use std::time::Instant;

fn main() {
    let field = PointMassField {
        beta: [0.2, 0.1],
        lenses: vec![(-0.8, 0.5, 0.6), (0.9, -0.4, 0.4)],
    };
    println!("omega,x2,re_th,im_th,est_th,t_th_ms,re_rc,im_rc,est_rc,t_rc_ms,rc_evals,ind");
    let n = 301;
    let (lo, hi) = (-3.0_f64, 3.0_f64);
    for &omega in &[3.0_f64, 10.0] {
        let th_cfg = ThimbleConfig {
            omega,
            ..ThimbleConfig::default()
        };
        let rc_cfg = SliceConfig {
            omega,
            ..SliceConfig::default()
        };
        for k in 0..n {
            let x2 = lo + (hi - lo) * k as f64 / (n - 1) as f64;
            // Exact pinches are measure-zero for the outer integral and sit on
            // the edge of every method's validity; the plot connects across.
            if field.lenses.iter().any(|&(_, b, _)| (x2 - b).abs() < 1e-9) {
                continue;
            }

            let reps = 3;
            let t = Instant::now();
            let mut th = slice_inner(&field, x2, &th_cfg);
            for _ in 1..reps {
                th = slice_inner(&field, x2, &th_cfg);
            }
            let t_th = t.elapsed().as_secs_f64() * 1e3 / reps as f64;

            let reps = 10;
            let t = Instant::now();
            let mut rc = slice_inner_rc_evals(&field, Complex64::from(x2), &rc_cfg);
            for _ in 1..reps {
                rc = slice_inner_rc_evals(&field, Complex64::from(x2), &rc_cfg);
            }
            let t_rc = t.elapsed().as_secs_f64() * 1e3 / reps as f64;

            let ind = branch_point_indicator(&field, x2, omega);
            let (rcv, rce, ev) = match &rc {
                Ok((o, e)) => (o.value, o.error, *e),
                Err(_) => (Complex64::new(f64::NAN, f64::NAN), f64::NAN, 0),
            };
            match &th {
                Ok(o) => println!(
                    "{omega},{x2:.5},{:.17e},{:.17e},{:.3e},{t_th:.4},{:.17e},{:.17e},{:.3e},{t_rc:.4},{ev},{ind:.3e}",
                    o.value.re, o.value.im, o.error, rcv.re, rcv.im, rce,
                ),
                Err(_) => println!(
                    "{omega},{x2:.5},nan,nan,nan,{t_th:.4},{:.17e},{:.17e},{:.3e},{t_rc:.4},{ev},{ind:.3e}",
                    rcv.re, rcv.im, rce,
                ),
            }
        }
    }
}
