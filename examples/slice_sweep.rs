//! Data generator for the M3A/A0 visualization: sweep the slice position x2 and
//! compare the inner thimble G(x2) against the 1D mesh oracle, with timings.
//! Prints CSV to stdout.

use fast_picard_lefshetz_integral_code::phases::PointMassField;
use fast_picard_lefshetz_integral_code::{
    Solver, SolverConfig, ThimbleConfig, X1SlicePhase, slice_inner,
};
use num_complex::Complex64;
use std::time::Instant;

fn mesh_inner(field: &PointMassField, x2: f64, omega: f64) -> (Complex64, f64) {
    let slice = X1SlicePhase {
        field,
        x2: Complex64::from(x2),
    };
    let cfg = SolverConfig {
        delta: 0.005,
        omega,
        ..SolverConfig::default()
    };
    let t = Instant::now();
    let out = Solver::<1>::new(&slice, cfg).run().unwrap().value;
    (out, t.elapsed().as_secs_f64() * 1e3)
}

fn main() {
    let omega = 3.0;
    let field = PointMassField {
        beta: [0.2, 0.1],
        lenses: vec![(-0.8, 0.5, 0.6), (0.9, -0.4, 0.4)],
    };
    let cfg = ThimbleConfig {
        omega,
        ..ThimbleConfig::default()
    };

    println!("x2,re_th,im_th,re_mesh,im_mesh,relerr,t_th_ms,t_mesh_ms,ok");
    let n = 301;
    let (lo, hi) = (-3.0_f64, 3.0_f64);
    for k in 0..n {
        let x2 = lo + (hi - lo) * k as f64 / (n - 1) as f64;

        let reps = 8;
        let t = Instant::now();
        let mut th_res = slice_inner(&field, x2, &cfg);
        for _ in 1..reps {
            th_res = slice_inner(&field, x2, &cfg);
        }
        let t_th = t.elapsed().as_secs_f64() * 1e3 / reps as f64;

        let (mesh, t_mesh) = mesh_inner(&field, x2, omega);

        match th_res {
            Ok(th) => {
                let rel = (th.value - mesh).norm() / mesh.norm();
                println!(
                    "{x2:.5},{:.8},{:.8},{:.8},{:.8},{rel:.3e},{t_th:.4},{t_mesh:.4},1",
                    th.value.re, th.value.im, mesh.re, mesh.im,
                );
            }
            Err(_) => {
                println!(
                    "{x2:.5},nan,nan,{:.8},{:.8},nan,{t_th:.4},{t_mesh:.4},0",
                    mesh.re, mesh.im,
                );
            }
        }
    }
}
