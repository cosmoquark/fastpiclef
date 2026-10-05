//! Quick check: on a FAR (un-swept) slice, is the ~9e-3 vs the damped brute the
//! flow's error or the reference's? Compare flow AND thimble (exact there) to brute.
use fast_picard_lefshetz_integral_code::phases::PointMassField;
use fast_picard_lefshetz_integral_code::{
    FlowContourConfig, Phase, ThimbleConfig, X1SlicePhase, flow_contour_integral, slice_inner,
};
use nalgebra::SVector;
use num_complex::Complex64;
fn damped(s: &X1SlicePhase, w: f64, eps: f64, r: f64, n: usize) -> Complex64 {
    let dx = 2.0*r/n as f64; let mut acc = Complex64::new(0.0,0.0);
    for k in 0..=n { let x=-r+dx*k as f64;
        let phi=s.value(&SVector::<Complex64,1>::new(Complex64::from(x)));
        let wt = if k==0||k==n {1.0} else if k%2==1 {4.0} else {2.0};
        acc += wt*(Complex64::i()*w*phi).exp()*(-eps*x*x).exp(); }
    acc*dx/3.0
}
fn main() {
    let field = PointMassField { beta:[0.2,0.1], lenses: vec![(-0.8,0.5,0.6),(0.9,-0.4,0.4)] };
    let (x2,w)=(2.5,3.0);
    let s = X1SlicePhase{field:&field,x2:Complex64::from(x2)};
    let th = slice_inner(&field,x2,&ThimbleConfig{omega:w,..Default::default()}).unwrap().value;
    let (fl,_) = flow_contour_integral(&s,&FlowContourConfig{omega:w,..Default::default()});
    println!("FAR slice x2={x2}, omega={w}:");
    println!("  thimble = {th:.8}");
    println!("  flow    = {fl:.8}   flow-vs-thimble = {:.2e}", (fl-th).norm()/th.norm());
    for &(eps,r,n) in &[(0.004,22.0,2_000_000usize),(0.002,30.0,4_000_000),(0.001,40.0,8_000_000),(0.0005,50.0,12_000_000)] {
        let d = damped(&s,w,eps,r,n);
        println!("  brute eps={eps} R={r}: {d:.8}   vs thimble = {:.2e}", (d-th).norm()/th.norm());
    }
}
