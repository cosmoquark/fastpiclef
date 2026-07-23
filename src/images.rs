//! Geometric-optics images and the stationary-phase amplification for a field
//! of point masses (Path 2, milestones M1 + M2).
//!
//! Fermat potential $\phi(\mathbf x) = \tfrac12|\mathbf x - \beta|^2
//! - \sum_j m_j \ln|\mathbf x - \mathbf z_j|$. In complex notation
//! $z = x_1 + i x_2$, $\zeta = \beta_1 + i\beta_2$, $z_j = a_j + i b_j$, the
//! lens equation $\nabla\phi = 0$ is (Witt 1990)
//! $\zeta = z - \sum_j m_j / (\bar z - \bar z_j)$.
//! It is non-holomorphic; substituting $\bar z = \bar\zeta + \sum_j m_j/(z-z_j)$
//! from the conjugate equation eliminates $\bar z$ and clears to the Witt–Mao
//! polynomial in $z$ of degree $N^2 + 1$. Its roots that satisfy the original
//! equation are the images (spurious roots are filtered out).
//!
//! At high frequency the amplification is the stationary-phase sum over images,
//! $F_{\rm geo}(\omega) = \sum_i |\mu_i|^{1/2}\exp(i\omega\phi_i - i\pi n_i)$,
//! with magnification $\mu_i = 1/\det\nabla^2\phi$ and Morse index
//! $n_i \in \{0, \tfrac12, 1\}$ (minimum / saddle / maximum). Exact as
//! $\omega\to\infty$ away from caustics ($\det\nabla^2\phi\to 0$); the finite-w
//! thimble corrections are milestone M3.

use crate::roots::{poly_add, poly_mul, poly_roots, poly_scale};
use num_complex::Complex64;
use std::f64::consts::PI;

fn poly_sub(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    poly_add(a, &poly_scale(b, Complex64::new(-1.0, 0.0)))
}

/// Real images of the point-mass field: the solutions $\mathbf x$ of the lens
/// equation, returned as `[x1, x2]`. `lenses` is a list of `(a, b, mass)`;
/// `source` is $\beta = (\beta_1, \beta_2)$. Empty `lenses` → the single free
/// image at the source.
pub fn point_mass_field_images(source: [f64; 2], lenses: &[(f64, f64, f64)]) -> Vec<[f64; 2]> {
    let zeta = Complex64::new(source[0], source[1]);
    let zj: Vec<Complex64> = lenses.iter().map(|&(a, b, _)| Complex64::new(a, b)).collect();
    let mj: Vec<Complex64> = lenses.iter().map(|&(_, _, m)| Complex64::new(m, 0.0)).collect();
    let n = lenses.len();
    let one = vec![Complex64::new(1.0, 0.0)];

    // Q(z) = prod_j (z - z_j); Q_without[j] = prod_{k != j} (z - z_k).
    let mut q = one.clone();
    for &z in &zj {
        q = poly_mul(&q, &[-z, Complex64::new(1.0, 0.0)]);
    }
    let q_without: Vec<Vec<Complex64>> = (0..n)
        .map(|j| {
            let mut p = one.clone();
            for (k, &z) in zj.iter().enumerate() {
                if k != j {
                    p = poly_mul(&p, &[-z, Complex64::new(1.0, 0.0)]);
                }
            }
            p
        })
        .collect();

    // P(z) = conj(zeta) Q + sum_j m_j * Q_without[j]  (= bar z * Q, cleared).
    let mut p = poly_scale(&q, zeta.conj());
    for j in 0..n {
        p = poly_add(&p, &poly_scale(&q_without[j], mj[j]));
    }
    // R_i(z) = P - conj(z_i) Q  (= (bar z - bar z_i) * Q).
    let r: Vec<Vec<Complex64>> = (0..n).map(|i| poly_sub(&p, &poly_scale(&q, zj[i].conj()))).collect();

    // prodR = prod_i R_i ; prodR_without[i] = prod_{k != i} R_k.
    let mut prod_r = one.clone();
    for ri in &r {
        prod_r = poly_mul(&prod_r, ri);
    }
    let prod_r_without: Vec<Vec<Complex64>> = (0..n)
        .map(|i| {
            let mut acc = one.clone();
            for (k, rk) in r.iter().enumerate() {
                if k != i {
                    acc = poly_mul(&acc, rk);
                }
            }
            acc
        })
        .collect();

    // poly = (z - zeta) prodR - sum_i m_i Q prodR_without[i]  (degree N^2 + 1).
    let mut poly = poly_mul(&[-zeta, Complex64::new(1.0, 0.0)], &prod_r);
    for i in 0..n {
        poly = poly_sub(&poly, &poly_scale(&poly_mul(&q, &prod_r_without[i]), mj[i]));
    }

    let roots = match poly_roots(&poly) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };

    // Keep roots that satisfy the *original* (non-holomorphic) lens equation
    // z - zeta - sum_j m_j / (conj(z) - conj(z_j)) = 0; dedup.
    let mut images: Vec<[f64; 2]> = Vec::new();
    for z in roots {
        if !z.is_finite() {
            continue;
        }
        let mut resid = z - zeta;
        let mut singular = false;
        for j in 0..n {
            let den = z.conj() - zj[j].conj();
            if den.norm() < 1e-12 {
                singular = true;
                break;
            }
            resid -= mj[j] / den;
        }
        if singular || resid.norm() > 1e-6 * (1.0 + z.norm()) {
            continue;
        }
        let cand = [z.re, z.im];
        if !images
            .iter()
            .any(|p| (p[0] - cand[0]).hypot(p[1] - cand[1]) < 1e-7 * (1.0 + z.norm()))
        {
            images.push(cand);
        }
    }
    images
}

/// Fermat potential and its (real, symmetric) Hessian at a point.
fn fermat_and_hessian(x: [f64; 2], source: [f64; 2], lenses: &[(f64, f64, f64)]) -> (f64, [[f64; 2]; 2]) {
    let mut phi = 0.5 * ((x[0] - source[0]).powi(2) + (x[1] - source[1]).powi(2));
    let mut h = [[1.0, 0.0], [0.0, 1.0]];
    for &(a, b, m) in lenses {
        let d0 = x[0] - a;
        let d1 = x[1] - b;
        let s = d0 * d0 + d1 * d1;
        phi -= 0.5 * m * s.ln();
        // Hessian of -m ln|x-z_j| = -m [ (1/s) I - (2/s^2) d d^T ].
        h[0][0] -= m * (1.0 / s - 2.0 * d0 * d0 / (s * s));
        h[1][1] -= m * (1.0 / s - 2.0 * d1 * d1 / (s * s));
        h[0][1] += m * 2.0 * d0 * d1 / (s * s);
    }
    h[1][0] = h[0][1];
    (phi, h)
}

/// Magnification $\mu = 1/\det H$ and Morse index $n \in \{0, \tfrac12, 1\}$
/// (minimum / saddle / maximum) from the $2\times2$ Hessian.
fn magnification_and_morse(h: [[f64; 2]; 2]) -> (f64, f64) {
    let det = h[0][0] * h[1][1] - h[0][1] * h[1][0];
    let tr = h[0][0] + h[1][1];
    let mu = 1.0 / det;
    let n = if det < 0.0 {
        0.5 // eigenvalues of opposite sign: a saddle
    } else if tr > 0.0 {
        0.0 // both positive: a minimum
    } else {
        1.0 // both negative: a maximum
    };
    (mu, n)
}

/// Stationary-phase (geometric-optics) amplification $F_{\rm geo}(\omega)$ —
/// the sum over images $\sum_i |\mu_i|^{1/2}\exp(i\omega\phi_i - i\pi n_i)$.
/// Exact as $\omega\to\infty$ away from caustics; O(number of images).
pub fn geometric_amplification(source: [f64; 2], lenses: &[(f64, f64, f64)], omega: f64) -> Complex64 {
    let mut f = Complex64::new(0.0, 0.0);
    for x in point_mass_field_images(source, lenses) {
        let (phi, h) = fermat_and_hessian(x, source, lenses);
        let (mu, n) = magnification_and_morse(h);
        f += Complex64::from(mu.abs().sqrt()) * (Complex64::i() * (omega * phi - PI * n)).exp();
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_mass_recovers_point_mass_images() {
        // N=1, mass 1 at origin, source (y, 0): images at ((y +- sqrt(y^2+4))/2, 0).
        let y = 1.0;
        let imgs = point_mass_field_images([y, 0.0], &[(0.0, 0.0, 1.0)]);
        assert_eq!(imgs.len(), 2);
        let rp = 0.5 * (y + (y * y + 4.0f64).sqrt());
        let rm = 0.5 * (y - (y * y + 4.0f64).sqrt());
        let mut xs: Vec<f64> = imgs.iter().map(|p| p[0]).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((xs[0] - rm).abs() < 1e-9, "rm: {} vs {}", xs[0], rm);
        assert!((xs[1] - rp).abs() < 1e-9, "rp: {} vs {}", xs[1], rp);
        assert!(imgs.iter().all(|p| p[1].abs() < 1e-9));
    }

    #[test]
    fn binary_lens_image_count_is_odd_and_valid() {
        // Equal binary, wide separation, source off-axis: 3 or 5 images, all
        // satisfying the lens equation (checked inside the finder).
        let lenses = [(-0.5, 0.0, 0.5), (0.5, 0.0, 0.5)];
        let imgs = point_mass_field_images([0.2, 0.15], &lenses);
        assert!(imgs.len() == 3 || imgs.len() == 5, "got {} images", imgs.len());
        assert_eq!(imgs.len() % 2, 1, "image count must be odd");
    }

    #[test]
    fn field_geometric_envelope_matches_total_magnification() {
        // For a 3-mass field (6 images — even is correct: point masses suppress
        // the central image, so #images = N+1 mod 2), the incoherent high-w
        // limit <|F_geo|^2> -> Sum_i |mu_i| holds. Wide, low-discrepancy band
        // (one image pair has |dphi| ~ 0.015, so narrow bands under-average).
        let lenses = [(0.6, 0.0, 0.4), (-0.5, 0.4, 0.3), (0.1, -0.6, 0.35)];
        let src = [0.2, 0.1];
        let imgs = point_mass_field_images(src, &lenses);
        assert_eq!(imgs.len(), 6);
        let mu_tot: f64 = imgs
            .iter()
            .map(|&x| magnification_and_morse(fermat_and_hessian(x, src, &lenses).1).0.abs())
            .sum();
        // Band must span many "periods" 2*pi/|dphi| ~ 420 of the slow pair.
        let n = 1600;
        let mean_sq: f64 = (0..n)
            .map(|k| {
                let w = 200.0 + 9800.0 * (k as f64 * 0.618_033_988_749_895).fract();
                geometric_amplification(src, &lenses, w).norm_sqr()
            })
            .sum::<f64>()
            / n as f64;
        assert!((mean_sq - mu_tot).abs() < 0.03 * mu_tot, "<|F|^2>={} vs mu_tot={}", mean_sq, mu_tot);
    }

    #[test]
    fn single_mass_geometric_matches_magnification_sum() {
        // |F_geo|^2 -> mu_tot = (y^2+2)/(y sqrt(y^2+4)) as the interference
        // averages out; check the magnitude is O(1) and the pieces are right.
        let y = 1.0;
        let mu_tot = (y * y + 2.0) / (y * (y * y + 4.0f64).sqrt());
        // <|F|^2> over a band of high omega should approach mu_tot.
        let mean_sq: f64 = (0..64)
            .map(|k| {
                let w = 200.0 + 0.37 * k as f64;
                geometric_amplification([y, 0.0], &[(0.0, 0.0, 1.0)], w).norm_sqr()
            })
            .sum::<f64>()
            / 64.0;
        assert!((mean_sq - mu_tot).abs() < 0.05, "<|F|^2>={} vs mu_tot={}", mean_sq, mu_tot);
    }
}
