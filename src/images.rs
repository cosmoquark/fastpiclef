//! Geometric-optics images and the stationary-phase amplification of a field
//! of point masses (Path 2, milestones M1 + M2; the M1 finder was reworked on
//! 2026-10-17 — see below).
//!
//! Fermat potential $\phi(\mathbf x) = \tfrac12|\mathbf x - \mathbf\beta|^2 -
//! \sum_j m_j \ln|\mathbf x - \mathbf z_j|$; the image condition $\nabla\phi =
//! 0$ is $\mathbf x = \mathbf\beta + \sum_j m_j\,(\mathbf x - \mathbf z_j)/
//! |\mathbf x - \mathbf z_j|^2$.  At high frequency the amplification is the
//! stationary-phase sum over images,
//! $F_{\rm geo}(\omega) = \sum_i |\mu_i|^{1/2}\exp(i\omega\phi_i - i\pi n_i)$,
//! with magnification $\mu_i = 1/\det\nabla^2\phi$ and Morse index
//! $n_i \in \{0, \tfrac12, 1\}$ (minimum / saddle / maximum).  Exact as
//! $\omega\to\infty$ away from caustics.
//!
//! # The finder, and why it was reworked
//!
//! M1 originally expanded the lens equation into the Witt–Mao polynomial of
//! degree $N^2+1$ and found all roots by Durand–Kerner.  Measured 2026-10-17
//! on the getting-started 40-mass field, that route is numerically dead in
//! f64 long before the field sizes this crate targets: the product of the $N$
//! degree-$N$ factors overflows f64 at $N \gtrsim 30$ (each factor has
//! coefficients up to $\sim 10^{14}$), and already at $N = 5$ the iteration
//! fails its residual contract on the product polynomial (clustered spurious
//! roots; the Cauchy-radius seeding starts the iterates $\sim 10^{11}$ from
//! every root).  Worse, the failure was folded into an *empty* image list, and
//! `geometric` then returned a zero amplification with a zero error estimate.
//!
//! The finder is now structural, and every failure is loud — [`Result`], all
//! the way to the Python bindings:
//!
//! 1. Multi-start Newton on the real lens equation (analytic gradient and
//!    Hessian).  Seeds: (a) a per-star ladder — the faint companion of star
//!    $j$ sits near $\mathbf z_j$ at distance $\sim m_j/|\mathbf z_j -
//!    \mathbf\beta|$ (weak regime) or at the Einstein scale $\sqrt{m_j}$
//!    (strong regime), so each star gets both geometric radius ladders over 8
//!    directions; (b) a golden-ratio-jittered global grid for the bright
//!    images.
//! 2. Every candidate is residual-verified,
//!    $\|\nabla\phi\|\le 10^{-13}(1+|\mathbf x|)$, and deduplicated.
//! 3. A completeness certificate.  For a generic source the index identity
//!    $\#\{\det H > 0\} - \#\{\det H < 0\} = 1 - N$ holds (excision on the
//!    sphere: the large-circle winding of $\nabla\phi$ is $+1$ and each star
//!    contributes $+1$), and every star must own an image within
//!    $2\sqrt{m_j} + 0.1$.  A violated certificate is an error, never a short
//!    list — a missed companion image changes the index sum and is caught.
//!    A degenerate image ($|\det H|$ below the caustic floor $10^{-8}$: at an
//!    exact caustic the image set is not discrete — e.g. the whole Einstein
//!    ring at $\beta = 0$ — and the stationary-phase sum is undefined) is
//!    reported as [`Error::DegenerateImage`] instead of poisoning the sum.
//!
//! Validated against exact mpmath image sets (80-digit root solves of the
//! degree-10 / degree-26 polynomials) at $N = 3$ and $N = 5$, and against an
//! independent ladder+grid reference for the 40-mass getting-started field:
//! 41 images (one companion per star plus the main image),
//! $\sum_i|\mu_i| = 1.4966407061178489$, $F_{\rm geo}(2) = -1.4586072556 +
//! 0.4592694845\,i$ — all pinned in the tests below.

use crate::{Error, Result};
use num_complex::Complex64;
use std::f64::consts::PI;

/// One verified geometric-optics image of a point-mass field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageData {
    /// Image position `[x1, x2]`.
    pub x: [f64; 2],
    /// Fermat potential at the image.
    pub phi: f64,
    /// Signed magnification `1 / det H`; positive (minimum or maximum) or
    /// negative (saddle).
    pub mu: f64,
    /// Morse index: `0` (minimum), `1/2` (saddle) or `1` (maximum).
    pub morse: f64,
}

/// Gradient of the Fermat potential at a real point.
fn grad_phi(x: [f64; 2], source: [f64; 2], lenses: &[(f64, f64, f64)]) -> [f64; 2] {
    let mut g = [x[0] - source[0], x[1] - source[1]];
    for &(a, b, m) in lenses {
        let d0 = x[0] - a;
        let d1 = x[1] - b;
        let s = d0 * d0 + d1 * d1;
        let f = m / s;
        g[0] -= f * d0;
        g[1] -= f * d1;
    }
    g
}

/// Fermat potential and its (real, symmetric) Hessian at a point.
fn fermat_and_hessian(
    x: [f64; 2],
    source: [f64; 2],
    lenses: &[(f64, f64, f64)],
) -> (f64, [[f64; 2]; 2]) {
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

/// One plain-Newton run on the real lens equation, from one seed.  `None` when
/// the iterate leaves the bound (or hits a star exactly) or the residual
/// criterion is not met within the iteration budget.
fn newton_image(
    mut x: [f64; 2],
    source: [f64; 2],
    lenses: &[(f64, f64, f64)],
    bound: f64,
) -> Option<[f64; 2]> {
    for _ in 0..120 {
        let (_, h) = fermat_and_hessian(x, source, lenses);
        let g = grad_phi(x, source, lenses);
        let det = h[0][0] * h[1][1] - h[0][1] * h[1][0];
        if !det.is_finite() || det == 0.0 {
            return None;
        }
        let dx = [
            (h[1][1] * g[0] - h[0][1] * g[1]) / det,
            (h[0][0] * g[1] - h[1][0] * g[0]) / det,
        ];
        x[0] -= dx[0];
        x[1] -= dx[1];
        if !(x[0].is_finite() && x[1].is_finite()) || x[0].hypot(x[1]) > bound {
            return None;
        }
        let g = grad_phi(x, source, lenses);
        if g[0].hypot(g[1]) <= 1e-13 * (1.0 + x[0].hypot(x[1])) {
            return Some(x);
        }
    }
    None
}

/// The image-search start set: a jittered global grid over `[-span, span]^2`
/// for the bright images, plus a per-star radius/angle ladder around every
/// star for the faint companion images (see the module docs).
fn image_seeds(source: [f64; 2], lenses: &[(f64, f64, f64)], span: f64) -> Vec<[f64; 2]> {
    let mut seeds = Vec::new();
    // (a) Global grid, deterministically golden-ratio jittered so no start
    // sits exactly on a star/singular centre or on an aligned degenerate line.
    const GRID: usize = 20;
    for i in 0..=GRID {
        for j in 0..=GRID {
            let k = i * (GRID + 1) + j;
            let jit = |s: usize| ((s as f64 + 1.0) * 0.618_033_988_749_895).fract() - 0.5;
            seeds.push([
                -span + 2.0 * span * (i as f64 + 0.8 * jit(k)) / GRID as f64,
                -span + 2.0 * span * (j as f64 + 0.8 * jit(k + 7)) / GRID as f64,
            ]);
        }
    }
    // (b) Per-star ladders.  Angle 0 of the sweep points away from the source
    // (the weak-field companion direction); both the weak-field scale m/|z-b|
    // and the Einstein scale sqrt(m) are covered geometrically.
    for &(a, b, m) in lenses {
        let (dx, dy) = (a - source[0], b - source[1]);
        let d = dx.hypot(dy);
        let einstein = m.sqrt();
        let (ex, ey) = if d > 1e-12 {
            (dx / d, dy / d)
        } else {
            (1.0, 0.0)
        };
        let weak = m / d.max(einstein);
        for base in [weak, einstein] {
            for scale in [0.125, 0.25, 0.5, 1.0, 2.0, 4.0] {
                let r = base * scale;
                for k in 0..8 {
                    let (s, c) = (k as f64 * PI / 4.0).sin_cos();
                    seeds.push([a + r * (c * ex - s * ey), b + r * (s * ex + c * ey)]);
                }
            }
        }
    }
    seeds
}

/// All verified images of the point-mass field, with the completeness
/// certificate enforced (see the module docs).  Errors instead of returning a
/// short list.
fn find_verified_images(source: [f64; 2], lenses: &[(f64, f64, f64)]) -> Result<Vec<[f64; 2]>> {
    if lenses.is_empty() {
        return Ok(vec![source]);
    }
    // Every image satisfies |x| <= max-star-|z| + |source| + sqrt(total mass)
    // (the deflection term cannot balance a larger |x|); span adds a margin.
    let a_star = lenses
        .iter()
        .fold(0.0f64, |acc, &(x, y, _)| acc.max(x.abs()).max(y.abs()));
    let m_tot: f64 = lenses.iter().map(|&(_, _, m)| m).sum();
    let span = a_star + source[0].abs().max(source[1].abs()) + m_tot.sqrt() + 1.0;
    let bound = span + 4.0;

    let mut images: Vec<[f64; 2]> = Vec::new();
    for seed in image_seeds(source, lenses, span) {
        if let Some(x) = newton_image(seed, source, lenses, bound) {
            let tol = 1e-8 * (1.0 + x[0].hypot(x[1]));
            if !images
                .iter()
                .any(|p| (p[0] - x[0]).hypot(p[1] - x[1]) < tol)
            {
                images.push(x);
            }
        }
    }
    if images.is_empty() {
        return Err(Error::ImageSearch("no image converged"));
    }

    // Certificate 1: every image must be non-degenerate (`|det H|` above the
    // caustic floor — at an exact caustic the image set is not discrete, e.g.
    // the whole Einstein ring at beta = 0, and the stationary-phase sum is
    // undefined), then the index identity (a missed companion flips it).
    const CAUSTIC_DET: f64 = 1e-8;
    let mut n_pos = 0i64;
    let mut n_neg = 0i64;
    for x in &images {
        let (_, h) = fermat_and_hessian(*x, source, lenses);
        let det = h[0][0] * h[1][1] - h[0][1] * h[1][0];
        if !det.is_finite() || det.abs() < CAUSTIC_DET {
            return Err(Error::DegenerateImage);
        }
        if det > 0.0 {
            n_pos += 1;
        } else {
            n_neg += 1;
        }
    }
    if n_pos - n_neg != 1 - lenses.len() as i64 {
        return Err(Error::ImageSearch(
            "index identity violated (a companion image was likely missed)",
        ));
    }
    // Certificate 2: every star owns an image within its local scale.
    for &(a, b, m) in lenses {
        let r = 2.0 * m.sqrt() + 0.1;
        if !images.iter().any(|x| (x[0] - a).hypot(x[1] - b) <= r) {
            return Err(Error::ImageSearch(
                "a star has no image nearby (a companion image was likely missed)",
            ));
        }
    }
    Ok(images)
}

/// Every verified geometric image of the point-mass field, with the Fermat
/// potential, signed magnification and Morse index of each.  See the module
/// docs for the finder and its completeness certificate; a failed certificate
/// is an error, never a silently short list.
pub fn point_mass_field_image_data(
    source: [f64; 2],
    lenses: &[(f64, f64, f64)],
) -> Result<Vec<ImageData>> {
    Ok(find_verified_images(source, lenses)?
        .iter()
        .map(|&x| {
            let (phi, h) = fermat_and_hessian(x, source, lenses);
            let (mu, morse) = magnification_and_morse(h);
            ImageData { x, phi, mu, morse }
        })
        .collect())
}

/// The verified geometric-optics images (real solutions of the lens equation),
/// each as `[x1, x2]`.  `source` is $\beta = (\beta_1,\beta_2)$; `lenses` is
/// a list of `(a, b, mass)`.  Empty `lenses` gives the single free image at
/// the source.
pub fn point_mass_field_images(
    source: [f64; 2],
    lenses: &[(f64, f64, f64)],
) -> Result<Vec<[f64; 2]>> {
    Ok(point_mass_field_image_data(source, lenses)?
        .into_iter()
        .map(|d| d.x)
        .collect())
}

/// Stationary-phase (geometric-optics) amplification $F_{\rm geo}(\omega)$ —
/// the sum over images $\sum_i |\mu_i|^{1/2}\exp(i\omega\phi_i - i\pi n_i)$.
/// Exact as $\omega\to\infty$ away from caustics; O(number of images).  The
/// image set (and hence the sum) comes from the verified finder: a failed
/// completeness certificate is an error.
pub fn geometric_amplification(
    source: [f64; 2],
    lenses: &[(f64, f64, f64)],
    omega: f64,
) -> Result<Complex64> {
    let mut f = Complex64::new(0.0, 0.0);
    for d in point_mass_field_image_data(source, lenses)? {
        f += Complex64::from(d.mu.abs().sqrt())
            * (Complex64::i() * (omega * d.phi - PI * d.morse)).exp();
    }
    Ok(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The getting-started notebook field stock: `np.random.default_rng(7)`
    /// uniform draws in `[-3, 3]^2`, every mass 0.05 — embedded at 17
    /// significant digits so the Rust tests use the exact same field.
    const NOTEBOOK_FIELD: [(f64, f64, f64); 40] = [
        (0.75057279962800205, 2.3832828058174531, 0.05),
        (1.6541141414711609, -1.6487568600564488, 0.05),
        (-1.1990022905326474, 2.2413206723775714, 0.05),
        (-2.9684081726065514, 1.9273705102965977, 0.05),
        (1.7824165725122771, -0.19239028293767557, 0.05),
        (-1.1818054390841188, -1.3294463273953601, 0.05),
        (-1.4707824740752524, -0.32954216470412057, 0.05),
        (0.027289553747719797, 0.32098411244695502, 0.05),
        (2.973001700606356, 1.7559715152825186, 0.05),
        (0.7330753766469762, 2.9337608860913091, 0.05),
        (-1.7081478105864063, -2.0387277968529327, 0.05),
        (0.6752376256381849, -2.7363479522317, 0.05),
        (-2.7859183273584232, 0.089332921628221484, 0.05),
        (-0.20276384804826542, 2.503006639157114, 0.05),
        (0.77535752694606241, 0.0847058795970832, 0.05),
        (-0.018759387638974445, -1.514910467836015, 0.05),
        (-2.9292358467449651, -1.8455871360881362, 0.05),
        (1.1521927252910356, -1.7963596560780288, 0.05),
        (-0.78278213638675975, -2.9775945476875441, 0.05),
        (1.9802863788104732, -2.0732335136313607, 0.05),
        (-1.3944041726172873, 2.2819929238849719, 0.05),
        (0.058744859210539069, 2.0829014781952164, 0.05),
        (0.83830300165515759, 1.4506256841711425, 0.05),
        (-2.4510263696217258, 0.24686292825893252, 0.05),
        (0.046633417802099508, 2.2280362601572836, 0.05),
        (-0.83241564591505446, 0.58910440324327862, 0.05),
        (-2.6444901459269783, -0.67420919333562779, 0.05),
        (-1.0617819224507601, -2.0988016255772886, 0.05),
        (1.8980286229144543, -0.72332297069812501, 0.05),
        (2.8724873064673293, 0.53995015806366187, 0.05),
        (0.63033752297910794, 0.82797948472999305, 0.05),
        (1.0587014628767299, -2.0952718849897876, 0.05),
        (-0.35811919687087457, -1.56261622902286, 0.05),
        (-0.58501021137610998, -2.4197754364095263, 0.05),
        (2.8069683062929283, -1.7099757758647198, 0.05),
        (1.0305909756677094, -1.1974795111255783, 0.05),
        (2.2444621568970264, 0.97328843003072274, 0.05),
        (-2.2103051051501659, 2.0704459252473173, 0.05),
        (2.669689026869877, 2.4235007291755606, 0.05),
        (0.41831488715566323, -2.1272402774344386, 0.05),
    ];
    const NOTEBOOK_BETA: [f64; 2] = [0.5, 0.2];

    #[test]
    fn single_mass_recovers_point_mass_images() {
        // N=1, mass 1 at origin, source (y, 0): images at ((y +- sqrt(y^2+4))/2, 0).
        let y = 1.0;
        let imgs = point_mass_field_images([y, 0.0], &[(0.0, 0.0, 1.0)]).unwrap();
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
        // satisfying the lens equation (checked inside the finder) and the
        // index certificate.
        let lenses = [(-0.5, 0.0, 0.5), (0.5, 0.0, 0.5)];
        let imgs = point_mass_field_images([0.2, 0.15], &lenses).unwrap();
        assert!(
            imgs.len() == 3 || imgs.len() == 5,
            "got {} images",
            imgs.len()
        );
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
        let imgs = point_mass_field_images(src, &lenses).unwrap();
        assert_eq!(imgs.len(), 6);
        let mu_tot: f64 = imgs
            .iter()
            .map(|&x| {
                magnification_and_morse(fermat_and_hessian(x, src, &lenses).1)
                    .0
                    .abs()
            })
            .sum();
        // Band must span many "periods" 2*pi/|dphi| ~ 420 of the slow pair.
        let n = 1600;
        let mean_sq: f64 = (0..n)
            .map(|k| {
                let w = 200.0 + 9800.0 * (k as f64 * 0.618_033_988_749_895).fract();
                geometric_amplification(src, &lenses, w).unwrap().norm_sqr()
            })
            .sum::<f64>()
            / n as f64;
        assert!(
            (mean_sq - mu_tot).abs() < 0.03 * mu_tot,
            "<|F|^2>={} vs mu_tot={}",
            mean_sq,
            mu_tot
        );
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
                geometric_amplification([y, 0.0], &[(0.0, 0.0, 1.0)], w)
                    .unwrap()
                    .norm_sqr()
            })
            .sum::<f64>()
            / 64.0;
        assert!(
            (mean_sq - mu_tot).abs() < 0.05,
            "<|F|^2>={} vs mu_tot={}",
            mean_sq,
            mu_tot
        );
    }

    #[test]
    fn weak_field_companions_complete() {
        // The N=3 prefix of the getting-started stock.  Exact reference: an
        // 80-digit mpmath root solve of the degree-10 Witt-Mao polynomial
        // (2026-10-17): 4 images (main + one companion per star),
        // sum |mu| = 1.0007201662805896309, F_geo(2) as pinned below.
        // The old grid Newton found only the main image here.
        let lenses = &NOTEBOOK_FIELD[..3];
        let data = point_mass_field_image_data(NOTEBOOK_BETA, lenses).unwrap();
        assert_eq!(data.len(), 4);
        let sum: f64 = data.iter().map(|d| d.mu.abs()).sum();
        assert!(
            (sum - 1.000_720_166_280_589_6).abs() < 1e-9,
            "sum|mu| = {sum}"
        );
        for &(a, b, m) in lenses {
            let r = 2.0 * m.sqrt() + 0.1;
            assert!(
                data.iter().any(|d| (d.x[0] - a).hypot(d.x[1] - b) <= r),
                "companion of ({a}, {b}) missed"
            );
        }
        let f = geometric_amplification(NOTEBOOK_BETA, lenses, 2.0).unwrap();
        let want = Complex64::new(0.954_319_205_893_619_8, -0.261_662_411_634_391_3);
        assert!((f - want).norm() < 1e-9, "F_geo(2) = {f} vs {want}");
    }

    #[test]
    fn caustic_image_set_is_reported_not_silent() {
        // Beta = 0 for an isolated point mass: the images are the whole
        // Einstein ring (degenerate; the stationary-phase sum is undefined).
        // The finder must report that, not return an empty or poisoned list.
        let err = point_mass_field_images([0.0, 0.0], &[(0.0, 0.0, 1.0)]).unwrap_err();
        assert!(matches!(err, Error::DegenerateImage), "{err:?}");
        assert!(geometric_amplification([0.0, 0.0], &[(0.0, 0.0, 1.0)], 2.0).is_err());
    }

    #[test]
    fn notebook_field_forty_masses_complete() {
        // The getting-started 40-mass field.  Independent reference (ladder +
        // grid multi-start, 2026-10-17; positions cross-checked against an
        // 80-digit mpmath solve of the degree-1601 polynomial's real images):
        // 41 images = 1 main + one faint companion per star,
        // sum |mu| = 1.4966407061178489, F_geo(2) as pinned below.
        let data = point_mass_field_image_data(NOTEBOOK_BETA, &NOTEBOOK_FIELD).unwrap();
        assert_eq!(data.len(), 41);
        let sum: f64 = data.iter().map(|d| d.mu.abs()).sum();
        assert!(
            (sum - 1.496_640_706_117_848_9).abs() < 1e-9,
            "sum|mu| = {sum}"
        );
        // The bright main image near beta, and its companions are all faint.
        let main = data
            .iter()
            .max_by(|a, b| a.mu.abs().partial_cmp(&b.mu.abs()).unwrap())
            .unwrap();
        assert!(
            (main.mu - 1.263_859_714_653_735_1).abs() < 1e-9,
            "main mu = {}",
            main.mu
        );
        assert!((main.x[0] - 0.509_894_758_243_287_5).abs() < 1e-9);
        assert!((main.x[1] - 0.280_295_650_081_515_2).abs() < 1e-9);
        assert!(data.iter().filter(|d| d.mu < 0.0).count() == 40);
        let f = geometric_amplification(NOTEBOOK_BETA, &NOTEBOOK_FIELD, 2.0).unwrap();
        let want = Complex64::new(-1.458_607_255_602_454_0, 0.459_269_484_456_860_0);
        assert!((f - want).norm() < 1e-8, "F_geo(2) = {f} vs {want}");
    }
}
