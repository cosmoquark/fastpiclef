use crate::{Error, Mesh, Phase, Result};
use nalgebra::{SMatrix, SVector};
use num_complex::Complex64;
use rayon::prelude::*;

/// Numerical integration of $e^{i \omega \phi}$ over a deformed contour
/// represented by `mesh`. Implementations return both a value and an
/// error estimate (e.g. obtained by comparing two quadrature orders).
pub trait Quadrature<const D: usize> {
    fn integrate<P: Phase<D> + ?Sized>(
        &self,
        mesh: &Mesh<D>,
        phase: &P,
        omega: f64,
    ) -> Result<(Complex64, f64)>;
}

/// Grundmann-Möller quadrature on D-simplices (Grundmann & Möller 1978).
///
/// `order` is the parameter $s$ of the family: the rule integrates
/// polynomials up to degree $2s+1$ exactly. The error estimate compares the
/// rule against its $s-1$ sibling, hence `order >= 1` is required. Some
/// weights are negative — standard for this family and harmless on the
/// smooth, exponentially damped integrands the descent flow produces.
#[derive(Debug, Clone, Copy)]
pub struct GrundmannMoeller {
    pub order: u32,
}

/// A concrete rule: barycentric nodes (flat, stride `dim + 1`) and one weight
/// per node, normalised so that the rule applied to $f \equiv 1$ returns the
/// volume $1/\mathrm{dim}!$ of the unit simplex.
struct GmRule {
    dim: usize,
    nodes: Vec<f64>,
    weights: Vec<f64>,
}

fn factorial(k: u32) -> f64 {
    (1..=k).map(f64::from).product()
}

/// Visit every β ∈ ℕ₀^`parts` with Σβ = `total`.
fn for_each_composition(total: u32, parts: usize, f: &mut impl FnMut(&[u32])) {
    fn rec(remaining: u32, parts: usize, buf: &mut Vec<u32>, f: &mut impl FnMut(&[u32])) {
        if buf.len() + 1 == parts {
            buf.push(remaining);
            f(buf);
            buf.pop();
            return;
        }
        for v in 0..=remaining {
            buf.push(v);
            rec(remaining - v, parts, buf, f);
            buf.pop();
        }
    }
    debug_assert!(parts >= 1);
    let mut buf = Vec::with_capacity(parts);
    rec(total, parts, &mut buf, f);
}

/// Determinant of a small complex matrix by Gaussian elimination with
/// partial pivoting. (nalgebra's `determinant` requires `DimMin` bounds
/// that are not derivable for a generic `Const<D>`, so we roll our own;
/// for the D <= 4 of interest this is also branch-cheap.)
pub(crate) fn complex_det<const D: usize>(mut m: SMatrix<Complex64, D, D>) -> Complex64 {
    let mut det = Complex64::new(1.0, 0.0);
    for col in 0..D {
        let mut piv = col;
        let mut best = m[(col, col)].norm_sqr();
        for r in col + 1..D {
            let v = m[(r, col)].norm_sqr();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best == 0.0 {
            return Complex64::new(0.0, 0.0);
        }
        if piv != col {
            m.swap_rows(piv, col);
            det = -det;
        }
        let diag = m[(col, col)];
        det *= diag;
        for r in col + 1..D {
            let factor = m[(r, col)] / diag;
            for c in col + 1..D {
                let sub = factor * m[(col, c)];
                m[(r, c)] -= sub;
            }
        }
    }
    det
}

/// Build the Grundmann-Möller rule of degree $2s+1$ on the unit `dim`-simplex.
fn gm_rule(dim: usize, s: u32) -> GmRule {
    let n = dim as u32;
    let degree = 2 * s + 1;
    let mut nodes = Vec::new();
    let mut weights = Vec::new();
    for i in 0..=s {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        let denom = f64::from(degree + n - 2 * i);
        let coeff = sign * 0.25f64.powi(s as i32) * denom.powi(degree as i32)
            / (factorial(i) * factorial(degree + n - i));
        for_each_composition(s - i, dim + 1, &mut |beta| {
            nodes.extend(beta.iter().map(|&b| f64::from(2 * b + 1) / denom));
            weights.push(coeff);
        });
    }
    GmRule {
        dim,
        nodes,
        weights,
    }
}

impl GmRule {
    fn num_nodes(&self) -> usize {
        self.weights.len()
    }

    fn node(&self, q: usize) -> &[f64] {
        let v = self.dim + 1;
        &self.nodes[q * v..(q + 1) * v]
    }

    /// Apply the rule to $e^{i\omega\phi}$ on the complex simplex whose
    /// vertex indices are `idx`. Returns the unit-simplex value; the caller
    /// multiplies by the complex Jacobian determinant of the chart.
    fn apply<const D: usize, P: Phase<D> + ?Sized>(
        &self,
        idx: &[u32],
        points: &[SVector<Complex64, D>],
        phase: &P,
        omega: f64,
    ) -> Complex64 {
        let mut acc = Complex64::new(0.0, 0.0);
        for q in 0..self.num_nodes() {
            let bary = self.node(q);
            let mut z = SVector::<Complex64, D>::zeros();
            for (lambda, &v) in bary.iter().zip(idx) {
                z += points[v as usize] * Complex64::from(*lambda);
            }
            let integrand = (Complex64::new(0.0, omega) * phase.value(&z)).exp();
            acc += integrand * Complex64::from(self.weights[q]);
        }
        acc
    }
}

impl<const D: usize> Quadrature<D> for GrundmannMoeller {
    fn integrate<P: Phase<D> + ?Sized>(
        &self,
        mesh: &Mesh<D>,
        phase: &P,
        omega: f64,
    ) -> Result<(Complex64, f64)> {
        if self.order < 1 {
            return Err(Error::InvalidInput(
                "GrundmannMoeller order must be >= 1 (the embedded error \
                 estimate compares orders s and s-1)",
            ));
        }
        let rule = gm_rule(D, self.order);
        let rule_lo = gm_rule(D, self.order - 1);

        // Integral over the simplicial chain: each active simplex with
        // vertices z_0..z_D carries the affine chart x -> z_0 + Σ x_j (z_j -
        // z_0), whose Jacobian is the constant complex determinant below
        // (orientation included; the mesh is built positively oriented and
        // deforms continuously, so no sign bookkeeping is needed).
        let zero = Complex64::new(0.0, 0.0);
        let (value, value_lo) = (0..mesh.num_simplices())
            .into_par_iter()
            .filter(|&i| mesh.active[i])
            .map(|i| {
                let s = mesh.simplex(i);
                let z0 = mesh.points[s[0] as usize];
                let jac = SMatrix::<Complex64, D, D>::from_fn(|r, c| {
                    mesh.points[s[c + 1] as usize][r] - z0[r]
                });
                let det = complex_det(jac);
                (
                    det * rule.apply(s, &mesh.points, phase, omega),
                    det * rule_lo.apply(s, &mesh.points, phase, omega),
                )
            })
            .reduce(|| (zero, zero), |a, b| (a.0 + b.0, a.1 + b.1));

        Ok((value, (value - value_lo).norm()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ∫_{T_n} Π x_j^{a_j} dx = Π a_j! / (n + Σ a_j)!
    fn monomial_integral(exponents: &[u32]) -> f64 {
        let n = exponents.len() as u32;
        let total: u32 = exponents.iter().sum();
        exponents.iter().map(|&a| factorial(a)).product::<f64>() / factorial(n + total)
    }

    /// Evaluate `rule` on the monomial with the given exponents over the unit
    /// simplex (cartesian coordinates = barycentric coordinates 1..=n).
    fn rule_on_monomial(rule: &GmRule, exponents: &[u32]) -> f64 {
        (0..rule.num_nodes())
            .map(|q| {
                let bary = rule.node(q);
                let m: f64 = exponents
                    .iter()
                    .enumerate()
                    .map(|(k, &a)| bary[k + 1].powi(a as i32))
                    .product();
                rule.weights[q] * m
            })
            .sum()
    }

    #[test]
    fn weights_sum_to_simplex_volume() {
        for dim in 1..=4 {
            for s in 0..=3 {
                let rule = gm_rule(dim, s);
                let total: f64 = rule.weights.iter().sum();
                let vol = 1.0 / factorial(dim as u32);
                assert!(
                    (total - vol).abs() < 1e-13,
                    "dim {dim}, s {s}: weight sum {total} != {vol}"
                );
            }
        }
    }

    #[test]
    fn exact_on_monomials_up_to_design_degree() {
        fn check_all(rule: &GmRule, exps: &mut Vec<u32>, slot: usize, budget: u32) {
            if slot == exps.len() {
                let exact = monomial_integral(exps);
                let got = rule_on_monomial(rule, exps);
                assert!(
                    (got - exact).abs() < 1e-13 * (1.0 + exact.abs()),
                    "dim {}, exps {exps:?}: got {got}, exact {exact}",
                    exps.len()
                );
                return;
            }
            for a in 0..=budget {
                exps[slot] = a;
                check_all(rule, exps, slot + 1, budget - a);
            }
            exps[slot] = 0;
        }

        // (dim, s) pairs; check every monomial with degree <= 2s+1.
        for &(dim, s) in &[(1usize, 1u32), (1, 3), (2, 2), (2, 3), (3, 2)] {
            let rule = gm_rule(dim, s);
            let mut exponents = vec![0u32; dim];
            check_all(&rule, &mut exponents, 0, 2 * s + 1);
        }
    }

    #[test]
    fn degree_one_past_design_is_not_exact() {
        // x^4 against the degree-3 rule on [0,1]: a clean miss guards
        // against the rule accidentally being graded higher than designed.
        let rule = gm_rule(1, 1);
        let got = rule_on_monomial(&rule, &[4]);
        assert!((got - 0.2).abs() > 1e-6);
    }

    /// φ ≡ 0 turns the integral into the plain volume of the meshed box.
    struct NullPhase;
    impl Phase<2> for NullPhase {
        fn value(&self, _z: &SVector<Complex64, 2>) -> Complex64 {
            Complex64::new(0.0, 0.0)
        }
        fn gradient(&self, _z: &SVector<Complex64, 2>) -> SVector<Complex64, 2> {
            SVector::zeros()
        }
    }

    #[test]
    fn integrates_unity_to_box_volume() {
        let mesh = Mesh::<2>::rectangular(&[(0.0, 2.0), (0.0, 1.5)], 0.5).unwrap();
        let quad = GrundmannMoeller { order: 2 };
        let (value, err) = quad.integrate(&mesh, &NullPhase, 1.0).unwrap();
        assert!((value - Complex64::new(3.0, 0.0)).norm() < 1e-12);
        assert!(err < 1e-12);
    }
}
