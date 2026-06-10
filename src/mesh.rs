use crate::{Error, Result};
use nalgebra::SVector;
use num_complex::Complex64;
use rayon::prelude::*;

/// Simplicial mesh living in $\mathbb{C}^D$.
///
/// `points` holds vertex coordinates as stack-sized `SVector`s.
/// `simplices` is a flat buffer of vertex indices with stride `D + 1`
/// (chosen over `Vec<[u32; D+1]>` because the latter requires
/// `generic_const_exprs`, which is unstable). `active[i]` flags whether
/// simplex `i` is currently part of the contour.
///
/// All simplices are kept positively oriented at construction time; the flow
/// deforms vertex positions continuously, so the (complex) orientation of the
/// chain is preserved and the quadrature can use the plain complex Jacobian
/// determinant without sign bookkeeping.
pub struct Mesh<const D: usize> {
    pub points: Vec<SVector<Complex64, D>>,
    pub simplices: Vec<u32>,
    pub active: Vec<bool>,
}

impl<const D: usize> Mesh<D> {
    /// Stride (= D + 1) of the `simplices` buffer.
    pub const VERTS_PER_SIMPLEX: usize = D + 1;

    pub fn empty() -> Self {
        Self {
            points: Vec::new(),
            simplices: Vec::new(),
            active: Vec::new(),
        }
    }

    pub fn num_simplices(&self) -> usize {
        self.simplices.len() / Self::VERTS_PER_SIMPLEX
    }

    pub fn simplex(&self, i: usize) -> &[u32] {
        let v = Self::VERTS_PER_SIMPLEX;
        &self.simplices[i * v..(i + 1) * v]
    }

    pub fn simplex_iter(&self) -> impl Iterator<Item = &[u32]> {
        self.simplices.chunks_exact(Self::VERTS_PER_SIMPLEX)
    }

    /// Build a uniform rectangular initial mesh on `domain`, embedded into
    /// $\mathbb{C}^D$ on the real slice (imaginary parts zero), with no edge
    /// of the underlying grid longer than `delta`.
    ///
    /// Each grid cell is split into `D!` simplices by the Kuhn (Freudenthal)
    /// triangulation: one simplex per permutation $\pi$ of the axes, walking
    /// from the cell's base corner along $e_{\pi(0)}, e_{\pi(1)}, \dots$.
    /// Odd permutations produce negatively oriented simplices; their last two
    /// vertices are swapped so that every simplex satisfies
    /// $\det[v_1 - v_0, \dots, v_D - v_0] > 0$.
    pub fn rectangular(domain: &[(f64, f64); D], delta: f64) -> Result<Self> {
        if !(delta > 0.0) {
            return Err(Error::InvalidInput("mesh spacing delta must be > 0"));
        }
        if domain.iter().any(|&(lo, hi)| !(hi > lo)) {
            return Err(Error::InvalidInput(
                "each domain interval must satisfy lo < hi",
            ));
        }

        // Cells per axis and the actual (≤ delta) spacing.
        let mut n_cells = [0usize; D];
        let mut spacing = [0.0f64; D];
        for k in 0..D {
            let (lo, hi) = domain[k];
            let n = ((hi - lo) / delta).ceil().max(1.0) as usize;
            n_cells[k] = n;
            spacing[k] = (hi - lo) / n as f64;
        }

        // Row-major strides over the (n_k + 1)-point lattice.
        let mut strides = [1usize; D];
        for k in (0..D.saturating_sub(1)).rev() {
            strides[k] = strides[k + 1] * (n_cells[k + 1] + 1);
        }
        let num_points: usize = (0..D).map(|k| n_cells[k] + 1).product();
        if num_points > u32::MAX as usize {
            return Err(Error::InvalidInput(
                "mesh exceeds u32 vertex indexing; increase delta",
            ));
        }

        // Lattice points, odometer order matching the strides above.
        let mut points = Vec::with_capacity(num_points);
        let mut idx = [0usize; D];
        loop {
            points.push(SVector::<Complex64, D>::from_fn(|k, _| {
                Complex64::new(domain[k].0 + idx[k] as f64 * spacing[k], 0.0)
            }));
            // Advance the odometer (last axis fastest).
            let mut k = D;
            loop {
                if k == 0 {
                    break;
                }
                k -= 1;
                if idx[k] < n_cells[k] {
                    idx[k] += 1;
                    break;
                }
                idx[k] = 0;
                if k == 0 {
                    break;
                }
            }
            if idx.iter().all(|&i| i == 0) {
                break;
            }
        }
        debug_assert_eq!(points.len(), num_points);

        // Kuhn triangulation: D! simplices per cell.
        let perms = permutations_with_parity(D);
        let num_cells: usize = n_cells.iter().product();
        let mut simplices = Vec::with_capacity(num_cells * perms.len() * (D + 1));
        let mut cell = [0usize; D];
        loop {
            let base: usize = (0..D).map(|k| cell[k] * strides[k]).sum();
            for (perm, odd) in &perms {
                let start = simplices.len();
                simplices.push(base as u32);
                let mut corner = base;
                for &axis in perm {
                    corner += strides[axis];
                    simplices.push(corner as u32);
                }
                if *odd {
                    let len = simplices.len();
                    simplices.swap(len - 1, len - 2);
                }
                debug_assert_eq!(simplices.len() - start, D + 1);
            }
            // Advance the cell odometer.
            let mut k = D;
            loop {
                if k == 0 {
                    break;
                }
                k -= 1;
                if cell[k] + 1 < n_cells[k] {
                    cell[k] += 1;
                    break;
                }
                cell[k] = 0;
                if k == 0 {
                    break;
                }
            }
            if cell.iter().all(|&i| i == 0) {
                break;
            }
        }

        let num_simplices = simplices.len() / (D + 1);
        Ok(Self {
            points,
            simplices,
            active: vec![true; num_simplices],
        })
    }

    /// Bisect every edge that exceeds `max_edge`.
    pub fn refine(&mut self, _max_edge: f64) -> Result<()> {
        Err(Error::NotImplemented("Mesh::refine"))
    }

    /// Deactivate every active simplex none of whose vertices satisfies
    /// `keep`. The predicate is evaluated once per vertex (in parallel), not
    /// once per (simplex, vertex) pair, so it may be moderately expensive
    /// (e.g. a phase evaluation).
    pub fn prune<F>(&mut self, keep: F)
    where
        F: Fn(&SVector<Complex64, D>) -> bool + Sync,
    {
        let mask = self.active_vertex_mask();
        let keep_flags: Vec<bool> = self
            .points
            .par_iter()
            .zip(mask.par_iter())
            .map(|(z, &alive)| alive && keep(z))
            .collect();
        for i in 0..self.num_simplices() {
            if !self.active[i] {
                continue;
            }
            if !self.simplex(i).iter().any(|&v| keep_flags[v as usize]) {
                self.active[i] = false;
            }
        }
    }

    /// `true` for every vertex that belongs to at least one active simplex.
    /// Vertices outside the mask no longer influence the integral and can be
    /// skipped by the flow.
    pub fn active_vertex_mask(&self) -> Vec<bool> {
        let mut mask = vec![false; self.points.len()];
        for (i, simplex) in self.simplex_iter().enumerate() {
            if self.active[i] {
                for &v in simplex {
                    mask[v as usize] = true;
                }
            }
        }
        mask
    }

    pub fn num_active(&self) -> usize {
        self.active.iter().filter(|&&a| a).count()
    }
}

/// All permutations of `0..d` together with their parity (`true` = odd).
/// `d` is tiny (≤ 4 in practice), so a simple recursive generation with an
/// O(d²) inversion count per permutation is plenty.
fn permutations_with_parity(d: usize) -> Vec<(Vec<usize>, bool)> {
    let mut out = Vec::new();
    let mut current = Vec::with_capacity(d);
    let mut used = vec![false; d];
    build_permutations(d, &mut current, &mut used, &mut out);
    out
}

fn build_permutations(
    d: usize,
    current: &mut Vec<usize>,
    used: &mut [bool],
    out: &mut Vec<(Vec<usize>, bool)>,
) {
    if current.len() == d {
        let inversions = (0..d)
            .flat_map(|i| (i + 1..d).map(move |j| (i, j)))
            .filter(|&(i, j)| current[i] > current[j])
            .count();
        out.push((current.clone(), inversions % 2 == 1));
        return;
    }
    for k in 0..d {
        if !used[k] {
            used[k] = true;
            current.push(k);
            build_permutations(d, current, used, out);
            current.pop();
            used[k] = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quadrature::complex_det;
    use nalgebra::SMatrix;

    /// Complex Jacobian determinant of simplex `i` of `mesh`.
    fn simplex_det<const D: usize>(mesh: &Mesh<D>, i: usize) -> Complex64 {
        let s = mesh.simplex(i);
        let z0 = mesh.points[s[0] as usize];
        let jac =
            SMatrix::<Complex64, D, D>::from_fn(|r, c| mesh.points[s[c + 1] as usize][r] - z0[r]);
        complex_det(jac)
    }

    #[test]
    fn rectangular_counts_and_orientation_2d() {
        let mesh = Mesh::<2>::rectangular(&[(0.0, 1.0), (0.0, 1.0)], 0.5).unwrap();
        assert_eq!(mesh.points.len(), 9); // 3 x 3 lattice
        assert_eq!(mesh.num_simplices(), 8); // 4 cells x 2!
        let mut total = Complex64::new(0.0, 0.0);
        for i in 0..mesh.num_simplices() {
            let det = simplex_det(&mesh, i);
            assert!(det.re > 0.0, "simplex {i} not positively oriented: {det}");
            assert!(det.im.abs() < 1e-15);
            total += det;
        }
        // Sum of det / D! = area of the box.
        assert!((total.re / 2.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rectangular_total_volume_3d() {
        let mesh = Mesh::<3>::rectangular(&[(0.0, 1.0), (0.0, 2.0), (0.0, 1.0)], 1.0).unwrap();
        assert_eq!(mesh.num_simplices(), 2 * 6); // 2 cells x 3!
        let total: Complex64 = (0..mesh.num_simplices())
            .map(|i| {
                let det = simplex_det(&mesh, i);
                assert!(det.re > 0.0);
                det
            })
            .sum();
        assert!((total.re / 6.0 - 2.0).abs() < 1e-12); // box volume 2
    }

    #[test]
    fn prune_deactivates_only_fully_dropped_simplices() {
        let mut mesh = Mesh::<1>::rectangular(&[(0.0, 1.0)], 0.25).unwrap();
        assert_eq!(mesh.num_simplices(), 4);
        // Keep only vertices with x < 0.6: segments [0,.25],[.25,.5],[.5,.75]
        // each retain at least one kept vertex; [0.75, 1.0] loses both.
        mesh.prune(|z| z[0].re < 0.6);
        assert_eq!(mesh.num_active(), 3);
        assert!(!mesh.active[3]);
    }
}
