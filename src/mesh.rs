use crate::{Error, Result};
use nalgebra::SVector;
use num_complex::Complex64;

/// Simplicial mesh living in $\mathbb{C}^D$.
///
/// `points` holds vertex coordinates as stack-sized `SVector`s.
/// `simplices` is a flat buffer of vertex indices with stride `D + 1`
/// (chosen over `Vec<[u32; D+1]>` because the latter requires
/// `generic_const_exprs`, which is unstable). `active[i]` flags whether
/// simplex `i` is currently part of the contour.
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
    /// $\mathbb{C}^D$ on the real axis, with no edge longer than `delta`.
    /// Stub for the algorithm to fill in.
    pub fn rectangular(_domain: &[(f64, f64); D], _delta: f64) -> Result<Self> {
        Err(Error::NotImplemented("Mesh::rectangular"))
    }

    /// Bisect every edge that exceeds `max_edge`.
    pub fn refine(&mut self, _max_edge: f64) -> Result<()> {
        Err(Error::NotImplemented("Mesh::refine"))
    }

    /// Drop simplices whose vertices all fail `keep`.
    pub fn prune<F>(&mut self, _keep: F) -> Result<()>
    where
        F: Fn(&SVector<Complex64, D>) -> bool,
    {
        Err(Error::NotImplemented("Mesh::prune"))
    }
}
