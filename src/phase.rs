use nalgebra::SVector;
use num_complex::Complex64;

/// A complex phase $\phi : \mathbb{C}^D \to \mathbb{C}$ supplied by the user.
///
/// The integrand we ultimately compute is $e^{i \omega \phi(z)}$ (frequency
/// $\omega$ is folded into the solver configuration, not the trait).
/// For the Picard-Lefshetz descent we also need $\nabla \phi$; the Morse
/// function used to drive the flow is $h = -\mathrm{Im}\,\phi$, and its real
/// gradient field is recovered from `gradient` inside the solver.
pub trait Phase<const D: usize>: Send + Sync {
    fn value(&self, z: &SVector<Complex64, D>) -> Complex64;
    fn gradient(&self, z: &SVector<Complex64, D>) -> SVector<Complex64, D>;
}

/// Object-safe, dimension-erased view of a phase used at the runtime-dispatch
/// boundary. Users normally implement [`Phase`] directly; this trait exists so
/// that `solve(d, ...)` can take a single `&dyn DynPhase` and route it to the
/// right const-generic specialisation.
pub trait DynPhase: Send + Sync {
    fn dim(&self) -> usize;
    fn value_slice(&self, z: &[Complex64]) -> Complex64;
    fn gradient_slice(&self, z: &[Complex64], out: &mut [Complex64]);
}

/// Blanket adapter: any concrete `Phase<D>` automatically becomes a `DynPhase`.
impl<P, const D: usize> DynPhase for PhaseAsDyn<P, D>
where
    P: Phase<D>,
{
    fn dim(&self) -> usize {
        D
    }

    fn value_slice(&self, z: &[Complex64]) -> Complex64 {
        let v = SVector::<Complex64, D>::from_column_slice(z);
        self.inner.value(&v)
    }

    fn gradient_slice(&self, z: &[Complex64], out: &mut [Complex64]) {
        let v = SVector::<Complex64, D>::from_column_slice(z);
        let g = self.inner.gradient(&v);
        out.copy_from_slice(g.as_slice());
    }
}

/// Wrapper used by the blanket impl above. Construct with [`as_dyn`].
pub struct PhaseAsDyn<P, const D: usize> {
    pub(crate) inner: P,
}

/// Wrap a typed `Phase<D>` so it can cross the runtime-dispatch boundary as
/// `&dyn DynPhase`.
pub fn as_dyn<P: Phase<D>, const D: usize>(phase: P) -> PhaseAsDyn<P, D> {
    PhaseAsDyn { inner: phase }
}

/// Reverse adapter: a borrowed `dyn DynPhase` viewed as a `Phase<D>`.
/// Used by the runtime dispatcher in `lib.rs`; checks the dimension at
/// construction so the `from_column_slice` calls below are safe.
pub(crate) struct DynAsPhase<'a, const D: usize> {
    dyn_phase: &'a dyn DynPhase,
}

impl<'a, const D: usize> DynAsPhase<'a, D> {
    pub(crate) fn new(dyn_phase: &'a dyn DynPhase) -> crate::Result<Self> {
        if dyn_phase.dim() != D {
            return Err(crate::Error::DimensionMismatch {
                phase: dyn_phase.dim(),
                requested: D,
            });
        }
        Ok(Self { dyn_phase })
    }
}

impl<'a, const D: usize> Phase<D> for DynAsPhase<'a, D> {
    fn value(&self, z: &SVector<Complex64, D>) -> Complex64 {
        self.dyn_phase.value_slice(z.as_slice())
    }

    fn gradient(&self, z: &SVector<Complex64, D>) -> SVector<Complex64, D> {
        let mut out = [Complex64::new(0.0, 0.0); D];
        self.dyn_phase.gradient_slice(z.as_slice(), &mut out);
        SVector::<Complex64, D>::from_column_slice(&out)
    }
}
