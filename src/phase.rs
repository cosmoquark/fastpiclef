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

/// Opt-in capability for 1D phases whose derivative is a rational function:
/// $\phi'(z) = N(z)/D(z)$ with polynomial $N, D$ and $D$ nonvanishing away
/// from the poles of $\phi$.
///
/// Declaring it unlocks the *complete* saddle source
/// ([`crate::saddles::saddles_from_polynomial`] via
/// [`crate::ThimbleSolver::new_rational`]): every stationary point is found,
/// with the guarantee grid-Newton searching cannot give. The coefficients
/// are only a *locator* — each root is polished against the phase's own
/// `gradient`, so they need to be accurate to Newton-basin size, not to
/// machine precision. General analytic phases simply don't implement this
/// trait and keep the grid-Newton path.
pub trait RationalPhase: Phase<1> {
    /// Ascending coefficients of $N(z)$ (index $k$ multiplies $z^k$).
    fn stationary_polynomial(&self) -> Vec<Complex64>;

    /// Poles of $\phi$ (zeros of $D$). Default: none (polynomial phases).
    /// Consumed by the (planned) mesh near-pole guard and diagnostics.
    fn poles(&self) -> Vec<Complex64> {
        Vec::new()
    }
}

/// 2D counterpart of [`RationalPhase`] for the polar decomposition: the
/// restriction of the phase to the ray $\mathbf{z} = r\,\hat{n}(\theta)$
/// must have a rational derivative in the complex radius $r$, with
/// coefficients depending on $\hat{n} = (\cos\theta, \sin\theta)$.
pub trait RationalPhase2D: Phase<2> {
    /// Ascending coefficients of the cleared radial $\phi_\theta'(r)$.
    fn radial_stationary_polynomial(&self, cos_t: f64, sin_t: f64) -> Vec<Complex64>;

    /// Poles of the radial restriction. Default: none.
    fn radial_poles(&self, _cos_t: f64, _sin_t: f64) -> Vec<Complex64> {
        Vec::new()
    }
}
