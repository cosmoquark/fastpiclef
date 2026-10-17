use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("dimension {0} is not supported (skeleton supports 1..=4)")]
    UnsupportedDimension(usize),
    #[error("phase reports dimension {phase}, but solve was called with d = {requested}")]
    DimensionMismatch { phase: usize, requested: usize },
    #[error("descent flow diverged")]
    FlowDiverged,
    /// A constant-phase trace failed to make progress — a Stokes-line
    /// configuration or a pole astride the contour. Signals the
    /// thimble→mesh fallback (see [`crate::thimble::solve_1d_hybrid`]).
    #[error("constant-phase trace stalled (Stokes line or pole on the contour?)")]
    TraceStalled,
    /// A stationary point with vanishing $\phi''$ (a caustic) was reached by
    /// the thimble solver, which has no uniform treatment for it. Signals the
    /// thimble→mesh fallback; the mesh-flow solver handles caustics natively.
    #[error("degenerate saddle (caustic); use the mesh-flow solver")]
    DegenerateSaddle,
    /// The geometric-optics image finder failed its completeness
    /// certificate (the index identity or the per-star companion check; see
    /// [`crate::images`]).  The image set, and anything derived from it, must
    /// not be trusted — this is deliberately *not* a silent short list.
    #[error("image search failed its completeness certificate: {0}")]
    ImageSearch(&'static str),
    /// The image search landed on a degenerate image (`|det H|` at the
    /// caustic floor): the stationary-phase form is invalid there — use the
    /// exact integral.
    #[error("degenerate image (caustic); the geometric-optics form is invalid — use solve()")]
    DegenerateImage,
    /// A cooperative interrupt probe ([`crate::SliceConfig::interrupt`]) fired
    /// at an orchestration boundary: the caller asked to abort a long call.
    /// The Python binding maps this to `KeyboardInterrupt`.
    #[error("interrupted")]
    Interrupted,
    #[error("invalid input: {0}")]
    InvalidInput(&'static str),
    #[error("not yet implemented: {0}")]
    NotImplemented(&'static str),
}

pub type Result<T> = std::result::Result<T, Error>;
