use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("dimension {0} is not supported (skeleton supports 1..=4)")]
    UnsupportedDimension(usize),
    #[error("phase reports dimension {phase}, but solve was called with d = {requested}")]
    DimensionMismatch { phase: usize, requested: usize },
    #[error("descent flow diverged")]
    FlowDiverged,
    #[error("pole encountered on contour")]
    PoleEncountered,
    #[error("invalid input: {0}")]
    InvalidInput(&'static str),
    #[error("not yet implemented: {0}")]
    NotImplemented(&'static str),
}

pub type Result<T> = std::result::Result<T, Error>;
