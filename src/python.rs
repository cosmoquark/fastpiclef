//! Python bindings (`fastpiclef`), built by maturin behind the `python`
//! feature. The module exposes the *parametrised* phase library and a choice
//! of solver method — deliberately **not** arbitrary Python phase callbacks,
//! which would pay an FFI cost per integrand evaluation and destroy the
//! performance the Rust core exists for.

use num_complex::Complex64;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::images::{geometric_amplification, point_mass_field_images};
use crate::phases::{
    AiryPhase, CustomRationalLens, PointMass, PointMassField, RationalLens, RationalLens2D,
    SimGLens,
};
use crate::saddles::saddles_from_polynomial;
use crate::{
    Error, PolarConfig, PolarSolver, RationalPhase, Solver, SolverConfig, ThimbleConfig,
    ThimbleSolver, solve_point_mass_polar,
};
use std::f64::consts::PI;

/// Free-space amplitude $(2\pi i/\omega)^{d/2}$ — the integral of the bare
/// quadratic phase, used to normalise the amplification factor to 1 in the
/// absence of a lens.
fn free_space(dim: usize, omega: f64) -> Complex64 {
    Complex64::from_polar(
        (2.0 * PI / omega).powf(dim as f64 / 2.0),
        dim as f64 * PI / 4.0,
    )
}

fn map_err(e: Error) -> PyErr {
    match e {
        Error::InvalidInput(_) | Error::UnsupportedDimension(_) | Error::DimensionMismatch { .. } => {
            PyValueError::new_err(e.to_string())
        }
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

/// Result of a solve: the raw Fresnel integral, the normalised amplification
/// factor (=1 for free space), an error estimate, which method ran, and a
/// method-dependent step/piece count.
#[pyclass(name = "Result", frozen)]
#[derive(Clone)]
struct PyOutput {
    #[pyo3(get)]
    integral: Complex64,
    #[pyo3(get)]
    amplification: Complex64,
    #[pyo3(get)]
    error: f64,
    #[pyo3(get)]
    method: String,
    #[pyo3(get)]
    steps: usize,
}

#[pymethods]
impl PyOutput {
    fn __repr__(&self) -> String {
        format!(
            "Result(amplification={:.6}, integral={:.6}, error={:.2e}, method='{}', steps={})",
            self.amplification, self.integral, self.error, self.method, self.steps
        )
    }
}

fn make_output(
    dim: usize,
    omega: f64,
    integral: Complex64,
    error: f64,
    method: &str,
    steps: usize,
) -> PyOutput {
    PyOutput {
        integral,
        amplification: integral / free_space(dim, omega),
        error,
        method: method.to_string(),
        steps,
    }
}

fn mesh_config_1d(
    omega: f64,
    delta: Option<f64>,
    flow_steps: Option<usize>,
    domain: Option<(f64, f64)>,
    quad_order: Option<u32>,
) -> SolverConfig {
    let mut c = SolverConfig {
        omega,
        ..SolverConfig::default()
    };
    if let Some(d) = delta {
        c.delta = d;
    }
    if let Some(n) = flow_steps {
        c.flow_steps = n;
    }
    if let Some(q) = quad_order {
        c.quad_order = q;
    }
    if let Some((lo, hi)) = domain {
        c.domain = vec![(lo, hi)];
    }
    c
}

/// Shared 1D solve over any rational phase: `method` in
/// {"mesh", "thimble", "auto"}.
fn solve_1d<P: RationalPhase>(
    phase: &P,
    omega: f64,
    method: &str,
    delta: Option<f64>,
    flow_steps: Option<usize>,
    domain: Option<(f64, f64)>,
    quad_order: Option<u32>,
) -> PyResult<PyOutput> {
    if !(omega > 0.0) {
        return Err(PyValueError::new_err("omega must be > 0"));
    }
    let tcfg = ThimbleConfig {
        omega,
        ..ThimbleConfig::default()
    };
    let mcfg = mesh_config_1d(omega, delta, flow_steps, domain, quad_order);
    match method {
        "mesh" => {
            let mut s = Solver::<1>::new(phase, mcfg);
            let o = s.run().map_err(map_err)?;
            Ok(make_output(1, omega, o.value, o.error, "mesh", s.steps_used()))
        }
        "thimble" => {
            let mut s = ThimbleSolver::new_rational(phase, tcfg);
            let o = s.run().map_err(map_err)?;
            Ok(make_output(1, omega, o.value, o.error, "thimble", s.pieces().len()))
        }
        "auto" => {
            let mut s = ThimbleSolver::new_rational(phase, tcfg);
            match s.run() {
                Ok(o) => Ok(make_output(
                    1,
                    omega,
                    o.value,
                    o.error,
                    "thimble",
                    s.pieces().len(),
                )),
                Err(Error::DegenerateSaddle) | Err(Error::TraceStalled) => {
                    let mut m = Solver::<1>::new(phase, mcfg);
                    let o = m.run().map_err(map_err)?;
                    Ok(make_output(
                        1,
                        omega,
                        o.value,
                        o.error,
                        "mesh (fallback)",
                        m.steps_used(),
                    ))
                }
                Err(e) => Err(map_err(e)),
            }
        }
        other => Err(PyValueError::new_err(format!(
            "unknown method '{other}'; use 'mesh', 'thimble', or 'auto'"
        ))),
    }
}

/// The traced thimble pieces of a 1D rational phase (ordered polylines) — for
/// plotting the deformed contour.
fn thimble_pieces<P: RationalPhase>(phase: &P, omega: f64) -> PyResult<Vec<Vec<Complex64>>> {
    let mut s = ThimbleSolver::new_rational(
        phase,
        ThimbleConfig {
            omega,
            ..ThimbleConfig::default()
        },
    );
    s.run().map_err(map_err)?;
    Ok(s.pieces().iter().cloned().collect())
}

// --- 1D phase classes -----------------------------------------------------

macro_rules! solve_thimble_methods {
    ($ty:ident) => {
        #[pymethods]
        impl $ty {
            #[pyo3(signature = (omega, *, method="auto", delta=None, flow_steps=None, domain=None, quad_order=None))]
            fn solve(
                &self,
                omega: f64,
                method: &str,
                delta: Option<f64>,
                flow_steps: Option<usize>,
                domain: Option<(f64, f64)>,
                quad_order: Option<u32>,
            ) -> PyResult<PyOutput> {
                solve_1d(&self.inner, omega, method, delta, flow_steps, domain, quad_order)
            }

            /// The traced Lefschetz-thimble pieces (lists of complex points).
            fn thimble(&self, omega: f64) -> PyResult<Vec<Vec<Complex64>>> {
                thimble_pieces(&self.inner, omega)
            }

            /// All stationary points (images) of the phase in the complex
            /// plane — the complete set from the polynomial saddle source.
            fn saddles(&self) -> PyResult<Vec<Complex64>> {
                let poly = self.inner.stationary_polynomial();
                let s = saddles_from_polynomial(&self.inner, &poly, 1e-11).map_err(map_err)?;
                Ok(s.iter().map(|sd| sd.z).collect())
            }

            /// Amplification factor across a list of frequencies (the
            /// wave→geometric-optics bridge). Returns `(amplifications,
            /// errors)`. A Python loop today; the signature reserves the
            /// seam for saddle warm-starting later.
            #[pyo3(signature = (omegas, *, method="auto"))]
            fn sweep(
                &self,
                omegas: Vec<f64>,
                method: &str,
            ) -> PyResult<(Vec<Complex64>, Vec<f64>)> {
                let mut amps = Vec::with_capacity(omegas.len());
                let mut errs = Vec::with_capacity(omegas.len());
                for w in omegas {
                    let out = solve_1d(&self.inner, w, method, None, None, None, None)?;
                    amps.push(out.amplification);
                    errs.push(out.error / free_space(1, w).norm());
                }
                Ok((amps, errs))
            }
        }
    };
}

/// Cubic ("Airy") phase $\phi(z) = z^3/3 - y z$.
#[pyclass(name = "AiryPhase")]
struct PyAiry {
    inner: AiryPhase,
}
#[pymethods]
impl PyAiry {
    #[new]
    fn new(y: f64) -> Self {
        Self {
            inner: AiryPhase { y },
        }
    }
}
solve_thimble_methods!(PyAiry);

/// 1D rational lens $\phi(z) = (z-\beta)^2/2 - \kappa/(1+z^2)$.
#[pyclass(name = "RationalLens")]
struct PyRationalLens {
    inner: RationalLens,
}
#[pymethods]
impl PyRationalLens {
    #[new]
    fn new(beta: f64, kappa: f64) -> Self {
        Self {
            inner: RationalLens { beta, kappa },
        }
    }
}
solve_thimble_methods!(PyRationalLens);

/// 1D 'simG' lens $\phi(z) = (z-\beta)^2/2 - \kappa/\sum_{k=0}^{k_\max}
/// z^{2k}/k!$.
#[pyclass(name = "SimGLens")]
struct PySimG {
    inner: SimGLens,
}
#[pymethods]
impl PySimG {
    #[new]
    fn new(beta: f64, kappa: f64, kmax: u32) -> Self {
        Self {
            inner: SimGLens { beta, kappa, kmax },
        }
    }
}
solve_thimble_methods!(PySimG);

/// User-defined 1D rational lens: $\phi(z) = (z-\beta)^2/2 - \kappa
/// A(z)/B(z)$, with `num` = $A$ and `den` = $B$ ascending complex
/// coefficients (`den` defaults to `[1]`, a polynomial potential).
#[pyclass(name = "CustomRationalLens")]
struct PyCustom {
    inner: CustomRationalLens,
}
#[pymethods]
impl PyCustom {
    #[new]
    #[pyo3(signature = (beta, kappa, num, den=None))]
    fn new(
        beta: f64,
        kappa: f64,
        num: Vec<Complex64>,
        den: Option<Vec<Complex64>>,
    ) -> PyResult<Self> {
        if num.is_empty() {
            return Err(PyValueError::new_err("num must be non-empty"));
        }
        let den = den.unwrap_or_else(|| vec![Complex64::from(1.0)]);
        if den.is_empty() {
            return Err(PyValueError::new_err("den must be non-empty (use [1])"));
        }
        Ok(Self {
            inner: CustomRationalLens {
                beta,
                kappa,
                num,
                den,
            },
        })
    }
}
solve_thimble_methods!(PyCustom);

// --- 2D phase classes -----------------------------------------------------

fn mesh_config_2d(
    omega: f64,
    delta: Option<f64>,
    flow_steps: Option<usize>,
    domain: Option<(f64, f64)>,
    quad_order: Option<u32>,
) -> SolverConfig {
    let mut c = mesh_config_1d(omega, delta, flow_steps, None, quad_order);
    if let Some((lo, hi)) = domain {
        c.domain = vec![(lo, hi); 2];
    }
    c
}

/// Axisymmetric 2D rational lens; `beta = (bx, by)`.
#[pyclass(name = "RationalLens2D")]
struct PyRationalLens2D {
    inner: RationalLens2D,
}
#[pymethods]
impl PyRationalLens2D {
    #[new]
    fn new(beta: (f64, f64), kappa: f64) -> Self {
        Self {
            inner: RationalLens2D {
                beta: [beta.0, beta.1],
                kappa,
            },
        }
    }

    #[pyo3(signature = (omega, *, method="auto", delta=None, flow_steps=None, domain=None, quad_order=None))]
    fn solve(
        &self,
        omega: f64,
        method: &str,
        delta: Option<f64>,
        flow_steps: Option<usize>,
        domain: Option<(f64, f64)>,
        quad_order: Option<u32>,
    ) -> PyResult<PyOutput> {
        if !(omega > 0.0) {
            return Err(PyValueError::new_err("omega must be > 0"));
        }
        match method {
            "polar" | "auto" => {
                let cfg = PolarConfig {
                    radial: ThimbleConfig {
                        omega,
                        ..ThimbleConfig::default()
                    },
                    ..PolarConfig::default()
                };
                let mut s = PolarSolver::new_rational(&self.inner, cfg);
                let o = s.run().map_err(map_err)?;
                Ok(make_output(2, omega, o.value, o.error, "polar", s.thetas_used()))
            }
            "mesh" => {
                let cfg = mesh_config_2d(omega, delta, flow_steps, domain, quad_order);
                let mut s = Solver::<2>::new(&self.inner, cfg);
                let o = s.run().map_err(map_err)?;
                Ok(make_output(2, omega, o.value, o.error, "mesh", s.steps_used()))
            }
            other => Err(PyValueError::new_err(format!(
                "unknown method '{other}'; use 'polar', 'mesh', or 'auto'"
            ))),
        }
    }
}

/// The isolated point-mass lens (psi = ln r), source at `(y, 0)`. Axisymmetric,
/// so the fast polar wrapper applies: each angle's radial saddle is the positive
/// root of `r^2 - b r - 1 = 0` (the point-mass lens equation). Validated against
/// the Nakamura-Deguchi closed form. `method` in {"polar", "auto", "mesh"}.
#[pyclass(name = "PointMass")]
struct PyPointMass {
    inner: PointMass,
}
#[pymethods]
impl PyPointMass {
    #[new]
    fn new(y: f64) -> Self {
        Self {
            inner: PointMass { y },
        }
    }

    #[pyo3(signature = (omega, *, method="auto", delta=None, flow_steps=None, domain=None, quad_order=None))]
    fn solve(
        &self,
        omega: f64,
        method: &str,
        delta: Option<f64>,
        flow_steps: Option<usize>,
        domain: Option<(f64, f64)>,
        quad_order: Option<u32>,
    ) -> PyResult<PyOutput> {
        if !(omega > 0.0) {
            return Err(PyValueError::new_err("omega must be > 0"));
        }
        match method {
            "polar" | "auto" => {
                // Wide u = ln r window (no branch cut in u, so a generous window
                // only sharpens the tail; the log cut that stalls the generic
                // half-line trace lives in r, not u).
                let cfg = PolarConfig {
                    radial: ThimbleConfig {
                        omega,
                        re_range: (-20.0, 12.0),
                        im_range: (-20.0, 20.0),
                        ..ThimbleConfig::default()
                    },
                    ..PolarConfig::default()
                };
                let (o, n) = solve_point_mass_polar(self.inner.y, &cfg).map_err(map_err)?;
                Ok(make_output(2, omega, o.value, o.error, "polar", n))
            }
            "mesh" => {
                let cfg = mesh_config_2d(omega, delta, flow_steps, domain, quad_order);
                let mut s = Solver::<2>::new(&self.inner, cfg);
                let o = s.run().map_err(map_err)?;
                Ok(make_output(2, omega, o.value, o.error, "mesh", s.steps_used()))
            }
            other => Err(PyValueError::new_err(format!(
                "unknown method '{other}'; use 'polar', 'mesh', or 'auto'"
            ))),
        }
    }
}

/// A field of point-mass lenses (the microlensing / stellar-field problem).
/// `lenses` is a list of `(x, y, mass)`; `beta = (bx, by)` is the source.
/// Only the general mesh solver applies (non-axisymmetric, log potentials).
#[pyclass(name = "PointMassField")]
struct PyPointMassField {
    inner: PointMassField,
}
#[pymethods]
impl PyPointMassField {
    #[new]
    #[pyo3(signature = (lenses, beta=(0.0, 0.0)))]
    fn new(lenses: Vec<(f64, f64, f64)>, beta: (f64, f64)) -> Self {
        Self {
            inner: PointMassField {
                beta: [beta.0, beta.1],
                lenses,
            },
        }
    }

    /// Number of point masses in the field.
    #[getter]
    fn n(&self) -> usize {
        self.inner.lenses.len()
    }

    /// The geometric-optics images (real solutions of the lens equation), each
    /// as `(x1, x2)` (Path 2, M1 — the Witt-Mao polynomial).
    fn images(&self) -> Vec<(f64, f64)> {
        point_mass_field_images(self.inner.beta, &self.inner.lenses)
            .into_iter()
            .map(|p| (p[0], p[1]))
            .collect()
    }

    /// Stationary-phase (geometric-optics) amplification: the sum over images
    /// `|mu_i|^(1/2) exp(i w phi_i - i pi n_i)`. Fast (O(#images)) and exact as
    /// `omega -> infinity` away from caustics (Path 2, M2). The `.error` field
    /// is 0 (this is an asymptotic form, not an error-controlled quadrature).
    fn geometric(&self, omega: f64) -> PyResult<PyOutput> {
        if !(omega > 0.0) {
            return Err(PyValueError::new_err("omega must be > 0"));
        }
        let f = geometric_amplification(self.inner.beta, &self.inner.lenses, omega);
        let integral = f * free_space(2, omega);
        Ok(make_output(2, omega, integral, 0.0, "geometric", self.inner.lenses.len()))
    }

    #[pyo3(signature = (omega, *, method="mesh", delta=None, flow_steps=None, domain=None, quad_order=None))]
    fn solve(
        &self,
        omega: f64,
        method: &str,
        delta: Option<f64>,
        flow_steps: Option<usize>,
        domain: Option<(f64, f64)>,
        quad_order: Option<u32>,
    ) -> PyResult<PyOutput> {
        if !(omega > 0.0) {
            return Err(PyValueError::new_err("omega must be > 0"));
        }
        match method {
            "mesh" | "auto" => {
                let cfg = mesh_config_2d(omega, delta, flow_steps, domain, quad_order);
                let mut s = Solver::<2>::new(&self.inner, cfg);
                let o = s.run().map_err(map_err)?;
                Ok(make_output(2, omega, o.value, o.error, "mesh", s.steps_used()))
            }
            other => Err(PyValueError::new_err(format!(
                "point-mass fields support only method='mesh' (got '{other}')"
            ))),
        }
    }
}

#[pymodule]
fn fastpiclef(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "Fast Picard-Lefschetz Fresnel-integral / wave-optics lensing solver.")?;
    m.add_class::<PyOutput>()?;
    m.add_class::<PyAiry>()?;
    m.add_class::<PyRationalLens>()?;
    m.add_class::<PySimG>()?;
    m.add_class::<PyCustom>()?;
    m.add_class::<PyRationalLens2D>()?;
    m.add_class::<PyPointMass>()?;
    m.add_class::<PyPointMassField>()?;
    Ok(())
}
