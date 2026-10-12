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
use crate::lens_field::{
    Component, HybridMethod, LensField, Profile, lens_field_geometric,
    lens_field_geometric_corrected, lens_field_hybrid, lens_field_images, lens_field_solve,
    point_mass_field_geometric_corrected, point_mass_field_hybrid,
};
use crate::rc1d::Rational1DModel;
use crate::slice_batch::slice_field_batch_g;
use crate::{
    Error, FieldBatch, PolarConfig, PolarSolver, RationalPhase, SliceConfig, Solver, SolverConfig,
    ThimbleConfig, ThimbleSolver, rc_1d_batch, slice_field, slice_field_batch,
    solve_point_mass_polar,
};
use rayon::prelude::*;
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

/// Result of a batched frequency sweep (`PointMassField.sweep`): one shared
/// contour for all frequencies; arrays in the caller's omega order.
#[pyclass(name = "SweepResult", frozen)]
#[derive(Clone)]
struct PySweepResult {
    #[pyo3(get)]
    omegas: Vec<f64>,
    #[pyo3(get)]
    integral: Vec<Complex64>,
    #[pyo3(get)]
    amplification: Vec<Complex64>,
    #[pyo3(get)]
    error: Vec<f64>,
    #[pyo3(get)]
    method: String,
    #[pyo3(get)]
    inner_solves: usize,
    #[pyo3(get)]
    groups: usize,
}

#[pymethods]
impl PySweepResult {
    fn __repr__(&self) -> String {
        format!(
            "SweepResult(n={}, method='{}', inner_solves={}, groups={})",
            self.omegas.len(),
            self.method,
            self.inner_solves,
            self.groups
        )
    }
    fn __len__(&self) -> usize {
        self.omegas.len()
    }
}

/// Result of a batched source map (`PointMassField.map`): nested lists in the
/// caller's (beta0, beta1) orders.
#[pyclass(name = "MapResult", frozen)]
#[derive(Clone)]
struct PyMapResult {
    #[pyo3(get)]
    beta0s: Vec<f64>,
    #[pyo3(get)]
    beta1s: Vec<f64>,
    #[pyo3(get)]
    omega: f64,
    #[pyo3(get)]
    integral: Vec<Vec<Complex64>>,
    #[pyo3(get)]
    amplification: Vec<Vec<Complex64>>,
    #[pyo3(get)]
    error: Vec<Vec<f64>>,
    #[pyo3(get)]
    method: String,
    #[pyo3(get)]
    inner_solves: usize,
    #[pyo3(get)]
    groups: usize,
}

#[pymethods]
impl PyMapResult {
    fn __repr__(&self) -> String {
        format!(
            "MapResult({}x{}, omega={}, method='{}', inner_solves={}, groups={})",
            self.beta0s.len(),
            self.beta1s.len(),
            self.omega,
            self.method,
            self.inner_solves,
            self.groups
        )
    }
}

/// Build the batched `SliceConfig` from the keyword knobs.
fn batch_cfg(
    omega: f64,
    tol: Option<f64>,
    margin: Option<f64>,
    margin_outer: Option<f64>,
    band_ratio: Option<f64>,
) -> SliceConfig {
    let mut cfg = SliceConfig { omega, ..SliceConfig::default() };
    if let Some(t) = tol {
        cfg.tol_inner = t;
        cfg.tol_outer = t;
    }
    if let Some(m) = margin {
        cfg.margin = m;
    }
    if let Some(m) = margin_outer {
        cfg.margin_outer = m;
    }
    if let Some(b) = band_ratio {
        cfg.band_ratio = b;
    }
    cfg
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

/// Batched source-grid solve for the rational 1D lenses: `method="rc"` runs
/// the shared-contour RC batch (exact by Cauchy; one contour for the grid),
/// `method="thimble"` the production per-point thimble in parallel. Returns
/// `(integrals, amplifications, errors)` in the caller's beta order.
macro_rules! rational_grid_method {
    ($ty:ident) => {
        #[pymethods]
        impl $ty {
            #[pyo3(signature = (betas, omega, *, method="rc", tol=None, margin=None))]
            fn grid(
                &self,
                betas: Vec<f64>,
                omega: f64,
                method: &str,
                tol: Option<f64>,
                margin: Option<f64>,
            ) -> PyResult<(Vec<Complex64>, Vec<Complex64>, Vec<f64>)> {
                if !(omega > 0.0) {
                    return Err(PyValueError::new_err("omega must be > 0"));
                }
                match method {
                    "rc" => {
                        let mut cfg = SliceConfig { omega, ..SliceConfig::default() };
                        if let Some(t) = tol {
                            cfg.tol_inner = t;
                        }
                        if let Some(m) = margin {
                            cfg.margin = m;
                        }
                        let (vals, errs) =
                            rc_1d_batch(&self.inner, &betas, omega, &cfg).map_err(map_err)?;
                        let nrm = free_space(1, omega);
                        let amps = vals.iter().map(|v| *v / nrm).collect();
                        Ok((vals, amps, errs))
                    }
                    "thimble" | "auto" => {
                        let lens = &self.inner;
                        let results: Vec<(Complex64, f64)> = betas
                            .par_iter()
                            .map(|&b| {
                                let l = Rational1DModel::with_beta(lens, b);
                                ThimbleSolver::new_rational(
                                    &l,
                                    ThimbleConfig { omega, ..Default::default() },
                                )
                                .run()
                                .map(|o| (o.value, o.error))
                            })
                            .collect::<Result<Vec<_>, _>>()
                            .map_err(map_err)?;
                        let nrm = free_space(1, omega);
                        let vals: Vec<Complex64> = results.iter().map(|r| r.0).collect();
                        let errs: Vec<f64> = results.iter().map(|r| r.1).collect();
                        let amps = vals.iter().map(|v| *v / nrm).collect();
                        Ok((vals, amps, errs))
                    }
                    other => Err(PyValueError::new_err(format!(
                        "unknown method '{other}'; use 'rc' or 'thimble'"
                    ))),
                }
            }
        }
    };
}

rational_grid_method!(PyRationalLens);
rational_grid_method!(PySimG);
rational_grid_method!(PyCustom);

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
/// The default solver is the exact RC slice reduction (`method="slice"`);
/// `method="mesh"` remains available (it is the legacy flow solver and
/// diverges on log potentials at moderate to high frequency).
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

    /// Geometric-optics amplification **with the stationary-phase next-order
    /// correction** (P4.1), with the polynomial-complete Witt-Mao image set:
    /// `F = sum_j |mu_j|^(1/2) e^{i w phi_j - i pi n_j} (1 + C_j/(i w))`, the
    /// conservative estimate `sum_j |G0_j| |C_j| / w` in `.error` (integral
    /// units).  `None` (Python `None`) at a caustic; use `hybrid` or `solve`.
    fn geometric_corrected(&self, omega: f64) -> PyResult<Option<PyOutput>> {
        if !(omega > 0.0) {
            return Err(PyValueError::new_err("omega must be > 0"));
        }
        let out = point_mass_field_geometric_corrected(self.inner.beta, &self.inner.lenses, omega)
            .map_err(map_err)?;
        Ok(out.map(|g| {
            make_output(
                2,
                omega,
                g.value * free_space(2, omega),
                g.error * free_space(2, omega).norm(),
                "geometric+correction",
                g.images,
            )
        }))
    }

    /// High-omega hybrid (P4.1): geometric optics with the stationary-phase
    /// next-order correction when its conservative estimate is within `tol`
    /// (integral units), and the exact RC engine otherwise.  `.method` is
    /// `'geometric+correction'` or `'slice'`; the image set comes from the
    /// polynomial-complete Witt-Mao solver, so the fast branch cannot miss an
    /// image.  The default `tol = 1e-9` stays on the exact RC; raise it (e.g.
    /// `1e-3`) to unlock the fast branch at high omega.
    #[pyo3(signature = (omega, tol = 1e-9))]
    fn hybrid(&self, omega: f64, tol: f64) -> PyResult<PyOutput> {
        if !(omega > 0.0) {
            return Err(PyValueError::new_err("omega must be > 0"));
        }
        if !(tol > 0.0) {
            return Err(PyValueError::new_err("tol must be > 0"));
        }
        let out =
            point_mass_field_hybrid(self.inner.beta, &self.inner.lenses, omega, tol)
                .map_err(map_err)?;
        let (method, steps) = match out.method {
            HybridMethod::GeometricCorrected => ("geometric+correction", out.images),
            HybridMethod::Rc => ("slice", out.inner_solves),
        };
        Ok(make_output(2, omega, out.value, out.error, method, steps))
    }

    /// Batched frequency sweep: one shared contour for every frequency
    /// (bands of ratio `band_ratio` get their own contour). Exact by the same
    /// Cauchy construction as `solve`; validated against the Nakamura-Deguchi
    /// closed form in `tests/batch.rs`. Returns a `SweepResult`.
    #[pyo3(signature = (omegas, *, tol=None, margin=None, margin_outer=None, band_ratio=None))]
    fn sweep(
        &self,
        omegas: Vec<f64>,
        tol: Option<f64>,
        margin: Option<f64>,
        margin_outer: Option<f64>,
        band_ratio: Option<f64>,
    ) -> PyResult<PySweepResult> {
        let cfg = batch_cfg(1.0, tol, margin, margin_outer, band_ratio);
        let batch = FieldBatch {
            omegas: omegas.clone(),
            beta0s: vec![self.inner.beta[0]],
            beta1s: vec![self.inner.beta[1]],
        };
        let out = slice_field_batch(&self.inner, &batch, &cfg).map_err(map_err)?;
        let amp = out
            .values
            .iter()
            .zip(omegas.iter())
            .map(|(v, &w)| *v / free_space(2, w))
            .collect();
        Ok(PySweepResult {
            omegas,
            integral: out.values,
            amplification: amp,
            error: out.errors,
            method: "slice-batch".to_string(),
            inner_solves: out.inner_solves,
            groups: out.groups,
        })
    }

    /// Batched source map at fixed omega: one shared contour for the whole
    /// `beta0s x beta1s` grid (the magnification-map protocol). Returns a
    /// `MapResult` with nested lists in the caller's orders.
    #[pyo3(signature = (beta0s, beta1s, omega, *, tol=None, margin=None, margin_outer=None, band_ratio=None))]
    fn map(
        &self,
        beta0s: Vec<f64>,
        beta1s: Vec<f64>,
        omega: f64,
        tol: Option<f64>,
        margin: Option<f64>,
        margin_outer: Option<f64>,
        band_ratio: Option<f64>,
    ) -> PyResult<PyMapResult> {
        if !(omega > 0.0) {
            return Err(PyValueError::new_err("omega must be > 0"));
        }
        let cfg = batch_cfg(omega, tol, margin, margin_outer, band_ratio);
        let batch = FieldBatch {
            omegas: vec![omega],
            beta0s: beta0s.clone(),
            beta1s: beta1s.clone(),
        };
        let out = slice_field_batch(&self.inner, &batch, &cfg).map_err(map_err)?;
        let m1 = beta1s.len();
        let norm = free_space(2, omega);
        let mut integral = Vec::with_capacity(beta0s.len());
        let mut amplification = Vec::with_capacity(beta0s.len());
        let mut error = Vec::with_capacity(beta0s.len());
        for i0 in 0..beta0s.len() {
            let lo = i0 * m1;
            let hi = lo + m1;
            integral.push(out.values[lo..hi].to_vec());
            amplification.push(out.values[lo..hi].iter().map(|v| *v / norm).collect());
            error.push(out.errors[lo..hi].to_vec());
        }
        Ok(PyMapResult {
            beta0s,
            beta1s,
            omega,
            integral,
            amplification,
            error,
            method: "slice-batch".to_string(),
            inner_solves: out.inner_solves,
            groups: out.groups,
        })
    }

    /// Exact finite-omega field integral. `method="slice"` is the RC
    /// Cartesian-slice reduction (real core + rotated tails in both
    /// variables; exact by Cauchy, validated against the point-mass
    /// Nakamura-Deguchi closed form); `method="mesh"` is the legacy flow
    /// solver (delta/flow_steps/domain/quad_order apply to it only). `tol`
    /// sets both inner and outer acceptance tolerances; `margin` /
    /// `margin_outer` resize the inner/outer cores.
    #[pyo3(signature = (omega, *, method="slice", tol=None, margin=None, margin_outer=None, delta=None, flow_steps=None, domain=None, quad_order=None))]
    fn solve(
        &self,
        omega: f64,
        method: &str,
        tol: Option<f64>,
        margin: Option<f64>,
        margin_outer: Option<f64>,
        delta: Option<f64>,
        flow_steps: Option<usize>,
        domain: Option<(f64, f64)>,
        quad_order: Option<u32>,
    ) -> PyResult<PyOutput> {
        if !(omega > 0.0) {
            return Err(PyValueError::new_err("omega must be > 0"));
        }
        match method {
            "slice" | "auto" => {
                let mut cfg = SliceConfig {
                    omega,
                    ..SliceConfig::default()
                };
                if let Some(t) = tol {
                    cfg.tol_inner = t;
                    cfg.tol_outer = t;
                }
                if let Some(m) = margin {
                    cfg.margin = m;
                }
                if let Some(m) = margin_outer {
                    cfg.margin_outer = m;
                }
                let o = slice_field(&self.inner, &cfg).map_err(map_err)?;
                Ok(make_output(2, omega, o.value, o.error, "slice", o.inner_solves))
            }
            "mesh" => {
                let cfg = mesh_config_2d(omega, delta, flow_steps, domain, quad_order);
                let mut s = Solver::<2>::new(&self.inner, cfg);
                let o = s.run().map_err(map_err)?;
                Ok(make_output(2, omega, o.value, o.error, "mesh", s.steps_used()))
            }
            other => Err(PyValueError::new_err(format!(
                "unknown method '{other}'; use 'slice', 'mesh', or 'auto'"
            ))),
        }
    }
}

/// A star/halo component for `LensField`: a point mass (optionally softened).
#[pyclass(name = "PointComp", frozen)]
#[derive(Clone)]
struct PyPointComp {
    x: f64,
    y: f64,
    mass: f64,
    xc: f64,
}

#[pymethods]
impl PyPointComp {
    #[new]
    #[pyo3(signature = (x, y, mass, xc=0.0))]
    fn new(x: f64, y: f64, mass: f64, xc: f64) -> Self {
        Self { x, y, mass, xc }
    }
}

/// A singular isothermal sphere component.
#[pyclass(name = "SISComp", frozen)]
#[derive(Clone)]
struct PySisComp {
    x: f64,
    y: f64,
    psi0: f64,
}

#[pymethods]
impl PySisComp {
    #[new]
    fn new(x: f64, y: f64, psi0: f64) -> Self {
        Self { x, y, psi0 }
    }
}

/// A cored isothermal sphere component.
#[pyclass(name = "CISComp", frozen)]
#[derive(Clone)]
struct PyCisComp {
    x: f64,
    y: f64,
    psi0: f64,
    xc: f64,
}

#[pymethods]
impl PyCisComp {
    #[new]
    fn new(x: f64, y: f64, psi0: f64, xc: f64) -> Self {
        Self { x, y, psi0, xc }
    }
}

/// An NFW halo component (`xs` the scale radius).
#[pyclass(name = "NFWComp", frozen)]
#[derive(Clone)]
struct PyNfwComp {
    x: f64,
    y: f64,
    psi0: f64,
    xs: f64,
}

#[pymethods]
impl PyNfwComp {
    #[new]
    fn new(x: f64, y: f64, psi0: f64, xs: f64) -> Self {
        Self { x, y, psi0, xs }
    }
}

/// Extract a [`Component`] from one of the component pyclasses.
fn component_from_any(obj: &Bound<'_, PyAny>) -> PyResult<Component> {
    if let Ok(p) = obj.extract::<PyPointComp>() {
        return Ok(Component { x: p.x, y: p.y, profile: Profile::Point { mass: p.mass, xc: p.xc } });
    }
    if let Ok(p) = obj.extract::<PySisComp>() {
        return Ok(Component { x: p.x, y: p.y, profile: Profile::Sis { psi0: p.psi0 } });
    }
    if let Ok(p) = obj.extract::<PyCisComp>() {
        return Ok(Component { x: p.x, y: p.y, profile: Profile::Cis { psi0: p.psi0, xc: p.xc } });
    }
    if let Ok(p) = obj.extract::<PyNfwComp>() {
        return Ok(Component { x: p.x, y: p.y, profile: Profile::Nfw { psi0: p.psi0, xs: p.xs } });
    }
    Err(PyValueError::new_err(
        "components must be PointComp/SISComp/CISComp/NFWComp instances",
    ))
}

/// The raw inner RC slice `G(x2) = int dz0 exp(i omega phi(z0, x2))` for a
/// point-mass field — the exact slice kernel, supporting complex `x2` inside
/// the validated sector (`Im((x2-b_j)^2) > 0`).  `amplification` is normalised
/// by the 1D free-space factor; the raw integral is the fundamental quantity.
#[pyfunction]
#[pyo3(signature = (lenses, beta, x2, omega, *, tol=None, margin=None, tail_angle=None))]
fn slice_inner_rc(
    lenses: Vec<(f64, f64, f64)>,
    beta: (f64, f64),
    x2: (f64, f64),
    omega: f64,
    tol: Option<f64>,
    margin: Option<f64>,
    tail_angle: Option<f64>,
) -> PyResult<PyOutput> {
    let mut cfg = SliceConfig { omega, ..SliceConfig::default() };
    if let Some(t) = tol {
        cfg.tol_inner = t;
    }
    if let Some(m) = margin {
        cfg.margin = m;
    }
    if let Some(a) = tail_angle {
        cfg.tail_angle = a;
    }
    let field = PointMassField {
        beta: [beta.0, beta.1],
        lenses,
    };
    let o = crate::slice::slice_inner_rc(&field, Complex64::new(x2.0, x2.1), &cfg)
        .map_err(map_err)?;
    Ok(make_output(1, omega, o.value, o.error, "slice-inner", 0))
}

/// General lens field: stars/halos (axisymmetric profiles) plus the external
/// macro model (`kappa`, `gamma1`, `gamma2`).  `solve`/`sweep`/`map` run the
/// same exact RC engine (P2a); `images`/`geometric` give the real-space
/// geometry (multi-start Newton + stationary phase).
#[pyclass(name = "LensField")]
struct PyLensField {
    inner: LensField,
}

#[pymethods]
impl PyLensField {
    #[new]
    #[pyo3(signature = (components, beta=(0.0, 0.0), kappa=0.0, gamma1=0.0, gamma2=0.0))]
    fn new(
        components: Vec<Bound<'_, PyAny>>,
        beta: (f64, f64),
        kappa: f64,
        gamma1: f64,
        gamma2: f64,
    ) -> PyResult<Self> {
        let comps = components
            .iter()
            .map(component_from_any)
            .collect::<PyResult<Vec<_>>>()?;
        let inner = LensField::new([beta.0, beta.1], kappa, gamma1, gamma2, comps)
            .map_err(map_err)?;
        Ok(Self { inner })
    }

    /// Number of components.
    #[getter]
    fn n(&self) -> usize {
        self.inner.components.len()
    }

    /// The macro parameters as stored (after the shear rotation).
    #[getter]
    fn lambdas(&self) -> (f64, f64) {
        (self.inner.lambda0, self.inner.lambda2)
    }

    /// The geometric-optics images (real solutions of the lens equation).
    fn images(&self) -> Vec<(f64, f64)> {
        lens_field_images(&self.inner).into_iter().map(|p| (p[0], p[1])).collect()
    }

    /// Stationary-phase (geometric-optics) amplification; exact as omega -> oo
    /// away from caustics.  `.error` is 0 (an asymptotic form, not quadrature).
    fn geometric(&self, omega: f64) -> PyResult<PyOutput> {
        if !(omega > 0.0) {
            return Err(PyValueError::new_err("omega must be > 0"));
        }
        let f = lens_field_geometric(&self.inner, omega);
        Ok(make_output(2, omega, f * free_space(2, omega), 0.0, "geometric", self.inner.components.len()))
    }

    /// Geometric-optics amplification **with the stationary-phase next-order
    /// correction** (P4.1): `F = sum_j |mu_j|^(1/2) e^{i w phi_j - i pi n_j}
    /// (1 + C_j/(i w))`, with the conservative error estimate
    /// `sum_j |G0_j| |C_j| / w` reported in `.error` (integral units — the same
    /// relative bound on the amplification).  `None` (Python `None`) at a
    /// caustic, where the asymptotic form is invalid; use `hybrid` or `solve`
    /// there.
    fn geometric_corrected(&self, omega: f64) -> PyResult<Option<PyOutput>> {
        if !(omega > 0.0) {
            return Err(PyValueError::new_err("omega must be > 0"));
        }
        Ok(lens_field_geometric_corrected(&self.inner, omega).map(|g| {
            make_output(
                2,
                omega,
                g.value * free_space(2, omega),
                g.error * free_space(2, omega).norm(),
                "geometric+correction",
                g.images,
            )
        }))
    }

    /// High-omega hybrid (P4.1): geometric optics with the validated
    /// stationary-phase correction when its conservative estimate is within
    /// `tol` (in integral units), and the exact RC engine otherwise.
    /// `.method` is `'geometric+correction'` or `'slice'`.  The switch is
    /// honest: on the geometric branch the true error is at most `tol` (the
    /// estimate was measured to cover the corrected residual by >= 3.5x in
    /// every measured family, and the reported budget carries a further
    /// factor 3, so `tol` bounds the true error on the geometric branch.  The default `tol = 1e-9` stays on the exact RC;
    /// raise it (e.g. `1e-3`) to unlock the fast branch at high omega.
    #[pyo3(signature = (omega, tol = 1e-9))]
    fn hybrid(&self, omega: f64, tol: f64) -> PyResult<PyOutput> {
        if !(omega > 0.0) {
            return Err(PyValueError::new_err("omega must be > 0"));
        }
        if !(tol > 0.0) {
            return Err(PyValueError::new_err("tol must be > 0"));
        }
        let out = lens_field_hybrid(&self.inner, omega, tol).map_err(map_err)?;
        let (method, steps) = match out.method {
            HybridMethod::GeometricCorrected => ("geometric+correction", out.images),
            HybridMethod::Rc => ("slice", out.inner_solves),
        };
        Ok(make_output(2, omega, out.value, out.error, method, steps))
    }

    /// Exact finite-omega field integral (RC engine; P2a).
    #[pyo3(signature = (omega, *, tol=None, margin=None, margin_outer=None))]
    fn solve(
        &self,
        omega: f64,
        tol: Option<f64>,
        margin: Option<f64>,
        margin_outer: Option<f64>,
    ) -> PyResult<PyOutput> {
        if !(omega > 0.0) {
            return Err(PyValueError::new_err("omega must be > 0"));
        }
        let mut cfg = SliceConfig { omega, ..SliceConfig::default() };
        if let Some(t) = tol {
            cfg.tol_inner = t;
            cfg.tol_outer = t;
        }
        if let Some(m) = margin {
            cfg.margin = m;
        }
        if let Some(m) = margin_outer {
            cfg.margin_outer = m;
        }
        let o = lens_field_solve(&self.inner, &cfg).map_err(map_err)?;
        Ok(make_output(2, omega, o.value, o.error, "slice", o.inner_solves))
    }

    /// Batched frequency sweep (see `PointMassField.sweep`).
    #[pyo3(signature = (omegas, *, tol=None, margin=None, margin_outer=None, band_ratio=None))]
    fn sweep(
        &self,
        omegas: Vec<f64>,
        tol: Option<f64>,
        margin: Option<f64>,
        margin_outer: Option<f64>,
        band_ratio: Option<f64>,
    ) -> PyResult<PySweepResult> {
        let cfg = batch_cfg(1.0, tol, margin, margin_outer, band_ratio);
        let batch = FieldBatch {
            omegas: omegas.clone(),
            beta0s: vec![self.inner.beta[0]],
            beta1s: vec![self.inner.beta[1]],
        };
        let out = slice_field_batch_g(&self.inner, &batch, &cfg).map_err(map_err)?;
        let amp = out
            .values
            .iter()
            .zip(omegas.iter())
            .map(|(v, &w)| *v / free_space(2, w))
            .collect();
        Ok(PySweepResult {
            omegas,
            integral: out.values,
            amplification: amp,
            error: out.errors,
            method: "slice-batch".to_string(),
            inner_solves: out.inner_solves,
            groups: out.groups,
        })
    }

    /// Batched source map (see `PointMassField.map`).
    #[pyo3(signature = (beta0s, beta1s, omega, *, tol=None, margin=None, margin_outer=None, band_ratio=None))]
    fn map(
        &self,
        beta0s: Vec<f64>,
        beta1s: Vec<f64>,
        omega: f64,
        tol: Option<f64>,
        margin: Option<f64>,
        margin_outer: Option<f64>,
        band_ratio: Option<f64>,
    ) -> PyResult<PyMapResult> {
        if !(omega > 0.0) {
            return Err(PyValueError::new_err("omega must be > 0"));
        }
        let cfg = batch_cfg(omega, tol, margin, margin_outer, band_ratio);
        let batch = FieldBatch {
            omegas: vec![omega],
            beta0s: beta0s.clone(),
            beta1s: beta1s.clone(),
        };
        let out = slice_field_batch_g(&self.inner, &batch, &cfg).map_err(map_err)?;
        let m1 = beta1s.len();
        let norm = free_space(2, omega);
        let mut integral = Vec::with_capacity(beta0s.len());
        let mut amplification = Vec::with_capacity(beta0s.len());
        let mut error = Vec::with_capacity(beta0s.len());
        for i0 in 0..beta0s.len() {
            let lo = i0 * m1;
            let hi = lo + m1;
            integral.push(out.values[lo..hi].to_vec());
            amplification.push(out.values[lo..hi].iter().map(|v| *v / norm).collect());
            error.push(out.errors[lo..hi].to_vec());
        }
        Ok(PyMapResult {
            beta0s,
            beta1s,
            omega,
            integral,
            amplification,
            error,
            method: "slice-batch".to_string(),
            inner_solves: out.inner_solves,
            groups: out.groups,
        })
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
    m.add_class::<PySweepResult>()?;
    m.add_class::<PyMapResult>()?;
    m.add_class::<PyLensField>()?;
    m.add_class::<PyPointComp>()?;
    m.add_class::<PySisComp>()?;
    m.add_class::<PyCisComp>()?;
    m.add_class::<PyNfwComp>()?;
    m.add_function(wrap_pyfunction!(slice_inner_rc, m)?)?;
    Ok(())
}
