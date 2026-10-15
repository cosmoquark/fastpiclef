# Type stub for the `fastpiclef` extension module (hand-written).
#
# maturin auto-detects a root `<module>.pyi` in a pure-Rust project and installs
# it next to the compiled extension as `fastpiclef/__init__.pyi`, together with
# the `py.typed` marker (maturin guide, "Adding Python type information"), so
# VSCode/Pylance and mypy type `import fastpiclef` from this file: keep the
# signatures and the short docstrings here in sync with `src/python.rs`.
#
# Drift against the runtime bindings (names, members, signatures, defaults) is
# checked by `tests/python/test_stub.py`.  After editing, run
#     pixi run build && pixi run python tests/python/test_stub.py

from collections.abc import Sequence
from typing import Literal

__all__ = [
    "Result",
    "AiryPhase",
    "RationalLens",
    "SimGLens",
    "CustomRationalLens",
    "RationalLens2D",
    "PointMass",
    "PointMassField",
    "SweepResult",
    "MapResult",
    "LensField",
    "PointComp",
    "SISComp",
    "CISComp",
    "NFWComp",
    "slice_inner_rc",
    "point_mass_nd",
]

_SolveMethod1D = Literal["mesh", "thimble", "auto"]
_GridMethod1D = Literal["rc", "thimble", "auto"]
_SolveMethod2D = Literal["polar", "mesh", "auto"]
_FieldMethod = Literal["slice", "mesh", "auto"]


class Result:
    """Outcome of one solve.

    Attributes:
        integral: the raw Fresnel integral.
        amplification: the normalised amplification factor (= 1 for free
            space).
        error: estimated absolute error of `integral` (integral units -- the
            same relative bound on `amplification`).
        method: which solver ran (e.g. "slice", "thimble", "mesh", "polar",
            "nd", "geometric+correction").
        steps: method-dependent count (quadrature pieces, outer nodes, or
            images).
    """

    integral: complex
    amplification: complex
    error: float
    method: str
    steps: int

    def __repr__(self) -> str: ...


class AiryPhase:
    """Cubic ("Airy") phase `phi(z) = z^3/3 - y z`."""

    def __init__(self, y: float) -> None: ...
    def solve(
        self,
        omega: float,
        *,
        method: _SolveMethod1D = "auto",
        delta: float | None = None,
        flow_steps: int | None = None,
        domain: tuple[float, float] | None = None,
        quad_order: int | None = None,
    ) -> Result:
        """Exact 1D Fresnel integral at `omega`.

        `method="auto"` runs the complete-saddle thimble and falls back to the
        mesh flow at caustics/stalls; `"thimble"` / `"mesh"` force one path.
        `delta`, `flow_steps`, `domain` and `quad_order` configure the mesh
        only.
        """
        ...

    def thimble(self, omega: float) -> list[list[complex]]:
        """The traced Lefschetz-thimble pieces (polylines of complex points)."""
        ...

    def saddles(self) -> list[complex]:
        """All stationary points (images) of the phase in the complex plane
        -- the complete set from the polynomial saddle source."""
        ...

    def sweep(
        self, omegas: Sequence[float], *, method: _SolveMethod1D = "auto"
    ) -> tuple[list[complex], list[float]]:
        """Amplification factors and error estimates across `omegas` (the
        wave -> geometric-optics bridge), in the caller's order."""
        ...


class RationalLens:
    """1D rational lens `phi(z) = (z - beta)^2/2 - kappa/(1 + z^2)`."""

    def __init__(self, beta: float, kappa: float) -> None: ...
    def solve(
        self,
        omega: float,
        *,
        method: _SolveMethod1D = "auto",
        delta: float | None = None,
        flow_steps: int | None = None,
        domain: tuple[float, float] | None = None,
        quad_order: int | None = None,
    ) -> Result:
        """Exact 1D Fresnel integral at `omega`.

        `method="auto"` runs the complete-saddle thimble and falls back to the
        mesh flow at caustics/stalls; `"thimble"` / `"mesh"` force one path.
        `delta`, `flow_steps`, `domain` and `quad_order` configure the mesh
        only.
        """
        ...

    def thimble(self, omega: float) -> list[list[complex]]:
        """The traced Lefschetz-thimble pieces (polylines of complex points)."""
        ...

    def saddles(self) -> list[complex]:
        """All stationary points (images) of the phase in the complex plane
        -- the complete set from the polynomial saddle source."""
        ...

    def sweep(
        self, omegas: Sequence[float], *, method: _SolveMethod1D = "auto"
    ) -> tuple[list[complex], list[float]]:
        """Amplification factors and error estimates across `omegas` (the
        wave -> geometric-optics bridge), in the caller's order."""
        ...

    def grid(
        self,
        betas: Sequence[float],
        omega: float,
        *,
        method: _GridMethod1D = "rc",
        tol: float | None = None,
        margin: float | None = None,
    ) -> tuple[list[complex], list[complex], list[float]]:
        """Batched source grid, returns `(integrals, amplifications, errors)`
        in the caller's beta order.

        `method="rc"` shares one exact RC contour across all `betas`;
        `"thimble"` runs the production per-point thimble in parallel.
        """
        ...


class SimGLens:
    """1D 'simG' lens `phi(z) = (z-beta)^2/2 -
    kappa / sum_{k=0}^{kmax} z^{2k}/k!`."""

    def __init__(self, beta: float, kappa: float, kmax: int) -> None: ...
    def solve(
        self,
        omega: float,
        *,
        method: _SolveMethod1D = "auto",
        delta: float | None = None,
        flow_steps: int | None = None,
        domain: tuple[float, float] | None = None,
        quad_order: int | None = None,
    ) -> Result:
        """Exact 1D Fresnel integral at `omega`.

        `method="auto"` runs the complete-saddle thimble and falls back to the
        mesh flow at caustics/stalls; `"thimble"` / `"mesh"` force one path.
        `delta`, `flow_steps`, `domain` and `quad_order` configure the mesh
        only.
        """
        ...

    def thimble(self, omega: float) -> list[list[complex]]:
        """The traced Lefschetz-thimble pieces (polylines of complex points)."""
        ...

    def saddles(self) -> list[complex]:
        """All stationary points (images) of the phase in the complex plane
        -- the complete set from the polynomial saddle source."""
        ...

    def sweep(
        self, omegas: Sequence[float], *, method: _SolveMethod1D = "auto"
    ) -> tuple[list[complex], list[float]]:
        """Amplification factors and error estimates across `omegas` (the
        wave -> geometric-optics bridge), in the caller's order."""
        ...

    def grid(
        self,
        betas: Sequence[float],
        omega: float,
        *,
        method: _GridMethod1D = "rc",
        tol: float | None = None,
        margin: float | None = None,
    ) -> tuple[list[complex], list[complex], list[float]]:
        """Batched source grid, returns `(integrals, amplifications, errors)`
        in the caller's beta order.

        `method="rc"` shares one exact RC contour across all `betas`;
        `"thimble"` runs the production per-point thimble in parallel.
        """
        ...


class CustomRationalLens:
    """User-defined 1D rational lens `phi(z) = (z-beta)^2/2 -
    kappa A(z)/B(z)`, with `num` = A and `den` = B as ascending complex
    coefficients (`den` defaults to `[1]`, a polynomial potential)."""

    def __init__(
        self,
        beta: float,
        kappa: float,
        num: Sequence[complex],
        den: Sequence[complex] | None = None,
    ) -> None: ...
    def solve(
        self,
        omega: float,
        *,
        method: _SolveMethod1D = "auto",
        delta: float | None = None,
        flow_steps: int | None = None,
        domain: tuple[float, float] | None = None,
        quad_order: int | None = None,
    ) -> Result:
        """Exact 1D Fresnel integral at `omega`.

        `method="auto"` runs the complete-saddle thimble and falls back to the
        mesh flow at caustics/stalls; `"thimble"` / `"mesh"` force one path.
        `delta`, `flow_steps`, `domain` and `quad_order` configure the mesh
        only.
        """
        ...

    def thimble(self, omega: float) -> list[list[complex]]:
        """The traced Lefschetz-thimble pieces (polylines of complex points)."""
        ...

    def saddles(self) -> list[complex]:
        """All stationary points (images) of the phase in the complex plane
        -- the complete set from the polynomial saddle source."""
        ...

    def sweep(
        self, omegas: Sequence[float], *, method: _SolveMethod1D = "auto"
    ) -> tuple[list[complex], list[float]]:
        """Amplification factors and error estimates across `omegas` (the
        wave -> geometric-optics bridge), in the caller's order."""
        ...

    def grid(
        self,
        betas: Sequence[float],
        omega: float,
        *,
        method: _GridMethod1D = "rc",
        tol: float | None = None,
        margin: float | None = None,
    ) -> tuple[list[complex], list[complex], list[float]]:
        """Batched source grid, returns `(integrals, amplifications, errors)`
        in the caller's beta order.

        `method="rc"` shares one exact RC contour across all `betas`;
        `"thimble"` runs the production per-point thimble in parallel.
        """
        ...


class RationalLens2D:
    """Axisymmetric 2D rational lens; `beta = (bx, by)`."""

    def __init__(self, beta: tuple[float, float], kappa: float) -> None: ...
    def solve(
        self,
        omega: float,
        *,
        method: _SolveMethod2D = "auto",
        delta: float | None = None,
        flow_steps: int | None = None,
        domain: tuple[float, float] | None = None,
        quad_order: int | None = None,
    ) -> Result:
        """Exact 2D Fresnel integral.

        `method="polar"` (or `"auto"`) uses the axisymmetric polar/thimble
        reduction; `"mesh"` is the legacy 2D flow solver (`delta`,
        `flow_steps`, `domain` and `quad_order` apply to it only).
        """
        ...


class PointMass:
    """The isolated point-mass lens `psi = ln r`, source at `(y, 0)`.

    Axisymmetric, so the fast polar wrapper applies: each angle's radial
    saddle is the positive root of `r^2 - b r - 1 = 0` (the point-mass lens
    equation). Validated against the Nakamura-Deguchi closed form.
    """

    def __init__(self, y: float) -> None: ...
    def solve(
        self,
        omega: float,
        *,
        method: _SolveMethod2D = "auto",
        delta: float | None = None,
        flow_steps: int | None = None,
        domain: tuple[float, float] | None = None,
        quad_order: int | None = None,
    ) -> Result:
        """Exact 2D Fresnel integral.

        `method="polar"` (or `"auto"`) uses the axisymmetric polar/thimble
        reduction; `"mesh"` is the legacy 2D flow solver (`delta`,
        `flow_steps`, `domain` and `quad_order` apply to it only).
        """
        ...


class PointMassField:
    """A field of point-mass lenses (the microlensing / stellar-field
    problem).

    `lenses` is a list of `(x, y, mass)`; `beta = (bx, by)` is the source.
    The default solver is the exact RC slice reduction (`method="slice"`);
    `method="mesh"` remains available (the legacy flow solver, which diverges
    on log potentials at moderate to high frequency).
    """

    n: int
    """Number of point masses in the field."""

    def __init__(
        self,
        lenses: Sequence[tuple[float, float, float]],
        beta: tuple[float, float] = (0.0, 0.0),
    ) -> None: ...
    def images(self) -> list[tuple[float, float]]:
        """The geometric-optics images (real solutions of the lens equation),
        each as `(x1, x2)` -- the polynomial-complete Witt-Mao finder."""
        ...

    def geometric(self, omega: float) -> Result:
        """Stationary-phase (geometric-optics) amplification: the sum over
        images `|mu_i|^(1/2) exp(i w phi_i - i pi n_i)`.

        Fast (O(#images)) and exact as `omega -> infinity` away from
        caustics.  `.error` is 0 (this is an asymptotic form, not an
        error-controlled quadrature).
        """
        ...

    def geometric_corrected(self, omega: float) -> Result | None:
        """Geometric-optics amplification **with the stationary-phase
        next-order correction**: `F = sum_j |mu_j|^(1/2) e^{i w phi_j - i pi
        n_j} (1 + C_j/(i w))`, with the conservative estimate
        `sum_j |G0_j| |C_j| / w` reported in `.error` (integral units).

        Returns `None` at a caustic, where the asymptotic form is invalid;
        use `hybrid` or `solve` there.
        """
        ...

    def hybrid(self, omega: float, tol: float = 1e-9) -> Result:
        """High-omega hybrid: geometric optics with the validated
        stationary-phase next-order correction when its conservative estimate
        is within `tol` (integral units), and the exact RC engine otherwise.

        `.method` is `"geometric+correction"` or `"slice"`.  The default
        `tol = 1e-9` stays on the exact RC; raise it (e.g. `1e-3`) to unlock
        the fast branch at high omega.
        """
        ...

    def sweep(
        self,
        omegas: Sequence[float],
        *,
        tol: float | None = None,
        margin: float | None = None,
        margin_outer: float | None = None,
        band_ratio: float | None = None,
    ) -> SweepResult:
        """Batched frequency sweep: one shared contour for every frequency
        (bands of ratio `band_ratio` get their own contour).

        Exact by the same Cauchy construction as `solve`.  Returns a
        `SweepResult` with lists in the caller's `omegas` order.
        """
        ...

    def map(
        self,
        beta0s: Sequence[float],
        beta1s: Sequence[float],
        omega: float,
        *,
        tol: float | None = None,
        margin: float | None = None,
        margin_outer: float | None = None,
        band_ratio: float | None = None,
    ) -> MapResult:
        """Batched source map at fixed `omega`: one shared contour for the
        whole `beta0s x beta1s` grid (the magnification-map protocol).

        Returns a `MapResult` with nested lists in the caller's orders.
        """
        ...

    def solve(
        self,
        omega: float,
        *,
        method: _FieldMethod = "slice",
        tol: float | None = None,
        margin: float | None = None,
        margin_outer: float | None = None,
        delta: float | None = None,
        flow_steps: int | None = None,
        domain: tuple[float, float] | None = None,
        quad_order: int | None = None,
    ) -> Result:
        """Exact finite-omega field integral.

        `method="slice"` is the RC Cartesian-slice reduction (real core +
        rotated tails in both variables; exact by Cauchy, validated against
        the point-mass Nakamura-Deguchi closed form); `method="mesh"` is the
        legacy flow solver (`delta`, `flow_steps`, `domain` and `quad_order`
        apply to it only).  `tol` sets both inner and outer acceptance
        tolerances; `margin` / `margin_outer` resize the inner/outer cores.
        """
        ...


class SweepResult:
    """Result of a batched frequency sweep (`PointMassField.sweep`): one
    shared contour for all frequencies; lists in the caller's omega order.

    Attributes:
        omegas: the frequencies evaluated.
        integral: raw Fresnel integrals.
        amplification: normalised amplification factors.
        error: absolute error estimates (integral units).
        method: solver tag (e.g. "slice-batch").
        inner_solves: number of inner (1D) solves performed by the batch.
        groups: number of shared contour groups (frequency bands).
    """

    omegas: list[float]
    integral: list[complex]
    amplification: list[complex]
    error: list[float]
    method: str
    inner_solves: int
    groups: int

    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...


class MapResult:
    """Result of a batched source map (`PointMassField.map`): nested lists in
    the caller's `(beta0s, beta1s)` orders.

    Attributes:
        beta0s: the first source coordinate grid.
        beta1s: the second source coordinate grid.
        omega: the frequency evaluated.
        integral: raw Fresnel integrals, indexed `[i0][i1]`.
        amplification: normalised amplification factors, `[i0][i1]`.
        error: absolute error estimates (integral units), `[i0][i1]`.
        method: solver tag (e.g. "slice-batch").
        inner_solves: number of inner (1D) solves performed by the batch.
        groups: number of shared contour groups (frequency bands).
    """

    beta0s: list[float]
    beta1s: list[float]
    omega: float
    integral: list[list[complex]]
    amplification: list[list[complex]]
    error: list[list[float]]
    method: str
    inner_solves: int
    groups: int

    def __repr__(self) -> str: ...


class LensField:
    """General lens field: stars/halos (axisymmetric profiles) plus the
    external macro model (`kappa`, `gamma1`, `gamma2`).

    `components` is a list of `PointComp` / `SISComp` / `CISComp` / `NFWComp`;
    `beta = (bx, by)` is the source.  `solve` / `sweep` / `map` run the same
    exact RC engine; `images` / `geometric` give the real-space geometry.
    """

    n: int
    """Number of components."""

    lambdas: tuple[float, float]
    """The macro parameters `(lambda0, lambda2)` as stored (after the shear
    rotation into the diagonal frame)."""

    def __init__(
        self,
        components: Sequence[PointComp | SISComp | CISComp | NFWComp],
        beta: tuple[float, float] = (0.0, 0.0),
        kappa: float = 0.0,
        gamma1: float = 0.0,
        gamma2: float = 0.0,
    ) -> None: ...
    def images(self) -> list[tuple[float, float]]:
        """The geometric-optics images (real solutions of the lens equation)."""
        ...

    def geometric(self, omega: float) -> Result:
        """Stationary-phase (geometric-optics) amplification; exact as
        `omega -> infinity` away from caustics.

        `.error` is 0 (an asymptotic form, not quadrature).
        """
        ...

    def geometric_corrected(self, omega: float) -> Result | None:
        """Geometric-optics amplification **with the stationary-phase
        next-order correction** (P4.1): `F = sum_j |mu_j|^(1/2) e^{i w phi_j -
        i pi n_j} (1 + C_j/(i w))`, with the conservative error estimate in
        `.error` (integral units).

        Returns `None` at a caustic, where the asymptotic form is invalid;
        use `hybrid` or `solve` there.
        """
        ...

    def hybrid(self, omega: float, tol: float = 1e-9) -> Result:
        """High-omega hybrid: geometric optics with the validated
        stationary-phase next-order correction when its conservative estimate
        is within `tol` (integral units), and the exact RC engine otherwise.

        `.method` is `"geometric+correction"` or `"slice"`.  The default
        `tol = 1e-9` stays on the exact RC; raise it (e.g. `1e-3`) to unlock
        the fast branch at high omega.
        """
        ...

    def solve(
        self,
        omega: float,
        *,
        tol: float | None = None,
        margin: float | None = None,
        margin_outer: float | None = None,
    ) -> Result:
        """Exact finite-omega field integral (RC engine)."""
        ...

    def sweep(
        self,
        omegas: Sequence[float],
        *,
        tol: float | None = None,
        margin: float | None = None,
        margin_outer: float | None = None,
        band_ratio: float | None = None,
    ) -> SweepResult:
        """Batched frequency sweep (see `PointMassField.sweep`)."""
        ...

    def map(
        self,
        beta0s: Sequence[float],
        beta1s: Sequence[float],
        omega: float,
        *,
        tol: float | None = None,
        margin: float | None = None,
        margin_outer: float | None = None,
        band_ratio: float | None = None,
    ) -> MapResult:
        """Batched source map (see `PointMassField.map`)."""
        ...


class PointComp:
    """A star/halo component for `LensField`: a point mass, optionally
    softened: `psi = (mass/2) ln(r^2 + xc^2)` (`xc = 0` is the pure point
    mass)."""

    def __init__(self, x: float, y: float, mass: float, xc: float = 0.0) -> None: ...


class SISComp:
    """A singular isothermal sphere component: `psi = psi0 sqrt(r^2)`."""

    def __init__(self, x: float, y: float, psi0: float) -> None: ...


class CISComp:
    """A cored isothermal sphere component:
    `psi = psi0 [ s + xc ln(2 xc/(s + xc)) ]`, `s = sqrt(r^2 + xc^2)`
    (`xc = 0` reduces to SIS)."""

    def __init__(self, x: float, y: float, psi0: float, xc: float) -> None: ...


class NFWComp:
    """An NFW halo component (`xs` the scale radius):
    `psi = (psi0/2) ln((1-s)/2) ln((1+s)/2)`, `s = sqrt(1 - r^2/xs^2)` (the
    factored, cancellation-free form)."""

    def __init__(self, x: float, y: float, psi0: float, xs: float) -> None: ...


def slice_inner_rc(
    lenses: Sequence[tuple[float, float, float]],
    beta: tuple[float, float],
    x2: tuple[float, float],
    omega: float,
    *,
    tol: float | None = None,
    margin: float | None = None,
    tail_angle: float | None = None,
) -> Result:
    """The raw inner RC slice `G(x2) = int dz0 exp(i omega phi(z0, x2))` for a
    point-mass field -- the exact slice kernel, supporting complex `x2` inside
    the validated sector (`Im((x2-b_j)^2) > 0`).

    `amplification` is normalised by the 1D free-space factor; the raw
    integral is the fundamental quantity.
    """
    ...


def point_mass_nd(omega: float, y: float) -> Result:
    """Nakamura-Deguchi closed form for the **isolated unit point mass** at
    the origin with source `(y, 0)`: the microsecond-class exact evaluator
    (`1F1(i w/2, 1; i w y^2/2)` in log space).

    Validated to ~1e-13 for `|z| = omega y^2 / 2 <~ 6`, degrading like
    `eps * exp(|z|)` beyond; `.error` is a conservative estimate (measured
    coverage >= 2x).  For larger `|z|` use `PointMass(y).solve(...)` (the
    polar 1D path, fast for high omega) or `LensField.hybrid`.  The phase
    convention is the standard Fermat-offset one (compare `|F|`, not the
    complex value, against this package's solvers).
    """
    ...
