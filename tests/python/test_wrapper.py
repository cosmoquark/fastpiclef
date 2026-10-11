"""Tests for the `fastpiclef` Python wrapper.

Runs with plain `python tests/python/test_wrapper.py` (no pytest needed); the
`test_*` functions are also collectable by pytest if it is installed. Kept
fast — the slow 2D point-mass field is exercised only at coarse settings.
"""

import cmath
import math

import numpy as np
from scipy.special import airy

import fastpiclef as fpl


def test_airy_matches_scipy():
    # Raw integral: 2π ω^{-1/3} Ai(-ω^{2/3} y).
    for y, omega in [(1.0, 1.0), (-1.0, 1.0), (1.0, 4.0)]:
        r = fpl.AiryPhase(y=y).solve(omega=omega)
        exact = 2 * math.pi * omega ** (-1 / 3) * airy(-(omega ** (2 / 3)) * y)[0]
        assert abs(r.integral - exact) / abs(exact) < 1e-8, (y, omega, r)


def test_methods_agree_on_rational_lens():
    # Near-caustic converging lens: thimble, mesh, and auto must coincide
    # (this is the Gap-2 case where the mesh used to give garbage).
    lens = fpl.RationalLens(beta=0.5, kappa=-0.3)
    thim = lens.solve(omega=1.0, method="thimble")
    mesh = lens.solve(omega=1.0, method="mesh", delta=0.01)
    auto = lens.solve(omega=1.0, method="auto")
    assert abs(thim.integral - mesh.integral) / abs(mesh.integral) < 1e-6
    assert auto.integral == thim.integral
    assert auto.method == "thimble"


def test_custom_reproduces_builtin():
    # psi = 1/(1+z^2): num=[1], den=[1,0,1].
    beta, kappa = 0.5, -0.3
    custom = fpl.CustomRationalLens(beta=beta, kappa=kappa, num=[1 + 0j], den=[1 + 0j, 0j, 1 + 0j])
    builtin = fpl.RationalLens(beta=beta, kappa=kappa)
    assert abs(custom.solve(omega=1.0).integral - builtin.solve(omega=1.0).integral) < 1e-10


def test_2d_polar_matches_mesh():
    g = fpl.RationalLens2D(beta=(0.4, 0.0), kappa=0.2)
    polar = g.solve(omega=1.0, method="polar")
    mesh = g.solve(omega=1.0, method="mesh", delta=0.04, domain=(-6.0, 6.0))
    assert abs(polar.integral - mesh.integral) / abs(mesh.integral) < 1e-6


def test_amplification_normalisation():
    # A vanishingly weak lens ⇒ amplification ≈ 1.
    weak = fpl.RationalLens(beta=0.0, kappa=1e-6)
    assert abs(weak.solve(omega=1.0).amplification - 1.0) < 1e-4


def test_thimble_pieces_shape():
    pcs = fpl.RationalLens(beta=0.0, kappa=-0.72).thimble(omega=1.0)
    assert len(pcs) >= 1
    assert all(isinstance(z, complex) for z in pcs[0])


def test_sweep():
    lens = fpl.RationalLens(beta=0.0, kappa=-0.72)
    omegas = [1.0, 3.0, 10.0]
    amps, errs = lens.sweep(omegas)
    assert len(amps) == len(errs) == 3
    assert all(isinstance(a, complex) for a in amps)
    assert all(e >= 0 for e in errs)


def test_errors():
    a = fpl.AiryPhase(y=1.0)
    for bad in [dict(omega=-1.0), dict(omega=1.0, method="nope")]:
        try:
            a.solve(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")


def test_point_mass_field_free_limit():
    # Empty field ⇒ free space, amplification = 1 (exact by construction now:
    # the RC slice solver is the default).
    empty = fpl.PointMassField([], beta=(0.4, 0.0))
    r = empty.solve(omega=2.0)
    assert r.method == "slice"
    assert abs(r.amplification - 1.0) < 1e-10, r


def test_point_mass_field_matches_nakamura_deguchi():
    # Unit mass at the origin, source (0.5, 0): raw integral vs the 30-digit
    # Nakamura-Deguchi closed form (comparison/referee.py::slice_raw_from_nd,
    # emitted 2026-10-11). The phase relation and method are validated in
    # tests/field.rs; this checks the Python path end to end.
    f = fpl.PointMassField([(0.0, 0.0, 1.0)], beta=(0.5, 0.0))
    r = f.solve(omega=1.0)
    want = 1.11515184895840158 + 10.6271092478315978j
    assert abs(r.integral - want) / abs(want) < 1e-12, r


def test_point_mass_field_methods():
    field = fpl.PointMassField([(1.0, 0.5, 0.1), (-0.8, -0.6, 0.1)], beta=(0.3, 0.0))
    default = field.solve(omega=2.0)
    explicit = field.solve(omega=2.0, method="slice")
    assert default.method == "slice"
    assert explicit.method == "slice"
    assert default.integral == explicit.integral
    # The legacy mesh solver stays callable (coarse settings; it is not the
    # trustworthy path on log potentials — that is why the default changed).
    mesh = field.solve(omega=2.0, method="mesh", delta=0.06, domain=(-6.0, 6.0))
    assert mesh.method == "mesh"
    assert math.isfinite(abs(mesh.amplification))
    # Unknown methods and bad frequencies are ValueError.
    for bad in [dict(omega=2.0, method="nope"), dict(omega=-1.0)]:
        try:
            field.solve(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
