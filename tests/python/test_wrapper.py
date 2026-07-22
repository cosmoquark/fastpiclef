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


def test_point_mass_field_runs_and_free_limit():
    # Empty field ⇒ free space, amplification = 1 (fast).
    empty = fpl.PointMassField([], beta=(0.4, 0.0))
    assert abs(empty.solve(omega=2.0, delta=0.05, domain=(-6.0, 6.0)).amplification - 1.0) < 1e-3
    # A small real field returns a finite, order-unity amplification (coarse).
    field = fpl.PointMassField([(1.0, 0.5, 0.1), (-0.8, -0.6, 0.1)], beta=(0.3, 0.0))
    r = field.solve(omega=2.0, delta=0.06, domain=(-6.0, 6.0))
    assert math.isfinite(abs(r.amplification)) and 0.2 < abs(r.amplification) < 5.0


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
