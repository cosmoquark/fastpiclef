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


def test_point_mass_field_images_complete():
    # The old Witt-Mao finder silently returned [] from N=5 on; the certified
    # structural finder replaced it (2026-10-17).  Field: the first 5 stars of
    # the getting-started 40-mass stock (seed 7, masses 0.05).  Reference: an
    # 80-digit mpmath root solve -- 6 images (main + one companion per star),
    # sum |mu| = 1.0020887392067522767, F_geo(2) as pinned below.
    stock = [
        (0.75057279962800205, 2.3832828058174531),
        (1.6541141414711609, -1.6487568600564488),
        (-1.1990022905326474, 2.2413206723775714),
        (-2.9684081726065514, 1.9273705102965977),
        (1.7824165725122771, -0.19239028293767557),
    ]
    field = fpl.PointMassField([(x, y, 0.05) for x, y in stock], beta=(0.5, 0.2))
    imgs = field.images()
    assert len(imgs) == 6, imgs
    data = field.images_detailed()
    assert len(data) == 6
    assert {(im.x, im.y) for im in data} == {tuple(p) for p in imgs}
    assert all(im.morse in (0.0, 0.5, 1.0) for im in data)
    mu_sum = sum(abs(im.mu) for im in data)
    assert abs(mu_sum - 1.0020887392067523) < 1e-9, mu_sum
    # every star owns its faint companion image
    for x, y in stock:
        assert any(
            math.hypot(im.x - x, im.y - y) <= 2 * math.sqrt(0.05) + 0.1 for im in data
        )
    g = field.geometric(2.0)
    want = 0.9266206441958274 - 0.4057561845676481j
    assert abs(g.amplification - want) < 1e-8, g


def test_field_sweep_matches_solve():
    # Batched sweep: one shared contour per band; each entry must match the
    # single-point production solve (a different contour — Cauchy), and the
    # omega=1 entry must hit the hard-coded ND raw integral.
    field = fpl.PointMassField([(0.0, 0.0, 1.0)], beta=(0.5, 0.0))
    omegas = [1.0, 3.0, 10.0, 30.0]  # ratio 30 -> two bands
    sw = field.sweep(omegas)
    assert sw.method == "slice-batch"
    assert len(sw) == len(omegas)
    assert sw.groups >= 2
    for k, w in enumerate(omegas):
        ref = field.solve(omega=w)
        d = abs(sw.integral[k] - ref.integral) / abs(ref.integral)
        assert d < 1e-11, (w, d)
    want = 1.11515184895840158 + 10.6271092478315978j
    assert abs(sw.integral[0] - want) / abs(want) < 1e-11


def test_field_map_matches_solve():
    # Batched source map: each grid entry must match the single-point solve.
    field = fpl.PointMassField([(0.0, 0.0, 1.0)], beta=(0.0, 0.0))
    b0s = [-0.5, 0.0, 0.5]
    b1s = [-0.25, 0.25]
    m = field.map(b0s, b1s, omega=3.0)
    assert len(m.amplification) == len(b0s)
    assert len(m.amplification[0]) == len(b1s)
    for i, b0 in enumerate(b0s):
        for j, b1 in enumerate(b1s):
            ref = fpl.PointMassField([(0.0, 0.0, 1.0)], beta=(b0, b1)).solve(omega=3.0)
            d = abs(m.integral[i][j] - ref.integral) / abs(ref.integral)
            assert d < 1e-11, (b0, b1, d)


def test_rational_grid_matches_thimble():
    # Shared-contour 1D batch vs the per-point production thimble.
    lens = fpl.RationalLens(beta=0.0, kappa=-2.0)
    betas = list(np.linspace(-2.0, 2.0, 33))
    omega = 10.0
    integ, amps, errs = lens.grid(betas, omega, method="rc")
    integ2, _, _ = lens.grid(betas, omega, method="thimble")
    scale = max(abs(v) for v in integ2)
    worst = max(abs(a - b) / scale for a, b in zip(integ, integ2))
    assert worst < 1e-11, worst
    for k in (0, 16, 32):
        ref = fpl.RationalLens(beta=betas[k], kappa=-2.0).solve(omega=omega)
        assert abs(integ[k] - ref.integral) / abs(ref.integral) < 1e-11
    try:
        lens.grid(betas, omega, method="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unknown grid method")


def test_lens_field_point_equals_point_mass():
    # kappa = gamma = 0 with point components: the general model must
    # reproduce the validated point-mass class.
    beta = (0.1, 0.2)
    comps = [fpl.PointComp(0.2, -0.1, 0.6), fpl.PointComp(-0.4, 0.3, 0.4)]
    lf = fpl.LensField(comps, beta=beta)
    pm = fpl.PointMassField([(0.2, -0.1, 0.6), (-0.4, 0.3, 0.4)], beta=beta)
    a = lf.solve(omega=3.0).integral
    b = pm.solve(omega=3.0).integral
    assert abs(a - b) / abs(b) < 1e-11


def test_lens_field_macro_only_closed_form():
    kappa, g1 = 0.3, 0.2
    lf = fpl.LensField([], beta=(0.4, -0.3), kappa=kappa, gamma1=g1)
    w = 3.0
    r = lf.solve(omega=w)
    l0, l2 = lf.lambdas
    b0, b1 = 0.4, -0.3
    kin = 0.5 * (b0 * b0 + b1 * b1) - b0 * b0 / (2 * l0) - b1 * b1 / (2 * l2)
    import cmath

    # sign(lambda) * pi/4 each, plus the kinematic phase
    sign = lambda x: 1.0 if x > 0 else -1.0
    # F = (1/i) * e^{i sgn*pi/4 ...} e^{i w kin} / sqrt|.|  (the 1/i from omega/(2 pi i))
    ref = (
        cmath.exp(1j * (sign(l0) + sign(l2)) * 3.141592653589793 / 4.0)
        * cmath.exp(1j * w * kin)
        / (1j * cmath.sqrt(abs(l0 * l2)))
    )
    assert abs(r.amplification - ref) < 1e-10, (r.amplification, ref)


def test_lens_field_sis_matches_polar_reference():
    # Raw integral for a unit SIS at the origin, source (0.5, 0): the
    # high-precision polar referee value (emitted 2026-10-12).
    lf = fpl.LensField([fpl.SISComp(0.0, 0.0, 1.0)], beta=(0.5, 0.0))
    r = lf.solve(omega=3.0)
    want = 2.21863952037528067e00 - 3.87023745646334216e00j
    assert abs(r.integral - want) / abs(want) < 1e-11


def test_lens_field_geometric_macro_only():
    lf = fpl.LensField([], beta=(0.4, -0.3), kappa=0.3, gamma1=0.2)
    ims = lf.images()
    assert len(ims) == 1, ims
    l0, l2 = lf.lambdas
    g = lf.geometric(omega=1e6)
    assert abs(abs(g.amplification) - 1.0 / math.sqrt(abs(l0 * l2))) < 1e-5


def test_point_mass_nd_closed_form():
    # P4.4: the isolated point mass via the Nakamura-Deguchi closed form
    # (mpmath dps 40 references): microseconds and exact at low |z|.
    r = fpl.point_mass_nd(omega=1.0, y=1.0)
    want = 1.37744791808669720 + 0.20921170471940767j
    assert abs(r.amplification - want) / abs(want) < 1e-12, r
    assert r.error >= abs(r.amplification - want)
    # the polar (1D thimble) path agrees in magnitude; the ND form carries the
    # standard Fermat-offset phase, so complex values are not comparable.
    p = fpl.PointMass(0.5).solve(omega=10.0)
    nd = fpl.point_mass_nd(omega=10.0, y=0.5)
    assert abs(abs(nd.amplification) - abs(p.amplification)) < 1e-9


def test_hybrid_point_mass_switch_and_budget():
    # P4.1: a tight tol stays on the exact RC engine; a loose tol takes the
    # corrected geometric branch, which must beat the bare geometric sum
    # against the external ND |F| at (omega, y) = (100, 0.5) = 1.762705414788,
    # with its reported budget covering the truth.
    field = fpl.PointMassField([(0.0, 0.0, 1.0)], beta=(0.5, 0.0))
    exact = field.solve(omega=100.0)
    tight = field.hybrid(omega=100.0, tol=1e-12)
    assert tight.method == "slice", tight
    assert abs(tight.integral - exact.integral) < 1e-9 * abs(exact.integral)
    loose = field.hybrid(omega=100.0, tol=1e-2)
    assert loose.method == "geometric+correction", loose
    fnd = 1.762705414788
    rel = abs(abs(loose.amplification) - fnd) / fnd
    rel_geo = abs(abs(field.geometric(omega=100.0).amplification) - fnd) / fnd
    assert rel < rel_geo, (rel, rel_geo)
    # .error is the integral budget; the amplification budget is its rescaling
    est_f = loose.error * 100.0 / (2.0 * math.pi)
    assert est_f >= rel * fnd, (est_f, rel * fnd)


def test_lens_field_hybrid_quadratic_and_caustic():
    # Quadratic phase: C = 0, so the corrected branch must equal the geometric
    # sum exactly and any tol takes the fast branch (zero estimate).
    lf = fpl.LensField([], beta=(0.4, -0.3), kappa=0.3, gamma1=0.2)
    g = lf.geometric(omega=10.0)
    c = lf.geometric_corrected(omega=10.0)
    assert abs(c.amplification - g.amplification) < 1e-12
    assert c.error < 1e-12
    h = lf.hybrid(omega=10.0, tol=1e-9)
    assert h.method == "geometric+correction", h
    assert abs(h.integral - g.integral) < 1e-12 * abs(g.integral)
    # Caustic (source at the point-mass caustic): the asymptotic branch is
    # invalid (None) or carries a huge estimate, and the hybrid falls back.
    lf2 = fpl.LensField([fpl.PointComp(0.0, 0.0, 1.0)], beta=(0.0, 0.0))
    c2 = lf2.geometric_corrected(omega=50.0)
    assert c2 is None or c2.error > 1.0, c2
    h2 = lf2.hybrid(omega=50.0, tol=1e-3)
    assert h2.method == "slice", h2
    ref = lf2.solve(omega=50.0)
    assert abs(h2.integral - ref.integral) < 1e-9 * abs(ref.integral)


def test_slice_inner_rc_free_space():
    # The raw slice kernel with no lenses must reproduce the free-space 1D
    # integral: G(x2) = sqrt(2 pi / w) e^{i pi/4} e^{i w (x2 - beta1)^2 / 2}.
    w, beta, x2 = 3.0, (0.2, -0.1), (0.4, 0.0)
    r = fpl.slice_inner_rc([], beta, x2, omega=w)
    exact = (
        cmath.sqrt(2 * math.pi / w)
        * cmath.exp(1j * math.pi / 4)
        * cmath.exp(1j * w * (x2[0] - beta[1]) ** 2 / 2)
    )
    assert abs(r.integral - exact) / abs(exact) < 1e-13, r
    # And a real unit point mass at a pinch height stays finite and honest.
    r2 = fpl.slice_inner_rc([(0.0, 0.5, 1.0)], (0.2, 0.0), (0.5, 0.0), omega=10.0)
    assert math.isfinite(abs(r2.integral)) and r2.error < 1e-10, r2


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
