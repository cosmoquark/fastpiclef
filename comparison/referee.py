"""Method-independent ground truth for Fresnel integrals.

Two families of truth live here:

* 1D lens phases phi(x) = (x-beta)^2/2 - kappa*psi(x) with a decaying potential:
  adaptive quadrature of the raw oscillatory integrand on [-L, L] plus the exact
  Fresnel-function tails of the leading (x-beta)^2/2 term (`lens_referee`;
  verified L-independent in the test suite — the technique that validated the
  Gap-1/2 work).

* Point-mass fields (the Cartesian-slice kernel G(x2) and the full 2D field
  integral): the "RC" referees below — a real-axis core over the range spanning
  all branch-point abscissae, plus tails rotated by pi/4 into the Gaussian
  valleys. No branch point is ever swept, so the deformation is exact by Cauchy
  with no sheet/Stokes bookkeeping (see the Journal entry of 2026-10-11).
  _Caveat:_ this shares its *method* with the Rust RC kernel; its independence
  comes from the different contour (margin 4.0 vs the kernel's 1.5), from the
  external Nakamura-Deguchi closed form, from transposed-slicing checks, and
  from FIONA.
"""

import math

import mpmath as mp
import numpy as np
from scipy.integrate import quad
from scipy.special import airy, fresnel


def _fresnel_tail(a, omega):
    """int_a^inf exp(i*omega*t^2/2) dt via the Fresnel integrals S, C."""
    x = a * math.sqrt(omega / math.pi)
    s, c = fresnel(x)
    return math.sqrt(math.pi / omega) * complex(0.5 - c, 0.5 - s)


def lens_referee(phi_real, beta, omega, L=40.0):
    """Raw integral int_R exp(i*omega*phi) dx for a lens with quadratic tails."""
    re = quad(lambda x: math.cos(omega * phi_real(x)), -L, L, limit=8000, epsabs=1e-11)[0]
    im = quad(lambda x: math.sin(omega * phi_real(x)), -L, L, limit=8000, epsabs=1e-11)[0]
    return complex(re, im) + _fresnel_tail(L - beta, omega) + _fresnel_tail(L + beta, omega)


def airy_exact(y, omega):
    """int_R exp(i*omega*(x^3/3 - y x)) dx = 2*pi*omega^{-1/3}*Ai(-omega^{2/3} y)."""
    return complex(2 * math.pi * omega ** (-1 / 3) * airy(-(omega ** (2 / 3)) * y)[0], 0.0)


def point_mass_exact(w, y):
    """Wave-optics amplification F(w, y) for the isolated point mass psi = ln|x|.

    Nakamura (1998) / Takahashi & Nakamura (2003) closed form:

        F = exp[pi*w/4 + i*(w/2)*(ln(w/2) - 2*phi_m)]
            * Gamma(1 - i*w/2) * 1F1(i*w/2, 1; i*w*y^2/2)

    with x_m = (y + sqrt(y^2+4))/2 the brighter image and the Fermat-potential
    offset phi_m = (x_m - y)^2/2 - ln(x_m). This is F for the convention
    F(w,y) = (w/2pi i) int d^2x exp{i w [ |x-y|^2/2 - ln|x| ]} (free space -> 1).

    Uses mpmath for the complex confluent hypergeometric 1F1 and Gamma (scipy's
    are real-argument only); robust up to large w. Returns a Python complex.
    """
    w = float(w)
    y = float(y)
    xm = 0.5 * (y + math.sqrt(y * y + 4.0))
    phi_m = 0.5 * (xm - y) ** 2 - math.log(xm)
    prefac = mp.e ** (mp.pi * w / 4 + 1j * (w / 2) * (mp.log(w / 2) - 2 * phi_m))
    val = prefac * mp.gamma(1 - 1j * w / 2) * mp.hyp1f1(1j * w / 2, 1, 1j * w * y * y / 2)
    return complex(val)


# ---------------------------------------------------------------------------
# Point-mass-field referees: "RC" = real core + rotated tails.
#
# Phase convention (matches the crate's PointMassField):
#     phi(z0, x2) = 1/2[(z0-b0)^2 + (x2-b1)^2] - sum_j (m_j/2) Log w_j,
#     w_j = (z0 - a_j)^2 + (x2 - b_j)^2,  Log = composite principal log.
# The core keeps the real axis over [A, B] spanning every branch-point
# abscissa (A <= min(b0, a_j) - margin, B >= max(b0, a_j) + margin, widened by
# |Im x2| for complex slice positions); the tails continue along
# B + e^{i pi/4} t and A - e^{i pi/4} t, where the quadratic term decays like
# e^{-omega t^2/2} and every w_j stays in the open upper half plane (so the
# principal Log is the analytic continuation of the physical value). No branch
# point lies in the swept wedges, hence Cauchy makes the deformation exact.
#
# Two implementation lessons baked in (both met on 2026-10-05, see Journal):
#   * anchored panels — near a pinched branch point, x - a_j computed as a
#     difference suffers cancellation; each half-interval is parametrised from
#     its own breakpoint (x = x0 +/- r) so x - a_j is formed exactly;
#   * phase-scaled roundoff floor — e^{i omega phi} carries an absolute phase
#     error ~ eps * |omega phi|, so panels are accepted at the ulp level of
#     omega*phi, not of the integrand.
# ---------------------------------------------------------------------------

# The shared two-mass test field (tests/slice.rs::test_field and the Part D sweep).
TWO_MASS_BETA = (0.2, 0.1)
TWO_MASS_LENSES = [(-0.8, 0.5, 0.6), (0.9, -0.4, 0.4)]

_X20, _W20 = np.polynomial.legendre.leggauss(20)
_E4 = np.exp(1j * np.pi / 4)


def _gl20(f, a, b):
    """GL20 of f on [a, b]; returns (integral, integral of |f|). A non-finite
    node value (a node exactly on a pinched branch point) counts as 0."""
    c, h = 0.5 * (a + b), 0.5 * (b - a)
    v = f(c + h * _X20)
    v = np.where(np.isfinite(v), v, 0.0)
    return h * np.sum(_W20 * v), abs(h) * np.sum(_W20 * np.abs(v))


def _adapt(f, a, b, tol, minlen, floor):
    """Adaptive GL20 on [a, b]: a panel is accepted when whole vs halves agree
    to `tol` per unit length, or to `floor` times the panel's |f| mass, or when
    it is shorter than `minlen` (its whole |f| mass is then left unaccounted —
    the caller budgets it)."""
    total = 0j
    whole, _ = _gl20(f, a, b)
    stack = [(a, b, whole)]
    while stack:
        lo, hi, whole = stack.pop()
        mid = 0.5 * (lo + hi)
        (left, al), (right, ar) = _gl20(f, lo, mid), _gl20(f, mid, hi)
        diff = abs(left + right - whole)
        if diff <= tol * (hi - lo) or diff <= floor * (al + ar) or hi - lo < minlen:
            total += left + right
        else:
            stack.append((lo, mid, left))
            stack.append((mid, hi, right))
    return total


def _phi_core(r, x2, x0, sgn, beta, lenses):
    """phi at x = x0 + sgn*r on the real axis; every x - a_k is formed as the
    exact constant (x0 - a_k) plus sgn*r (exactly sgn*r when x0 = a_k)."""
    r = np.asarray(r, dtype=float)
    v = 0.5 * (((x0 - beta[0]) + sgn * r) ** 2 + (x2 - beta[1]) ** 2)
    for a, b, m in lenses:
        t = (x0 - a) + sgn * r
        v = v - 0.5 * m * np.log(t * t + (x2 - b) ** 2)
    return v


def _phi_c(z, x2, beta, lenses):
    """phi at complex z (tail segments), composite principal log."""
    z = np.asarray(z, dtype=complex)
    v = 0.5 * ((z - beta[0]) ** 2 + (x2 - beta[1]) ** 2)
    for a, b, m in lenses:
        v = v - 0.5 * m * np.log((z - a) ** 2 + (x2 - b) ** 2)
    return v


def slice_referee(x2, omega, lenses=TWO_MASS_LENSES, beta=TWO_MASS_BETA,
                  margin=4.0, tol=1e-14):
    """G(x2) = int_R exp(i*omega*phi(z0, x2)) dz0 for the point-mass field, by
    the RC construction. Supports complex x2 (outer rotated tails evaluate the
    inner integral there), provided Im((x2 - b_j)^2) > 0 for every mass — the
    condition that keeps every w_j in the upper half plane along the inner
    contour. Two different margins give two different contours: their
    agreement is the built-in self-check (see __main__)."""
    x2 = complex(x2)
    if x2.imag != 0.0:
        for _, b, _ in lenses:
            if ((x2 - b) ** 2).imag <= 0.0:
                raise ValueError(
                    f"complex slice position {x2} outside the validated sector: "
                    f"Im((x2-{b})^2) <= 0")
    sh = abs(x2.imag)
    avals = [a for a, _, _ in lenses]
    A = min([beta[0]] + avals) - margin - sh
    B = max([beta[0]] + avals) + margin + sh
    pts = sorted(set([A, B] + avals))
    phimax = (0.5 * max(abs(A - beta[0]), abs(B - beta[0])) ** 2
              + 0.5 * abs(x2 - beta[1]) ** 2 + 5.0)
    floor = 4e-16 * (1.0 + omega * phimax)
    total = 0j
    for lo, hi in zip(pts[:-1], pts[1:]):
        half = 0.5 * (hi - lo)
        for x0, sgn in [(lo, 1.0), (hi, -1.0)]:
            f = lambda r, x0=x0, sgn=sgn: np.exp(
                1j * omega * _phi_core(r, x2, x0, sgn, beta, lenses))
            total += _adapt(f, 0.0, half, tol, 1e-15, floor)
    T = math.sqrt(2.0 * 45.0 / omega) + 2.0
    total += _adapt(
        lambda t: _E4 * np.exp(1j * omega * _phi_c(B + _E4 * t, x2, beta, lenses)),
        0.0, T, tol, 1e-14, floor)
    total += _adapt(
        lambda t: _E4 * np.exp(1j * omega * _phi_c(A - _E4 * t, x2, beta, lenses)),
        0.0, T, tol, 1e-14, floor)
    return total


def field_referee(omega, lenses, beta, margin_outer=1.5, tol=1e-13):
    """I = int_R G(x2) dx2 for the point-mass field: the RC construction in the
    outer variable too — real core with breakpoints at the pinch heights
    x2 = b_j (anchored halves x2 = x0 +/- rho^2, dx2 = 2 rho d rho), plus
    pi/4-rotated tails on which the inner `slice_referee` runs at complex x2.
    The inner margin is fixed at 1.5; `margin_outer` varies the outer contour
    (the Cauchy self-check compares two values of it)."""
    bvals = [b for _, b, _ in lenses]
    A2 = min([beta[1]] + bvals) - margin_outer
    B2 = max([beta[1]] + bvals) + margin_outer
    pts = sorted(set([A2, B2] + bvals))
    floor = 4e-16 * (1.0 + omega * (
        0.5 * max(abs(A2 - beta[1]), abs(B2 - beta[1])) ** 2 + 10.0))

    def G(x2):
        return slice_referee(x2, omega, lenses, beta, margin=1.5)

    total = 0j
    for lo, hi in zip(pts[:-1], pts[1:]):
        rmax = math.sqrt(0.5 * (hi - lo))
        for x0, sgn in [(lo, 1.0), (hi, -1.0)]:
            g = lambda rho, x0=x0, sgn=sgn: np.array(
                [G(x0 + sgn * q * q) * 2.0 * q for q in rho])
            total += _adapt(g, 0.0, rmax, tol, 1e-12, floor)
    S = math.sqrt(2.0 * 45.0 / omega) + 2.0
    total += _adapt(lambda s: np.array([_E4 * G(B2 + _E4 * q) for q in s]),
                    0.0, S, tol, 1e-12, floor)
    total += _adapt(lambda s: np.array([_E4 * G(A2 - _E4 * q) for q in s]),
                    0.0, S, tol, 1e-12, floor)
    return total


def slice_raw_from_nd(omega, y):
    """The raw 2D integral I for a single unit point mass at the origin with
    source (y, 0), from the Nakamura-Deguchi closed form:

        I = (2 pi i / omega) * F * exp(i omega phi_m),

    with F the free-space-normalised amplification (`point_mass_exact`) and
    phi_m the Fermat-potential offset of the brighter image. The phase
    relation was measured against the brute-force RC field integral on
    2026-10-06 (agreement 2e-15; Journal entry of 2026-10-11). High precision
    (dps 30) so the values serve as 17-digit test references."""
    with mp.extraprec(30):
        w, yv = mp.mpf(float(omega)), mp.mpf(float(y))
        xm = (yv + mp.sqrt(yv * yv + 4)) / 2
        phim = (xm - yv) ** 2 / 2 - mp.log(xm)
        F = (mp.exp(mp.pi * w / 4 + 1j * (w / 2) * (mp.log(w / 2) - 2 * phim))
             * mp.gamma(1 - 1j * w / 2) * mp.hyp1f1(1j * w / 2, 1, 1j * w * yv * yv / 2))
        return complex((2 * mp.pi * 1j / w) * F * mp.exp(1j * w * phim))


if __name__ == "__main__":
    import sys
    import time

    if "--emit-rust-cases" in sys.argv:
        # (a) slice cases for the two-mass test field: (omega, x2, Re G, Im G)
        print("// (a) two-mass field slices: (omega, x2, re, im)")
        for omega in [1.0, 10.0, 100.0]:
            for x2 in [2.5, 1.3, -1.1, 0.0, 0.49, -0.41, 0.5 + 1e-6]:
                g = slice_referee(x2, omega)
                print(f"    ({omega:.1}, {x2!r}, {g.real:.17e}, {g.imag:.17e}),")
        # (b) complex slice positions on the rotated rays: (omega, x2.re, x2.im, re, im)
        print("// (b) complex x2 on rotated rays: (omega, x2re, x2im, re, im)")
        for omega in [1.0, 10.0]:
            for s in [0.3, 1.0]:
                for x2 in [2.0 + _E4 * s, -1.9 - _E4 * s]:
                    g = slice_referee(x2, omega)
                    print(f"    ({omega:.1}, {x2.real:.17e}, {x2.imag:.17e}, "
                          f"{g.real:.17e}, {g.imag:.17e}),")
        # (c) raw ND references for the single unit mass: (omega, y, re, im)
        print("// (c) single-mass raw ND references: (omega, y, re, im)")
        for omega, y in [(0.3, 1.0), (1.0, 0.5), (3.0, 0.1), (10.0, 2.0),
                         (30.0, 1.0), (100.0, 0.5)]:
            I = slice_raw_from_nd(omega, y)
            print(f"    ({omega:.1}, {y:.1}, {I.real:.17e}, {I.imag:.17e}),")
        sys.exit(0)

    print("RC referee self-test (two contours must agree — Cauchy).")
    X2S = [-3.0, -1.1, -0.5, -0.41, -0.4, 0.0, 0.3, 0.49, 0.5, 1.3, 2.5]
    for omega in [1.0, 3.0, 10.0, 30.0, 100.0]:
        worst = 0.0
        for x2 in X2S:
            g1 = slice_referee(x2, omega, margin=1.5)
            g2 = slice_referee(x2, omega, margin=4.0)
            worst = max(worst, abs(g1 - g2) / abs(g1))
        print(f"  slice omega={omega:6.1f}  max rel |G(m=1.5) - G(m=4)| = {worst:.1e}")

    t0 = time.time()
    I = field_referee(1.0, [(0.0, 0.0, 1.0)], (0.5, 0.0))
    ref = slice_raw_from_nd(1.0, 0.5)
    print(f"  field  omega= 1.0  single mass vs Nakamura-Deguchi: "
          f"{abs(I - ref) / abs(ref):.1e}  ({time.time() - t0:.0f}s)")

    for omega, anchor in [(3.0, -0.437797336628663 + 2.968066928420613j),
                          (10.0, 0.620555704800363 + 1.035804262604527j)]:
        t0 = time.time()
        I1 = field_referee(omega, TWO_MASS_LENSES, TWO_MASS_BETA, margin_outer=1.5)
        I2 = field_referee(omega, TWO_MASS_LENSES, TWO_MASS_BETA, margin_outer=3.0)
        print(f"  field  omega={omega:5.1f}  two contours: {abs(I1 - I2) / abs(I1):.1e}"
              f"   vs 2026-10-06 anchor: {abs(I1 - anchor) / abs(anchor):.1e}"
              f"  ({time.time() - t0:.0f}s)")
