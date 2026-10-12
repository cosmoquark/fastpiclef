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
from scipy.special import airy, fresnel, jv


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


def _gl20(f, a, b, counter=None):
    """GL20 of f on [a, b]; returns (integral, integral of |f|). A non-finite
    node value (a node exactly on a pinched branch point) counts as 0. With
    `counter` (a one-element list) the 20 node evaluations are tallied."""
    c, h = 0.5 * (a + b), 0.5 * (b - a)
    v = f(c + h * _X20)
    v = np.where(np.isfinite(v), v, 0.0)
    if counter is not None:
        counter[0] += 20
    return h * np.sum(_W20 * v), abs(h) * np.sum(_W20 * np.abs(v))


def _adapt(f, a, b, tol, minlen, floor, counter=None):
    """Adaptive GL20 on [a, b]: a panel is accepted when whole vs halves agree
    to `tol` per unit length, or to `floor` times the panel's |f| mass, or when
    it is shorter than `minlen` (its whole |f| mass is then left unaccounted —
    the caller budgets it). `floor` may be a callable of the panel's upper end
    (for integrands whose phase magnitude varies strongly along [a, b])."""
    total = 0j
    whole, _ = _gl20(f, a, b, counter)
    stack = [(a, b, whole)]
    while stack:
        lo, hi, whole = stack.pop()
        mid = 0.5 * (lo + hi)
        (left, al), (right, ar) = _gl20(f, lo, mid, counter), _gl20(f, mid, hi, counter)
        diff = abs(left + right - whole)
        fl = floor(hi) if callable(floor) else floor
        if diff <= tol * (hi - lo) or diff <= fl * (al + ar) or hi - lo < minlen:
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


# ---------------------------------------------------------------------------
# P0 of the batched/general-lens-solver plan: referees for the general model
# (point masses + macro model (kappa, gamma1, gamma2) + radial profiles).
#
# Phase convention (matches the planned Rust `LensField`):
#     phi = 1/2 L0 x0^2 + 1/2 L2 x2^2 - b0 x0 - b1 x2 + 1/2|b|^2 - sum_j psi_j(w_j),
#     L0 = 1 - kappa - gamma,   L2 = 1 - kappa + gamma,
#     w_j = (x0 - a_j)^2 + (x2 - b_j)^2,
# in the frame where the external shear is diagonal: gamma = |gamma_vec| and
# every component centre and the source are rotated by -atan2(g2, g1)/2 (an
# exact reduction: all components are axisymmetric and the rotation preserves
# measure).  With L0 = L2 = 1 this is exactly the crate's PointMassField phase.
#
# The RC construction generalises with two changes:
#   * the inner tails rotate by sign(L0)*alpha and the outer tails by
#     sign(L2)*alpha, alpha in (0, pi/4] (the arbiter family);
#   * every tail/core length scales like |L|^{-1/2}, and the Gaussian centres
#     c0 = b0/L0, c2 = b1/L2 join the core extents.
# Exactness invariant (checked numerically by `check_sector`; the pen-and-paper
# argument is in the plan): along the whole contour Re w_j > 0 for every j
# (except a node exactly on a pinch), so no w_j ever crosses the (-inf, 0] cut
# and the principal log is the analytic continuation.  alpha <= pi/4 is what
# makes cos(2 alpha) >= 0 and closes the argument (mixed-sign saddle images).
#
# Profiles psi(w) are analytic in C \ (-inf, 0] with principal branches:
#     point(m, xc)   : (m/2) ln(w + xc^2)
#     sis(psi0)      : psi0 sqrt(w)
#     cis(psi0, xc)  : psi0 [s + xc ln(2 xc/(s + xc))],  s = sqrt(w + xc^2)
#     nfw(psi0, xs)  : (psi0/2) [ (1/4) ln^2(w/(4 xs^2)) - atanh^2(sqrt(1-u^2)) ]
#                      (= FIONA/GLoW's form on the real axis; evaluated through
#                       the factored form below, verified in the self-test)
# ---------------------------------------------------------------------------


def _shear_rotation(gamma1, gamma2):
    """(gamma, theta): shear magnitude and the rotation that diagonalises
    ẑ½ (gamma1 (x0^2 - x2^2) + 2 gamma2 x0 x2) = ½ gamma (u0^2 - u2^2)."""
    gamma = math.hypot(gamma1, gamma2)
    th = 0.5 * math.atan2(gamma2, gamma1) if gamma else 0.0
    return gamma, th


def _rot(x, y, th):
    """Coordinates in the frame rotated by `th` (i.e. R(-th) applied to the
    vector: u0 = x cos th + y sin th, u1 = -x sin th + y cos th)."""
    c, s = math.cos(th), math.sin(th)
    return (c * x + s * y, -s * x + c * y)


def _log1p_c(z):
    """Complex log1p: numpy's complex `log1p` computes log(1+z) internally and
    loses all relative accuracy for tiny |z| (1 + 1e-16 rounds to 1.11e-16),
    which bites the NFW ln((1+s)/2) factor as u -> 0.  Use a Horner series for
    |z| < 0.1 (relative truncation ~1e-29) and numpy elsewhere."""
    z = np.asarray(z, dtype=complex)
    az = np.abs(z)
    out = np.empty_like(z)
    small = az < 0.1
    if np.any(small):
        zs = z[small]
        acc = np.full_like(zs, 1.0 / 30.0)
        for k in range(29, 0, -1):
            acc = acc * (-zs) + 1.0 / k
        out[small] = zs * acc
    big = ~small
    if np.any(big):
        out[big] = np.log1p(z[big])
    return out


def _psi_w(kind, params, w):
    """Complex profile potential psi_j(w), vectorized over a complex array `w`.
    Analytic in C minus the cut (-inf, 0] with principal branches (see the
    section header). The non-finite policy for nodes exactly on a pinch lives
    in `_gl20`, not here."""
    w = np.asarray(w, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == "point":
            m, xc = params
            return 0.5 * m * np.log(w + xc * xc)
        if kind == "sis":
            (psi0,) = params
            return psi0 * np.sqrt(w)
        if kind == "cis":
            psi0, xc = params
            if xc == 0.0:
                return psi0 * np.sqrt(w)
            s = np.sqrt(w + xc * xc)
            return psi0 * (s + xc * np.log(2.0 * xc / (s + xc)))
        if kind == "nfw":
            psi0, xs = params
            u2 = w / (xs * xs)
            v = 1.0 - u2
            out = np.empty_like(u2)
            small = np.abs(v) < 1e-4
            if np.any(small):
                # atanh(sqrt(v))^2 = v (1 + v/3 + v^2/5 + ...)^2, series in v
                # (accurate well past |v| = 1e-4).
                vv = v[small]
                poly = 1.0 + vv * (1.0 / 3.0 + vv * (1.0 / 5.0))
                a2 = vv * poly * poly
                l2 = 0.25 * np.log(u2[small] / 4.0) ** 2
                out[small] = 0.5 * psi0 * (l2 - a2)
            big = ~small
            if np.any(big):
                u2b = u2[big]
                s = np.sqrt(1.0 - u2b)
                # ln((1-s)/2) = ln(u^2/(2(1+s)));  ln((1+s)/2) = log1p(-u^2/(2(1+s)))
                first = np.log(u2b / (2.0 * (1.0 + s)))
                half = _log1p_c(-u2b / (2.0 * (1.0 + s)))
                out[big] = 0.5 * psi0 * first * half
            return out
    raise ValueError(f"unknown profile kind {kind!r}")


def make_psi_w(components):
    """Sum psi_j(w) of *co-centred* components [(kind, params), ...] as a single
    callable w -> psi (for the axisymmetric `hankel_referee`)."""
    def psi_w(w):
        w = np.asarray(w, dtype=complex)
        acc = np.zeros_like(w)
        for kind, params in components:
            acc = acc + _psi_w(kind, params, w)
        return acc
    return psi_w


def point_components(lenses):
    """[(a, b, m), ...] -> the general component list [(kind, a, b, params)]."""
    return [("point", float(a), float(b), (float(m), 0.0)) for a, b, m in lenses]


def macro_only_integral(omega, beta, kappa=0.0, gamma1=0.0, gamma2=0.0):
    """Raw integral of the lensless macro model, exactly:
        I = (2 pi / (omega sqrt|L0 L2|)) e^{i (sgn L0 + sgn L2) pi/4} e^{i omega Kc},
        Kc = 1/2|b|^2 - b0^2/(2L0) - b1^2/(2L2)
    (b = beta in the shear-diagonal frame).  F = omega I / (2 pi i); the
    lensless free-space case gives I = 2 pi i / omega."""
    gamma, th = _shear_rotation(gamma1, gamma2)
    L0 = 1.0 - kappa - gamma
    L2 = 1.0 - kappa + gamma
    if L0 == 0.0 or L2 == 0.0:
        raise ValueError("critical macro model: L0 or L2 = 0")
    b = _rot(float(beta[0]), float(beta[1]), th)
    Kc = 0.5 * (b[0] ** 2 + b[1] ** 2) - b[0] ** 2 / (2.0 * L0) - b[1] ** 2 / (2.0 * L2)
    return (2.0 * math.pi / (omega * math.sqrt(abs(L0 * L2)))
            * np.exp(1j * (math.copysign(math.pi / 4, L0) + math.copysign(math.pi / 4, L2)))
            * np.exp(1j * omega * Kc))


def mass_sheet_nd(omega, beta, m, kappa):
    """Raw integral for a point mass `m` at the origin under a pure convergence
    sheet kappa (gamma = 0, lambda = 1 - kappa > 0), reduced exactly to the
    unit-mass Nakamura-Deguchi integral by x = u/sqrt(lambda), u = sqrt(m) v:

        I_kappa = lambda^{-1} m e^{i omega [Delta - (m/2) ln m]} J(m omega, y_eff),
        Delta   = (m/2) ln lambda + 1/2 |beta|^2 (1 - 1/lambda),
        y_eff   = |beta| / sqrt(lambda m),

    with J the unit-mass raw integral (= `slice_raw_from_nd`).  F_kappa =
    omega I/(2 pi i) = lambda^{-1} e^{i omega [Delta-(m/2) ln m]} F_ND(m omega, y_eff).
    Validated against `hankel_referee` and the general Cartesian referee in the
    P0 experiments (this is the external anchor for the macro model)."""
    lam = 1.0 - float(kappa)
    if not lam > 0.0:
        raise ValueError("mass_sheet_nd requires 1 - kappa > 0")
    y2 = float(beta[0]) ** 2 + float(beta[1]) ** 2
    Delta = 0.5 * m * math.log(lam) + 0.5 * y2 * (1.0 - 1.0 / lam)
    y_eff = math.sqrt(y2) / math.sqrt(lam * m)
    J = slice_raw_from_nd(m * omega, y_eff)
    return J * m * np.exp(1j * omega * (Delta - 0.5 * m * math.log(m))) / lam


def hankel_referee(omega, y, psi_w, kappa=0.0, tol=1e-13, counter=None):
    """Axisymmetric referee (gamma = 0; every component centred at the origin):

        F = -2i omega d2 e^{i omega y^2/2} int_0^inf rho^3 J0(omega y d rho^2)
                exp(i omega [a rho^4 - psi_w(d2 rho^4)]) d rho,

    where d = e^{i sgn(lambda) pi/4}, d2 = d^2, a = lambda d2 / 2 (Im a > 0), on
    the deformed ray r = d rho^2 -- a *polar* reduction, independent of the
    Cartesian RC construction.  With psi_w = 0 the identity gives F = 1 for
    every y; with the point mass it must reproduce Nakamura-Deguchi (checked in
    the self-test).  Validated range: omega*y up to a few tens.  Returns the
    raw integral I, like every referee here (F = omega I/(2 pi i))."""
    lam = 1.0 - float(kappa)
    if lam == 0.0:
        raise ValueError("critical macro model: lambda = 0")
    sg = 1.0 if lam > 0.0 else -1.0
    d = np.exp(0.25j * sg * math.pi)
    d2 = d * d
    a = 0.5 * lam * d2

    def f(rho):
        rho = np.asarray(rho, dtype=float)
        w = d2 * rho ** 4
        return (rho ** 3 * jv(0, omega * y * d * rho ** 2)
                * np.exp(1j * omega * (a * rho ** 4 - psi_w(w))))

    # Tail length: decay like e^{-omega |lambda| rho^4/2} beating the J0 growth
    # e^{omega y rho^2/sqrt(2)}; probe upward from the balance scale (the grid
    # starts just off 0.0: profile logs diverge there but contribute nothing).
    rho_max = max(math.sqrt(2.0 * abs(y) / abs(lam) + 2.0),
                  (2.0 * 45.0 / (omega * abs(lam))) ** 0.25)
    for _ in range(40):
        grid = np.linspace(1e-12, rho_max, 257)
        vals = np.abs(f(grid))
        if not np.all(np.isfinite(vals)):
            raise RuntimeError("hankel_referee: non-finite integrand on the ray")
        if abs(f(rho_max)) <= 1e-17 * max(vals.max(), 1e-300) or rho_max > 1e4:
            break
        rho_max *= 1.5
    if rho_max > 1e4:
        raise RuntimeError("hankel_referee: tail decay probe failed")

    # Local phase-roundoff floor: the phase magnitude of the exponent (quadratic
    # + profile) *and* of the J0 argument at the panel's own position — the
    # global-rho_max version over-accepts the high-mass panels and costs ~1e-11.
    def floor_fn(t):
        ps = float(np.abs(psi_w(np.asarray([d2 * t ** 4], dtype=complex)))[0])
        return 4e-16 * (1.0 + omega * (0.5 * abs(lam) * t ** 4 + ps + abs(y) * t ** 2) + 5.0)

    I_rho = _adapt(f, 0.0, rho_max, tol, 1e-12, floor_fn, counter)
    F = -2j * omega * d2 * np.exp(1j * omega * y * y / 2.0) * I_rho
    # Return the raw integral, as every referee here does; F = omega I/(2 pi i).
    return F * (2j * math.pi / omega)


def _general_ctx(components, beta, kappa, gamma1, gamma2):
    """Shared setup: shear-diagonal frame, Gauss centres, constants."""
    gamma, th = _shear_rotation(gamma1, gamma2)
    L0 = 1.0 - kappa - gamma
    L2 = 1.0 - kappa + gamma
    if L0 == 0.0 or L2 == 0.0:
        raise ValueError("critical macro model: L0 or L2 = 0")
    b = _rot(float(beta[0]), float(beta[1]), th)
    comps = []
    for kind, a, bb, params in components:
        a2, b2 = _rot(float(a), float(bb), th)
        comps.append((kind, a2, b2, tuple(float(p) for p in params)))
    c0, c2 = b[0] / L0, b[1] / L2
    Kc = 0.5 * (b[0] ** 2 + b[1] ** 2) - b[0] ** 2 / (2.0 * L0) - b[1] ** 2 / (2.0 * L2)
    return dict(L0=L0, L2=L2, b=b, comps=comps, c0=c0, c2=c2, Kc=Kc,
                amin=min([c0] + [a for _, a, _, _ in comps]),
                amax=max([c0] + [a for _, a, _, _ in comps]),
                bmin=min([c2] + [bb for _, _, bb, _ in comps]),
                bmax=max([c2] + [bb for _, _, bb, _ in comps]))


def _sector_check(ctx, x2, alpha, where):
    """Assert Re w_j > 0 at sampled nodes (the exactness invariant). Raises on
    violation. `x2` may be complex (outer tails)."""
    A = ctx["amin"] - 1.5 - abs(x2.imag)
    B = ctx["amax"] + 1.5 + abs(x2.imag)
    d1 = np.exp(1j * np.copysign(alpha, ctx["L0"]))
    T = math.sqrt(90.0 / abs(ctx["L0"])) + 2.0
    for t in np.linspace(A, B, 7):          # inner core
        for kind, a, bb, params in ctx["comps"]:
            w = (t - a) ** 2 + (x2 - bb) ** 2
            if w.real <= 0.0:
                raise ValueError(f"sector violated ({where}, core): Re w = {w.real:.3e}")
    for X, sgn in [(B, 1.0), (A, -1.0)]:    # inner tails
        for tt in np.linspace(0.0, T, 5):
            z = X + sgn * d1 * tt
            for kind, a, bb, params in ctx["comps"]:
                w = (z - a) ** 2 + (x2 - bb) ** 2
                if w.real <= 0.0:
                    raise ValueError(f"sector violated ({where}, tail): Re w = {w.real:.3e}")


def general_slice_referee(x2, omega, components, beta, kappa=0.0, gamma1=0.0, gamma2=0.0,
                          margin=4.0, alpha=math.pi / 4, tol=1e-14, counter=None,
                          subtract_free=False, check_sector=False):
    """G(x2) = int_R exp(i omega phi) dz0 for the general model (points + macro
    + profiles), by the RC construction in z0.  `x2` may be complex (the outer
    rotated tails of `general_field_referee` evaluate the inner here).
    `subtract_free` integrates e^{i omega phi_free}(e^{-i omega Psi} - 1)
    instead (for the low-omega F - 1 regime; the caller adds back the exact
    `macro_only_integral`).  With `check_sector`, samples the exactness
    invariant Re w_j > 0 along the contour."""
    if not (0.0 < alpha <= 0.25 * math.pi + 1e-15):
        raise ValueError("alpha must be in (0, pi/4]")
    if not omega > 0.0:
        raise ValueError("omega must be > 0")
    ctx = _general_ctx(components, beta, kappa, gamma1, gamma2)
    if check_sector:
        _sector_check(ctx, complex(x2), alpha, "inner")
    L0, L2, c0, c2, Kc = ctx["L0"], ctx["L2"], ctx["c0"], ctx["c2"], ctx["Kc"]
    x2 = complex(x2)
    sh = abs(x2.imag)
    A = ctx["amin"] - margin - sh
    B = ctx["amax"] + margin + sh
    pts = sorted(set([A, B] + [a for _, a, _, _ in ctx["comps"]]))
    span0 = max(abs(A - c0), abs(B - c0))
    dquad = (x2 - c2) ** 2
    psib = 1.0
    for kind, a, bb, params in ctx["comps"]:
        wA = (A - a) ** 2 + (x2 - bb) ** 2
        wB = (B - a) ** 2 + (x2 - bb) ** 2
        psib = psib + float(max(np.abs(_psi_w(kind, params, np.asarray([wA])))[0],
                                np.abs(_psi_w(kind, params, np.asarray([wB])))[0]))
    floor = 4e-16 * (1.0 + omega * (0.5 * abs(L0) * span0 ** 2 + 0.5 * abs(L2) * abs(dquad)
                                    + abs(Kc) + psib + 5.0))

    def psi_sum_core(r, x0, sgn):
        rr = np.asarray(r, dtype=float)
        acc = np.zeros_like(rr, dtype=complex)
        for kind, a, bb, params in ctx["comps"]:
            t = (x0 - a) + sgn * rr
            acc = acc + _psi_w(kind, params, t * t + (x2 - bb) ** 2)
        return acc

    def psi_sum_tail(t, X, sgn, d1):
        tt = np.asarray(t, dtype=float)
        acc = np.zeros_like(tt, dtype=complex)
        for kind, a, bb, params in ctx["comps"]:
            tvec = (X - a) + sgn * d1 * tt
            acc = acc + _psi_w(kind, params, tvec * tvec + (x2 - bb) ** 2)
        return acc

    def _expm1_integrand(ps):
        # e^{-i theta} - 1 with theta = omega*ps, stable for small theta:
        # -2i e^{-i theta/2} sin(theta/2)  (the direct subtraction floors at
        # eps/theta relative -- catastrophic at low omega).
        th = omega * ps
        return -2j * np.exp(-0.5j * th) * np.sin(0.5 * th)

    total = 0j
    for lo, hi in zip(pts[:-1], pts[1:]):
        half = 0.5 * (hi - lo)
        for x0, sgn in [(lo, 1.0), (hi, -1.0)]:
            def f(r, x0=x0, sgn=sgn):
                rr = np.asarray(r, dtype=float)
                quad = 0.5 * L0 * (((x0 - c0) + sgn * rr) ** 2) + 0.5 * L2 * dquad + Kc
                if subtract_free:
                    ps = psi_sum_core(r, x0, sgn)
                    return np.exp(1j * omega * quad) * _expm1_integrand(ps)
                return np.exp(1j * omega * (quad - psi_sum_core(r, x0, sgn)))
            total += _adapt(f, 0.0, half, tol, 1e-15, floor, counter)
    d1 = np.exp(1j * np.copysign(alpha, L0))
    T = math.sqrt(90.0 / (omega * abs(L0))) + abs(ctx["b"][0]) / (abs(L0) * math.cos(alpha)) + 2.0
    for X, sgn in [(B, 1.0), (A, -1.0)]:
        def f(t, X=X, sgn=sgn):
            tt = np.asarray(t, dtype=float)
            zc = (X - c0) + sgn * d1 * tt
            quad = 0.5 * L0 * zc * zc + 0.5 * L2 * dquad + Kc
            if subtract_free:
                ps = psi_sum_tail(t, X, sgn, d1)
                return d1 * np.exp(1j * omega * quad) * _expm1_integrand(ps)
            return d1 * np.exp(1j * omega * (quad - psi_sum_tail(t, X, sgn, d1)))
        total += _adapt(f, 0.0, T, tol, 1e-14, floor, counter)
    return total


def general_field_referee(omega, components, beta, kappa=0.0, gamma1=0.0, gamma2=0.0,
                          margin=4.0, margin_outer=1.5, alpha=math.pi / 4, tol=1e-13,
                          counter=None, subtract_free=False, check_sector=False):
    """I = int_R G(x2) dx2 for the general model: the RC construction in both
    variables (real core with anchored halves and breakpoints at every pinch
    height b_j and Gaussian centre c2, plus sign(L2)-rotated tails on which the
    inner runs at complex x2).  Returns the raw integral; F = omega I/(2 pi i).
    `subtract_free` adds back the exact lensless `macro_only_integral`; with
    kappa = gamma = 0 and point components this must reproduce `field_referee`
    (the self-test checks <= 1e-13)."""
    if not (0.0 < alpha <= 0.25 * math.pi + 1e-15):
        raise ValueError("alpha must be in (0, pi/4]")
    if not omega > 0.0:
        raise ValueError("omega must be > 0")
    ctx = _general_ctx(components, beta, kappa, gamma1, gamma2)
    if check_sector:
        # Sample the inner contour at several outer positions, including the
        # outer tails (where x2 is complex) -- that is where the mixed-sign
        # saddle case could in principle cross a cut.
        L0, L2 = ctx["L0"], ctx["L2"]
        c2 = ctx["c2"]
        A2 = ctx["bmin"] - margin_outer
        B2 = ctx["bmax"] + margin_outer
        S = math.sqrt(90.0 / (omega * abs(L2))) + 2.0
        d2 = np.exp(1j * np.copysign(alpha, L2))
        for x2 in [complex(A2), complex(B2), complex(c2),
                   complex(B2 + d2 * 0.5), complex(B2 + d2 * 2.0),
                   complex(A2 - d2 * 0.5), complex(A2 - d2 * 2.0)]:
            _sector_check(ctx, x2, alpha, "outer")
    L0, L2, c2, Kc = ctx["L0"], ctx["L2"], ctx["c2"], ctx["Kc"]
    A2 = ctx["bmin"] - margin_outer
    B2 = ctx["bmax"] + margin_outer
    pts = sorted(set([A2, B2] + [bb for _, _, bb, _ in ctx["comps"]]))
    span2 = max(abs(A2 - c2), abs(B2 - c2))
    S = math.sqrt(90.0 / (omega * abs(L2))) + abs(ctx["b"][1]) / (abs(L2) * math.cos(alpha)) + 2.0
    span0_max = max(abs(ctx["amin"] - 1.5 - S * math.sin(alpha) - ctx["c0"]),
                    abs(ctx["amax"] + 1.5 + S * math.sin(alpha) - ctx["c0"]), 1.0)
    floor = 4e-16 * (1.0 + omega * (0.5 * abs(L0) * span0_max ** 2 + 0.5 * abs(L2) * span2 ** 2
                                    + abs(Kc) + 10.0))

    def G(x2):
        return general_slice_referee(x2, omega, components, beta, kappa, gamma1, gamma2,
                                     margin=margin, alpha=alpha, tol=tol,
                                     counter=counter, subtract_free=subtract_free)

    total = 0j
    if subtract_free:
        total += macro_only_integral(omega, beta, kappa, gamma1, gamma2)
    for lo, hi in zip(pts[:-1], pts[1:]):
        rmax = math.sqrt(0.5 * (hi - lo))
        for x0, sgn in [(lo, 1.0), (hi, -1.0)]:
            def f(rho, x0=x0, sgn=sgn):
                q = np.asarray(rho, dtype=float)
                return np.array([G(x0 + sgn * qq * qq) * 2.0 * qq for qq in q])
            total += _adapt(f, 0.0, rmax, tol, 1e-12, floor)
    d2 = np.exp(1j * np.copysign(alpha, L2))
    for X, sgn in [(B2, 1.0), (A2, -1.0)]:
        def f(s, X=X, sgn=sgn):
            q = np.asarray(s, dtype=float)
            return np.array([d2 * G(X + sgn * d2 * qq) for qq in q])
        total += _adapt(f, 0.0, S, tol, 1e-12, floor)
    return total


def make_psi_w_mp(components):
    """mpmath version of `make_psi_w` (dps whatever is active; robust forms:
    the NFW branch uses ln((1-s)/2) ln((1+s)/2) with log1p, never the
    cancelling (1/4)ln^2 - atanh^2 difference, and never mp.atanh near 1)."""
    def psi_w(w):
        w = mp.mpc(w)
        acc = mp.mpc(0)
        for kind, params in components:
            if kind == "point":
                m, xc = params
                acc += mp.mpf(m) / 2 * mp.log(w + xc * xc)
            elif kind == "sis":
                acc += mp.mpf(params[0]) * mp.sqrt(w)
            elif kind == "cis":
                psi0, xc = params
                if xc == 0.0:
                    acc += mp.mpf(psi0) * mp.sqrt(w)
                else:
                    s = mp.sqrt(w + xc * xc)
                    acc += mp.mpf(psi0) * (s + xc * mp.log(2 * xc / (s + xc)))
            elif kind == "nfw":
                psi0, xs = params
                u2 = w / (xs * xs)
                s = mp.sqrt(1 - u2)
                first = mp.log(u2 / (2 * (1 + s)))
                half = mp.log1p(-u2 / (2 * (1 + s)))
                acc += mp.mpf(psi0) / 2 * first * half
            else:
                raise ValueError(f"unknown profile kind {kind!r}")
        return acc
    return psi_w


_GL_RULE_MP_CACHE = {}


def _gl_rule_mp(n=20):
    """GL(n) nodes/weights at the *current* mp.dps (numpy's double-precision
    rule caps any mp quadrature at ~1e-16 relative — the reason the first hp
    attempt hung refining forever: its whole-vs-halves differences could never
    fall below 1e-28).  Newton on P_n with an mp recurrence; cached per dps."""
    dps = mp.mp.dps
    key = (n, dps)
    if key in _GL_RULE_MP_CACHE:
        return _GL_RULE_MP_CACHE[key]
    nodes, weights = [], []
    for i in range(1, n + 1):
        x = mp.cos(mp.pi * (mp.mpf(i) - mp.mpf("0.25")) / (n + mp.mpf("0.5")))
        for _ in range(200):
            p_prev, p_cur = mp.mpf(1), x
            for k in range(2, n + 1):
                p_prev, p_cur = p_cur, ((2 * k - 1) * x * p_cur - (k - 1) * p_prev) / k
            dp = n * (x * p_cur - p_prev) / (x * x - 1)
            dx = p_cur / dp
            x = x - dx
            if abs(dx) <= mp.mpf(10) ** (-(dps + 5)):
                break
        p_prev, p_cur = mp.mpf(1), x
        for k in range(2, n + 1):
            p_prev, p_cur = p_cur, ((2 * k - 1) * x * p_cur - (k - 1) * p_prev) / k
        dp = n * (x * p_cur - p_prev) / (x * x - 1)
        nodes.append(x)
        weights.append(2 / ((1 - x * x) * dp * dp))
    _GL_RULE_MP_CACHE[key] = (nodes, weights)
    return nodes, weights


def _gl20_mp(f, a, b):
    """GL20 of an mp-scalar -> mp-complex function on [a, b], with the
    mp-precision rule (no |f| mass returned)."""
    nodes, weights = _gl_rule_mp()
    c, h = (a + b) / 2, (b - a) / 2
    s = mp.mpc(0)
    for x, w in zip(nodes, weights):
        s += w * f(c + h * x)
    return h * s


def _adapt_mp(f, a, b, tol, minlen, max_depth=80):
    """Adaptive GL20 in mp arithmetic: accept when whole vs halves agree to
    `tol` in absolute value or the panel is shorter than `minlen`."""
    whole = _gl20_mp(f, a, b)
    value = mp.mpc(0)
    stack = [(a, b, whole, 0)]
    while stack:
        lo, hi, whole, depth = stack.pop()
        mid = (lo + hi) / 2
        left, right = _gl20_mp(f, lo, mid), _gl20_mp(f, mid, hi)
        if abs(left + right - whole) <= tol * (hi - lo) or hi - lo < minlen or depth > max_depth:
            value += left + right
        else:
            stack.append((lo, mid, left, depth + 1))
            stack.append((mid, hi, right, depth + 1))
    return value


def hankel_referee_hp(omega, y, psi_w_mp, kappa=0.0, tol_rel=1e-28, dps=45):
    """High-precision `hankel_referee` (same polar construction, mpmath
    arithmetic): the authoritative referee for profiles / large omega*y, where
    the float version's ~eps*(mass/|I|) accumulation reaches 1e-8.  `psi_w_mp`
    from `make_psi_w_mp`.  Returns the raw integral I (F = omega I/(2 pi i));
    the float `hankel_referee` and this one share only the *method*, so
    agreement between them (within the float one's floor) checks the mp code,
    while the point-mass ND / macro-only closed forms are the external anchors."""
    with mp.workdps(dps):
        w, yv, lam = mp.mpf(omega), mp.mpf(y), 1 - mp.mpf(kappa)
        if lam == 0:
            raise ValueError("critical macro model: lambda = 0")
        sg = 1 if lam > 0 else -1
        d = mp.e ** (mp.mpc(0, 1) * sg * mp.pi / 4)
        d2 = d * d
        a = lam * d2 / 2
        rho_max = mp.sqrt(2 * abs(yv) / abs(lam) + 2)
        rho_max = max(rho_max, (mp.mpf(90) / (mp.mpf(omega) * abs(lam))) ** mp.mpf("0.25"))

        def f(rho):
            wq = d2 * rho ** 4
            return (rho ** 3 * mp.besselj(0, mp.mpf(omega) * yv * d * rho ** 2)
                    * mp.e ** (mp.mpc(0, 1) * mp.mpf(omega) * (a * rho ** 4 - psi_w_mp(wq))))

        # decay probe (mp is slow, so probe coarsely)
        for _ in range(12):
            if abs(f(rho_max)) <= mp.mpf("1e-30") * max(abs(f(rho_max / 2)), mp.mpf("1e-300")):
                break
            rho_max *= mp.mpf("1.5")
        if rho_max > 1e5:
            raise RuntimeError("hankel_referee_hp: tail decay probe failed")
        I_rho = _adapt_mp(f, mp.mpf(0), rho_max, tol_rel, mp.mpf("1e-16"))
        F = -2j * mp.mpf(omega) * d2 * mp.e ** (mp.mpc(0, 1) * mp.mpf(omega) * yv * yv / 2) * I_rho
        return complex(F * (2j * mp.pi / mp.mpf(omega)))


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
