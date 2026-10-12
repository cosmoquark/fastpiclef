"""P4.1: derive and validate the 2D stationary-phase next-order correction.

Model: I(w) = int d^2u exp{i w phi(u)} with phi expanded at a stationary point
(phi0 + 1/2 u.H.u + 1/6 phi3.u^3 + 1/24 phi4.u^4).  Expanding the cubic and
quartic terms to the order that contributes O(1/w) after Gaussian integration
(Isserlis moments) gives

    I = e^{i w phi0} G0 [1 + (1/(i w)) C + O(w^-2)],
    G0 = (2 pi / (w sqrt|det H|)) e^{i (pi/4) sigma},
    C  = (1/8) H^-1_ab H^-1_cd phi4_abcd - (1/72) Sum_15,

with Sum_15 the 15 pairings of phi3_abc phi3_def H^-1 H^-1 H^-1.

This script (a) verifies the 1D reduction C = phi''''/(8 phi''^2)
- 5 phi'''^2/(24 phi''^3) (sympy), (b) checks the correction on the Airy
integral (exact closed form: the residual must fall like w^-2 instead of w^-1),
and (c) checks it on the point mass against the Nakamura-Deguchi form.

Run: `pixi run -e comparison python -m comparison.sp_correction`
"""

import os

os.environ.setdefault("RAYON_NUM_THREADS", "1")

import math

import numpy as np

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def derive_1d_check():
    """sympy: the general 2D formula reduced to 1D must give the textbook form."""
    try:
        import sympy as sp
    except ImportError:
        print("  (sympy unavailable: skipping the symbolic 1D reduction check)")
        return None

    h = sp.symbols("h", positive=True)
    p3 = sp.symbols("p3")
    p4 = sp.symbols("p4")
    # 1D: H = h, phi3 = p3, phi4 = p4.  H^-1 = 1/h.
    # Sum_15 for the 1D cubic-squared: <u^6> = 15 ((i/w)/h)^3 -> the pairing sum
    # contributes 15 * p3^2 / h^3 (the coefficient multiplying (i/w)^3 G0).
    C = sp.Rational(1, 8) * p4 / h**2 - sp.Rational(1, 72) * 15 * p3**2 / h**3
    textbook = p4 / (8 * h**2) - 5 * p3**2 / (24 * h**3)
    diff = sp.simplify(C - textbook)
    print(f"  1D reduction: C = {sp.simplify(C)}; textbook {textbook}; diff {diff}")
    assert diff == 0
    return True


def _airy_exact(y, w):
    from scipy.special import airy

    return 2.0 * math.pi * w ** (-1.0 / 3.0) * airy(-(w ** (2.0 / 3.0)) * y)[0]


def airy_check(y=1.0, ws=(10.0, 30.0, 100.0, 300.0, 1000.0)):
    """Airy (1D, exact): geometric vs geometric+correction residuals."""
    print("  Airy y=1: residual scaling (geo vs geo+corr)")
    out = []
    for w in ws:
        geo = 0.0
        corr = 0.0
        for u in (math.sqrt(y), -math.sqrt(y)):
            phi0 = u**3 / 3 - y * u
            phi2 = 2 * u
            phi3 = 2.0
            phi4 = 0.0
            G0 = (math.sqrt(2 * math.pi / (w * abs(phi2)))
                  * np.exp(1j * math.pi / 4 * (1 if phi2 > 0 else -1)))
            g = G0 * np.exp(1j * w * phi0)
            C = phi4 / (8 * phi2**2) - 5 * phi3**2 / (24 * phi2**3)
            geo += g
            corr += g * (1 + C / (1j * w))
        exact = _airy_exact(y, w)
        eg = abs(geo - exact) / abs(exact)
        ec = abs(corr - exact) / abs(exact)
        out.append((w, eg, ec))
        print(f"    w={w:6.0f}: geo {eg:.2e}  geo+corr {ec:.2e}")
    return out


def _sum15(phi3, Hinv):
    """The 15-pairing sum: pair the six indices of phi3 x phi3 into three
    H^-1 contractions (Isserlis).  Direct enumeration (15 x 2^6 terms)."""
    import itertools

    def matchings(items):
        if not items:
            yield []
            return
        a, rest = items[0], items[1:]
        for i, b in enumerate(rest):
            for m in matchings(rest[:i] + rest[i + 1:]):
                yield [(a, b)] + m

    total = 0.0
    for M in matchings(list(range(6))):
        for idx in itertools.product(range(2), repeat=6):
            term = phi3[idx[0], idx[1], idx[2]] * phi3[idx[3], idx[4], idx[5]]
            for (p, q) in M:
                term *= Hinv[idx[p], idx[q]]
            total += term
    return total


def _fd_tensor_check(x, y):
    """Cross-check the analytic H/phi3/phi4 at (x, 0) by finite differences."""
    h = 1e-3

    def phi(a, b):
        return 0.5 * ((a - y) ** 2 + b ** 2) - 0.5 * math.log(a * a + b * b)

    def d2(a, b, i, j):
        if i == j:
            return (phi(a + h, b) - 2 * phi(a, b) + phi(a - h, b)) / h ** 2 if i == 0 \
                else (phi(a, b + h) - 2 * phi(a, b) + phi(a, b - h)) / h ** 2
        return (phi(a + h, b + h) - phi(a + h, b - h) - phi(a - h, b + h)
                + phi(a - h, b - h)) / (4 * h ** 2)

    def d3(a, b, i, j, k):
        # mixed finite difference of order 3 (i,j,k in {0,1})
        eps = 1e-2
        def shift(a2, b2, idx, s):
            return (a2 + s * eps, b2) if idx == 0 else (a2, b2 + s * eps)
        # fall back to a simple combination for the components we use
        vals = {}
        for idx, s in [((i, j, k), 0)]:
            pass
        # direct formulas are easier: use central differences per axis
        def d1(a2, b2, idx):
            return ((phi(*shift(a2, b2, idx, 1)) - phi(*shift(a2, b2, idx, -1)))
                    / (2 * eps))
        def dd1(a2, b2, idx, jdx):
            return (d1(*shift(a2, b2, jdx, 1), idx) - d1(*shift(a2, b2, jdx, -1), idx)) / (2 * eps)
        return (dd1(*shift(a, b, k, 1), i, j) - dd1(*shift(a, b, k, -1), i, j)) / (2 * eps)

    print("    FD tensor check at x=%s:" % x)
    for (i, j) in [(0, 0), (0, 1), (1, 1)]:
        print(f"      d2[{i}{j}]: FD {d2(x, 0.0, i, j):+.6f} vs analytic "
              f"{[1.0 + 1.0 / x ** 2, 0.0, 1.0 - 1.0 / x ** 2][{(0, 0): 0, (0, 1): 1, (1, 1): 2}[(i, j)]]:+.6f}")
    for (i, j, k), an in [((0, 0, 0), -2.0 / x ** 3), ((0, 1, 1), 2.0 / x ** 3),
                          ((1, 1, 1), 0.0)]:
        print(f"      d3[{i}{j}{k}]: FD {d3(x, 0.0, i, j, k):+.4f} vs analytic {an:+.4f}")


def point_mass_2d_check(ys=(0.1, 0.5), ws=(10.0, 30.0, 100.0, 300.0)):
    """Point mass (2D, exact ND): per-image Hessian + the 15-pairing correction;
    the residual must fall like w^-2 once the correction is included."""
    from .referee import point_mass_exact

    def derivatives(x):
        """phi = (x1-y)^2/2 + x2^2/2 - ln sqrt(x1^2+x2^2) at (x, 0)."""
        H = np.diag([1.0 + 1.0 / x**2, 1.0 - 1.0 / x**2])
        phi3 = np.zeros((2, 2, 2))
        phi3[0, 0, 0] = -2.0 / x**3
        phi3[0, 1, 1] = phi3[1, 0, 1] = phi3[1, 1, 0] = 2.0 / x**3
        phi4 = np.zeros((2, 2, 2, 2))
        phi4[0, 0, 0, 0] = 6.0 / x**4
        for p in {(0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1),
                  (1, 0, 1, 0), (1, 1, 0, 0)}:
            phi4[p] = -6.0 / x**4
        phi4[1, 1, 1, 1] = 6.0 / x**4  # FD-verified (was 4/x^4 by hand: wrong)
        return H, phi3, phi4

    print("  point mass (2D): residual scaling (geo vs geo+corr) vs ND")
    _fd_tensor_check(1.0512, 0.1)
    out = []
    for y in ys:
        xm = 0.5 * (y + math.sqrt(y * y + 4.0))
        xs = 0.5 * (y - math.sqrt(y * y + 4.0))
        for w in ws:
            F_geo = 0.0
            F_corr = 0.0
            for x in (xm, xs):
                H, phi3, phi4 = derivatives(x)
                det = float(np.linalg.det(H))
                Hinv = np.linalg.inv(H)
                # Morse signature: the sum of the Hessian eigenvalue signs
                # (2 = minimum, 0 = saddle, -2 = maximum) -- NOT sign(det).
                ev = np.linalg.eigvalsh(H)
                sigma = int(np.sign(ev[0]) + np.sign(ev[1]))
                phi0 = 0.5 * (x - y) ** 2 - math.log(abs(x))
                G0 = (1.0 / (1j * math.sqrt(abs(det)))) * np.exp(
                    1j * math.pi / 4 * sigma) * np.exp(1j * w * phi0)
                C = (0.125 * np.einsum("ab,cd,abcd->", Hinv, Hinv, phi4)
                     - _sum15(phi3, Hinv) / 72.0)
                F_geo += G0
                F_corr += G0 * (1.0 + C / (1j * w))
            # |F| comparison: the ND closed form carries a constant
            # Fermat-offset phase relative to our convention (AGENTS note).
            exact = abs(point_mass_exact(w, y))
            eg = abs(abs(F_geo) - exact) / exact
            ec = abs(abs(F_corr) - exact) / exact
            # error estimate for the hybrid: |C|/w times the geometric modulus
            # (a safety factor is applied downstream; here raw)
            est = 0.0
            for x in (xm, xs):
                H, phi3, phi4 = derivatives(x)
                Hinv = np.linalg.inv(H)
                C = (0.125 * np.einsum("ab,cd,abcd->", Hinv, Hinv, phi4)
                     - _sum15(phi3, Hinv) / 72.0)
                est += abs(C) / w
            out.append((y, w, eg, ec, est))
            cover = est / max(ec, 1e-300)
            print(f"    y={y} w={w:6.0f}: geo {eg:.2e}  geo+corr {ec:.2e}  "
                  f"est {est:.2e}  cover {cover:6.2f}x")
    return out


if __name__ == "__main__":
    print("== 1D reduction (sympy) ==")
    derive_1d_check()
    print("== Airy ==")
    airy_check()
    print("== point mass (radial reduction) ==")
    point_mass_2d_check()
