"""P0 experiments for the batched/general-lens-solver plan (see
`/home/paul/.opencode/plan/batched-general-lens-solver.md`, section 2).

Runs the planning experiments *before* any Rust work:

  E1  source-map batching (1D rational grid; 2D point-mass-field grid):
      accuracy vs the per-point production solver / validated referees, and
      the node-count inflation of a shared contour;
  E2  frequency batching (single mass, two-mass field; log bands; Chebyshev
      interpolation as the alternative) + low-omega tests (E3);
  E4  macro model (kappa, gamma) sanity sweeps (alpha family, margins,
      transposed slicing, sector checks); closed-form anchors are already in
      the referee self-tests;
  E5  halo profiles (SIS/CIS/NFW) vs the high-precision polar (Hankel) referee,
      real-axis conventions vs FIONA's formulas, analyticity loops, and the
      outer-integrand panel counts near the pinch heights.

Usage (from the repo root, comparison environment):
    python -m comparison.p0_experiments E1 E2 E3 E4 E5
or a subset.  Results are appended to comparison/results/p0_*.csv (17
significant digits for anything graded below 1e-10) and printed.

This module is *exploratory*: the Rust ports (P1/P2) get their own tests with
hard-coded references emitted from here.
"""

import csv
import math
import os
import sys
import time

import numpy as np

from .referee import (  # noqa: E402
    TWO_MASS_BETA, TWO_MASS_LENSES, _X20, _W20, _gl_rule_mp, _psi_w,
    general_field_referee, general_slice_referee, hankel_referee_hp,
    make_psi_w_mp, macro_only_integral, mass_sheet_nd, point_components,
    point_mass_exact, slice_raw_from_nd,
)

RESULTS = os.path.join(os.path.dirname(__file__), "results")
E4 = np.exp(1j * np.pi / 4)


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def _append_csv(name, header, rows):
    path = os.path.join(RESULTS, name)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(header)
        w.writerows(rows)
    print(f"    -> {path} (+{len(rows)} rows)")


# ---------------------------------------------------------------------------
# Vector-valued adaptive GL20 (the prototype of the planned Rust
# `adaptive::integrate_vec`): the acceptance rule must hold for *every*
# batch component.
# ---------------------------------------------------------------------------

def _gl20_vec(f, a, b, counter=None):
    c, h = 0.5 * (a + b), 0.5 * (b - a)
    v = f(c + h * _X20)                      # (20, M)
    v = np.where(np.isfinite(v), v, 0.0)
    if counter is not None:
        counter[0] += 20
    return (h * np.sum(_W20[:, None] * v, axis=0),
            abs(h) * np.sum(_W20[:, None] * np.abs(v), axis=0))


def _adapt_vec(f, a, b, tol, minlen, floor, counter=None):
    whole, _ = _gl20_vec(f, a, b, counter)
    total = np.zeros_like(whole)
    stack = [(a, b, whole)]
    while stack:
        lo, hi, whole = stack.pop()
        mid = 0.5 * (lo + hi)
        (l, al), (r, ar) = _gl20_vec(f, lo, mid, counter), _gl20_vec(f, mid, hi, counter)
        diff = np.abs(l + r - whole)
        if np.all(diff <= tol * (hi - lo)) or np.all(diff <= floor * (al + ar)) or hi - lo < minlen:
            total += l + r
        else:
            stack.append((lo, mid, l))
            stack.append((mid, hi, r))
    return total


# ---------------------------------------------------------------------------
# E1a: 1D rational lens, shared-contour batch on the P-LPI source grid.
#     phi = (z-beta)^2/2 - kappa/(1+z^2)   [RationalLens; poles at +-i]
# ---------------------------------------------------------------------------

def _rc_1d_batch(betas, omega, kappa, *, A=None, B=None, margin=4.0,
                 tol=1e-14, counter=None):
    """Batched RC for the rational lens: one shared contour for all betas,
    vectorized over betas.  If A/B are None, the contour spans just these
    betas (independent-solve mode)."""
    betas = np.asarray(betas, dtype=float)
    M = betas.size
    if A is None:
        A = min(0.0, betas.min()) - margin
    if B is None:
        B = max(0.0, betas.max()) + margin
    pts = sorted(set([A, B, 0.0]))
    phimax = (0.5 * max(abs(A), abs(B)) ** 2 + 0.5 * betas.max() ** 2
              + abs(kappa) + 5.0)
    floor = 4e-16 * (1.0 + omega * phimax)
    total = np.zeros(M, dtype=complex)

    def core_f(r, x0, sgn):
        rr = np.asarray(r, dtype=float)
        z = x0 + sgn * rr                                          # (20,)
        S = np.exp(1j * omega * (0.5 * z * z - kappa / (1.0 + z * z)))  # (20,)
        T = np.exp(-1j * omega * np.outer(z, betas)) * np.exp(0.5j * omega * betas ** 2)
        return S[:, None] * T

    for lo, hi in zip(pts[:-1], pts[1:]):
        half = 0.5 * (hi - lo)
        for x0, sgn in [(lo, 1.0), (hi, -1.0)]:
            total += _adapt_vec(lambda r, x0=x0, sgn=sgn: core_f(r, x0, sgn),
                                0.0, half, tol, 1e-15, floor, counter)

    T_len = (math.sqrt(90.0 / omega) + abs(betas).max() / math.cos(math.pi / 4) + 2.0)
    for X, sgn in [(B, 1.0), (A, -1.0)]:
        def tail_f(t, X=X, sgn=sgn):
            tt = np.asarray(t, dtype=float)
            z = X + sgn * E4 * tt
            S = np.exp(1j * omega * (0.5 * z * z - kappa / (1.0 + z * z)))
            T = np.exp(-1j * omega * np.outer(z, betas)) * np.exp(0.5j * omega * betas ** 2)
            return E4 * S[:, None] * T
        total += _adapt_vec(tail_f, 0.0, T_len, tol, 1e-14, floor, counter)
    return total


def e1a(n_mu=512, omega=10.0, kappa=-2.0):
    print("E1a: 1D rational shared-contour batch (P-LPI grid)")
    mus = np.linspace(-4.0, 4.0, n_mu)
    # truth: the production solver (complete-saddle thimble, machine precision)
    import fastpiclef as fpl
    F_prod = np.empty(n_mu, dtype=complex)
    t0 = time.time()
    for i, mu in enumerate(mus):
        F_prod[i] = fpl.RationalLens(beta=float(mu), kappa=kappa).solve(omega).integral
    t_prod = time.time() - t0
    print(f"    production thimble truth: {n_mu} solves in {t_prod:.2f} s "
          f"({t_prod / n_mu * 1e3:.3f} ms/point)")

    c_shared = [0]
    t0 = time.time()
    v_shared = _rc_1d_batch(mus, omega, kappa, counter=c_shared)
    t_shared = time.time() - t0
    err_shared = np.abs(v_shared - F_prod) / np.abs(F_prod)

    c_ind = [0]
    v_ind = np.empty(n_mu, dtype=complex)
    t0 = time.time()
    for i, mu in enumerate(mus):
        v_ind[i] = _rc_1d_batch(mus[i:i + 1], omega, kappa, counter=c_ind)[0]
    t_ind = time.time() - t0
    err_ind = np.abs(v_ind - F_prod) / np.abs(F_prod)

    nodes_shared = c_shared[0] // 20
    nodes_ind = c_ind[0] // 20
    print(f"    shared  batch: max rel err {err_shared.max():.2e}  nodes {nodes_shared} "
          f"({nodes_shared / n_mu:.1f}/point)  python wall {t_shared:.2f} s")
    print(f"    independent RC: max rel err {err_ind.max():.2e}  nodes {nodes_ind} "
          f"({nodes_ind / n_mu:.1f}/point)  python wall {t_ind:.2f} s")
    cross = np.max(np.abs(v_shared - v_ind) / np.abs(v_ind))
    print(f"    cross-check (shared vs per-point RC): max rel {cross:.2e}")

    rows = [[f"{omega:.1f}", str(n_mu),
             f"{err_shared.max():.17e}", f"{err_ind.max():.17e}",
             str(nodes_shared), str(nodes_ind), f"{t_shared:.3f}", f"{t_ind:.3f}",
             f"{t_prod:.3f}"]]
    _append_csv("p0_batch_source.csv",
                ["experiment", "omega", "n_points", "err_shared_max", "err_independent_max",
                 "nodes_shared", "nodes_independent", "wall_shared_s", "wall_ind_s",
                 "wall_production_s"],
                [["1d-grid"] + rows[0]])


# ---------------------------------------------------------------------------
# E1b: 2D two-mass field, shared-contour batch over a source grid.
# ---------------------------------------------------------------------------

def _inner_vec(x2, beta0s, omega, lenses, a_lo, a_hi, margin, tol, counter):
    """Inner G(x2; beta0) vectorized over beta0s, with the core spanning every
    branch-point abscissa over the *whole grid* -- and widened by |Im x2| for
    complex slice positions (the outer tails), exactly as the validated
    single-point referee does; without the widening a branch point leaves the
    core and the contour crosses to the wrong sheet."""
    M0 = beta0s.size
    x2 = complex(x2)
    sh = abs(x2.imag)
    A = min(a_lo, beta0s.min()) - margin - sh
    B = max(a_hi, beta0s.max()) + margin + sh
    pts = sorted(set([A, B] + [a for a, _, _ in lenses]))
    span = max(abs(A), abs(B))
    floor = 4e-16 * (1.0 + omega * (0.5 * span ** 2 + 0.5 * abs(x2) ** 2 + 5.0))
    total = np.zeros(M0, dtype=complex)

    def core_f(r, x0, sgn):
        rr = np.asarray(r, dtype=float)
        z = x0 + sgn * rr
        S = np.exp(1j * omega * 0.5 * z * z)
        for a, b, m in lenses:
            t = (x0 - a) + sgn * rr
            S = S * np.exp(-0.5j * omega * m * np.log(t * t + (x2 - b) ** 2))
        T = np.exp(-1j * omega * np.outer(z, beta0s)) * np.exp(0.5j * omega * beta0s ** 2)
        return S[:, None] * T

    for lo, hi in zip(pts[:-1], pts[1:]):
        half = 0.5 * (hi - lo)
        for x0, sgn in [(lo, 1.0), (hi, -1.0)]:
            total += _adapt_vec(lambda r, x0=x0, sgn=sgn: core_f(r, x0, sgn),
                                0.0, half, tol, 1e-15, floor, counter)
    T_len = math.sqrt(90.0 / omega) + abs(beta0s).max() / math.cos(math.pi / 4) + 2.0
    for X, sgn in [(B, 1.0), (A, -1.0)]:
        def tail_f(t, X=X, sgn=sgn):
            tt = np.asarray(t, dtype=float)
            z = X + sgn * E4 * tt
            S = np.exp(1j * omega * 0.5 * z * z)
            for a, b, m in lenses:
                w = (z - a) ** 2 + (x2 - b) ** 2
                S = S * np.exp(-0.5j * omega * m * np.log(w))
            T = np.exp(-1j * omega * np.outer(z, beta0s)) * np.exp(0.5j * omega * beta0s ** 2)
            return E4 * S[:, None] * T
        total += _adapt_vec(tail_f, 0.0, T_len, tol, 1e-14, floor, counter)
    return total


def _rc_2d_batch(beta0s, beta1s, omega, lenses, *, shared=True, margin=4.0,
                 margin_outer=1.5, tol=1e-13, counter=None):
    """2D RC for the point-mass field, vectorized over a (beta0, beta1) grid.
    `shared=True`: one contour spanning the whole grid; `shared=False`: the
    contour spans only the single beta (independent-solve mode)."""
    beta0s = np.asarray(beta0s, dtype=float)
    beta1s = np.asarray(beta1s, dtype=float)
    a_lo = min([beta0s.min()] + [a for a, _, _ in lenses])
    a_hi = max([beta0s.max()] + [a for a, _, _ in lenses])
    b_lo = min([beta1s.min()] + [b for _, b, _ in lenses])
    b_hi = max([beta1s.max()] + [b for _, b, _ in lenses])
    A2 = b_lo - margin_outer
    B2 = b_hi + margin_outer
    pts2 = sorted(set([A2, B2] + [b for _, b, _ in lenses]))
    span2 = max(abs(A2), abs(B2))
    floor = 4e-16 * (1.0 + omega * (0.5 * span2 ** 2 + 10.0))
    M0, M1 = beta0s.size, beta1s.size
    M = M0 * M1

    def G(x2):
        return _inner_vec(x2, beta0s, omega, lenses, a_lo, a_hi, margin, tol, counter)

    def flatten(x2, g):
        outer = np.exp(1j * omega * (0.5 * x2 * x2
                                     - np.outer(np.ones(M0), beta1s) * x2))
        outer = outer * np.exp(0.5j * omega * beta1s ** 2)[None, :]
        return (g[:, None] * outer).reshape(M)

    total = np.zeros(M, dtype=complex)
    for lo, hi in zip(pts2[:-1], pts2[1:]):
        rmax = math.sqrt(0.5 * (hi - lo))
        for x0, sgn in [(lo, 1.0), (hi, -1.0)]:
            def f(rho, x0=x0, sgn=sgn):
                q = np.asarray(rho, dtype=float)
                return np.array([flatten(x0 + sgn * qq * qq, G(x0 + sgn * qq * qq))
                                 for qq in q]) * (2.0 * np.asarray(rho, dtype=float))[:, None]
            total += _adapt_vec(f, 0.0, rmax, tol, 1e-12, floor, counter)
    S_len = math.sqrt(90.0 / omega) + abs(beta1s).max() / math.cos(math.pi / 4) + 2.0
    for X, sgn in [(B2, 1.0), (A2, -1.0)]:
        def f(s, X=X, sgn=sgn):
            q = np.asarray(s, dtype=float)
            # +E4 from the Jacobian·orientation bookkeeping (both outer tails),
            # exactly as in the validated single-point construction.
            return np.array([E4 * flatten(X + sgn * E4 * qq, G(X + sgn * E4 * qq))
                             for qq in q])
        total += _adapt_vec(f, 0.0, S_len, tol, 1e-12, floor, counter)
    return total.reshape(M0, M1)


def e1b(n_side=6, omega=3.0, n_referee=4, ind_stride=2):
    print("E1b: 2D two-mass field, shared-contour source-grid batch")
    grid = np.linspace(-1.0, 1.0, n_side)
    c = [0]
    t0 = time.time()
    v = _rc_2d_batch(grid, grid, omega, TWO_MASS_LENSES, counter=c)
    t_vec = time.time() - t0
    nodes_vec = c[0] // 20
    print(f"    vectorized batch done ({time.time() - t0:.1f} s)")

    ci = [0]
    v_ind = np.empty((n_side // ind_stride, n_side // ind_stride), dtype=complex)
    t0 = time.time()
    for i in range(0, n_side, ind_stride):
        for j in range(0, n_side, ind_stride):
            v_ind[i // ind_stride, j // ind_stride] = _rc_2d_batch(
                np.array([grid[i]]), np.array([grid[j]]), omega,
                TWO_MASS_LENSES, counter=ci)[0, 0]
    t_ind = time.time() - t0
    nodes_ind = ci[0] // 20
    n_sub = v_ind.size
    print(f"    vectorized batch: max|.| {np.abs(v).max():.3f}  gl20-calls {nodes_vec} "
          f"({nodes_vec / v.size:.1f}/point)  wall {t_vec:.1f} s")
    print(f"    independent RC  : gl20-calls {nodes_ind} ({nodes_ind / n_sub:.1f}/point)  "
          f"wall {t_ind:.1f} s")
    worst_cross = 0.0
    for i in range(0, n_side, ind_stride):
        for j in range(0, n_side, ind_stride):
            worst_cross = max(worst_cross, rel(v[i, j],
                                               v_ind[i // ind_stride, j // ind_stride]))
    print(f"    shared vs independent (Cauchy, different contours): max rel {worst_cross:.2e}")

    # independent referee (different construction: margins 4.0/1.5, per point)
    rng = np.random.default_rng(3)
    idx = rng.choice(n_side * n_side, size=n_referee, replace=False)
    worst = 0.0
    for k in idx:
        i, j = divmod(int(k), n_side)
        I = general_field_referee(omega, point_components(TWO_MASS_LENSES),
                                  (float(grid[i]), float(grid[j])))
        worst = max(worst, rel(v[i, j], I))
    print(f"    vectorized vs independent referee ({n_referee} pts): max rel {worst:.2e}")
    _append_csv("p0_batch_source.csv",
                ["experiment", "omega", "n_points", "err_shared_max", "err_independent_max",
                 "nodes_shared", "nodes_independent", "wall_shared_s", "wall_ind_s",
                 "wall_production_s"],
                [[f"2d-grid-{n_side}x{n_side}", f"{omega:.1f}", str(v.size),
                  f"{worst:.17e}", f"{worst_cross:.17e}", str(nodes_vec), str(nodes_ind),
                  f"{t_vec:.3f}", f"{t_ind:.3f}", ""]])


# ---------------------------------------------------------------------------
# E2: frequency batching.
# ---------------------------------------------------------------------------

def _rc_field_omega_batch(omegas, beta, lenses, *, margin=4.0, margin_outer=1.5,
                          tol=1e-13, counter=None):
    """2D point-mass field, vectorized over omega (one shared contour)."""
    omegas = np.asarray(omegas, dtype=float)
    omax = omegas.max()
    A = min([beta[0]] + [a for a, _, _ in lenses]) - margin
    B = max([beta[0]] + [a for a, _, _ in lenses]) + margin
    A2 = min([beta[1]] + [b for _, b, _ in lenses]) - margin_outer
    B2 = max([beta[1]] + [b for _, b, _ in lenses]) + margin_outer
    pts = sorted(set([A, B] + [a for a, _, _ in lenses]))
    pts2 = sorted(set([A2, B2] + [b for _, b, _ in lenses]))
    span = max(abs(A), abs(B))
    span2 = max(abs(A2), abs(B2))
    floor = 4e-16 * (1.0 + omax * (0.5 * span ** 2 + 0.5 * span2 ** 2 + 10.0))

    def inner(x2):
        pts_i = pts
        tot = np.zeros(omegas.size, dtype=complex)

        def core_f(r, x0, sgn):
            rr = np.asarray(r, dtype=float)
            z = x0 + sgn * rr
            K = 0.5 * z * z
            for a, b, m in lenses:
                t = (x0 - a) + sgn * rr
                K = K - 0.5 * m * np.log(t * t + (x2 - b) ** 2)
            beta0 = beta[0]
            return (np.exp(1j * omega_outer * K[:, None])
                    * np.exp(-1j * omega_outer * (z[:, None] * beta0))
                    * np.exp(0.5j * omega_outer * beta0 ** 2))
        omega_outer = omegas[None, :]
        for lo, hi in zip(pts_i[:-1], pts_i[1:]):
            half = 0.5 * (hi - lo)
            for x0, sgn in [(lo, 1.0), (hi, -1.0)]:
                tot += _adapt_vec(lambda r, x0=x0, sgn=sgn: core_f(r, x0, sgn),
                                  0.0, half, tol, 1e-15, floor, counter)
        T_len = (math.sqrt(90.0 / omegas.min()) + abs(beta[0]) / math.cos(math.pi / 4) + 2.0)
        for X, sgn in [(B, 1.0), (A, -1.0)]:
            def tail_f(t, X=X, sgn=sgn):
                tt = np.asarray(t, dtype=float)
                z = X + sgn * E4 * tt
                K = 0.5 * z * z
                for a, b, m in lenses:
                    w = (z - a) ** 2 + (x2 - b) ** 2
                    K = K - 0.5 * m * np.log(w)
                return E4 * (np.exp(1j * omega_outer * K[:, None])
                             * np.exp(-1j * omega_outer * (z[:, None] * beta[0]))
                             * np.exp(0.5j * omega_outer * beta[0] ** 2))
            tot += _adapt_vec(tail_f, 0.0, T_len, tol, 1e-14, floor, counter)
        return tot

    def outer(x2):
        g = inner(x2)
        return g * np.exp(1j * omegas * (0.5 * x2 * x2 - beta[1] * x2)
                          + 0.5j * omegas * beta[1] ** 2)

    total = np.zeros(omegas.size, dtype=complex)
    for lo, hi in zip(pts2[:-1], pts2[1:]):
        rmax = math.sqrt(0.5 * (hi - lo))
        for x0, sgn in [(lo, 1.0), (hi, -1.0)]:
            def f(rho, x0=x0, sgn=sgn):
                q = np.asarray(rho, dtype=float)
                return np.array([outer(x0 + sgn * qq * qq) * 2.0 * qq for qq in q])
            total += _adapt_vec(f, 0.0, rmax, tol, 1e-12, floor, counter)
    S_len = (math.sqrt(90.0 / omegas.min()) + abs(beta[1]) / math.cos(math.pi / 4) + 2.0)
    for X, sgn in [(B2, 1.0), (A2, -1.0)]:
        def f(s, X=X, sgn=sgn):
            q = np.asarray(s, dtype=float)
            return np.array([E4 * outer(X + sgn * E4 * qq) for qq in q])
        total += _adapt_vec(f, 0.0, S_len, tol, 1e-12, floor, counter)
    return total


def e2(band=(1.0, 10.0), n=24):
    print(f"E2: frequency batching over a log band {band}")
    omegas = np.geomspace(band[0], band[1], n)

    # single mass vs ND
    c = [0]
    t0 = time.time()
    v = _rc_field_omega_batch(omegas, (0.5, 0.0), [(0.0, 0.0, 1.0)], counter=c)
    t_batch = time.time() - t0
    truth = np.array([slice_raw_from_nd(w, 0.5) for w in omegas])
    err = np.abs(v - truth) / np.abs(truth)
    nodes = c[0] // 20
    print(f"    single mass: batch max rel vs ND {err.max():.2e}  nodes {nodes} "
          f"({nodes / n:.1f}/omega) wall {t_batch:.1f} s")

    # per-omega independent referee node counts (only a few omega, the referee
    # is slow): subsample
    sub = np.geomspace(band[0], band[1], 5)
    nodes_ind = []
    for w in sub:
        cc = [0]
        general_field_referee(w, point_components([(0.0, 0.0, 1.0)]), (0.5, 0.0),
                              counter=cc)
        nodes_ind.append(cc[0] // 20)
    print(f"    independent referee nodes at {['%.2g' % s for s in sub]}: {nodes_ind}")

    # two-mass field: batch vs per-omega referee
    c = [0]
    v2 = _rc_field_omega_batch(omegas, TWO_MASS_BETA, TWO_MASS_LENSES, counter=c)
    nodes2 = c[0] // 20
    sub = np.geomspace(band[0], band[1], 4)
    worst = 0.0
    for w in sub:
        I = general_field_referee(float(w), point_components(TWO_MASS_LENSES), TWO_MASS_BETA)
        vv = _rc_field_omega_batch(np.array([w]), TWO_MASS_BETA, TWO_MASS_LENSES)
        worst = max(worst, rel(vv[0], I))
    print(f"    two-mass field: batch nodes {nodes2} ({nodes2 / n:.1f}/omega); "
          f"batch vs per-omega referee ({len(sub)} pts): max rel {worst:.2e}")

    # Chebyshev interpolation of F(omega) on the band (single mass, ND truth)
    print("    Chebyshev interpolation of F(omega) (single mass, y=0.5):")
    rows = []
    test = np.geomspace(band[0], band[1], 40)
    F_test = np.array([point_mass_exact(w, 0.5) for w in test])
    for ncheb in (6, 8, 10, 12, 16):
        samples = np.geomspace(band[0], band[1], ncheb)
        cheb = np.polynomial.chebyshev.Chebyshev.fit(
            np.linspace(-1, 1, ncheb),
            np.array([point_mass_exact(w, 0.5) for w in samples]),
            ncheb - 1)
        # evaluate at test points mapped the same way (log-spaced variable)
        mapped = (2.0 * (np.log(test) - math.log(band[0]))
                  / (math.log(band[1]) - math.log(band[0])) - 1.0)
        pred = cheb(mapped)
        err_c = np.max(np.abs(pred - F_test) / np.abs(F_test))
        print(f"      n={ncheb:2d}: max rel {err_c:.2e}")
        rows.append([ncheb, err_c])
    _append_csv("p0_batch_omega.csv",
                ["case", "n_omega", "max_rel_err", "nodes", "nodes_per_omega", "wall_s", "note"],
                [["single-mass-batch", n, f"{err.max():.17e}", nodes, f"{nodes / n:.2f}",
                  f"{t_batch:.3f}", "vs ND"],
                 ["two-mass-batch-referee", n, f"{worst:.17e}", nodes2, f"{nodes2 / n:.2f}",
                  "", "vs per-omega referee"],
                 *[["chebyshev", k, f"{e:.17e}", "", "", "", "single mass, ND test points"]
                   for k, e in rows]])
    print(f"    -> {os.path.join(RESULTS, 'p0_batch_omega.csv')}")


# ---------------------------------------------------------------------------
# E3: low omega.
# ---------------------------------------------------------------------------

def e3(omegas=(1e-4, 1e-3, 1e-2, 0.1, 0.3), ys=(0.1, 1.0, 3.0)):
    print("E3: low-omega accuracy (F-1) and cost")
    rows = []
    for w in omegas:
        for y in ys:
            Iref = slice_raw_from_nd(w, y)
            Fref = w * Iref / (2j * math.pi)
            c1 = [0]
            t0 = time.time()
            I1 = general_field_referee(float(w), point_components([(0.0, 0.0, 1.0)]),
                                       (float(y), 0.0), counter=c1)
            t1 = time.time() - t0
            F1 = w * I1 / (2j * math.pi)
            c2 = [0]
            t0 = time.time()
            I2 = general_field_referee(float(w), point_components([(0.0, 0.0, 1.0)]),
                                       (float(y), 0.0), counter=c2, subtract_free=True)
            t2 = time.time() - t0
            F2 = w * I2 / (2j * math.pi)
            e1 = abs(F1 - Fref) / max(abs(Fref - 1.0), 1e-300)
            e2 = abs(F2 - Fref) / max(abs(Fref - 1.0), 1e-300)
            print(f"    w={w:7.1e} y={y:3.1f}: F-1 rel err direct {e1:.2e} / subtracted "
                  f"{e2:.2e}   nodes {c1[0] // 20} / {c2[0] // 20}   wall {t1:.1f}/{t2:.1f}s")
            rows.append([f"{w:.17e}", f"{y:.1f}", f"{e1:.17e}", f"{e2:.17e}",
                         c1[0] // 20, c2[0] // 20, f"{t1:.3f}", f"{t2:.3f}"])
    _append_csv("p0_low_omega.csv",
                ["omega", "y", "rel_err_Fm1_direct", "rel_err_Fm1_subtracted",
                 "nodes_direct", "nodes_subtracted", "wall_direct_s", "wall_subtracted_s"],
                rows)


# ---------------------------------------------------------------------------
# E4: macro model sweeps (the closed-form anchors are in the referee self-test;
# this row-set is the *parametric* sweep: alpha family, margins, transposed).
# ---------------------------------------------------------------------------

def e4():
    print("E4: macro model (kappa, gamma) sweep on the two-mass field")
    rows = []
    cases = [("I", 0.3, 0.2), ("II", 0.5, 0.7), ("III", 1.5, 0.2), ("near-crit", 0.45, 0.5)]
    base = general_field_referee
    for name, k, g in cases:
        t0 = time.time()
        I15 = base(3.0, point_components(TWO_MASS_LENSES), TWO_MASS_BETA,
                   kappa=k, gamma1=g, alpha=math.pi / 4, margin_outer=1.5,
                   check_sector=True)
        I30 = base(3.0, point_components(TWO_MASS_LENSES), TWO_MASS_BETA,
                   kappa=k, gamma1=g, alpha=math.pi / 4, margin_outer=3.0,
                   check_sector=True)
        Ia = base(3.0, point_components(TWO_MASS_LENSES), TWO_MASS_BETA,
                  kappa=k, gamma1=g, alpha=math.pi / 8, check_sector=True)
        dt = time.time() - t0
        # transposed: reflection (x0,x2)->(x2,x0) maps gamma1 -> -gamma1
        lens_t = [(b, a, m) for a, b, m in TWO_MASS_LENSES]
        It = base(3.0, point_components(lens_t), (TWO_MASS_BETA[1], TWO_MASS_BETA[0]),
                  kappa=k, gamma1=-g)
        print(f"    {name:9s} k={k} g={g}: margin {rel(I15, I30):.2e}  "
              f"alpha {rel(I15, Ia):.2e}  transposed {rel(I15, It):.2e}  ({dt:.1f}s)")
        rows.append([name, f"{k}", f"{g}", f"{rel(I15, I30):.17e}", f"{rel(I15, Ia):.17e}",
                     f"{rel(I15, It):.17e}"])
    # mass-sheet anchor (external): kappa, single mass, gamma=0, vs ND mapping
    for k in (0.3, 0.45):
        I = base(3.0, point_components([(0.0, 0.0, 1.0)]), (0.5, 0.0), kappa=k)
        Im = mass_sheet_nd(3.0, (0.5, 0.0), 1.0, k)
        print(f"    mass-sheet k={k}: rel vs ND mapping {rel(I, Im):.2e}")
        rows.append([f"mass-sheet-k{k}", f"{k}", "0", f"{rel(I, Im):.17e}", "", ""])
    # macro-only closed form with shear
    for k, g1, g2 in [(0.3, 0.2, 0.0), (0.1, 0.35, 0.1)]:
        I = base(3.0, [], (0.4, -0.3), kappa=k, gamma1=g1, gamma2=g2)
        Im = macro_only_integral(3.0, (0.4, -0.3), kappa=k, gamma1=g1, gamma2=g2)
        print(f"    macro-only k={k} g=({g1},{g2}): rel {rel(I, Im):.2e}")
        rows.append([f"macro-only-{k}", f"{k}", f"{g1},{g2}", f"{rel(I, Im):.17e}", "", ""])
    _append_csv("p0_macro.csv",
                ["case", "kappa", "gamma", "rel_margin", "rel_alpha", "rel_transposed"],
                rows)


# ---------------------------------------------------------------------------
# E5: halo profiles.
# ---------------------------------------------------------------------------

def _profile_components(kind, params, x=0.0, y=0.0):
    return [(kind, x, y, params)]


def e5(cases=None):
    print("E5: halo profiles vs the high-precision polar referee")
    if cases is None:
        cases = [("sis", (1.0,)), ("cis", (1.0, 0.2)), ("nfw", (1.0, 0.5))]
    wsy = [(0.3, 1.0), (1.0, 0.5), (3.0, 0.3), (10.0, 0.5)]
    rows = []
    for kind, params in cases:
        for w, y in wsy:
            c = [0]
            t0 = time.time()
            I = general_field_referee(w, _profile_components(kind, params), (y, 0.0),
                                      counter=c)
            t_car = time.time() - t0
            t0 = time.time()
            Ihp = hankel_referee_hp(w, y, make_psi_w_mp([(kind, params)]))
            t_hp = time.time() - t0
            print(f"    {kind:4s} w={w:5.1f} y={y:3.1f}: rel(Cartesian, polar-hp) "
                  f"{rel(I, Ihp):.2e}  nodes {c[0] // 20}  wall {t_car:.1f}/{t_hp:.1f}s")
            rows.append([kind, ",".join(str(p) for p in params), f"{w:.1f}", f"{y:.1f}",
                         f"{rel(I, Ihp):.17e}", c[0] // 20, f"{t_car:.3f}", f"{t_hp:.3f}"])

    # real-axis convention vs FIONA's lens classes (restricted to u >= 0.1:
    # FIONA's own log^2 + (u^2-1)F^2 form cancels catastrophically as u -> 0,
    # so below that its *implementation* is the unreliable side, not ours).
    # FIONA reads its GL-cache env var at import time: set it first (the
    # driver's gotcha -- Dropbox case-conflicts on .W.npy/.w.npy).
    os.environ.setdefault("FIONA_GL2D_DIR", "/tmp/fastpiclef_fiona_gl")
    os.makedirs(os.environ["FIONA_GL2D_DIR"], exist_ok=True)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "refs", "FIONA"))
    from fiona import lenses as fl  # noqa: E402
    checks = [("sis", (1.0,), fl.SIS(psi0=1.0), 0.0),
              ("cis", (1.0, 0.2), fl.CIS(psi0=1.0, xc=0.2), 0.0),
              ("nfw", (1.0, 0.5), fl.NFW(psi0=1.0, xs=0.5), 0.1)]
    for kind, params, flens, u_min in checks:
        rs = np.geomspace(max(1e-6, u_min * params[-1] if kind == "nfw" else 1e-6),
                          20.0, 200)
        ours = np.array([complex(_psi_w(kind, params, np.asarray([complex(r * r)]))[0]).real
                         for r in rs])
        ref = np.asarray(flens.psi_r(rs))
        m = np.abs(ref) > 1e-30
        worst = np.max(np.abs(ours[m] - ref[m]) / np.abs(ref[m]))
        print(f"    real axis vs FIONA {kind}: max rel {worst:.2e}")
        rows.append([f"{kind}-fiona-reals", "", "", "", f"{worst:.17e}", "", "", ""])

    # analyticity loops: closed loops in the complex w plane (cyclic trapezoid —
    # endpoint=True so no half-endpoint defect)
    def loop(gamma_c, R, npts=2000):
        th = np.linspace(0, 2 * np.pi, npts, endpoint=True)
        w = gamma_c + R * np.exp(1j * th)
        v = None
        for kind, params in cases_all:
            p = _psi_w(kind, params, w)
            v = p if v is None else v + p
        dz = 1j * R * np.exp(1j * th)
        f = v * dz
        return abs(np.sum(f[:-1]) * (2 * np.pi / (npts - 1)) / (2 * np.pi * 1j))
    cases_all = cases
    res_away = loop(2.0 + 2.0j, 0.5)          # loop far from the cut
    res_around0 = loop(-1.0 + 0.0j, 0.5)      # loop around the negative axis
    print(f"    analyticity: |loop mean| away from cut {res_away:.2e}; "
          f"around the cut {res_around0:.2e} (must be large)")
    rows.append(["analyticity", "", "", "", f"{res_away:.17e}", "", "", ""])

    _append_csv("p0_profiles.csv",
                ["profile", "params", "omega", "y", "rel_cartesian_vs_polarhp",
                 "cartesian_nodes", "wall_cartesian_s", "wall_polarhp_s"],
                rows)


def e6():
    """FIONA cross-checks: the macro model on the two-mass field, and the
    axisymmetric profiles (third-party arbitration; |F| only, convention-free).
    refs/ stays read-only: the macro lens is a subclass defined here."""
    print("E6: FIONA cross-checks")
    import multiprocessing as mp
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    os.environ.setdefault("FIONA_GL2D_DIR", "/tmp/fastpiclef_fiona_gl")
    os.makedirs(os.environ["FIONA_GL2D_DIR"], exist_ok=True)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "refs", "FIONA"))
    import fiona  # noqa: E402
    from fiona import lenses as fl  # noqa: E402
    rows = []

    class MacroPointLens(fiona.Lens):
        def __init__(self, lenses, kappa=0.0, gamma1=0.0, gamma2=0.0):
            self.lenses = [(float(a), float(b), float(m)) for a, b, m in lenses]
            self.kappa = float(kappa)
            self.g1 = float(gamma1)
            self.g2 = float(gamma2)

        def psi_xy(self, x1, x2):
            x1 = np.asarray(x1, dtype=float)
            x2 = np.asarray(x2, dtype=float)
            acc = (0.5 * self.kappa * (x1 * x1 + x2 * x2)
                   + 0.5 * self.g1 * (x1 * x1 - x2 * x2) + self.g2 * x1 * x2)
            for a, b, m in self.lenses:
                acc = acc + 0.5 * m * np.log((x1 - a) ** 2 + (x2 - b) ** 2)
            return acc

    def fiona_run(lens, beta, wlist, threads=1):
        fiona.set_num_threads(threads)
        solver = fiona.FresnelNUFFT3(lens, auto_R_from_gl_nodes=True,
                                     parallel_frequencies=(threads > 1),
                                     nufft_nthreads=threads, verbose=False)
        w = np.asarray(wlist, dtype=float)
        return np.asarray(solver(w, np.array([beta[0]]), np.array([beta[1]]))).reshape(-1)

    # macro model + two point masses, type I and II
    for name, k, g1 in [("type-I", 0.3, 0.2), ("type-II", 0.5, 0.7)]:
        wlist = [1.0, 3.0, 10.0]
        Ff = fiona_run(MacroPointLens(TWO_MASS_LENSES, k, g1), TWO_MASS_BETA, wlist)
        for wi, Fi in zip(wlist, Ff):
            I = general_field_referee(wi, point_components(TWO_MASS_LENSES), TWO_MASS_BETA,
                                      kappa=k, gamma1=g1)
            Fo = wi * I / (2j * math.pi)
            print(f"    macro {name}: w={wi:4.1f}  |F| FIONA {abs(Fi):.8f}  ours "
                  f"{abs(Fo):.8f}  rel {abs(abs(Fi) - abs(Fo)) / abs(Fo):.2e}")
            rows.append([f"fiona-macro-{name}", f"{wi:.1f}", f"{abs(abs(Fi)-abs(Fo))/abs(Fo):.17e}"])
    # profiles
    for kind, params, lens in [("sis", (1.0,), fl.SIS(psi0=1.0)),
                               ("cis", (1.0, 0.2), fl.CIS(psi0=1.0, xc=0.2)),
                               ("nfw", (1.0, 0.5), fl.NFW(psi0=1.0, xs=0.5))]:
        wlist = [1.0, 3.0, 10.0]
        Ff = fiona_run(lens, (0.5, 0.0), wlist)
        for wi, Fi in zip(wlist, Ff):
            Ihp = hankel_referee_hp(wi, 0.5, make_psi_w_mp([(kind, params)]))
            Fo = wi * Ihp / (2j * math.pi)
            print(f"    {kind}: w={wi:4.1f}  |F| FIONA {abs(Fi):.8f}  ours {abs(Fo):.8f}"
                  f"  rel {abs(abs(Fi) - abs(Fo)) / abs(Fo):.2e}")
            rows.append([f"fiona-{kind}", f"{wi:.1f}", f"{abs(abs(Fi)-abs(Fo))/abs(Fo):.17e}"])
    # macro-only (lensless) saddle-type check: the *trivial* closed form
    # |F| = 1/sqrt|L0 L2|.  FIONA's real-box + window scheme cannot represent
    # the conditionally-convergent saddle directions (lambda < 0): its 30-50%
    # errors here are the reference code's limitation, not ours.
    for name, k, g in [("I", 0.3, 0.2), ("II", 0.5, 0.7), ("III", 1.5, 0.2), ("near-crit", 0.45, 0.5)]:
        w = 3.0
        F_closed = w * macro_only_integral(w, (0.4, -0.3), kappa=k, gamma1=g) / (2j * math.pi)
        Iref = general_field_referee(w, [], (0.4, -0.3), kappa=k, gamma1=g)
        F_ref = w * Iref / (2j * math.pi)
        Ff = fiona_run(MacroPointLens([], k, g), (0.4, -0.3), [w])[0]
        print(f"    macro-only {name:9s}: |F| ours {abs(F_ref):.10f} "
              f"(vs closed {abs(F_ref - F_closed) / abs(F_closed):.1e})  "
              f"FIONA {abs(Ff):.10f} (rel {abs(abs(Ff) - abs(F_closed)) / abs(F_closed):.1e})")
        rows.append([f"fiona-macro-only-{name}", f"{w:.1f}",
                     f"{abs(abs(Ff) - abs(F_closed)) / abs(F_closed):.17e}"])
    _append_csv("p0_fiona.csv", ["case", "omega", "rel_absF"], rows)


def main(argv):
    which = argv or ["E1", "E2", "E3", "E4", "E5", "E6"]
    t0 = time.time()
    if "E1" in which or "E1a" in which:
        e1a()
    if "E1" in which or "E1b" in which:
        e1b()
    if "E2" in which:
        e2()
    if "E3" in which:
        e3()
    if "E4" in which:
        e4()
    if "E5" in which:
        e5()
    if "E6" in which:
        e6()
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main(sys.argv[1:])
