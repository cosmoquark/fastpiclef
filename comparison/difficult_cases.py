"""P3 — difficult-case taxonomy.

Measures the solver on the stress cases of the plan's §5.3 and writes
`comparison/results/difficult_cases.csv` plus a markdown rendering
(`difficult_cases.md`).  Columns: our accuracy, our cost, whether the error
budget covers the true error, the reference used, competitors where they run.

Truths: closed forms (Airy, Nakamura-Deguchi) where they exist; the Python RC
referee (a *different* margin — Cauchy) for the slice rows; the margin/alpha
family spread for the macro and profile rows (documented as a truth proxy).

Run: `pixi run -e comparison python -m comparison.difficult_cases`
"""

import os

os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import math
import time

import numpy as np
import pandas as pd

import fastpiclef as fpl

from .referee import airy_exact, slice_raw_from_nd, slice_referee

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def _measured(fn):
    t0 = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t0


def _rel(v, t):
    return abs(v - t) / abs(t)


def _budget(r, true_err):
    """Error-budget coverage: the estimate plus an 8-ulp allowance of the value
    (the estimator sums accepted-panel disagreements, so at the roundoff floor
    it can under-cover by a couple of ulps — same convention as the test suite)."""
    allow = 8.0 * float(np.finfo(float).eps) * abs(complex(r.integral))
    return (r.error + allow) / max(true_err, 1e-300)


def _row(case, definition, params, ours_err, cost, budget, referee, note=""):
    return dict(
        case=case,
        definition=definition,
        params=params,
        ours_rel_err=float(ours_err),
        ours_cost_s=float(cost),
        budget_ratio=float(budget),
        referee=referee,
        note=note,
    )


def _cauchy_spread(builder, omega, tol=1e-14):
    """(value, spread) from margin/alpha-family Cauchy variations."""
    b = builder
    v = complex(b().solve(omega=omega, tol=tol).integral)
    v2 = complex(b().solve(omega=omega, tol=tol, margin=2.5).integral)
    v3 = complex(b().solve(omega=omega, tol=tol, margin_outer=3.0).integral)
    return v, max(abs(v2 - v), abs(v3 - v)) / abs(v)


def _point_field(y):
    return fpl.PointMassField([(0.0, 0.0, 1.0)], beta=(y, 0.0))


# ------------------------------------------------------------------- rows


def row_near_caustic_airy():
    worst, cost, budget = 0.0, 0.0, np.inf
    notes = []
    for y in (0.05, 0.0):
        truth = airy_exact(y, 1.0)
        try:
            r, dt = _measured(lambda: fpl.AiryPhase(y=y).solve(omega=1.0, method="thimble"))
            method = "thimble"
        except Exception as e:  # noqa: BLE001
            notes.append(f"thimble at y={y}: {type(e).__name__} ({e})")
            r, dt = _measured(lambda: fpl.AiryPhase(y=y).solve(
                omega=1.0, method="mesh", delta=0.01, domain=(-8.0, 8.0)))
            method = "mesh"
        e = _rel(complex(r.integral), truth)
        worst = max(worst, e)
        cost = max(cost, dt)
        budget = min(budget, _budget(r, e * abs(truth)))
        notes.append(f"y={y} via {method}: {e:.1e}")
    return _row("near_caustic_airy", "Airy cubic phase at/near the caustic (y->0)",
                "y = 0.05 (thimble) and y = 0.0 (degenerate saddle: the thimble refuses "
                "by design, the mesh runs)", worst, cost, budget,
                "Airy closed form (external)", "; ".join(notes))


def row_near_caustic_2d():
    worst, cost, budget = 0.0, 0.0, np.inf
    for y in (0.9, 0.99, 1.0, 1.01):
        truth = slice_raw_from_nd(10.0, y)
        r, dt = _measured(lambda: _point_field(y).solve(omega=10.0, tol=1e-14))
        e = _rel(complex(r.integral), truth)
        worst = max(worst, e)
        cost = max(cost, dt)
        budget = min(budget, _budget(r, e * abs(truth)))
    # FIONA on the caustic point (its native protocol, for the record)
    from . import fiona_driver

    truth_caustic = slice_raw_from_nd(10.0, 1.0)
    (F, _), dt = _measured(lambda: fiona_driver.fiona_point_mass_sweep(1.0, [10.0]))
    ef = _rel(complex(F[0]), truth_caustic * 10.0 / (2.0 * math.pi * 1j))
    return _row("near_caustic_2d", "point mass, source on/near the caustic (y_crit = 1)",
                "y in {0.9, 0.99, 1.0, 1.01}, w = 10", worst, cost, budget,
                "Nakamura-Deguchi (mpmath dps 30)",
                f"FIONA at the caustic: rel {ef:.1e} ({dt:.1f} s)")


def row_near_pinch():
    lenses = [(0.0, 0.5, 1.0)]
    beta = (0.2, 0.0)
    omega = 10.0
    worst, cost, budget = 0.0, 0.0, np.inf
    for delta in (1e-6, 1e-9, 1e-12, 0.0):
        x2 = 0.5 + delta
        r, dt = _measured(lambda: fpl.slice_inner_rc(lenses, beta, (x2, 0.0), omega, tol=1e-14))
        ref = slice_referee(complex(x2, 0.0), omega, lenses=tuple(lenses), beta=beta,
                            margin=4.0, tol=1e-14)
        r2, _ = _measured(lambda: fpl.slice_inner_rc(lenses, beta, (x2, 0.0), omega,
                                                    tol=1e-14, margin=2.0))
        e = _rel(complex(r.integral), ref)
        cauchy = abs(complex(r.integral) - complex(r2.integral)) / abs(complex(r.integral))
        worst = max(worst, e)
        cost = max(cost, dt)
        budget = min(budget, _budget(r, e * abs(ref)))
    return _row("near_pinch_slice", "inner RC slice at the pinch height x2 = b_j",
                "x2 = b + {1e-6, 1e-9, 1e-12, 0}, w = 10", worst, cost, budget,
                "Python RC referee at margin 4 (different contour)",
                f"kernel-vs-referee; worst Cauchy margin spread {cauchy:.1e}")


def row_large_omega():
    worst, cost, budget = 0.0, 0.0, np.inf
    for omega in (100.0, 300.0):
        truth = slice_raw_from_nd(omega, 1.0)
        r, dt = _measured(lambda: _point_field(1.0).solve(omega=omega, tol=1e-14))
        e = _rel(complex(r.integral), truth)
        worst = max(worst, e)
        cost = max(cost, dt)
        budget = min(budget, _budget(r, e * abs(truth)))
    return _row("large_omega", "high frequency (cost grows ~w^1.5; see S3/P4.1)",
                "w in {100, 300}, y = 1", worst, cost, budget,
                "Nakamura-Deguchi (mpmath dps 30)",
                "FIONA's node count also grows with w (see C1 sweep)")


def row_large_offset():
    worst, cost, budget = 0.0, 0.0, np.inf
    for y in (5.0, 10.0):
        truth = slice_raw_from_nd(10.0, y)
        r, dt = _measured(lambda: _point_field(y).solve(omega=10.0, tol=1e-14))
        e = _rel(complex(r.integral), truth)
        worst = max(worst, e)
        cost = max(cost, dt)
        budget = min(budget, _budget(r, e * abs(truth)))
    from . import fiona_driver

    (F, _), dt = _measured(lambda: fiona_driver.fiona_point_mass_sweep(5.0, [10.0]))
    truth5 = slice_raw_from_nd(10.0, 5.0)
    ef = _rel(complex(F[0]), truth5 * 10.0 / (2.0 * math.pi * 1j))
    return _row("large_offset", "source far outside the lens (weak field)",
                "y in {5, 10}, w = 10", worst, cost, budget,
                "Nakamura-Deguchi (mpmath dps 30)",
                f"FIONA at y=5: rel {ef:.1e} ({dt:.1f} s)")


def row_low_omega():
    worst, cost, budget = 0.0, 0.0, np.inf
    worst_f1 = 0.0
    for omega in (1e-3, 1e-4):
        truth = slice_raw_from_nd(omega, 0.5)
        r, dt = _measured(lambda: _point_field(0.5).solve(omega=omega, tol=1e-14))
        e = _rel(complex(r.integral), truth)
        # the physical quantity at low w is F - 1
        fs = (2.0 * math.pi / omega) ** 0.5 * np.exp(1j * math.pi / 4.0)
        f1_ref = abs(truth / fs - 1.0)
        worst_f1 = max(worst_f1, abs((complex(r.integral) - truth) / fs) / f1_ref)
        worst = max(worst, e)
        cost = max(cost, dt)
        budget = min(budget, _budget(r, e * abs(truth)))
    return _row("low_omega", "low frequency (F - 1 is the physical quantity)",
                "w in {1e-3, 1e-4}, y = 0.5", worst, cost, budget,
                "Nakamura-Deguchi (mpmath dps 30)",
                f"worst rel error on F - 1: {worst_f1:.1e}; "
                "the optional free-space subtraction is not used (P4)")


def _det_lenses(n, seed=0x9E3779B97F4A7C15):
    s = seed | 1

    def nxt():
        nonlocal s
        s ^= (s << 13) & ((1 << 64) - 1)
        s ^= s >> 7
        s ^= (s << 17) & ((1 << 64) - 1)
        return (s >> 11) / float(1 << 53)

    return [(nxt() * 6.0 - 3.0, nxt() * 6.0 - 3.0, 0.02) for _ in range(n)]


def row_many_masses():
    rows = []
    for n in (25, 100):
        lenses = _det_lenses(n)
        v, spread = _cauchy_spread(lambda: fpl.PointMassField(lenses, beta=(0.3, 0.1)), 1.0)
        r, dt = _measured(lambda: fpl.PointMassField(lenses, beta=(0.3, 0.1)).solve(omega=1.0, tol=1e-14))
        rows.append((n, dt, spread, _budget(r, spread * abs(v))))
    return _row("many_masses", "many point masses (per-node cost O(N))",
                "N in {25, 100}, w = 1, m_j = 0.02",
                max(sp for _, _, sp, _ in rows), max(c for _, c, _, _ in rows),
                min(b for *_, b in rows), "margin-family Cauchy spread",
                "cost scaling: " + ", ".join(f"N={n}: {c:.1f} s" for n, c, _, _ in rows))


def row_near_critical_macro():
    b = lambda: fpl.LensField([fpl.PointComp(0.0, 0.0, 1.0)], beta=(0.4, 0.0),
                              kappa=0.45, gamma1=0.5)
    v, spread = _cauchy_spread(b, 3.0)
    r, dt = _measured(lambda: b().solve(omega=3.0, tol=1e-14))
    return _row("near_critical_macro", "macro model near critical (lambda0 = 0.05: flat Gaussian)",
                "kappa = 0.45, gamma1 = 0.5, w = 3", spread, dt,
                _budget(r, spread * abs(v)), "margin-family Cauchy spread",
                "long tails / slow decay: the cost is the price of exactness")


def row_saddle_macro():
    b = lambda: fpl.LensField([fpl.PointComp(0.0, 0.0, 1.0)], beta=(0.4, 0.0),
                              kappa=0.5, gamma1=0.7)
    v, spread = _cauchy_spread(b, 10.0)
    r, dt = _measured(lambda: b().solve(omega=10.0, tol=1e-14))
    return _row("saddle_macro", "type-II saddle macro image (lambda0 = -0.2)",
                "kappa = 0.5, gamma1 = 0.7, w = 10", spread, dt,
                _budget(r, spread * abs(v)), "margin-family Cauchy spread",
                "FIONA errs 29-49% here (P0 finding of record) - never use it as truth")


def row_small_core_cis():
    b = lambda: fpl.LensField([fpl.CISComp(0.0, 0.0, 1.0, 0.01)], beta=(0.5, 0.0))
    v, spread = _cauchy_spread(b, 3.0)
    r, dt = _measured(lambda: b().solve(omega=3.0, tol=1e-14))
    return _row("small_core_cis", "CIS with a tiny core (xc = 0.01; stiff potential)",
                "psi0 = 1, xc = 0.01, w = 3", spread, dt,
                _budget(r, spread * abs(v)), "margin-family Cauchy spread",
                "the P2 polar-referee anchor was at xc = 0.2 (P0 E5)")


def row_nfw():
    worst, cost, budget = 0.0, 0.0, np.inf
    for omega in (1.0, 10.0):
        b = lambda: fpl.LensField([fpl.NFWComp(0.0, 0.0, 1.0, 1.0)], beta=(0.5, 0.0))
        v, spread = _cauchy_spread(b, omega)
        r, dt = _measured(lambda: b().solve(omega=omega, tol=1e-14))
        worst = max(worst, spread)
        cost = max(cost, dt)
        budget = min(budget, _budget(r, spread * abs(v)))
    return _row("nfw", "NFW halo (log^2 potential, u ~ 1 cancellation handled)",
                "psi0 = 1, xs = 1, w in {1, 10}, y = 0.5", worst, cost, budget,
                "margin-family Cauchy spread (P2 tests: <= 8e-15 vs the Hankel referee)",
                "glworia would be the third-party comparator (blocked by JAX drift, see notes)")


def row_mixed_field():
    b = lambda: fpl.LensField(
        [fpl.PointComp(0.0, 0.0, 0.3), fpl.NFWComp(0.5, 0.2, 0.2, 0.4)],
        beta=(0.3, 0.1), kappa=0.3, gamma1=0.2)
    v, spread = _cauchy_spread(b, 3.0)
    r, dt = _measured(lambda: b().solve(omega=3.0, tol=1e-14))
    return _row("mixed_field", "stars + halo + macro (the science configuration)",
                "point + NFW + (kappa=0.3, gamma1=0.2), w = 3", spread, dt,
                _budget(r, spread * abs(v)), "margin-family Cauchy spread")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    for fn in (
        row_near_caustic_airy,
        row_near_caustic_2d,
        row_near_pinch,
        row_large_omega,
        row_large_offset,
        row_low_omega,
        row_many_masses,
        row_near_critical_macro,
        row_saddle_macro,
        row_small_core_cis,
        row_nfw,
        row_mixed_field,
    ):
        print(f"== {fn.__name__} ==")
        row = fn()
        rows.append(row)
        print(f"   err {row['ours_rel_err']:.2e}  cost {row['ours_cost_s']:.2f} s  "
              f"budget {row['budget_ratio']:.1f}x  ({row['referee']})")

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "difficult_cases.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}")

    md = ["# Difficult-case taxonomy (P3)", "",
          "| case | definition | params | our rel err | cost [s] | budget | referee | note |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(
            f"| {r['case']} | {r['definition']} | {r['params']} | "
            f"{r['ours_rel_err']:.1e} | {r['ours_cost_s']:.2f} | "
            f"{r['budget_ratio']:.1f}x | {r['referee']} | {r['note']} |"
        )
    with open(os.path.join(RESULTS, "difficult_cases.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("wrote results/difficult_cases.md")


if __name__ == "__main__":
    main()
