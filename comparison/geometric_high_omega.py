"""P4.1 strategy study: how good is geometric optics, and can we *certify* it?

The exact RC solver is the right tool up to omega ~ 100 but its cost grows like
omega^1.5; the LISA-halo regime (omega ~ 1e3-1e4) needs the stationary-phase
(geometric) answer plus an honest error estimate.  This script measures, against
external truth:

  * the point mass (Nakamura-Deguchi closed form, exact at every omega): the
    geometric error vs omega for off-caustic / mid / near-caustic sources, and
    the *self-consistency* estimator |F_geo(w) - F_geo(2w)| (a proxy computable
    without any exact solve);
  * spot checks on the macro (type I and II) and 3-mass cases: geometric vs the
    exact RC at omega = 30, 100, 300 (slow; the load-dependent times are noted).

Writes `comparison/results/geometric_high_omega.csv`.

Run: `pixi run -e comparison python -m comparison.geometric_high_omega`
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

from .referee import point_mass_exact

RESULTS = os.path.join(os.path.dirname(__file__), "results")
FIELD_LENSES = [(0.6, 0.0, 0.4), (-0.5, 0.4, 0.3), (0.1, -0.6, 0.35)]
FIELD_SOURCE = (0.2, 0.1)


def _geo_absF(builder, omega):
    return abs(complex(builder().geometric(omega).amplification))


def _to_F(I, omega):
    return I * omega / (2.0 * math.pi * 1j)


def point_mass_study(ys=(0.1, 0.5, 0.9), w_grid=None):
    """Geometric error vs the exact ND form + the self-consistency estimator."""
    if w_grid is None:
        w_grid = np.geomspace(10.0, 1e4, 13)
    rows = []
    for y in ys:
        builder = lambda: fpl.PointMassField([(0.0, 0.0, 1.0)], beta=(y, 0.0))
        for w in w_grid:
            truth = abs(point_mass_exact(float(w), y))
            g1 = _geo_absF(builder, float(w))
            g2 = _geo_absF(builder, float(2.0 * w))
            rows.append(dict(
                case="point_mass", y=y, omega=float(w),
                absF_geo=g1, absF_exact=truth,
                geo_rel_err=abs(g1 - truth) / truth,
                selfcons_rel=abs(g1 - g2) / truth,
            ))
    return rows


def field_study(omegas=(30.0, 100.0, 300.0)):
    """Geometric vs the exact RC on the macro and 3-mass cases (RC is slow)."""
    cases = {
        "type1_macro": lambda: fpl.LensField([fpl.PointComp(0.0, 0.0, 1.0)],
                                             beta=(0.3, 0.1), kappa=0.3, gamma1=0.2),
        "type2_macro": lambda: fpl.LensField([fpl.PointComp(0.0, 0.0, 1.0)],
                                             beta=(0.3, 0.1), kappa=0.5, gamma1=0.7),
        "field3": lambda: fpl.PointMassField(FIELD_LENSES, beta=FIELD_SOURCE),
    }
    rows = []
    for name, builder in cases.items():
        for w in omegas:
            g1 = _geo_absF(builder, w)
            g2 = _geo_absF(builder, 2.0 * w)
            t0 = time.perf_counter()
            r = builder().solve(omega=w, tol=1e-14)
            wall = time.perf_counter() - t0
            exact = abs(_to_F(complex(r.integral), w))
            rows.append(dict(
                case=name, y=float("nan"), omega=w,
                absF_geo=g1, absF_exact=exact,
                geo_rel_err=abs(g1 - exact) / exact,
                selfcons_rel=abs(g1 - g2) / exact,
                rc_wall_s=wall,
            ))
            print(f"  {name:12s} w={w:6.0f}: geo {g1:.6f} exact {exact:.6f} "
                  f"err {abs(g1 - exact) / exact:.2e} selfcons "
                  f"{abs(g1 - g2) / exact:.2e} ({wall:.1f} s)")
    return rows


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rows = point_mass_study()
    print("== point mass: geometric error vs the exact ND form ==")
    df = pd.DataFrame(rows)
    for y, g in df.groupby("y"):
        g = g.sort_values("omega")
        fit = np.polyfit(np.log10(g["omega"]), np.log10(g["geo_rel_err"]), 1)
        print(f"  y={y}: geo err ~ omega^{fit[0]:+.2f}; at omega=1e3: "
              f"{g[g['omega'] >= 1e3]['geo_rel_err'].iloc[0]:.1e}; "
              f"selfcons tracks: ratio "
              f"{(g['selfcons_rel'] / g['geo_rel_err']).median():.2f}")
    rows += field_study()
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "geometric_high_omega.csv"), index=False)
    print(f"wrote {os.path.join(RESULTS, 'geometric_high_omega.csv')}")


if __name__ == "__main__":
    main()
