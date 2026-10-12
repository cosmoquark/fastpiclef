"""P4: the wolensing comparison (approved 2026-10-12).

Ours vs wolensing vs the high-precision Hankel referee on the axisymmetric
single-lens cases where the parameter mapping is exact (SIS, point mass) or
calibrated by the deflection at the scale radius (NFW).  ``|F|`` is compared
(convention-free); ours is the exact RC value at tol = 1e-14.

Parameter mapping (see `wolensing_driver` for the conventions):
  * SIS:  lenstronomy ``theta_E = thetaE_ref``  <->  our ``psi0 = 1``;
  * point: ``POINT_MASS(theta_E = thetaE_ref)``  <->  our ``m = 1``;
  * NFW:  ``Rs = xs * thetaE_ref`` and ``alpha_Rs = thetaE_ref * psi0 * (1 - ln 2)``
    (the deflection of our dimensionless NFW at u = 1; the overall constant in
    psi is irrelevant for |F|).

Writes ``comparison/results/wolensing_compare.csv``.

Run: ``pixi run -e comparison python -m comparison.wolensing_compare``
"""

import os

os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import math

import numpy as np
import pandas as pd

import fastpiclef as fpl

from . import wolensing_driver as wd
from .referee import hankel_referee_hp, make_psi_w_mp, point_mass_exact

RESULTS = os.path.join(os.path.dirname(__file__), "results")
W_GRID = (1.0, 3.0, 10.0, 30.0)
HANKEL_W = (1.0, 10.0)  # mpmath: only a couple of points


def _ours_absF(builder, w):
    return abs(complex(builder().solve(omega=w, tol=1e-14).amplification))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    cases = [
        ("sis", 0.5,
         lambda: fpl.LensField([fpl.SISComp(0.0, 0.0, 1.0)], beta=(0.5, 0.0)),
         ["SIS"], [{"theta_E": 1.0, "center_x": 0.0, "center_y": 0.0}],
         ("sis", (1.0,))),
        ("point_mass", 0.5,
         lambda: fpl.LensField([fpl.PointComp(0.0, 0.0, 1.0)], beta=(0.5, 0.0)),
         ["POINT_MASS"], [{"theta_E": 1.0, "center_x": 0.0, "center_y": 0.0}],
         ("point", (1.0,))),
        ("nfw", 0.3,
         lambda: fpl.LensField([fpl.NFWComp(0.0, 0.0, 1.0, 1.0)], beta=(0.3, 0.0)),
         ["NFW"], [{"Rs": 1.0, "alpha_Rs": 1.0 * (1.0 - math.log(2.0)),
                    "center_x": 0.0, "center_y": 0.0}],
         ("nfw", (1.0, 1.0))),
    ]
    for name, y, builder, lm_list, kw_lens, psi_spec in cases:
        print(f"== {name} (y={y}) ==")
        # ours
        ours = {}
        for w in W_GRID:
            ours[w] = _ours_absF(builder, w)
            print(f"   ours  w={w:5.1f}: |F| = {ours[w]:.10f}")
        # wolensing
        Fw = wd.F_at(W_GRID, lm_list, kw_lens, (y, 0.0))
        wol = {w: abs(Fw[i]) for i, w in enumerate(W_GRID)}
        print("   wolensing: " + ", ".join(f"w={w:g}: {wol[w]:.10f}" for w in W_GRID))
        # Hankel referee (mpmath; a couple of points)
        hank = {}
        for w in HANKEL_W:
            if name == "point_mass":
                hank[w] = abs(point_mass_exact(w, y))
            else:
                psi_w = make_psi_w_mp([psi_spec])
                hank[w] = abs(hankel_referee_hp(w, y, psi_w)) * w / (2.0 * math.pi)
            print(f"   referee w={w:5.1f}: |F| = {hank[w]:.10f}")
        for w in W_GRID:
            rows.append(dict(
                case=name, y=y, omega=w,
                ours=ours[w], wolensing=wol[w], hankel=hank.get(w, float("nan")),
                rel_wol_vs_ours=abs(wol[w] - ours[w]) / ours[w],
                rel_wol_vs_hankel=(abs(wol[w] - hank[w]) / hank[w]
                                   if w in hank else float("nan")),
                rel_ours_vs_hankel=(abs(ours[w] - hank[w]) / hank[w]
                                    if w in hank else float("nan")),
            ))
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "wolensing_compare.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    print(df.groupby("case")[["rel_wol_vs_ours", "rel_wol_vs_hankel",
                              "rel_ours_vs_hankel"]].max().to_string())


if __name__ == "__main__":
    main()
