"""O2: the low-omega free-space subtraction, Rust vs the E3 referee.

The E3 experiment (`comparison/p0_experiments.py`, dataset
`results/p0_low_omega.csv`) measured the Python referee's direct vs
`subtract_free` `F - 1` accuracy on the grid
`omega in {1e-4, 1e-3, 1e-2, 0.1, 0.3} x y in {0.1, 1, 3}`.  This script runs
the same grid through the Rust engine (`PointMassField.solve`, with and
without `subtract_free=True`) against the 30-digit mpmath ND closed form
(`referee.slice_raw_from_nd` --- the truth of the E3 table), prints the four
columns side by side, and writes `results/low_omega_rust.csv`.

`rel_err_Fm1 = |F - F_ref| / |F_ref - 1|` with `F = omega I / (2 pi i)`.
`cover_*` is the engine's reported integral budget over the true integral
error (`|I - I_ref|`); `nodes_*` are inner solves (load-independent).  The
direct path's `F - 1` floor is the quadrature's `eps |I_free|` accumulation;
the subtracted path replaces it by the final `free + J` addition's
`eps / |F - 1|`.

Run: `pixi run -e comparison low-omega`
"""

import csv
import math
import os

import fastpiclef as fpl

from comparison.referee import slice_raw_from_nd

RESULTS = os.path.join(os.path.dirname(__file__), "results")
GRID = [(w, y) for w in (1e-4, 1e-3, 1e-2, 0.1, 0.3) for y in (0.1, 1.0, 3.0)]


def main():
    # The referee's own rows (same grid), for the side-by-side columns.
    ref = {}
    with open(os.path.join(RESULTS, "p0_low_omega.csv"), newline="") as fh:
        for row in csv.DictReader(fh):
            ref[(float(row["omega"]), float(row["y"]))] = row

    rows = []
    print("omega      y   |F-1|        referee d/s        rust d/s           d/s     cover_d cover_s nodes")
    for w, y in GRID:
        iref = slice_raw_from_nd(w, y)
        fref = w * iref / (2j * math.pi)
        scale = abs(fref - 1.0)
        f = fpl.PointMassField([(0.0, 0.0, 1.0)], beta=(y, 0.0))
        d = f.solve(omega=w)
        s = f.solve(omega=w, subtract_free=True)
        e_d = abs(w * d.integral / (2j * math.pi) - fref) / scale
        e_s = abs(w * s.integral / (2j * math.pi) - fref) / scale
        cover_d = d.error / max(abs(d.integral - iref), 1e-300)
        cover_s = s.error / max(abs(s.integral - iref), 1e-300)
        rrow = ref.get((w, y), {})
        rd = float(rrow.get("rel_err_Fm1_direct", "nan"))
        rs = float(rrow.get("rel_err_Fm1_subtracted", "nan"))
        print(f"{w:8.1e} {y:4.1f}  {scale:9.2e}   {rd:9.2e}/{rs:9.2e}   "
              f"{e_d:9.2e}/{e_s:9.2e}   {e_d / e_s:6.2f}  "
              f"{cover_d:7.1f} {cover_s:7.1f} {d.steps:5}/{s.steps:<5}")
        rows.append(dict(omega=w, y=y, rel_err_Fm1_ref_direct=rd,
                         rel_err_Fm1_ref_subtracted=rs,
                         rel_err_Fm1_direct=e_d, rel_err_Fm1_subtracted=e_s,
                         cover_direct=cover_d, cover_subtracted=cover_s,
                         nodes_direct=d.steps, nodes_subtracted=s.steps))

    path = os.path.join(RESULTS, "low_omega_rust.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    worst_d = max(r["rel_err_Fm1_direct"] for r in rows)
    worst_s = max(r["rel_err_Fm1_subtracted"] for r in rows)
    print(f"worst F-1 rel err: direct {worst_d:.2e} / subtracted {worst_s:.2e}")
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
