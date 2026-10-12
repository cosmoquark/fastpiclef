"""P4.1 reconnaissance: the exact field solver's cost and accuracy vs omega.

The RC cost grew ~omega^1.5 between omega = 30 and 100 (S3/P0).  This script
measures the growth to omega = 1000 (single-threaded in-process + a threaded
subprocess), the accuracy against the external Nakamura-Deguchi closed form for
the point mass, the geometric solver's error at the same frequencies (the
asymptotic anchor), and the work breakdown (`steps` = inner solves per outer
node set; wall/steps = per-inner-call cost).

Writes `comparison/results/high_omega.csv`.

Run: `pixi run -e comparison python -m comparison.high_omega`
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

from .referee import slice_raw_from_nd

RESULTS = os.path.join(os.path.dirname(__file__), "results")

OMEGAS = (10.0, 30.0, 100.0, 300.0, 1000.0)
FIELD_LENSES = [(0.6, 0.0, 0.4), (-0.5, 0.4, 0.3), (0.1, -0.6, 0.35)]
FIELD_SOURCE = (0.2, 0.1)


def _point(y):
    return fpl.PointMassField([(0.0, 0.0, 1.0)], beta=(y, 0.0))


def _to_F(I, omega):
    return I * omega / (2.0 * math.pi * 1j)


def measure_point(omega, y=1.0, reps=1):
    """Single-threaded point-mass solve: (value, wall, steps, err vs ND, geo err)."""
    truth = slice_raw_from_nd(omega, y)
    r = None
    t0 = time.perf_counter()
    for _ in range(reps):
        r = _point(y).solve(omega=omega, tol=1e-14)
    wall = (time.perf_counter() - t0) / reps
    err = abs(complex(r.integral) - truth) / abs(truth)
    g = _point(y).geometric(omega)
    geo_err = abs(abs(g.amplification) - abs(_to_F(truth, omega))) / abs(_to_F(truth, omega))
    return complex(r.integral), wall, int(r.steps), err, geo_err


def measure_field(omega, reps=1):
    r = None
    t0 = time.perf_counter()
    for _ in range(reps):
        r = fpl.PointMassField(FIELD_LENSES, beta=FIELD_SOURCE).solve(omega=omega, tol=1e-14)
    wall = (time.perf_counter() - t0) / reps
    g = fpl.PointMassField(FIELD_LENSES, beta=FIELD_SOURCE).geometric(omega)
    return wall, int(r.steps), complex(g.amplification)


def threaded_time(code, timeout=7200):
    """Run a snippet in a subprocess with the thread pins removed."""
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(RESULTS)))
    env = {k: v for k, v in os.environ.items()
           if k not in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=env, cwd=root, timeout=timeout, check=True)
    return float(out.stdout.strip().splitlines()[-1])


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    print("== point mass, y=1 (external ND truth) ==")
    print(f"{'omega':>7} {'wall [s]':>10} {'steps':>7} {'s/step [ms]':>12} "
          f"{'err vs ND':>10} {'geo err':>10}")
    for w in OMEGAS:
        v, wall, steps, err, geo = measure_point(w)
        per = 1e3 * wall / max(steps, 1)
        print(f"{w:7.0f} {wall:10.3f} {steps:7d} {per:12.3f} {err:10.1e} {geo:10.1e}")
        rows.append(dict(case="point_mass", omega=w, threads=1, wall_s=wall, steps=steps,
                         per_step_ms=per, rel_err_nd=err, geometric_rel_err=geo))
    print("== 3-mass field ==")
    for w in (10.0, 30.0, 100.0, 300.0):
        wall, steps, geo = measure_field(w)
        print(f"{w:7.0f} {wall:10.3f} {steps:7d}")
        rows.append(dict(case="field3", omega=w, threads=1, wall_s=wall, steps=steps,
                         per_step_ms=1e3 * wall / max(steps, 1),
                         rel_err_nd=float("nan"), geometric_rel_err=float("nan")))
    print("== threaded (12 cores) point mass ==")
    for w in (100.0, 300.0, 1000.0):
        code = (
            "import time, fastpiclef as fpl\n"
            f"f = fpl.PointMassField([(0.0, 0.0, 1.0)], beta=(1.0, 0.0))\n"
            f"f.solve(omega={w!r})\n"
            "t0 = time.perf_counter()\n"
            f"f.solve(omega={w!r})\n"
            "print(time.perf_counter() - t0)\n"
        )
        wall = threaded_time(code)
        print(f"{w:7.0f} {wall:10.3f} (threaded)")
        rows.append(dict(case="point_mass", omega=w, threads=os.cpu_count(), wall_s=wall,
                         steps=0, per_step_ms=float("nan"), rel_err_nd=float("nan"),
                         geometric_rel_err=float("nan")))

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "high_omega.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    for case in ("point_mass", "field3"):
        g = df[(df.case == case) & (df.threads == 1)].sort_values("omega")
        if len(g) >= 3:
            x = np.log10(g["omega"].to_numpy())
            y = np.log10(g["wall_s"].to_numpy())
            slope = float(np.polyfit(x, y, 1)[0])
            print(f"  {case}: wall ~ omega^{slope:.2f} (single-threaded, "
                  f"{g['omega'].min():g}..{g['omega'].max():g})")


if __name__ == "__main__":
    main()
