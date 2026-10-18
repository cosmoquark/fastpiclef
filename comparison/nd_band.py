"""P4.4: the honest band of the Nakamura-Deguchi series (the ND fast path).

The ND closed form (`point_mass_nd`) is the microsecond path of the isolated
lens.  The documented criterion used to be `|z| = omega y^2/2 <= ~10`; a
quiet-window probe on 2026-10-18 found the true accuracy driver is the
compound series parameter `omega*y` --- at fixed `|z|` the error grows with
omega (e.g. omega = 100, y = 0.3: `|z| = 4.5`, but only ~5e-4 true accuracy;
omega = 10, y = 1: `|z| = 5`, 5.6e-13).  This script maps the band against
the exact RC truth.

For each `(omega, y)`: the ND row's own error estimate, the true |mod| error
against the exact RC slice solve at `tol = 1e-14`, the estimate's coverage,
the ND wall (best-of-blocks, microseconds) and the RC wall.  Contrast pairs
share `omega*y` across different `(omega, y)` splits to separate the drivers.

Writes `results/nd_band.csv`.

Run: `pixi run -e comparison nd-band`
"""

import os

os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv
import time

import fastpiclef as fpl

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def loadavg() -> float:
    with open("/proc/loadavg") as fh:
        return float(fh.read().split()[0])


def best_wall(fn, reps: int, blocks: int = 3):
    """Best block's mean wall per call; returns (wall, last result)."""
    best = float("inf")
    r = None
    for _ in range(blocks):
        t0 = time.perf_counter()
        for _ in range(reps):
            r = fn()
        best = min(best, (time.perf_counter() - t0) / reps)
    return best, r


def rc_truth(w: float, y: float):
    """The exact RC row: its modulus is the truth, its own wall the reference."""
    t0 = time.perf_counter()
    r = fpl.PointMassField([(0.0, 0.0, 1.0)], beta=(y, 0.0)).solve(omega=w, tol=1e-14)
    return r, time.perf_counter() - t0


def main():
    # (omega, y) pairs: the diagonal through the old |z| claim plus omega*y
    # contrast pairs at 20 and 30, and the cliff beyond.
    pairs = [
        (2.0, 1.0), (5.0, 1.0), (8.0, 1.0), (10.0, 1.0), (12.0, 1.0),
        (15.0, 1.0), (20.0, 1.0), (25.0, 1.0), (30.0, 1.0), (40.0, 1.0),
        (40.0, 0.5), (60.0, 0.35), (60.0, 0.5), (100.0, 0.2), (100.0, 0.3),
    ]
    rows = []
    print(f"{'omega':>7} {'y':>5} {'omega*y':>8} {'|z|':>6} {'nd wall [us]':>12} "
          f"{'true err':>10} {'nd est':>10} {'cover':>8} {'rc [s]':>7}")
    for w, y in pairs:
        z = 0.5 * w * y * y
        r = fpl.point_mass_nd(omega=w, y=y)
        wall, _ = best_wall(lambda w=w, y=y: fpl.point_mass_nd(omega=w, y=y),
                            reps=1000)
        truth, rc_wall = rc_truth(w, y)
        rel = abs(abs(r.amplification) - abs(truth.amplification)) / abs(truth.amplification)
        cover = r.error / max(rel, 1e-300)
        print(f"{w:7.1f} {y:5.2f} {w * y:8.2f} {z:6.2f} {1e6 * wall:12.2f} "
              f"{rel:10.2e} {r.error:10.2e} {cover:8.1f} {rc_wall:7.2f}")
        rows.append(dict(omega=w, y=y, omega_y=w * y, z=z, nd_wall_s=wall,
                         true_err=rel, nd_est=r.error, cover=cover,
                         rc_wall_s=rc_wall, loadavg=loadavg()))

    path = os.path.join(RESULTS, "nd_band.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["omega", "y", "omega_y", "z", "nd_wall_s",
                                            "true_err", "nd_est", "cover", "rc_wall_s",
                                            "loadavg"])
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
