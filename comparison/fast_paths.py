"""P4.4: the isolated-lens fast-path table (timing, single thread).

The report's fast-path guidance in one reproducible table:

* the ND closed form (`point_mass_nd`) --- microseconds, but only inside its
  honest band: the accuracy driver is the compound parameter `omega*y`, not
  `|z|` alone (see `nd_band.csv` for the map and `nd_band.py` for the story);
* the polar 1D path (`PointMass(y).solve(method="polar")`) --- the high-omega
  route, seconds and loose at low omega;
* the exact 2D RC slice path (`PointMassField(...).solve`) --- the low-omega
  route, exact (the truth of this table at `tol = 1e-14`).

All walls are single-threaded (`RAYON_NUM_THREADS=1`) and the machine load is
recorded per row; the ND rows use many reps (microsecond calls), the
polar/RC rows a best-of-blocks at seconds-or-less.  `rel_rc` is the modulus
error against the *tight* exact RC (so every row is judged against one truth,
including the ND rows --- which is how the omega*y band shows up here); where
the path reports its own error budget, `cover` is the budget over the true
error.

Writes `results/fast_paths.csv`.

Run: `pixi run -e comparison fast-paths`
"""

import os

os.environ.setdefault("RAYON_NUM_THREADS", "1")

import csv
import time

import fastpiclef as fpl

RESULTS = os.path.join(os.path.dirname(__file__), "results")
TIGHT = 1e-14


def loadavg() -> float:
    with open("/proc/loadavg") as fh:
        return float(fh.read().split()[0])


def best_wall(fn, reps: int, blocks: int):
    """Best block's mean wall per call; returns (wall, last result)."""
    best = float("inf")
    r = None
    for _ in range(blocks):
        t0 = time.perf_counter()
        for _ in range(reps):
            r = fn()
        best = min(best, (time.perf_counter() - t0) / reps)
    return best, r


def main():
    rows = []
    truth = {}

    def tight(w: float, y: float):
        if (w, y) not in truth:
            truth[(w, y)] = fpl.PointMassField([(0.0, 0.0, 1.0)],
                                               beta=(y, 0.0)).solve(omega=w, tol=TIGHT)
        return truth[(w, y)]

    def rel(r, w: float, y: float):
        t = abs(tight(w, y).amplification)
        return abs(abs(r.amplification) - t) / t

    print("== ND closed form (the microsecond path) ==")
    for w, y in ((1.0, 1.0), (10.0, 1.0), (10.0, 0.3), (100.0, 0.3), (1.0, 4.0)):
        wall, r = best_wall(lambda w=w, y=y: fpl.point_mass_nd(omega=w, y=y),
                            reps=1000, blocks=3)
        rc = rel(r, w, y)
        cover = r.error / max(rc, 1e-300)
        print(f"  w={w:7.2f} y={y:4.2f}  |z|={0.5 * w * y * y:5.2f}  w*y={w * y:5.1f}  "
              f"{1e6 * wall:7.2f} us   true {rc:.2e}  est {r.error:.2e}  cover {cover:.1f}")
        rows.append(dict(path="nd", omega=w, y=y, z=0.5 * w * y * y, omega_y=w * y,
                         wall_s=wall, reps=1000, blocks=3, rel_rc=rc, cover=cover,
                         budget=r.error, steps=r.steps, loadavg=loadavg(),
                         note="closed form"))

    print("== polar 1D path, y = 1 (the high-omega route) ==")
    for w in (0.3, 1.0, 10.0, 100.0):
        blocks = 1 if w >= 100.0 else 2
        wall, r = best_wall(lambda w=w: fpl.PointMass(1.0).solve(omega=w, method="polar"),
                            reps=1, blocks=blocks)
        rc = rel(r, w, 1.0)
        cover = r.error / max(rc, 1e-300)
        print(f"  w={w:7.2f}  {1e3 * wall:9.2f} ms   true {rc:.2e}  est {r.error:.2e}  "
              f"cover {cover:.1f}  steps {r.steps}")
        rows.append(dict(path="polar", omega=w, y=1.0, z=0.5 * w, omega_y=w,
                         wall_s=wall, reps=1, blocks=blocks, rel_rc=rc, cover=cover,
                         budget=r.error, steps=r.steps, loadavg=loadavg(),
                         note="1D thimble"))

    print("== exact 2D RC (slice path, default tol), y = 1 (the low-omega route) ==")
    for w in (0.3, 1.0, 3.0, 10.0, 100.0):
        blocks = 1 if w >= 10.0 else 2

        def solve(w=w):
            return fpl.PointMassField([(0.0, 0.0, 1.0)], beta=(1.0, 0.0)).solve(omega=w)

        wall, r = best_wall(solve, reps=1, blocks=blocks)
        rc = rel(r, w, 1.0)
        print(f"  w={w:7.2f}  {1e3 * wall:9.2f} ms   default-vs-tight {rc:.2e}  "
              f"est {r.error:.2e}  steps {r.steps}")
        rows.append(dict(path="rc-slice", omega=w, y=1.0, z=0.5 * w, omega_y=w,
                         wall_s=wall, reps=1, blocks=blocks, rel_rc=rc,
                         cover=float("nan"), budget=r.error, steps=r.steps,
                         loadavg=loadavg(), note="truth = same solve at tol 1e-14"))

    path = os.path.join(RESULTS, "fast_paths.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(
            fh,
            fieldnames=["path", "omega", "y", "z", "omega_y", "wall_s", "reps", "blocks",
                        "rel_rc", "cover", "budget", "steps", "loadavg", "note"],
        )
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
