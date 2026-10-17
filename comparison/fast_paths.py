"""P4.4: the isolated-lens fast-path table (timing, single thread).

The report's fast-path guidance in one reproducible table:

* the ND closed form (`point_mass_nd`) --- microseconds, inside its band
  $|z| = omega y^2/2 <= ~10$ (its error estimate is the in-band honesty
  check);
* the polar 1D path (`PointMass(y).solve(method="polar")`) --- the high-omega
  route (~tens of ms at omega = 10, ~hundreds at 100), but seconds and loose
  at omega <= 3;
* the exact 2D RC slice path (`PointMassField(...).solve`) --- the low-omega
  route (~0.1-0.4 s at omega <= 3, seconds at 10-100).

All walls are single-threaded (`RAYON_NUM_THREADS=1`) and the machine load is
recorded per row; the ND rows use many reps (microsecond calls), the
polar/RC rows a best-of-blocks at seconds-or-less.  The `rel_nd` column
compares *moduli* against `point_mass_nd` where the series is in band (the
phase conventions differ: the ND form carries the Fermat-offset phase).

Writes `results/fast_paths.csv`.

Run: `pixi run -e comparison fast-paths`
"""

import os

os.environ.setdefault("RAYON_NUM_THREADS", "1")

import csv
import time

import fastpiclef as fpl

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def loadavg() -> float:
    with open("/proc/loadavg") as fh:
        return float(fh.read().split()[0])


def best_wall(fn, reps: int, blocks: int) -> float:
    """Best block's mean wall per call (sheds scheduler noise)."""
    best = float("inf")
    for _ in range(blocks):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        best = min(best, (time.perf_counter() - t0) / reps)
    return best


def main():
    rows = []

    print("== ND closed form (the microsecond path) ==")
    for w, y in ((1.0, 1.0), (10.0, 1.0), (10.0, 0.3), (100.0, 0.3), (1.0, 4.0)):
        z = 0.5 * w * y * y
        r = fpl.point_mass_nd(omega=w, y=y)
        wall = best_wall(lambda w=w, y=y: fpl.point_mass_nd(omega=w, y=y), reps=1000, blocks=3)
        print(f"  w={w:7.2f} y={y:4.2f}  |z|={z:5.2f}  {1e6 * wall:8.2f} us   est {r.error:.2e}")
        rows.append(dict(path="nd", omega=w, y=y, z=z, wall_s=wall, reps=1000, blocks=3,
                         rel_nd=0.0, budget=r.error, steps=r.steps, loadavg=loadavg(),
                         note="closed form"))

    print("== polar 1D path, y = 1 (the high-omega route) ==")
    nd = {w: fpl.point_mass_nd(omega=w, y=1.0) for w in (1.0, 10.0, 100.0)}
    for w in (1.0, 10.0, 100.0):
        blocks = 1 if w >= 100.0 else 2
        r = fpl.PointMass(1.0).solve(omega=w, method="polar")
        wall = best_wall(lambda w=w: fpl.PointMass(1.0).solve(omega=w, method="polar"),
                         reps=1, blocks=blocks)
        rel = abs(abs(r.amplification) - abs(nd[w].amplification)) / abs(nd[w].amplification)
        print(f"  w={w:7.2f}  {1e3 * wall:9.2f} ms   |mod| vs ND {rel:.2e}   "
              f"est {r.error:.2e}  steps {r.steps}")
        rows.append(dict(path="polar", omega=w, y=1.0, z=0.5 * w, wall_s=wall, reps=1,
                         blocks=blocks, rel_nd=rel, budget=r.error, steps=r.steps,
                         loadavg=loadavg(), note="1D thimble"))

    print("== exact 2D RC (slice path), y = 1 (the low-omega route) ==")
    for w in (0.3, 1.0, 3.0, 10.0, 100.0):
        blocks = 1 if w >= 10.0 else 2

        def solve(w=w):
            f = fpl.PointMassField([(0.0, 0.0, 1.0)], beta=(1.0, 0.0))
            return f.solve(omega=w)

        r = solve()
        wall = best_wall(solve, reps=1, blocks=blocks)
        ref = nd.get(w)
        rel = (abs(abs(r.amplification) - abs(ref.amplification)) / abs(ref.amplification)
               if ref is not None else float("nan"))
        print(f"  w={w:7.2f}  {1e3 * wall:9.2f} ms   |mod| vs ND {rel:.2e}   "
              f"est {r.error:.2e}  steps {r.steps}")
        rows.append(dict(path="rc-slice", omega=w, y=1.0, z=0.5 * w, wall_s=wall, reps=1,
                         blocks=blocks, rel_nd=rel, budget=r.error, steps=r.steps,
                         loadavg=loadavg(), note="exact"))

    path = os.path.join(RESULTS, "fast_paths.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(
            fh,
            fieldnames=["path", "omega", "y", "z", "wall_s", "reps", "blocks", "rel_nd",
                        "budget", "steps", "loadavg", "note"],
        )
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
