"""P4.1: the *installed Rust* high-omega hybrid — coverage table.

Runs the `fastpiclef` bindings (not the numpy prototype in `sp_correction.py`):

* **Part A (complex truth, our convention)** — for a point mass, compare the
  corrected geometric branch (`geometric_corrected` / the hybrid's fast branch)
  against the exact RC engine at omega = 30 and 100: the achieved error must be
  covered by the reported estimate (the coverage claim of the prototype, now on
  the Rust tensors), and must beat the bare geometric sum.
* **Part B (external truth, |F|)** — against the Nakamura-Deguchi closed form
  over omega = 30 .. 3000 (where the RC would be expensive): the branch the
  hybrid takes, the relative errors of the geometric and corrected sums, and
  the estimate coverage.
* **Part C (macro + multi-mass)** — the quadratic-phase invariance (C = 0) and
  the point-mass-field path vs the general LensField path.

Writes `results/hybrid_check.csv`.

Run: `pixi run -e comparison python -m comparison.hybrid_check`
"""

import os

os.environ.setdefault("RAYON_NUM_THREADS", "1")

import csv
import math

import fastpiclef as fpl

from .referee import point_mass_exact

RESULTS = os.path.join(os.path.dirname(__file__), "results")
TAU = 2.0 * math.pi


def part_a(rows):
    print("== A: complex truth vs the exact RC (point mass, y = 0.1, 0.5, 1.0) ==")
    for y in (0.1, 0.5, 1.0):
        field = fpl.PointMassField([(0.0, 0.0, 1.0)], beta=(y, 0.0))
        for w in (30.0, 100.0):
            rc = field.solve(omega=w)
            geo = field.geometric(omega=w)
            corr = field.geometric_corrected(omega=w)
            truth = rc.amplification
            r_geo = abs(geo.amplification - truth)
            r_corr = abs(corr.amplification - truth)
            est_f = corr.error * w / TAU  # integral budget -> amplification budget
            cover = est_f / max(r_corr, 1e-300)
            rows.append(
                dict(
                    part="A",
                    case=f"point y={y}",
                    omega=w,
                    method="geometric+correction",
                    rel_corr=r_corr / abs(truth),
                    rel_geo=r_geo / abs(truth),
                    est=est_f,
                    cover=cover,
                    rc_steps=rc.steps,
                    fast_steps=corr.steps,
                )
            )
            print(
                f"  y={y} w={w:6.0f}: geo {r_geo / abs(truth):.2e}  "
                f"corr {r_corr / abs(truth):.2e}  est {est_f:.2e}  cover {cover:7.2f}x  "
                f"RC steps {rc.steps} -> {corr.steps} image solves"
            )


def part_b(rows):
    print("== B: |F| vs the ND closed form (no RC) ==")
    for y in (0.1, 0.5, 1.0, 2.0):
        field = fpl.PointMassField([(0.0, 0.0, 1.0)], beta=(y, 0.0))
        for w in (30.0, 100.0, 300.0, 1000.0, 3000.0):
            fnd = abs(point_mass_exact(w, y))
            geo = abs(field.geometric(omega=w).amplification)
            hyb = field.hybrid(omega=w, tol=1e-2)
            fh = abs(hyb.amplification)
            rel_corr = abs(fh - fnd) / fnd
            rel_geo = abs(geo - fnd) / fnd
            est_f = hyb.error * w / TAU if hyb.method != "slice" else float("nan")
            cover = est_f / max(rel_corr, 1e-300)
            rows.append(
                dict(
                    part="B",
                    case=f"point y={y}",
                    omega=w,
                    method=hyb.method,
                    rel_corr=rel_corr,
                    rel_geo=rel_geo,
                    est=est_f,
                    cover=cover,
                    rc_steps=0,
                    fast_steps=hyb.steps,
                )
            )
            print(
                f"  y={y} w={w:6.0f}: [{hyb.method:>20s}] geo {rel_geo:.2e}  "
                f"corr {rel_corr:.2e}  est {est_f:.2e}  cover {cover:8.2f}x"
            )


def part_c(rows):
    print("== C: macro + multi-mass ==")
    lf = fpl.LensField([], beta=(0.4, -0.3), kappa=0.3, gamma1=0.2)
    g = lf.geometric(omega=10.0)
    c = lf.geometric_corrected(omega=10.0)
    d = abs(c.amplification - g.amplification) / abs(g.amplification)
    print(f"  macro-only (C = 0): |F_corr - F_geo|/|F| = {d:.2e}, est = {c.error:.2e}")
    rows.append(
        dict(part="C", case="macro-only", omega=10.0, method="geometric+correction",
             rel_corr=d, rel_geo=0.0, est=c.error, cover=float("nan"), rc_steps=0, fast_steps=c.steps)
    )
    lens = [(0.6, 0.0, 0.4), (-0.5, 0.4, 0.3), (0.1, -0.6, 0.35)]
    src = (0.2, 0.1)
    pm = fpl.PointMassField(lens, beta=src)
    comps = [fpl.PointComp(x, y, m) for (x, y, m) in lens]
    lfx = fpl.LensField(comps, beta=src)
    for w in (100.0, 300.0):
        h = pm.hybrid(omega=w, tol=1e-1)
        est_f = h.error * w / TAU
        rc = pm.solve(omega=w) if w <= 100.0 else None
        if rc is not None:
            r = abs(h.amplification - rc.amplification)
            cover = est_f / max(r, 1e-300)
            print(
                f"  3 masses w={w:6.0f}: [{h.method}] residual vs RC {r:.2e}  "
                f"est {est_f:.2e}  cover {cover:7.2f}x  RC steps {rc.steps} -> {h.steps}"
            )
            rows.append(
                dict(part="C", case="3 masses", omega=w, method=h.method, rel_corr=r,
                     rel_geo=float("nan"), est=est_f, cover=cover, rc_steps=rc.steps,
                     fast_steps=h.steps)
            )
        else:
            print(f"  3 masses w={w:6.0f}: [{h.method}] est {est_f:.2e} (no RC at this omega)")
            rows.append(
                dict(part="C", case="3 masses", omega=w, method=h.method, rel_corr=float("nan"),
                     rel_geo=float("nan"), est=est_f, cover=float("nan"), rc_steps=0,
                     fast_steps=h.steps)
            )
    # the polynomial path must agree with the general LensField path
    a = pm.hybrid(omega=300.0, tol=1e-1)
    b = lfx.hybrid(omega=300.0, tol=1e-1)
    d = abs(a.integral - b.integral) / abs(b.integral)
    print(f"  path agreement at w=300: |I_pm - I_lf|/|I| = {d:.2e}")
    assert d < 1e-9, d


def main():
    rows = []
    part_a(rows)
    part_b(rows)
    part_c(rows)
    path = os.path.join(RESULTS, "hybrid_check.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(
            fh,
            fieldnames=["part", "case", "omega", "method", "rel_corr", "rel_geo", "est",
                        "cover", "rc_steps", "fast_steps"],
        )
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
