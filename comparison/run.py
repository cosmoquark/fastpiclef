"""Comparison harness: fastpiclef vs the reference codes.

Phase 1 — fastpiclef (thimble + mesh) vs plind (PL flow), 1D. Single-threaded
for a fair core comparison (threaded speed is a separate axis). Emits a summary
table and the headline error-vs-wall-time convergence curve.

Run: `pixi run -e comparison bench`  (or `python -m comparison.run`).
"""

import os

# Pin to one thread BEFORE importing the solvers (rayon/OpenMP read these at
# pool init). Threaded performance is reported separately, later.
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd

from . import drivers
from .cases import CASES, CONVERGENCE_CASE
from .metrics import timed

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def rel(value, truth):
    return abs(value - truth) / abs(truth)


def to_md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = [f"{r[c]:.3e}" if isinstance(r[c], float) else str(r[c]) for c in cols]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def summary_table():
    rows = []
    for case in CASES:
        t = case.truth
        v, s = timed(lambda: drivers.fastpiclef_run(case.builder, case.omega, "auto"))
        rows.append([case.name, "fastpiclef/auto", rel(v, t), s])
        v, s = timed(
            lambda: drivers.fastpiclef_run(
                case.builder, case.omega, "mesh", delta=0.01, domain=(-8.0, 8.0)
            )
        )
        rows.append([case.name, "fastpiclef/mesh", rel(v, t), s])
        try:
            v, s = timed(lambda: drivers.plind_run(case.phi, case.dphi, case.omega), reps=3)
            rows.append([case.name, "plind", rel(v, t), s])
        except Exception as e:  # noqa: BLE001
            print(f"  plind FAILED on {case.name}: {type(e).__name__}: {e}")
            rows.append([case.name, "plind", float("nan"), float("nan")])
    return pd.DataFrame(rows, columns=["case", "solver", "rel_err", "time_s"])


def convergence_curve():
    """Error vs wall-time for the two PL-flow methods (apples-to-apples), with
    the fastpiclef thimble path shown as a single machine-precise point."""
    case = CONVERGENCE_CASE
    t = case.truth
    series = {}

    pts = []
    for delta in (0.06, 0.04, 0.02, 0.01, 0.005):
        v, s = timed(
            lambda d=delta: drivers.fastpiclef_run(
                case.builder, case.omega, "mesh", delta=d, domain=(-8.0, 8.0)
            )
        )
        pts.append((s, rel(v, t)))
    series["fastpiclef / mesh"] = pts

    pts = []
    for pd_delta in (1.2, 0.8, 0.5, 0.3, 0.2):
        try:
            v, s = timed(
                lambda d=pd_delta: drivers.plind_run(
                    case.phi, case.dphi, case.omega, N=2001, delta=d
                ),
                reps=3,
            )
            pts.append((s, rel(v, t)))
        except Exception as e:  # noqa: BLE001
            print(f"  plind convergence point delta={pd_delta} FAILED: {e}")
    series["plind"] = pts

    v, s = timed(lambda: drivers.fastpiclef_run(case.builder, case.omega, "auto"))
    thimble_pt = (s, rel(v, t))
    return case, series, thimble_pt


def plot_convergence(ax, case, series, thimble_pt):
    """Render the error-vs-cost curve on a given Matplotlib axis (reused by
    the CLI and the results notebook)."""
    for name, pts in series.items():
        if pts:
            xs, ys = zip(*sorted(pts))
            ax.plot(xs, ys, marker="o", label=name)
    # The thimble IS the reference here (error vs itself = 0); show it at a
    # nominal floor (its test-suite-validated ~1e-13 level) so the marker lands.
    ax.plot(thimble_pt[0], max(thimble_pt[1], 1e-13), marker="*", ms=16, ls="",
            color="C3", label="fastpiclef / thimble (reference)")
    ax.set(xscale="log", yscale="log", xlabel="wall time  [s]", ylim=(3e-15, 3e-7),
           ylabel="relative error", title=f"Error vs cost — {case.name}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(RESULTS, exist_ok=True)
    print("== Reference sanity: thimble truth vs brute-force referee ==")
    for case in CASES:
        print(f"  {case.name:<28} |thimble - referee| = {rel(case.truth, case.referee):.2e}")

    print("\n== Summary (single-threaded; truth = validated thimble / closed form) ==")
    df = summary_table()
    with pd.option_context("display.float_format", lambda x: f"{x:.3e}"):
        print(df.to_string(index=False))
    df.to_csv(os.path.join(RESULTS, "summary.csv"), index=False)
    with open(os.path.join(RESULTS, "summary.md"), "w") as f:
        f.write(to_md(df))

    print("\n== Convergence (error vs wall-time) ==")
    case, series, thimble_pt = convergence_curve()
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    plot_convergence(ax, case, series, thimble_pt)
    fig.tight_layout()
    out = os.path.join(RESULTS, "convergence.png")
    fig.savefig(out, dpi=120)
    print(f"  thimble: err={thimble_pt[1]:.2e} at {thimble_pt[0]*1e3:.2f} ms")
    print(f"  wrote {out} and results/summary.{{csv,md}}")


if __name__ == "__main__":
    main()
