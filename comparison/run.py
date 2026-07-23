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

import math
import time

import numpy as np
import pandas as pd

import fastpiclef as fpl

from . import drivers, plpi_driver
from .cases import CASES, CONVERGENCE_CASE
from .metrics import timed

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def _normF(integral, omega):
    """Normalised amplification F = integral / (2*pi*i/omega)^{1/2} (free = 1)."""
    fs = (2 * math.pi / omega) ** 0.5 * np.exp(1j * math.pi / 4)
    return np.asarray(integral) / fs


def grid_comparison(plind_points=15):
    """Phase 2 — the (mu)-grid protocol at P-LPI's native benchmark
    (RationalLens kappa=-2, nu=10). Returns everything the tables/plots need.

    fastpiclef is the truth (validated thimble); P-LPI and plind are graded
    against it across the grid. plind is subsampled (it is ~2 s/point here) and
    its full-grid time is extrapolated.
    """
    kappa, omega = plpi_driver.PLPI_KAPPA, plpi_driver.PLPI_NU
    mu, F_plpi_raw, plpi_t1 = plpi_driver.run_plpi_1d(threads=1)
    _, _, plpi_tN = plpi_driver.run_plpi_1d(threads=os.cpu_count())

    # fastpiclef over the whole grid (independent solves — no warm-start yet).
    t0 = time.perf_counter()
    F_fpl_raw = np.array(
        [complex(fpl.RationalLens(beta=float(m), kappa=kappa)
                 .solve(omega=omega, method="auto").integral) for m in mu]
    )
    fpl_t = time.perf_counter() - t0

    # plind, subsampled; per-point time -> extrapolated grid time.
    idx = np.linspace(0, len(mu) - 1, plind_points).round().astype(int)
    phi = lambda z, b: 0.5 * (z - b) ** 2 - kappa / (1 + z * z)
    dphi = lambda z, b: (z - b) + 2 * kappa * z / (1 + z * z) ** 2
    F_plind_raw, plind_times = [], []
    for j in idx:
        b = float(mu[j])
        t0 = time.perf_counter()
        v = drivers.plind_run(lambda z: phi(z, b), lambda z: dphi(z, b), omega, N=2001, delta=0.5)
        plind_times.append(time.perf_counter() - t0)
        F_plind_raw.append(v)
    F_plind_raw = np.array(F_plind_raw)

    err_plpi = np.abs(F_plpi_raw - F_fpl_raw) / np.abs(F_fpl_raw)
    err_plind = np.abs(F_plind_raw - F_fpl_raw[idx]) / np.abs(F_fpl_raw[idx])
    plind_per_pt = float(np.median(plind_times))
    return {
        "mu": mu, "idx": idx, "omega": omega,
        "F_fpl": _normF(F_fpl_raw, omega),
        "F_plpi": _normF(F_plpi_raw, omega),
        "F_plind": _normF(F_plind_raw, omega),
        "err_plpi": err_plpi, "err_plind": err_plind,
        "times": {
            "fastpiclef (512, 1 thr)": fpl_t,
            "P-LPI (512, 1 thr)": plpi_t1,
            f"P-LPI (512, {os.cpu_count()} thr)": plpi_tN,
            "plind/pt": plind_per_pt,
            "plind (512, extrapolated)": plind_per_pt * len(mu),
        },
    }


def plot_grid_overlay(ax, res):
    mu = res["mu"]
    ax.plot(mu, np.abs(res["F_fpl"]), lw=1.6, label="fastpiclef (reference)")
    ax.plot(mu, np.abs(res["F_plpi"]), lw=1, ls="--", label="P-LPI")
    ax.plot(mu[res["idx"]], np.abs(res["F_plind"]), "o", ms=4, label="plind (subsampled)")
    ax.set(xlabel="source position  mu", ylabel="|F|  amplification",
           title=f"Amplification across the grid (kappa=-2, nu={res['omega']:g}) — 3-way agreement")
    ax.legend()


def plot_grid_error(ax, res):
    ax.semilogy(res["mu"], np.maximum(res["err_plpi"], 1e-16), lw=1, label="P-LPI")
    ax.semilogy(res["mu"][res["idx"]], np.maximum(res["err_plind"], 1e-16), "o", ms=4, label="plind")
    ax.set(xlabel="source position  mu", ylabel="rel. error vs fastpiclef thimble",
           title="Accuracy across the grid", ylim=(1e-9, 1e-1))
    ax.legend()


def freq_comparison(y=1.0, w_grid=None, fiona_threads=None):
    """Phase 3 / C1 -- FIONA's native frequency-sweep protocol on the isolated
    point mass psi = ln r. BOTH codes are graded against the Nakamura-Deguchi
    closed form (external truth -- no circularity). The story is the high-w tail:
    the thimble stays O(1) and machine-precise while FIONA's node count (hence
    cost) grows and its accuracy degrades.
    """
    from . import fiona_driver
    from .referee import point_mass_exact

    if w_grid is None:
        w_grid = np.geomspace(0.5, 30.0, 16)
    w_grid = np.asarray(w_grid, dtype=float)

    F_nd = np.array([point_mass_exact(w, y) for w in w_grid])

    # fastpiclef: independent per-frequency polar solves (no warm-start).
    t0 = time.perf_counter()
    F_fpl = np.array(
        [complex(fpl.PointMass(y).solve(omega=float(w), method="polar").amplification)
         for w in w_grid]
    )
    fpl_t = time.perf_counter() - t0

    # FIONA: adaptive NUFFT3 batched over the grid (its native fast mode).
    F_fiona, fiona_t = fiona_driver.fiona_point_mass_sweep(y, w_grid, threads=1)

    err_fpl = np.abs(np.abs(F_fpl) - np.abs(F_nd)) / np.abs(F_nd)
    err_fiona = np.abs(np.abs(F_fiona) - np.abs(F_nd)) / np.abs(F_nd)
    times = {
        f"fastpiclef ({w_grid.size} w, 1 thr)": fpl_t,
        f"FIONA ({w_grid.size} w, 1 thr)": fiona_t,
    }
    if fiona_threads and fiona_threads > 1:
        _, fiona_tN = fiona_driver.fiona_point_mass_sweep(y, w_grid, threads=fiona_threads)
        times[f"FIONA ({w_grid.size} w, {fiona_threads} thr)"] = fiona_tN
    return {
        "y": y, "w": w_grid,
        "F_fpl": F_fpl, "F_fiona": F_fiona, "F_nd": F_nd,
        "err_fpl": err_fpl, "err_fiona": err_fiona, "times": times,
    }


def plot_freq_overlay(ax, res):
    w = res["w"]
    ax.plot(w, np.abs(res["F_nd"]), "k-", lw=2.4, alpha=0.35, label="Nakamura-Deguchi (exact)")
    ax.plot(w, np.abs(res["F_fpl"]), lw=1.5, label="fastpiclef (thimble)")
    ax.plot(w, np.abs(res["F_fiona"]), "--", lw=1.2, label="FIONA (NUFFT3)")
    ax.set(xscale="log", xlabel="frequency  w", ylabel="|F|  amplification",
           title=f"Point mass |F(w)|, y={res['y']:g} -- 3-way vs the exact closed form")
    ax.legend()


def plot_freq_error(ax, res):
    w = res["w"]
    ax.loglog(w, np.maximum(res["err_fpl"], 1e-16), lw=1.6, label="fastpiclef (thimble)")
    ax.loglog(w, np.maximum(res["err_fiona"], 1e-16), "--", lw=1.2, label="FIONA (NUFFT3)")
    ax.set(xlabel="frequency  w", ylabel="rel. error in |F| vs exact",
           title="Accuracy vs the external truth (both codes)")
    ax.legend()


# A small, well-separated, weak field -- the mesh's most favourable case (the
# 40-microlens field proper is far beyond it). (x, y, mass) and the 2D source.
FIELD_LENSES = [(0.6, 0.0, 0.4), (-0.5, 0.4, 0.3), (0.1, -0.6, 0.35)]
FIELD_SOURCE = (0.2, 0.1)


def field_comparison(lenses=None, source=None, w_fiona=None, w_mesh=None, mesh_delta=0.03):
    """Phase 3 / C2 -- the genuinely 2D field of point masses (non-axisymmetric,
    no closed form). FIONA (NUFFT3) handles all w robustly; our only field path
    is the mesh (the polar/thimble fast path needs axisymmetry, which the field
    lacks). Trust is calibrated by C1 (both codes machine-checked against the
    point-mass closed form); here FIONA is the reference and we measure where the
    mesh can follow it. The mesh diverges above low w -- the evidence for Path 2.
    """
    from . import fiona_driver

    if lenses is None:
        lenses = FIELD_LENSES
    if source is None:
        source = FIELD_SOURCE
    if w_fiona is None:
        w_fiona = np.geomspace(1.0, 30.0, 12)
    if w_mesh is None:
        w_mesh = np.array([1.0, 2.0, 4.0, 8.0])
    w_fiona = np.asarray(w_fiona, dtype=float)
    w_mesh = np.asarray(w_mesh, dtype=float)

    F_fiona, fiona_t = fiona_driver.fiona_field(lenses, source, w_fiona)
    F_ref_at_mesh, _ = fiona_driver.fiona_field(lenses, source, w_mesh)

    # Path 2 / M2: the stationary-phase (geometric-optics) solver -- fast
    # (O(#images)), exact as w -> infinity away from caustics. Its high-w
    # accuracy is anchored by the N=1 Nakamura-Deguchi match.
    pmf = fpl.PointMassField(lenses=lenses, beta=source)
    t0 = time.perf_counter()
    F_geo = np.array([complex(pmf.geometric(float(w)).amplification) for w in w_fiona])
    geo_t = time.perf_counter() - t0

    F_mesh = np.full(w_mesh.size, np.nan, dtype=complex)
    mesh_t = np.full(w_mesh.size, np.nan)
    for i, w in enumerate(w_mesh):
        try:
            t0 = time.perf_counter()
            r = fpl.PointMassField(lenses=lenses, beta=source).solve(
                omega=float(w), method="mesh", delta=mesh_delta, domain=(-6.0, 6.0))
            F_mesh[i] = complex(r.amplification)
            mesh_t[i] = time.perf_counter() - t0
        except Exception:  # noqa: BLE001  (mesh may fail outright at high w)
            pass
    err_mesh = np.abs(np.abs(F_mesh) - np.abs(F_ref_at_mesh)) / np.abs(F_ref_at_mesh)
    n_images = len(pmf.images())
    return {
        "lenses": lenses, "source": source, "n_images": n_images,
        "w_fiona": w_fiona, "F_fiona": F_fiona, "fiona_t": fiona_t,
        "F_geo": F_geo, "geo_t": geo_t,
        "w_mesh": w_mesh, "F_mesh": F_mesh, "F_ref_at_mesh": F_ref_at_mesh,
        "err_mesh": err_mesh, "mesh_t": mesh_t,
    }


def plot_field(ax, res):
    ax.plot(res["w_fiona"], np.abs(res["F_fiona"]), "-o", ms=3, color="C1",
            label="FIONA (NUFFT3), all w")
    ax.plot(res["w_fiona"], np.abs(res["F_geo"]), ":", lw=2, color="C0",
            label="fastpiclef geometric (M2)")
    ok = res["err_mesh"] < 0.1
    ax.plot(res["w_mesh"][ok], np.abs(res["F_mesh"][ok]), "s", ms=9, color="C2",
            label="fastpiclef mesh (converged)")
    ax.plot(res["w_mesh"][~ok], np.abs(res["F_mesh"][~ok]), "x", ms=11, mew=2.5, color="C3",
            label="fastpiclef mesh (diverged)")
    ax.set(xscale="log", xlabel="frequency  w", ylabel="|F|  amplification",
           title=f"Field of {res['n_images']} images ({len(res['lenses'])} masses) "
                 f"-- mesh & geometric vs FIONA")
    ax.legend()


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

    print("\n== Phase 2: (mu)-grid protocol vs P-LPI (kappa=-2, nu=10) ==")
    res = grid_comparison()
    for k, v in res["times"].items():
        print(f"  time  {k:<32} {v:.3f} s")
    print(f"  accuracy vs fastpiclef thimble:  P-LPI median {np.median(res['err_plpi']):.1e} "
          f"max {res['err_plpi'].max():.1e} | plind median {np.median(res['err_plind']):.1e}")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    plot_grid_overlay(a1, res)
    plot_grid_error(a2, res)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "grid.png"), dpi=120)
    pd.DataFrame(res["times"].items(), columns=["quantity", "seconds"]).to_csv(
        os.path.join(RESULTS, "grid_times.csv"), index=False)
    print(f"  wrote results/grid.png and results/grid_times.csv")

    print("\n== Phase 3 / C1: frequency sweep vs FIONA (point mass, ND truth) ==")
    fres = freq_comparison(y=1.0)
    for k, v in fres["times"].items():
        print(f"  time  {k:<28} {v:.2f} s")
    hi = fres["w"] >= 10.0
    print(f"  |F| rel err vs ND (external truth):")
    print(f"    fastpiclef  median {np.median(fres['err_fpl']):.1e}  (w>=10: {np.median(fres['err_fpl'][hi]):.1e})")
    print(f"    FIONA       median {np.median(fres['err_fiona']):.1e}  (w>=10: {np.median(fres['err_fiona'][hi]):.1e})")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    plot_freq_overlay(a1, fres)
    plot_freq_error(a2, fres)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "freq.png"), dpi=120)
    pd.DataFrame(fres["times"].items(), columns=["quantity", "seconds"]).to_csv(
        os.path.join(RESULTS, "freq_times.csv"), index=False)
    print(f"  wrote results/freq.png and results/freq_times.csv")


if __name__ == "__main__":
    main()
