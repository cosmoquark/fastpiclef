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
    its full-grid time is extrapolated.  P3 adds the *shared-contour* 1D batch
    (`RationalLens.grid`, method="rc") as fastpiclef's grid answer — the
    protocol-matching competitor to P-LPI's batched C++ run.
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

    # fastpiclef: the shared-contour 1D batch (rc_1d_batch) over the same grid.
    t0 = time.perf_counter()
    integ, _, _ = fpl.RationalLens(beta=0.0, kappa=kappa).grid(
        [float(m) for m in mu], omega, method="rc")
    F_grid_raw = np.array(integ, dtype=complex)
    fpl_grid_t = time.perf_counter() - t0
    err_fpl_grid = np.abs(F_grid_raw - F_fpl_raw) / np.abs(F_fpl_raw)
    fpl_grid_tN = _threaded_grid_time(kappa, omega, [float(m) for m in mu])

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
        "err_fpl_grid": err_fpl_grid,
        "times": {
            "fastpiclef (512, 1 thr)": fpl_t,
            "fastpiclef grid rc (512, 1 thr)": fpl_grid_t,
            f"fastpiclef grid rc (512, {os.cpu_count()} thr)": fpl_grid_tN,
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
    ax.semilogy(res["mu"], np.maximum(res["err_fpl_grid"], 1e-16), lw=1.2, ls=(0, (1, 2)),
                label="fastpiclef grid (rc batch) vs thimble")
    ax.set(xlabel="source position  mu", ylabel="rel. error vs fastpiclef thimble",
           title="Accuracy across the grid", ylim=(1e-9, 1e-1))
    ax.legend()


def _threaded_grid_time(kappa, omega, mu):
    """All-threads wall time of the shared-contour 1D grid (`RationalLens.grid`,
    rc), measured in a subprocess with the thread pins removed."""
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(RESULTS)))
    code = (
        "import time, fastpiclef as fpl\n"
        f"l = fpl.RationalLens(beta=0.0, kappa={kappa!r})\n"
        f"l.grid({mu!r}, {omega!r}, method='rc')\n"
        "t0 = time.perf_counter()\n"
        f"l.grid({mu!r}, {omega!r}, method='rc')\n"
        "print(time.perf_counter() - t0)\n"
    )
    env = {k: v for k, v in os.environ.items()
           if k not in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, cwd=root, check=True)
    return float(out.stdout.strip().splitlines()[-1])


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

    # fastpiclef: the exact RC slice solver (the field solver, here on the
    # single mass) — same protocol, machine precision at every frequency.
    t0 = time.perf_counter()
    F_slice = np.array(
        [complex(fpl.PointMassField([(0.0, 0.0, 1.0)], beta=(y, 0.0))
                 .solve(omega=float(w), method="slice").amplification)
         for w in w_grid]
    )
    slice_t = time.perf_counter() - t0

    # FIONA: adaptive NUFFT3 batched over the grid (its native fast mode).
    F_fiona, fiona_t = fiona_driver.fiona_point_mass_sweep(y, w_grid, threads=1)

    err_fpl = np.abs(np.abs(F_fpl) - np.abs(F_nd)) / np.abs(F_nd)
    err_slice = np.abs(np.abs(F_slice) - np.abs(F_nd)) / np.abs(F_nd)
    err_fiona = np.abs(np.abs(F_fiona) - np.abs(F_nd)) / np.abs(F_nd)
    times = {
        f"fastpiclef polar thimble ({w_grid.size} w, 1 thr)": fpl_t,
        f"fastpiclef slice ({w_grid.size} w, 1 thr)": slice_t,
        f"FIONA ({w_grid.size} w, 1 thr)": fiona_t,
    }
    if fiona_threads and fiona_threads > 1:
        _, fiona_tN = fiona_driver.fiona_point_mass_sweep(y, w_grid, threads=fiona_threads)
        times[f"FIONA ({w_grid.size} w, {fiona_threads} thr)"] = fiona_tN
    return {
        "y": y, "w": w_grid,
        "F_fpl": F_fpl, "F_slice": F_slice, "F_fiona": F_fiona, "F_nd": F_nd,
        "err_fpl": err_fpl, "err_slice": err_slice, "err_fiona": err_fiona,
        "times": times,
    }


def plot_freq_overlay(ax, res):
    w = res["w"]
    ax.plot(w, np.abs(res["F_nd"]), "k-", lw=2.4, alpha=0.35, label="Nakamura-Deguchi (exact)")
    ax.plot(w, np.abs(res["F_fpl"]), lw=1.5, label="fastpiclef (polar thimble)")
    ax.plot(w, np.abs(res["F_slice"]), lw=1.1, ls=(0, (1, 2)), label="fastpiclef (slice, exact)")
    ax.plot(w, np.abs(res["F_fiona"]), "--", lw=1.2, label="FIONA (NUFFT3)")
    ax.set(xscale="log", xlabel="frequency  w", ylabel="|F|  amplification",
           title=f"Point mass |F(w)|, y={res['y']:g} -- 4-way vs the exact closed form")
    ax.legend()


def plot_freq_error(ax, res):
    w = res["w"]
    ax.loglog(w, np.maximum(res["err_fpl"], 1e-16), lw=1.6, label="fastpiclef (polar thimble)")
    ax.loglog(w, np.maximum(res["err_slice"], 1e-16), ls=(0, (1, 2)), lw=1.2,
              label="fastpiclef (slice, exact)")
    ax.loglog(w, np.maximum(res["err_fiona"], 1e-16), "--", lw=1.2, label="FIONA (NUFFT3)")
    ax.set(xlabel="frequency  w", ylabel="rel. error in |F| vs exact",
           title="Accuracy vs the external truth (all codes)")
    ax.legend()


def fiona_grid_contest(ys=(0.2, 0.5, 1.0, 2.0), w_grid=None, fiona_threads=None):
    """P3.2 — FIONA's native (w, y) protocol vs our batched entry points.

    FIONA batches frequencies per source; ours batches frequencies per source
    (`PointMassField.sweep`) and sources per frequency (`PointMassField.map`).
    Everything is graded against the Nakamura-Deguchi closed form (external).
    """
    from . import fiona_driver
    from .referee import point_mass_exact

    if w_grid is None:
        w_grid = np.geomspace(0.5, 30.0, 12)
    w_grid = np.asarray(w_grid, dtype=float)
    ys = np.asarray(ys, dtype=float)

    # FIONA: one solver call per source (its native batched-frequency mode).
    t0 = time.perf_counter()
    F_fiona = np.array([
        fiona_driver.fiona_point_mass_sweep(float(y), w_grid, threads=1)[0] for y in ys
    ])
    fiona_t = time.perf_counter() - t0
    fiona_tN = None
    if fiona_threads and fiona_threads > 1:
        t0 = time.perf_counter()
        for y in ys:
            fiona_driver.fiona_point_mass_sweep(float(y), w_grid, threads=fiona_threads)
        fiona_tN = time.perf_counter() - t0

    # ours: batched frequency sweeps (one per source)
    t0 = time.perf_counter()
    F_sweep = np.array([
        np.array(fpl.PointMassField([(0.0, 0.0, 1.0)], beta=(float(y), 0.0))
                 .sweep(list(w_grid)).amplification) for y in ys
    ])
    sweep_t = time.perf_counter() - t0

    # ours: batched source maps (one per frequency)
    t0 = time.perf_counter()
    F_map = np.empty((ys.size, w_grid.size), dtype=complex)
    for i, w in enumerate(w_grid):
        m = fpl.PointMassField([(0.0, 0.0, 1.0)], beta=(0.0, 0.0)).map(
            [float(y) for y in ys], [0.0], float(w))
        F_map[:, i] = [complex(m.amplification[j][0]) for j in range(ys.size)]
    map_t = time.perf_counter() - t0

    F_nd = np.array([[point_mass_exact(float(w), float(y)) for w in w_grid] for y in ys])

    def err(F):
        return np.abs(np.abs(F) - np.abs(F_nd)) / np.abs(F_nd)

    times = {
        f"FIONA ({ys.size}x{w_grid.size}, 1 thr)": fiona_t,
        f"fastpiclef sweep ({ys.size}x{w_grid.size}, 1 thr)": sweep_t,
        f"fastpiclef map ({ys.size}x{w_grid.size}, 1 thr)": map_t,
    }
    if fiona_tN is not None:
        times[f"FIONA ({ys.size}x{w_grid.size}, {fiona_threads} thr)"] = fiona_tN
    return {
        "ys": ys, "w": w_grid,
        "F_fiona": F_fiona, "F_sweep": F_sweep, "F_map": F_map, "F_nd": F_nd,
        "err_fiona": err(F_fiona), "err_sweep": err(F_sweep), "err_map": err(F_map),
        "times": times,
    }


def plot_fiona_grid(ax, res):
    w = res["w"]
    y0 = res["ys"][0]
    ax.plot(w, np.abs(res["F_nd"][0]), "k-", lw=2.4, alpha=0.35, label="Nakamura-Deguchi (exact)")
    ax.plot(w, np.abs(res["F_fiona"][0]), "--", lw=1.2, label=f"FIONA (y={y0:g})")
    ax.plot(w, np.abs(res["F_sweep"][0]), lw=1.5, label=f"fastpiclef sweep (y={y0:g})")
    ax.plot(w, np.abs(res["F_map"][0]), ls=(0, (1, 2)), lw=1.2, label=f"fastpiclef map (y={y0:g})")
    ax.set(xscale="log", xlabel="frequency  w", ylabel="|F|  amplification",
           title="(w, y) grid protocol — first source row")
    ax.legend()


def plot_fiona_grid_error(ax, res):
    w = res["w"]
    ax.loglog(w, np.maximum(res["err_fiona"].max(axis=0), 1e-16), "--", lw=1.2, label="FIONA")
    ax.loglog(w, np.maximum(res["err_sweep"].max(axis=0), 1e-16), lw=1.5,
              label="fastpiclef sweep")
    ax.loglog(w, np.maximum(res["err_map"].max(axis=0), 1e-16), ls=(0, (1, 2)), lw=1.2,
              label="fastpiclef map")
    ax.set(xlabel="frequency  w", ylabel="max rel. error in |F| over y",
           title="(w, y) grid: accuracy vs the exact closed form")
    ax.legend()


# A small, well-separated, weak field -- the mesh's most favourable case (the
# 40-microlens field proper is far beyond it). (x, y, mass) and the 2D source.
FIELD_LENSES = [(0.6, 0.0, 0.4), (-0.5, 0.4, 0.3), (0.1, -0.6, 0.35)]
FIELD_SOURCE = (0.2, 0.1)


def _threaded_slice_time(lenses, source, omega):
    """All-threads wall time of one field solve, measured in a subprocess with
    the thread pins removed (rayon reads RAYON_NUM_THREADS at pool init, so
    the pinned harness process cannot measure this itself)."""
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(RESULTS)))
    code = (
        "import time, fastpiclef as fpl\n"
        f"f = fpl.PointMassField({lenses!r}, beta={source!r})\n"
        f"f.solve(omega={omega!r})\n"  # warm (thread-pool init)
        "t0 = time.perf_counter()\n"
        f"f.solve(omega={omega!r})\n"
        "print(time.perf_counter() - t0)\n"
    )
    env = {k: v for k, v in os.environ.items()
           if k not in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, cwd=root, check=True)
    return float(out.stdout.strip().splitlines()[-1])


def field_comparison(lenses=None, source=None, w_fiona=None, w_mesh=None, mesh_delta=0.03):
    """Phase 3 / C2 -- the genuinely 2D field of point masses (non-axisymmetric,
    no closed form). The *slice* solver (exact RC in both variables; validated
    against Nakamura-Deguchi on the single mass) is the reference; FIONA is
    graded against it, the geometric-optics solver tracks it at high w, and the
    mesh's divergence is kept as history. The mesh is not a field solver: it
    diverges on log potentials, which is why Path 2 exists.
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

    # fastpiclef slice: the exact reference (independent per-frequency solves;
    # warm-starting is not implemented — see the roadmap).
    slice_t = np.full(w_fiona.size, np.nan)
    F_slice = np.full(w_fiona.size, np.nan, dtype=complex)
    for i, w in enumerate(w_fiona):
        t0 = time.perf_counter()
        r = fpl.PointMassField(lenses=lenses, beta=source).solve(omega=float(w), method="slice")
        F_slice[i] = complex(r.amplification)
        slice_t[i] = time.perf_counter() - t0

    # Path 2 / M2: the stationary-phase (geometric-optics) solver -- fast
    # (O(#images)), exact as w -> infinity away from caustics.
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
    err_fiona = np.abs(np.abs(F_fiona) - np.abs(F_slice)) / np.abs(F_slice)
    err_geo = np.abs(np.abs(F_geo) - np.abs(F_slice)) / np.abs(F_slice)
    t_threaded = _threaded_slice_time(lenses, source, float(w_fiona[-1]))
    n_images = len(pmf.images())
    return {
        "lenses": lenses, "source": source, "n_images": n_images,
        "w_fiona": w_fiona, "F_fiona": F_fiona, "fiona_t": fiona_t,
        "F_slice": F_slice, "slice_t": slice_t, "slice_t_threaded": t_threaded,
        "err_fiona": err_fiona,
        "F_geo": F_geo, "geo_t": geo_t, "err_geo": err_geo,
        "w_mesh": w_mesh, "F_mesh": F_mesh, "F_ref_at_mesh": F_ref_at_mesh,
        "err_mesh": err_mesh, "mesh_t": mesh_t,
    }


def plot_field(ax, res):
    ax.plot(res["w_fiona"], np.abs(res["F_fiona"]), "-o", ms=3, color="C1",
            label="FIONA (NUFFT3), all w")
    ax.plot(res["w_fiona"], np.abs(res["F_slice"]), "-", lw=2.4, alpha=0.35, color="C0",
            label="fastpiclef slice (exact reference)")
    ax.plot(res["w_fiona"], np.abs(res["F_geo"]), ":", lw=2, color="C0",
            label="fastpiclef geometric (M2)")
    ok = res["err_mesh"] < 0.1
    ax.plot(res["w_mesh"][ok], np.abs(res["F_mesh"][ok]), "s", ms=9, color="C2",
            label="fastpiclef mesh (converged)")
    ax.plot(res["w_mesh"][~ok], np.abs(res["F_mesh"][~ok]), "x", ms=11, mew=2.5, color="C3",
            label="fastpiclef mesh (diverged)")
    ax.set(xscale="log", xlabel="frequency  w", ylabel="|F|  amplification",
           title=f"Field of {res['n_images']} images ({len(res['lenses'])} masses) "
                 f"-- exact slice reference, FIONA, geometric, mesh")
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
    print(f"  fastpiclef rc batch vs thimble:  median {np.median(res['err_fpl_grid']):.1e} "
          f"max {res['err_fpl_grid'].max():.1e}")
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

    print("\n== P3 / grid protocol: FIONA's (w,y) grid vs our sweep/map ==")
    gres = fiona_grid_contest(fiona_threads=os.cpu_count())
    for k, v in gres["times"].items():
        print(f"  time  {k:<40} {v:.2f} s")
    print(f"  max rel |F| err vs ND: FIONA {gres['err_fiona'].max():.1e} | "
          f"sweep {gres['err_sweep'].max():.1e} | map {gres['err_map'].max():.1e}")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    plot_fiona_grid(a1, gres)
    plot_fiona_grid_error(a2, gres)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fiona_grid.png"), dpi=120)
    pd.DataFrame(gres["times"].items(), columns=["quantity", "seconds"]).to_csv(
        os.path.join(RESULTS, "fiona_grid_times.csv"), index=False)
    rows = []
    for i, y in enumerate(gres["ys"]):
        for j, w in enumerate(gres["w"]):
            rows.append(dict(
                y=float(y), w=float(w),
                absF_fiona=float(abs(gres["F_fiona"][i, j])),
                absF_sweep=float(abs(gres["F_sweep"][i, j])),
                absF_map=float(abs(gres["F_map"][i, j])),
                absF_nd=float(abs(gres["F_nd"][i, j])),
                err_fiona=float(gres["err_fiona"][i, j]),
                err_sweep=float(gres["err_sweep"][i, j]),
                err_map=float(gres["err_map"][i, j]),
            ))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "fiona_grid.csv"), index=False)
    print("  wrote results/fiona_grid.png, fiona_grid_times.csv, fiona_grid.csv")


if __name__ == "__main__":
    main()
