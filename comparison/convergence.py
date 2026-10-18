"""P3 — convergence-order study across the codes on the P3 case list.

For each case and each code that applies, sweep the code's resolution knob and
record (relative error vs the case's referee, wall time, evaluations) to
`comparison/results/convergence_p3.csv`, with error-vs-cost figures in
`comparison/results/convergence_p3_{1d,2d}.png` and an empirical-slopes summary
in `comparison/results/convergence_p3_orders.md`.

Truths: closed forms (Airy, Nakamura-Deguchi) where they exist; the validated
production thimble for the 1D rational case; for the 2D cases without a closed
form the *tight* RC value at tol = 1e-14 with a margin-family Cauchy spread
recorded as its uncertainty; the NFW truth is the tight RC value, itself
validated against the Hankel referee to <= 8e-15 (P2 tests).

Run: `pixi run -e comparison python -m comparison.convergence`
"""

import os

os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import math

import json
import numpy as np
import pandas as pd

import fastpiclef as fpl

from . import drivers, plpi_driver
from .cases import airy_case, rational_case
from .metrics import timed
from .referee import slice_raw_from_nd

RESULTS = os.path.join(os.path.dirname(__file__), "results")

# The weak 3-mass field of C2 (same as run.py's FIELD_*).
FIELD_LENSES = [(0.6, 0.0, 0.4), (-0.5, 0.4, 0.3), (0.1, -0.6, 0.35)]
FIELD_SOURCE = (0.2, 0.1)


def _star_lenses(n: int, seed: int = 12345):
    """The deterministic star field of `examples/many_masses.rs` (replicated
    here, xorshift64, seed 12345): n equal masses 0.05, uniform in [-3, 3]^2.
    Cross-links the tolerance x N study to the many-mass cost dataset."""
    mask = (1 << 64) - 1
    s = seed | 1
    out = []
    for _ in range(n):
        s = (s ^ (s << 13)) & mask
        s ^= s >> 7
        s = (s ^ (s << 17)) & mask
        x = (s >> 11) / float(1 << 53)
        s = (s ^ (s << 13)) & mask
        s ^= s >> 7
        s = (s ^ (s << 17)) & mask
        y = (s >> 11) / float(1 << 53)
        out.append((x * 6.0 - 3.0, y * 6.0 - 3.0, 0.05))
    return out

RC_TOLS = (1e-4, 1e-6, 1e-8, 1e-11, 1e-14)
BATCH_TOLS = (1e-4, 1e-8, 1e-11, 1e-14)
MESH_DELTAS = (0.06, 0.03, 0.015, 0.0075, 0.00375)
PLIND_DELTAS = (1.2, 0.8, 0.5, 0.3, 0.2)
PLPI_NS = (2, 3, 4, 5, 6)
# NMu must stay >= 16: main.cpp prints progress via `index % (NMu / 16)`.
PLPI_NMUS = (16, 32, 64, 128, 256)
FIONA_GL = (32, 64, 128, 256)
FIONA_TOLS = (1e-3, 1e-6, 1e-9, 1e-12)


def _rel(v, t):
    return abs(v - t) / abs(t)


def _to_F(I, omega):
    """Raw integral -> amplification F (2D)."""
    return I * omega / (2.0 * math.pi * 1j)


def _relF(F_val, truth_I, omega):
    """|F| relative error.  FIONA's phase convention differs from the
    Nakamura-Deguchi raw integral by a constant ~pi (measured at y=1, w=10), so
    the C1 protocol's convention-free |F| is the right comparison for it."""
    t = abs(_to_F(truth_I, omega))
    return abs(abs(complex(F_val)) - t) / t


def _row(case, code, knob, value, rel_err, wall_s, evals, note=""):
    return {
        "case": case,
        "code": code,
        "knob": knob,
        "knob_value": str(value),
        "rel_err": float(rel_err),
        "wall_s": float(wall_s),
        "evals": evals,
        "note": note,
    }


def _case(name, dim, omega, builder, truth=None, truth_src="", source=None, **kw):
    return dict(
        name=name,
        dim=dim,
        omega=omega,
        builder=builder,
        truth=truth,
        truth_src=truth_src,
        truth_spread=0.0,
        source=source,
        **kw,
    )


def build_cases():
    cases = []
    # ---- 1D: Airy (exact) at the smooth and the caustic settings, rational ----
    c = airy_case(1.0, 1.0)
    cases.append(_case("airy_y1", 1, 1.0, c.builder, truth=c.truth,
                       truth_src="Airy closed form", phi=c.phi, dphi=c.dphi))
    c = airy_case(0.05, 1.0)
    cases.append(_case("airy_caustic", 1, 1.0, c.builder, truth=c.truth,
                       truth_src="Airy closed form", phi=c.phi, dphi=c.dphi))
    c = rational_case(0.3, -2.0, 10.0)
    cases.append(_case("rational_k2", 1, 10.0, c.builder, truth=c.truth,
                       truth_src="production thimble (validated)", phi=c.phi, dphi=c.dphi,
                       plpi=True))
    # ---- 2D ----
    cases.append(_case(
        "point_mass", 2, 10.0,
        lambda: fpl.PointMassField([(0.0, 0.0, 1.0)], beta=(1.0, 0.0)),
        truth=slice_raw_from_nd(10.0, 1.0), truth_src="Nakamura-Deguchi (mpmath dps 30)",
        source=(1.0, 0.0), fiona="point",
    ))
    cases.append(_case(
        "field3", 2, 10.0,
        lambda: fpl.PointMassField(FIELD_LENSES, beta=FIELD_SOURCE),
        truth_src="tight RC + Cauchy spread", source=FIELD_SOURCE, fiona="field",
        lenses=FIELD_LENSES,
    ))
    cases.append(_case(
        "type2_macro", 2, 10.0,
        lambda: fpl.LensField([fpl.PointComp(0.0, 0.0, 1.0)], beta=(0.4, 0.0),
                              kappa=0.5, gamma1=0.7),
        truth_src="tight RC + Cauchy spread (lambda0 = -0.2: FIONA cannot represent it)",
        source=(0.4, 0.0),
    ))
    cases.append(_case(
        "nfw", 2, 3.0,
        lambda: fpl.LensField([fpl.NFWComp(0.0, 0.0, 1.0, 1.0)], beta=(0.5, 0.0)),
        truth_src="tight RC (validated vs the Hankel referee <= 8e-15)", source=(0.5, 0.0),
    ))
    # ---- star fields: the tolerance x N (and omega) study ----
    for n in (1, 3, 10, 30):
        cases.append(_case(
            f"starfield_n{n}", 2, 3.0,
            lambda n=n: fpl.PointMassField(_star_lenses(n), beta=(0.5, 0.0)),
            truth_src="tight RC + Cauchy spread", source=(0.5, 0.0),
        ))
    cases.append(_case(
        "starfield_n100", 2, 3.0,
        lambda: fpl.PointMassField(_star_lenses(100), beta=(0.5, 0.0)),
        truth_src="tight RC + Cauchy spread", source=(0.5, 0.0),
        tols=(1e-8, 1e-14), batch=False,
    ))
    for w in (0.3, 10.0, 30.0):
        cases.append(_case(
            f"starfield_n10_w{str(w).replace('.', 'p')}", 2, w,
            lambda: fpl.PointMassField(_star_lenses(10), beta=(0.5, 0.0)),
            truth_src="tight RC + Cauchy spread", source=(0.5, 0.0),
            tols=(1e-8, 1e-14), batch=False,
        ))
    return cases


def resolve_truths(cases):
    """Fill in the RC truths for cases without a closed form, with a
    margin-family Cauchy spread as the truth uncertainty.

    Truths are checkpointed to `results/convergence_p3_truths.json` after each
    case (the N = 100 star field costs minutes per solve), so an interrupted
    run resumes without recomputing them.
    """
    ckpt_path = os.path.join(RESULTS, "convergence_p3_truths.json")
    ckpt = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as fh:
            ckpt = json.load(fh)
    for c in cases:
        if c["truth"] is not None:
            continue
        if c["name"] in ckpt:
            c["truth"] = complex(*ckpt[c["name"]]["truth"])
            c["truth_spread"] = ckpt[c["name"]]["spread"]
            continue
        b = c["builder"]
        v = complex(b().solve(omega=c["omega"], tol=1e-14).integral)
        v2 = complex(b().solve(omega=c["omega"], tol=1e-14, margin=2.5).integral)
        v3 = complex(b().solve(omega=c["omega"], tol=1e-14, margin_outer=3.0).integral)
        c["truth"] = v
        c["truth_spread"] = max(abs(v2 - v), abs(v3 - v)) / abs(v)
        ckpt[c["name"]] = {"truth": [v.real, v.imag], "spread": c["truth_spread"]}
        with open(ckpt_path, "w") as fh:
            json.dump(ckpt, fh, indent=1)
    return cases


# ------------------------------------------------------------------ studies


def study_ours_rc(case, tols=RC_TOLS):
    rows = []
    for tol in tols:
        b = case["builder"]
        r, s = timed(lambda: b().solve(omega=case["omega"], tol=tol), reps=2)
        rows.append(_row(case["name"], "fastpiclef-rc", "tol", f"{tol:g}",
                         _rel(complex(r.integral), case["truth"]), s, int(r.steps),
                         "single point"))
    return rows


def study_ours_rc_batch(case, tols=BATCH_TOLS):
    y0, y1 = case["source"]
    grid = [y0 - 0.1, y0, y0 + 0.1]
    rows = []
    for tol in tols:
        b = case["builder"]
        r, s = timed(lambda: b().map(grid, [y1], case["omega"], tol=tol), reps=2)
        v = complex(r.integral[1][0])
        rows.append(_row(case["name"], "fastpiclef-rc-batch", "tol", f"{tol:g}",
                         _rel(v, case["truth"]), s / len(grid), int(r.inner_solves),
                         "3-point batch, per-point time, centre entry"))
    return rows


def study_ours_geometric(case):
    b = case["builder"]
    try:
        r, s = timed(lambda: b().geometric(case["omega"]), reps=2)
        return [_row(case["name"], "fastpiclef-geometric", "none", "-",
                     _rel(complex(r.integral), case["truth"]), s, 0,
                     "stationary phase, O(#images)")]
    except Exception as e:  # noqa: BLE001  (the certified finder may refuse)
        return [_row(case["name"], "fastpiclef-geometric", "none", "-",
                     float("nan"), float("nan"), 0,
                     f"failed: {type(e).__name__}: {e}")]


def study_ours_thimble(case):
    b = case["builder"]
    r, s = timed(lambda: b().solve(omega=case["omega"], method="thimble"), reps=3)
    return [_row(case["name"], "fastpiclef-thimble", "none", "-",
                 _rel(complex(r.integral), case["truth"]), s, int(r.steps),
                 "machine precision (reference)")]


def study_ours_mesh(case, deltas=MESH_DELTAS):
    rows = []
    for d in deltas:
        try:
            r, s = timed(lambda: case["builder"]().solve(
                omega=case["omega"], method="mesh", delta=d, domain=(-8.0, 8.0)), reps=2)
            rows.append(_row(case["name"], "fastpiclef-mesh", "delta", f"{d:g}",
                             _rel(complex(r.integral), case["truth"]), s, int(r.steps)))
        except Exception as e:  # noqa: BLE001
            rows.append(_row(case["name"], "fastpiclef-mesh", "delta", f"{d:g}",
                             float("nan"), float("nan"), 0, f"failed: {type(e).__name__}"))
    return rows


def study_plind(case, deltas=PLIND_DELTAS):
    rows = []
    for d in deltas:
        try:
            v, s = timed(lambda: drivers.plind_run(case["phi"], case["dphi"],
                                                   case["omega"], N=2001, delta=d), reps=2)
            rows.append(_row(case["name"], "plind", "delta", f"{d:g}",
                             _rel(v, case["truth"]), s, 2001, "N=2001"))
        except Exception as e:  # noqa: BLE001
            rows.append(_row(case["name"], "plind", "delta", f"{d:g}",
                             float("nan"), float("nan"), 0, f"failed: {type(e).__name__}"))
    return rows


def study_plpi(case, Ns=PLPI_NS, NMus=PLPI_NMUS):
    """P-LPI's resolution knobs: `N` (per-thimble integration resolution) and
    `NMu` (thimbles reused by nearest-neighbour lookup).  The truth is the
    per-point production thimble over the same 512-point grid (independent)."""
    mu = plpi_driver.mu_grid()
    truth = np.array([
        complex(fpl.RationalLens(beta=float(m), kappa=-2.0)
                .solve(omega=10.0, method="thimble").integral)
        for m in mu
    ])
    rows = []
    for n in Ns:
        try:
            _, data, wall = plpi_driver.run_plpi_1d(threads=1, patches={"N": n})
        except Exception as e:  # noqa: BLE001
            rows.append(_row(case["name"], "P-LPI", "N", n, float("nan"),
                             float("nan"), 0, f"failed: {type(e).__name__}"))
            continue
        err = np.abs(data - truth) / np.abs(truth)
        rows.append(_row(case["name"], "P-LPI", "N", n, float(np.median(err)), wall,
                         plpi_driver.PLPI_NM, f"median over 512 grid points; max {err.max():.2e}"))
    for nmu in NMus:
        try:
            _, data, wall = plpi_driver.run_plpi_1d(threads=1, patches={"NMu": nmu})
        except Exception as e:  # noqa: BLE001
            rows.append(_row(case["name"], "P-LPI", "NMu", nmu, float("nan"),
                             float("nan"), 0, f"failed: {type(e).__name__}"))
            continue
        err = np.abs(data - truth) / np.abs(truth)
        rows.append(_row(case["name"], "P-LPI", "NMu", nmu, float(np.median(err)), wall,
                         nmu * plpi_driver.PLPI_NM, f"median; max {err.max():.2e}"))
    return rows


def study_fiona(case, gls=(64, 128, 256, 512), radii=(5.0, 10.0, 20.0, 40.0),
                tols=FIONA_TOLS):
    """FIONA's resolution knobs.  Its default adaptive mode fixes
    `n_gl(w) = 1000*clip(floor(w/10)+1, 1, 10)` and `R(w) = sqrt(n_gl/2w)`
    itself, so the exposed `gl_nodes_per_dim`/`nufft_tol` do not move its
    accuracy (measured flat, ~1e-3 here).  The fixed-box mode
    (`auto_R_from_gl_nodes=False`) is where the knobs bite: sweep the GL order
    at the adaptive R(w), then the box radius at gl = 256."""
    from . import fiona_driver

    w = [case["omega"]]
    if case["fiona"] == "point":
        def call(**kw):
            return fiona_driver.fiona_point_mass_sweep(1.0, w, threads=1, **kw)[0]
    else:
        def call(**kw):
            return fiona_driver.fiona_field(case["lenses"], case["source"], w,
                                            threads=1, **kw)[0]

    n_gl = 1000 * min(max(int(case["omega"] // 10) + 1, 1), 10)
    r_adaptive = math.sqrt(n_gl / (2.0 * case["omega"]))

    rows = []
    for gl in gls:
        F, s = timed(lambda: call(auto_R_from_gl_nodes=False, gl_nodes_per_dim=gl,
                                  min_physical_radius=r_adaptive, nufft_tol=1e-9), reps=2)
        rows.append(_row(case["name"], "FIONA", "gl_nodes_per_dim", gl,
                         _relF(F[0], case["truth"], case["omega"]), s, gl * gl,
                         f"fixed mode, R={r_adaptive:.2f} (the adaptive R(w))"))
    for r in radii:
        F, s = timed(lambda: call(auto_R_from_gl_nodes=False, gl_nodes_per_dim=256,
                                  min_physical_radius=r, nufft_tol=1e-9), reps=2)
        rows.append(_row(case["name"], "FIONA", "min_physical_radius", f"{r:g}",
                         _relF(F[0], case["truth"], case["omega"]), s, 256 * 256,
                         "fixed mode, gl=256"))
    for tol in tols:
        F, s = timed(lambda: call(nufft_tol=tol), reps=2)
        rows.append(_row(case["name"], "FIONA", "nufft_tol", f"{tol:g}",
                         _relF(F[0], case["truth"], case["omega"]), s, 2000 * 2000,
                         "adaptive mode (geometry-limited: flat)"))
    return rows


# ------------------------------------------------------------------ reporting


def slopes_md(df):
    """Empirical order per (case, code) series: log-log slope over the points
    whose error is above the truth floor (10x the recorded spread, or 1e-13)."""
    lines = [
        "# P3 convergence — empirical orders",
        "",
        "Series at the truth floor (machine-precision solvers, or a solver already "
        "at its own floor) show no usable points — that is the finding, not a gap. "
        "For the adaptive-GL codes the error falls super-algebraically, so the "
        "log-log slope understates the convergence; the figures carry the shape.",
        "",
    ]
    for (case, code), g in df.groupby(["case", "code"]):
        g = g[np.isfinite(g["rel_err"]) & (g["rel_err"] > 0)].sort_values("wall_s")
        floor = 1e-13
        g = g[g["rel_err"] > 10.0 * floor]
        if len(g) >= 3:
            x = np.log10(g["wall_s"].to_numpy())
            y = np.log10(g["rel_err"].to_numpy())
            slope = float(np.polyfit(x, y, 1)[0])
            lines.append(
                f"- {case:14s} {code:22s}: slope {slope:+.2f} over {len(g)} points "
                f"(err {g['rel_err'].min():.1e}..{g['rel_err'].max():.1e}, "
                f"time {g['wall_s'].min():.2e}..{g['wall_s'].max():.2e} s)"
            )
        else:
            lines.append(
                f"- {case:14s} {code:22s}: {len(g)} usable points; "
                f"best err {g['rel_err'].min() if len(g) else float('nan'):.1e}"
            )
    lines.append("")
    lines.append("## Rate vs evaluations (adaptive codes: decades of error per evaluation)")
    for (case, code), g in df.groupby(["case", "code"]):
        g = g[np.isfinite(g["rel_err"]) & (g["rel_err"] > 0) & (g["evals"] > 0)]
        g = g.sort_values("evals")
        if len(g) >= 3 and g["rel_err"].max() / max(g["rel_err"].min(), 1e-300) > 100:
            x = g["evals"].to_numpy(dtype=float)
            if x.max() - x.min() <= 0:
                continue  # the counter does not vary with the knob (e.g. mesh)
            y = np.log10(g["rel_err"].to_numpy())
            slope = float(np.polyfit(x, y, 1)[0])
            lines.append(
                f"- {case:14s} {code:20s}: {slope:+.3e} decades per evaluation "
                f"({len(g)} points, evals {int(x.min())}..{int(x.max())})"
            )
    return "\n".join(lines) + "\n"


def make_figures(df, cases):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [
        ("1d", [c for c in cases if c["dim"] == 1]),
        ("2d", [c for c in cases
                if c["dim"] == 2 and not c["name"].startswith("starfield")]),
    ]
    for tag, cs in groups:
        fig, axes = plt.subplots(1, len(cs), figsize=(4.6 * len(cs), 4.3), squeeze=False)
        for ax, c in zip(axes[0], cs):
            sub = df[df["case"] == c["name"]]
            for (code, knob), g in sub.groupby(["code", "knob"]):
                g = g[np.isfinite(g["rel_err"])].sort_values("wall_s")
                if g.empty:
                    continue
                ax.plot(g["wall_s"], np.maximum(g["rel_err"], 1e-16), "o-", ms=4,
                        label=f"{code} ({knob})")
            spread = max(c.get("truth_spread", 0.0), 1e-14)
            ax.axhline(spread, color="0.4", lw=0.8, ls=":")
            ax.set(xscale="log", yscale="log", xlabel="wall time [s]",
                   ylabel="rel. error vs truth",
                   title=f"{c['name']} (w={c['omega']:g})", ylim=(1e-16, 3e-1))
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        out = os.path.join(RESULTS, f"convergence_p3_{tag}.png")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  wrote {out}")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    print("== resolving truths ==")
    cases = resolve_truths(build_cases())
    for c in cases:
        print(f"  {c['name']:14s} truth |I|={abs(c['truth']):.6g}  "
              f"({c['truth_src']}; spread {c['truth_spread']:.1e})")

    rows = []
    out_csv = os.path.join(RESULTS, "convergence_p3.csv")
    for c in cases:
        print(f"== {c['name']} ==")
        try:
            if c["dim"] == 1:
                rows += study_ours_mesh(c)
                rows += study_ours_thimble(c)
                rows += study_plind(c)
                if c.get("plpi"):
                    rows += study_plpi(c)
            else:
                rows += study_ours_rc(c, c.get("tols", RC_TOLS))
                if c.get("batch", True):
                    rows += study_ours_rc_batch(c, c.get("tols_batch", BATCH_TOLS))
                rows += study_ours_geometric(c)
                if c.get("fiona"):
                    rows += study_fiona(c)
        except Exception as e:  # noqa: BLE001
            print(f"  !! {c['name']} failed: {type(e).__name__}: {e}")
            rows.append(_row(c["name"], "study-error", "-", "-", float("nan"),
                             float("nan"), 0, f"{type(e).__name__}: {e}"))
        print(f"  rows: {len(rows)}")
        # incremental snapshot: an interrupted run keeps the finished cases
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(df)} rows)")
    make_figures(df, cases)
    with open(os.path.join(RESULTS, "convergence_p3_orders.md"), "w") as fh:
        fh.write(slopes_md(df))
    print("wrote results/convergence_p3_orders.md")


if __name__ == "__main__":
    main()
