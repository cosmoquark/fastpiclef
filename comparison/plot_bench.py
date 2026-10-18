"""Quiet-window benchmark figures (reads the result CSVs, writes PNGs).

Figures (all into `comparison/results/`):

* `bench_high_omega.png` --- the cost law (wall ~ omega^s), the accuracy
  evolution (RC vs ND, geometric sum), and the threading speedup;
* `bench_many_masses.png` --- wall and *load-independent* inner-solve count
  vs the number of stars;
* `bench_batch_map.png` --- batched-vs-scalar amortisation for the source
  maps (incl. the N = 100 4x4 case) and the frequency sweeps;
* `bench_fast_paths.png` --- ND / polar / exact-RC walls and errors, and the
  ND band (true error vs omega*y against the eps*e^(omega*y) envelope);
* `bench_starfield.png` --- the tolerance x N x omega convergence of the
  adaptive RC on star fields.

Deterministic plotting only; data loads are recorded inside the CSVs.

Run: `pixi run -e comparison plot-bench`
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def read(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        print(f"  (missing {name}; skipped)")
        return None
    df = pd.read_csv(path)
    loads = df["loadavg"].dropna() if "loadavg" in df else pd.Series(dtype=float)
    extra = f", load {loads.min():.2f}..{loads.max():.2f}" if len(loads) else ""
    print(f"  read {name}: {len(df)} rows{extra}")
    return df


def slope(x, y):
    x = np.log10(np.asarray(x, dtype=float))
    y = np.log10(np.asarray(y, dtype=float))
    return float(np.polyfit(x, y, 1)[0])


# ---------------------------------------------------------------- high omega


def plot_high_omega(df):
    pm1 = df[(df.case == "point_mass") & (df.threads == 1)].sort_values("omega")
    f3 = df[(df.case == "field3") & (df.threads == 1)].sort_values("omega")
    pmt = df[(df.case == "point_mass") & (df.threads > 1)].sort_values("omega")

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3))
    ax = axes[0]
    ax.plot(pm1.omega, pm1.wall_s, "o-", label="point mass, 1 thread")
    ax.plot(f3.omega, f3.wall_s, "s-", label="3-mass field, 1 thread")
    if len(pmt):
        ax.plot(pmt.omega, pmt.wall_s, "^-", label=f"point mass, {int(pmt.threads.iloc[0])} threads")
    s_all = slope(pm1.omega, pm1.wall_s)
    hi = pm1[pm1.omega >= 30]
    s_hi = slope(hi.omega, hi.wall_s) if len(hi) >= 2 else float("nan")
    ax.set(xscale="log", yscale="log", xlabel="$\\omega$", ylabel="wall [s]",
           title=f"cost law: point mass $\\omega^{{{s_hi:.2f}}}$ (30..1000), "
                 f"$\\omega^{{{s_all:.2f}}}$ (10..1000)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(pm1.omega, pm1.rel_err_nd.clip(lower=1e-16), "o-", label="RC vs ND (rel.)")
    ax.plot(pm1.omega, pm1.geometric_rel_err.clip(lower=1e-16), "s--",
            label="geometric sum vs ND")
    ax.set(xscale="log", yscale="log", xlabel="$\\omega$", ylabel="relative error",
           title="point mass: exactness vs the closed form")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    if len(pmt):
        m = (pm1.set_index("omega").wall_s / pmt.set_index("omega").wall_s).dropna()
        ax.plot(m.index, m.values, "o-")
        for w, v in m.items():
            ax.annotate(f"{v:.1f}x", (w, v), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=8)
        ax.axhline(pmt.threads.iloc[0] / 2.0, color="0.4", ls=":", lw=0.8,
                   label="6 physical cores")
        ax.set(xscale="log", xlabel="$\\omega$", ylabel="speedup (1 thread / N threads)",
               title="threading (rayon, bitwise-identical)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("high_omega.csv — single-thread walls, quiet window", y=1.02)
    fig.tight_layout()
    out = os.path.join(RESULTS, "bench_high_omega.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")

    print(f"  slope: point-mass 1-thread over 30..1000: {s_hi:.2f} (10..1000: {s_all:.2f})")
    if len(pmt):
        for w, v in m.items():
            print(f"  speedup at w={w:g}: {v:.2f}x")
    print(f"  accuracy at w={pm1.omega.max():g}: rel vs ND {pm1.rel_err_nd.iloc[-1]:.1e}, "
          f"geometric {pm1.geometric_rel_err.iloc[-1]:.1e}")
    return s_hi


# ------------------------------------------------------------- many masses


def plot_many_masses(df):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
    for ax, col, lab in ((axes[0], "wall_s", "wall [s]"),
                         (axes[1], "inner_solves", "inner solves (counts)")):
        for w, g in df.groupby("omega"):
            g = g.sort_values("n_lenses")
            ax.plot(g.n_lenses, g[col], "o-", label=f"$\\omega$ = {w:g}")
            if len(g) >= 2:
                print(f"  {col} slope at w={w:g}: {slope(g.n_lenses, g[col]):.2f} "
                      f"(N = {g.n_lenses.min():g}..{g.n_lenses.max():g})")
        ax.set(xscale="log", yscale="log", xlabel="number of stars $N$", ylabel=lab,
               title=lab)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    la = df["loadavg"]
    axes[0].set_title(f"wall [s] (recorded load {la.min():.2f}..{la.max():.2f})")
    fig.suptitle("many_masses.csv — single point solve in an N-star field", y=1.02)
    fig.tight_layout()
    out = os.path.join(RESULTS, "bench_many_masses.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    for _, r in df.sort_values(["omega", "n_lenses"]).iterrows():
        print(f"  N={int(r.n_lenses):5d} w={r.omega:5g}: wall {r.wall_s:9.3f} s, "
              f"{int(r.inner_solves):7d} inner solves, {int(r.threads)} threads, "
              f"load {r.loadavg:.2f}")


# ------------------------------------------------------------------ batch


def plot_batch(df):
    d1 = df[df.threads == 1]
    labels, speeds, notes = [], [], []
    for case, scalar_mode in (("2d-map", "scalar-sub8"), ("map-100", "scalar-anchor")):
        sub = d1[d1.case == case]
        for (n, grid), g in sub.groupby(["n_lenses", "grid"], sort=False):
            piv = g.set_index("mode")["wall_s"]
            if "batch" in piv and scalar_mode in piv:
                labels.append(f"N={int(n)}, {grid}")
                speeds.append(float(piv[scalar_mode] / piv["batch"]))
                notes.append(float(piv["batch"]))
    sw = d1[d1.case == "sweep"]
    for n, g in sw.groupby("n_lenses", sort=False):
        piv = g.set_index("mode")["wall_s"]
        if "batch" in piv and "scalar-sub6" in piv:
            labels.append(f"N={int(n)}, 200 ω")
            speeds.append(float(piv["scalar-sub6"] / piv["batch"]))
            notes.append(float(piv["batch"]))
    fig, ax = plt.subplots(figsize=(8.6, max(0.42 * len(labels) + 1.6, 3.5)))
    y = np.arange(len(labels))
    ax.barh(y, speeds, color="tab:blue", alpha=0.8)
    for yy, s, wall in zip(y, speeds, notes):
        ax.annotate(f"{s:.1f}x  (batch {wall:.3g} s)", (s, yy), textcoords="offset points",
                    xytext=(4, -3), fontsize=8, va="center")
    ax.set_yticks(y, labels, fontsize=8)
    ax.set(xlabel="amortisation: scalar estimate / batch wall (1 thread)",
           title="batch_bench.csv — shared-contour amortisation (serial run)")
    ax.set_xlim(0, max(speeds) * 1.25)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS, "bench_batch_map.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    for lab, s, wall in zip(labels, speeds, notes):
        print(f"  {lab:18s} amortised {s:7.1f}x  (batch {wall:.3g} s)")


# -------------------------------------------------------------- fast paths


def plot_fast_paths(fp, band):
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3))
    ax = axes[0]
    for path, style in (("nd", "o-"), ("polar", "s-"), ("rc-slice", "^-")):
        g = fp[fp.path == path].sort_values("omega")
        ax.plot(g.omega, g.wall_s, style, label=path)
    ax.set(xscale="log", yscale="log", xlabel="$\\omega$ (y = 1)",
           ylabel="wall [s]", title="isolated lens: the three paths (1 thread)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for path, style in (("nd", "o-"), ("polar", "s-"), ("rc-slice", "^-")):
        g = fp[fp.path == path].sort_values("omega")
        ax.plot(g.omega, g.rel_rc.clip(lower=1e-17), style, label=path)
    ax.set(xscale="log", yscale="log", xlabel="$\\omega$ (y = 1)",
           ylabel="|mod| error vs the tight RC", title="accuracy of the three paths")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(band.omega_y, band.true_err, "o-", label="ND true error (vs RC)")
    xs = np.linspace(band.omega_y.min(), band.omega_y.max(), 100)
    ax.plot(xs, 2e-17 * np.exp(xs), "k--", lw=0.9, label="$\\varepsilon\\,e^{\\omega y}$ envelope")
    ax.set(xscale="linear", yscale="log", xlabel="$\\omega\\, y$",
           ylabel="rel. error", title="the ND band: driver is $\\omega y$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle("fast_paths.csv + nd_band.csv", y=1.02)
    fig.tight_layout()
    out = os.path.join(RESULTS, "bench_fast_paths.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    for path in ("nd", "polar", "rc-slice"):
        g = fp[fp.path == path].sort_values("omega")
        unit = 1e6 if path == "nd" else 1e3
        tag = "us" if path == "nd" else "ms"
        print(f"  {path:9s} " + "  ".join(
            f"w={r.omega:g}: {unit * r.wall_s:8.2f} {tag}, err {r.rel_rc:.1e}"
            for _, r in g.iterrows()))
    print("  nd band (omega*y, true err): " + "  ".join(
        f"{r.omega_y:.0f}:{r.true_err:.1e}" for _, r in band.sort_values("omega_y").iterrows()))


# --------------------------------------------------------------- starfield


def plot_starfield(conv):
    star = conv[conv.case.astype(str).str.startswith("starfield")]
    if star.empty:
        print("  (no starfield rows)")
        return
    fam = star[star.case.str.match(r"starfield_n\d+$") & (star.code == "fastpiclef-rc")]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3))
    ax = axes[0]
    for case, g in fam.groupby("case"):
        g = g.copy()
        g["tol"] = g.knob_value.astype(float)
        g = g.sort_values("tol")
        ax.plot(g.tol, g.rel_err.clip(lower=1e-17), "o-", label=case.replace("starfield_", ""))
    lims = np.array([1e-15, 1e-3])
    ax.plot(lims, lims, "0.4", lw=0.8, ls=":", label="achieved = requested")
    ax.axhline(1e-13, color="0.4", lw=0.8, ls="--")
    ax.set(xscale="log", yscale="log", xlabel="requested tol",
           ylabel="rel. error vs tight truth",
           title="tolerance honored ($\\omega$ = 3)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for case, g in fam.groupby("case"):
        g = g.copy()
        g["tol"] = g.knob_value.astype(float)
        g = g.sort_values("tol")
        ax.plot(g.tol, g.wall_s, "o-", label=case.replace("starfield_", ""))
    ax.set(xscale="log", yscale="log", xlabel="requested tol", ylabel="wall [s]",
           title="cost of tightening the tolerance")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    wfam = star[(star.case.str.contains("_w") | (star.case == "starfield_n10"))
                & (star.code == "fastpiclef-rc")].copy()
    wfam["tol"] = wfam.knob_value.astype(float)
    for tol in (1e-8, 1e-14):
        pts = []
        g = wfam[wfam["tol"] == tol]
        for _, r in g.iterrows():
            om = 3.0 if r.case == "starfield_n10" else float(
                r.case.split("_w")[1].replace("p", "."))
            pts.append((om, r.rel_err, r.wall_s))
        pts.sort()
        ax.plot([p[0] for p in pts], [max(p[1], 1e-17) for p in pts], "o-",
                label=f"tol = {tol:g}")
    ax.set(xscale="log", yscale="log", xlabel="$\\omega$ (N = 10)",
           ylabel="rel. error", title="frequency independence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle("convergence_p3.csv — star-field tolerance study (1 thread)", y=1.02)
    fig.tight_layout()
    out = os.path.join(RESULTS, "bench_starfield.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    for _, r in star.sort_values(["case", "knob_value"]).iterrows():
        if r.code == "fastpiclef-rc":
            print(f"  {r.case:22s} tol {float(r.knob_value):8.0e}: err {r.rel_err:.2e}, "
                  f"wall {r.wall_s:8.3f} s, evals {int(r.evals)}")


def main():
    print("== reading datasets ==")
    ho = read("high_omega.csv")
    mm = read("many_masses.csv")
    bb = read("batch_bench.csv")
    fp = read("fast_paths.csv")
    band = read("nd_band.csv")
    conv = read("convergence_p3.csv")

    print("== figures ==")
    if ho is not None:
        plot_high_omega(ho)
    if mm is not None:
        plot_many_masses(mm)
    if bb is not None:
        plot_batch(bb)
    if fp is not None and band is not None:
        plot_fast_paths(fp, band)
    if conv is not None:
        plot_starfield(conv)


if __name__ == "__main__":
    main()
