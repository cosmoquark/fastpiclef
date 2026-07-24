"""Visualize the M3A/A0 inner-slice kernel: the thimble G(x2) vs the 1D mesh
oracle across the slice position x2, and the relative error that localizes the
swept-branch-point gap Option 1 (Hankel terms) will close.

Reads comparison/results/slice_sweep.csv (written by `cargo run --example
slice_sweep`), writes notebooks/figures/slice_inner_validation.png.
"""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "comparison" / "results" / "slice_sweep.csv"
OUT = ROOT / "notebooks" / "figures" / "slice_inner_validation.png"

# --- palette (validated dataviz default, light mode) ---
BLUE = "#2a78d6"   # thimble (the new method)
ORANGE = "#eb6834"  # mesh oracle
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
SHADE = "#eda100"   # swept-region highlight
SURF = "#fcfcfb"
FLOOR = 2e-14       # ~ machine precision

# masses of the field used in slice_sweep.rs: (a, b, m); the slice sweeps x2,
# so branch points pinch the real axis as x2 -> b_j.
MASS_B = [0.5, -0.4]
SWEPT_THRESH = 1e-9  # rel. error above the machine floor => a branch point is swept


def swept_spans(x2, rel, ok):
    """Contiguous x2 intervals where the thimble deviates from the oracle
    (rel error above the floor, or a stall) — the region Option 1 must fix."""
    dev = [(not o) or (not math.isnan(r) and r > SWEPT_THRESH)
           for r, o in zip(rel, ok)]
    spans, start = [], None
    for i, d in enumerate(dev):
        if d and start is None:
            start = x2[i]
        elif not d and start is not None:
            spans.append((start, x2[i]))
            start = None
    if start is not None:
        spans.append((start, x2[-1]))
    return spans


def load(path):
    x2, mag_th, mag_me, rel, ok, t_th, t_me = [], [], [], [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            x2.append(float(row["x2"]))
            good = row["ok"] == "1"
            ok.append(good)
            me = complex(float(row["re_mesh"]), float(row["im_mesh"]))
            mag_me.append(abs(me))
            if good:
                th = complex(float(row["re_th"]), float(row["im_th"]))
                mag_th.append(abs(th))
                rel.append(float(row["relerr"]))
            else:
                mag_th.append(math.nan)
                rel.append(math.nan)
            t_th.append(float(row["t_th_ms"]))
            t_me.append(float(row["t_mesh_ms"]))
    return x2, mag_th, mag_me, rel, ok, t_th, t_me


def median(v):
    v = sorted(x for x in v if not math.isnan(x))
    return v[len(v) // 2] if v else math.nan


def main():
    x2, mag_th, mag_me, rel, ok, t_th, t_me = load(CSV)

    med_th, med_me = median(t_th), median(t_me)
    speedup = med_me / med_th if med_th else float("nan")

    plt.rcParams.update({
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "figure.facecolor": SURF,
        "axes.facecolor": SURF,
    })

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(9, 6.6), height_ratios=[1.0, 1.15], sharex=True
    )
    fig.subplots_adjust(hspace=0.12, top=0.9, right=0.97, left=0.1, bottom=0.1)

    spans = swept_spans(x2, rel, ok)

    # ---- panel 0: |G(x2)|, thimble vs oracle ----
    for lo, hi in spans:
        ax0.axvspan(lo, hi, color=SHADE, alpha=0.11, lw=0)
    ax0.plot(x2, mag_me, color=ORANGE, lw=3.2, alpha=0.35, solid_capstyle="round",
             label="1D mesh (oracle)")
    ax0.plot(x2, mag_th, color=BLUE, lw=1.6, solid_capstyle="round",
             label="slice thimble")
    ax0.set_ylabel(r"$|G(x_2)|$   (inner integral)", color=INK2)
    ax0.legend(frameon=False, loc="upper right", fontsize=10)
    ax0.grid(True, color=MUTED, alpha=0.14, lw=0.6)
    ax0.set_axisbelow(True)
    for s in ("top", "right"):
        ax0.spines[s].set_visible(False)
    ax0.set_title(
        "Cartesian-slice inner kernel (M3A / A0): the thimble reproduces the "
        "true inner integral,\nexactly except where it sweeps a log branch point",
        fontsize=12.5, color=INK, loc="left", pad=12,
    )

    # ---- panel 1: relative error (log) ----
    for j, (lo, hi) in enumerate(spans):
        ax1.axvspan(lo, hi, color=SHADE, alpha=0.11, lw=0,
                    label="swept branch point (Option 1 target)" if j == 0 else None)
    ax1.axhline(FLOOR, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    ax1.text(x2[2], FLOOR * 1.5, "machine precision", color=MUTED, fontsize=9,
             va="bottom")
    ax1.semilogy(x2, rel, color=BLUE, lw=1.6, solid_capstyle="round",
                 label="slice thimble vs oracle")
    ax1.legend(frameon=False, loc="lower right", fontsize=9.5)
    ax1.set_ylabel(r"rel. error  vs oracle", color=INK2)
    ax1.set_xlabel(r"slice position  $x_2$", color=INK2)
    ax1.set_ylim(FLOOR * 0.4, 5.0)
    ax1.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax1.grid(True, which="major", color=MUTED, alpha=0.14, lw=0.6)
    ax1.set_axisbelow(True)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # mass markers (thin verticals; a small tick label near the x-axis)
    for b in MASS_B:
        ax1.axvline(b, color=SHADE, lw=1.0, alpha=0.55)
        ax0.axvline(b, color=SHADE, lw=1.0, alpha=0.55)
        ax0.text(b, ax0.get_ylim()[0] + 0.05, r"$\times$", color="#b8860b",
                 fontsize=12, va="bottom", ha="center")
    ax0.text(-2.85, ax0.get_ylim()[1] * 0.86,
             r"$\times$ = mass ($x_2 = b_j$)", color="#a35a00", fontsize=9,
             va="center", ha="left")

    # annotate the two regimes
    ax1.annotate(
        "exact where no branch point is swept\n(machine precision, ~$10^{-14}$)",
        xy=(2.6, 2e-14), xytext=(1.4, 5e-11),
        color=INK2, fontsize=9.5,
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0),
    )
    ax1.annotate(
        "thimble misses a branch-cut (Hankel) term\n"
        "(oracle also unconverged near the pinches)",
        xy=(-0.6, 4e-1), xytext=(-1.65, 3.0),
        color="#a35a00", fontsize=9, ha="left", va="top",
        arrowprops=dict(arrowstyle="->", color=SHADE, lw=1.1),
    )
    # the smooth exponential Stokes turn-on on the negative side
    ax1.annotate(
        "smooth (Stokes)\nturn-on",
        xy=(-1.78, 4e-11), xytext=(-2.95, 2e-7),
        color=INK2, fontsize=8.5, ha="left",
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9),
    )

    # speed callout
    fig.text(
        0.1, 0.005,
        f"single-thread inner solve:  thimble {med_th*1e3:.0f} µs   "
        f"vs   mesh oracle {med_me*1e3:.0f} µs   "
        f"(≈{speedup:.0f}× faster), ω = 3",
        fontsize=9.5, color=INK2,
    )

    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {OUT}")
    print(f"median times: thimble {med_th*1e3:.1f} us, mesh {med_me*1e3:.1f} us, "
          f"speedup {speedup:.1f}x")
    def in_span(x):
        return any(lo <= x <= hi for lo, hi in spans)

    unswept = [r for x, r, o in zip(x2, rel, ok) if o and not in_span(x)]
    print(f"slices: {sum(ok)}/{len(ok)} thimble-ok; "
          f"swept spans: {[(round(a, 2), round(b, 2)) for a, b in spans]}; "
          f"un-swept max rel err {max(unswept):.1e}")


if __name__ == "__main__":
    main()
