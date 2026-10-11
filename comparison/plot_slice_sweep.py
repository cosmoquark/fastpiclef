"""The slice inner kernel G(x2) graded against the independent RC referee.

Reads `comparison/results/slice_sweep.csv` (written by
`cargo run --release --example slice_sweep`), evaluates
`comparison.referee.slice_referee` at the same slice positions, and renders a
3x2 figure (rows: |G|, relative error with the thimble's own estimate and the
swept-branch-point indicator, per-call time; columns: omega = 3 and 10) to
`notebooks/figures/slice_inner_validation.png`.

The story the figure must carry (2026-10-11): the thimble kernel is exact
where no log branch point is swept and misses a branch-cut (Hankel)
contribution where one is — its error estimate (dotted) stays tiny there by
construction, which is exactly why the indicator (gold) exists and why the
estimate now includes 100x it; the RC kernel is machine-exact everywhere at a
fraction of the cost.
"""

import csv
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from comparison.referee import TWO_MASS_BETA, TWO_MASS_LENSES, slice_referee  # noqa: E402

CSV = ROOT / "comparison" / "results" / "slice_sweep.csv"
OUT = ROOT / "notebooks" / "figures" / "slice_inner_validation.png"

# --- palette (validated dataviz default, light mode) ---
BLUE = "#2a78d6"    # thimble (A0)
ORANGE = "#eb6834"  # RC (A0', exact)
GOLD = "#b8860b"    # indicator / swept-region highlight
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
SURF = "#fcfcfb"
FLOOR = 2e-14

MASS_B = [0.5, -0.4]


def load(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def col(rows, key):
    out = []
    for r in rows:
        v = r[key]
        out.append(float(v) if v not in ("nan", "") else math.nan)
    return out


def median(v):
    v = sorted(x for x in v if not math.isnan(x))
    return v[len(v) // 2] if v else math.nan


def main():
    rows = load(CSV)
    omegas = sorted({float(r["omega"]) for r in rows})

    data = {}
    t0 = time.time()
    for w in omegas:
        sel = [r for r in rows if float(r["omega"]) == w]
        x2 = col(sel, "x2")
        ref = [slice_referee(x, w, TWO_MASS_LENSES, TWO_MASS_BETA, margin=4.0) for x in x2]
        data[w] = {
            "x2": x2,
            "ref": ref,
            "mag_ref": [abs(g) for g in ref],
            "mag_th": [abs(complex(re, im)) for re, im in zip(col(sel, "re_th"), col(sel, "im_th"))],
            "mag_rc": [abs(complex(re, im)) for re, im in zip(col(sel, "re_rc"), col(sel, "im_rc"))],
            "err_th": [abs(complex(re, im) - g) / abs(g) for re, im, g in zip(col(sel, "re_th"), col(sel, "im_th"), ref)],
            "err_rc": [abs(complex(re, im) - g) / abs(g) for re, im, g in zip(col(sel, "re_rc"), col(sel, "im_rc"), ref)],
            "est_th": col(sel, "est_th"),
            "ind": col(sel, "ind"),
            "t_th": col(sel, "t_th_ms"),
            "t_rc": col(sel, "t_rc_ms"),
        }
    print(f"referee evaluated at {sum(len(v['x2']) for v in data.values())} slices "
          f"in {time.time() - t0:.1f}s")

    plt.rcParams.update({
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "figure.facecolor": SURF,
        "axes.facecolor": SURF,
    })
    fig, axes = plt.subplots(3, len(omegas), figsize=(12.6, 9.4), sharex=True,
                             height_ratios=[1.0, 1.35, 0.9])
    fig.subplots_adjust(hspace=0.13, wspace=0.12, top=0.9, right=0.985, left=0.075,
                        bottom=0.085)

    for j, w in enumerate(omegas):
        d = data[w]
        x2 = d["x2"]
        ax0, ax1, ax2 = axes[0][j], axes[1][j], axes[2][j]

        # ---- row 0: |G| ----
        ax0.plot(x2, d["mag_ref"], color=INK, lw=3.2, alpha=0.25,
                 solid_capstyle="round", label="RC referee (independent)")
        ax0.plot(x2, d["mag_th"], color=BLUE, lw=1.5, solid_capstyle="round",
                 label="slice thimble (A0)")
        ax0.plot(x2, d["mag_rc"], color=ORANGE, lw=1.2, ls="--",
                 label="RC kernel (A0$'$, exact)")
        ax0.set_ylabel(r"$|G(x_2)|$" if j == 0 else "", color=INK2)
        ax0.set_title(f"ω = {w:g}", fontsize=13, color=INK, loc="left", pad=8)
        ax0.grid(True, color=MUTED, alpha=0.14, lw=0.6)
        ax0.set_axisbelow(True)
        for s in ("top", "right"):
            ax0.spines[s].set_visible(False)
        if j == 0:
            ax0.legend(frameon=False, loc="upper right", fontsize=9)

        # shade the swept band (where the thimble error exceeds the floor)
        swept = [x for x, e in zip(x2, d["err_th"]) if not math.isnan(e) and e > 1e-8]
        if swept:
            ax0.axvspan(min(swept), max(swept), color=GOLD, alpha=0.10, lw=0)
            ax1.axvspan(min(swept), max(swept), color=GOLD, alpha=0.10, lw=0)

        # ---- row 1: relative error + estimate + indicator ----
        ax1.axhline(FLOOR, color=MUTED, lw=1.0, ls=(0, (4, 3)))
        ax1.semilogy(x2, d["err_th"], color=BLUE, lw=1.5, solid_capstyle="round",
                     label="thimble err vs referee" if j == 0 else None)
        ax1.semilogy(x2, d["est_th"], color=BLUE, lw=1.0, ls=":",
                     label="thimble error estimate (100× indicator + quadrature)" if j == 0 else None)
        ax1.semilogy(x2, [max(i, 1e-16) for i in d["ind"]], color=GOLD, lw=1.4,
                     label="branch-point indicator" if j == 0 else None)
        ax1.semilogy(x2, [max(e, 1e-16) for e in d["err_rc"]], color=ORANGE, lw=1.2,
                     ls="--", label="RC err vs referee" if j == 0 else None)
        ax1.set_ylabel("relative error / bound" if j == 0 else "", color=INK2)
        ax1.set_ylim(1e-16, 30)
        ax1.yaxis.set_major_locator(LogLocator(base=10, numticks=9))
        ax1.grid(True, which="major", color=MUTED, alpha=0.14, lw=0.6)
        ax1.set_axisbelow(True)
        for s in ("top", "right"):
            ax1.spines[s].set_visible(False)
        if j == 0:
            ax1.legend(frameon=False, loc="upper left", fontsize=8.2, ncol=1)

        # ---- row 2: timing ----
        ax2.semilogy(x2, [max(t, 1e-3) for t in d["t_th"]], color=BLUE, lw=1.4,
                     label="thimble" if j == 0 else None)
        ax2.semilogy(x2, [max(t, 1e-3) for t in d["t_rc"]], color=ORANGE, lw=1.4,
                     label="RC (exact)" if j == 0 else None)
        med_th, med_rc = median(d["t_th"]), median(d["t_rc"])
        ax2.axhline(med_th, color=BLUE, lw=0.8, alpha=0.5)
        ax2.axhline(med_rc, color=ORANGE, lw=0.8, alpha=0.5)
        ax2.set_ylabel("time / slice  (ms)" if j == 0 else "", color=INK2)
        ax2.set_xlabel(r"slice position  $x_2$")
        ax2.grid(True, which="major", color=MUTED, alpha=0.14, lw=0.6)
        ax2.set_axisbelow(True)
        for s in ("top", "right"):
            ax2.spines[s].set_visible(False)
        if j == 0:
            ax2.legend(frameon=False, loc="upper right", fontsize=9)
        ax2.text(0.02, 0.06,
                 f"median: thimble {med_th*1e3:.0f} µs vs RC {med_rc*1e3:.0f} µs "
                 f"({med_th/max(med_rc, 1e-9):.0f}×)",
                 transform=ax2.transAxes, fontsize=9, color=INK2, va="bottom")
        # mass pinches
        for b in MASS_B:
            for ax in (ax0, ax1, ax2):
                ax.axvline(b, color=GOLD, lw=1.0, alpha=0.45)

    axes[0][0].text(0.02, 0.06, r"$\times$ = mass pinch ($x_2 = b_j$)",
                    transform=axes[0][0].transAxes, fontsize=8.5, color=GOLD)

    fig.suptitle("The Cartesian-slice inner kernel: thimble (A0) vs the exact RC kernel (A0'), "
                 "graded against the independent referee",
                 fontsize=13.5, color=INK, x=0.075, ha="left", y=0.965)
    fig.text(0.075, 0.935,
             "The thimble is machine-exact where no log branch point is swept and misses the "
             "branch-cut (Hankel) term where one is; the indicator bounds that deficit. "
             "RC is exact everywhere, an order of magnitude faster.",
             fontsize=10, color=INK2)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {OUT}")
    for w in omegas:
        d = data[w]
        print(f"omega={w:g}: median thimble/RC time {median(d['t_th'])*1e3:.1f}/"
              f"{median(d['t_rc'])*1e3:.1f} us; max thimble err {max(e for e in d['err_th'] if not math.isnan(e)):.2e}; "
              f"max RC err {max(d['err_rc']):.2e}")


if __name__ == "__main__":
    main()
