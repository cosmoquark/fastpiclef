"""P3.4 — glworia third-party comparison: attempt and outcome (2026-10-12).

glworia's own shipped reference test is NOT reproduced under the only JAX
available for our Python 3.14 env (jax 0.10.2, conda-forge): for its NFW
`kappa = 3`, `y = (0.1, 0)` case the live `compute_F` differs from its own
`test/data/F_interp.npy` (median 0.79 / max 355 relative), and the live F has
no smooth y-dependence — silently wrong output, not a convention shift.  The
package metadata (Python < 3.12, jax 0.3/0.4 era) explains the drift; a
dedicated pinned env (python <= 3.11 + jax ~ 0.4.27) would be needed.

This script records the evidence in `results/glworia_attempt.csv`; the driver
(`glworia_driver.py`, including the `host_callback` shim) stays in place for
that future env.  A likely better third-party comparator is `wolensing`
(numpy/scipy + lenstronomy, light jax use, shipped SIS references) — see the
P3 journal entry.

Run: `pixi run -e comparison python -m comparison.glworia_compare`
"""

import os

os.environ.setdefault("RAYON_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from . import glworia_driver as gd

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def main():
    rows = []
    w_ref, F_ref = gd.glworia_reference_data()
    print(f"glworia reference curve: {F_ref.size} points, w in [{w_ref[0]:g}, {w_ref[-1]:g}]")

    # Their own test settings (conftest.py), on their own dense grid.
    ws = np.linspace(1e-2, 1e3, 10**5)
    F = gd.glworia_F(
        "NFW", ws, (0.1, 0.0), [3.0], N=200, T0_max=1000.0,
        param_arr=np.linspace(0.1, 10.0, 100000),
    )
    rel = np.abs(F - F_ref) / np.abs(F_ref)
    print(f"live vs its own reference: median rel {np.median(rel):.2e}, max {rel.max():.2e}")
    rows.append(["live_vs_own_reference", "NFW kappa=3 y=0.1, N=200, T0_max=1000",
                 float(np.median(rel)), float(rel.max())])

    # Smoothness in y (a working wave-optics code varies smoothly; this does not).
    ws_small = np.array([1.0, 3.0, 10.0, 30.0])
    vals = {}
    for y in (0.01, 0.03, 0.10, 0.30):
        Fy = gd.glworia_F(
            "NFW", ws_small, (y, 0.0), [3.0], N=200, T0_max=1000.0,
            param_arr=np.linspace(0.1, 10.0, 100000),
        )
        vals[y] = np.abs(Fy)
        print(f"  live y={y:5.3f}: |F| = {np.array2string(np.abs(Fy), precision=3)}")
    spread = max(
        np.abs(vals[y1] - vals[y2]).max() / max(vals[y1].max(), 1e-30)
        for y1 in vals for y2 in vals if y1 < y2
    )
    rows.append(["y_smoothness", "max relative jump between adjacent y values",
                 float(spread), float("nan")])

    df = pd.DataFrame(rows, columns=["check", "detail", "value", "extra"])
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "glworia_attempt.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
