"""Driver for the FIONA reference code (Python, refs/FIONA), the main competitor.

FIONA computes F(w, y) = (w / 2 pi i) int d^2x exp{i w [ |x-y|^2/2 - psi(x) ]}
via Gauss-Legendre quadrature + FINUFFT (the fast production path; its compiled
Hankel extension `_pynufht` is not shipped in refs/). Its native protocol is a
*frequency sweep* at fixed source position -- the opposite axis to P-LPI's
source grid, and the setting where an FFT method is judged fairly.

The shared lens is the isolated point mass psi = ln r (FIONA `PointLens(xc=0)`),
whose amplification has the Nakamura-Deguchi closed form -- an external truth for
both codes. The returned F uses FIONA's own convention, identical to fastpiclef's
`.amplification` and to `referee.point_mass_exact`; we compare |F| (physical,
convention-free). refs/ is read-only, so the multi-mass field lens (C2) is a
`fiona.Lens` subclass defined *here*, not a change to refs/.

Two environment gotchas handled at import (neither needs a Python downgrade):
  * Python 3.14 defaults multiprocessing to a non-fork start method, which breaks
    FIONA's parallel GL-node builder -> force `fork`.
  * FIONA caches GL nodes in files differing only in case (`.W.npy` vs `.w.npy`);
    under Dropbox's case-conflict handling these collide, so the cache must live
    outside any synced folder -> default to /tmp (override with FIONA_GL2D_DIR).
"""

import multiprocessing as mp
import os
import sys
import time

import numpy as np

# --- multiprocessing start method (Python 3.14 default breaks FIONA's GL pool).
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

# --- GL-node cache outside Dropbox (case-insensitive-conflict safe).
os.environ.setdefault("FIONA_GL2D_DIR", "/tmp/fastpiclef_fiona_gl")
os.makedirs(os.environ["FIONA_GL2D_DIR"], exist_ok=True)

_REFS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "refs", "FIONA"))
if _REFS not in sys.path:
    sys.path.insert(0, _REFS)

import fiona  # noqa: E402


class PointMassFieldLens(fiona.Lens):
    """A field of unit point masses, psi(x) = sum_j (m_j/2) ln((x1-a_j)^2+(x2-b_j)^2),
    the FIONA-side counterpart of fastpiclef `PointMassField` (C2). `lenses` is a
    list of (x, y, mass); a small `soft` core keeps GL nodes off the log poles."""

    def __init__(self, lenses, soft=0.0):
        self.lenses = [(float(a), float(b), float(m)) for a, b, m in lenses]
        self.soft2 = float(soft) ** 2

    def psi_xy(self, x1, x2):
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        acc = np.zeros(np.broadcast(x1, x2).shape)
        for a, b, m in self.lenses:
            acc = acc + 0.5 * m * np.log((x1 - a) ** 2 + (x2 - b) ** 2 + self.soft2)
        return acc


def point_mass_solver(threads=1, **kwargs):
    """FIONA's fast 2D solver on the exact point mass (PointLens, xc=0), in its
    production adaptive mode (n_gl grows with w to resolve the oscillation)."""
    fiona.set_num_threads(threads)
    lens = fiona.PointLens(psi0=1.0, xc=0.0)
    opts = dict(
        auto_R_from_gl_nodes=True,
        parallel_frequencies=(threads > 1),
        nufft_nthreads=threads,
        verbose=False,
    )
    opts.update(kwargs)
    return fiona.FresnelNUFFT3(lens, **opts)


def fiona_point_mass_sweep(y, w_grid, threads=1, **kwargs):
    """Sweep FIONA F(w, y) over a frequency grid at fixed source position y.
    Returns (F_complex_array, wall_seconds). Uses one solver call over the whole
    grid (FIONA batches frequencies).  `kwargs` go to FresnelNUFFT3 (e.g.
    `gl_nodes_per_dim`, `nufft_tol` — the P3 convergence knobs)."""
    solver = point_mass_solver(threads=threads, **kwargs)
    w = np.asarray(w_grid, dtype=float)
    y1 = np.array([float(y)])
    y2 = np.array([0.0])
    t0 = time.perf_counter()
    F = np.asarray(solver(w, y1, y2)).reshape(-1)
    wall = time.perf_counter() - t0
    return F, wall


def fiona_field(lenses, source, w_grid, threads=1, soft=0.0, **kwargs):
    """FIONA F(w, y) over a frequency grid for a *field* of point masses (C2).
    `source` is the 2D source position (y1, y2), matching fastpiclef's `beta`.
    `kwargs` go to FresnelNUFFT3 (P3 convergence knobs)."""
    fiona.set_num_threads(threads)
    y1, y2 = float(source[0]), float(source[1])
    lens = PointMassFieldLens(lenses, soft=soft)
    opts = dict(
        auto_R_from_gl_nodes=True,
        parallel_frequencies=(threads > 1),
        nufft_nthreads=threads,
        verbose=False,
    )
    opts.update(kwargs)
    solver = fiona.FresnelNUFFT3(lens, **opts)
    w = np.asarray(w_grid, dtype=float)
    t0 = time.perf_counter()
    F = np.asarray(solver(w, np.array([y1]), np.array([y2]))).reshape(-1)
    wall = time.perf_counter() - t0
    return F, wall


if __name__ == "__main__":
    from referee import point_mass_exact

    w = np.array([1.0, 10.0, 100.0])
    F, wall = fiona_point_mass_sweep(1.0, w)
    print(f"FIONA point-mass sweep {w.size} freqs in {wall:.2f}s")
    for wi, Fi in zip(w, F):
        Fnd = point_mass_exact(wi, 1.0)
        print(f"  w={wi:6.1f}: |F|={abs(Fi):.6f}  rel={abs(abs(Fi) - abs(Fnd)) / abs(Fnd):.2e}")
