"""P2b cross-check: the Rust `LensField` profiles against FIONA (third-party
arbitration at FIONA's own accuracy; |F| only, convention-free).

Writes `comparison/results/profiles_fiona.csv`.  Run in the comparison env
after `maturin develop --release` there.
"""

import csv
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass
os.environ.setdefault("FIONA_GL2D_DIR", "/tmp/fastpiclef_fiona_gl")
os.makedirs(os.environ["FIONA_GL2D_DIR"], exist_ok=True)

_REFS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "refs", "FIONA"))
if _REFS not in sys.path:
    sys.path.insert(0, _REFS)

import fiona  # noqa: E402
from fiona import lenses as fl  # noqa: E402

import fastpiclef as fpl  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results", "profiles_fiona.csv")


def main():
    rows = []
    w_list = [1.0, 3.0, 10.0]
    for name, lens, comp in [
        ("sis", fl.SIS(psi0=1.0), fpl.SISComp(0.0, 0.0, 1.0)),
        ("cis", fl.CIS(psi0=1.0, xc=0.2), fpl.CISComp(0.0, 0.0, 1.0, 0.2)),
        ("nfw", fl.NFW(psi0=1.0, xs=0.5), fpl.NFWComp(0.0, 0.0, 1.0, 0.5)),
    ]:
        fiona.set_num_threads(1)
        solver = fiona.FresnelNUFFT3(
            lens, auto_R_from_gl_nodes=True, parallel_frequencies=False,
            nufft_nthreads=1, verbose=False,
        )
        Ff = np.asarray(
            solver(np.array(w_list), np.array([0.5]), np.array([0.0]))
        ).reshape(-1)
        lf = fpl.LensField([comp], beta=(0.5, 0.0))
        for wi, Fi in zip(w_list, Ff):
            ours = lf.solve(omega=wi)
            rel = abs(abs(Fi) - abs(ours.amplification)) / abs(ours.amplification)
            print(f"  {name:4s} w={wi:4.1f}: FIONA |F| {abs(Fi):.8f}  ours "
                  f"{abs(ours.amplification):.8f}  rel {rel:.2e}")
            rows.append([name, f"{wi:.1f}", f"{rel:.17e}"])
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["profile", "omega", "rel_absF"])
        w.writerows(rows)
    print(f"-> {RESULTS}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time() - t0:.0f} s")
