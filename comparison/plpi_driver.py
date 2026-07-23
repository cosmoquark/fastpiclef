"""Driver for the P-LPI reference code (C++, refs/P-LPI/code_1D).

code_1D computes the raw integral int exp(i*nu*phi) dx for
phi = (x - mu)^2/2 + 2/(1 + x^2)  [= our RationalLens(kappa=-2)]  at nu=10,
over a grid of NM=512 source positions mu in [-4, 4], written to result.bin.
refs/ is read-only, so we build a scratch copy with the env's C++ compiler.
"""

import os
import shutil
import subprocess
import tempfile
import time

import numpy as np

# Grid hardcoded in main.cpp.
PLPI_MU_MIN, PLPI_MU_MAX, PLPI_NM = -4.0, 4.0, 2 ** 9  # 512
PLPI_NU = 10.0
PLPI_KAPPA = -2.0  # phi = (x-mu)^2/2 + 2/(1+x^2) = RationalLens(beta=mu, kappa=-2)

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "refs", "P-LPI", "code_1D"))
_FLAGS = ["-fopenmp", "-std=c++11", "-O3", "-march=native", "-ffast-math"]


def mu_grid():
    return np.linspace(PLPI_MU_MIN, PLPI_MU_MAX, PLPI_NM)


def run_plpi_1d(threads=1, keep=False):
    """Build and run P-LPI code_1D; return (mu_grid, raw_integrals, wall_s).

    The reported wall time is the batched 512-point run only (not compilation).
    `threads` pins OMP_NUM_THREADS; `keep` leaves the build dir for inspection.
    """
    build = tempfile.mkdtemp(prefix="plpi1d_")
    for f in os.listdir(_SRC):
        if f.endswith((".cpp", ".h")):
            shutil.copy(os.path.join(_SRC, f), build)

    cxx = os.environ.get("CXX", "g++")
    comp = subprocess.run(
        [cxx, *_FLAGS, "-o", "PL1D", "main.cpp"],
        cwd=build, capture_output=True, text=True,
    )
    if comp.returncode != 0:
        raise RuntimeError(f"P-LPI compile failed:\n{comp.stderr}")

    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    t0 = time.perf_counter()
    run = subprocess.run(["./PL1D"], cwd=build, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if run.returncode != 0:
        raise RuntimeError(f"P-LPI run failed:\n{run.stderr}")

    binpath = os.path.join(build, "result.bin")
    size = os.path.getsize(binpath)
    data = np.fromfile(binpath, dtype=np.complex128)
    if data.size != PLPI_NM:
        raise RuntimeError(
            f"result.bin has {data.size} complex values ({size} bytes), "
            f"expected {PLPI_NM} ({PLPI_NM * 16} bytes) — format mismatch?"
        )
    if not keep:
        shutil.rmtree(build, ignore_errors=True)
    return mu_grid(), data, wall


if __name__ == "__main__":
    mu, F, wall = run_plpi_1d()
    print(f"P-LPI code_1D: {F.size} points, {wall:.3f} s (1 thread)")
    print(f"  |integral| at mu=0: {abs(F[F.size // 2]):.6f}")
