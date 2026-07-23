"""Shared 1D test cases: a fastpiclef phase builder + numpy-vectorised phi/phi'
(for plind) + method-independent ground truth.
"""

from dataclasses import dataclass
from typing import Callable

import fastpiclef as fpl

from .referee import airy_exact, lens_referee


@dataclass
class Case:
    name: str
    omega: float
    builder: Callable[[], object]  # () -> fastpiclef phase
    phi: Callable                  # numpy-vectorised, complex-capable
    dphi: Callable
    truth: complex                 # fine reference (thimble / closed form)
    referee: complex               # independent brute-force cross-check (~1e-6 floor)


def rational_case(beta, kappa, omega):
    phi = lambda z: 0.5 * (z - beta) ** 2 - kappa / (1 + z * z)
    dphi = lambda z: (z - beta) + 2 * kappa * z / (1 + z * z) ** 2
    # Fine truth: our machine-precision thimble (validated in the test suite by
    # brute-force agreement to ~1e-6, thimble==mesh to 1e-13, and 3-way PL vs
    # plind/P-LPI). The brute-force referee is kept as an external sanity check.
    thimble = complex(
        fpl.RationalLens(beta=beta, kappa=kappa).solve(omega=omega, method="thimble").integral
    )
    referee = lens_referee(lambda x: 0.5 * (x - beta) ** 2 - kappa / (1 + x * x), beta, omega)
    return Case(
        f"rational b={beta} k={kappa} w={omega:g}",
        omega,
        lambda: fpl.RationalLens(beta=beta, kappa=kappa),
        phi,
        dphi,
        thimble,
        referee,
    )


def airy_case(y, omega):
    phi = lambda z: z ** 3 / 3 - y * z
    dphi = lambda z: z * z - y
    exact = airy_exact(y, omega)  # closed form is the truth AND the sanity check
    return Case(
        f"airy y={y} w={omega:g}",
        omega,
        lambda: fpl.AiryPhase(y=y),
        phi,
        dphi,
        exact,
        exact,
    )


# The headline set: P-LPI's native strong-lens regime, a gentle lens, and Airy.
CASES = [
    rational_case(0.3, -2.0, 10.0),   # strong converging, multi-image (P-LPI native, nu=10)
    rational_case(0.3, 0.2, 1.0),     # gentle diverging
    airy_case(1.0, 1.0),              # cubic phase, exact reference
]

# For the convergence-order curve, a case where both flow methods converge cleanly.
CONVERGENCE_CASE = rational_case(0.3, 0.2, 2.0)
