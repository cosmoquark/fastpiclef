"""Method-independent ground truth for 1D Fresnel integrals.

For lens phases phi(x) = (x-beta)^2/2 - kappa*psi(x) with a decaying potential,
adaptive quadrature of the raw oscillatory integrand on [-L, L] plus the exact
Fresnel-function tails of the leading (x-beta)^2/2 term. Verified L-independent
in the test suite; this is the same technique used to validate the Gap-1/2 work.
"""

import math

import mpmath as mp
from scipy.integrate import quad
from scipy.special import airy, fresnel


def _fresnel_tail(a, omega):
    """int_a^inf exp(i*omega*t^2/2) dt via the Fresnel integrals S, C."""
    x = a * math.sqrt(omega / math.pi)
    s, c = fresnel(x)
    return math.sqrt(math.pi / omega) * complex(0.5 - c, 0.5 - s)


def lens_referee(phi_real, beta, omega, L=40.0):
    """Raw integral int_R exp(i*omega*phi) dx for a lens with quadratic tails."""
    re = quad(lambda x: math.cos(omega * phi_real(x)), -L, L, limit=8000, epsabs=1e-11)[0]
    im = quad(lambda x: math.sin(omega * phi_real(x)), -L, L, limit=8000, epsabs=1e-11)[0]
    return complex(re, im) + _fresnel_tail(L - beta, omega) + _fresnel_tail(L + beta, omega)


def airy_exact(y, omega):
    """int_R exp(i*omega*(x^3/3 - y x)) dx = 2*pi*omega^{-1/3}*Ai(-omega^{2/3} y)."""
    return complex(2 * math.pi * omega ** (-1 / 3) * airy(-(omega ** (2 / 3)) * y)[0], 0.0)


def point_mass_exact(w, y):
    """Wave-optics amplification F(w, y) for the isolated point mass psi = ln|x|.

    Nakamura (1998) / Takahashi & Nakamura (2003) closed form:

        F = exp[pi*w/4 + i*(w/2)*(ln(w/2) - 2*phi_m)]
            * Gamma(1 - i*w/2) * 1F1(i*w/2, 1; i*w*y^2/2)

    with x_m = (y + sqrt(y^2+4))/2 the brighter image and the Fermat-potential
    offset phi_m = (x_m - y)^2/2 - ln(x_m). This is F for the convention
    F(w,y) = (w/2pi i) int d^2x exp{i w [ |x-y|^2/2 - ln|x| ]} (free space -> 1).

    Uses mpmath for the complex confluent hypergeometric 1F1 and Gamma (scipy's
    are real-argument only); robust up to large w. Returns a Python complex.
    """
    w = float(w)
    y = float(y)
    xm = 0.5 * (y + math.sqrt(y * y + 4.0))
    phi_m = 0.5 * (xm - y) ** 2 - math.log(xm)
    prefac = mp.e ** (mp.pi * w / 4 + 1j * (w / 2) * (mp.log(w / 2) - 2 * phi_m))
    val = prefac * mp.gamma(1 - 1j * w / 2) * mp.hyp1f1(1j * w / 2, 1, 1j * w * y * y / 2)
    return complex(val)
