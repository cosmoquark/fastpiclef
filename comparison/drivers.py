"""Per-code runners. Each returns the *raw* integral int exp(i*omega*phi) dx,
the fundamental computed quantity, so accuracy is compared convention-free.
(The normalised amplification F = integral / (2*pi*i/omega)^{d/2} is derived
for display only.)
"""

import numpy as np

import fastpiclef as fpl

# plind (read-only in refs/) predates numpy 2.0 and calls np.in1d, removed in
# numpy 2. Shim it to the modern np.isin (matching in1d's flatten-first shape).
if not hasattr(np, "in1d"):
    np.in1d = lambda ar1, ar2, **kw: np.isin(np.ravel(ar1), ar2, **kw)


def fastpiclef_run(builder, omega, method, **knobs):
    """builder() -> a fastpiclef phase; returns the raw integral."""
    return complex(builder().solve(omega=omega, method=method, **knobs).integral)


def plind_run(phi, dphi, omega, N=2001, domain=(-8.0, 8.0),
              delta=0.6, thresh=-20.0, tmax=1.5, dt=1e-2):
    """plind PL flow. phi, dphi are numpy-vectorised (accept complex arrays).

    plind integrates exp(expfun) with expfun = i*omega*phi and flows along the
    gradient of the Morse function h = Re(expfun); its complex gradient is
    grad = i*omega*conj(phi') (matches plind's Airy example gradh = 2i*conj(z)).
    """
    from plind.contour_dict import real_contour_1d
    from plind.plmodel import PLModel

    expfun = lambda z: 1j * omega * phi(z)
    grad = lambda z: 1j * omega * np.conj(dphi(z))
    pl = PLModel(real_contour_1d(N, domain), expfun, grad, expargs=[])
    pl.descend(delta, thresh, tmax, dt)
    pl.integrate()
    return complex(pl.integral[0])
