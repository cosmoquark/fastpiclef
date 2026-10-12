"""Driver for glworia (`refs/glworia`, JAX), the third-party wave-optics code.

Conventions: glworia's ``F(w, y) = (w / 2 pi i) int d^2x exp{i w [|x-y|^2/2 -
psi(x)]}`` — identical to fastpiclef's ``.amplification``.  Its ``Psi`` is in
units of the scale radius with ``lens_params = [kappa]``; for NFW that kappa is
the same normalisation as our ``Profile::Nfw { psi0, xs }`` evaluated at
``xs = 1``.  The convention is verified against glworia's own shipped reference
curve and our Hankel referee before any comparison (see
``comparison/glworia_compare.py``).

Environment notes (2026-10-12, jax 0.10.2):
  * glworia's metadata says Python <3.12, but the source runs on our 3.14 env
    when imported from ``refs/`` (as with FIONA/P-LPI; ``refs/`` stays
    read-only and is never modified).
  * ``glworia/amp/root.py`` imports ``call`` from
    ``jax.experimental.host_callback``, removed in JAX >= 0.5.  The import is
    *unused* (only line 6 references it), so a no-op shim module is registered
    here before glworia is imported.
  * JAX owns its XLA thread pool; the harness's ``RAYON``/``OMP`` pins do not
    apply.  Timings are warm-up + repeat and are labelled as such.
"""

import os
import sys
import types

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_GL = os.path.join(_ROOT, "refs", "glworia")
if _GL not in sys.path:
    sys.path.insert(0, _GL)


def _shim_host_callback():
    if "jax.experimental.host_callback" in sys.modules:
        return
    try:
        import jax.experimental.host_callback  # noqa: F401

        return
    except Exception:  # noqa: BLE001
        pass
    mod = types.ModuleType("jax.experimental.host_callback")

    def call(*args, **kwargs):  # pragma: no cover - never called by glworia
        raise NotImplementedError("host_callback shim: glworia does not call it")

    mod.call = call
    sys.modules["jax.experimental.host_callback"] = mod
    try:
        import jax.experimental as _exp

        _exp.host_callback = mod
    except Exception:  # noqa: BLE001
        pass


_shim_host_callback()

from jax import config  # noqa: E402

config.update("jax_enable_x64", True)


def _api():
    from glworia.amp.amplification_factor import (
        amplification_computation_prep,
        compute_F,
        crtical_curve_interpolants,
    )
    from glworia.amp.lens_model import get_lens_model
    from glworia.amp.root import make_crit_curve_helper_func

    return (
        get_lens_model,
        amplification_computation_prep,
        crtical_curve_interpolants,
        compute_F,
        make_crit_curve_helper_func,
    )


def glworia_F(lm_name, ws, y, lens_params, N=200, T0_max=1000.0, param_arr=None, n_param=48):
    """F(w) at one source position via glworia's production path.

    ``lm_name`` in {"NFW", "gSIS", "CIS"}; ``y`` a 2-vector (glworia uses the
    first component); ``lens_params`` a 1-vector; ``N``/``T0_max`` are the
    convergence knobs (interpolation nodes / max time delay).
    """
    import jax.numpy as jnp

    get_lens_model, prep, crit_interp, compute_F, crit_helper = _api()
    lm = get_lens_model(lm_name)
    Psi = lm.get_Psi()
    T_funcs, helper_funcs = prep(Psi)
    if param_arr is None:
        p0 = float(np.atleast_1d(np.asarray(lens_params, dtype=float))[0])
        param_arr = jnp.linspace(max(0.05, p0 - 0.5), p0 + 0.5, n_param)
    crit_funcs = crit_interp(param_arr, T_funcs, crit_helper(T_funcs))
    F, *_ = compute_F(
        jnp.asarray(np.atleast_1d(ws), dtype=float),
        jnp.asarray(y, dtype=float),
        jnp.atleast_1d(jnp.asarray(lens_params, dtype=float)),
        T_funcs,
        helper_funcs,
        crit_funcs,
        N,
        T0_max,
    )
    return np.asarray(F)


def glworia_reference_data():
    """glworia's shipped NFW reference curve (`test/data/F_interp.npy`):
    kappa = 3, y = (0.1, 0), w = linspace(1e-2, 1e3, 1e5).  Read-only."""
    import jax.numpy as jnp

    F = np.asarray(jnp.load(os.path.join(_GL, "test", "data", "F_interp.npy")))
    w = np.linspace(1e-2, 1e3, 10**5)
    return w, F


if __name__ == "__main__":
    w, F = glworia_reference_data()
    print(f"glworia reference: {F.shape} at w in [{w[0]:g}, {w[-1]:g}]")
    Fs = glworia_F("NFW", [0.5, 1.0, 10.0], (0.1, 0.0), [3.0], N=200, T0_max=1000.0)
    print("live:", Fs)
