"""Driver for wolensing (`refs/wolensing`, git 7b9101f), the third-party
wave-optics code (P4, approved 2026-10-12).

wolensing computes the wave-optics amplification factor by integrating the
Fresnel integral in the *time* domain and Fourier transforming, using
lenstronomy for the lens models.  Its conventions (verified against its own
shipped SIS reference and against Nakamura-Deguchi on the point mass — see
`validate()`):

  * ``F(w, y) = (w / 2 pi i) int d^2x exp{i w [|x-y|^2/2 - psi(x)]}`` — the same
    normalisation as fastpiclef's ``.amplification`` (F = 1 for no lens);
  * its dimensionless frequency is ``w = 8 pi G M_Lz f / c^3 = 2 pi f Tscale``
    with ``Tscale = 4 (1+z_L) M_ref G / c^3`` — i.e. exactly fastpiclef's
    ``omega``.  **Watch the units**: the pipeline's *returned* array ``fs`` is the
    *physical* frequency (Hz) and its shipped test files store the physical
    *angular* frequency ``ws = 2 pi f`` (rad/s for the fixture's Tscale in
    seconds); the dimensionless ``w`` used here is therefore ``2 pi fs Tscale``
    (``run()`` converts; comparing the raw physical array with our omega
    produces a fake ``1/Tscale`` frequency stretch — the 2026-10-12 finding);
  * the Fermat potential is normalised by the *reference* point-mass Einstein
    radius ``thetaE(zL, zS, M_ref)``: the caller supplies lens kwargs in those
    dimensionless units and this driver scales them by ``thetaE`` before the
    call.  For a lens whose own scale equals the reference (the natural unit
    setup) the dimensionless potentials match fastpiclef's profiles exactly:
    e.g. SIS ``theta_E = thetaE`` <-> our ``psi0 = 1``, point mass
    ``POINT_MASS(theta_E = thetaE)`` <-> our ``m = 1``.

The integrator settings mirror wolensing's own test fixture (`test/conftest.py`)
so the tested code path is what runs.  Their pipeline is slow (time-domain
integration + FFT, ~2-5 min per case) and its accuracy is FFT/time-grid limited
(~1e-3); it is a *third-party* arbiter, not a truth.

Run: ``pixi run -e comparison python -m comparison.wolensing_driver``
"""

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_WS = os.path.join(_ROOT, "refs", "wolensing")
if _WS not in sys.path:
    sys.path.insert(0, _WS)


def theta_e(zL, zS, m_ref):
    """The reference point-mass Einstein radius (radians), wolensing's helper."""
    from wolensing.utils.utils import Einstein_radius

    return float(Einstein_radius(zL, zS, m_ref))


def t_scale(zL, m_ref):
    """wolensing's time scale ``4 (1+z_L) M_ref G / c^3`` (seconds)."""
    import wolensing.utils.constants as const

    return 4.0 * (1.0 + zL) * m_ref * const.M_sun * const.G / const.c**3


def run(
    lens_model_list,
    kwargs_lens_dimless,
    beta_dimless,
    *,
    zL=0.5,
    zS=1.0,
    m_ref=1e3,
    m_window=10.0,
    pixel_num=20000,
    time_step=1e-5,
    time_length=4.0,
    last_image_t=0.02,
    textend=10.0,
    freq_end=2000.0,
    type2=False,
    mu=1.0,
    verbose=False,
):
    """Run wolensing's full pipeline for a *dimensionless* lens setup.

    ``kwargs_lens_dimless``: lenstronomy kwargs in units of the reference
    Einstein radius (all length-like entries are scaled by ``thetaE`` here).
    ``beta_dimless``: source position in the same units.
    Returns ``(w, F)`` on their dense frequency grid (``w = 2 pi f Tscale`` =
    fastpiclef's omega).

    Defaults mirror their *notebook* (production) settings: ``TimeLength =
    4/Tscale``, ``TExtend = 10/Tscale``.  The shipped-SIS-reference validation
    passes ``time_length=0.13`` (their conftest value) to match the stored data.
    """
    from lenstronomy.LensModel.lens_model import LensModel

    import wolensing.amplification_factor.amplification_factor as af

    thetaE = theta_e(zL, zS, m_ref)
    thetaE_win = theta_e(zL, zS, m_window)
    Tsc = t_scale(zL, m_ref)

    kwargs_lens = [
        {k: float(v) * thetaE for k, v in kw.items()} for kw in kwargs_lens_dimless
    ]
    beta0, beta1 = (float(beta_dimless[0]) * thetaE, float(beta_dimless[1]) * thetaE)

    lens = LensModel(lens_model_list=lens_model_list)
    T = lens.fermat_potential
    t_centre = float(np.asarray(thetaE ** (-2) * T(0.0, 0.0, kwargs_lens, beta0, beta1)).ravel()[0])
    # Anchor guard: for a POINT_MASS the centre potential diverges and
    # lenstronomy returns a numerical-floor constant (~57.69, thetaE-
    # independent; their own notebook does exactly this), which throws the time
    # window far outside the image delays and silently drops the image signals.
    # When the centre value sits far above the potential's minimum, fall back to
    # the grid minimum (the primary image) -- the analogue of the finite
    # SIS/SIE centre value that makes the conftest recipe work.
    ng = 201
    gx = np.linspace(-100.0, 100.0, ng) * thetaE
    X, Y = np.meshgrid(gx, gx)
    with np.errstate(divide="ignore", invalid="ignore"):
        Tgrid = thetaE ** (-2) * T(X, Y, kwargs_lens, beta0, beta1)
    t_min = float(np.nanmin(Tgrid))
    T0 = t_centre
    anchor = "centre"
    if not np.isfinite(T0) or (T0 - t_min) > 5.0:
        T0 = t_min
        anchor = "grid-min"
    if verbose or anchor != "centre":
        print(f"  [T0 anchor: {anchor}; T0 = {T0:.6g} (centre value {t_centre:.6g}, grid min {t_min:.6g})]")

    kwargs_macro = {
        "source_pos_x": beta0,
        "source_pos_y": beta1,
        "theta_E": thetaE,
        "mu": mu,
    }
    kwargs_integrator = {
        "PixelNum": int(pixel_num),
        "PixelBlockMax": 2000,
        "WindowSize": 210.0 * thetaE_win,
        "WindowCenterX": 0,
        "WindowCenterY": 0,
        "T0": T0,
        "TimeStep": time_step / Tsc,
        "TimeMax": T0 + 1.0 / Tsc,
        "TimeMin": T0 - 0.1 / Tsc,
        "TimeLength": time_length / Tsc,
        "TExtend": textend / Tsc,
        "LastImageT": last_image_t / Tsc,
        "Tbuffer": 0,
        "Tscale": Tsc,
    }
    amp = af.amplification_factor(
        lens_model_list=lens_model_list,
        kwargs_lens=kwargs_lens,
        kwargs_macro=kwargs_macro,
        **kwargs_integrator,
    )
    amp.integrator(gpu=False)
    fs, Fws = amp.fourier(freq_end=freq_end, type2=type2)
    # Convert the physical frequency (Hz) to the dimensionless Fresnel
    # frequency w = 2 pi f Tscale = fastpiclef's omega.
    w = 2.0 * np.pi * np.asarray(fs) * Tsc
    return w, np.asarray(Fws)


def F_at(w_targets, *args, **kwargs):
    """``|F|`` interpolated onto ``w_targets`` (their grid is dense)."""
    w, F = run(*args, **kwargs)
    order = np.argsort(w)
    w, F = w[order], F[order]
    out = np.interp(np.asarray(w_targets, dtype=float), w, F.real) + 1j * np.interp(
        np.asarray(w_targets, dtype=float), w, F.imag
    )
    return out


def _validate_stored_vs_ours(ws_sample=(10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)):
    """The decisive convention check, no live pipeline: wolensing's *stored*
    SIS reference (physical angular frequency ws, rad/s) against our exact RC
    engine at the dimensionless w = ws * Tscale.  If the mapping is right the
    two curves must coincide to wolensing's FFT/time-window accuracy."""
    import fastpiclef as fpl

    ws_file = os.path.join(_WS, "test", "test_sis_ws.txt")
    Fws_file = os.path.join(_WS, "test", "test_sis_Fws.txt")
    ws_stored = np.loadtxt(ws_file)
    F_stored = np.array([complex(line.strip()) for line in open(Fws_file)])
    zL, zS, m_ref = 0.5, 1.0, 1e3
    Tsc = t_scale(zL, m_ref)
    y = 0.1
    print("  stored |F| vs our exact SIS at w = ws * Tscale:")
    rows = []
    for ws in ws_sample:
        w = ws * Tsc
        ours = abs(
            complex(
                fpl.LensField([fpl.SISComp(0.0, 0.0, 1.0)], beta=(y, 0.0))
                .solve(omega=w, tol=1e-14)
                .amplification
            )
        )
        ref = float(np.interp(ws, ws_stored, np.abs(F_stored)))
        rows.append((ws, w, ref, ours, abs(ref - ours) / ours))
        print(
            f"    ws={ws:8.1f} rad/s -> w={w:8.4f}: wolensing {ref:.6f}  ours {ours:.6f}  "
            f"rel {abs(ref - ours) / ours:.2e}"
        )
    return rows


def _validate_sis_reference(w_targets=(0.5, 1.0, 3.0, 10.0)):
    """Reproduce wolensing's own shipped SIS case (their conftest) live and
    compare with its stored frequency-domain reference data.

    The shipped `test_sis_ws.txt` holds the *physical* angular frequency
    `ws = 2 pi f` (rad/s; its maximum 1.26e4 = 2 pi * freq_end/Hz), so the
    dimensionless comparison frequency is `ws * Tscale`."""
    ws_file = os.path.join(_WS, "test", "test_sis_ws.txt")
    Fws_file = os.path.join(_WS, "test", "test_sis_Fws.txt")
    ws_stored = np.loadtxt(ws_file)
    F_stored = np.array(
        [complex(line.strip()) for line in open(Fws_file)]
    )
    zL, zS, m_ref = 0.5, 1.0, 1e3
    Tsc = t_scale(zL, m_ref)
    print(f"  stored SIS reference: {ws_stored.size} points, physical ws in "
          f"[{ws_stored[0]:.3g}, {ws_stored[-1]:.3g}] rad/s "
          f"= dimensionless w in [{ws_stored[0] * Tsc:.3g}, {ws_stored[-1] * Tsc:.3g}]")

    w, F = run(["SIS"], [{"theta_E": 1.0, "center_x": 0.0, "center_y": 0.0}],
               (0.1, 0.0), zL=zL, zS=zS, m_ref=m_ref, time_length=0.13)
    print(f"  live run: {w.size} points, dimensionless w in [{w[0]:.3g}, {w[-1]:.3g}]")
    ref_at = np.interp(np.asarray(w_targets), ws_stored * Tsc, np.abs(F_stored))
    live_at = np.interp(np.asarray(w_targets), w, np.abs(F))
    rel = np.abs(live_at - ref_at) / ref_at
    print("  |F| live vs stored (dimensionless w):  w = " + ", ".join(f"{t:g}" for t in w_targets))
    print("                                         rel = " + ", ".join(f"{r:.2e}" for r in rel))
    return float(np.max(rel))


def _validate_point_mass_nd(w_targets=(0.5, 1.0, 3.0, 10.0), y=0.1):
    """The point-mass convention check against Nakamura-Deguchi (external)."""
    from .referee import point_mass_exact

    zL, zS, m_ref = 0.5, 1.0, 1e3
    F = F_at(w_targets, ["POINT_MASS"],
             [{"theta_E": 1.0, "center_x": 0.0, "center_y": 0.0}], (y, 0.0),
             zL=zL, zS=zS, m_ref=m_ref)
    ref = np.array([point_mass_exact(w, y) for w in w_targets])
    rel = np.abs(np.abs(F) - np.abs(ref)) / np.abs(ref)
    print("  |F| wolensing vs ND:  w = " + ", ".join(f"{t:g}" for t in w_targets))
    print("                        rel = " + ", ".join(f"{r:.2e}" for r in rel))
    return float(np.max(rel))


if __name__ == "__main__":
    print("== wolensing driver validation ==")
    print("-- (a) its own shipped SIS reference (live pipeline) --")
    r_sis = _validate_sis_reference()
    print("-- (b) point mass vs the Nakamura-Deguchi closed form --")
    r_nd = _validate_point_mass_nd()
    print(f"summary: max rel vs stored SIS {r_sis:.2e}; vs ND {r_nd:.2e}")
