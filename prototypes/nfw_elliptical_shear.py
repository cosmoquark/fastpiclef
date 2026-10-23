#!/usr/bin/env python3
"""Elliptical NFW + external shear: |F(omega, y)| sky maps.

This is the case of ``prototypes/new_test.txt``, which is a summary of
``refs/FIONA/elliptical_sheared_lenses.ipynb`` section 5.4:

    M200 = 1e13 Msun, z_lens = 0.5, z_source = 1.5,
    c200 = 15, 30, 60,  e1 = 0.2, e2 = 0,  gamma_ext = (0.03, 0.01),
    y1, y2 in [-1.5, 1.5] arcsec,  omega = 1, 10, 100.

The cosmological conversion is the notebook's
``FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.048)`` + ``LensCosmo.nfw_physical2angle``;
the resulting (R_s, alpha_Rs) are tabulated below so the script does not need
astropy/lenstronomy at run time.  ``--from-lenstronomy`` recomputes them.

Two solvers
-----------
``fpl``    -- the exact RC slice engine of this repository.  The elliptical
              NFW potential is *not* in fpl's native class, but it reduces to
              it exactly (see below), so these maps are exact (1e-5 or better
              at the default tolerance).
``gwlens`` -- the third-party GWLENS ``Solver2D`` (a local clone, see
              ``--gwlens-dir``), which accepts the elliptical potential
              directly and evaluates a whole source grid with one NUFFT.
              It is the fast dense-map engine; fpl is the precision anchor.

The exact reduction
-------------------
Write the elliptical NFW potential as ``psi_NFW(s)`` with

    s^2 = (1-e) x1'^2 + (1+e) x2'^2        (x' = ellipticity principal frame)

where lenstronomy's ``(e1, e2)`` maps to an axis ratio ``q = (1-|e|)/(1+|e|)``,
a position angle ``phi = atan2(e2,e1)/2``, and ``e = |1-q^2|/(1+q^2)``.
With ``Se = diag(1/sqrt(1-e), 1/sqrt(1+e))`` and ``x = Se u`` the NFW becomes
circular in ``u`` and the Fermat quadratic becomes

    1/2 x^T (I - G) x = 1/2 u^T A u,    A = Se (I - G) Se,

with ``G = [[gamma1, gamma2], [gamma2, -gamma1]]``.  ``A`` is symmetric, so it
is diagonalised by a rotation; fpl's ``LensField`` takes exactly a diagonal
macro quadratic ``(kappa, gamma1, gamma2)`` plus a circular ``NFWComp``.  The
source transforms as ``beta_u = Se R beta`` (R the ellipticity rotation) and
the Jacobian is ``d^2x = d^2u / sqrt(1-e^2)``, hence

    F_ell(beta) = F_fpl(beta_u) / sqrt(1-e^2).

That is exact: the reduction was checked against GWLENS's direct elliptical
potential at 5 sources x 3 frequencies to <= 6e-13 in |F|.

Gotcha: ``LensField.solve`` rotates its ``beta`` argument into the macro
diagonal frame internally, while ``LensField.map`` expects ``beta0s/beta1s``
*already in that diagonal frame*.  The helper functions below keep the two
conventions apart (``physical_to_solver_input`` vs ``physical_to_map``).

Measured behaviour (this box, 6 physical cores)
-----------------------------------------------
fpl exact maps at 500x500, tol 1e-6:

    c200   omega=1   omega=10   omega=100
    15     ~3 s      ~10 s      ~90 s
    30     ~3 s      ~10 s      stalls (the compact cusp defeats the adaptive
    60     ~3 s      ~11 s      contour: > minutes for even a 4x4 probe)

GWLENS maps at 500x500: ~0.4 s (omega=1, 10), ~11 s (omega=100); the full
3x3 GWLENS run measured 50 s in total on this box.

The fpl maps are exact at their native grid nodes (node checks 1e-13..1.8e-9
against point solves).  The dark interference nulls at c200=60, omega=10 are
narrower than the 500-grid spacing (~0.006 arcsec): fpl and GWLENS agree to
~1e-9 *at the nulls* in scalar solves, but a map value *between* nodes is not
meaningful (fpl samples the nulls, GWLENS's NUFFT smooths them).

So for the requested 3x3 figure this script uses fpl for every panel it can
reach exactly and GWLENS for the two omega=100 compact-halo panels (clearly
labelled in the figure).  ``--solver fpl --force-fpl`` attempts the hard
panels anyway; ``--solver gwlens`` uses GWLENS for all nine.

Usage
-----
    pixi run -e comparison python prototypes/nfw_elliptical_shear.py
    pixi run -e comparison python prototypes/nfw_elliptical_shear.py --quick
    pixi run -e comparison python prototypes/nfw_elliptical_shear.py --solver gwlens
    pixi run -e comparison python prototypes/nfw_elliptical_shear.py --from-lenstronomy

GWLENS is available two ways:

* the dedicated pixi environment (preferred):

      pixi run -e gwlens nfw-elliptical

* a scratch clone used from the comparison environment:

      git clone https://github.com/hsreekanth15/GWLENS /tmp/GWLENS
      pixi run -e comparison python prototypes/nfw_elliptical_shear.py \
          --gwlens-dir /tmp/GWLENS

See ``prototypes/nfw_elliptical_shear_options.md`` for the full options list
and derivations, and ``prototypes/nfw_elliptical_shear_STATUS.md`` for the
current status / resume checklist.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

import fastpiclef as fpl

LN2 = np.log(2.0)
Y_EXTENT = 1.5
DEFAULT_N = 500
DEFAULT_OMEGAS = (1.0, 10.0, 100.0)
DEFAULT_C200 = (15.0, 30.0, 60.0)

# FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.048), z_l=0.5, z_s=1.5, M200=1e13:
# (R_s [arcsec], alpha_Rs [arcsec]) from LensCosmo.nfw_physical2angle.
NFW_TABLE = {
    15.0: (4.05686713256022546, 1.51259845796567971),
    30.0: (2.02843356628011273, 2.25099458385512863),
    60.0: (1.01421678314005637, 3.55038709413520559),
}


# ---------------------------------------------------------------------------
# The exact reduction
# ---------------------------------------------------------------------------
def ellipse_params(e1: float, e2: float):
    """lenstronomy ellipticity -> (position angle, axis ratio, scaling e)."""
    c = min(float(np.hypot(e1, e2)), 0.9999)
    q = (1.0 - c) / (1.0 + c)
    e = abs(1.0 - q * q) / (1.0 + q * q)
    phi = 0.5 * np.arctan2(e2, e1)
    return phi, q, e


def build_field(Rs: float, alpha_Rs: float, e1: float = 0.2, e2: float = 0.0,
                gamma1: float = 0.03, gamma2: float = 0.01):
    """Exact fpl LensField equivalent of elliptical NFW + external shear."""
    phi, q, e = ellipse_params(e1, e2)
    R = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
    Se = np.diag([1.0 / np.sqrt(1.0 - e), 1.0 / np.sqrt(1.0 + e)])
    G = np.array([[gamma1, gamma2], [gamma2, -gamma1]])
    A = Se @ (np.eye(2) - G) @ Se
    mu, V = np.linalg.eigh(A)
    psi_ang = float(np.arctan2(V[1, 0], V[0, 0]))
    kappa = 1.0 - 0.5 * (mu[0] + mu[1])
    gamma = 0.5 * (mu[1] - mu[0])
    psi0 = alpha_Rs * Rs / (1.0 - LN2)  # fpl NFW normalisation
    lf = fpl.LensField(
        [fpl.NFWComp(0.0, 0.0, psi0, Rs)],
        beta=(0.0, 0.0),
        kappa=float(kappa),
        gamma1=float(gamma * np.cos(2.0 * psi_ang)),
        gamma2=float(gamma * np.sin(2.0 * psi_ang)),
    )
    info = dict(
        Rs=float(Rs), alpha_Rs=float(alpha_Rs), psi0=float(psi0),
        e1=float(e1), e2=float(e2), gamma1=float(gamma1), gamma2=float(gamma2),
        phi=float(phi), q=float(q), e=float(e),
        kappa=float(kappa), gamma=float(gamma), psi_ang=psi_ang,
        Se=Se, R=R, A=A, mu=mu, measure=1.0 / np.sqrt(1.0 - e * e),
    )
    return lf, info


def physical_to_solver_input(x, y, info):
    """Physical source -> the beta passed to LensField.solve (u frame)."""
    R = info["R"]
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xl = R[0, 0] * x + R[0, 1] * y
    yl = R[1, 0] * x + R[1, 1] * y
    return xl / np.sqrt(1.0 - info["e"]), yl / np.sqrt(1.0 + info["e"])


def physical_to_map(x, y, info):
    """Physical source -> the beta frame expected by LensField.map.

    ``map`` does *not* apply the internal shear rotation, so this includes it.
    """
    u0, u1 = physical_to_solver_input(x, y, info)
    th = info["psi_ang"]
    c, s = np.cos(th), np.sin(th)
    return c * u0 + s * u1, -s * u0 + c * u1


def map_to_physical(b0, b1, info):
    """Inverse of physical_to_map; used for pcolormesh coordinates."""
    th = info["psi_ang"]
    c, s = np.cos(th), np.sin(th)
    u0 = c * b0 - s * b1
    u1 = s * b0 + c * b1
    xl = u0 * np.sqrt(1.0 - info["e"])
    yl = u1 * np.sqrt(1.0 + info["e"])
    R = info["R"]
    x = R[0, 0] * xl + R[1, 0] * yl
    y = R[0, 1] * xl + R[1, 1] * yl
    return x, y


def fpl_solve_point(info, beta, omega, tol=1e-10):
    """Exact |F| at one physical source position."""
    bu = physical_to_solver_input(beta[0], beta[1], info)
    lf = fpl.LensField(
        [fpl.NFWComp(0.0, 0.0, info["psi0"], info["Rs"])],
        beta=(float(bu[0]), float(bu[1])),
        kappa=info["kappa"],
        gamma1=info["gamma"] * np.cos(2.0 * info["psi_ang"]),
        gamma2=info["gamma"] * np.sin(2.0 * info["psi_ang"]),
    )
    r = lf.solve(omega, tol=tol)
    return abs(r.amplification) * info["measure"], r.error * info["measure"], r


def fpl_map_native(lf, info, n, omega, tol=1e-6, omega_map=None):
    """Exact map on a native rectangular grid in the map beta frame.

    The grid covers the bounding box of the transformed physical square, so
    ``pcolormesh(map_to_physical(b0, b1), F)`` shows the exact values (no
    interpolation) over the requested [-1.5, 1.5]^2 field.
    """
    corners = np.array([[x, y] for x in (-Y_EXTENT, Y_EXTENT)
                        for y in (-Y_EXTENT, Y_EXTENT)])
    d0, d1 = physical_to_map(corners[:, 0], corners[:, 1], info)
    b0 = np.linspace(d0.min(), d0.max(), n)
    b1 = np.linspace(d1.min(), d1.max(), n)
    t = time.perf_counter()
    m = lf.map(list(b0), list(b1), omega if omega_map is None else omega_map, tol=tol)
    wall = time.perf_counter() - t
    F = np.abs(np.array(m.amplification)) * info["measure"]
    E = np.array(m.error) * info["measure"]
    B0, B1 = np.meshgrid(b0, b1, indexing="ij")
    X, Y = map_to_physical(B0, B1, info)
    return dict(X=X, Y=Y, F=F, err=E, b0=b0, b1=b1, wall=wall, info=info,
                inner_solves=int(m.inner_solves), method="fpl")


# ---------------------------------------------------------------------------
# GWLENS (optional fast dense-map engine)
# ---------------------------------------------------------------------------
def import_gwlens(gwlens_dir=None):
    """Import GWLENS: installed package (gwlens env) or a scratch clone."""
    if gwlens_dir:
        if gwlens_dir not in sys.path:
            sys.path.insert(0, gwlens_dir)
    try:
        from gwlens.solver import solver2d as gs
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "GWLENS is not importable.  Preferred: use the dedicated env\n"
            "    pixi run -e gwlens nfw-elliptical\n"
            "or `pixi install -e gwlens`.  Fallback: clone it outside refs/\n"
            "    git clone https://github.com/hsreekanth15/GWLENS /tmp/GWLENS\n"
            "and pass --gwlens-dir /tmp/GWLENS from the comparison env.\n"
            f"Original error: {exc}"
        ) from exc
    return gs


def gwlens_lens(gs, Rs, alpha_Rs, e1=0.2, e2=0.0, gamma1=0.03, gamma2=0.01):
    """The same elliptical NFW potential + external shear, for GWLENS."""
    phi, q, e = ellipse_params(e1, e2)
    psi0 = alpha_Rs * Rs / (1.0 - LN2)
    c, s = np.cos(phi), np.sin(phi)
    sq1, sq2 = np.sqrt(1.0 - e), np.sqrt(1.0 + e)

    def psi(x1, x2):
        xp = (c * x1 + s * x2) * sq1
        yp = (-s * x1 + c * x2) * sq2
        v = np.sqrt(xp * xp + yp * yp + 0j) / Rs
        return 0.5 * psi0 * (np.log(v / 2.0) ** 2 + np.arccos(1.0 / v) ** 2)

    return gs.Lens(psi, singular=[(0.0, 0.0)], q_inf=0.0) + \
        gs.external_shear(gamma1=gamma1, gamma2=gamma2)


def gwlens_map(gs, lens, n, omega, y_extent=Y_EXTENT):
    g = np.linspace(-y_extent, y_extent, n)
    Y1, Y2 = np.meshgrid(g, g, indexing="xy")
    y1, y2 = Y1.ravel(), Y2.ravel()
    t0 = time.perf_counter()
    solver = gs.Solver2D(lens, (0.0, 0.0), omega, omega,
                         y_radius=y_extent * np.sqrt(2.0) * 1.01,
                         map_ext=1.5, threads="auto", n_workers=1)
    build = time.perf_counter() - t0
    t = time.perf_counter()
    F = np.asarray(solver.map(omega, y1, y2)).reshape(n, n)
    wall = time.perf_counter() - t
    solver.close()
    return dict(X=Y1, Y=Y2, F=np.abs(F), err=np.zeros_like(np.abs(F)),
                wall=wall + build, build=build, method="gwlens")


def gwlens_point(gs, lens, beta, omega):
    return float(abs(complex(gs.amplification(omega, beta, lens))))


# ---------------------------------------------------------------------------
# Feasibility of the exact engine (measured envelope, not a hard theorem)
# ---------------------------------------------------------------------------
def fpl_feasible(c200: float, omega: float) -> bool:
    """Whether the exact RC engine reaches this panel in reasonable time.

    Measured 2026-10-22: all c200 up to omega ~ 20; c200=15 up to omega=100;
    c200=30 stalls above ~50; c200=60 above ~20 (the compact NFW cusp makes
    the adaptive contour refine without bound at high omega).
    """
    if omega <= 20.0:
        return True
    return c200 <= 15.0 and omega <= 100.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def panel_validation(info, panel, c200, omega, gs, lens, rng, nval=6):
    """Validate a computed panel.

    ``node_max_rel``   -- the map at four of its *own* nodes vs a high-accuracy
                          point solve of the same solver (checks the batched
                          contour / beta frame; exact at the nodes).
    ``fpl_vs_gwlens``  -- six random physical points, fpl exact vs GWLENS
                          (checks the model reduction; only where fpl reaches).
    """
    out = {}
    Fp = np.abs(panel["F"])
    if panel["method"] == "fpl" and fpl_feasible(c200, omega):
        rels = []
        for _ in range(4):
            i = int(rng.integers(0, len(panel["b0"])))
            j = int(rng.integers(0, len(panel["b1"])))
            x, y = map_to_physical(panel["b0"][i], panel["b1"][j], info)
            v, _, _ = fpl_solve_point(info, (x, y), omega, tol=1e-10)
            rels.append(abs(panel["F"][i, j] - v) / v)
        out["node_max_rel"] = float(max(rels))
    elif gs is not None and lens is not None:
        # GWLENS map vs its own point solver *at a grid node* (not between
        # nodes: the dark interference nulls are narrower than the spacing).
        rels = []
        for _ in range(4):
            i = int(rng.integers(0, Fp.shape[0]))
            j = int(rng.integers(0, Fp.shape[1]))
            x, y = float(panel["X"][i, j]), float(panel["Y"][i, j])
            vg = gwlens_point(gs, lens, (x, y), omega)
            rels.append(abs(Fp[i, j] - vg) / vg)
        out["node_max_rel"] = float(max(rels))
    if gs is not None and lens is not None and fpl_feasible(c200, omega):
        rels = []
        for x, y in rng.uniform(-1.35, 1.35, size=(nval, 2)):
            vf, _, _ = fpl_solve_point(info, (x, y), omega, tol=1e-10)
            vg = gwlens_point(gs, lens, (x, y), omega)
            rels.append(abs(vf - vg) / vg)
        out["fpl_vs_gwlens"] = float(max(rels))
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_figure(results, c200s, omegas, outdir, scale="linear", dpi=200):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import Normalize, LogNorm

    nrow, ncol = len(c200s), len(omegas)
    fig = Figure(figsize=(4.6 * ncol, 4.4 * nrow), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = fig.subplots(nrow, ncol, squeeze=False)

    for i, c200 in enumerate(c200s):
        row = [results[(c200, w)]["F"] for w in omegas]
        lo = min(float(np.nanmin(a)) for a in row)
        hi = max(float(np.nanmax(a)) for a in row)
        if scale == "log":
            norm = LogNorm(vmin=max(lo, hi * 1e-3), vmax=hi)
        else:
            norm = Normalize(vmin=lo, vmax=hi)
        ims = []
        for j, w in enumerate(omegas):
            r = results[(c200, w)]
            ax = axes[i, j]
            im = ax.pcolormesh(r["X"], r["Y"], r["F"], shading="auto",
                               cmap="magma", norm=norm, rasterized=True)
            ims.append(im)
            ax.set_xlim(-Y_EXTENT, Y_EXTENT)
            ax.set_ylim(-Y_EXTENT, Y_EXTENT)
            ax.set_aspect("equal")
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([-1, 0, 1])
            if i == 0:
                ax.set_title(rf"$\omega={w:g}$", fontsize=13)
            if j == 0:
                ax.set_ylabel(rf"$c_{{200}}={c200:g}$" + "\n" + r"$y_2$")
            else:
                ax.set_ylabel("")
            ax.set_xlabel(r"$y_1$")
            tag = "fpl exact" if r["method"] == "fpl" else "GWLENS"
            ax.text(0.03, 0.96, tag, transform=ax.transAxes, va="top",
                    color="white", fontsize=8, alpha=0.9)
        cb = fig.colorbar(ims[-1], ax=axes[i, :], shrink=0.95, pad=0.02)
        cb.set_label(r"$|F(\omega,\mathbf{y})|$")
    path = os.path.join(outdir, f"nfw_elliptical_shear_{scale}.png")
    fig.savefig(path, dpi=dpi)
    return path


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="grid points per axis")
    ap.add_argument("--omegas", type=float, nargs="+", default=list(DEFAULT_OMEGAS))
    ap.add_argument("--c200", type=float, nargs="+", default=list(DEFAULT_C200))
    ap.add_argument("--tol", type=float, default=1e-6, help="fpl map tolerance")
    ap.add_argument("--solver", choices=["auto", "fpl", "gwlens"], default="auto")
    ap.add_argument("--force-fpl", action="store_true",
                    help="attempt fpl even outside its measured envelope")
    ap.add_argument("--gwlens-dir", default=os.environ.get("GWLENS_DIR", ""),
                    help="local GWLENS clone; empty = use the installed package "
                         "(the dedicated `gwlens` pixi environment)")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__),
                                                     "results", "nfw_elliptical_shear"))
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--save-data", action="store_true",
                    help="also write a compact maps.npz (float32 |F| grids; "
                         "the default keeps only the figure and summary)")
    ap.add_argument("--quick", action="store_true", help="n=64, no validation")
    ap.add_argument("--from-lenstronomy", action="store_true",
                    help="recompute (Rs, alpha_Rs) with astropy/lenstronomy")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    if args.quick:
        args.n = min(args.n, 64)
        # keep a quick smoke test from overwriting the full-resolution outputs
        args.outdir = args.outdir.rstrip("/") + "_quick"

    nfw = dict(NFW_TABLE)
    if args.from_lenstronomy:
        from astropy.cosmology import FlatLambdaCDM
        from lenstronomy.Cosmo.lens_cosmo import LensCosmo
        cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3, Ob0=0.048)
        lc = LensCosmo(z_lens=0.5, z_source=1.5, cosmo=cosmo)
        for c in args.c200:
            Rs, aRs = lc.nfw_physical2angle(M=1e13, c=c)
            nfw[c] = (float(Rs), float(aRs))
            print(f"lenstronomy: c200={c:g}  Rs={Rs:.12f}  alpha_Rs={aRs:.12f}")

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    results = {}
    gs = None
    use_gwlens = args.solver in ("auto", "gwlens")
    if use_gwlens:
        gs = import_gwlens(args.gwlens_dir)

    for c200 in args.c200:
        Rs, aRs = nfw[float(c200)]
        lf, info = build_field(Rs, aRs)
        lens = gwlens_lens(gs, Rs, aRs) if gs is not None else None
        for omega in args.omegas:
            want_fpl = args.solver == "fpl" or (
                args.solver == "auto" and (fpl_feasible(c200, omega) or args.force_fpl))
            if args.solver == "fpl" and not fpl_feasible(c200, omega) and not args.force_fpl:
                raise SystemExit(
                    f"fpl cannot reach c200={c200:g}, omega={omega:g} "
                    f"(see fpl_feasible); use --solver auto/gwlens or --force-fpl")
            print(f"[{c200:g} / {omega:g}] ", end="", flush=True)
            if want_fpl:
                try:
                    panel = fpl_map_native(lf, info, args.n, omega, tol=args.tol)
                    print(f"fpl {panel['wall']:.1f}s", end="", flush=True)
                except Exception as exc:
                    if args.solver == "fpl" and not args.force_fpl:
                        raise
                    print(f"fpl failed ({exc}); falling back to GWLENS ", end="",
                          flush=True)
                    panel = gwlens_map(gs, lens, args.n, omega)
            else:
                panel = gwlens_map(gs, lens, args.n, omega)
                print(f"GWLENS {panel['wall']:.1f}s", end="", flush=True)
            if not args.quick:
                val = panel_validation(info, panel, c200, omega, gs, lens, rng)
                panel["validation"] = val
                print("  " + " ".join(f"{k}={v:.1e}" for k, v in val.items()),
                      end="", flush=True)
            print(flush=True)
            results[(float(c200), float(omega))] = panel

    # summary
    lines = []
    lines.append("# Elliptical NFW + external shear -- prototype run\n")
    lines.append("Case: M200=1e13 Msun, z_l=0.5, z_s=1.5, "
                 "e1=0.2, e2=0, gamma=(0.03,0.01), "
                 f"y in [-{Y_EXTENT},{Y_EXTENT}] arcsec, n={args.n}.\n")
    lines.append("| c200 | omega | solver | wall [s] | inner_solves | "
                 "validation |")
    lines.append("|---|---|---|---|---|---|")
    for (c200, w), r in results.items():
        val = r.get("validation", {})
        vs = ", ".join(f"{k}={v:.1e}" for k, v in val.items()) or "-"
        lines.append(f"| {c200:g} | {w:g} | {r['method']} | {r['wall']:.2f} | "
                     f"{r.get('inner_solves', 0)} | {vs} |")
    n_fpl = sum(1 for r in results.values() if r["method"] == "fpl")
    n_gw = sum(1 for r in results.values() if r["method"] == "gwlens")
    tot_fpl = sum(r["wall"] for r in results.values() if r["method"] == "fpl")
    tot_gw = sum(r["wall"] for r in results.values() if r["method"] == "gwlens")
    lines.append("")
    if n_fpl == 0:
        lines.append(f"GWLENS total for all {n_gw} panels: {tot_gw:.1f} s "
                     "(wall includes the adaptive grid build).")
    elif n_gw == 0:
        lines.append(f"fpl total for all {n_fpl} panels: {tot_fpl:.1f} s.")
    else:
        lines.append(f"fpl total for the {n_fpl} fpl panels: {tot_fpl:.1f} s.  "
                     f"GWLENS total for the {n_gw} GWLENS-only panels: "
                     f"{tot_gw:.1f} s (wall includes the adaptive grid build).  "
                     f"Hybrid total: {tot_fpl + tot_gw:.1f} s.  This is not "
                     "like-for-like: fpl's exact RC engine stalls for the compact "
                     "NFW at omega=100 (c200=30, 60), so those two panels are "
                     "GWLENS and are labelled as such in the figure.")
    lines.append("")
    lines.append("Validation: `node_max_rel` compares each map at four of its "
                 "own nodes with a high-accuracy point solve of the same solver "
                 "(the maps are exact at their nodes); `fpl_vs_gwlens` compares "
                 "six random source positions between the two independent "
                 "solvers.  The dark interference nulls are narrower than the "
                 "pixel spacing, so a *value between nodes* (or a nearest-pixel "
                 "comparison) is not meaningful; at the node the two scalar "
                 "solvers agree to ~1e-9 even at the nulls.")
    lines.append("")
    lines.append("The exact reduction (elliptical NFW -> fpl's circular NFW + "
                 "diagonal macro via x = Se u) was also checked against GWLENS "
                 "directly at 5 sources x 3 frequencies: |dF|/|F| <= 6e-13.")
    lines.append("")
    lines.append("Reference claims: FIONA 130 s, GWLENS 6.2 s for this figure "
                 "(from the GWLENS authors; not reproduced here).  On this 6-core "
                 "box GWLENS measures ~0.4 s map + a few seconds of grid build at "
                 "omega=1,10, and ~11 s map + ~8 s build at omega=100, i.e. "
                 "18-21 s per omega=100 panel end-to-end.  A separate "
                 "`--solver gwlens` run over all nine panels measured 50.1 s "
                 "in total on this box.  The absolute numbers are "
                 "hardware-dependent; the ratio is what the authors quote.")
    with open(os.path.join(args.outdir, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # data (compact, opt-in: the figure and summary are the default products)
    if args.save_data:
        save = {"c200": np.array(args.c200), "omegas": np.array(args.omegas),
                "y": np.linspace(-Y_EXTENT, Y_EXTENT, args.n)}
        for (c200, w), r in results.items():
            key = f"c{c200:g}_w{w:g}"
            save[f"F_{key}"] = r["F"].astype(np.float32)
            save[f"solver_{key}"] = r["method"]
            save[f"wall_{key}"] = r["wall"]
            if r["method"] == "fpl":
                save[f"b0_{key}"] = r["b0"]
                save[f"b1_{key}"] = r["b1"]
                save[f"e_{key}"] = r["info"]["e"]
                save[f"phi_{key}"] = r["info"]["phi"]
                save[f"theta_{key}"] = r["info"]["psi_ang"]
        np.savez_compressed(os.path.join(args.outdir, "maps.npz"), **save)
        print(f"-> {os.path.join(args.outdir, 'maps.npz')}")

    if not args.no_plot:
        for scale in ("linear", "log"):
            path = make_figure(results, args.c200, args.omegas, args.outdir,
                               scale=scale)
            print(f"-> {path}")
    print(f"-> {os.path.join(args.outdir, 'summary.md')}")


if __name__ == "__main__":
    main()
