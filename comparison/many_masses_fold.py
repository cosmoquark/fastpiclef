"""P4.2 Phase 1: far-field folding prototype (go/no-go for the many-mass mode).

Plan of record: ~/.opencode/plan/p42-far-field-folding.md

Method (all in numpy, only the *solver* comes from the installed package):

* near/far split at R_cut around the fold centre c;
* far stars folded into beta_shift + shear + a global phase (their potential is
  expanded through quadratic order at c);
* the exact RC engine solves the small folded `LensField`;
* each far star's own micro-image (the saddle at ~m/R from it, amplitude
  ~m/R^2) is restored by corrected stationary phase;
* optional first-order smooth-remainder correction at the near images;
* a measured error budget (E_smooth + E_micro, x3 safety).

Subcommands:
    sanity   the plan's four sanity checks (must pass before any table)
    run      the case matrix; resumable, writes results/many_masses_fold.csv
    all      sanity + run

Run: `pixi run -e comparison python -m comparison.many_masses_fold [sanity|run|all]`
"""

import argparse
import csv
import itertools
import json
import math
import os
import time
from itertools import permutations

import numpy as np

import fastpiclef as fpl

from .sp_correction import _sum15

RESULTS = os.path.join(os.path.dirname(__file__), "results")
TAU = 2.0 * math.pi
CSV_NAME = "many_masses_fold.csv"
TRUTHS_NAME = "many_masses_fold_truths.json"


# --------------------------------------------------------------------------
# star fields and the model
# --------------------------------------------------------------------------

def star_field(seed, kappa_star, R_field, spectrum=None):
    """Uniform disc of stars; masses equal (or a power law rescaled to mean 1)."""
    rng = np.random.default_rng(seed)
    n = int(round(kappa_star * math.pi * R_field ** 2))
    r = R_field * np.sqrt(rng.random(n))
    t = TAU * rng.random(n)
    xy = np.column_stack([r * np.cos(t), r * np.sin(t)])
    if spectrum == "power":
        mmin, mmax = 0.1, 1.0
        u = rng.random(n)
        m = (1.0 / mmin - u * (1.0 / mmin - 1.0 / mmax)) ** (-1.0)
        m = m / m.mean()
    else:
        m = np.ones(n)
    return np.column_stack([xy, m])


def _fill_sym(tensor, order, values):
    """Fill a symmetric tensor of rank `order` from values[(n_zeros, n_ones)]."""
    for k in range(order + 1):
        l = order - k
        v = values[(k, l)]
        for idx in set(permutations([0] * k + [1] * l)):
            tensor[idx] = v


class Model:
    """A lens model in the original frame: point masses + (kappa, gamma1, gamma2)."""

    def __init__(self, stars, kappa=0.0, gamma1=0.0, gamma2=0.0, beta=(0.0, 0.0)):
        stars = np.atleast_2d(np.asarray(stars, float))
        if stars.ndim == 1:
            stars = stars[None, :]
        self.stars = stars
        self.kappa = float(kappa)
        self.gamma1 = float(gamma1)
        self.gamma2 = float(gamma2)
        self.beta = np.asarray(beta, float)

    # -- first derivatives -------------------------------------------------
    def grad_psi(self, x, mask=None, with_macro=True):
        s = self.stars if mask is None else self.stars[mask]
        dx = x[0] - s[:, 0]
        dy = x[1] - s[:, 1]
        r2 = dx * dx + dy * dy
        gx = np.sum(s[:, 2] * dx / r2)
        gy = np.sum(s[:, 2] * dy / r2)
        if with_macro:
            gx += (self.kappa + self.gamma1) * x[0] + self.gamma2 * x[1]
            gy += (self.kappa - self.gamma1) * x[1] + self.gamma2 * x[0]
        return np.array([gx, gy])

    def hess_psi(self, x, mask=None, with_macro=True):
        s = self.stars if mask is None else self.stars[mask]
        dx = x[0] - s[:, 0]
        dy = x[1] - s[:, 1]
        r2 = dx * dx + dy * dy
        r4 = r2 * r2
        m = s[:, 2]
        hxx = np.sum(m * (r2 - 2.0 * dx * dx) / r4)
        hxy = np.sum(-2.0 * m * dx * dy / r4)
        hyy = np.sum(m * (r2 - 2.0 * dy * dy) / r4)
        if with_macro:
            hxx += self.kappa + self.gamma1
            hxy += self.gamma2
            hyy += self.kappa - self.gamma1
        return np.array([[hxx, hxy], [hxy, hyy]])

    def psi(self, x, mask=None):
        s = self.stars if mask is None else self.stars[mask]
        dx = x[0] - s[:, 0]
        dy = x[1] - s[:, 1]
        val = np.sum(s[:, 2] * 0.5 * np.log(dx * dx + dy * dy))
        val += (0.5 * self.kappa * (x[0] ** 2 + x[1] ** 2)
                + 0.5 * self.gamma1 * (x[0] ** 2 - x[1] ** 2)
                + self.gamma2 * x[0] * x[1])
        return val

    # -- tensors of the phase ---------------------------------------------
    def phase_tensors(self, x):
        """Phase phi = |x-beta|^2/2 - psi: value, Hessian, phi3, phi4.

        The point masses are handled analytically through 4th order via
        d/dz on the analytic potential; the macro contributes to the
        Hessian only.
        """
        z = x[0] + 1j * x[1]
        R = z - (self.stars[:, 0] + 1j * self.stars[:, 1])
        m = self.stars[:, 2]
        inv = 1.0 / R
        # phi = |x-beta|^2/2 - psi;  derivatives of -sum psi_star at x:
        t2 = np.zeros((2, 2))
        t3 = np.zeros((2, 2, 2))
        t4 = np.zeros((2, 2, 2, 2))
        d2 = -inv * inv
        vals = {(k, 2 - k): -np.sum(m * np.real((1j ** (2 - k)) * d2)) for k in range(3)}
        _fill_sym(t2, 2, vals)
        d3 = 2.0 * inv ** 3
        vals = {(k, 3 - k): -np.sum(m * np.real((1j ** (3 - k)) * d3)) for k in range(4)}
        _fill_sym(t3, 3, vals)
        d4 = -6.0 * inv ** 4
        vals = {(k, 4 - k): -np.sum(m * np.real((1j ** (4 - k)) * d4)) for k in range(5)}
        _fill_sym(t4, 4, vals)
        H = np.eye(2) - self.hess_psi(x)
        ux = x[0] - self.beta[0]
        uy = x[1] - self.beta[1]
        phi = 0.5 * (ux * ux + uy * uy) - self.psi(x)
        return phi, H, t3, t4

    def image_contribution(self, x, omega):
        """Leading stationary-phase contribution to F, plus C and its parts."""
        phi, H, t3, t4 = self.phase_tensors(x)
        det = float(np.linalg.det(H))
        Hinv = np.linalg.inv(H)
        ev = np.linalg.eigvalsh(H)
        sigma = int(np.sum(np.sign(ev)))
        G0 = np.exp(1j * (omega * phi + math.pi / 4.0 * sigma)) / (1j * math.sqrt(abs(det)))
        C = (0.125 * np.einsum("ab,cd,abcd->", Hinv, Hinv, t4)
             - _sum15(t3, Hinv) / 72.0)
        return dict(x=np.asarray(x, float), phi=phi, H=H, det=det, sigma=sigma,
                    G0=G0, C=C, saddle=det < 0.0)


# --------------------------------------------------------------------------
# folding
# --------------------------------------------------------------------------

def fold(stars, kappa, gamma1, gamma2, beta, c, R_cut, near_extra=None):
    """Far-field fold at centre c with radius R_cut.

    `near_extra` is a list of catalogue indices force-kept in the near set
    (the promotion rule for far stars whose micro-image search flags).

    Returns the near stars, the folded model parameters, the global phase
    const, and the quad data needed for the remainder R(x).
    """
    c = np.asarray(c, float)
    d = stars[:, :2] - c
    rr = np.hypot(d[:, 0], d[:, 1])
    far_mask = rr >= R_cut
    if near_extra:
        far_mask[np.asarray(near_extra, int)] = False
    near = stars[~far_mask]
    far = stars[far_mask]
    zc = -(d[far_mask, 0] + 1j * d[far_mask, 1])       # c - z_j (complex)
    mf = far[:, 2]
    C0 = float(np.sum(mf * np.log(np.abs(zc))))
    B = [None] + [np.sum(mf * zc ** (-n)) for n in range(1, 5)]
    A1, A2 = B[1], -B[2] / 2.0
    g = np.array([A1.real, -A1.imag])
    g1f = 2.0 * A2.real
    g2f = -2.0 * A2.imag
    Gamma = np.array([[g1f, g2f], [g2f, -g1f]])
    beta_eff = np.asarray(beta, float) + g - Gamma @ c
    const = (0.5 * float(beta @ np.asarray(beta, float))
             - 0.5 * float(beta_eff @ beta_eff)
             - C0 + float(g @ c) - 0.5 * float(c @ (Gamma @ c)))
    return dict(near=near, far=far, c=c, C0=C0, A=[B[n] for n in range(1, 5)],
                g=g, gamma=(g1f, g2f), Gamma=Gamma, const=const, beta_eff=beta_eff,
                kappa=kappa, gamma1=gamma1, gamma2=gamma2)


def folded_model(fr):
    return Model(fr["near"], kappa=fr["kappa"],
                 gamma1=fr["gamma1"] + fr["gamma"][0],
                 gamma2=fr["gamma2"] + fr["gamma"][1], beta=fr["beta_eff"])


def remainder(fr, x):
    """true far-star potential minus its quadratic expansion at the fold centre."""
    far = fr["far"]
    z = x[0] + 1j * x[1]
    R = z - (far[:, 0] + 1j * far[:, 1])
    psiF = np.sum(far[:, 2] * np.log(np.abs(R)))
    u = np.asarray(x, float) - fr["c"]
    quad = fr["C0"] + float(fr["g"] @ u) + 0.5 * float(u @ (fr["Gamma"] @ u))
    return psiF - quad


def remainder_bound(fr):
    """Rigorous disc bound: |R| <= sum m rho^3/(3(1-rho)) (at the given x)."""
    return None  # evaluated per-point in the sanity checks


# --------------------------------------------------------------------------
# micro-images of the far stars
# --------------------------------------------------------------------------

def _newton_full(model, x0, beta, exclude=None, itmax=50, tol=1e-11):
    x = np.asarray(x0, float).copy()
    n = len(model.stars)
    for _ in range(itmax):
        mask = np.ones(n, bool)
        if exclude is not None:
            mask[exclude] = False
        F = x - model.grad_psi(x, mask=mask) - beta
        if np.hypot(*F) < tol:
            return x, True
        J = np.eye(2) - model.hess_psi(x, mask=mask)
        try:
            step = np.linalg.solve(J, F)
        except np.linalg.LinAlgError:
            return x, False
        x = x - step
        if not np.all(np.isfinite(x)) or np.hypot(*(x - x0)) > 50.0:
            return x, False
    F = x - model.grad_psi(x, mask=mask) - beta
    return x, np.hypot(*F) < 1e-9


BETA_MIN = 3.0     # far-star promotion threshold on |beta_loc| (see the single-star study)


def find_micro_images(model, fr, beta, R_search=2.0, R_keep=8.0, beta_min=BETA_MIN):
    """One saddle micro-image near each far star (all-frame Newton on the full
    model, star excluded from the local deflection for the start only).

    Every far star is searched; `beta_min` only *flags* weak stars (their
    stationary-phase treatment is the coarse one — see the single-star study).
    Ownership: a converged saddle counts for star j only if j is its nearest
    star (shared/global images are discarded).  Flags carry the *catalogue*
    index and are informational (no promotion: promoting a distant strong star
    drags the image set outside the far field's Taylor convergence radius).
    """
    out = []
    flags = []
    n = len(model.stars)
    xall = model.stars[:, :2]
    for j in range(len(fr["far"])):
        xj = fr["far"][j, :2]
        mj = fr["far"][j, 2]
        # locate the star's index in the full catalogue
        dist = np.hypot(model.stars[:, 0] - xj[0], model.stars[:, 1] - xj[1])
        idx = int(np.argmin(dist))
        if dist[idx] > 1e-9:
            raise RuntimeError("far star not found in the model")
        mask = np.ones(n, bool)
        mask[idx] = False
        a_rest = model.grad_psi(xj, mask=mask)
        h_rest = model.hess_psi(xj, mask=mask)
        A = np.eye(2) - h_rest
        beta_loc = np.asarray(beta, float) - xj + a_rest
        nv = float(np.hypot(*beta_loc))
        if nv < 1e-9:
            flags.append((idx, "degenerate"))
            continue
        weak = nv < beta_min
        d0 = abs(mj) / nv
        v = A @ beta_loc
        nvv = float(np.hypot(*v))
        vhat = v / nvv if nvv > 1e-12 else beta_loc / nv
        sols = []
        starts = [xj - d0 * vhat, xj + d0 * vhat]
        for s in range(8):
            ang = s * math.pi / 4.0
            starts.append(xj + d0 * np.array([math.cos(ang), math.sin(ang)]))
            starts.append(xj + 0.5 * d0 * np.array([math.cos(ang), math.sin(ang)]))
        beyond = False
        weak_own = False
        for x0 in starts:
            # Newton on the *full* model (the micro-image is a stationary
            # point of the true phase; the exclusion above is only for the
            # local-deflection start).
            xs, ok = _newton_full(model, x0, beta)
            if not ok:
                continue
            r = float(np.hypot(*(xs - xj)))
            if r > R_keep:
                continue
            H = np.eye(2) - model.hess_psi(xs)
            det = float(np.linalg.det(H))
            if det >= 0.0:
                continue
            if any(np.hypot(*(xs - s)) < 1e-7 for s in sols):
                continue
            # ownership: nearest star must be j, with a margin
            da = np.hypot(xall[:, 0] - xs[0], xall[:, 1] - xs[1])
            order = np.argsort(da)
            if order[0] != idx:
                continue
            d2 = da[order[1]] if n > 1 else np.inf
            if r > 0.5 * d2:
                weak_own = True
            if r > R_search:
                beyond = True
            sols.append(xs)
        if len(sols) == 0:
            flags.append((idx, "no_saddle"))
        elif len(sols) > 1:
            flags.append((idx, f"multi={len(sols)}"))
        elif beyond:
            flags.append((idx, "saddle_beyond_search"))
        elif weak:
            flags.append((idx, f"weak:{nv:.2f}"))
        elif weak_own:
            flags.append((idx, "weak_ownership"))
        for xs in sols:
            out.append(dict(star=idx, x=xs, r=float(np.hypot(*(xs - xj))),
                            beta_loc=nv, weak=weak))
    return out, flags


# --------------------------------------------------------------------------
# the folded estimate + budget
# --------------------------------------------------------------------------

def folded_field(fr):
    comps = [fpl.PointComp(float(x), float(y), float(m)) for (x, y, m) in fr["near"]]
    return fpl.LensField(comps, beta=(float(fr["beta_eff"][0]), float(fr["beta_eff"][1])),
                         kappa=fr["kappa"], gamma1=fr["gamma1"] + fr["gamma"][0],
                         gamma2=fr["gamma2"] + fr["gamma"][1])


def estimate(model_true, fr, omega, F_fold, lf_fold, beta_min=BETA_MIN):
    """All variants + the budget for one (fold, omega).

    Budget = 3 x (phase-remainder at the folded images + micro-image
    approximation terms + missing-saddle bounds).  The first-order correction
    is applied only where omega * |R| <= 1 (validity gate).
    """
    lf_mod = folded_model(fr)
    const_phase = np.exp(1j * omega * fr["const"])

    # images of the folded model: the phase remainder there is the first error
    # channel (the folded saddle positions are displaced, so no C-difference
    # term is meaningful — measured 2026-10-13)
    ims = lf_fold.images()
    E_shift = 0.0
    dF1 = 0.0
    max_wR = 0.0
    for (xx, yy) in ims:
        x = np.array([xx, yy])
        fold_img = lf_mod.image_contribution(x, omega)
        G = fold_img["G0"]
        R = remainder(fr, x)
        max_wR = max(max_wR, omega * abs(R))
        E_shift += abs(G) * omega * abs(R)
        dF1 += -1j * omega * G * R

    micro, flags = find_micro_images(model_true, fr, model_true.beta, beta_min=beta_min)
    sumG = 0j
    E_micro = 0.0
    for mm in micro:
        img = model_true.image_contribution(mm["x"], omega)
        g = img["G0"] * (1.0 + img["C"] / (1j * omega))
        sumG += g
        E_micro += abs(img["G0"]) * abs(img["C"]) / omega

    # far stars without a found saddle: bound their missing image by m/beta_loc^2
    found = {mm["star"] for mm in micro}
    E_missing = 0.0
    for j in range(len(fr["far"])):
        xj = fr["far"][j, :2]
        dist = np.hypot(model_true.stars[:, 0] - xj[0], model_true.stars[:, 1] - xj[1])
        idx = int(np.argmin(dist))
        if idx in found:
            continue
        mask = np.ones(len(model_true.stars), bool)
        mask[idx] = False
        nv = float(np.hypot(*(model_true.beta - xj + model_true.grad_psi(xj, mask=mask))))
        E_missing += fr["far"][j, 2] / max(nv, 1e-9) ** 2

    budget = 3.0 * (E_shift + E_micro + E_missing)
    F_plain = const_phase * F_fold
    F_micro = F_plain + sumG
    F_first = (const_phase * (F_fold + dF1) + sumG) if max_wR <= 1.0 else F_micro
    return dict(F_plain=F_plain, F_micro=F_micro, F_first=F_first,
                E_smooth=E_shift, E_micro=E_micro, E_missing=E_missing,
                budget=budget, max_wR=max_wR,
                n_micro=len(micro), n_flags=len(flags), flags=flags,
                n_images=len(ims), micro=micro)


def _fmt(v):
    return f"{v:.17e}"


def _c(re_im):
    c = re_im
    return _fmt(c.real), _fmt(c.imag)


# --------------------------------------------------------------------------
# sanity checks
# --------------------------------------------------------------------------

def sanity(n=30, omegas=(3.0,), seed=20261013):
    ok = True
    kappa_star = 100.0 / (math.pi * 100.0)
    R_field = math.sqrt(n / (math.pi * kappa_star))
    print(f"== sanity 1: K = N reproduces the unfolded value (N={n}, R_field={R_field:.2f}) ==")
    stars = star_field(seed, kappa_star, R_field)
    beta = (0.05, 0.0)
    kappa, g1 = 0.55, 0.05
    fr = fold(stars, kappa, g1, 0.0, beta, (0.0, 0.0), 2.0 * R_field)
    assert len(fr["far"]) == 0
    assert abs(fr["const"]) == 0.0
    for w in omegas:
        truth = fpl.LensField([fpl.PointComp(x, y, m) for (x, y, m) in stars],
                              beta=beta, kappa=kappa, gamma1=g1).solve(omega=w).amplification
        ff = folded_field(fr).solve(omega=w).amplification * np.exp(1j * w * fr["const"])
        d = abs(ff - truth) / abs(truth)
        print(f"   w={w}: |F_fold - truth|/|truth| = {d:.2e}")
        ok &= d < 1e-12

    print("== sanity 2: the folding algebra (phase identity in the disc) ==")
    for R_cut in (4.0, 8.0):
        fr = fold(stars, kappa, g1, 0.0, beta, (0.0, 0.0), R_cut)
        m_true = Model(stars, kappa, g1, 0.0, beta)
        m_fold = folded_model(fr)
        rng = np.random.default_rng(1)
        worst = 0.0
        worst_bound = 0.0
        worst_R = 0.0
        for _ in range(200):
            r = 0.5 * R_cut * math.sqrt(rng.random())
            th = TAU * rng.random()
            x = np.array([r * math.cos(th), r * math.sin(th)])
            lhs = m_true.phase_tensors(x)[0]
            R = remainder(fr, x)
            rhs = m_fold.phase_tensors(x)[0] + fr["const"] - R
            worst = max(worst, abs(lhs - rhs))
            worst_R = max(worst_R, abs(R))
            far = fr["far"]
            d = np.hypot(fr["c"][0] - far[:, 0], fr["c"][1] - far[:, 1])
            rho = r / d                     # the Taylor convergence ratio
            if np.all(rho < 0.95):
                worst_bound = max(worst_bound,
                                  float(np.sum(far[:, 2] * rho ** 3 / (3 * (1 - rho)))))
        print(f"   R_cut={R_cut}: |phi_true - (phi_fold + const - R)| <= {worst:.2e}  "
              f"max |R| {worst_R:.2e} <= disc bound {worst_bound:.2e}")
        ok &= worst < 1e-12 and worst_R <= worst_bound * (1 + 1e-12)

    print("== sanity 3: micro finder vs the closed form (single star) ==")
    y = 0.5
    star = np.array([[0.0, 0.0, 1.0]])
    m1 = Model(star, 0.0, 0.0, 0.0, (y, 0.0))
    # fold centre away from the star so the star is in the far set
    fr = fold(star, 0.0, 0.0, 0.0, (y, 0.0), (0.0, 5.0), 1.0)
    assert len(fr["far"]) == 1
    micro, flags = find_micro_images(m1, fr, (y, 0.0), beta_min=0.0)
    xs_exact = 0.5 * (y - math.sqrt(y * y + 4.0))
    print(f"   found {len(micro)} micro image(s), flags {flags}")
    if len(micro) == 1:
        xs = micro[0]["x"][0]
        print(f"   x- Newton {xs:.12f} vs closed form {xs_exact:.12f}  diff {abs(xs - xs_exact):.2e}")
        ok &= abs(xs - xs_exact) < 1e-10
    else:
        ok = False

    print("== sanity 4: single far star, full pipeline vs truth ==")
    star = np.array([[0.0, 50.0, 1.0]])
    m1 = Model(star, 0.0, 0.0, 0.0, (0.0, 0.0))
    fr = fold(star, 0.0, 0.0, 0.0, (0.0, 0.0), (0.0, 0.0), 1.0)
    lf_fold = folded_field(fr)
    for w in (3.0, 10.0, 30.0):
        truth = fpl.LensField([fpl.PointComp(0.0, 50.0, 1.0)], beta=(0.0, 0.0)).solve(omega=w)
        Ff = lf_fold.solve(omega=w).amplification
        est = estimate(m1, fr, w, Ff, lf_fold)
        d_plain = abs(est["F_plain"] - truth.amplification) / abs(truth.amplification)
        d_micro = abs(est["F_micro"] - truth.amplification) / abs(truth.amplification)
        d_first = abs(est["F_first"] - truth.amplification) / abs(truth.amplification)
        # the flipped-const control
        flipped = np.exp(-1j * w * fr["const"]) * Ff + sum(
            (lambda img: img["G0"] * (1 + img["C"] / (1j * w)))(
                m1.image_contribution(mm["x"], w)) for mm in est["micro"])
        d_flip = abs(flipped - truth.amplification) / abs(truth.amplification)
        print(f"   w={w}: plain {d_plain:.2e}  micro {d_micro:.2e}  first {d_first:.2e}  "
              f"[flipped const: {d_flip:.2e}]  budget {est['budget']:.2e} "
              f"cover {est['budget'] / max(d_micro, 1e-300):.2f}x")
        ok &= d_micro < est["budget"]

    print("== sanity:", "PASS" if ok else "FAIL")
    return ok


# --------------------------------------------------------------------------
# the run matrix
# --------------------------------------------------------------------------

def case_list(mode="main"):
    """The Phase-1 test matrix (plan §3).  mode: main | small | big | maps.

    Two density families: *dense* (kappa_star ~ 0.3183: ~1 star per Einstein
    area — the hard, strongly interacting case) and *diluted*
    (R_field >> R_cut: the many-mass science geometry, where the far field is
    a genuinely distant tail).  The single-star study (2026-10-13) showed the
    far-star micro-image error falls like beta_loc^-1.8 (beta_loc ~ 0.45 R):
    ~7e-3 at beta_loc = 1.2, ~1.6e-4 at 4, ~4e-5 at 8 (omega = 3).
    """
    fracs = (0.27, 0.37, 0.46, 0.55, 0.73, 0.91)
    out = []
    if mode == "main":
        # dense, N = 30 (hard case; every far star is strong -> promotion)
        n, rf = 30, 5.477
        ks = n / (math.pi * rf ** 2)
        out.append(dict(name="min5_a", macro="min", kappa=0.55, gamma1=0.05,
                        omegas=(0.3, 1.0, 3.0), seed=20261013, kappa_star=ks,
                        R_field=rf, beta=(0.05, 0.0)))
        out.append(dict(name="sad5_a", macro="sad", kappa=0.95, gamma1=0.45,
                        omegas=(1.0, 3.0), seed=20261013, kappa_star=ks,
                        R_field=rf, beta=(0.05, 0.0)))
        # diluted, N = 30 in R_field = 15 (the fold's natural geometry)
        n, rf = 30, 15.0
        ks = n / (math.pi * rf ** 2)
        out.append(dict(name="min5_dil", macro="min", kappa=0.55, gamma1=0.05,
                        omegas=(1.0, 3.0), seed=20261013, kappa_star=ks,
                        R_field=rf, beta=(0.05, 0.0)))
        out.append(dict(name="sad5_dil", macro="sad", kappa=0.95, gamma1=0.45,
                        omegas=(3.0,), seed=20261013, kappa_star=ks,
                        R_field=rf, beta=(0.05, 0.0)))
        # min20_a (dense mu~20, kappa=0.775) was DROPPED 2026-10-13: its exact
        # truth at omega=3 ran >3.5 h wall (vs ~150 s for the kappa=0.55 field
        # -- a cost cliff, plausibly a near-pinch/near-critical configuration;
        # recorded).  The dense verdict is settled by min5_a/min5_b/min5_c/
        # sad5_a at 0.15-1.4, and each row on that field would be similarly
        # expensive.
        n, rf = 30, 5.477
        ks = n / (math.pi * rf ** 2)
        out.append(dict(name="min5_b", macro="min", kappa=0.55, gamma1=0.05,
                        omegas=(3.0,), seed=777, kappa_star=ks, R_field=rf,
                        beta=(0.05, 0.0), spectrum="power"))
        out.append(dict(name="min5_c", macro="min", kappa=0.55, gamma1=0.05,
                        omegas=(3.0,), seed=4242, kappa_star=ks, R_field=rf,
                        beta=(-0.08, 0.05)))
    elif mode == "small":
        n, rf = 10, 3.162
        ks = n / (math.pi * rf ** 2)
        out.append(dict(name="min5_small", macro="min", kappa=0.55, gamma1=0.05,
                        omegas=(10.0, 30.0), seed=20261013, kappa_star=ks,
                        R_field=rf, beta=(0.05, 0.0)))
    elif mode == "big":
        # only the *diluted* N=100 case: the dense N=100 is settled by the N=30
        # dense verdict (2026-10-13) and its rows would cost ~1.5 h to add nothing
        out.append(dict(name="min5_n100d", macro="min", kappa=0.55, gamma1=0.05,
                        omegas=(3.0,), seed=20261014, kappa_star=100 / (math.pi * 900),
                        R_field=30.0, beta=(0.05, 0.0)))
    elif mode == "maps":
        n, rf = 30, 15.0
        ks = n / (math.pi * rf ** 2)
        out.append(dict(name="map_min5_dil", macro="min", kappa=0.55, gamma1=0.05,
                        omegas=(3.0,), seed=20261013, kappa_star=ks,
                        R_field=rf, beta=(0.05, 0.0), maps=True))
    defaults = {"main": (1.0, 2.0, 3.0, 5.0, 8.0, 12.0),
                "small": (1.0, 2.0),
                "big": (1.0, 2.0, 3.0, 5.0, 8.0),
                "maps": (1.0, 3.0)}
    for c in out:
        c.setdefault("rcuts", defaults[mode])
    return out


def load_json(path):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def save_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1)


def truth_for(case, stars, omega, cache, cache_path):
    key = f"{case['name']}|{omega}"
    if key in cache:
        return cache[key]
    from fastpiclef import PointComp, LensField
    comps = [PointComp(float(x), float(y), float(m)) for (x, y, m) in stars]
    lf = LensField(comps, beta=tuple(case["beta"]), kappa=case["kappa"], gamma1=case["gamma1"])
    load = os.getloadavg()[0]
    t0 = time.time()
    out = lf.solve(omega=omega)
    wall = time.time() - t0
    rec = dict(re=float(out.amplification.real), im=float(out.amplification.imag),
               error=float(out.error), steps=int(out.steps), wall=wall, load=load)
    print(f"   [truth {key}] {wall:.1f} s (load {load:.2f}), "
          f"F = {out.amplification:.12g}, err {out.error:.1e}, steps {out.steps}")
    cache[key] = rec
    save_json(cache_path, cache)
    return rec


def run(args):
    os.makedirs(RESULTS, exist_ok=True)
    csv_path = os.path.join(RESULTS, CSV_NAME)
    truth_path = os.path.join(RESULTS, TRUTHS_NAME)
    cache = load_json(truth_path)
    existing = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                existing.add((r["case"], r["omega"], r["R_cut"]))
    fields = ["case", "macro", "kappa", "gamma1", "omega", "seed", "spectrum",
              "kappa_star", "R_field", "N", "R_cut", "beta_min", "K", "const_re",
              "const_im", "truth_re", "truth_im", "truth_err", "truth_steps",
              "truth_wall", "ff_re", "ff_im", "ff_steps",
              "err_plain", "err_micro", "err_first",
              "E_smooth", "E_micro", "E_missing", "budget", "cover_best",
              "max_wR", "indicator", "n_micro", "n_flags", "promoted", "flags_detail",
              "n_images", "wall_fold", "load"]

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as fh:
            csv.writer(fh).writerow(fields)

    for case in case_list(args.mode):
        if args.cases and case["name"] not in args.cases:
            continue
        if case.get("maps"):
            continue
        stars = star_field(case["seed"], case["kappa_star"], case["R_field"],
                           spectrum=case.get("spectrum"))
        np.savez(os.path.join(RESULTS, f"many_masses_fold_stars_{case['name']}.npz"),
                 stars=stars)
        print(f"== case {case['name']} ({case['macro']}, kappa={case['kappa']}, "
              f"N={len(stars)}, sigma={case.get('spectrum')}) ==")
        for omega in case["omegas"]:
            if args.omegas and omega not in args.omegas:
                continue
            rec = truth_for(case, stars, omega, cache, truth_path)
            truth = complex(rec["re"], rec["im"])
            beta_min = case.get("beta_min", BETA_MIN)
            prev_F = None
            for rcut in (getattr(args, "rcuts", None) or case["rcuts"]):
                key = (case["name"], _fmt(omega), _fmt(rcut))
                if key in existing and not args.force:
                    continue
                est, out_f, fr, wall = _fold_row(case, stars, omega, rcut, beta_min)
                t = abs(truth)
                e_plain = abs(est["F_plain"] - truth) / t
                e_micro = abs(est["F_micro"] - truth) / t
                e_first = abs(est["F_first"] - truth) / t

                # indicator: this split's estimate vs the previous (coarser) split
                indicator = (abs(est["F_first"] - prev_F) / t
                             if prev_F is not None else float("nan"))
                prev_F = est["F_first"]

                row = dict(
                    case=case["name"], macro=case["macro"], kappa=_fmt(case["kappa"]),
                    gamma1=_fmt(case["gamma1"]), omega=_fmt(omega), seed=case["seed"],
                    spectrum=case.get("spectrum") or "equal",
                    kappa_star=_fmt(case["kappa_star"]), R_field=_fmt(case["R_field"]),
                    N=len(stars), R_cut=_fmt(rcut), beta_min=_fmt(beta_min),
                    K=len(fr["near"]),
                    const_re=_fmt(fr["const"]), const_im=_fmt(0.0),
                    truth_re=_fmt(truth.real), truth_im=_fmt(truth.imag),
                    truth_err=_fmt(rec["error"]), truth_steps=rec["steps"],
                    truth_wall=_fmt(rec["wall"]),
                    ff_re=_fmt(out_f.amplification.real), ff_im=_fmt(out_f.amplification.imag),
                    ff_steps=out_f.steps,
                    err_plain=_fmt(e_plain), err_micro=_fmt(e_micro), err_first=_fmt(e_first),
                    E_smooth=_fmt(est["E_smooth"]), E_micro=_fmt(est["E_micro"]),
                    E_missing=_fmt(est["E_missing"]), budget=_fmt(est["budget"]),
                    cover_best=_fmt(est["budget"] / max(min(e_micro, e_first), 1e-300)),
                    max_wR=_fmt(est["max_wR"]), indicator=_fmt(indicator),
                    n_micro=est["n_micro"],
                    n_flags=est["n_flags"], promoted=est.get("promoted", 0),
                    flags_detail=";".join(f"{i}:{w}" for (i, w) in est["flags"]),
                    n_images=est["n_images"],
                    wall_fold=_fmt(wall), load=_fmt(os.getloadavg()[0]))
                with open(csv_path, "a", newline="") as fh:
                    csv.writer(fh).writerow([row[f] for f in fields])
                print(f"   w={omega:6.3g} R_cut={rcut:5.2f} K={len(fr['near']):4d}: "
                      f"plain {e_plain:.2e}  micro {e_micro:.2e}  first {e_first:.2e}  "
                      f"budget {est['budget']:.2e}  cover {est['budget'] / max(min(e_micro, e_first), 1e-300):5.2f}x  "
                      f"ind {indicator:.1e}  flags {est['n_flags']} "
                      f"(promoted {est.get('promoted', 0)}) [{wall:.1f} s]")
    print(f"wrote {csv_path}")
    if args.mode == "maps":
        run_maps(args, cache, truth_path)


def _fold_row(case, stars, omega, rcut, beta_min, beta=None):
    """One fold + folded solve + estimate (compact near split, no promotion).

    Returns (est, out_f, fr, wall).  The weak-star and ownership flags are
    informational; the budget carries the micro-image approximation terms.
    """
    beta = case["beta"] if beta is None else beta
    fr = fold(stars, case["kappa"], case["gamma1"], 0.0, beta, (0.0, 0.0), rcut)
    lf_fold = folded_field(fr)
    t0 = time.time()
    out_f = lf_fold.solve(omega=omega)
    wall = time.time() - t0
    model_true = Model(stars, case["kappa"], case["gamma1"], 0.0, beta)
    est = estimate(model_true, fr, omega, out_f.amplification, lf_fold,
                   beta_min=beta_min)
    est["promoted"] = 0
    return est, out_f, fr, wall


def run_maps(args, cache, truth_path):
    """Uniform and jittered 3x3 source grids: the fold (with per-point
    promotion) vs the exact truth map."""
    path = os.path.join(RESULTS, "many_masses_fold_maps.csv")
    fields = ["case", "grid", "i", "j", "beta0", "beta1", "omega", "beta_min", "R_cut", "K",
              "truth_re", "truth_im", "err_plain", "err_micro", "err_first",
              "E_smooth", "E_micro", "budget", "cover_best", "n_micro", "n_flags",
              "promoted", "wall", "load"]
    existing = set()
    if not os.path.exists(path):
        with open(path, "w", newline="") as fh:
            csv.writer(fh).writerow(fields)
    else:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                existing.add((r["case"], r["grid"], r["i"], r["j"], r["R_cut"]))
    for case in [c for c in case_list("maps")
                 if not args.cases or c["name"] in args.cases]:
        stars = star_field(case["seed"], case["kappa_star"], case["R_field"])
        omega = case["omegas"][0]
        b0 = np.asarray(case["beta"], float)
        step = 0.06
        off = np.array([-step, 0.0, step])
        grids = {}
        b0s, b1s, tags = [], [], []
        for i, dx in enumerate(off):
            for j, dy in enumerate(off):
                b0s.append(b0[0] + dx); b1s.append(b0[1] + dy); tags.append((i, j))
        grids["uniform"] = (list(b0s), list(b1s), tags)
        rng = np.random.default_rng(99)
        jit = 0.02 * (rng.random((len(b0s), 2)) - 0.5)
        grids["jittered"] = ([x + d[0] for x, d in zip(b0s, jit)],
                             [y + d[1] for y, d in zip(b1s, jit)], tags)
        for grid, (b0s_g, b1s_g, tags_g) in grids.items():
            # truth map (batched engine)
            key = f"map|{case['name']}|{grid}|{omega}"
            if key not in cache:
                comps = [fpl.PointComp(float(x), float(y), float(m)) for (x, y, m) in stars]
                lf = fpl.LensField(comps, beta=tuple(case["beta"]),
                                   kappa=case["kappa"], gamma1=case["gamma1"])
                load = os.getloadavg()[0]
                t0 = time.time()
                out = lf.map(b0s_g, b1s_g, omega)
                wall = time.time() - t0
                cache[key] = dict(b0s=b0s_g, b1s=b1s_g,
                                  re=[[v.real for v in row] for row in out.amplification],
                                  im=[[v.imag for v in row] for row in out.amplification],
                                  wall=wall, load=load, steps=out.inner_solves)
                save_json(truth_path, cache)
                print(f"   [truth map {key}] {wall:.1f} s (load {load:.2f}), "
                      f"steps {out.inner_solves}")
            truth = cache[key]
            for rcut in case["rcuts"]:
                for k, ((i, j), x0, x1) in enumerate(zip(tags_g, b0s_g, b1s_g)):
                    if (case["name"], grid, str(i), str(j), _fmt(rcut)) in existing:
                        continue
                    est, out_f, frk, wall = _fold_row(case, stars, omega,
                                                      rcut, case.get("beta_min", BETA_MIN),
                                                      beta=(x0, x1))
                    truth_c = complex(truth["re"][i][j], truth["im"][i][j])
                    t = abs(truth_c)
                    e_plain = abs(est["F_plain"] - truth_c) / t
                    e_micro = abs(est["F_micro"] - truth_c) / t
                    e_first = abs(est["F_first"] - truth_c) / t
                    row = dict(case=case["name"], grid=grid, i=i, j=j,
                               beta0=_fmt(x0), beta1=_fmt(x1), omega=_fmt(omega),
                               beta_min=_fmt(case.get("beta_min", BETA_MIN)),
                               R_cut=_fmt(rcut), K=len(frk["near"]),
                               truth_re=_fmt(truth_c.real), truth_im=_fmt(truth_c.imag),
                               err_plain=_fmt(e_plain), err_micro=_fmt(e_micro),
                               err_first=_fmt(e_first), E_smooth=_fmt(est["E_smooth"]),
                               E_micro=_fmt(est["E_micro"]), budget=_fmt(est["budget"]),
                               cover_best=_fmt(est["budget"] / max(min(e_micro, e_first), 1e-300)),
                               n_micro=est["n_micro"], n_flags=est["n_flags"],
                               promoted=est.get("promoted", 0),
                               wall=_fmt(wall), load=_fmt(os.getloadavg()[0]))
                    with open(path, "a", newline="") as fh:
                        csv.writer(fh).writerow([row[f] for f in fields])
                    print(f"   {grid:8s} ({i},{j}) w={omega:5.3g} R_cut={rcut:5.2f} "
                          f"K={len(frk['near']):3d}: plain {e_plain:.2e} micro {e_micro:.2e} "
                          f"first {e_first:.2e} budget {est['budget']:.2e} "
                          f"flags {est['n_flags']} [{wall:.1f} s]")
    print(f"wrote {path}")


# --------------------------------------------------------------------------

def report():
    """Read the two CSVs and write results/many_masses_fold.md."""
    import collections
    csv_path = os.path.join(RESULTS, CSV_NAME)
    maps_path = os.path.join(RESULTS, "many_masses_fold_maps.csv")
    out_path = os.path.join(RESULTS, "many_masses_fold.md")
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    lines = ["# P4.2 far-field folding — Phase 1 prototype results",
             "",
             "Prototype: `comparison/many_masses_fold.py` (plan: "
             "`~/.opencode/plan/p42-far-field-folding.md`).  Truth = the exact RC on all N "
             "stars (the same engine, full field).  Variants: `plain` = folded only; "
             "`micro` = folded + restored far-star micro-images; `first` = + the first-order "
             "smooth-remainder correction at the near images.  Errors are relative to "
             "|F_truth|; `budget` = 3x(E_smooth + E_micro) (raw estimate either side).",
             ""]
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["case"], r["omega"])].append(r)

    # gate summary: best error per (case, omega), its K, and the coverage
    lines.append("## summary (gate view)")
    lines.append("")
    lines.append("| case | omega | best err | variant | K | cover at best | min cover | "
                 "max wR |")
    lines.append("|------|-------|----------|---------|---|---------------|-----------|"
                 "--------|")
    for (case, omega) in sorted(groups, key=lambda k: (k[0], float(k[1]))):
        rs = sorted(groups[(case, omega)], key=lambda r: int(r["K"]))
        best = None
        for r in rs:
            for v in ("err_micro", "err_first", "err_plain"):
                e = float(r[v])
                if best is None or e < best[0]:
                    best = (e, v, r)
        min_cover = min(float(r["cover_best"]) for r in rs)
        max_wr = max(float(r["max_wR"]) for r in rs)
        lines.append(f"| {case} | {omega} | {best[0]:.2e} | {best[1]} | {best[2]['K']} | "
                     f"{float(best[2]['cover_best']):.1f}x | {min_cover:.1f}x | "
                     f"{max_wr:.2e} |")
    lines.append("")
    for (case, omega) in sorted(groups, key=lambda k: (k[0], float(k[1]))):
        rs = sorted(groups[(case, omega)], key=lambda r: int(r["K"]))
        r0 = rs[0]
        lines.append(f"## {case}, omega = {omega} (N = {r0['N']}, macro {r0['macro']}, "
                     f"kappa = {float(r0['kappa']):.3f}, gamma1 = {float(r0['gamma1']):.3f}, "
                     f"kappa_star = {float(r0['kappa_star']):.4f}"
                     + (f", spectrum = {r0['spectrum']}" if r0["spectrum"] != "equal" else "")
                     + ")")
        lines.append("")
        lines.append("| K | R_cut | err plain | err micro | err first | budget | cover "
                     "| E_shift | E_micro | E_missing | max wR | indicator | n_micro "
                     "| flags | fold wall [s] |")
        lines.append("|---|-------|-----------|-----------|-----------|--------|-------|"
                     "---------|---------|-----------|--------|-----------|---------|-------|"
                     "----------------|")
        for r in rs:
            lines.append(
                f"| {r['K']} | {float(r['R_cut']):.2f} | {float(r['err_plain']):.2e} | "
                f"{float(r['err_micro']):.2e} | {float(r['err_first']):.2e} | "
                f"{float(r['budget']):.2e} | {float(r['cover_best']):.1f}x | "
                f"{float(r['E_smooth']):.2e} | {float(r['E_micro']):.2e} | "
                f"{float(r['E_missing']):.2e} | {float(r['max_wR']):.2e} | "
                f"{float(r['indicator']):.1e} | {r['n_micro']} | {r['n_flags']} | "
                f"{float(r['wall_fold']):.1f} |")
        lines.append("")
    if os.path.exists(maps_path):
        with open(maps_path, newline="") as fh:
            mrows = list(csv.DictReader(fh))
        if mrows:
            lines.append("## source maps (uniform / jittered 3x3)")
            lines.append("")
            lines.append("| grid | i | j | R_cut | K | err micro | err first | budget | "
                         "cover | flags |")
            lines.append("|------|---|---|-------|---|-----------|-----------|--------|"
                         "-------|-------|")
            for r in mrows:
                lines.append(
                    f"| {r['grid']} | {r['i']} | {r['j']} | {float(r['R_cut']):.2f} | "
                    f"{r['K']} | {float(r['err_micro']):.2e} | {float(r['err_first']):.2e} | "
                    f"{float(r['budget']):.2e} | {float(r['cover_best']):.1f}x | "
                    f"{r['n_flags']} |")
            lines.append("")
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["sanity", "run", "all", "report"], default="all",
                    nargs="?")
    ap.add_argument("--mode", choices=["main", "small", "big", "maps"], default="main")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--omegas", nargs="*", type=float, default=None)
    ap.add_argument("--betamins", nargs="*", type=float, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.command in ("sanity", "all"):
        if not sanity():
            raise SystemExit("sanity checks failed")
    if args.command in ("run", "all"):
        run(args)
    if args.command == "report":
        report()


if __name__ == "__main__":
    main()
