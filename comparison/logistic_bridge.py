"""P5: the logistic-bridging comparator -- the heuristic competitor.

Paul's end-stage item (Journal 2026-05-24; decided 2026-10-13): instead of
evaluating the Fresnel integral, a cheap empirical implementation bridges the
two known regimes with a logistic-like function of log omega, and we quantify
how close such a heuristic can get to the exact solver.

Variants
--------
V1 (complex)   F_h = 1 + s(omega) (F_geo(omega) - 1)
V2 (amplitude) |F_h| = 1 + s(omega) (|F_geo(omega)| - 1), arg F_h = arg F_geo

with the logistic window ``s(omega) = 1 / (1 + (omega_c / omega)^p)``:
``s -> 0`` (no lensing, F -> 1) at low omega, ``s -> 1`` (geometric optics)
at high omega.  ``F_geo`` is the package's stationary-phase sum (min/saddle/
max Morse phases 0 / -pi/2 / -pi, same convention as ``solve``).

Tiers
-----
tuned : ``(omega_c, p)`` fitted per case, minimising the max relative error
        over omega in [0.3, 30] (the maximally fair heuristic -- it sees the
        exact answer when fitting);
fixed : one ``(omega_c, p)`` for the three primary cases (minimax over
        cases: no per-case knowledge at all).

Cases
-----
P1  isolated point mass, y = 1.
M1  quadrupole macro (kappa = 0.3, gamma1 = 0.15) + central point mass
    (m = 0.2), beta = (0.4, 0.1) -- the substructure configuration, chosen
    off-caustic so the anchor itself converges: rel_g = 1.1e-2 at omega = 30
    (corrected 2.2e-3), so the bridge's transition-band error is not
    dominated by the ingredient.
C1  the caustic family: the point mass at y = 0, 0.02, 0.05, 0.1, 0.2.  At
    y = 0 there is no discrete geometric sum: the image set is the whole
    Einstein ring (a degenerate image), and since the finder fix of
    2026-10-17 the package *reports* that (`geometric` raises) instead of
    returning the old silent empty list (F_geo = 0).  The case is recorded
    with F_geo = NaN and carries no fits; the exact |F| grows like
    sqrt(omega) (the Einstein-ring stationary *manifold*) -- a logistic
    blend between 1 and a finite anchor cannot represent that.

Reference: the exact RC engine via the batched ``sweep``.  Referees: the
Nakamura-Deguchi closed form (moduli, machine-exact for the point-mass cases
at every tabulated z) and a second contour margin (2.0 vs 1.5) for M1.

Data: ``results/logistic_bridge{,.md,.png,_fits.csv,_checks.csv}``.
Subcommands: run | report | all (default).
"""
import argparse
import csv
import os

import numpy as np

import fastpiclef as fpl

RESULTS = os.path.join(os.path.dirname(__file__), "results")
CSV_NAME = "logistic_bridge.csv"
FITS_NAME = "logistic_bridge_fits.csv"
CHECKS_NAME = "logistic_bridge_checks.csv"
MD_NAME = "logistic_bridge.md"
PNG_NAME = "logistic_bridge.png"

WMIN, WMAX, NW = 0.3, 30.0, 121
WC_GRID = np.geomspace(0.03, 300.0, 161)
P_GRID = np.geomspace(0.25, 20.0, 121)


def _fmt(x):
    return f"{float(x):.17e}"


def cases():
    out = [
        dict(name="P1_point_y1", kind="point", y=1.0, primary=True),
        dict(name="M1_macro_point", kind="macro", primary=True),
    ]
    for y in (0.2, 0.1, 0.05, 0.02, 0.0):
        out.append(dict(name=f"C1_y{y:g}", kind="point", y=y,
                        primary=(y == 0.2)))
    return out


def make_field(case):
    if case["kind"] == "point":
        return fpl.PointMassField([(0.0, 0.0, 1.0)], beta=(case["y"], 0.0))
    return fpl.LensField([fpl.PointComp(0.0, 0.0, 0.2)], beta=(0.40, 0.10),
                         kappa=0.3, gamma1=0.15)


def window(omegas, wc, p):
    """Logistic window s(omega) = 1 / (1 + (omega_c/omega)^p)."""
    lx = np.clip(p * (np.log(wc) - np.log(omegas)), -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(lx))


def blend(variant, omegas, Fgeo, wc, p):
    s = window(omegas, wc, p)
    if variant == "v1":
        return 1.0 + s * (Fgeo - 1.0)
    mag = 1.0 + s * (np.abs(Fgeo) - 1.0)
    ph = np.where(np.abs(Fgeo) > 0.0, np.angle(Fgeo), 0.0)
    return mag * np.exp(1j * ph)


def rel_err(Fh, Fex):
    return np.abs(Fh - Fex) / np.abs(Fex)


def fit_case(omegas, Fex, Fgeo, variant):
    """Brute-force (omega_c, p) minimax fit over the band.

    Returns the max-error landscape (len(WC_GRID) x len(P_GRID)) and the best
    ``(max_err, omega_c, p)``.
    """
    grid = np.empty((len(WC_GRID), len(P_GRID)))
    best = (float("inf"), 0.0, 0.0)
    lw = np.log(omegas)
    for i, wc in enumerate(WC_GRID):
        lx = np.clip(P_GRID[None, :] * (np.log(wc) - lw[:, None]), -700.0, 700.0)
        s = 1.0 / (1.0 + np.exp(lx))
        if variant == "v1":
            Fh = 1.0 + s * (Fgeo[:, None] - 1.0)
        else:
            mag = 1.0 + s * (np.abs(Fgeo)[:, None] - 1.0)
            ph = np.where(np.abs(Fgeo) > 0.0, np.angle(Fgeo), 0.0)
            Fh = mag * np.exp(1j * ph)[:, None]
        er = np.abs(Fh - Fex[:, None]) / np.abs(Fex)[:, None]
        mx = er.max(axis=0)
        grid[i, :] = mx
        j = int(np.argmin(mx))
        if mx[j] < best[0]:
            best = (float(mx[j]), float(wc), float(P_GRID[j]))
    return grid, best


def fit_global(grids):
    """One (omega_c, p) minimising the worst case error across `grids`."""
    stack = np.maximum.reduce(list(grids.values()))
    idx = np.unravel_index(int(np.argmin(stack)), stack.shape)
    return float(WC_GRID[idx[0]]), float(P_GRID[idx[1]]), float(stack[idx])


def run(args):
    os.makedirs(RESULTS, exist_ok=True)
    omegas = np.geomspace(WMIN, WMAX, int(args.nw))
    data = {}
    fit_rows = []
    checks = []
    for case in cases():
        name = case["name"]
        field = make_field(case)
        print(f"== case {name} ==")
        sw = field.sweep([float(w) for w in omegas])
        Fex = np.asarray(sw.amplification, dtype=complex)
        ex_err = np.asarray(sw.error, dtype=float)
        try:
            Fgeo = np.array([field.geometric(omega=float(w)).amplification
                             for w in omegas], dtype=complex)
            n_im = len(field.images())
            print(f"   {n_im} images, |F_geo| = {np.abs(Fgeo).min():.3e}"
                  f"..{np.abs(Fgeo).max():.3e}; sweep: {sw.inner_solves} inner "
                  f"solves in {sw.groups} groups")
        except RuntimeError as exc:
            # A caustic (e.g. y = 0 in C1): no discrete image set, so no
            # geometric anchor.  The finder reports it (see the module
            # docstring); record NaNs and skip this case's fits.
            print(f"   no discrete image set (caustic): {exc}")
            Fgeo = np.full(len(omegas), complex(np.nan, np.nan))
            n_im = 0
        fit_ok = bool(np.all(np.isfinite(Fgeo)))

        # referees -------------------------------------------------------
        if case["kind"] == "point":
            nd = [fpl.point_mass_nd(omega=float(w), y=case["y"])
                  for w in omegas]
            Fnd = np.array([r.amplification for r in nd], dtype=complex)
            nd_err = np.array([r.error for r in nd])
            ok = nd_err < 1e-8
            dmod = np.abs(np.abs(Fnd[ok]) - np.abs(Fex[ok]))
            checks.append(dict(
                case=name, check="ND modulus referee", value=_fmt(dmod.max()),
                detail=f"max |d|F|| over {int(ok.sum())}/{len(ok)} omegas "
                       f"with nd.err < 1e-8 (nd err at w=30: {nd_err[-1]:.1e})"))
        else:
            for w in (1.0, 3.0, 10.0, 30.0):
                a = field.solve(omega=w)
                b = field.solve(omega=w, margin=2.0)
                checks.append(dict(
                    case=name, check="contour margin (2.0 vs 1.5)",
                    value=_fmt(abs(a.amplification - b.amplification)
                               / abs(a.amplification)),
                    detail=f"omega = {w:g}"))
        for w in (0.3, 3.0, 30.0):
            k = int(np.argmin(np.abs(omegas - w)))
            s1 = field.solve(omega=float(omegas[k]))
            checks.append(dict(
                case=name, check="sweep vs solve", value=_fmt(
                    abs(sw.amplification[k] - s1.amplification)
                    / abs(s1.amplification)),
                detail=f"omega = {omegas[k]:.6g}"))

        # fitting --------------------------------------------------------
        if fit_ok:
            grid1, best1 = fit_case(omegas, Fex, Fgeo, "v1")
            grid2, best2 = fit_case(omegas, Fex, Fgeo, "v2")
            print(f"   tuned V1: omega_c = {best1[1]:.4g}, p = {best1[2]:.4g}, "
                  f"max rel = {best1[0]:.3e}")
            print(f"   tuned V2: omega_c = {best2[1]:.4g}, p = {best2[2]:.4g}, "
                  f"max rel = {best2[0]:.3e}")
            for variant, best in (("v1", best1), ("v2", best2)):
                er = rel_err(blend(variant, omegas, Fgeo, best[1], best[2]), Fex)
                fit_rows.append(dict(
                    case=name, variant=variant, tier="tuned",
                    omega_c=_fmt(best[1]), p=_fmt(best[2]),
                    max_err=_fmt(er.max()),
                    rms_err=_fmt(np.sqrt((er ** 2).mean())),
                    omega_at_max=_fmt(omegas[int(np.argmax(er))])))
        else:
            # Caustic: no geometric anchor, no fits (all-NaN rows below).
            grid1 = grid2 = None
            best1 = best2 = (float("nan"), float("nan"), float("nan"))
        data[name] = dict(case=case, Fex=Fex, Fgeo=Fgeo, ex_err=ex_err,
                          grid1=grid1, grid2=grid2, best1=best1, best2=best2)
        for variant, label in (("geo anchor", rel_err(Fgeo, Fex)),
                               ("F=1 only", rel_err(np.ones_like(Fex), Fex))):
            fit_rows.append(dict(
                case=name, variant=variant, tier="--", omega_c="", p="",
                max_err=_fmt(label.max()),
                rms_err=_fmt(np.sqrt((label ** 2).mean())),
                omega_at_max=_fmt(omegas[int(np.argmax(label))])))

    # fixed prescription over the primaries ------------------------------
    prim = [c["name"] for c in cases() if c["primary"]]
    wc_g, p_g, obj = fit_global({n: data[n]["grid1"] for n in prim})
    print(f"== fixed prescription (V1, minimax over {prim}): "
          f"omega_c = {wc_g:.4g}, p = {p_g:.4g}, worst = {obj:.3e}")
    fit_rows.append(dict(case="ALL primaries", variant="v1", tier="fixed",
                         omega_c=_fmt(wc_g), p=_fmt(p_g), max_err=_fmt(obj),
                         rms_err="", omega_at_max=""))
    for n in prim:
        er = rel_err(blend("v1", omegas, data[n]["Fgeo"], wc_g, p_g),
                     data[n]["Fex"])
        fit_rows.append(dict(
            case=n, variant="v1", tier="fixed", omega_c=_fmt(wc_g),
            p=_fmt(p_g), max_err=_fmt(er.max()),
            rms_err=_fmt(np.sqrt((er ** 2).mean())),
            omega_at_max=_fmt(omegas[int(np.argmax(er))])))

    # blind canonical pairs (no calibration at all) ------------------------
    for wc_b, p_b in ((1.0, 1.0), (1.0, 2.0), (1.0, 4.0)):
        for n in prim:
            er = rel_err(blend("v1", omegas, data[n]["Fgeo"], wc_b, p_b),
                         data[n]["Fex"])
            fit_rows.append(dict(
                case=n, variant="v1", tier=f"blind({wc_b:g},{p_b:g})",
                omega_c=_fmt(wc_b), p=_fmt(p_b), max_err=_fmt(er.max()),
                rms_err=_fmt(np.sqrt((er ** 2).mean())),
                omega_at_max=_fmt(omegas[int(np.argmax(er))])))

    # per-omega rows ------------------------------------------------------
    fields = ["case", "kind", "y", "omega", "ex_re", "ex_im", "geo_re",
              "geo_im", "h1_re", "h1_im", "h2_re", "h2_im", "hg_re", "hg_im",
              "err1", "err2", "errg", "ref_err"]
    rows = []
    for case in cases():
        name = case["name"]
        d = data[name]
        Fgeo, Fex = d["Fgeo"], d["Fex"]
        h1 = blend("v1", omegas, Fgeo, d["best1"][1], d["best1"][2])
        h2 = blend("v2", omegas, Fgeo, d["best2"][1], d["best2"][2])
        hg = blend("v1", omegas, Fgeo, wc_g, p_g)
        e1, e2, eg = rel_err(h1, Fex), rel_err(h2, Fex), rel_err(hg, Fex)
        for k, w in enumerate(omegas):
            ref_err = d["ex_err"][k] * w / (2.0 * np.pi) / abs(Fex[k])
            rows.append(dict(
                case=name, kind=case["kind"],
                y=("" if case["kind"] == "macro" else _fmt(case["y"])),
                omega=_fmt(w),
                ex_re=_fmt(Fex[k].real), ex_im=_fmt(Fex[k].imag),
                geo_re=_fmt(Fgeo[k].real), geo_im=_fmt(Fgeo[k].imag),
                h1_re=_fmt(h1[k].real), h1_im=_fmt(h1[k].imag),
                h2_re=_fmt(h2[k].real), h2_im=_fmt(h2[k].imag),
                hg_re=_fmt(hg[k].real), hg_im=_fmt(hg[k].imag),
                err1=_fmt(e1[k]), err2=_fmt(e2[k]), errg=_fmt(eg[k]),
                ref_err=_fmt(ref_err)))

    with open(os.path.join(RESULTS, CSV_NAME), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {os.path.join(RESULTS, CSV_NAME)} ({len(rows)} rows)")

    with open(os.path.join(RESULTS, FITS_NAME), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["case", "variant", "tier",
                                            "omega_c", "p", "max_err",
                                            "rms_err", "omega_at_max"])
        wr.writeheader()
        wr.writerows(fit_rows)
    print(f"wrote {os.path.join(RESULTS, FITS_NAME)}")

    with open(os.path.join(RESULTS, CHECKS_NAME), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["case", "check", "value", "detail"])
        wr.writeheader()
        wr.writerows(checks)
    print(f"wrote {os.path.join(RESULTS, CHECKS_NAME)}")


def _read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def report():
    rows = _read_csv(os.path.join(RESULTS, CSV_NAME))
    fits = _read_csv(os.path.join(RESULTS, FITS_NAME))
    checks = _read_csv(os.path.join(RESULTS, CHECKS_NAME))
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case"], []).append(r)
    primary = [r for r in fits if r["tier"] == "tuned" and r["variant"] == "v1"]

    lines = []
    lines.append("# P5 logistic-bridging comparator -- results")
    lines.append("")
    lines.append(
        "The heuristic competitor (Paul's end-stage item): bridge the wave "
        "limit `F -> 1` and the geometric sum `F_geo(omega)` with a logistic "
        "window `s(omega) = 1/(1 + (omega_c/omega)^p)` in log omega, instead "
        "of evaluating the Fresnel integral.  V1 blends the complex values; "
        "V2 blends the moduli and takes the phase from `F_geo`.  `tuned` = "
        "`(omega_c, p)` fitted per case (sees the exact answer); `fixed` = "
        "one pair for the three primaries (P1, M1, C1 at y = 0.2).")
    lines.append("")
    lines.append("Reference: the exact RC engine (`sweep`); referees: the "
                 "Nakamura-Deguchi closed form (moduli) for the point-mass "
                 "cases and a second contour margin for M1.")
    lines.append("")

    lines.append("## tuned per case (maximally fair)")
    lines.append("")
    lines.append("| case | variant | omega_c | p | max rel err | RMS | at omega |")
    lines.append("|------|---------|---------|---|-------------|-----|----------|")
    for r in fits:
        if r["tier"] != "tuned":
            continue
        lines.append(f"| {r['case']} | {r['variant']} | {float(r['omega_c']):.4g} "
                     f"| {float(r['p']):.4g} | {float(r['max_err']):.3e} "
                     f"| {float(r['rms_err']):.3e} | {float(r['omega_at_max']):.4g} |")
    lines.append("")

    fixed = [r for r in fits if r["tier"] == "fixed"]
    if fixed:
        head = fixed[0]
        lines.append("## fixed prescription (one pair, no per-case knowledge)")
        lines.append("")
        lines.append(f"`omega_c = {float(head['omega_c']):.4g}`, "
                     f"`p = {float(head['p']):.4g}` (minimax over the three "
                     f"primaries; worst-case max error {float(head['max_err']):.3e}).")
        lines.append("")
        lines.append("| case | max rel err | RMS | at omega |")
        lines.append("|------|-------------|-----|----------|")
        for r in fixed[1:]:
            lines.append(f"| {r['case']} | {float(r['max_err']):.3e} "
                         f"| {float(r['rms_err']):.3e} "
                         f"| {float(r['omega_at_max']):.4g} |")
        lines.append("")

    blind = [r for r in fits if r["tier"].startswith("blind(")]
    if blind:
        pairs = sorted({r["tier"] for r in blind})
        order = ["P1_point_y1", "M1_macro_point", "C1_y0.2"]
        lines.append("### blind canonical pairs (written down without data)")
        lines.append("")
        lines.append("| case | " +
                     " | ".join(p.replace("blind", "") for p in pairs) +
                     " |")
        lines.append("|" + "---|" * (len(pairs) + 1))
        for n in order:
            cells = []
            for p in pairs:
                v = next(r for r in blind if r["case"] == n
                         and r["tier"] == p)
                cells.append(f"{float(v['max_err']):.2e}")
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## ingredient floors (context)")
    lines.append("")
    lines.append("| case | floor | max rel err |")
    lines.append("|------|-------|-------------|")
    for r in fits:
        if r["tier"] != "--":
            continue
        lines.append(f"| {r['case']} | {r['variant']} | {float(r['max_err']):.3e} |")
    lines.append("")

    lines.append("## referees")
    lines.append("")
    lines.append("| case | check | value | detail |")
    lines.append("|------|-------|-------|--------|")
    for r in checks:
        lines.append(f"| {r['case']} | {r['check']} | {float(r['value']):.2e} "
                     f"| {r['detail']} |")
    lines.append("")

    # caustic structure ---------------------------------------------------
    lines.append("## the caustic family (structural failure)")
    lines.append("")
    z = by_case.get("C1_y0", [])
    if z:
        w = np.array([float(r["omega"]) for r in z])
        m = np.array([abs(complex(float(r["ex_re"]), float(r["ex_im"])))
                      for r in z])
        slope = np.polyfit(np.log(w), np.log(m), 1)[0]
        lines.append(f"At `y = 0` the image finder returns no images (the two "
                     f"solutions sit on the critical curve and are discarded), "
                     f"so `|F_geo| = 0`; the exact `|F|` grows as a power law "
                     f"over the band with fitted slope **{slope:.3f}** "
                     f"(`sqrt(omega)` for the Einstein-ring stationary "
                     f"manifold; `|F| = {m[0]:.3f} .. {m[-1]:.3f}`).")
        lines.append("")
    lines.append("| y | tuned V1 max err | geo-anchor max err | F=1 max err |")
    lines.append("|---|------------------|--------------------|-------------|")
    for r in fits:
        if not r["case"].startswith("C1_") or r["tier"] != "tuned" \
                or r["variant"] != "v1":
            continue
        geo = next(x for x in fits if x["case"] == r["case"]
                   and x["variant"] == "geo anchor")
        one = next(x for x in fits if x["case"] == r["case"]
                   and x["variant"] == "F=1 only")
        lines.append(f"| {r['case'].replace('C1_y', '')} "
                     f"| {float(r['max_err']):.3e} "
                     f"| {float(geo['max_err']):.3e} "
                     f"| {float(one['max_err']):.3e} |")
    lines.append("")
    cand = [r for r in rows if r["case"] != "C1_y0"]
    wr = max(cand, key=lambda r: float(r["err1"]))
    exw = abs(complex(float(wr["ex_re"]), float(wr["ex_im"])))
    gew = abs(complex(float(wr["geo_re"]), float(wr["geo_im"])))
    lines.append(
        f"The largest tuned-bridge error in the family sits at a *fringe "
        f"minimum* ({wr['case']}, omega = {float(wr['omega']):.2f}: "
        f"|F| = {exw:.3f}, |F_geo| = {gew:.3f} — a near-cancellation of the "
        f"two images), where any additive bridge inherits the geometric "
        f"cancellation; the absolute error there is small and the RMS column "
        f"averages this out.  The fit metric is the max relative error, so "
        f"the floors above are conservative at such nulls.")
    lines.append("")

    with open(os.path.join(RESULTS, MD_NAME), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.join(RESULTS, MD_NAME)}")

    # figure --------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11.0, 7.5))
    colors = {"P1_point_y1": "C0", "M1_macro_point": "C1", "C1_y0.2": "C2"}

    a = ax[0, 0]
    for name, col in (("P1_point_y1", "C0"),):
        rs = by_case[name]
        w = np.array([float(r["omega"]) for r in rs])
        a.loglog(w, [abs(complex(float(r["ex_re"]), float(r["ex_im"]))) for r in rs],
                 col, label="exact")
        a.loglog(w, [abs(complex(float(r["geo_re"]), float(r["geo_im"]))) for r in rs],
                 col + "--", alpha=0.6, label="geometric anchor")
        a.loglog(w, [abs(complex(float(r["h1_re"]), float(r["h1_im"]))) for r in rs],
                 col + ":", label="tuned V1")
    a.set_title("P1: |F| vs omega (point mass, y = 1)")
    a.set_xlabel("omega")
    a.set_ylabel("|F|")
    a.legend(fontsize=8)

    a = ax[0, 1]
    for name in ("P1_point_y1", "M1_macro_point", "C1_y0.2"):
        rs = by_case[name]
        w = np.array([float(r["omega"]) for r in rs])
        a.loglog(w, [float(r["err1"]) for r in rs], colors[name],
                 label=f"{name} tuned V1")
        a.loglog(w, [float(r["errg"]) for r in rs], colors[name] + "--",
                 alpha=0.6, label=f"{name} fixed V1")
    a.set_title("relative error, V1 blends")
    a.set_xlabel("omega")
    a.set_ylabel("|F_h - F| / |F|")
    a.legend(fontsize=7)

    a = ax[1, 0]
    for name, col in (("C1_y0", "k"), ("C1_y0.02", "C3"), ("C1_y0.2", "C2")):
        rs = by_case[name]
        w = np.array([float(r["omega"]) for r in rs])
        ef = [abs(complex(float(r["ex_re"]), float(r["ex_im"]))) for r in rs]
        gf = [abs(complex(float(r["geo_re"]), float(r["geo_im"]))) for r in rs]
        a.loglog(w, ef, col, label=f"exact, {name}")
        a.loglog(w, gf, col + "--", alpha=0.6, label=f"geo, {name}")
    a.set_title("caustic family: |F| (geo=0 at y=0)")
    a.set_xlabel("omega")
    a.legend(fontsize=7)

    a = ax[1, 1]
    ys, e1, eg = [], [], []
    for r in fits:
        if r["case"].startswith("C1_") and r["tier"] == "tuned" \
                and r["variant"] == "v1":
            ys.append(float(r["case"].replace("C1_y", "")))
            e1.append(float(r["max_err"]))
            eg.append(float(next(x["max_err"] for x in fits
                                 if x["case"] == r["case"]
                                 and x["variant"] == "geo anchor")))
    order = np.argsort(ys)
    ys = np.clip(np.array(ys)[order], 0.01, None)
    a.loglog(ys, np.array(e1)[order], "o-", label="tuned V1 max err")
    a.loglog(ys, np.array(eg)[order], "s--", alpha=0.6, label="geo-anchor max err")
    a.set_title("caustic approach: error vs y (left edge = y=0)")
    a.set_xlabel("y")
    a.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(RESULTS, PNG_NAME)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "report", "all"], nargs="?",
                    default="all")
    ap.add_argument("--nw", type=int, default=NW)
    args = ap.parse_args()
    if args.command in ("run", "all"):
        run(args)
    if args.command in ("report", "all"):
        report()


if __name__ == "__main__":
    main()
