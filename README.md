# `fastpiclef`: the faster-phan-light wave-optics solver!

- Uses Picard–Lefschetz theory 🧐
- Is written in Rust 🦀
- Includes a Python wrapper 🐍 (`import fastpiclef`)

Exact, fast evaluation of the gravitational-lensing **Fresnel integral**

$$
  F(\omega, \beta) = \left(\frac{\omega}{2 \pi i}\right)^\frac{d}{2} \cdot \int_R^d \exp(i \omega \phi(x; \beta)) \mathrm{d}^d x
$$

normalised so that free space gives $F = 1$. The integrand oscillates on the
Fresnel scale $\omega^\frac{1}{2}$, converges only conditionally at infinity,
and point-mass lenses add logarithmic branch points that real-space quadratures
cannot cross. `fastpiclef` deforms the contour into the complex plane along
Picard–Lefschetz steepest-descent directions instead: a real core anchored at
every branch-point pinch plus $\pi/4$-rotated Gaussian-decay tails, applied
recursively in the two integration variables. The result is exact by Cauchy,
with a validated error budget, instead of an oscillatory quadrature.

## Installation

Requirements:

- **Python ≥ 3.12**.  The extension is an `abi3` module, so one build serves
  3.12 and later (CPython 3.14 additionally needs
  `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`, which the pixi environments set for
  you).
- **Rust ≥ 1.85** and a C linker (`cc` / `gcc` / `clang`) for source builds.
  [rustup](https://rustup.rs) is the easy route.

⚠️ N.B.: `pixi`, `uv` & `pip` compile the Rust components. However, **`pixi`, `uv` & `pip` do not provide Rust, so `cargo`, `rustc` and a linker must be on `PATH`;** without them the install fails inside maturin.

### Pixi

Inside a Pixi project, you can add the dependency via

```bash
pixi add --pypi fastpiclef --git https://github.com/cosmoquark/fastpiclef.git
```

### UV

Inside a UV project, you can add the dependency via
```bash
uv add "fastpiclef @ git+https://github.com/cosmoquark/fastpiclef.git"
```

In a bare virtual environment, one may us `uv pip install` instead of
`uv add` above, which installs only and writes no project files.

### pip

As for Pip, one may write

```bash
pip install "fastpiclef @ git+https://github.com/cosmoquark/fastpiclef.git"
```

### Rust crate only (no Python)

The crate is a normal Cargo library (plus a `demo` binary), so it builds and
tests without Python:

```bash
cargo build --release             # library + demo binary
cargo build --release --examples  # cost/timing drivers in examples/
cargo test --release              # full suite (~6 min)
cargo run --release --bin demo    # descent + quadrature pipeline demo
```

Do **not** run `cargo build --features python`: that feature turns the crate
into a Python extension module and breaks the demo binary by design.  The
bindings are always built through maturin (`pixi run build`, or
`maturin develop --release`).

### Prebuilt wheel (installing later without Rust)

```bash
bash scripts/build-dist.sh     # from the checkout: wheel + sdist in dist/
pip install dist/*.whl         # installs with no Rust toolchain
```

`bash scripts/release.sh` does the same and also regenerates the static
PEP 503 index in `docs/simple/` (served by GitHub Pages); with `--publish` it
tags, pushes and attaches `dist/*` to the GitHub release.  Collaborators who
prefer the index add it once and then install normally:

```toml
# pyproject.toml
[[tool.uv.index]]
url = "https://cosmoquark.github.io/fastpiclef/simple"
```

The wheel is one `abi3` build for CPython 3.12+ on x86_64 Linux, verified
bit-identical to a source build with `cargo` removed from `PATH`.  Zig is a
link-time tool only (it links the Rust objects against an old glibc so the
wheel carries the portable `manylinux_2_17` tag); it is not a dependency of
the package.  The sdist carries the sources so other platforms can build
locally.

## Quick start

```python
import fastpiclef as fpl

# 1D cubic phase: exact, through the caustic at y = 0
print(fpl.AiryPhase(y=0.0).solve(omega=4.0).amplification)

# Isolated point mass: polar path, fast at high frequency
r = fpl.PointMass(y=1.0).solve(omega=100.0)
print(r.amplification, r.error)

# Nakamura-Deguchi closed form: microseconds, good for omega*y <~ 10
print(fpl.point_mass_nd(2.0, 0.3).amplification)

# A field of point masses: exact 2D RC slice solver (the default)
field = fpl.PointMassField(
    [(0.0, 0.0, 1.0), (2.0, 0.0, 0.3), (-1.5, 1.0, 0.5)], beta=(0.2, 0.1)
)
print(field.solve(omega=2.0))
print(field.images())

# Batched frequency sweep: one shared contour for all omegas
sw = field.sweep([0.5, 1.0, 2.0, 5.0])
print(sw.amplification, sw.groups)

# General field: stars + halo + external convergence and shear (a macro image)
lf = fpl.LensField(
    [fpl.PointComp(0.0, 0.0, 1.0), fpl.NFWComp(2.0, 0.0, 0.3, 1.0)],
    beta=(0.1, 0.0), kappa=0.1, gamma1=0.05,
)
print(lf.solve(3.0))                # exact finite-frequency integral
print(lf.geometric(100.0))          # geometric optics
print(lf.hybrid(100.0, tol=1e-3))   # asymptotic branch when its budget < tol
```

which prints, e.g.

```
Result(amplification=1.715402-1.443162i, integral=4.533828+5.389093i, error=1.11e-13, method='slice', steps=2566)
```

For gravitational waves, the dimensionless frequency is
`omega = 8 pi G M_Lz f / c^3 = 1.2379e-4 (M_Lz / M_sun) (f / Hz)` in units of
the Einstein radius of the lens, so a `10 M_sun` microlens in the LIGO band
spans `omega ~ 0.01–1` — the regime this solver was built for.

More: [`getting_started.ipynb`](notebooks/getting_started.ipynb) (tour),
[`method.ipynb`](notebooks/method.ipynb) (how the deformation works),
[`time_domain.ipynb`](notebooks/time_domain.ipynb) (arrival-time view),
[`comparison.ipynb`](notebooks/comparison.ipynb) (head-to-head with the
reference codes).

## Capabilities

Every solve returns a `Result` with `.integral` (the raw Fresnel integral),
`.amplification` (the normalised `F`), `.error` (absolute error of `.integral`,
i.e. the same relative bound on `.amplification`), `.method` and `.steps`.

| Model | Entry points | Solver |
| --- | --- | --- |
| 1D cubic phase `z^3/3 - y z` | `AiryPhase` | complete-saddle Lefschetz thimble, machine precision |
| 1D rational lenses | `RationalLens`, `SimGLens`, `CustomRationalLens` | thimble (mesh flow as caustic fallback); `.thimble()`, `.saddles()` for inspection |
| Axisymmetric 2D | `PointMass`, `RationalLens2D` | polar / radial-thimble reduction (fast at high `omega`) |
| Fields of point masses | `PointMassField` | **exact 2D RC slice** solver (real core + rotated tails, both variables), the default `method="slice"` |
| General lens fields | `LensField` + `PointComp` / `SISComp` / `CISComp` / `NFWComp` | the same RC engine, with external `kappa`, `gamma1`, `gamma2` |
| Isolated unit point mass | `point_mass_nd(omega, y)` | Nakamura–Deguchi closed form (`1F1` in log space), microsecond class |

Beyond single evaluations:

- **Batched sweeps, maps and grids** — `.sweep(omegas)` and
  `.map(beta0s, beta1s, omega)` on the field solvers, `.sweep()` / `.grid()`
  on the 1D lenses.  One contour is shared across all frequencies (or all
  sources) in a band, so the geometric cost is amortised.
- **Verified image finder** — `.images()` / `.images_detailed()` return the
  real solutions of the lens equation from a multi-start Newton search that
  carries a **completeness certificate**.  A failed certificate or a
  degenerate (caustic) image set raises instead of returning a silently short
  list.
- **Geometric optics and its high-`omega` hybrid** — `.geometric(omega)` is the
  stationary-phase sum, `.geometric_corrected(omega)` adds the next-order
  `1/(i omega)` correction, and `.hybrid(omega, tol)` uses that fast branch
  only when its conservative estimate is below `tol`, falling back to the
  exact RC engine otherwise.
- **Low-frequency subtraction** — `solve(..., subtract_free=True)` removes the
  `eps * |I_free|` floor from `F - 1` below `omega ~ 1e-2`.
- **Raw kernels** — `slice_inner_rc(...)` exposes the exact 1D slice kernel
  `G(x2) = ∫ dz0 exp(i omega phi(z0, x2))` for complex `x2`.

## Accuracy and validation

- The Airy integral (1D, closed form) is reproduced to `1e-12`–`1e-16`
  relative, including through the caustic.
- Point-mass fields agree with the independent Nakamura–Deguchi closed form at
  the `1e-15` level over the validated band `omega ~ 0.3–1000`, and the
  reported `.error` covers the true error.
- Saddle-type macro images (`lambda0 < 0` or `lambda2 < 0`) are exact too —
  the regime where real-box/FFT methods miss the conditionally convergent
  directions by tens of percent.

## Repository layout

```
src/            Rust core (RC quadrature, slice/field engine, thimble, polar,
                batching, image finder, lens profiles, Python bindings)
tests/          Rust integration tests (incl. bitwise baselines)
tests/python/   wrapper and type-stub tests
examples/       timing/cost drivers (many_masses, batch_bench, ...)
notebooks/      executed notebooks
docs/report/    the technical report (methods, validation, performance)
```

## Status

Known limitations:
- the Python API may still change
- the validated range is documented per solver rather than universal.
- exact field evaluations cost roughly linearly in `omega` and quadratically-ish in the number of masses (seconds, for a 40-star field at `omega = 2`);
- the `geometric` / `hybrid` paths exist precisely to cover the high-frequency regime cheaply.
