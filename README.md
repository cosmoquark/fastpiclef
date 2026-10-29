# `fastpiclef`: the faster-phan-light wave-optics solver!

- Uses Picard–Lefschetz theory 🧐
- Written in Rust 🦀 (with deterministic task-level parallelism)
- Python wrapper 🐍 (`import fastpiclef`)

Exact, fast evaluation of the gravitational-lensing **Fresnel integral**

$$
  F(\omega, \beta) = \left(\frac{\omega}{2 \pi i}\right)^\frac{d}{2} \cdot \int_R^d \exp(i \omega \phi(x; \beta)) \mathrm{d}^d x
$$

normalised so that free space gives $F = 1$. The integrand oscillates on the
Fresnel scale $\omega^\frac{1}{2}$, converges only conditionally at infinity,
and point-mass lenses add logarithmic branch points that real-space quadratures
cannot cross.  `fastpiclef` deforms the contour into the complex plane along
Picard-Lefschetz steepest-descent directions instead: a real core anchored at
every branch-point pinch plus $pi/4$-rotated Gaussian-decay tails, applied
recursively in the two integration variables. The result is exact by Cauchy,
with a validated error budget, instead of an oscillatory quadrature.

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

- **Batched sweeps, maps and grids** — `PointMassField` / `LensField` expose
  `.sweep(omegas)` and `.map(beta0s, beta1s, omega)`, the 1D lenses expose
  `.sweep()` and `.grid()`.  One contour is shared across all frequencies (or
  all sources) in a band, so the geometric cost is amortised rather than paid
  per point.
- **Verified image finder** — `.images()` / `.images_detailed()` return the
  real solutions of the lens equation from a multi-start Newton search that
  carries a **completeness certificate** (index identity plus per-cusp-group
  companions).  A failed certificate or a degenerate (caustic) image set
  raises instead of returning a silently short list.
- **Geometric optics and its high-`omega` hybrid** — `.geometric(omega)` is the
  stationary-phase sum (exact as `omega -> infinity` away from caustics),
  `.geometric_corrected(omega)` adds the next-order `1/(i omega)` correction,
  and `.hybrid(omega, tol)` uses that fast branch only when its conservative
  estimate is below `tol`, falling back to the exact RC engine otherwise.
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
- Self-agreement is not treated as truth: the suite grades against external
  closed forms and referees at a *different* contour margin, independent
  contour families, transposed-slicing symmetry, and bitwise baselines.
  Details, benchmarks against `plind` / `P-LPI` / FIONA, and the measured error
  budget are in [`docs/report/main.tex`](docs/report/main.tex)
  (`latexmk -pdf main.tex`).

## Installation

`fastpiclef` is **not on PyPI yet, and no prebuilt wheels are published** —
there are no git tags or release assets either.  So every route below
compiles the Rust core once (about a minute on a laptop) and then installs
it; only a wheel you build yourself installs without a compiler (see
[Building a wheel once](#building-a-wheel-once-installing-later-without-rust)).

Requirements:

- **Python >= 3.12**.  The extension is an `abi3` module, so one build serves
  3.12 and later (3.14 additionally needs
  `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`, which the pixi environments set for
  you).
- **Rust >= 1.85** and a C linker (`cc` / `gcc` / `clang`) for source builds.
  [rustup](https://rustup.rs) is the easy route:
  `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`.

### uv

The source is on GitHub (`https://github.com/cosmoquark/fastpiclef.git`,
primary) with a Codeberg mirror
(`https://codeberg.org/cosmoquark/fast_piclef.git`); substitute whichever you
have access to.

```bash
git clone https://github.com/cosmoquark/fastpiclef.git
cd fastpiclef
uv venv            # or use an existing environment
uv pip install .
```

Straight from the repository, or into an existing uv project:

```bash
uv pip install "fastpiclef @ git+https://github.com/cosmoquark/fastpiclef.git"
uv add "fastpiclef @ git+https://github.com/cosmoquark/fastpiclef.git"
```

Both forms build the extension in release mode through maturin and also
install the type stub, so editors and mypy see the API.  `uv pip install`
needs an environment to install into — `uv venv` first, or pass `--system`.

### pip

The same, with pip instead of uv:

```bash
git clone https://github.com/cosmoquark/fastpiclef.git
cd fastpiclef
python -m venv .venv && . .venv/bin/activate
pip install .
# or: pip install "fastpiclef @ git+https://github.com/cosmoquark/fastpiclef.git"
```

For an editable install while developing the bindings, install `maturin`
(`pip install maturin`) and run `maturin develop --release` inside the
activated environment; that is exactly what `pixi run build` does.

### Building a wheel once (installing later without Rust)

One command, no docker:

```bash
bash scripts/build-dist.sh     # artifacts only
bash scripts/release.sh        # artifacts + regenerated index, then prints the publish steps
```

It builds in a throwaway `uv` venv holding `maturin` + `zig` (from PyPI) and
writes to `dist/`:

```
fastpiclef-0.1.0-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
fastpiclef-0.1.0.tar.gz          # ~260 KB sdist
```

The wheel installs in milliseconds with **no Rust toolchain**: verified on
CPython 3.12, 3.13 and 3.14 (one `abi3` wheel covers all three) with `cargo`
removed from `PATH`, giving bit-identical results.  The sdist is deliberately
lean — the comparison harness, notebooks, prototypes and report are excluded
in `[tool.maturin]`.

Why zig at all: the crate has no C dependencies (the wheel needs only
`libc`/`libm`), but a native build records the build machine's glibc symbols
(`GLIBC_2.44` here), which PyPI rejects and older systems cannot import.  Zig
links the same Rust objects against an old glibc (`GLIBC_2.14`), which is what
the `manylinux_2_17` tag promises.  It is a link-time tool only: not a
dependency of the package, not in `Cargo.toml`, not in the wheel.  Building
inside a manylinux container (as CI does) is the alternative; `auditwheel`
alone cannot help, since it bundles libraries rather than rewriting symbol
versions.

Then upload — TestPyPI rehearsal first:

```bash
maturin upload --repository-url https://test.pypi.org/legacy/ dist/*   # token
maturin upload dist/*                                                  # MATURIN_PYPI_TOKEN
```

### Consuming it as a dependency

```bash
uv add fastpiclef                    # once the wheel is on PyPI — no Rust
uv add "fastpiclef @ <wheel-url>"    # a release asset — no Rust
uv add "fastpiclef @ git+https://github.com/cosmoquark/fastpiclef.git"  # source: Rust required
```

The first form is the one to aim for: one line, no URL, no compiler, and
`uv run --with fastpiclef script.py` works for one-off scripts too.  Until the
wheel is published, the git form is a working fallback, but note the cost
difference: `uv` records the commit in `uv.lock` and caches the built wheel, so
the ~45 s Rust build happens once per commit per machine; pip re-builds on
every fresh environment and fails outright where no Rust toolchain exists.

#### Without PyPI: a static index on GitHub Pages (recommended)

A plain HTML PEP 503 index needs no server: GitHub Pages can serve it, and its
links point at the release assets, so no wheel is committed to git.  One-time
setup: enable Pages for this repository from `main` / `docs` (free plans need a
public repository).  At each release, `bash scripts/release.sh --publish`
builds, regenerates `docs/simple/`, tags, pushes and attaches `dist/*`.

Collaborators then configure this **once** and never touch a URL again:

```toml
# pyproject.toml
[project]
dependencies = ["fastpiclef"]

[[tool.uv.index]]
url = "https://cosmoquark.github.io/fastpiclef/simple"
```

```bash
uv sync                 # installs the newest published wheel, no Rust
uv lock --upgrade       # picks up the next release
```

(`uv add fastpiclef --index https://cosmoquark.github.io/fastpiclef/simple`
writes the same block.)  Verified with the index and the wheels on *different*
hosts — the realistic Pages-plus-release-assets split: uv resolved the
version, downloaded the wheel from the asset URL, installed it with `cargo`
removed from `PATH`, and `uv.lock` records the registry plus the artifact's
sha256.

#### Without PyPI: a pinned release-asset URL (zero setup)

If you would rather not enable Pages, the same assets work as a direct URL —
but the URL names one exact file, so each new release means editing every
consumer's `pyproject.toml`.  Attach `dist/*` to a tag on the public
repository (GitHub and Codeberg both serve
`…/releases/download/<tag>/<file>`), and the collaborator runs:

```bash
uv add "fastpiclef @ https://github.com/cosmoquark/fastpiclef/releases/download/v0.1.0/fastpiclef-0.1.0-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
```

`uv` records that as a normal source and pins it (URL + sha256) in `uv.lock`:

```toml
[tool.uv.sources]
fastpiclef = { url = "https://…/fastpiclef-0.1.0-…whl" }
```

Verified: with `cargo` removed from `PATH` the install is instant, and deleting
`uv.lock` and re-resolving still works (the URL lives in `pyproject.toml`, not
only in the lock).  Put that block in a starter `pyproject.toml` and nobody
ever types the URL.

For several platforms, list one source per platform with a marker — `uv`
resolves universally, so every listed wheel must exist:

```toml
[tool.uv.sources]
fastpiclef = [
  { url = "https://…/fastpiclef-0.1.0-cp312-abi3-manylinux_2_17_x86_64.…whl", marker = "sys_platform == 'linux' and platform_machine == 'x86_64'" },
  { url = "https://…/fastpiclef-0.1.0-cp312-abi3-macosx_11_0_arm64.whl",      marker = "sys_platform == 'darwin' and platform_machine == 'arm64'" },
]
```

Two caveats.  `uv add --find-links <page>` resolves once but is *not* written to
`pyproject.toml`, so a later re-resolve forgets it — prefer a URL source or a
PEP 503 index (`uv add fastpiclef --index https://…/simple`, which does
persist).  And release assets on a *private* repository require
authentication, so this route assumes a public repository.

### pixi (development, notebooks, comparison harness)

pixi provisions Rust, Python and all analysis dependencies, so it is the
zero-setup route:

```bash
pixi run build        # maturin develop --release into the active environment
pixi run test-py      # wrapper + type-stub tests
pixi run -e notebook lab   # JupyterLab with the notebooks
```

Each pixi environment carries its own installed extension, so rebuild with
`pixi run -e <env> build` after Rust changes.

### Compiling the Rust crate directly

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

### Full check

`bash scripts/ci.sh` runs clippy, builds the examples, runs the release test
suite, then rebuilds and tests the Python bindings (`fast` and `rust` modes
trim it down).

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

## Repository layout

```
src/            Rust core (RC quadrature, slice/field engine, thimble, polar,
                batching, image finder, lens profiles, Python bindings)
tests/          Rust integration tests (incl. bitwise baselines)
tests/python/   wrapper and type-stub tests
examples/       timing/cost drivers (many_masses, batch_bench, ...)
comparison/     harness vs plind / P-LPI / FIONA / wolensing, referees, datasets
notebooks/      executed notebooks
docs/report/    the technical report (methods, validation, performance)
Journal.typ     append-only development log / roadmap
```

## Status

Research code at version `0.1.0`: the Python API may still change, and the
validated range is documented per solver rather than universal.  Known
limitation worth stating up front: exact field evaluations cost roughly
linearly in `omega` and quadratically-ish in the number of masses
(seconds, not milliseconds, for a 40-star field at `omega = 2`); the
`geometric` / `hybrid` paths exist precisely to cover the high-frequency
regime cheaply.
