"""Drift test for the hand-written `fastpiclef.pyi` type stub.

Checks that the stub and the compiled extension agree on the public surface:
names, class members (methods and getters), call signatures and defaults.
Catches the case where `src/python.rs` gains/renames an argument and the stub
(which is what VSCode/Pylance and mypy see) is not updated.

Also checks the stub that maturin installed next to the extension is current
(`maturin` ships the root `fastpiclef.pyi` as `fastpiclef/__init__.pyi` plus
`py.typed` for pure-Rust projects).

Run (from the repo root, after `pixi run build`):
    python tests/python/test_stub.py
"""

import ast
import inspect
import pathlib

import fastpiclef as fpl

STUB_PATH = pathlib.Path(__file__).resolve().parents[2] / "fastpiclef.pyi"

# Our stub convention: parameters before `*` / after `*`.
POS = "positional_or_keyword"
KW = "keyword_only"


def _tree() -> ast.Module:
    return ast.parse(STUB_PATH.read_text())


def _top_level_defs(tree: ast.Module):
    classes, funcs = {}, {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes[node.name] = node
        elif isinstance(node, ast.FunctionDef):
            funcs[node.name] = node
    return classes, funcs


def _stub_all(tree: ast.Module) -> list[str]:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return list(ast.literal_eval(node.value))
    raise AssertionError("fastpiclef.pyi has no top-level __all__")


def _stub_members(cls: ast.ClassDef):
    """({method name: node}, {attribute name}) for one stub class."""
    methods, attrs = {}, set()
    for stmt in cls.body:
        if isinstance(stmt, ast.FunctionDef):
            methods[stmt.name] = stmt
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            attrs.add(stmt.target.id)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    attrs.add(target.id)
    return methods, attrs


def _stub_params(fn: ast.FunctionDef):
    """[(name, kind, default-node-or-None)], `self`/`cls` excluded."""
    args = fn.args
    out = []
    positional = list(args.posonlyargs) + list(args.args)
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for arg, default in zip(positional, defaults):
        if arg.arg in ("self", "cls"):
            continue
        out.append((arg.arg, POS, default))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        out.append((arg.arg, KW, default))
    return out


def _runtime_params(obj):
    """[(name, kind, default)], `self` excluded; default is the raw value,
    `inspect.Parameter.empty` when required, `Ellipsis` when PyO3 could not
    render it (e.g. tuple defaults)."""
    out = []
    for p in inspect.signature(obj).parameters.values():
        if p.name in ("self", "cls"):
            continue
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            kind = POS
        elif p.kind == p.KEYWORD_ONLY:
            kind = KW
        else:
            raise AssertionError(f"{obj}: unexpected parameter kind {p.kind}")
        out.append((p.name, kind, p.default))
    return out


def _compare_signatures(what: str, stub_fn: ast.FunctionDef, runtime_obj) -> None:
    stub = _stub_params(stub_fn)
    runtime = _runtime_params(runtime_obj)
    assert [s[0] for s in stub] == [r[0] for r in runtime], (
        f"{what}: parameter names differ: "
        f"stub {[s[0] for s in stub]} vs runtime {[r[0] for r in runtime]}"
    )
    for (sname, skind, sdefault), (_, rkind, rdefault) in zip(stub, runtime):
        assert skind == rkind, f"{what}.{sname}: kind {skind} != {rkind}"
        if rdefault is inspect.Parameter.empty:
            assert sdefault is None, (
                f"{what}.{sname}: stub default {ast.unparse(sdefault)} "
                "but the runtime parameter is required"
            )
        elif rdefault is Ellipsis:
            # PyO3 could not render the default (e.g. a tuple); the parameter
            # must still be optional in the stub, but the value is opaque.
            assert sdefault is not None, (
                f"{what}.{sname}: runtime parameter has a default, stub marks "
                "it required"
            )
        else:
            assert sdefault is not None, (
                f"{what}.{sname}: runtime default {rdefault!r} missing in the stub"
            )
            stub_value = ast.literal_eval(sdefault)
            assert stub_value == rdefault, (
                f"{what}.{sname}: stub default {stub_value!r} != runtime {rdefault!r}"
            )


def test_all_matches_runtime():
    tree = _tree()
    stub_defs = _top_level_defs(tree)
    stub_all = _stub_all(tree)
    runtime_all = [name for name in fpl.__all__ if name != "__doc__"]
    assert set(stub_all) == set(runtime_all), (
        f"__all__ differs: stub-only {set(stub_all) - set(runtime_all)}, "
        f"runtime-only {set(runtime_all) - set(stub_all)}"
    )
    defined = set(stub_defs[0]) | set(stub_defs[1])
    assert set(stub_all) == defined, (
        f"__all__ lists undefined names {set(stub_all) - defined}; "
        f"defined but unlisted {defined - set(stub_all)}"
    )


def test_module_functions_match_runtime():
    _, stub_funcs = _top_level_defs(_tree())
    for name, node in stub_funcs.items():
        assert hasattr(fpl, name), f"stub defines {name}, runtime has no such function"
        _compare_signatures(name, node, getattr(fpl, name))


def test_classes_match_runtime():
    stub_classes, _ = _top_level_defs(_tree())
    for name, cls_node in stub_classes.items():
        runtime_cls = getattr(fpl, name, None)
        assert runtime_cls is not None, f"stub defines class {name}, missing at runtime"

        methods, attrs = _stub_members(cls_node)
        runtime_public = {m for m in dir(runtime_cls) if not m.startswith("_")}
        stub_public = {m for m in methods if not m.startswith("_")} | attrs
        assert stub_public == runtime_public, (
            f"{name}: stub-only members {stub_public - runtime_public}, "
            f"runtime-only members {runtime_public - stub_public}"
        )

        for attr in attrs:
            assert hasattr(runtime_cls, attr), f"{name}.{attr}: attribute missing"
            assert not callable(getattr(runtime_cls, attr)), (
                f"{name}.{attr}: stub declares an attribute but runtime has a method"
            )

        for method, fn_node in methods.items():
            assert hasattr(runtime_cls, method), f"{name}.{method}: method missing"
            runtime_obj = (
                runtime_cls if method == "__init__" else getattr(runtime_cls, method)
            )
            _compare_signatures(f"{name}.{method}", fn_node, runtime_obj)


def test_installed_stub_is_current():
    installed = pathlib.Path(fpl.__file__).with_name("__init__.pyi")
    assert installed.exists(), (
        f"{installed} is missing: run `pixi run build` (maturin installs the "
        "root fastpiclef.pyi as fastpiclef/__init__.pyi)"
    )
    assert installed.read_text() == STUB_PATH.read_text(), (
        "the installed stub differs from fastpiclef.pyi: run `pixi run build`"
    )
    assert installed.with_name("py.typed").exists(), (
        "py.typed marker missing next to the installed stub"
    )


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
