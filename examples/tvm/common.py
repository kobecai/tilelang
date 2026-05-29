from __future__ import annotations

import textwrap
import sys
import importlib
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def print_section(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def print_obj(obj: Any, title: str | None = None) -> None:
    if title:
        print_section(title)
    if hasattr(obj, "script"):
        print(obj.script())
    else:
        print(obj)


def _candidate_tvm_python_paths(tilelang_module=None) -> list[str]:
    roots: list[Path] = []
    if tilelang_module is not None and getattr(tilelang_module, "__file__", None):
        tilelang_root = Path(tilelang_module.__file__).resolve().parent
        roots.extend([tilelang_root, tilelang_root.parent])

    roots.append(REPO_ROOT)

    paths: list[str] = []
    seen: set[str] = set()
    for root in roots:
        tvm_python = root / "3rdparty" / "tvm" / "python"
        if tvm_python.is_dir():
            path = str(tvm_python)
            if path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def _prepend_to_sys_path(paths: list[str]) -> None:
    for path in reversed(paths):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def import_tvm():
    try:
        tilelang = importlib.import_module("tilelang")
    except ModuleNotFoundError as err:
        if err.name == "torch":
            raise SystemExit(
                textwrap.dedent(
                    """
                    TileLang should bootstrap TVM before these examples import `tvm`,
                    but this environment is missing `torch`, which TileLang preloads
                    before loading its runtime libraries.

                    Install the normal TileLang Python dependencies, then rerun this
                    example from the repository root.
                    """
                ).strip()
            ) from err
        raise

    _prepend_to_sys_path(_candidate_tvm_python_paths(tilelang))

    try:
        return importlib.import_module("tvm")
    except ModuleNotFoundError as tvm_err:
        raise SystemExit(
            textwrap.dedent(
                """
                TVM is not importable after importing TileLang.

                Check that TileLang was installed with its bundled TVM Python files
                and native libraries. In a source checkout, rebuild/install with
                `pip install .`, then rerun this example from the repository root.
                """
            ).strip()
        ) from tvm_err


def import_tilelang():
    try:
        import tilelang
        import tilelang.language as T

        return tilelang, T
    except ModuleNotFoundError as err:
        if err.name == "torch":
            raise SystemExit(
                textwrap.dedent(
                    """
                    This example needs TileLang's full Python import path. The current
                    environment is missing `torch`, which TileLang preloads before
                    loading its TVM runtime libraries.
                    """
                ).strip()
            ) from err
        raise


def tir_namespace(tvm):
    if hasattr(tvm, "tirx"):
        return tvm.tirx
    return tvm.tir


def schedule_namespace(tvm):
    if hasattr(tvm, "s_tir"):
        return tvm.s_tir
    return tvm.tir


def make_ir_module(tvm, prim_func, name: str):
    attrs = getattr(prim_func, "attrs", None)
    global_symbol = attrs.get("global_symbol") if attrs else None
    return tvm.IRModule({str(global_symbol or name): prim_func})


def apply_available_passes(mod, passes):
    for pass_name, pass_factory in passes:
        if pass_factory is None:
            print(f"skip {pass_name}: not available in this TVM build")
            continue
        print(f"apply {pass_name}")
        mod = pass_factory()(mod)
    return mod
