from __future__ import annotations

import ast
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace


EXAMPLE_FILES = [
    "01_ir_module_primfunc.py",
    "02_schedule_sblock.py",
    "03_pass_context_target.py",
    "04_tilelang_tile_ops.py",
    "05_tir_transformation_tutorial.py",
]


def test_learning_examples_exist_and_parse() -> None:
    root = Path(__file__).resolve().parent
    expected_files = ["README.md", "common.py", *EXAMPLE_FILES]

    missing = [name for name in expected_files if not (root / name).exists()]
    assert missing == []

    for name in ["common.py", *EXAMPLE_FILES]:
        ast.parse((root / name).read_text(encoding="utf-8"), filename=name)


def test_readme_mentions_each_example() -> None:
    root = Path(__file__).resolve().parent
    readme = (root / "README.md").read_text(encoding="utf-8")

    for name in EXAMPLE_FILES:
        assert name in readme


def test_common_finds_installed_tilelang_tvm_python_path(tmp_path) -> None:
    root = Path(__file__).resolve().parent
    spec = spec_from_file_location("tvm_example_common", root / "common.py")
    assert spec is not None and spec.loader is not None
    common = module_from_spec(spec)
    spec.loader.exec_module(common)

    fake_tilelang_root = tmp_path / "site-packages" / "tilelang"
    expected = fake_tilelang_root / "3rdparty" / "tvm" / "python"
    expected.mkdir(parents=True)
    fake_tilelang = SimpleNamespace(__file__=str(fake_tilelang_root / "__init__.py"))

    paths = common._candidate_tvm_python_paths(fake_tilelang)
    assert str(expected.resolve()) in paths
