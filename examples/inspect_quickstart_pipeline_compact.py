from __future__ import annotations

# pyright: reportMissingImports=false, reportInvalidTypeForm=false, reportCallIssue=false, reportRedeclaration=false

import argparse
import importlib
import importlib.util
import json
import os
import re
import shutil
import sys
import sysconfig
import textwrap
import traceback
from pathlib import Path
from typing import Any, Callable


def _preload_stdlib_inspect() -> None:
    current_dir = Path(__file__).resolve().parent
    existing_inspect = sys.modules.get("inspect")
    if existing_inspect is not None:
        existing_file = getattr(existing_inspect, "__file__", None)
        if not existing_file or Path(existing_file).resolve().parent != current_dir:
            return

    inspect_path = Path(sysconfig.get_path("stdlib")) / "inspect.py"
    spec = importlib.util.spec_from_file_location("inspect", inspect_path)
    if spec is None or spec.loader is None:
        return

    inspect_module = importlib.util.module_from_spec(spec)
    sys.modules["inspect"] = inspect_module
    spec.loader.exec_module(inspect_module)


_preload_stdlib_inspect()

from dataclasses import dataclass


KERNEL_CHOICES = ("issue_2307", "quickstart_matmul")

PASS_ORDER = [
    "03_bind_target",
    "04_let_inline",
    "05_add_wrapper_for_single_buf_store",
    "06_legalize_negative_index",
    "07_verify_parallel_loop",
    "08_inject_assumes",
    "09_tilelang_simplify",
    "10_layout_reducer",
    "11_producer_consumer_warp_specialized",
    "12_lower_blackwell_2sm",
    "13_if_stmt_binding",
    "14_pipeline_planning",
    "15_inject_software_pipeline",
    "16_simplify_after_pipeline",
    "17_layout_inference",
    "18_lower_tile_op",
    "19_lower_l2_persistent",
    "20_decouple_type_cast",
    "21_legalize_vectorized_loop",
    "22_legalize_safe_memory_access",
    "23_lower_access_ptr",
    "24_simplify_after_safe_access",
    "25_hoist_non_restrict_params",
    "26_lower_shared_tmem",
    "27_plan_update_buffer_allocation_location",
    "28_lower_shared_barrier",
    "29_fuse_mbarrier_arrive_expect_tx",
    "30_hoist_global_buffer_allocations",
    "31_lower_opaque_block",
    "32_simplify_after_opaque_block",
    "33_narrow_data_type_32",
    "34_flatten_buffer",
    "35_config_index_bitwidth",
    "36_tirx_simplify_after_flatten",
    "37_vectorize_loop",
    "38_storage_rewrite",
    "39_loop_unswitching",
    "40_unroll_loop",
    "41_renormalize_split_pattern",
    "42_tirx_simplify_after_unroll",
    "43_remove_no_op",
    "44_hoist_if_then_else",
    "45_verify_memory",
    "46_annotate_entry_func",
    "47_infer_fragment",
    "48_lower_thread_allreduce",
    "49_lower_ldg_stg",
    "50_lower_hopper_intrin",
    "51_annotate_device_regions",
    "52_split_host_device",
    "53_mark_cuda_sync_calls",
    "54_annotate_read_only_params",
    "55_merge_shared_memory_allocations",
    "56_inject_fence_proxy",
    "57_thread_sync_shared",
    "58_thread_sync_shared_dyn",
    "59_inject_tcgen05_fence",
    "60_merge_if_stmt",
    "61_annotate_warp_group_reg_alloc",
    "62_make_packed_api",
    "63_simplify_after_packed_api",
    "64_lower_device_kernel_launch",
    "65_persist_threadblock",
    "68_device_codegen_lower_intrin",
    "69_device_codegen_simplify",
    "70_device_codegen_hoist_broadcast_values",
]
PASS_INDEX = {name: index for index, name in enumerate(PASS_ORDER)}

BASELINE_TITLES = {
    "00_frontend_primfunc",
    "01_kernel_params",
    "02_after_prelower_semantic_check",
    "03_bind_target",
}


@dataclass(frozen=True)
class CompactSelection:
    selected_passes: set[str]
    dump_all_passes: bool
    stop_after_pass: str | None
    explicit_selectors: list[str]


@dataclass(frozen=True)
class KernelSpec:
    name: str
    jit_entry: Any
    tir_kwargs: Callable[[argparse.Namespace], dict[str, Any]]
    description: str


@dataclass(frozen=True)
class PassStep:
    name: str
    factory: Callable[[], Any]
    note: str
    enabled: Callable[[Any], bool] | None = None
    after_run: Callable[[Any], None] | None = None


@dataclass
class Artifact:
    step: int
    file: str
    title: str
    note: str


class StopAfterSelectedPass(Exception):
    def __init__(self, title: str) -> None:
        super().__init__(title)
        self.title = title


_SOURCE_CHECKOUT_ROOTS: list[Path] = []
_TVM_PYTHON_CANDIDATES: list[Path] = []
_TVM_PYTHON_ROOTS: list[Path] = []
_TVM_PYTHON_ROOTS_MISSING_TIRX: list[Path] = []
_SOURCE_PATHS_ADDED: list[Path] = []
_TVM_PACKAGE_PATHS_ADDED: list[Path] = []

tilelang: Any = None
T: Any = None
tvm: Any = None
nvcc: Any = None
PassConfigKey: Any = None
CallingConv: Any = None
Target: Any = None
tirx: Any = None
_TIRX_IMPORT_ERROR: Exception | None = None
_TIRX_INITIAL_IMPORT_ERROR: Exception | None = None
KERNEL_SPECS: dict[str, KernelSpec] = {}


def safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception as err:
        return f"<repr failed: {type(err).__name__}: {err}>"


def _normalize_source_root(candidate: Path) -> Path | None:
    candidate = candidate.expanduser().resolve()
    if (candidate / "tilelang" / "__init__.py").is_file():
        return candidate
    if candidate.name == "tilelang" and (candidate / "__init__.py").is_file():
        return candidate.parent
    return None


def _explicit_source_roots() -> list[Path]:
    raw_roots: list[str] = []
    env_roots = os.environ.get("TILELANG_SOURCE_ROOT")
    if env_roots:
        raw_roots.extend(root for root in env_roots.split(os.pathsep) if root)

    index = 1
    while index < len(sys.argv):
        arg = sys.argv[index]
        if arg == "--tilelang-source-root" and index + 1 < len(sys.argv):
            raw_roots.append(sys.argv[index + 1])
            index += 2
            continue
        if arg.startswith("--tilelang-source-root="):
            raw_roots.append(arg.split("=", 1)[1])
        index += 1

    roots = []
    for raw_root in raw_roots:
        root = _normalize_source_root(Path(raw_root))
        if root is not None and root not in roots:
            roots.append(root)
    return roots


def _source_checkout_roots() -> list[Path]:
    roots = _explicit_source_roots()
    for candidate in Path(__file__).resolve().parents:
        root = _normalize_source_root(candidate)
        if root is not None and root not in roots:
            roots.append(root)
    return roots


def _prepend_existing_path(path: Path) -> bool:
    path_str = str(path)
    if path.is_dir() and path_str not in sys.path:
        sys.path.insert(0, path_str)
        return True
    return False


def _installed_tilelang_tvm_python_roots() -> list[Path]:
    try:
        spec = importlib.util.find_spec("tilelang")
    except Exception:
        return []
    locations = getattr(spec, "submodule_search_locations", None) if spec is not None else None
    if not locations:
        return []
    candidates = []
    for location in locations:
        tvm_python = Path(location) / "3rdparty" / "tvm" / "python"
        if (tvm_python / "tvm" / "__init__.py").is_file():
            candidates.append(tvm_python)
    return candidates


def _prepend_source_checkout_root() -> None:
    global _SOURCE_CHECKOUT_ROOTS, _TVM_PYTHON_CANDIDATES, _TVM_PYTHON_ROOTS
    global _TVM_PYTHON_ROOTS_MISSING_TIRX, _SOURCE_PATHS_ADDED

    _SOURCE_CHECKOUT_ROOTS = _source_checkout_roots()
    paths: list[Path] = []
    tvm_python_roots: list[Path] = []
    tvm_python_candidates: list[Path] = []
    for root in _SOURCE_CHECKOUT_ROOTS:
        paths.append(root)
        tvm_python = root / "3rdparty" / "tvm" / "python"
        if (tvm_python / "tvm" / "__init__.py").is_file() and tvm_python not in tvm_python_roots:
            tvm_python_roots.append(tvm_python)
        if (tvm_python / "tvm" / "tirx" / "__init__.py").is_file() and tvm_python not in tvm_python_candidates:
            paths.append(tvm_python)
            tvm_python_candidates.append(tvm_python)

    for tvm_python in _installed_tilelang_tvm_python_roots():
        if tvm_python not in tvm_python_roots:
            tvm_python_roots.append(tvm_python)
        if (tvm_python / "tvm" / "tirx" / "__init__.py").is_file() and tvm_python not in tvm_python_candidates:
            paths.append(tvm_python)
            tvm_python_candidates.append(tvm_python)

    _TVM_PYTHON_ROOTS = tvm_python_roots
    _TVM_PYTHON_CANDIDATES = tvm_python_candidates
    _TVM_PYTHON_ROOTS_MISSING_TIRX = [path for path in tvm_python_roots if path not in tvm_python_candidates]
    for path in reversed(paths):
        if _prepend_existing_path(path):
            _SOURCE_PATHS_ADDED.append(path)


def _extend_imported_tvm_package_path() -> None:
    tvm_module = sys.modules.get("tvm")
    tvm_path = getattr(tvm_module, "__path__", None)
    if tvm_path is None:
        return
    for tvm_python in _TVM_PYTHON_CANDIDATES:
        tvm_package_path = tvm_python / "tvm"
        if not (tvm_package_path / "tirx" / "__init__.py").is_file():
            continue
        path_str = str(tvm_package_path)
        if path_str not in tvm_path:
            tvm_path.append(path_str)
            _TVM_PACKAGE_PATHS_ADDED.append(tvm_package_path)


def issue_2307_kernel_source():
    @T.prim_func
    def kernel(score: T.Tensor((16,), T.float)):
        with T.Kernel(1, threads=32) as _:
            score_fragment = T.alloc_fragment((32,), T.float)
            for i in T.Parallel(32):
                if i < 16:
                    score_fragment[i] = score[i]
                else:
                    score_fragment[i] = T.infinity(T.float)
            for i in T.Parallel(32):
                if i < 16:
                    T.device_assert(T.isfinite(score_fragment[i]))

    return kernel


def quickstart_matmul_source(A, B, block_M: int, block_N: int, block_K: int):
    M, N, K = T.const("M, N, K")
    dtype = T.float16
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)

        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            T.copy(B[ko * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        for i, j in T.Parallel(block_M, block_N):
            C_local[i, j] = T.max(C_local[i, j], 0)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


def issue_2307_tir_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    del args
    return {}


def quickstart_matmul_tir_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "M": args.M,
        "N": args.N,
        "K": args.K,
        "block_M": args.block_M,
        "block_N": args.block_N,
        "block_K": args.block_K,
    }


def _initialize_kernel_specs() -> None:
    global KERNEL_SPECS
    if KERNEL_SPECS:
        return
    KERNEL_SPECS = {
        "issue_2307": KernelSpec(
            name="issue_2307",
            jit_entry=tilelang.jit(issue_2307_kernel_source),
            tir_kwargs=issue_2307_tir_kwargs,
            description="Minimal repro from tile-ai/tilelang#2307: fragment layout with alloc_fragment(32, float).",
        ),
        "quickstart_matmul": KernelSpec(
            name="quickstart_matmul",
            jit_entry=tilelang.jit(quickstart_matmul_source),
            tir_kwargs=quickstart_matmul_tir_kwargs,
            description="Original examples/quickstart.py-style matmul pipeline inspector kernel.",
        ),
    }


def initialize_tilelang_environment() -> None:
    global tilelang, T, tvm, nvcc, PassConfigKey, CallingConv, Target
    global tirx, _TIRX_IMPORT_ERROR, _TIRX_INITIAL_IMPORT_ERROR

    if tilelang is not None:
        return

    _prepend_source_checkout_root()

    import tilelang as tilelang_module
    import tilelang.language as language_module
    from tilelang import tvm as tvm_module
    from tilelang.contrib import nvcc as nvcc_module
    from tilelang.transform import PassConfigKey as PassConfigKeyType
    from tvm.ir import CallingConv as CallingConvType
    from tvm.target import Target as TargetType

    tilelang = tilelang_module
    T = language_module
    tvm = tvm_module
    nvcc = nvcc_module
    PassConfigKey = PassConfigKeyType
    CallingConv = CallingConvType
    Target = TargetType

    try:
        tirx = importlib.import_module("tvm.tirx")
    except Exception as err:
        _extend_imported_tvm_package_path()
        try:
            tirx = importlib.import_module("tvm.tirx")
        except Exception as retry_err:
            tirx = None
            _TIRX_IMPORT_ERROR = retry_err
            _TIRX_INITIAL_IMPORT_ERROR = err
        else:
            _TIRX_IMPORT_ERROR = None
            _TIRX_INITIAL_IMPORT_ERROR = err
    else:
        _TIRX_IMPORT_ERROR = None
        _TIRX_INITIAL_IMPORT_ERROR = None

    _initialize_kernel_specs()


def selector_key(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", value.lower())


def pass_suffix(pass_name: str) -> str:
    return re.sub(r"^\d+_", "", pass_name)


def pass_keys(pass_name: str) -> set[str]:
    number = pass_name.split("_", 1)[0]
    return {
        selector_key(pass_name),
        selector_key(pass_suffix(pass_name)),
        selector_key(number),
        selector_key(str(int(number))),
    }


def split_pass_values(values: list[str] | None) -> list[str]:
    tokens: list[str] = []
    for value in values or []:
        tokens.extend(token.strip() for token in value.split(","))
    return [token for token in tokens if token]


def resolve_pass_name(raw_value: str) -> str:
    key = selector_key(raw_value)
    if not key:
        raise ValueError("empty pass selector")
    matches = [name for name in PASS_ORDER if key in pass_keys(name)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"ambiguous pass selector `{raw_value}`: {', '.join(matches)}")
    raise ValueError(f"unknown pass selector `{raw_value}`; run with --list-passes")


def build_selection(args: argparse.Namespace) -> CompactSelection:
    explicit_selectors = split_pass_values(args.dump_lowering_pass)
    raw_passes = explicit_selectors or ["05_add_wrapper_for_single_buf_store"]
    normalized = {selector_key(value) for value in raw_passes}

    dump_all_passes = "all" in normalized
    selected_passes: set[str] = set()
    if "none" not in normalized and not dump_all_passes:
        selected_passes = {resolve_pass_name(value) for value in raw_passes}

    if args.stop_after_pass is not None:
        stop_key = selector_key(args.stop_after_pass)
        stop_after_pass = None if stop_key in {"none", "off", "false", "full"} else resolve_pass_name(args.stop_after_pass)
    elif dump_all_passes:
        stop_after_pass = None
    elif selected_passes:
        stop_after_pass = max(selected_passes, key=lambda name: PASS_INDEX[name])
    else:
        stop_after_pass = "03_bind_target"

    if stop_after_pass is not None:
        later_passes = [name for name in selected_passes if PASS_INDEX[name] > PASS_INDEX[stop_after_pass]]
        if later_passes:
            raise ValueError(
                f"--stop-after-pass {stop_after_pass} is before selected pass output(s): {', '.join(sorted(later_passes))}"
            )

    return CompactSelection(selected_passes, dump_all_passes, stop_after_pass, explicit_selectors)


def budgeted_lines(lines: list[str], budget: int, artifact_file: str, tail_lines: int) -> list[str]:
    if budget <= 0:
        return []
    if len(lines) <= budget:
        return lines
    if budget <= 5:
        return lines[:budget]
    marker = f"... <{len(lines) - budget + 1} lines omitted; see {artifact_file}> ..."
    if tail_lines <= 0 or budget <= tail_lines + 6:
        return lines[: budget - 1] + [marker]
    head_count = budget - tail_lines - 1
    return lines[:head_count] + [marker] + lines[-tail_lines:]


class CompactArtifactDumper:
    def __init__(
        self,
        out_dir: Path,
        selection: CompactSelection,
        print_ir: bool,
        all_txt_line_budget: int,
        combined_tail_lines: int,
    ) -> None:
        self.out_dir = out_dir
        self.selection = selection
        self.print_ir = print_ir
        self.all_txt_line_budget = max(200, all_txt_line_budget)
        self.combined_tail_lines = max(0, combined_tail_lines)
        self.step = 0
        self.artifacts: list[Artifact] = []
        self.executed_passes: list[str] = []
        self.skipped_titles: list[str] = []

    def should_dump_title(self, title: str) -> bool:
        if title.startswith("error_") or title == "pipeline_failure":
            return True
        if title in BASELINE_TITLES:
            return True
        if self.selection.dump_all_passes and title in PASS_INDEX:
            return True
        return title in self.selection.selected_passes

    def _next_path(self, title: str, suffix: str) -> Path:
        safe_title = re.sub(r"[^0-9A-Za-z_.-]+", "_", title).strip("_").lower()
        safe_title = re.sub(r"^\d+_", "", safe_title) or "artifact"
        path = self.out_dir / f"{self.step:02d}_{safe_title}{suffix}"
        self.step += 1
        return path

    def _record(self, path: Path, title: str, note: str) -> None:
        self.artifacts.append(Artifact(self.step - 1, path.name, title, note))

    def dump_text(self, title: str, text: str, note: str, suffix: str = ".txt") -> Path:
        if not self.should_dump_title(title):
            self.skipped_titles.append(title)
            print(f"[skip dump] {title}")
            return self.out_dir / f"skipped_{title}{suffix}"

        path = self._next_path(title, suffix)
        path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        self._record(path, title, note)
        if self.print_ir:
            print(f"\n===== {path.name}: {title} =====\n{text}")
        else:
            print(f"[{self.step - 1:02d}] wrote {path}")
        return path

    def dump_module(self, title: str, module: Any, note: str) -> Path:
        if not self.should_dump_title(title):
            self.skipped_titles.append(title)
            print(f"[skip dump] {title}")
            return self.out_dir / f"skipped_{title}.py"

        body = module.script() if hasattr(module, "script") else str(module)
        header = textwrap.dedent(
            f"""
            # Step {self.step}: {title}
            # {note}

            """
        ).lstrip()
        return self.dump_text(title, header + body + "\n", note, suffix=".py")

    def _compact_config(self, config: dict[str, Any]) -> dict[str, Any]:
        keys = ["kernel", "kernel_description", "tir_kwargs", "target", "arch", "target_host", "mode", "stopped_after_pass", "error_type", "error"]
        result = {key: config[key] for key in keys if key in config}
        result.update(
            {
                "baseline_outputs": sorted(BASELINE_TITLES),
                "dump_lowering_passes": "all" if self.selection.dump_all_passes else sorted(self.selection.selected_passes),
                "explicit_pass_selectors": self.selection.explicit_selectors,
                "stop_after_pass": self.selection.stop_after_pass,
                "executed_lowering_passes": self.executed_passes,
                "skipped_dumps": self.skipped_titles,
                "all_txt_line_budget": self.all_txt_line_budget,
            }
        )
        return result

    def dump_manifest(self, config: dict[str, Any]) -> Path:
        compact = self._compact_config(config)
        lines = [
            "# TileLang Compact Pipeline Artifacts",
            "",
            "Generated by standalone compact inspector.",
            "",
            "This view keeps baseline frontend artifacts and selects lowering pass dumps with `--dump-lowering-pass`.",
            f"`all.txt` is line-budgeted to about {self.all_txt_line_budget} lines; selected individual artifact files remain complete.",
            "",
            "## Config",
            "",
            "```json",
            json.dumps(compact, indent=2, sort_keys=True, default=safe_repr),
            "```",
            "",
            "## Files",
            "",
            "| Step | File | What to look for |",
            "| ---: | --- | --- |",
        ]
        for artifact in self.artifacts:
            note = artifact.note.replace("|", "\\|")
            lines.append(f"| {artifact.step:02d} | `{artifact.file}` | {note} |")
        path = self.out_dir / "README.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {path}")
        return path

    def dump_all_text(self, config: dict[str, Any]) -> Path:
        lines = [
            "# TileLang Compact Pipeline Artifacts Combined Dump",
            "",
            "Generated by standalone compact inspector.",
            "",
            "## Compact Config",
            "",
            "```json",
            json.dumps(self._compact_config(config), indent=2, sort_keys=True, default=safe_repr),
            "```",
            "",
            "## Artifacts",
            "",
        ]

        for index, artifact in enumerate(self.artifacts):
            block_header = [
                "=" * 96,
                f"Step {artifact.step:02d}: {artifact.title}",
                f"File: {artifact.file}",
                f"Note: {artifact.note}",
                "=" * 96,
                "",
            ]
            remaining_artifacts = len(self.artifacts) - index
            remaining_budget = self.all_txt_line_budget - len(lines)
            reserve_for_later = max(0, remaining_artifacts - 1) * 8
            content_budget = max(1, remaining_budget - reserve_for_later - len(block_header) - 1)
            path = self.out_dir / artifact.file
            try:
                content_lines = path.read_text(encoding="utf-8").splitlines()
            except Exception as err:
                content_lines = [f"<failed to read {artifact.file}: {type(err).__name__}: {err}>"]
            lines.extend(block_header)
            lines.extend(budgeted_lines(content_lines, content_budget, artifact.file, self.combined_tail_lines))
            lines.append("")
            if len(lines) >= self.all_txt_line_budget:
                break

        if len(lines) > self.all_txt_line_budget:
            lines = lines[: self.all_txt_line_budget - 1] + ["... <all.txt line budget reached; see individual artifact files> ..."]
        path = self.out_dir / "all.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {path} ({len(lines)} lines, budget {self.all_txt_line_budget})")
        return path


def module_location(module_name: str) -> str | None:
    module = sys.modules.get(module_name)
    if module is None:
        return None
    module_file = getattr(module, "__file__", None)
    if module_file:
        return str(module_file)
    module_path = getattr(module, "__path__", None)
    return safe_repr(list(module_path)) if module_path else None


def import_diagnostics() -> dict[str, Any]:
    tvm_path = getattr(tvm, "__path__", None)
    return {
        "python_executable": sys.executable,
        "script_path": str(Path(__file__).resolve()),
        "source_checkout_roots": [str(path) for path in _SOURCE_CHECKOUT_ROOTS],
        "tilelang_source_root_env": os.environ.get("TILELANG_SOURCE_ROOT"),
        "tvm_python_roots": [str(path) for path in _TVM_PYTHON_ROOTS],
        "tvm_python_candidates": [str(path) for path in _TVM_PYTHON_CANDIDATES],
        "tvm_python_roots_missing_tirx": [str(path) for path in _TVM_PYTHON_ROOTS_MISSING_TIRX],
        "source_paths_added": [str(path) for path in _SOURCE_PATHS_ADDED],
        "tvm_package_paths_added": [str(path) for path in _TVM_PACKAGE_PATHS_ADDED],
        "tilelang_file": module_location("tilelang"),
        "tilelang_version": getattr(tilelang, "__version__", None),
        "tvm_file": module_location("tvm"),
        "tvm_package_path": [str(path) for path in tvm_path] if tvm_path is not None else None,
        "tirx_file": module_location("tvm.tirx"),
        "tirx_initial_import_error": repr(_TIRX_INITIAL_IMPORT_ERROR) if _TIRX_INITIAL_IMPORT_ERROR is not None else None,
        "tirx_import_error": repr(_TIRX_IMPORT_ERROR) if _TIRX_IMPORT_ERROR is not None else None,
    }


def _resolve_tvm_transform(name: str) -> Callable[[], Any]:
    candidates = ("tvm.s_tir.transform", "tvm.tir.transform", "tvm.tirx.transform")
    for module_name in candidates:
        try:
            transform_module = importlib.import_module(module_name)
        except ImportError:
            continue
        transform_factory = getattr(transform_module, name, None)
        if transform_factory is not None:
            return transform_factory
    raise ImportError(f"Cannot find TVM transform `{name}` in {', '.join(candidates)}")


def tvm_transform(name: str) -> Callable[[], Any]:
    return lambda: _resolve_tvm_transform(name)()


def pass_config_key(name: str, default: str) -> str:
    key = getattr(PassConfigKey, name, None)
    if key is None:
        return default
    return key.value if hasattr(key, "value") else str(key)


def selected_kernel_spec(args: argparse.Namespace) -> KernelSpec:
    return KERNEL_SPECS[args.kernel]


def make_cuda_target(arch: str) -> Any:
    return Target({"kind": "cuda", "arch": arch})


def _pass_config_get(pass_ctx: Any, key: Any, default: Any = None) -> Any:
    key_value = key.value if isinstance(key, PassConfigKey) else key
    return pass_ctx.config.get(key_value, default)


def allow_vectorize(pass_ctx: Any | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return not bool(_pass_config_get(pass_ctx, "tirx.disable_vectorize", False))


def should_enable_aggressive_merge(pass_ctx: Any | None = None, target: Any | None = None) -> bool:
    del target
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(_pass_config_get(pass_ctx, PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE, False))


def should_force_let_inline(pass_ctx: Any | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(_pass_config_get(pass_ctx, PassConfigKey.TL_FORCE_LET_INLINE, False))


def should_enable_race_check(pass_ctx: Any | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return not bool(_pass_config_get(pass_ctx, PassConfigKey.TL_DISABLE_DATA_RACE_CHECK, False))


def should_disable_shared_memory_reuse(pass_ctx: Any | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(_pass_config_get(pass_ctx, PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE, False))


def _have_pdl(target: Any) -> bool:
    return bool(nvcc.have_pdl(target))


def _have_tma(target: Any | None = None) -> bool:
    have_tma = getattr(nvcc, "have_tma", None)
    return bool(have_tma(target)) if have_tma is not None else False


def allow_warp_specialized(pass_ctx: Any | None = None, target: Any | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    if target is None or target.kind.name != "cuda" or not _have_tma(target):
        return False
    return not bool(_pass_config_get(pass_ctx, "tl.disable_warp_specialized", False))


def module_has_tma(module: Any) -> bool:
    return any(func.attrs and func.attrs.get("tl.has_tma", False) for _, func in module.functions.items())


def is_cpu_device_backend(target: Any) -> bool:
    return target.kind.name == "c"


def _has_device_kernel_launch(attrs: Any) -> bool:
    return bool(attrs and "calling_conv" in attrs and attrs["calling_conv"] == CallingConv.DEVICE_KERNEL_LAUNCH)


def _is_device_call_c_device(func: Any) -> bool:
    attrs = func.attrs
    calling_conv = attrs.get("calling_conv", CallingConv.DEFAULT)
    is_cpacked = calling_conv == CallingConv.C_PACKED_FUNC
    if "target" in attrs and attrs["target"].kind.name == "c" and not is_cpacked:
        return True
    return _has_device_kernel_launch(attrs)


def _is_device_call(func: Any) -> bool:
    return _has_device_kernel_launch(func.attrs)


def get_device_call(is_device_c: bool = False) -> Callable[[Any], bool]:
    return _is_device_call_c_device if is_device_c else _is_device_call


def get_host_call(is_device_c: bool = False) -> Callable[[Any], bool]:
    return lambda func: not get_device_call(is_device_c)(func)


def canon_target_host(target: str | Any, target_host: str | Any | None) -> str | Any:
    del target
    return target_host or ("llvm" if tvm.runtime.enabled("llvm") else "c")


def extrac_params(func: Any) -> list[str]:
    params = []
    for var in func.params:
        if var in func.buffer_map:
            buffer = func.buffer_map[var]
            shape = ", ".join(str(dim) for dim in buffer.shape)
            params.append(f"{buffer.name}: Tensor([{shape}], {buffer.dtype})")
        else:
            params.append(f"{var}: {var.dtype}")
    return params


def PreLowerSemanticCheck(module: Any) -> None:
    pass_ctx = tilelang.transform.get_pass_context()
    if _pass_config_get(pass_ctx, PassConfigKey.TL_DISABLE_PRELOWER_SEMANTIC_CHECK, False):
        return
    if _pass_config_get(pass_ctx, PassConfigKey.TL_AST_PRINT_ENABLE, False):
        tilelang.analysis.ASTPrinter()(module)
    tilelang.analysis.NestedLoopChecker()(module)
    tilelang.analysis.FragmentLoopChecker()(module)


def _layout_visual_formats(pass_ctx: Any | None = None) -> list[str]:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    formats_value = _pass_config_get(pass_ctx, PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS, "")
    if not formats_value:
        return ["txt"]
    formats_str = str(formats_value).strip().lower()
    if formats_str == "all":
        return ["txt", "png", "pdf", "svg"]
    return [fmt.strip() for fmt in formats_str.split(",") if fmt.strip()]


def LayoutVisual(module: Any) -> None:
    pass_ctx = tilelang.transform.get_pass_context()
    if _pass_config_get(pass_ctx, PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE, False):
        tilelang.analysis.LayoutVisual(formats=_layout_visual_formats(pass_ctx))(module)


def build_pass_steps(tirx_module: Any, target: Any) -> list[PassStep]:
    return [
        PassStep("03_bind_target", lambda: tirx_module.transform.BindTarget(target), "Attaches the CUDA target to the PrimFunc so later passes can make target-specific choices."),
        PassStep("04_let_inline", tilelang.transform.LetInline, "Optional let inlining enabled by pass config.", lambda _module: should_force_let_inline()),
        PassStep("05_add_wrapper_for_single_buf_store", tilelang.transform.AddWrapperForSingleBufStore, "Normalizes single-buffer stores before later region and memory passes."),
        PassStep("06_legalize_negative_index", tilelang.transform.LegalizeNegativeIndex, "Canonicalizes negative indices into non-negative index expressions."),
        PassStep("07_verify_parallel_loop", tilelang.transform.VerifyParallelLoop, "Checks parallel loop legality; this is mostly validation unless it reports an error.", lambda _module: should_enable_race_check()),
        PassStep("08_inject_assumes", tilelang.transform.InjectAssumes, "Injects shape/bound assumptions that help the prover and simplifier."),
        PassStep("09_tilelang_simplify", tilelang.transform.Simplify, "Simplifies expressions while the IR still contains high-level TileLang constructs."),
        PassStep("10_layout_reducer", tilelang.transform.LayoutReducer, "Sets reducer layout metadata before the main layout inference pass."),
        PassStep("11_producer_consumer_warp_specialized", tilelang.cuda.transform.ProducerConsumerWarpSpecialized, "Optional Hopper+ warp specialization while tile ops are still high level.", lambda _module: allow_warp_specialized(target=target)),
        PassStep("12_lower_blackwell_2sm", tilelang.cuda.transform.LowerBlackwell2SM, "Handles Blackwell 2CTA/2SM GEMM annotations before LayoutInference."),
        PassStep("13_if_stmt_binding", tilelang.transform.IfStmtBinding, "Normalizes if/bind structure so software pipeline planning sees canonical bodies."),
        PassStep("14_pipeline_planning", tilelang.transform.PipelinePlanning, "Plans the software pipeline around the T.Pipelined loop."),
        PassStep("15_inject_software_pipeline", tilelang.transform.InjectSoftwarePipeline, "Rewrites the loop body into prologue/main/epilogue-style pipelined IR."),
        PassStep("16_simplify_after_pipeline", tilelang.transform.Simplify, "Cleans up expressions introduced by software pipeline injection."),
        PassStep("17_layout_inference", tilelang.transform.LayoutInference, "Infers fragment/shared/parallel-loop layouts that later tile-op lowering consumes.", after_run=LayoutVisual),
        PassStep("18_lower_tile_op", tilelang.transform.LowerTileOp, "Lowers tl.tileop.copy/gemm/reduce into lower-level TIR/intrinsic structure."),
        PassStep("19_lower_l2_persistent", tilelang.cuda.transform.LowerL2Persistent, "CUDA-specific lowering for L2 persistent mapping metadata."),
        PassStep("20_decouple_type_cast", tilelang.transform.DecoupleTypeCast, "Decouples type casts from vectorization constraints before vectorization."),
        PassStep("21_legalize_vectorized_loop", tilelang.transform.LegalizeVectorizedLoop, "Makes vectorized loops structurally legal for later lowering."),
        PassStep("22_legalize_safe_memory_access", tilelang.transform.LegalizeSafeMemoryAccess, "Adds predicates/guards for memory accesses that may be out of bounds at tile edges."),
        PassStep("23_lower_access_ptr", tilelang.transform.LowerAccessPtr, "Lowers TileLang pointer metadata ops to standard tvm_access_ptr-style nodes."),
        PassStep("24_simplify_after_safe_access", tilelang.transform.Simplify, "Simplifies duplicated conditions and expressions from access legalization."),
        PassStep("25_hoist_non_restrict_params", tilelang.transform.HoistNonRestrictParams, "Hoists root-block non-restrict annotations to function attrs when present."),
        PassStep("26_lower_shared_tmem", tilelang.cuda.transform.LowerSharedTmem, "Lowers shared.tmem allocations and initialization placement."),
        PassStep("27_plan_update_buffer_allocation_location", tilelang.transform.PlanAndUpdateBufferAllocationLocation, "Moves buffer allocations to planned locations after pipeline/barrier structure is known."),
        PassStep("28_lower_shared_barrier", tilelang.cuda.transform.LowerSharedBarrier, "Lowers TileLang shared barrier constructs."),
        PassStep("29_fuse_mbarrier_arrive_expect_tx", tilelang.cuda.transform.FuseMBarrierArriveExpectTx, "Fuses TMA mbarrier arrive/expect-tx when LowerTileOp produced TMA ops.", module_has_tma),
        PassStep("30_hoist_global_buffer_allocations", tilelang.transform.HoistGlobalBufferAllocations, "Hoists global buffer allocations to the appropriate scope."),
        PassStep("31_lower_opaque_block", tilelang.transform.LowerOpaqueBlock, "Lowers opaque TileLang blocks into standard TIR structure."),
        PassStep("32_simplify_after_opaque_block", tilelang.transform.Simplify, "Cleans up IR after opaque block lowering."),
        PassStep("33_narrow_data_type_32", lambda: tirx_module.transform.NarrowDataType(32), "Narrows index/data expressions where legal to 32-bit forms."),
        PassStep("34_flatten_buffer", tilelang.transform.FlattenBuffer, "Flattens multi-dimensional buffers and buffer indexing."),
        PassStep("35_config_index_bitwidth", tilelang.transform.ConfigIndexBitwidth, "Applies the configured index bitwidth after buffer flattening."),
        PassStep("36_tirx_simplify_after_flatten", tirx_module.transform.Simplify, "Runs TVM/TIRX simplification on flattened index expressions."),
        PassStep("37_vectorize_loop", lambda: tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize()), "Vectorizes eligible loops unless vectorization is disabled by pass config."),
        PassStep("38_storage_rewrite", tilelang.transform.StorageRewrite, "Rewrites storage allocation/reuse for local/shared temporaries."),
        PassStep("39_loop_unswitching", tilelang.transform.LoopUnswitching, "Hoists loop-invariant branches out of loops when profitable/legal."),
        PassStep("40_unroll_loop", tilelang.transform.UnrollLoop, "Applies loop unrolling decisions."),
        PassStep("41_renormalize_split_pattern", tvm_transform("RenormalizeSplitPattern"), "Normalizes split-loop patterns after unrolling and simplification."),
        PassStep("42_tirx_simplify_after_unroll", tirx_module.transform.Simplify, "Cleans up after loop transformations."),
        PassStep("43_remove_no_op", tirx_module.transform.RemoveNoOp, "Removes no-op statements introduced by previous rewrites."),
        PassStep("44_hoist_if_then_else", tvm_transform("HoistIfThenElse"), "Hoists suitable if-then-else constructs for cleaner lowered IR."),
        PassStep("45_verify_memory", tirx_module.transform.VerifyMemory, "Verifies memory access legality before host/device annotation."),
        PassStep("46_annotate_entry_func", tirx_module.transform.AnnotateEntryFunc, "Marks the entry function for later host/device split and codegen."),
        PassStep("47_infer_fragment", tvm_transform("InferFragment"), "Infers fragment-related metadata needed by thread-level reductions."),
        PassStep("48_lower_thread_allreduce", tilelang.transform.LowerThreadAllreduce, "Lowers thread-level allreduce constructs if present."),
        PassStep("49_lower_ldg_stg", tilelang.cuda.transform.LowerLDGSTG, "Lowers eligible global loads/stores to CUDA ldg/stg intrinsics."),
        PassStep("50_lower_hopper_intrin", tilelang.cuda.transform.LowerHopperIntrin, "Lowers Hopper-specific CUDA intrinsics when target/IR require them."),
        PassStep("51_annotate_device_regions", tilelang.transform.AnnotateDeviceRegions, "Marks device regions so SplitHostDevice can separate host and kernel functions."),
        PassStep("52_split_host_device", tilelang.transform.SplitHostDevice, "Splits the mixed module into host wrapper and device kernel functions."),
        PassStep("53_mark_cuda_sync_calls", lambda: tilelang.cuda.transform.MarkCudaSyncCalls(_have_pdl(target)), "Marks CUDA sync calls such as PDL sync/trigger when the architecture supports them."),
        PassStep("54_annotate_read_only_params", tilelang.transform.AnnotateReadOnlyParams, "Annotates read-only kernel parameters for downstream codegen."),
        PassStep("55_merge_shared_memory_allocations", lambda: tilelang.transform.MergeSharedMemoryAllocations(enable_aggressive_merge=should_enable_aggressive_merge(target=target), disable_reuse=should_disable_shared_memory_reuse()), "Merges shared memory allocations after SplitHostDevice, where device functions are explicit."),
        PassStep("56_inject_fence_proxy", tilelang.cuda.transform.InjectFenceProxy, "Injects CUDA async-proxy fences when the target and IR require them."),
        PassStep("57_thread_sync_shared", lambda: tilelang.transform.ThreadSync("shared"), "Inserts synchronization for shared memory hazards."),
        PassStep("58_thread_sync_shared_dyn", lambda: tilelang.transform.ThreadSync("shared.dyn"), "Inserts synchronization for dynamic shared memory hazards."),
        PassStep("59_inject_tcgen05_fence", tilelang.cuda.transform.InjectTcgen05Fence, "Injects conservative Blackwell tcgen05 fences when needed."),
        PassStep("60_merge_if_stmt", tilelang.transform.MergeIfStmt, "Merges compatible adjacent if statements after synchronization insertion."),
        PassStep("61_annotate_warp_group_reg_alloc", tilelang.cuda.transform.AnnotateWarpGroupRegAlloc, "Annotates warp-group register allocation for warp-specialized kernels.", lambda _module: allow_warp_specialized(target=target)),
        PassStep("62_make_packed_api", tilelang.transform.MakePackedAPI, "Converts the host-facing ABI to TVM packed API form."),
        PassStep("63_simplify_after_packed_api", tilelang.transform.Simplify, "Final simplification before lowering device kernel launch calls."),
        PassStep("64_lower_device_kernel_launch", tilelang.transform.LowerDeviceKernelLaunch, "Lowers abstract device kernel launch calls inside the host wrapper."),
        PassStep("65_persist_threadblock", tilelang.cuda.transform.PersistThreadblock, "Applies CUDA persistent threadblock transformation when requested by annotations."),
    ]


def run_pass(dumper: CompactArtifactDumper, title: str, module: Any, transform_factory: Callable[[], Any], note: str) -> Any:
    print(f"running {title}")
    try:
        next_module = transform_factory()(module)
    except Exception as err:
        dumper.dump_text(f"error_{title}", f"Pass `{title}` failed.\n\n{type(err).__name__}: {err}\n", f"Failure while running {title}.")
        raise

    dumper.executed_passes.append(title)
    dumper.dump_module(title, next_module, note)
    if title == dumper.selection.stop_after_pass:
        raise StopAfterSelectedPass(title)
    return next_module


def maybe_run_pass(dumper: CompactArtifactDumper, step: PassStep, module: Any) -> Any:
    enabled = True if step.enabled is None else bool(step.enabled(module))
    if not enabled:
        if step.name in dumper.selection.selected_passes:
            dumper.skipped_titles.append(f"{step.name} (disabled by pass config or target capability)")
        if step.name == dumper.selection.stop_after_pass:
            raise StopAfterSelectedPass(f"{step.name} (skipped)")
        return module

    module = run_pass(dumper, step.name, module, step.factory, step.note)
    if step.after_run is not None:
        step.after_run(module)
    return module


def selected_kernel_spec(args: argparse.Namespace) -> KernelSpec:
    return KERNEL_SPECS[args.kernel]


def build_initial_module(args: argparse.Namespace) -> tuple[Any, list[str]]:
    spec = selected_kernel_spec(args)
    prim_func = spec.jit_entry.get_tir(**spec.tir_kwargs(args))
    global_symbol = str(prim_func.attrs["global_symbol"])
    module = tvm.IRModule({global_symbol: prim_func})
    return module, extrac_params(prim_func)


def manifest_config(args: argparse.Namespace, target_host: Any, pass_configs: dict[str, Any], **extra: Any) -> dict[str, Any]:
    spec = selected_kernel_spec(args)
    config = {
        "kernel": spec.name,
        "kernel_description": spec.description,
        "tir_kwargs": spec.tir_kwargs(args),
        "target": "cuda",
        "arch": args.arch,
        "target_host": str(target_host),
        "pass_configs": pass_configs,
        "import_diagnostics": import_diagnostics(),
    }
    config.update(extra)
    return config


def finalize_artifacts(dumper: CompactArtifactDumper, config: dict[str, Any]) -> None:
    dumper.dump_manifest(config)
    dumper.dump_all_text(config)


def dump_cuda_pipeline(args: argparse.Namespace, dumper: CompactArtifactDumper, pass_configs: dict[str, Any]) -> None:
    if tirx is None:
        raise RuntimeError(f"Manual pass-by-pass mode requires tvm.tirx. Import diagnostics:\n{json.dumps(import_diagnostics(), indent=2, default=safe_repr)}")

    target = make_cuda_target(args.arch)
    target_host = Target(canon_target_host(target, args.target_host))
    target = Target(target, target_host)
    pass_instruments = []
    if args.enable_dump_ir_instrument:
        pass_instruments.append(tvm.ir.instrument.DumpIR(dump_dir=str(dumper.out_dir / "tvm_dump_ir")))
    pass_context = tilelang.transform.PassContext(opt_level=3, config=pass_configs, instruments=pass_instruments)

    with pass_context, target:
        module, params = build_initial_module(args)
        dumper.dump_module(
            "00_frontend_primfunc",
            module,
            "Eager JIT/AST builder output: high-level TIRX with Kernel launch, shared/fragment buffers, and tile ops.",
        )
        dumper.dump_text(
            "01_kernel_params",
            "\n".join(str(param) for param in params) + "\n",
            "Kernel parameters extracted from the frontend PrimFunc buffer map and scalar args.",
        )
        PreLowerSemanticCheck(module)
        dumper.dump_module(
            "02_after_prelower_semantic_check",
            module,
            "Semantic check is validation-only; this file is intentionally usually identical to the frontend module.",
        )

        for step in build_pass_steps(tirx, target):
            module = maybe_run_pass(dumper, step, module)

        host_filter = get_host_call(is_device_c=is_cpu_device_backend(target))
        device_filter = get_device_call(is_device_c=is_cpu_device_backend(target))
        host_module = tirx.transform.Filter(host_filter)(module)
        device_module = tirx.transform.Filter(device_filter)(module)
        dumper.dump_module("66_host_module_after_filter", host_module, "Host-side module selected from the lowered mixed module.")
        dumper.dump_module("67_device_module_after_filter", device_module, "Device-side module selected from the lowered mixed module.")

        codegen_ir = run_pass(dumper, "68_device_codegen_lower_intrin", device_module, tilelang.transform.LowerIntrin, "First device-codegen cleanup: lower remaining TileLang intrinsics.")
        codegen_ir = run_pass(dumper, "69_device_codegen_simplify", codegen_ir, tirx.transform.Simplify, "Simplifies the device-only module immediately before source generation.")
        codegen_ir = run_pass(dumper, "70_device_codegen_hoist_broadcast_values", codegen_ir, tilelang.transform.HoistBroadcastValues, "Hoists broadcast values in the device-only module before CUDA source emission.")

        if not args.skip_codegen:
            global_func = "target.build.tilelang_cuda_without_compile"
            if "cutedsl" in target.keys:
                global_func = "target.build.tilelang_cutedsl_without_compile"
            try:
                codegen_module = tvm.ffi.get_global_func(global_func)(codegen_ir, target)
                cuda_source = codegen_module.inspect_source()
            except Exception as err:
                dumper.dump_text("71_generated_cuda_source_error", f"CUDA source generation failed.\n\n{type(err).__name__}: {err}\n", "Final CUDA source generation failed; earlier IR dumps are still useful for pipeline inspection.")
            else:
                dumper.dump_text("71_generated_cuda_source", cuda_source, "Final CUDA C++ source emitted by TileLang codegen, without invoking NVCC.", suffix=".cu")

    finalize_artifacts(dumper, manifest_config(args, target_host, pass_configs, mode="manual_pass_pipeline"))


def load_pass_configs(raw_json: str | None, out_dir: Path, enable_dump_ir_instrument: bool) -> dict[str, Any]:
    if not raw_json:
        pass_configs: dict[str, Any] = {}
    else:
        path = Path(raw_json)
        pass_configs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else json.loads(raw_json)
    if enable_dump_ir_instrument:
        pass_configs[pass_config_key("TL_ENABLE_DUMP_IR", "tl.enable_dump_ir")] = True
        pass_configs[pass_config_key("TL_DUMP_IR_DIR", "tl.dump_ir_path")] = str(out_dir / "tvm_dump_ir")
    return pass_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump a compact, standalone TileLang lowering view for selected passes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python 2inspect.py --clean
              python 2inspect.py --dump-lowering-pass 17_layout_inference --clean
              python 2inspect.py --dump-pass lower_tile_op --clean
              python 2inspect.py --dump-lowering-pass none --clean
              python 2inspect.py --dump-lowering-pass 52_split_host_device --stop-after-pass none --clean
            """
        ),
    )
    parser.add_argument("--kernel", choices=KERNEL_CHOICES, default="issue_2307", help="Kernel spec to inspect.")
    parser.add_argument("--M", type=int, default=1024, help="quickstart_matmul only: M dimension.")
    parser.add_argument("--N", type=int, default=1024, help="quickstart_matmul only: N dimension.")
    parser.add_argument("--K", type=int, default=1024, help="quickstart_matmul only: K dimension.")
    parser.add_argument("--block-M", type=int, default=128, dest="block_M", help="quickstart_matmul only: M tile size.")
    parser.add_argument("--block-N", type=int, default=128, dest="block_N", help="quickstart_matmul only: N tile size.")
    parser.add_argument("--block-K", type=int, default=32, dest="block_K", help="quickstart_matmul only: K tile size.")
    parser.add_argument("--arch", default="sm_80", help="CUDA architecture to encode in the target, e.g. sm_80 or sm_90.")
    parser.add_argument("--target-host", default=None, help="Optional host target. Defaults to llvm if enabled, otherwise c.")
    parser.add_argument("--tilelang-source-root", default=None, help="Optional TileLang source checkout root used before imports to find vendored tvm.tirx.")
    parser.add_argument("--out-dir", default="debug/issue_2307_pipeline_compact", help="Directory for generated artifacts.")
    parser.add_argument("--clean", action="store_true", help="Remove the output directory before generating artifacts.")
    parser.add_argument("--print-ir", action="store_true", help="Also print selected artifacts to stdout.")
    parser.add_argument("--pass-config-json", default=None, help="Optional JSON string or JSON file path with TileLang/TVM PassContext config.")
    parser.add_argument(
        "--dump-lowering-pass",
        "--dump-pass",
        action="append",
        default=None,
        dest="dump_lowering_pass",
        help=(
            "Lowering pass to dump. Accepts numeric ids, full names, or suffixes, "
            "e.g. 05, 17_layout_inference, lower_tile_op, LayoutInference. Repeat or comma-separate. "
            "Defaults to 05_add_wrapper_for_single_buf_store. Use 'none' for only baseline anchors or 'all' for every pass."
        ),
    )
    parser.add_argument(
        "--stop-after-pass",
        "--stop-after",
        default=None,
        dest="stop_after_pass",
        help="Stop after this pass. Defaults to the latest selected pass. Use 'none' to run the full pipeline while dumping selected passes.",
    )
    parser.add_argument("--all-txt-line-budget", type=int, default=1000, help="Approximate line limit for OUT_DIR/all.txt.")
    parser.add_argument("--combined-tail-lines", type=int, default=40, help="Tail lines kept when a long artifact is truncated in all.txt.")
    parser.add_argument("--enable-dump-ir-instrument", action="store_true", help="Also enable TVM DumpIR instrument into OUT_DIR/tvm_dump_ir for comparison.")
    parser.add_argument("--with-codegen", dest="skip_codegen", action="store_false", help="Continue through CUDA source generation when the pipeline is not stopped early.")
    parser.set_defaults(skip_codegen=True)
    parser.add_argument("--list-passes", action="store_true", help="Print selectable lowering pass names and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_passes:
        print("Selectable lowering passes:")
        for pass_name in PASS_ORDER:
            print(f"  {pass_name}")
        return

    selection = build_selection(args)
    out_dir = Path(args.out_dir).expanduser().resolve()
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    initialize_tilelang_environment()
    pass_configs = load_pass_configs(args.pass_config_json, out_dir, args.enable_dump_ir_instrument)
    dumper = CompactArtifactDumper(out_dir, selection, args.print_ir, args.all_txt_line_budget, args.combined_tail_lines)

    try:
        try:
            dump_cuda_pipeline(args, dumper, pass_configs)
        except StopAfterSelectedPass as stop:
            print(f"stopped after {stop.title}")
            target_host = Target(canon_target_host(make_cuda_target(args.arch), args.target_host))
            finalize_artifacts(
                dumper,
                manifest_config(
                    args,
                    target_host,
                    pass_configs,
                    mode="compact_manual_pass_pipeline_stopped",
                    stopped_after_pass=stop.title,
                ),
            )
    except Exception as err:
        dumper.dump_text(
            "pipeline_failure",
            traceback.format_exc(),
            "Unhandled exception while dumping the compact pipeline; earlier selected artifacts are included above this failure record.",
        )
        target_host = Target(canon_target_host(make_cuda_target(args.arch), args.target_host))
        finalize_artifacts(
            dumper,
            manifest_config(args, target_host, pass_configs, mode="compact_failed", error_type=type(err).__name__, error=str(err)),
        )
        raise

    print(f"\nDone. Open {out_dir / 'README.md'} to browse artifacts, or {out_dir / 'all.txt'} for the compact combined dump.")


if __name__ == "__main__":
    main()