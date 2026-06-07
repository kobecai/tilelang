from __future__ import annotations

# pyright: reportMissingImports=false, reportInvalidTypeForm=false, reportCallIssue=false

import argparse
import importlib
import json
import re
import shutil
import sys
import textwrap
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


THIS_DIR = Path(__file__).resolve().parent
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
    "00_public_dumpir_fallback",
    "00_frontend_primfunc",
    "01_frontend_primfunc",
    "01_kernel_params",
    "02_kernel_params",
    "02_after_prelower_semantic_check",
    "03_bind_target",
}


class StopAfterSelectedPass(Exception):
    def __init__(self, title: str) -> None:
        super().__init__(title)
        self.title = title


@dataclass(frozen=True)
class CompactSelection:
    selected_passes: set[str]
    dump_all_passes: bool
    stop_after_pass: str | None
    explicit_selectors: list[str]


@dataclass
class Artifact:
    step: int
    file: str
    title: str
    note: str


def safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception as err:
        return f"<repr failed: {type(err).__name__}: {err}>"


def load_verbose_inspector() -> Any:
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    return importlib.import_module("inspect_quickstart_pipeline")


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

    return CompactSelection(
        selected_passes=selected_passes,
        dump_all_passes=dump_all_passes,
        stop_after_pass=stop_after_pass,
        explicit_selectors=explicit_selectors,
    )


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
        include_ir_generator: bool,
    ) -> None:
        self.out_dir = out_dir
        self.selection = selection
        self.print_ir = print_ir
        self.all_txt_line_budget = max(200, all_txt_line_budget)
        self.combined_tail_lines = max(0, combined_tail_lines)
        self.include_ir_generator = include_ir_generator
        self.step = 0
        self.dumped_ir_generator = False
        self.dumped_ir_generator_runtime = False
        self.artifacts: list[Artifact] = []
        self.executed_passes: list[str] = []
        self.skipped_titles: list[str] = []

    def should_dump_title(self, title: str) -> bool:
        if title.startswith("error_") or title == "pipeline_failure":
            return True
        if self.include_ir_generator and title.startswith("ir_generator_"):
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

    def _skipped_path(self, title: str, suffix: str) -> Path:
        safe_title = re.sub(r"[^0-9A-Za-z_.-]+", "_", title).strip("_").lower() or "artifact"
        return self.out_dir / f"skipped_{safe_title}{suffix}"

    def _record(self, path: Path, title: str, note: str) -> None:
        self.artifacts.append(Artifact(self.step - 1, path.name, title, note))

    def dump_text(self, title: str, text: str, note: str, suffix: str = ".txt") -> Path:
        if not self.should_dump_title(title):
            self.skipped_titles.append(title)
            print(f"[skip dump] {title}")
            return self._skipped_path(title, suffix)

        path = self._next_path(title, suffix)
        path.write_text(text, encoding="utf-8")
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
            return self._skipped_path(title, ".py")

        body = module.script() if hasattr(module, "script") else str(module)
        header = textwrap.dedent(
            f"""
            # Step {self.step}: {title}
            # {note}

            """
        ).lstrip()
        return self.dump_text(title, header + body + "\n", note, suffix=".py")

    def _compact_config(self, config: dict[str, Any]) -> dict[str, Any]:
        keys = [
            "kernel",
            "kernel_description",
            "tir_kwargs",
            "target",
            "arch",
            "target_host",
            "mode",
            "stopped_after_pass",
            "missing_tirx",
            "dump_ir_dir",
            "error_type",
            "error",
        ]
        result = {key: config[key] for key in keys if key in config}
        selection = self.selection
        result.update(
            {
                "baseline_outputs": sorted(BASELINE_TITLES),
                "dump_lowering_passes": "all" if selection.dump_all_passes else sorted(selection.selected_passes),
                "explicit_pass_selectors": selection.explicit_selectors,
                "stop_after_pass": selection.stop_after_pass,
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
            "Generated by `examples/inspect_quickstart_pipeline_compact.py`.",
            "",
            "This learning-oriented view keeps baseline frontend artifacts and selects lowering pass dumps with `--dump-lowering-pass`.",
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
        compact = self._compact_config(config)
        lines = [
            "# TileLang Compact Pipeline Artifacts Combined Dump",
            "",
            "Generated by `examples/inspect_quickstart_pipeline_compact.py`.",
            "Individual artifact files contain full selected dumps; this file is intentionally line-budgeted.",
            "",
            "## Compact Config",
            "",
            "```json",
            json.dumps(compact, indent=2, sort_keys=True, default=safe_repr),
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


def make_compact_run_pass() -> Callable[..., Any]:
    def run_pass(
        dumper: CompactArtifactDumper,
        title: str,
        module: Any,
        transform_factory: Callable[[], Any],
        note: str,
    ) -> Any:
        print(f"running {title}")
        try:
            next_module = transform_factory()(module)
        except Exception as err:
            dumper.dump_text(
                f"error_{title}",
                f"Pass `{title}` failed.\n\n{type(err).__name__}: {err}\n",
                f"Failure while running {title}.",
                suffix=".txt",
            )
            raise

        dumper.executed_passes.append(title)
        if dumper.should_dump_title(title):
            dumper.dump_module(title, next_module, note)
        else:
            dumper.skipped_titles.append(title)
            print(f"[skip dump] {title}")

        if title == dumper.selection.stop_after_pass:
            raise StopAfterSelectedPass(title)
        return next_module

    return run_pass


def add_common_kernel_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--kernel", choices=KERNEL_CHOICES, default="issue_2307", help="Kernel spec to inspect.")
    parser.add_argument("--M", type=int, default=1024, help="quickstart_matmul only: M dimension.")
    parser.add_argument("--N", type=int, default=1024, help="quickstart_matmul only: N dimension.")
    parser.add_argument("--K", type=int, default=1024, help="quickstart_matmul only: K dimension.")
    parser.add_argument("--block-M", type=int, default=128, dest="block_M", help="quickstart_matmul only: M tile size.")
    parser.add_argument("--block-N", type=int, default=128, dest="block_N", help="quickstart_matmul only: N tile size.")
    parser.add_argument("--block-K", type=int, default=32, dest="block_K", help="quickstart_matmul only: K tile size.")
    parser.add_argument("--arch", default="sm_80", help="CUDA architecture to encode in the target, e.g. sm_80 or sm_90.")
    parser.add_argument("--target-host", default=None, help="Optional host target. Defaults to llvm if enabled, otherwise c.")
    parser.add_argument(
        "--tilelang-source-root",
        default=None,
        help="Optional TileLang source checkout root used before imports to find vendored tvm.tirx.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump a compact, learning-oriented TileLang lowering view for selected passes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python examples/inspect_quickstart_pipeline_compact.py --clean
              python examples/inspect_quickstart_pipeline_compact.py --dump-lowering-pass 17_layout_inference --clean
              python examples/inspect_quickstart_pipeline_compact.py --dump-pass lower_tile_op --clean
              python examples/inspect_quickstart_pipeline_compact.py --dump-lowering-pass none --clean
              python examples/inspect_quickstart_pipeline_compact.py --dump-lowering-pass 52_split_host_device --stop-after-pass none --clean
            """
        ),
    )
    add_common_kernel_args(parser)
    parser.add_argument("--out-dir", default="debug/issue_2307_pipeline_compact", help="Directory for generated artifacts.")
    parser.add_argument("--clean", action="store_true", help="Remove the output directory before generating artifacts.")
    parser.add_argument("--print-ir", action="store_true", help="Also print selected artifacts to stdout.")
    parser.add_argument(
        "--pass-config-json",
        default=None,
        help="Optional JSON string or JSON file path with TileLang/TVM PassContext config.",
    )
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
        help=(
            "Stop after this pass. Defaults to the latest selected --dump-lowering-pass. "
            "Use 'none' to run the full pipeline while still dumping only selected passes."
        ),
    )
    parser.add_argument("--all-txt-line-budget", type=int, default=1000, help="Approximate line limit for OUT_DIR/all.txt.")
    parser.add_argument("--combined-tail-lines", type=int, default=40, help="Tail lines kept when a long artifact is truncated in all.txt.")
    parser.add_argument(
        "--dump-ir-generator",
        action="store_true",
        help="Also include TileLang eager IRGenerator source/metadata artifacts. Off by default to keep all.txt compact.",
    )
    parser.add_argument(
        "--enable-dump-ir-instrument",
        action="store_true",
        help="Also enable TVM DumpIR instrument into OUT_DIR/tvm_dump_ir for comparison.",
    )
    parser.add_argument("--with-codegen", dest="skip_codegen", action="store_false", help="Continue through CUDA source generation when the pipeline is not stopped early.")
    parser.set_defaults(skip_codegen=True)
    parser.add_argument(
        "--require-tirx",
        action="store_true",
        help="Fail instead of falling back to public DumpIR when tvm.tirx is unavailable.",
    )
    parser.add_argument("--list-passes", action="store_true", help="Print selectable lowering pass names and exit.")
    return parser.parse_args()


def finalize_stopped_run(verbose: Any, args: argparse.Namespace, dumper: CompactArtifactDumper, pass_configs: dict[str, Any], stopped_after: str) -> None:
    target_host = verbose.Target(verbose.canon_target_host(verbose.make_cuda_target(args.arch), args.target_host))
    verbose.finalize_artifacts(
        dumper,
        verbose.manifest_config(
            args,
            target_host,
            pass_configs,
            mode="compact_manual_pass_pipeline_stopped",
            stopped_after_pass=stopped_after,
        ),
    )


def load_pass_configs(verbose: Any, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    pass_configs = verbose.load_pass_configs(args.pass_config_json)
    if args.enable_dump_ir_instrument:
        pass_configs[verbose.pass_config_key("TL_ENABLE_DUMP_IR", "tl.enable_dump_ir")] = True
        pass_configs[verbose.pass_config_key("TL_DUMP_IR_DIR", "tl.dump_ir_path")] = str(out_dir / "tvm_dump_ir")
    return pass_configs


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

    verbose = load_verbose_inspector()
    pass_configs = load_pass_configs(verbose, args, out_dir)
    dumper = CompactArtifactDumper(
        out_dir,
        selection=selection,
        print_ir=args.print_ir,
        all_txt_line_budget=args.all_txt_line_budget,
        combined_tail_lines=args.combined_tail_lines,
        include_ir_generator=args.dump_ir_generator,
    )

    original_run_pass = verbose.run_pass
    verbose.run_pass = make_compact_run_pass()
    try:
        try:
            verbose.dump_cuda_pipeline(args, dumper, pass_configs)
        except StopAfterSelectedPass as stop:
            print(f"stopped after {stop.title}")
            finalize_stopped_run(verbose, args, dumper, pass_configs, stop.title)
    except Exception as err:
        dumper.dump_text(
            "pipeline_failure",
            traceback.format_exc(),
            "Unhandled exception while dumping the compact pipeline; earlier selected artifacts are included above this failure record.",
        )
        target_host = verbose.Target(verbose.canon_target_host(verbose.make_cuda_target(args.arch), args.target_host))
        verbose.finalize_artifacts(
            dumper,
            verbose.manifest_config(
                args,
                target_host,
                pass_configs,
                mode="compact_failed",
                error_type=type(err).__name__,
                error=str(err),
            ),
        )
        raise
    finally:
        verbose.run_pass = original_run_pass

    print(f"\nDone. Open {out_dir / 'README.md'} to browse artifacts, or {out_dir / 'all.txt'} for the compact combined dump.")


if __name__ == "__main__":
    main()