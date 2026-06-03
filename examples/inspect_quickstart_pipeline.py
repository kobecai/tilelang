from __future__ import annotations

# pyright: reportMissingImports=false, reportInvalidTypeForm=false, reportCallIssue=false, reportRedeclaration=false

import argparse
import importlib
import importlib.util
import json
import re
import shutil
import sys
import sysconfig
import textwrap
from pathlib import Path


def _preload_stdlib_inspect() -> None:
    """Avoid self-import when this standalone script is renamed to inspect.py."""
    if Path(__file__).name != "inspect.py":
        return

    current_file = Path(__file__).resolve()
    existing_inspect = sys.modules.get("inspect")
    if existing_inspect is not None:
        existing_file = getattr(existing_inspect, "__file__", None)
        if existing_file and Path(existing_file).resolve() != current_file:
            return

    inspect_path = Path(sysconfig.get_path("stdlib")) / "inspect.py"
    spec = importlib.util.spec_from_file_location("inspect", inspect_path)
    if spec is None or spec.loader is None:
        return

    inspect_module = importlib.util.module_from_spec(spec)
    sys.modules["inspect"] = inspect_module
    spec.loader.exec_module(inspect_module)


_preload_stdlib_inspect()


def _prepend_source_checkout_root() -> None:
    for candidate in (Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent):
        if (candidate / "tilelang" / "__init__.py").is_file() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            return


_prepend_source_checkout_root()

from dataclasses import dataclass
from typing import Any, Callable

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.contrib import nvcc
from tilelang.transform import PassConfigKey
from tvm.ir import CallingConv
from tvm.target import Target

try:
    tirx = importlib.import_module("tvm.tirx")
except ImportError as err:
    tirx = None
    _TIRX_IMPORT_ERROR = err
else:
    _TIRX_IMPORT_ERROR = None


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


def safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception as err:
        return f"<repr failed: {type(err).__name__}: {err}>"


def object_summary(value: Any) -> str:
    if value is None:
        return "None"
    ty = type(value)
    return f"{ty.__module__}.{ty.__qualname__}(id=0x{id(value):x})"


def format_mapping(title: str, mapping: Any) -> list[str]:
    lines = [f"{title}:"]
    if not mapping:
        lines.append("  <empty>")
        return lines
    try:
        items = mapping.items()
    except AttributeError:
        lines.append(f"  {safe_repr(mapping)}")
        return lines
    for key, value in sorted(items, key=lambda item: str(item[0])):
        lines.append(f"  {key}: {safe_repr(value)}")
    return lines


@tilelang.jit
def matmul(A, B, block_M: int, block_N: int, block_K: int):
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


@dataclass
class Artifact:
    step: int
    file: str
    title: str
    note: str


class ArtifactDumper:
    def __init__(self, out_dir: Path, print_ir: bool = False) -> None:
        self.out_dir = out_dir
        self.print_ir = print_ir
        self.step = 0
        self.dumped_ir_generator = False
        self.dumped_ir_generator_runtime = False
        self.artifacts: list[Artifact] = []

    def _next_path(self, title: str, suffix: str) -> Path:
        safe_title = re.sub(r"[^0-9A-Za-z_.-]+", "_", title).strip("_").lower()
        safe_title = re.sub(r"^\d+_", "", safe_title)
        path = self.out_dir / f"{self.step:02d}_{safe_title}{suffix}"
        self.step += 1
        return path

    def _record(self, path: Path, title: str, note: str) -> None:
        self.artifacts.append(Artifact(self.step - 1, path.name, title, note))

    def dump_text(self, title: str, text: str, note: str, suffix: str = ".txt") -> Path:
        path = self._next_path(title, suffix)
        path.write_text(text, encoding="utf-8")
        self._record(path, title, note)
        if self.print_ir:
            print(f"\n===== {path.name}: {title} =====\n{text}")
        else:
            print(f"[{self.step - 1:02d}] wrote {path}")
        return path

    def dump_module(self, title: str, module: Any, note: str) -> Path:
        if hasattr(module, "script"):
            body = module.script()
        else:
            body = str(module)
        header = textwrap.dedent(
            f"""
            # Step {self.step}: {title}
            # {note}

            """
        ).lstrip()
        return self.dump_text(title, header + body + "\n", note, suffix=".py")

    def dump_manifest(self, config: dict[str, Any]) -> Path:
        lines = [
            "# Quickstart Pipeline Artifacts",
            "",
            "This directory was generated by `examples/inspect_quickstart_pipeline.py`.",
            "",
            "## Config",
            "",
            "```json",
            json.dumps(config, indent=2, sort_keys=True),
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


def make_cuda_target(arch: str) -> Target:
    return Target({"kind": "cuda", "arch": arch})


def _pass_config_get(pass_ctx: Any, key: str | PassConfigKey, default: Any = None) -> Any:
    key_value = key.value if isinstance(key, PassConfigKey) else key
    return pass_ctx.config.get(key_value, default)


def allow_vectorize(pass_ctx: Any | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return not bool(_pass_config_get(pass_ctx, "tirx.disable_vectorize", False))


def should_enable_aggressive_merge(pass_ctx: Any | None = None, target: Target | None = None) -> bool:
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


def LayoutVisual(module: tvm.IRModule) -> None:
    pass_ctx = tilelang.transform.get_pass_context()
    if _pass_config_get(pass_ctx, PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE, False):
        tilelang.analysis.LayoutVisual(formats=_layout_visual_formats(pass_ctx))(module)


def _have_pdl(target: Target) -> bool:
    return bool(nvcc.have_pdl(target))


def _have_tma(target: Target | None = None) -> bool:
    have_tma = getattr(nvcc, "have_tma", None)
    return bool(have_tma(target)) if have_tma is not None else False


def allow_warp_specialized(pass_ctx: Any | None = None, target: Target | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    if target is None or target.kind.name != "cuda" or not _have_tma(target):
        return False
    return not bool(_pass_config_get(pass_ctx, "tl.disable_warp_specialized", False))


def module_has_tma(module: tvm.IRModule) -> bool:
    return any(func.attrs and func.attrs.get("tl.has_tma", False) for _, func in module.functions.items())


def is_cpu_device_backend(target: Target) -> bool:
    return target.kind.name == "c"


def _has_device_kernel_launch(attrs: Any) -> bool:
    return bool(attrs and "calling_conv" in attrs and attrs["calling_conv"] == CallingConv.DEVICE_KERNEL_LAUNCH)


def _is_device_call_c_device(func: tirx.PrimFunc) -> bool:
    attrs = func.attrs
    calling_conv = attrs.get("calling_conv", CallingConv.DEFAULT)
    is_cpacked = calling_conv == CallingConv.C_PACKED_FUNC
    if "target" in attrs and attrs["target"].kind.name == "c" and not is_cpacked:
        return True
    return _has_device_kernel_launch(attrs)


def _is_device_call(func: tirx.PrimFunc) -> bool:
    return _has_device_kernel_launch(func.attrs)


def get_device_call(is_device_c: bool = False) -> Callable[[tirx.PrimFunc], bool]:
    return _is_device_call_c_device if is_device_c else _is_device_call


def get_host_call(is_device_c: bool = False) -> Callable[[tirx.PrimFunc], bool]:
    return lambda func: not get_device_call(is_device_c)(func)


def canon_target_host(target: str | Target, target_host: str | Target | None) -> str | Target:
    del target
    return target_host or ("llvm" if tvm.runtime.enabled("llvm") else "c")


def extrac_params(func: tirx.PrimFunc) -> list[str]:
    params = []
    for var in func.params:
        if var in func.buffer_map:
            buffer = func.buffer_map[var]
            shape = ", ".join(str(dim) for dim in buffer.shape)
            params.append(f"{buffer.name}: Tensor([{shape}], {buffer.dtype})")
        else:
            params.append(f"{var}: {var.dtype}")
    return params


def PreLowerSemanticCheck(module: tvm.IRModule) -> None:
    pass_ctx = tilelang.transform.get_pass_context()
    if _pass_config_get(pass_ctx, PassConfigKey.TL_DISABLE_PRELOWER_SEMANTIC_CHECK, False):
        return
    if _pass_config_get(pass_ctx, PassConfigKey.TL_AST_PRINT_ENABLE, False):
        tilelang.analysis.ASTPrinter()(module)
    tilelang.analysis.NestedLoopChecker()(module)
    tilelang.analysis.FragmentLoopChecker()(module)


def run_pass(
    dumper: ArtifactDumper,
    title: str,
    module: tvm.IRModule,
    transform_factory: Callable[[], Any],
    note: str,
) -> tvm.IRModule:
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
    dumper.dump_module(title, next_module, note)
    return next_module


def dump_ir_generator_artifacts(args: argparse.Namespace, dumper: ArtifactDumper) -> None:
    if not args.dump_ir_generator or dumper.dumped_ir_generator:
        return
    dumper.dumped_ir_generator = True

    original_source = getattr(matmul, "func_source", None)
    if original_source:
        dumper.dump_text(
            "ir_generator_00_original_jit_source",
            textwrap.dedent(str(original_source)).strip() + "\n",
            "Original Python function source captured by @tilelang.jit before eager IRGenerator mutation.",
            suffix=".py",
        )

    jit_func = getattr(matmul, "func", None)
    ir_gen = getattr(jit_func, "ir_gen", None)
    if ir_gen is None:
        dumper.dump_text(
            "ir_generator_unavailable",
            "This TileLang version does not expose matmul.func.ir_gen, so IRGenerator source cannot be dumped.\n",
            "IRGenerator object is unavailable on this installed TileLang version.",
        )
        return

    ir_source = getattr(ir_gen, "source", None)
    if ir_source:
        dumper.dump_text(
            "ir_generator_01_mutated_source",
            str(ir_source).rstrip() + "\n",
            "AST-mutated Python source compiled by TileLang into the eager IRGenerator closure.",
            suffix=".py",
        )

    metadata_lines = [
        "# IRGenerator Metadata",
        "",
        f"JIT object: {object_summary(matmul)}",
        f"JIT mode: {safe_repr(getattr(matmul, 'mode', None))}",
        f"Signature: {safe_repr(getattr(matmul, 'signature', None))}",
        f"JITFunc object: {object_summary(jit_func)}",
        f"IRGenerator object: {object_summary(ir_gen)}",
        f"IRGenerator gen: {safe_repr(getattr(ir_gen, 'gen', None))}",
        "",
        *format_mapping("JITFunc arg_names", {idx: name for idx, name in enumerate(getattr(jit_func, "arg_names", []) or [])}),
        "",
        *format_mapping("JITFunc tensor_args", getattr(jit_func, "tensor_args", None)),
        "",
        *format_mapping("JITFunc tensor_args_defaults", getattr(jit_func, "tensor_args_defaults", None)),
        "",
        *format_mapping("IRGenerator extra_type_hints", getattr(ir_gen, "extra_type_hints", None)),
        "",
    ]
    dumper.dump_text(
        "ir_generator_02_metadata",
        "\n".join(metadata_lines),
        "Metadata that controls how IRGenerator phase1/phase2 maps Python arguments and type hints into TIR.",
    )


def dump_ir_generator_runtime_state(args: argparse.Namespace, dumper: ArtifactDumper) -> None:
    if not args.dump_ir_generator or dumper.dumped_ir_generator_runtime:
        return
    dumper.dumped_ir_generator_runtime = True

    jit_func = getattr(matmul, "func", None)
    p1_cache = getattr(jit_func, "p1_cache", None)
    if not p1_cache:
        dumper.dump_text(
            "ir_generator_03_runtime_state",
            "No phase1 template cache entries were found after matmul.get_tir(...).\n",
            "IRGenerator runtime cache state after frontend TIR generation.",
        )
        return

    lines = ["# IRGenerator Runtime State", ""]
    for index, (key, template) in enumerate(p1_cache.items()):
        lines.extend(
            [
                f"## Phase1 template {index}",
                f"cache_key: {safe_repr(key)}",
                f"name: {safe_repr(getattr(template, 'name', None))}",
                f"is_lazy_style: {safe_repr(getattr(template, 'is_lazy_style', None))}",
                f"constexprs: {safe_repr(getattr(template, 'constexprs', None))}",
                f"matcher: {safe_repr(getattr(template, 'matcher', None))}",
                "",
            ]
        )
    dumper.dump_text(
        "ir_generator_03_runtime_state",
        "\n".join(lines),
        "Phase1 TirTemplate cache and constexpr matcher state after matmul.get_tir(...).",
    )

    first_template = next(iter(p1_cache.values()))
    phase1_prim_func = getattr(first_template, "prim_func", None)
    if phase1_prim_func is not None:
        dumper.dump_module(
            "ir_generator_04_phase1_template_primfunc",
            phase1_prim_func,
            "Phase1 template PrimFunc before phase2 constexpr/tensor-shape substitution produces the frontend module.",
        )


def build_initial_module(args: argparse.Namespace) -> tuple[tvm.IRModule, Any]:
    prim_func = matmul.get_tir(
        M=args.M,
        N=args.N,
        K=args.K,
        block_M=args.block_M,
        block_N=args.block_N,
        block_K=args.block_K,
    )
    global_symbol = str(prim_func.attrs["global_symbol"])
    module = tvm.IRModule({global_symbol: prim_func})
    params = extrac_params(prim_func)
    return module, params


def call_public_lower(tilelang_lower: Callable[..., Any], prim_func: Any, target: Target, target_host: Target) -> Any:
    try:
        return tilelang_lower(
            prim_func,
            target=target,
            target_host=target_host,
            enable_host_codegen=False,
            enable_device_compile=False,
        )
    except TypeError as err:
        try:
            return tilelang_lower(prim_func, target=target, target_host=target_host)
        except TypeError:
            raise err


def dump_public_lowering(args: argparse.Namespace, dumper: ArtifactDumper, pass_configs: dict[str, Any]) -> None:
    fallback_configs = dict(pass_configs)
    dump_ir_dir = dumper.out_dir / "tvm_dump_ir"
    fallback_configs[pass_config_key("TL_ENABLE_DUMP_IR", "tl.enable_dump_ir")] = True
    fallback_configs[pass_config_key("TL_DUMP_IR_DIR", "tl.dump_ir_path")] = str(dump_ir_dir)

    dumper.dump_text(
        "00_public_dumpir_fallback",
        "\n".join(
            [
                "Manual pass-by-pass mode is unavailable because this TVM package does not provide tvm.tirx.",
                f"Original import error: {_TIRX_IMPORT_ERROR!r}",
                "Falling back to tilelang.lower under TVM DumpIR instrumentation.",
                f"TVM DumpIR directory: {dump_ir_dir}",
                "",
            ]
        ),
        "The installed TileLang/TVM version lacks tvm.tirx, so this run follows the public lowering path.",
    )

    target = make_cuda_target(args.arch)
    target_host = Target(canon_target_host(target, args.target_host))
    target = Target(target, target_host)
    pass_instruments = []
    dump_ir = getattr(tvm.ir.instrument, "DumpIR", None)
    if dump_ir is not None:
        pass_instruments.append(dump_ir(dump_dir=str(dump_ir_dir)))

    with tilelang.transform.PassContext(opt_level=3, config=fallback_configs, instruments=pass_instruments), target:
        dump_ir_generator_artifacts(args, dumper)
        module, params = build_initial_module(args)
        dump_ir_generator_runtime_state(args, dumper)
        prim_func = next(iter(module.functions.values()))
        dumper.dump_module(
            "01_frontend_primfunc",
            module,
            "Eager JIT/AST builder output before public TileLang lowering.",
        )
        dumper.dump_text(
            "02_kernel_params",
            "\n".join(str(param) for param in params) + "\n",
            "Kernel parameters extracted from the frontend PrimFunc buffer map and scalar args.",
        )

        tilelang_lower = getattr(tilelang, "lower", None)
        if tilelang_lower is None:
            raise RuntimeError("This TileLang package does not expose tilelang.lower, so public DumpIR fallback cannot run.")

        artifact = call_public_lower(tilelang_lower, prim_func, target, target_host)

    if hasattr(artifact, "host_mod"):
        dumper.dump_module("03_public_host_module", artifact.host_mod, "Host module returned by tilelang.lower.")
    if hasattr(artifact, "device_mod"):
        dumper.dump_module("04_public_device_module", artifact.device_mod, "Device module returned by tilelang.lower.")
    kernel_source = getattr(artifact, "kernel_source", None)
    if kernel_source is not None:
        dumper.dump_text(
            "05_public_generated_source",
            str(kernel_source),
            "Kernel source returned by tilelang.lower without device compilation.",
            suffix=".cu",
        )

    dumper.dump_manifest(
        {
            "mode": "public_dumpir_fallback",
            "missing_tirx": repr(_TIRX_IMPORT_ERROR),
            "M": args.M,
            "N": args.N,
            "K": args.K,
            "block_M": args.block_M,
            "block_N": args.block_N,
            "block_K": args.block_K,
            "target": "cuda",
            "arch": args.arch,
            "target_host": str(target_host),
            "pass_configs": fallback_configs,
            "dump_ir_dir": str(dump_ir_dir),
        }
    )


def dump_cuda_pipeline(args: argparse.Namespace, dumper: ArtifactDumper, pass_configs: dict[str, Any]) -> None:
    if tirx is None:
        dump_public_lowering(args, dumper, pass_configs)
        return

    target = make_cuda_target(args.arch)
    target_host = Target(canon_target_host(target, args.target_host))
    target = Target(target, target_host)
    pass_instruments = []
    if args.enable_dump_ir_instrument:
        pass_instruments.append(tvm.ir.instrument.DumpIR(dump_dir=str(dumper.out_dir / "tvm_dump_ir")))
    pass_context = tilelang.transform.PassContext(opt_level=3, config=pass_configs, instruments=pass_instruments)

    with pass_context, target:
        dump_ir_generator_artifacts(args, dumper)
        module, params = build_initial_module(args)
        dump_ir_generator_runtime_state(args, dumper)
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

        module = run_pass(
            dumper,
            "03_bind_target",
            module,
            lambda: tirx.transform.BindTarget(target),
            "Attaches the CUDA target to the PrimFunc so later passes can make target-specific choices.",
        )

        if should_force_let_inline():
            module = run_pass(
                dumper,
                "04_let_inline",
                module,
                tilelang.transform.LetInline,
                "Optional let inlining enabled by pass config.",
            )

        module = run_pass(
            dumper,
            "05_add_wrapper_for_single_buf_store",
            module,
            tilelang.transform.AddWrapperForSingleBufStore,
            "Normalizes single-buffer stores before later region and memory passes.",
        )
        module = run_pass(
            dumper,
            "06_legalize_negative_index",
            module,
            tilelang.transform.LegalizeNegativeIndex,
            "Canonicalizes negative indices into non-negative index expressions.",
        )
        if should_enable_race_check():
            module = run_pass(
                dumper,
                "07_verify_parallel_loop",
                module,
                tilelang.transform.VerifyParallelLoop,
                "Checks parallel loop legality; this is mostly validation unless it reports an error.",
            )
        module = run_pass(
            dumper,
            "08_inject_assumes",
            module,
            tilelang.transform.InjectAssumes,
            "Injects shape/bound assumptions that help the prover and simplifier.",
        )
        module = run_pass(
            dumper,
            "09_tilelang_simplify",
            module,
            tilelang.transform.Simplify,
            "Simplifies expressions while the IR still contains high-level TileLang constructs.",
        )
        module = run_pass(
            dumper,
            "10_layout_reducer",
            module,
            tilelang.transform.LayoutReducer,
            "Sets reducer layout metadata before the main layout inference pass.",
        )

        if allow_warp_specialized(target=target):
            module = run_pass(
                dumper,
                "11_producer_consumer_warp_specialized",
                module,
                tilelang.cuda.transform.ProducerConsumerWarpSpecialized,
                "Optional Hopper+ warp specialization while tile ops are still high level.",
            )

        module = run_pass(
            dumper,
            "12_lower_blackwell_2sm",
            module,
            tilelang.cuda.transform.LowerBlackwell2SM,
            "Handles Blackwell 2CTA/2SM GEMM annotations before LayoutInference.",
        )
        module = run_pass(
            dumper,
            "13_if_stmt_binding",
            module,
            tilelang.transform.IfStmtBinding,
            "Normalizes if/bind structure so software pipeline planning sees canonical bodies.",
        )
        module = run_pass(
            dumper,
            "14_pipeline_planning",
            module,
            tilelang.transform.PipelinePlanning,
            "Plans the software pipeline around the T.Pipelined loop.",
        )
        module = run_pass(
            dumper,
            "15_inject_software_pipeline",
            module,
            tilelang.transform.InjectSoftwarePipeline,
            "Rewrites the loop body into prologue/main/epilogue-style pipelined IR.",
        )
        module = run_pass(
            dumper,
            "16_simplify_after_pipeline",
            module,
            tilelang.transform.Simplify,
            "Cleans up expressions introduced by software pipeline injection.",
        )
        module = run_pass(
            dumper,
            "17_layout_inference",
            module,
            tilelang.transform.LayoutInference,
            "Infers fragment/shared/parallel-loop layouts that later tile-op lowering consumes.",
        )
        LayoutVisual(module)
        module = run_pass(
            dumper,
            "18_lower_tile_op",
            module,
            tilelang.transform.LowerTileOp,
            "Lowers tl.tileop.copy/gemm/reduce into lower-level TIR/intrinsic structure.",
        )
        module = run_pass(
            dumper,
            "19_lower_l2_persistent",
            module,
            tilelang.cuda.transform.LowerL2Persistent,
            "CUDA-specific lowering for L2 persistent mapping metadata.",
        )
        module = run_pass(
            dumper,
            "20_decouple_type_cast",
            module,
            tilelang.transform.DecoupleTypeCast,
            "Decouples type casts from vectorization constraints before vectorization.",
        )
        module = run_pass(
            dumper,
            "21_legalize_vectorized_loop",
            module,
            tilelang.transform.LegalizeVectorizedLoop,
            "Makes vectorized loops structurally legal for later lowering.",
        )
        module = run_pass(
            dumper,
            "22_legalize_safe_memory_access",
            module,
            tilelang.transform.LegalizeSafeMemoryAccess,
            "Adds predicates/guards for memory accesses that may be out of bounds at tile edges.",
        )
        module = run_pass(
            dumper,
            "23_lower_access_ptr",
            module,
            tilelang.transform.LowerAccessPtr,
            "Lowers TileLang pointer metadata ops to standard tvm_access_ptr-style nodes.",
        )
        module = run_pass(
            dumper,
            "24_simplify_after_safe_access",
            module,
            tilelang.transform.Simplify,
            "Simplifies duplicated conditions and expressions from access legalization.",
        )
        module = run_pass(
            dumper,
            "25_hoist_non_restrict_params",
            module,
            tilelang.transform.HoistNonRestrictParams,
            "Hoists root-block non-restrict annotations to function attrs when present.",
        )
        module = run_pass(
            dumper,
            "26_lower_shared_tmem",
            module,
            tilelang.transform.LowerSharedTmem,
            "Lowers shared.tmem allocations and initialization placement.",
        )
        module = run_pass(
            dumper,
            "27_plan_update_buffer_allocation_location",
            module,
            tilelang.transform.PlanAndUpdateBufferAllocationLocation,
            "Moves buffer allocations to planned locations after pipeline/barrier structure is known.",
        )
        module = run_pass(
            dumper,
            "28_lower_shared_barrier",
            module,
            tilelang.transform.LowerSharedBarrier,
            "Lowers TileLang shared barrier constructs.",
        )

        if module_has_tma(module):
            module = run_pass(
                dumper,
                "29_fuse_mbarrier_arrive_expect_tx",
                module,
                tilelang.transform.FuseMBarrierArriveExpectTx,
                "Fuses TMA mbarrier arrive/expect-tx when LowerTileOp produced TMA ops.",
            )

        module = run_pass(
            dumper,
            "30_hoist_global_buffer_allocations",
            module,
            tilelang.transform.HoistGlobalBufferAllocations,
            "Hoists global buffer allocations to the appropriate scope.",
        )
        module = run_pass(
            dumper,
            "31_lower_opaque_block",
            module,
            tilelang.transform.LowerOpaqueBlock,
            "Lowers opaque TileLang blocks into standard TIR structure.",
        )
        module = run_pass(
            dumper,
            "32_simplify_after_opaque_block",
            module,
            tilelang.transform.Simplify,
            "Cleans up IR after opaque block lowering.",
        )
        module = run_pass(
            dumper,
            "33_narrow_data_type_32",
            module,
            lambda: tirx.transform.NarrowDataType(32),
            "Narrows index/data expressions where legal to 32-bit forms.",
        )
        module = run_pass(
            dumper,
            "34_flatten_buffer",
            module,
            tilelang.transform.FlattenBuffer,
            "Flattens multi-dimensional buffers and buffer indexing.",
        )
        module = run_pass(
            dumper,
            "35_config_index_bitwidth",
            module,
            tilelang.transform.ConfigIndexBitwidth,
            "Applies the configured index bitwidth after buffer flattening.",
        )
        module = run_pass(
            dumper,
            "36_tirx_simplify_after_flatten",
            module,
            tirx.transform.Simplify,
            "Runs TVM/TIRX simplification on flattened index expressions.",
        )
        module = run_pass(
            dumper,
            "37_vectorize_loop",
            module,
            lambda: tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize()),
            "Vectorizes eligible loops unless vectorization is disabled by pass config.",
        )
        module = run_pass(
            dumper,
            "38_storage_rewrite",
            module,
            tilelang.transform.StorageRewrite,
            "Rewrites storage allocation/reuse for local/shared temporaries.",
        )
        module = run_pass(
            dumper,
            "39_loop_unswitching",
            module,
            tilelang.transform.LoopUnswitching,
            "Hoists loop-invariant branches out of loops when profitable/legal.",
        )
        module = run_pass(
            dumper,
            "40_unroll_loop",
            module,
            tilelang.transform.UnrollLoop,
            "Applies loop unrolling decisions.",
        )
        module = run_pass(
            dumper,
            "41_renormalize_split_pattern",
            module,
            tvm_transform("RenormalizeSplitPattern"),
            "Normalizes split-loop patterns after unrolling and simplification.",
        )
        module = run_pass(
            dumper,
            "42_tirx_simplify_after_unroll",
            module,
            tirx.transform.Simplify,
            "Cleans up after loop transformations.",
        )
        module = run_pass(
            dumper,
            "43_remove_no_op",
            module,
            tirx.transform.RemoveNoOp,
            "Removes no-op statements introduced by previous rewrites.",
        )
        module = run_pass(
            dumper,
            "44_hoist_if_then_else",
            module,
            tvm_transform("HoistIfThenElse"),
            "Hoists suitable if-then-else constructs for cleaner lowered IR.",
        )
        module = run_pass(
            dumper,
            "45_verify_memory",
            module,
            tirx.transform.VerifyMemory,
            "Verifies memory access legality before host/device annotation.",
        )
        module = run_pass(
            dumper,
            "46_annotate_entry_func",
            module,
            tirx.transform.AnnotateEntryFunc,
            "Marks the entry function for later host/device split and codegen.",
        )
        module = run_pass(
            dumper,
            "47_infer_fragment",
            module,
            tvm_transform("InferFragment"),
            "Infers fragment-related metadata needed by thread-level reductions.",
        )
        module = run_pass(
            dumper,
            "48_lower_thread_allreduce",
            module,
            tilelang.transform.LowerThreadAllreduce,
            "Lowers thread-level allreduce constructs if present.",
        )
        module = run_pass(
            dumper,
            "49_lower_ldg_stg",
            module,
            tilelang.transform.LowerLDGSTG,
            "Lowers eligible global loads/stores to CUDA ldg/stg intrinsics.",
        )
        module = run_pass(
            dumper,
            "50_lower_hopper_intrin",
            module,
            tilelang.cuda.transform.LowerHopperIntrin,
            "Lowers Hopper-specific CUDA intrinsics when target/IR require them.",
        )
        module = run_pass(
            dumper,
            "51_annotate_device_regions",
            module,
            tilelang.transform.AnnotateDeviceRegions,
            "Marks device regions so SplitHostDevice can separate host and kernel functions.",
        )
        module = run_pass(
            dumper,
            "52_split_host_device",
            module,
            tilelang.transform.SplitHostDevice,
            "Splits the mixed module into host wrapper and device kernel functions.",
        )
        module = run_pass(
            dumper,
            "53_mark_cuda_sync_calls",
            module,
            lambda: tilelang.cuda.transform.MarkCudaSyncCalls(_have_pdl(target)),
            "Marks CUDA sync calls such as PDL sync/trigger when the architecture supports them.",
        )
        module = run_pass(
            dumper,
            "54_annotate_read_only_params",
            module,
            tilelang.transform.AnnotateReadOnlyParams,
            "Annotates read-only kernel parameters for downstream codegen.",
        )
        module = run_pass(
            dumper,
            "55_merge_shared_memory_allocations",
            module,
            lambda: tilelang.transform.MergeSharedMemoryAllocations(
                enable_aggressive_merge=should_enable_aggressive_merge(target=target),
                disable_reuse=should_disable_shared_memory_reuse(),
            ),
            "Merges shared memory allocations after SplitHostDevice, where device functions are explicit.",
        )
        module = run_pass(
            dumper,
            "56_inject_fence_proxy",
            module,
            tilelang.transform.InjectFenceProxy,
            "Injects CUDA async-proxy fences when the target and IR require them.",
        )
        module = run_pass(
            dumper,
            "57_thread_sync_shared",
            module,
            lambda: tilelang.transform.ThreadSync("shared"),
            "Inserts synchronization for shared memory hazards.",
        )
        module = run_pass(
            dumper,
            "58_thread_sync_shared_dyn",
            module,
            lambda: tilelang.transform.ThreadSync("shared.dyn"),
            "Inserts synchronization for dynamic shared memory hazards.",
        )
        module = run_pass(
            dumper,
            "59_inject_tcgen05_fence",
            module,
            tilelang.transform.InjectTcgen05Fence,
            "Injects conservative Blackwell tcgen05 fences when needed.",
        )
        module = run_pass(
            dumper,
            "60_merge_if_stmt",
            module,
            tilelang.transform.MergeIfStmt,
            "Merges compatible adjacent if statements after synchronization insertion.",
        )

        if allow_warp_specialized(target=target):
            module = run_pass(
                dumper,
                "61_annotate_warp_group_reg_alloc",
                module,
                tilelang.transform.AnnotateWarpGroupRegAlloc,
                "Annotates warp-group register allocation for warp-specialized kernels.",
            )

        module = run_pass(
            dumper,
            "62_make_packed_api",
            module,
            tilelang.transform.MakePackedAPI,
            "Converts the host-facing ABI to TVM packed API form.",
        )
        module = run_pass(
            dumper,
            "63_simplify_after_packed_api",
            module,
            tilelang.transform.Simplify,
            "Final simplification before lowering device kernel launch calls.",
        )
        module = run_pass(
            dumper,
            "64_lower_device_kernel_launch",
            module,
            tilelang.transform.LowerDeviceKernelLaunch,
            "Lowers abstract device kernel launch calls inside the host wrapper.",
        )
        module = run_pass(
            dumper,
            "65_persist_threadblock",
            module,
            tilelang.cuda.transform.PersistThreadblock,
            "Applies CUDA persistent threadblock transformation when requested by annotations.",
        )

        host_filter = get_host_call(is_device_c=is_cpu_device_backend(target))
        device_filter = get_device_call(is_device_c=is_cpu_device_backend(target))
        host_module = tirx.transform.Filter(host_filter)(module)
        device_module = tirx.transform.Filter(device_filter)(module)
        dumper.dump_module(
            "66_host_module_after_filter",
            host_module,
            "Host-side module selected from the lowered mixed module.",
        )
        dumper.dump_module(
            "67_device_module_after_filter",
            device_module,
            "Device-side module selected from the lowered mixed module.",
        )

        codegen_ir = run_pass(
            dumper,
            "68_device_codegen_lower_intrin",
            device_module,
            tilelang.transform.LowerIntrin,
            "First device-codegen cleanup: lower remaining TileLang intrinsics.",
        )
        codegen_ir = run_pass(
            dumper,
            "69_device_codegen_simplify",
            codegen_ir,
            tirx.transform.Simplify,
            "Simplifies the device-only module immediately before source generation.",
        )
        codegen_ir = run_pass(
            dumper,
            "70_device_codegen_hoist_broadcast_values",
            codegen_ir,
            tilelang.transform.HoistBroadcastValues,
            "Hoists broadcast values in the device-only module before CUDA source emission.",
        )

        if not args.skip_codegen:
            global_func = "target.build.tilelang_cuda_without_compile"
            if "cutedsl" in target.keys:
                global_func = "target.build.tilelang_cutedsl_without_compile"
            try:
                codegen_module = tvm.ffi.get_global_func(global_func)(codegen_ir, target)
                cuda_source = codegen_module.inspect_source()
            except Exception as err:
                dumper.dump_text(
                    "71_generated_cuda_source_error",
                    f"CUDA source generation failed.\n\n{type(err).__name__}: {err}\n",
                    "Final CUDA source generation failed; earlier IR dumps are still useful for pipeline inspection.",
                    suffix=".txt",
                )
            else:
                dumper.dump_text(
                    "71_generated_cuda_source",
                    cuda_source,
                    "Final CUDA C++ source emitted by TileLang codegen, without invoking NVCC.",
                    suffix=".cu",
                )

    dumper.dump_manifest(
        {
            "M": args.M,
            "N": args.N,
            "K": args.K,
            "block_M": args.block_M,
            "block_N": args.block_N,
            "block_K": args.block_K,
            "target": "cuda",
            "arch": args.arch,
            "target_host": str(target_host),
            "pass_configs": pass_configs,
        }
    )


def load_pass_configs(raw_json: str | None) -> dict[str, Any]:
    if not raw_json:
        return {}
    path = Path(raw_json)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(raw_json)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump step-by-step compilation artifacts for examples/quickstart.py matmul.",
    )
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--block-M", type=int, default=128, dest="block_M")
    parser.add_argument("--block-N", type=int, default=128, dest="block_N")
    parser.add_argument("--block-K", type=int, default=32, dest="block_K")
    parser.add_argument("--arch", default="sm_80", help="CUDA architecture to encode in the target, e.g. sm_80 or sm_90.")
    parser.add_argument("--target-host", default=None, help="Optional host target. Defaults to llvm if enabled, otherwise c.")
    parser.add_argument("--out-dir", default="debug/quickstart_pipeline", help="Directory for generated IR/source artifacts.")
    parser.add_argument("--clean", action="store_true", help="Remove the output directory before generating artifacts.")
    parser.add_argument("--print-ir", action="store_true", help="Also print every dumped artifact to stdout.")
    parser.add_argument(
        "--pass-config-json",
        default=None,
        help="Optional JSON string or JSON file path with TileLang/TVM PassContext config.",
    )
    parser.add_argument(
        "--enable-dump-ir-instrument",
        action="store_true",
        help="Also enable TVM DumpIR instrument into OUT_DIR/tvm_dump_ir for comparison with the manual dumps.",
    )
    parser.add_argument(
        "--no-dump-ir-generator",
        dest="dump_ir_generator",
        action="store_false",
        help="Do not dump TileLang eager IRGenerator source/metadata artifacts before lowering.",
    )
    parser.set_defaults(dump_ir_generator=True)
    parser.add_argument("--skip-codegen", action="store_true", help="Stop after host/device IR dumps and skip CUDA source generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pass_configs = load_pass_configs(args.pass_config_json)
    if args.enable_dump_ir_instrument:
        pass_configs[pass_config_key("TL_ENABLE_DUMP_IR", "tl.enable_dump_ir")] = True
        pass_configs[pass_config_key("TL_DUMP_IR_DIR", "tl.dump_ir_path")] = str(out_dir / "tvm_dump_ir")

    dumper = ArtifactDumper(out_dir, print_ir=args.print_ir)
    dump_cuda_pipeline(args, dumper, pass_configs)
    print(f"\nDone. Open {out_dir / 'README.md'} to browse the artifacts in order.")


if __name__ == "__main__":
    main()