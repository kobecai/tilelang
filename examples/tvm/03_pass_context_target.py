from __future__ import annotations

import argparse

from common import apply_available_passes, import_tvm, make_ir_module, print_obj, print_section, tir_namespace


tvm = import_tvm()
tir = tir_namespace(tvm)

from tvm.script import tir as T  # noqa: E402


@T.prim_func
def simplify_me(
    A: T.Buffer((16,), "int32"),
    C: T.Buffer((16,), "int32"),
):
    T.func_attr({"global_symbol": "simplify_me", "tir.noalias": True})
    for i in T.serial(16):
        with T.block("store"):
            vi = T.axis.spatial(16, i)
            C[vi] = ((A[vi] + T.int32(0)) * T.int32(1)) + (vi - vi)


def target_summary(target) -> str:
    kind = getattr(getattr(target, "kind", None), "name", None)
    keys = list(getattr(target, "keys", []))
    return f"kind={kind}, keys={keys}, target={target}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="llvm", help="Try cuda, hip, metal, llvm, or a full TVM target string.")
    args = parser.parse_args()

    target = tvm.target.Target(args.target)
    mod = make_ir_module(tvm, simplify_me, "simplify_me")

    print_obj(mod, "1. Before passes")
    print_section("2. Target object")
    print(target_summary(target))

    transform = tir.transform
    passes = [
        ("BindTarget", getattr(transform, "BindTarget", None) and (lambda: transform.BindTarget(target))),
        ("Simplify", getattr(transform, "Simplify", None)),
        ("NarrowDataType(32)", getattr(transform, "NarrowDataType", None) and (lambda: transform.NarrowDataType(32))),
        ("RemoveNoOp", getattr(transform, "RemoveNoOp", None)),
    ]

    print_section("3. PassContext + TargetContext")
    with tvm.transform.PassContext(opt_level=3) as pass_ctx:
        with target:
            current = tvm.target.Target.current(allow_none=True)
            print("PassContext opt_level:", pass_ctx.opt_level)
            print("Target.current:", target_summary(current))
            lowered = apply_available_passes(mod, passes)

    print_obj(lowered, "4. After available passes")

    print_section("5. TileLang connection")
    print("tilelang/engine/phase.py builds larger pipelines from the same ideas:")
    print("  IRModule -> transform pass -> new IRModule")
    print("  PassContext carries pass config such as dump/debug/vectorize flags.")
    print("  TargetContext lets passes query hardware capabilities.")


if __name__ == "__main__":
    main()
