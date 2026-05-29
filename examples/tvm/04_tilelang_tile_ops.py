from __future__ import annotations

import argparse

from common import import_tilelang, import_tvm, make_ir_module, print_obj, print_section


tvm = import_tvm()
tilelang, T = import_tilelang()


@tilelang.jit(out_idx=[2])
def add_with_tile_ops(M: int, N: int, block_M: int = 16, block_N: int = 16, threads: int = 128):
    dtype = T.float32

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            B_shared = T.alloc_shared((block_M, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), dtype)

            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(B[by * block_M, bx * block_N], B_shared)
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = A_shared[i, j] + B_shared[i, j]
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def print_ffi_hooks() -> None:
    print_section("3. Selected FFI/PackedFunc hooks")
    names = [
        "tl.transform.LayoutInference",
        "tl.transform.LowerTileOp",
        "target.build.tilelang_cuda_without_compile",
        "target.build.tilelang_hip_without_compile",
        "target.build.tilelang_metal",
    ]
    for name in names:
        func = tvm.ffi.get_global_func(name, allow_missing=True)
        print(f"{name}: {'registered' if func is not None else 'missing in this build'}")


def maybe_lower(prim_func, target: str, target_host: str | None) -> None:
    from tilelang.engine.lower import lower_to_host_device_ir

    print_section("4. Optional TileLang lowering")
    host_mod, device_mod, params, resolved_target, resolved_host = lower_to_host_device_ir(
        prim_func,
        target=target,
        target_host=target_host,
    )
    print("resolved target:", resolved_target)
    print("resolved host  :", resolved_host)
    print("params         :", params)
    print_obj(host_mod, "host IR after SplitHostDevice")
    print_obj(device_mod, "device IR after SplitHostDevice")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--lower", action="store_true", help="Run TileLang's lowering pipeline and split host/device IR.")
    parser.add_argument("--target", default="cuda", help="Target for --lower, for example cuda, hip, metal, llvm, or c.")
    parser.add_argument("--target-host", default=None)
    args = parser.parse_args()

    prim_func = add_with_tile_ops.get_tir(args.m, args.n)
    mod = make_ir_module(tvm, prim_func, "main")

    print_obj(prim_func, "1. TileLang DSL elaborates to a TVM/TIR PrimFunc")
    print_obj(mod, "2. Wrapped as IRModule")
    print_ffi_hooks()

    if args.lower:
        maybe_lower(prim_func, args.target, args.target_host)
    else:
        print_section("4. Lowering is opt-in")
        print("Run with --lower after your TileLang build and target runtime are ready.")
        print("That path exercises LowerAndLegalize, OptimizeForTarget, and SplitHostDevice.")


if __name__ == "__main__":
    main()
