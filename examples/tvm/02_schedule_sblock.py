from __future__ import annotations

from common import import_tvm, make_ir_module, print_obj, print_section, schedule_namespace


tvm = import_tvm()
schedule_ns = schedule_namespace(tvm)

from tvm.script import tir as T  # noqa: E402


M = 32
N = 32
K = 32


@T.prim_func
def matmul(
    A: T.Buffer((M, K), "float32"),
    B: T.Buffer((K, N), "float32"),
    C: T.Buffer((M, N), "float32"),
):
    T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
    for i, j, k in T.grid(M, N, K):
        with T.block("C"):
            vi = T.axis.spatial(M, i)
            vj = T.axis.spatial(N, j)
            vk = T.axis.reduce(K, k)
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def get_block(sch, name: str):
    try:
        return sch.get_block(name, func_name="matmul")
    except TypeError:
        return sch.get_block(name)


def main() -> None:
    mod = make_ir_module(tvm, matmul, "matmul")
    print_obj(mod, "1. Original TensorIR matmul")

    sch = schedule_ns.Schedule(mod)
    block_c = get_block(sch, "C")
    loops = sch.get_loops(block_c)

    print_section("2. Schedule sees the SBlock/Block boundary")
    print("block rv:", block_c)
    print("loop rvs:", loops)

    i_outer, i_inner = sch.split(loops[0], factors=[None, 8])
    j_outer, j_inner = sch.split(loops[1], factors=[None, 8])
    sch.reorder(i_outer, j_outer, i_inner, j_inner, loops[2])

    print_obj(
        sch.mod,
        "3. After split + reorder",
    )

    print_section("4. What this maps to in TileLang")
    print("TileLang usually hides this manual schedule behind T.Kernel, T.Parallel,")
    print("layout inference, tile ops, and target-aware lowering passes.")


if __name__ == "__main__":
    main()
