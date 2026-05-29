from __future__ import annotations

from common import import_tvm, make_ir_module, print_obj, print_section, tir_namespace


tvm = import_tvm()
tir = tir_namespace(tvm)

from tvm.script import tir as T  # noqa: E402


@T.prim_func
def vector_add(
    A: T.Buffer((16,), "float32"),
    B: T.Buffer((16,), "float32"),
    C: T.Buffer((16,), "float32"),
):
    T.func_attr({"global_symbol": "vector_add", "tir.noalias": True})
    for i in T.serial(16):
        with T.block("add"):
            vi = T.axis.spatial(16, i)
            C[vi] = A[vi] + B[vi]


def collect_blocks(func):
    block_types = []
    for type_name in ("SBlock", "Block"):
        if hasattr(tir, type_name):
            block_types.append(getattr(tir, type_name))

    blocks = []

    def visit(node):
        if block_types and isinstance(node, tuple(block_types)):
            blocks.append(node)

    tir.stmt_functor.post_order_visit(func.body, visit)
    return blocks


def main() -> None:
    mod = make_ir_module(tvm, vector_add, "vector_add")
    func = mod["vector_add"]

    print_obj(mod, "1. IRModule wraps one or more PrimFunc objects")

    print_section("2. PrimFunc metadata")
    print("attrs:", dict(func.attrs))
    print("number of params:", len(func.params))
    print("params:", [str(param) for param in func.params])

    print_section("3. Buffer map")
    for handle_var, buffer in func.buffer_map.items():
        print(f"{handle_var} -> {buffer.name}: shape={buffer.shape}, dtype={buffer.dtype}")

    print_section("4. SBlock/Block read-write regions")
    for block in collect_blocks(func):
        reads = getattr(block, "reads", [])
        writes = getattr(block, "writes", [])
        print(f"block={block.name_hint}")
        print("  reads :", reads)
        print("  writes:", writes)

    print_section("5. Structural equality")
    same = tvm.ir.structural_equal(mod, make_ir_module(tvm, vector_add, "vector_add"))
    print("structural_equal(mod, cloned_mod):", same)


if __name__ == "__main__":
    main()
