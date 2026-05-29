import tilelang  # noqa: F401
import tilelang.language as T
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget


@T.prim_func
def gemm_128x128(
    A: T.Tensor((128, 128), "float16"),
    B: T.Tensor((128, 128), "float16"),
    C: T.Tensor((128, 128), "float16"),
):
    with T.Kernel(1, threads=128) as (_,):
        A_s = T.alloc_shared((128, 128), "float16")
        B_s = T.alloc_shared((128, 128), "float16")
        C_l = T.alloc_fragment((128, 128), "float16")
        T.copy(A, A_s)
        T.copy(B, B_s)
        T.clear(C_l)
        T.gemm(A_s, B_s, C_l)
        T.copy(C_l, C)


print(gemm_128x128.script())

target = tvm.target.Target("cuda -arch=sm_80")
mod = tvm.IRModule.from_expr(
    gemm_128x128.with_attr("global_symbol", "gemm_128x128")
)
with target:
    mod = LowerAndLegalize(mod, target)
    mod = OptimizeForTarget(mod, target)

print(mod.script())  # 应该能看到 ptx_mma / cp_async 等低层 intrinsic