import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def _boundary_values(dtype):
    bits = int(dtype.removeprefix("uint").removeprefix("int"))
    if dtype.startswith("uint"):
        return [0, 1, 50, (1 << (bits - 1)) - 1, 1 << (bits - 1), (1 << (bits - 1)) + 1, (1 << bits) - 2, (1 << bits) - 1]
    return [-128, -127, -1, 0, 1, 50, 126, 127]


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [
        ("uint8", "int16"),
        ("uint8", "int32"),
        ("uint8", "float16"),
        ("uint16", "int32"),
        ("uint16", "float32"),
        ("uint32", "int64"),
        ("uint32", "float32"),
        ("int8", "int32"),
    ],
)
def test_cutedsl_unsigned_scalar_cast(src_dtype, dst_dtype):
    pytest.importorskip("cutlass")

    @T.prim_func
    def kernel(A: T.Tensor((32,), src_dtype), C: T.Tensor((3, 32), dst_dtype)):
        with T.Kernel(1, threads=32):
            i = T.get_thread_binding()
            local = T.alloc_local((1,), src_dtype)
            shared = T.alloc_shared((32,), src_dtype)
            local[0] = A[i]
            shared[i] = A[i]
            T.sync_threads()
            C[0, i] = T.Cast(dst_dtype, A[i])
            C[1, i] = T.Cast(dst_dtype, local[0])
            C[2, i] = T.Cast(dst_dtype, shared[i])

    a = torch.tensor(_boundary_values(src_dtype) * 4, dtype=getattr(torch, src_dtype), device="cuda")
    compiled = tilelang.compile(kernel, target="cutedsl", out_idx=-1)
    actual = compiled(a)
    expected = a.to(getattr(torch, dst_dtype)).expand(3, -1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["uint8", "uint16", "uint32"])
def test_cutedsl_unsigned_scalar_comparison_and_extrema(dtype):
    pytest.importorskip("cutlass")

    @T.prim_func
    def kernel(
        A: T.Tensor((32,), dtype),
        B: T.Tensor((32,), dtype),
        C: T.Tensor((4, 32), "bool"),
        D: T.Tensor((3, 32), dtype),
    ):
        with T.Kernel(1, threads=32):
            i = T.get_thread_binding()
            C[0, i] = A[i] < B[i]
            C[1, i] = A[i] <= B[i]
            C[2, i] = A[i] > B[i]
            C[3, i] = A[i] >= B[i]
            D[0, i] = A[i]
            D[1, i] = T.min(A[i], B[i])
            D[2, i] = T.max(A[i], B[i])

    a_ref = torch.tensor(_boundary_values(dtype) * 4, dtype=torch.int64, device="cuda")
    b_ref = torch.full_like(a_ref, 50)
    compiled = tilelang.compile(kernel, target="cutedsl", out_idx=[2, 3])
    comparisons, extrema = compiled(a_ref.to(getattr(torch, dtype)), b_ref.to(getattr(torch, dtype)))
    torch.testing.assert_close(comparisons, torch.stack([a_ref < b_ref, a_ref <= b_ref, a_ref > b_ref, a_ref >= b_ref]))
    expected_extrema = torch.stack([a_ref, torch.minimum(a_ref, b_ref), torch.maximum(a_ref, b_ref)])
    torch.testing.assert_close(extrema.to(torch.int64), expected_extrema, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
