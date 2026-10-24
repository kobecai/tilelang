"""TMA multicast validation."""

import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


_BARRIER_REUSE_CASES = ["single", "fresh", "ring", "two_calls", "conditional", "nested", "unrolled"]


def make_sync_multicast_barrier_kernel(mode):
    @T.macro
    def transfer(A, B, shared, bar, rank, iteration, slot):
        T.copy_cluster(A[iteration, :, :], shared[slot, :, :], cluster_mask=0b0011, remote_barrier=bar[slot])
        T.copy(shared[slot, :, :], B[rank, iteration, :, :], prefer_instruction="sync")
        # Isolate phase selection from cross-CTA destination/barrier reuse.
        T.cluster_sync()

    @T.prim_func
    def main(A: T.Tensor((8, 32, 32), T.float32), B: T.Tensor((4, 8, 32, 32), T.float32)):
        with T.ClusterKernel(4, cluster_dims=(4, 1, 1), threads=128) as rank:
            shared = T.alloc_shared((8, 32, 32), T.float32)
            bar = T.alloc_cluster_barrier([1] * 8)
            if mode == "two_calls":
                for i in T.serial(4):
                    transfer(A, B, shared, bar, rank, 2 * i, 0)
                    transfer(A, B, shared, bar, rank, 2 * i + 1, 0)
            elif mode == "nested":
                for outer in T.serial(2):
                    for inner in T.serial(3):
                        transfer(A, B, shared, bar, rank, 3 * outer + inner, 0)
            elif mode == "unrolled":
                for i in T.unroll(8):
                    transfer(A, B, shared, bar, rank, i, 0)
            elif mode == "conditional":
                for i in T.serial(8):
                    if i % 2 == 0:
                        transfer(A, B, shared, bar, rank, i, 0)
            else:
                for i in T.serial(8):
                    slot = 0 if mode == "single" else (i if mode == "fresh" else i % 4)
                    transfer(A, B, shared, bar, rank, i, slot)

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("mode", _BARRIER_REUSE_CASES)
def test_sync_multicast_barrier_reuse_codegen(mode):
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})
    with target, tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}):
        artifact = tilelang.lower(make_sync_multicast_barrier_kernel(mode), target=target)
    source = artifact.kernel_source
    assert source is not None
    assert "tma_load_multicast" in source
    assert ".arrive_and_wait();" in source
    assert ".wait(" not in source


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
@pytest.mark.parametrize("mode", _BARRIER_REUSE_CASES)
def test_sync_multicast_barrier_reuse(mode):
    kernel = tilelang.compile(
        make_sync_multicast_barrier_kernel(mode),
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    # Every iteration differs, so a stale tile cannot accidentally compare equal.
    A = torch.arange(8 * 32 * 32, device="cuda", dtype=torch.float32).reshape(8, 32, 32)
    expected = A.clone()
    if mode == "conditional":
        expected[1::2] = 0
    elif mode == "nested":
        expected[6:] = 0
    for _ in range(3):
        B = torch.zeros((4, 8, 32, 32), device="cuda", dtype=torch.float32)
        kernel(A, B)
        torch.testing.assert_close(B, expected.unsqueeze(0).expand_as(B), rtol=0, atol=0)


def make_tma_multicast_demo_kernel(M, N, block_M, block_N, cluster_mask):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, N), "float16"),
    ):
        with T.ClusterKernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=128,
            cluster_dims=(4, 1, 1),
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), "float16")
            T.copy_cluster(A[by * block_M, bx * block_N], A_shared, cluster_mask=cluster_mask)
            T.copy(A_shared, B[by * block_M, bx * block_N])

    return kernel


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_multicast_demo():
    """Verify TMA multicast: rank 1's B region should equal rank 0's A region within the same cluster."""
    M, N = 1024, 1024
    block_M, block_N = 128, 64
    # mask=0b0011: rank 0 multicasts, rank 1 receives, ranks 2/3 each do regular tma_load
    cluster_mask = 0b0011

    kernel = make_tma_multicast_demo_kernel(M, N, block_M, block_N, cluster_mask)
    mod = tilelang.compile(
        kernel,
        out_idx=[1],
        execution_backend="cython",
    )

    A = torch.randn(M, N, device="cuda", dtype=torch.float16)
    B = mod(A)

    # Within a cluster: the first 4 blocks in the grid are (0,0),(1,0),(2,0),(3,0) -> by=0, bx=0,1,2,3
    # rank 0 -> bx=0: A[0:block_M, 0:block_N] -> B[0:block_M, 0:block_N]
    # rank 1 -> bx=1: multicast receives A[0:block_M, 0:block_N] -> B[0:block_M, block_N:2*block_N]
    # rank 2 -> bx=2: A[0:block_M, 2*block_N:3*block_N] -> B[0:block_M, 2*block_N:3*block_N]
    # rank 3 -> bx=3: A[0:block_M, 3*block_N:4*block_N] -> B[0:block_M, 3*block_N:4*block_N]

    # Multicast check: rank 1's B region should equal rank 0's A region
    B_rank1 = B[0:block_M, block_N : 2 * block_N]
    A_rank0 = A[0:block_M, 0:block_N]
    torch.testing.assert_close(B_rank1, A_rank0, rtol=1e-2, atol=1e-2)

    # rank 0 itself: B should equal A
    torch.testing.assert_close(B[0:block_M, 0:block_N], A[0:block_M, 0:block_N], rtol=1e-2, atol=1e-2)
    # ranks 2, 3: each B region equals its own A region
    torch.testing.assert_close(
        B[0:block_M, 2 * block_N : 3 * block_N],
        A[0:block_M, 2 * block_N : 3 * block_N],
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        B[0:block_M, 3 * block_N : 4 * block_N],
        A[0:block_M, 3 * block_N : 4 * block_N],
        rtol=1e-2,
        atol=1e-2,
    )


if __name__ == "__main__":
    tilelang.testing.main()
