"""TMA stores must follow writer-side proxy fences and shared synchronization."""

import re

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def tma_store_reused_shared(columns, threads, explicit_sync):
    rows = 4096
    block_rows = block_cols = 64

    @T.prim_func
    def main(A: T.Tensor((rows, columns), "float32"), O: T.Tensor((rows, columns), "float32")):
        with T.Kernel(T.ceildiv(columns, block_cols), T.ceildiv(rows, block_rows), threads=threads) as (bx, by):
            shared = T.alloc_shared((block_rows, block_cols), "float32")
            for iteration in T.serial(16):
                for i, j in T.Parallel(block_rows, block_cols):
                    shared[i, j] = A[by * block_rows + i, bx * block_cols + j] + T.float32(iteration)
                if explicit_sync:
                    T.sync_threads()
                T.copy(shared, O[by * block_rows, bx * block_cols], annotations={"prefer_instruction": "tma"})

    return main


def _check_store_fence_order(source):
    stores = list(re.finditer(r"tl::tma_store(?:<[^>]*>)?\(", source))
    assert stores, "Expected TMA stores; a generic-copy fallback does not exercise this regression"
    leaders = list(re.finditer(r"if\s*\(tl::tl_shuffle_elect<\d+>\(\)\)", source))
    assert leaders, "Expected elected-thread TMA stores"
    for store in stores:
        leader = max((match.start() for match in leaders if match.start() < store.start()), default=-1)
        fence = source.rfind("tl::fence_proxy_async();", 0, store.start())
        barrier = max(
            source.rfind("__syncthreads();", 0, store.start()),
            source.rfind("tl::__sync_thread_partial(", 0, store.start()),
        )
        assert 0 <= fence < barrier < leader < store.start(), source


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@pytest.mark.parametrize("columns", [64, 128])
@pytest.mark.parametrize("threads", [64, 128])
@pytest.mark.parametrize("explicit_sync", [False, True])
def test_tma_store_writer_fence_with_shared_reuse(columns, threads, explicit_sync):
    kernel = tilelang.compile(
        tma_store_reused_shared(columns, threads, explicit_sync),
        out_idx=[1],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    source = kernel.get_kernel_source()
    _check_store_fence_order(source)
    if columns == 64:
        assert "CUtensorMap" not in source, "Expected contiguous 1D TMA stores"
    else:
        assert "CUtensorMap" in source, "Expected descriptor-based TMA stores"

    values = torch.arange(4096 * columns, device="cuda", dtype=torch.float32).reshape(4096, columns) % 1024
    expected = values + 15
    for _ in range(20):
        torch.testing.assert_close(kernel(values), expected, rtol=0, atol=0)
