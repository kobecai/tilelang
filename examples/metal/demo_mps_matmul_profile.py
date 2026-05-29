from __future__ import annotations

import argparse
import time

import torch


def synchronize_mps() -> None:
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "synchronize"):
        mps.synchronize()


def benchmark_mps_event(fn, warmup: int = 10, repeat: int = 50, use_events: bool = False) -> float:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeat <= 0:
        raise ValueError("repeat must be positive")

    for _ in range(warmup):
        fn()
    synchronize_mps()

    if not use_events:
        start = time.perf_counter()
        for _ in range(repeat):
            fn()
        synchronize_mps()
        return (time.perf_counter() - start) * 1_000.0 / repeat

    mps = getattr(torch, "mps", None)
    event_type = getattr(mps, "Event", None) if mps is not None else None
    if event_type is None:
        raise RuntimeError("torch.mps.Event is not available in this PyTorch build.")

    start_event = event_type(enable_timing=True)
    end_event = event_type(enable_timing=True)
    start_event.record()
    for _ in range(repeat):
        fn()
    end_event.record()
    end_event.synchronize()
    return start_event.elapsed_time(end_event) / repeat


def create_matmul_prim_func(
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
):
    import tilelang.language as T

    dtype = T.float32
    accum_dtype = T.float32

    @T.prim_func
    def matmul_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared, coalesced_width=2)
                T.copy(B[ko * block_K, bx * block_N], B_shared, coalesced_width=2)

                for i, j in T.Parallel(block_M, block_N):
                    for k in T.Serial(block_K):
                        C_local[i, j] += A_shared[i, k] * B_shared[k, j]

            T.copy(C_local, C[by * block_M, bx * block_N], coalesced_width=2)

    return matmul_kernel


def require_mps() -> None:
    if not torch.backends.mps.is_built():
        raise RuntimeError("This PyTorch build does not include MPS support.")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this machine.")
    if not hasattr(torch, "mps") or not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError("torch.mps.compile_shader is required; install a recent PyTorch build.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small TileLang matmul demo on Apple Metal/MPS.")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=16)
    parser.add_argument("--block-k", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument(
        "--use-mps-events",
        action="store_true",
        help="Use torch.mps.Event timing. Wall-clock timing is the default because MPS events can hang on some stacks.",
    )
    return parser.parse_args()


def main() -> None:
    require_mps()

    import tilelang

    tilelang.disable_cache()

    args = parse_args()
    torch.manual_seed(0)

    func = create_matmul_prim_func(args.m, args.n, args.k, args.block_m, args.block_n, args.block_k)
    kernel = tilelang.compile(func, target="metal", execution_backend="torch")

    a = torch.randn(args.m, args.k, device="mps", dtype=torch.float32)
    b = torch.randn(args.k, args.n, device="mps", dtype=torch.float32)
    c = torch.zeros(args.m, args.n, device="mps", dtype=torch.float32)

    kernel(a, b, c)
    synchronize_mps()

    reference = a @ b
    synchronize_mps()
    torch.testing.assert_close(c.cpu(), reference.cpu(), rtol=1e-3, atol=1e-3)

    latency_ms = benchmark_mps_event(
        lambda: kernel(a, b, c),
        warmup=args.warmup,
        repeat=args.repeat,
        use_events=args.use_mps_events,
    )
    kernel_source = kernel.get_kernel_source()

    print(f"TileLang: {tilelang.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Target: metal, device: mps")
    print(f"Shape: M={args.m}, N={args.n}, K={args.k}")
    print(f"Block: M={args.block_m}, N={args.block_n}, K={args.block_k}")
    print(f"Correctness: PASS")
    print(f"Kernel source bytes: {len(kernel_source.encode('utf-8'))}")
    print(f"Timer: {'torch.mps.Event' if args.use_mps_events else 'synchronized wall clock'}")
    print(f"Latency: {latency_ms:.4f} ms")


if __name__ == "__main__":
    main()
