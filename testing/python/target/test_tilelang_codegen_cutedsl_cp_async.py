import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm
from tilelang.engine.lower import lower
from tilelang.backend.target import determine_target
from tilelang.cuda.target import normalize_cutedsl_target


def test_cutedsl_dict_target_normalizes_to_cuda_marker(monkeypatch):
    from tilelang.jit.adapter.cutedsl import checks

    monkeypatch.setattr(checks, "check_cutedsl_available", lambda: None)
    target = determine_target({"kind": "cutedsl", "arch": "sm_80"}, return_object=True)

    assert target.kind.name == "cuda"
    assert target.attrs["arch"] == "sm_80"
    assert {"cuda", "gpu", "cutedsl"}.issubset(set(target.keys))


def test_cutedsl_string_target_uses_detected_cuda_arch(monkeypatch):
    """Verify bare CuTeDSL targets use TileLang's CUDA arch normalization."""

    from tilelang.jit.adapter.cutedsl import checks

    monkeypatch.setattr(checks, "check_cutedsl_available", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))

    target = determine_target("cutedsl", return_object=True)

    assert target.kind.name == "cuda"
    assert target.attrs["arch"] == "sm_90a"
    assert {"cuda", "gpu", "cutedsl"}.issubset(set(target.keys))


def test_cutedsl_codegen_supports_tl_ptx_cp_async():
    if not tvm.runtime.enabled("cuda"):
        pytest.skip("TileLang CuTeDSL codegen requires TVM built with CUDA support.")

    build_cutedsl = tvm.ffi.get_global_func("target.build.tilelang_cutedsl_without_compile", allow_missing=True)
    if build_cutedsl is None:
        pytest.skip("TileLang CuTeDSL backend is not enabled in this build.")

    target = normalize_cutedsl_target({"kind": "cutedsl", "arch": "sm_80"})
    assert target is not None

    @T.prim_func
    def prog(A: T.Tensor((16,), "uint8"), B: T.Tensor((16,), "uint8")):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((16,), "uint8", scope="shared")
            T.ptx_cp_async(T.access_ptr(A_shared[0], "w", 16), T.access_ptr(A[0], "r", 16), 16)
            B[0] = A_shared[0]

    artifact = lower(prog.with_attr("global_symbol", "main"), target=target)
    assert "tl.cp_async_gs(" in artifact.kernel_source


def _sync_tma_barrier_kernel():
    @T.prim_func
    def main(A: T.Tensor((6, 64, 64), T.float32), B: T.Tensor((6, 32, 32), T.float32)):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((32, 32), T.float32)
            bar = T.alloc_barrier([1] * 6)
            for i in T.serial(6):
                T.copy(A[i, 0:32, 0:32], shared, prefer_instruction="tma", annotations={"barrier": bar[i]})
                T.copy(shared, B[i, :, :], prefer_instruction="sync")

    return main


@tilelang.testing.requires_cuda
def test_cutedsl_sync_tma_uses_arrival_state():
    # Invoke source emission directly, without requiring the optional CuTeDSL
    # Python runtime or a physical Hopper GPU.
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})

    with target, tvm.transform.PassContext(config={"tl.disable_warp_specialized": True}):
        artifact = lower(_sync_tma_barrier_kernel(), target=target)
        build = tvm.get_global_func("target.build.tilelang_cutedsl_without_compile")
        device_mod = tilelang.transform.LowerIntrin()(artifact.device_mod)
        device_mod = tvm.tirx.transform.Simplify()(device_mod)
        source = build(device_mod, target).inspect_source()
    assert "tl.mbarrier_arrive_and_wait(" in source
    assert "tl.sync_threads()" in source
    assert "tl.mbarrier_wait(" not in source


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_cutedsl_sync_tma_barrier_slots():
    pytest.importorskip("cutlass.cute")
    kernel = tilelang.compile(
        _sync_tma_barrier_kernel(),
        target="cutedsl",
        execution_backend="cutedsl",
        pass_configs={"tl.disable_warp_specialized": True},
    )
    A = torch.arange(6 * 64 * 64, dtype=torch.float32, device="cuda").reshape(6, 64, 64)
    B = torch.empty((6, 32, 32), dtype=torch.float32, device="cuda")
    kernel(A, B)
    torch.testing.assert_close(B, A[:, :32, :32], rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
