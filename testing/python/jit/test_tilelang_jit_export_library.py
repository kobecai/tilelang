from types import SimpleNamespace

import pytest

from tilelang.jit.kernel import JITKernel


class _RecordingRuntimeModule:
    def __init__(self):
        self.exported_paths = []

    def export_library(self, path):
        self.exported_paths.append(path)


def _make_kernel(*, execution_backend, artifact, libpath=None):
    kernel = JITKernel.__new__(JITKernel)
    kernel.execution_backend = execution_backend
    kernel.artifact = artifact
    kernel.adapter = SimpleNamespace(libpath=libpath)
    return kernel


def test_disk_cached_tvm_ffi_library_is_copied(tmp_path):
    cached_library = tmp_path / "cache" / "executable.so"
    cached_library.parent.mkdir()
    cached_library.write_bytes(b"cached-library")
    exported_library = tmp_path / "exports" / "kernel.so"
    kernel = _make_kernel(execution_backend="tvm_ffi", artifact=None, libpath=str(cached_library))

    kernel.export_library(str(exported_library))

    assert exported_library.read_bytes() == b"cached-library"


def test_disk_cached_tvm_ffi_library_can_be_exported_to_same_path(tmp_path):
    cached_library = tmp_path / "executable.so"
    cached_library.write_bytes(b"cached-library")
    kernel = _make_kernel(execution_backend="tvm_ffi", artifact=None, libpath=str(cached_library))

    kernel.export_library(str(cached_library))

    assert cached_library.read_bytes() == b"cached-library"


def test_cold_tvm_ffi_library_still_exports_runtime_module(tmp_path):
    runtime_module = _RecordingRuntimeModule()
    kernel = _make_kernel(
        execution_backend="tvm_ffi",
        artifact=SimpleNamespace(rt_mod=runtime_module),
    )
    exported_library = tmp_path / "exports" / "kernel.so"

    kernel.export_library(str(exported_library))

    assert runtime_module.exported_paths == [str(exported_library)]
    assert exported_library.parent.is_dir()


def test_non_tvm_ffi_library_path_is_not_exported(tmp_path):
    backend_library = tmp_path / "kernel.cubin"
    backend_library.write_bytes(b"backend-library")
    exported_library = tmp_path / "kernel.so"
    kernel = _make_kernel(execution_backend="nvrtc", artifact=None, libpath=str(backend_library))

    with pytest.raises(AttributeError, match="tvm_ffi"):
        kernel.export_library(str(exported_library))

    assert not exported_library.exists()


def test_missing_cached_tvm_ffi_library_preserves_export_error(tmp_path):
    exported_library = tmp_path / "kernel.so"
    kernel = _make_kernel(
        execution_backend="tvm_ffi",
        artifact=None,
        libpath=str(tmp_path / "missing.so"),
    )

    with pytest.raises(AttributeError, match="tvm_ffi"):
        kernel.export_library(str(exported_library))

    assert not exported_library.exists()
