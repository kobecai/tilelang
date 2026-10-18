import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tvm import tirx


def _cuda_source(func):
    func = func.with_attr("global_symbol", "scalar_int2_cast")
    func = func.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    build = tvm.get_global_func("target.build.tilelang_cuda_without_compile")
    return build(tvm.IRModule({"main": func}), tvm.target.Target({"kind": "cuda", "arch": "sm_80"})).inspect_source()


def _cast_source(dtype, source_dtype="int8", materialize=False):
    source = tirx.decl_buffer((1,), source_dtype, name="A")
    output = tirx.decl_buffer((1,), source_dtype, name="C")
    value = tirx.Cast(dtype, source[0])
    if materialize:
        temporary = tirx.Var("narrow", dtype)
        body = tirx.SeqStmt([tirx.Bind(temporary, value), output.vstore([0], tirx.Cast(source_dtype, temporary))])
    else:
        body = output.vstore([0], tirx.Cast(source_dtype, value))
    return _cuda_source(tirx.PrimFunc([source.data, output.data], body, buffer_map={source.data: source, output.data: output}))


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int2", "uint2"])
@pytest.mark.parametrize("source_dtype", ["int8", "uint8", "int32", "int64", "uint64"])
def test_scalar_int2_cast_codegen(dtype, source_dtype):
    source = _cast_source(dtype, source_dtype, materialize=True)
    assert "& 3u" in source
    assert "unsigned int" in source
    assert ("^ 2" in source) == (dtype == "int2")


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int2", "uint2"])
def test_scalar_int2_cast_evaluates_operand_once(dtype):
    value = tirx.call_extern("int32", "next_value")
    func = tirx.PrimFunc([], tirx.Evaluate(tirx.Cast("int32", tirx.Cast(dtype, value))))
    assert _cuda_source(func).count("next_value()") == 1


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int1", "int4", "uint4", "int8"])
def test_scalar_int2_cast_existing_widths(dtype):
    assert "& 3u" not in _cast_source(dtype)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", ["int2", "uint2"])
@pytest.mark.parametrize("use", ["buffer", "allocation", "pointer", "parameter", "vector", "arithmetic", "float", "call", "annotated"])
def test_scalar_int2_cast_rejects_unsupported_uses(dtype, use):
    params = []
    buffer_map = {}
    if use == "buffer":
        buffer = tirx.decl_buffer((4,), dtype, name="packed")
        params = [buffer.data]
        buffer_map = {buffer.data: buffer}
        body = tirx.Evaluate(buffer[0])
        match = "packed 2-bit buffers"
    elif use == "allocation":
        buffer = tirx.decl_buffer((4,), dtype, name="packed", scope="local")
        body = tirx.AllocBuffer(buffer)
        match = "packed 2-bit buffers"
    elif use == "pointer":
        pointer = tirx.Var("packed", tvm.ir.PointerType(tvm.ir.PrimType(dtype), "global"))
        params = [pointer]
        body = tirx.Evaluate(0)
        match = "packed 2-bit buffers"
    elif use == "parameter":
        value = tirx.Var("value", dtype)
        params = [value]
        body = tirx.Evaluate(tirx.Cast("int32", value))
        match = "2-bit kernel parameters"
    elif use == "vector":
        value = tirx.Broadcast(tirx.IntImm("int8", 1), 8)
        body = tirx.Evaluate(tirx.Cast("int8x8", tirx.Cast(f"{dtype}x8", value)))
        match = "vector expressions"
    elif use == "arithmetic":
        value = tirx.Add(tirx.IntImm(dtype, 1), tirx.IntImm(dtype, 1))
        body = tirx.Evaluate(tirx.Cast("int32", value))
        match = "arithmetic"
    elif use == "float":
        value = tirx.Var("value", "float32")
        params = [value]
        body = tirx.Evaluate(tirx.Cast("int32", tirx.Cast(dtype, value)))
        match = "require integer operands"
    elif use == "call":
        body = tirx.Evaluate(tirx.call_extern("int32", "consume", tirx.IntImm(dtype, 1)))
        match = "2-bit call arguments"
    else:
        value = tirx.Var("value", "int32")
        params = [value]
        body = tirx.Evaluate(tirx.Cast(dtype, value, annotations={"sat": False}))
        match = "no rounding or saturation annotations"
    with pytest.raises(ValueError, match=match):
        _cuda_source(tirx.PrimFunc(params, body, buffer_map=buffer_map))


def _scalar_int2_kernel(size, source_dtype, narrow_dtype):
    @T.prim_func
    def main(A: T.Tensor((size,), source_dtype), C: T.Tensor((size,), source_dtype)):
        with T.Kernel(T.ceildiv(size, 128), threads=128) as bx:
            i = bx * 128 + T.get_thread_binding(0)
            if i < size:
                value = T.Cast(narrow_dtype, A[i])
                C[i] = T.Cast(source_dtype, value)

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("narrow_dtype", ["int2", "uint2"])
@pytest.mark.parametrize("source_dtype", ["int8", "uint8", "int32", "int64"])
def test_scalar_int2_cast_runtime(source_dtype, narrow_dtype):
    # Exhaust int8/uint8, then cover values whose upper 32 bits are nonzero.
    values = list(range(256)) if source_dtype == "uint8" else list(range(-128, 128))
    if source_dtype == "int32":
        values += [-(2**31), 2**31 - 1]
    elif source_dtype == "int64":
        values += [-(2**63), 2**63 - 1, 2**32 + 3, -(2**32) - 1]
    # Python's modulo gives the unsigned residue; shift it into the signed
    # interval independently of the bitwise code emitted by the compiler.
    expected = [(value + 2) % 4 - 2 if narrow_dtype == "int2" else value % 4 for value in values]
    torch_dtype = getattr(torch, source_dtype)
    source = torch.tensor(values, dtype=torch_dtype, device="cuda")
    reference = torch.tensor(expected, dtype=torch.int64, device="cuda").to(torch_dtype)
    kernel = tilelang.compile(_scalar_int2_kernel(len(values), source_dtype, narrow_dtype), out_idx=[1], target="cuda")
    torch.testing.assert_close(kernel(source), reference, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
