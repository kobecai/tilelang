"""Writer-side proxy fences must precede synchronization with a TMA issuer."""

import pytest

import tilelang
import tilelang.testing
from tilelang import tvm
from tvm import tirx


def _call(name, args=(), dtype="handle"):
    return tirx.Call(dtype, tvm.ir.Op.get(name), list(args))


def _make_module(scope="shared", sync="none", store_op="tl.tma_store", consumer_only=False):
    tx = tirx.Var("tx", "int32")
    shared = tirx.decl_buffer((128,), "float32", name="S", scope=scope)
    output = tirx.decl_buffer((128,), "float32", name="O")
    index = tx - 128 if consumer_only else tx
    writer = tirx.BufferStore(shared, tirx.Cast("float32", tx), [index])
    store = tirx.Evaluate(_call(store_op, [output.access_ptr("w"), shared.access_ptr("r"), 512]))
    leader = tirx.IfThenElse(_call("tl.tl_shuffle_elect", [128], "bool"), store, None)
    region = [writer]
    if sync != "none":
        args = [tirx.StringImm(scope)]
        if sync == "partial":
            args.extend([tirx.IntImm("int32", 3), tirx.IntImm("int32", 128)])
        region.append(tirx.Evaluate(_call("tirx.tvm_storage_sync", args, "int32")))
    region.append(leader)
    body = tirx.SeqStmt(region)
    if consumer_only:
        body = tirx.IfThenElse(tx >= 128, body, None)
    threads = 256 if consumer_only else 128
    thread = tirx.IterVar(tvm.ir.Range(0, threads), tx, tirx.IterVar.ThreadIndex, "threadIdx.x")
    body = tirx.AttrStmt(thread, "thread_extent", threads, tirx.SeqStmt([tirx.AllocBuffer(shared), body]))
    func = tirx.PrimFunc([output.data], body, buffer_map={output.data: output})
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"})
    return tvm.IRModule({"main": func.with_attr("target", target)}), shared


def _events(stmt, conditions=()):
    if isinstance(stmt, tirx.SeqStmt):
        return [event for child in stmt.seq for event in _events(child, conditions)]
    if isinstance(stmt, tirx.IfThenElse):
        events = _events(stmt.then_case, conditions + (str(stmt.condition),))
        if stmt.else_case is not None:
            events += _events(stmt.else_case, conditions + ("else",))
        return events
    if isinstance(stmt, tirx.AttrStmt):
        return _events(stmt.body, conditions)
    if isinstance(stmt, tirx.BufferStore):
        return [("write", conditions)]
    if isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call):
        name = stmt.value.op.name
        if name in ("tl.fence_proxy_async", "tirx.tvm_storage_sync", "tl.tma_store", "tl.tma_store_scatter4"):
            return [(name, conditions)]
    return []


def _lower(mod):
    mod = tilelang.cuda.transform.InjectFenceProxy()(mod)
    mod = tilelang.transform.ThreadSync("shared")(mod)
    return tilelang.transform.ThreadSync("shared.dyn")(mod)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("scope", ["shared", "shared.dyn"])
@pytest.mark.parametrize("sync", ["none", "plain", "partial"])
@pytest.mark.parametrize("store_op", ["tl.tma_store", "tl.tma_store_scatter4"])
@pytest.mark.parametrize("consumer_only", [False, True])
def test_tma_store_writer_fence_before_barrier(scope, sync, store_op, consumer_only):
    mod, _ = _make_module(scope, sync, store_op, consumer_only)
    events = _events(_lower(mod)["main"].body)
    assert [name for name, _ in events] == ["write", "tl.fence_proxy_async", "tirx.tvm_storage_sync", store_op]
    writer_conditions = events[0][1]
    assert events[1][1] == writer_conditions
    assert events[2][1] == writer_conditions
    assert len(events[3][1]) == len(writer_conditions) + 1
    assert "tl_shuffle_elect" in events[3][1][-1]


@tilelang.testing.requires_cuda
def test_tma_store_fence_does_not_cross_a_later_shared_write():
    mod, shared = _make_module(sync="plain")

    def add_write(node):
        if isinstance(node, tirx.IfThenElse):
            return tirx.SeqStmt([tirx.BufferStore(shared, tirx.FloatImm("float32", 42), [0]), node])
        return None

    body = tirx.stmt_functor.ir_transform(mod["main"].body, None, add_write)
    mod = tvm.IRModule({"main": mod["main"].with_body(body)})
    events = _events(_lower(mod)["main"].body)
    assert [name for name, _ in events][-4:] == ["write", "tl.fence_proxy_async", "tirx.tvm_storage_sync", "tl.tma_store"]


@tilelang.testing.requires_cuda
def test_tma_store_fence_crosses_only_read_only_bindings_before_barrier():
    mod, _ = _make_module(sync="plain")

    def add_binding(node):
        if isinstance(node, tirx.IfThenElse):
            return tirx.SeqStmt([tirx.Bind(tirx.Var("offset", "int32"), tirx.IntImm("int32", 16)), node])
        return None

    body = tirx.stmt_functor.ir_transform(mod["main"].body, None, add_binding)
    mod = tvm.IRModule({"main": mod["main"].with_body(body)})
    events = _events(_lower(mod)["main"].body)
    assert [name for name, _ in events] == ["write", "tl.fence_proxy_async", "tirx.tvm_storage_sync", "tl.tma_store"]


@tilelang.testing.requires_cuda
def test_tma_store_fence_is_not_hoisted_before_a_write_inside_the_leader():
    mod, shared = _make_module()

    def move_write(node):
        if isinstance(node, tirx.BufferStore):
            return tirx.Evaluate(0)
        if isinstance(node, tirx.IfThenElse):
            write = tirx.BufferStore(shared, tirx.FloatImm("float32", 42), [0])
            return tirx.IfThenElse(node.condition, tirx.SeqStmt([write, node.then_case]), None)
        return None

    body = tirx.stmt_functor.ir_transform(mod["main"].body, None, move_write)
    mod = tvm.IRModule({"main": mod["main"].with_body(body)})
    events = _events(tilelang.cuda.transform.InjectFenceProxy()(mod)["main"].body)
    assert [name for name, _ in events] == ["write", "tl.fence_proxy_async", "tl.tma_store"]
    assert events[0][1] == events[1][1] == events[2][1]


@tilelang.testing.requires_cuda
def test_tma_store_without_generic_writes_does_not_add_a_fence():
    mod, _ = _make_module()

    def remove_write(node):
        if isinstance(node, tirx.BufferStore):
            return tirx.Evaluate(0)
        return None

    body = tirx.stmt_functor.ir_transform(mod["main"].body, None, remove_write)
    mod = tvm.IRModule({"main": mod["main"].with_body(body)})
    events = _events(tilelang.cuda.transform.InjectFenceProxy()(mod)["main"].body)
    assert [name for name, _ in events] == ["tl.tma_store"]
