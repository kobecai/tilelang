# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# ruff: noqa: E402

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from common import import_tvm, print_section, schedule_namespace

if TYPE_CHECKING:
    import numpy as np


# Adapted from:
# https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/tir_transformation.html
#
# The official tutorial imports `tvm` directly.  In this repository we go through
# `common.import_tvm()` so TileLang can bootstrap its bundled TVM/TIRX build first.
tvm = import_tvm()
schedule_ns = schedule_namespace(tvm)

try:
    from tvm.script import ir as I
    from tvm.script import tirx as T
except ImportError as err:
    raise SystemExit(
        "This tutorial uses TVM TIRX (`from tvm.script import tirx as T`). "
        "Run it with TileLang's bundled TVM build."
    ) from err


@I.ir_module
class MyModule:
    @T.prim_func(s_tir=True)
    def main(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
        T.func_attr({"tirx.noalias": True})
        with T.sblock("root"):
            T.reads()
            T.writes()
            Y = T.sblock_alloc_buffer((128, 128))
            for i, j, k in T.grid(128, 128, 128):
                with T.sblock("Y"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        Y[vi, vj] = T.float32(0)
                    Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
            for i, j in T.grid(128, 128):
                with T.sblock("C"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    C[vi, vj] = T.max(Y[vi, vj], T.float32(0))


def tensor_from_numpy(array: np.ndarray):
    if hasattr(tvm.runtime, "tensor"):
        return tvm.runtime.tensor(array)
    return tvm.nd.array(array)


def import_numpy():
    try:
        import numpy as np

        return np
    except ModuleNotFoundError as err:
        raise SystemExit(
            "This tutorial needs `numpy` for input generation, correctness checks, and timing. "
            "Install numpy or rerun with `--no-evaluate` to inspect only the schedule steps."
        ) from err


def ndarray_to_numpy(array):
    if hasattr(array, "numpy"):
        return array.numpy()
    return array.asnumpy()


def build_tirx(mod, target: str):
    if not hasattr(tvm, "tirx") or not hasattr(tvm.tirx, "build"):
        raise SystemExit("This tutorial requires `tvm.tirx.build`, which is missing from this TVM build.")
    return tvm.tirx.build(mod, target=target)


def device_for_target(target: str):
    kind = tvm.target.Target(target).kind.name
    if kind == "cuda":
        return tvm.cuda()
    if kind in {"rocm", "hip"}:
        return tvm.rocm()
    if kind == "metal":
        return tvm.metal()
    return tvm.cpu()


def get_sblock(sch, name: str):
    if hasattr(sch, "get_sblock"):
        return sch.get_sblock(name)
    try:
        return sch.get_block(name, func_name="main")
    except TypeError:
        return sch.get_block(name)


def show_obj(obj, title: str, enabled: bool) -> None:
    if not enabled:
        return

    print_section(title)
    show = getattr(obj, "show", None)
    if show is not None:
        show()
    elif hasattr(obj, "script"):
        print(obj.script())
    else:
        print(obj)


def make_inputs(seed: int):
    np = import_numpy()
    rng = np.random.default_rng(seed)
    a_np = rng.uniform(size=(128, 128)).astype("float32")
    b_np = rng.uniform(size=(128, 128)).astype("float32")
    c_np = a_np @ b_np

    return a_np, b_np, c_np


def evaluate(mod, a_np: np.ndarray, b_np: np.ndarray, c_np: np.ndarray, target: str) -> None:
    np = import_numpy()
    lib = build_tirx(mod, target=target)

    a_nd = tensor_from_numpy(a_np)
    b_nd = tensor_from_numpy(b_np)
    c_nd = tensor_from_numpy(np.zeros((128, 128), dtype="float32"))

    lib(a_nd, b_nd, c_nd)
    np.testing.assert_allclose(ndarray_to_numpy(c_nd), c_np, rtol=1e-5)

    f_timer = lib.time_evaluator("main", device_for_target(target))
    print(f_timer(a_nd, b_nd, c_nd))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the official TVM TensorIR transformation tutorial with TileLang's TVM import path."
    )
    parser.add_argument("--target", default="llvm", help="Tutorial build target. The official tutorial uses llvm.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible input tensors.")
    parser.add_argument("--no-evaluate", action="store_true", help="Skip build, correctness, and timing steps.")
    parser.add_argument("--no-show", action="store_true", help="Skip printing intermediate IR and trace output.")
    args = parser.parse_args()

    should_show = not args.no_show
    should_evaluate = not args.no_evaluate
    a_np = b_np = c_np = None

    if should_evaluate:
        a_np, b_np, c_np = make_inputs(args.seed)

    if should_evaluate:
        print_section("Evaluate original MyModule")
        assert a_np is not None and b_np is not None and c_np is not None
        evaluate(MyModule, a_np, b_np, c_np, args.target)

    print_section("Initialization Schedule")
    sch = schedule_ns.Schedule(MyModule)

    print_section("Loop Tiling")
    block_Y = get_sblock(sch, "Y")
    i, j, k = sch.get_loops(block_Y)
    print("loops:", i, j, k)

    j0, j1 = sch.split(j, factors=[None, 8])
    print("split j into:", j0, j1)
    show_obj(sch.mod, "After split(j, factors=[None, 8])", should_show)

    sch.reorder(j0, k, j1)
    show_obj(sch.mod, "After reorder(j0, k, j1)", should_show)
    if should_evaluate:
        print_section("Evaluate after split + reorder")
        assert a_np is not None and b_np is not None and c_np is not None
        evaluate(sch.mod, a_np, b_np, c_np, args.target)

    print_section("Leverage Localities")
    block_C = get_sblock(sch, "C")
    sch.reverse_compute_at(block_C, j0)
    show_obj(sch.mod, "After reverse_compute_at(C, j0)", should_show)

    print_section("Rewrite Reduction")
    sch.decompose_reduction(block_Y, k)
    show_obj(sch.mod, "After decompose_reduction(Y, k)", should_show)
    if should_evaluate:
        print_section("Evaluate after decompose_reduction")
        assert a_np is not None and b_np is not None and c_np is not None
        evaluate(sch.mod, a_np, b_np, c_np, args.target)

    print_section("Trace the Transformation")
    show_obj(sch.trace, "sch.trace", should_show)
    show_obj(sch, "sch.show()", should_show)


if __name__ == "__main__":
    main()
