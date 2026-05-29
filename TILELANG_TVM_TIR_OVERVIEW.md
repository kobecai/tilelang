# TileLang 使用 TVM/TIR 的组件与学习路线

## 总览

TileLang 是建立在 TVM/TIR 之上的 DSL + lowering + codegen 系统。它自己定义了 `T.copy`、`T.gemm`、layout、fragment、pipeline、GPU tile op 等高层语义，但底层程序表示、IR 容器、pass 框架、Target、FFI、runtime module、host/device split 和后端 codegen 基本都沿用 TVM/TIR 体系。

核心链路是：

```text
TileLang Python DSL
-> tvm.script.tirx / tirx.PrimFunc
-> tvm.IRModule
-> TileLang lowering passes
-> tirx/s_tir/tl transform passes
-> host/device IR split
-> target.build.tilelang_cuda / hip / metal / llvm
-> TVM runtime module or Cython/NVRTC adapter
```

## TileLang 用到的 TVM/TIR 组件

### 1. `IRModule` / `PrimFunc` / TIR 节点

TileLang kernel 最终会变成 `tirx.PrimFunc` 或 `tvm.IRModule`。在 `tilelang/engine/lower.py`，如果输入是 `tirx.PrimFunc`，TileLang 会取它的 `global_symbol`，包装成：

```python
tvm.IRModule({func.attrs["global_symbol"]: func})
```

然后执行：

```text
PreLowerSemanticCheck
-> LowerAndLegalize
-> OptimizeForTarget
-> Filter(host)
-> Filter(device)
```

所以 TileLang 不是直接从 Python 生成 CUDA，而是先生成 TIR，再通过 pass pipeline 降到可 codegen 的 host/device IR。

### 2. `tirx` / `s_tir`

当前仓库使用的是带 `tirx` / `s_tir` 分层的 TVM。官方 TensorIR 文档也把 TensorIR 分为：

- `tirx`：核心 IR 定义和 lowering，包括 `PrimFunc`、`Buffer`、`SBlock`、表达式、语句、lowering passes。
- `s_tir`：schedulable TIR，包括 schedule、MetaSchedule、DLight、tensor intrinsics 等。

TileLang 里大量直接使用：

```python
from tvm import tirx, s_tir, IRModule
```

例如 `tilelang/engine/phase.py` 使用 `tirx.transform.BindTarget`、`tirx.transform.NarrowDataType`、`s_tir.transform.RenormalizeSplitPattern`、`s_tir.transform.InferFragment`。

简单说：TileLang 的 IR 基座是 `tirx`，部分 schedulable / tensor intrinsic 相关能力来自 `s_tir`，TileLang 自己再加 `tl.*` 对象和 `tl.transform.*` pass。

### 3. TVM pass 框架

TileLang 编译过程本质是 TVM pass pipeline。最核心文件是 `tilelang/engine/phase.py`。

`LowerAndLegalize` 主要把 TileLang 前端语义降成更标准、更低层的 TIR：

```text
BindTarget
AddWrapperForSingleBufStore
LegalizeNegativeIndex
VerifyParallelLoop
InjectAssumes
Simplify
LayoutReducer
ProducerConsumerWarpSpecialized
LowerBlackwell2SM
PipelinePlanning
InjectSoftwarePipeline
LayoutInference
LowerTileOp
LowerL2Persistent
DecoupleTypeCast
LegalizeVectorizedLoop
LegalizeSafeMemoryAccess
LowerAccessPtr
HoistNonRestrictParams
```

`OptimizeForTarget` 则偏后端化和 codegen 前准备：

```text
LowerSharedTmem
PlanAndUpdateBufferAllocationLocation
LowerSharedBarrier
FlattenBuffer
ConfigIndexBitwidth
VectorizeLoop
StorageRewrite
UnrollLoop
RenormalizeSplitPattern
VerifyMemory
InferFragment
LowerThreadAllreduce
LowerLDGSTG
LowerHopperIntrin
AnnotateDeviceRegions
SplitHostDevice
ThreadSync
MakePackedAPI
LowerDeviceKernelLaunch
PersistThreadblock
```

注意这里很多不是 TVM 原版 pass，而是 TileLang 自己实现的 `tl.transform.*` pass，只是它们遵守 TVM pass 体系：输入 `IRModule`，输出新的 `IRModule`。

### 4. `PassContext`

JIT 编译时，TileLang 会进入 TVM 的 `PassContext`。在 `tilelang/jit/kernel.py`：

```python
with tvm.transform.PassContext(opt_level=3, config=pass_configs, instruments=pass_instruments), self.target:
    artifact = tilelang.lower(...)
```

这里 `PassContext` 管 pass config、IR dump instrument、优化等级等。TileLang 的很多开关，比如 dump IR、layout visualization、disable vectorize、disable race check，也都是通过 pass config 影响 pipeline。

### 5. `Target` / `TargetContext`

TileLang 强依赖 TVM `Target`。Target 不只是 `"cuda"` 字符串，而是目标设备能力描述。比如 `tilelang/utils/target.py` 负责解析：

```text
auto
cuda
hip
metal
llvm
webgpu
c
cutedsl
{"kind": "cuda", "arch": "sm_80"}
{"kind": "hip", "mcpu": "gfx942"}
```

C++ 侧 `src/target/utils.cc` 根据 Target 判断：

```text
CUDA / ROCm / Metal / CPU
Volta / Turing / Ampere / Hopper / Blackwell
async copy
ldmatrix / stmatrix
TMA / bulk copy
TMEM
warp size
```

所以 Target 会影响 GEMM 选 MMA/WGMMA/TCGEN5MMA，copy 选普通 SIMT/cp.async/TMA/ldmatrix/stmatrix，pipeline 是否启用 warp specialization 等。

如果遇到 `Target context required`，本质就是某些 pass 或 helper 调用了 `Target::Current(false)`，但当前线程没有进入 `with target:` 作用域。JIT 正常路径里已经通过 `with ..., self.target:` 处理了。

### 6. Buffer / BufferRegion / Range / IndexMap / Layout / Fragment

TileLang 对 TIR buffer 用得很深。比如 `tilelang/language/allocate.py`：

```python
alloc_shared -> scope="shared.dyn"
alloc_local -> scope="local"
alloc_fragment -> scope="local.fragment"
alloc_barrier -> scope="shared.barrier"
alloc_tmem -> scope="shared.tmem"
```

这些最终都是 TVM/TIR buffer，只是 TileLang 用不同 memory scope 表达 GPU 内存层级。

`T.copy` 在 `tilelang/language/copy_op.py` 会把输入规范化成 `BufferRegion`，然后生成：

```python
tirx.call_intrin("handle", tirx.op.Op.get("tl.tileop.copy"), src, dst, ...)
```

Layout 系统在 `tilelang/layout/layout.py` 和 `tilelang/layout/fragment.py`。它注册成 TVM object：`tl.Layout`、`tl.Fragment`，内部用 `IterVar`、`Range`、`IndexMap` 表达逻辑 index 到线程、寄存器、物理布局的映射。

### 7. Tile op 机制

TileLang 的核心抽象是 tile op。`T.gemm`、`T.copy`、`T.reduce`、`T.fill` 等不是普通 Python 函数，而是生成 `tl.tileop.*` intrinsic call。

例如 `tilelang/language/gemm_op.py`：

```python
tirx.call_intrin(
    "handle",
    tirx.op.Op.get("tl.tileop.gemm"),
    ...
)
```

C++ 侧 `src/op/operator.h` 定义了统一接口：

```cpp
TileOperatorNode::Lower
TileOperatorNode::InferLayout
TileOperatorNode::Clone
TileOperatorNode::GetAccessRegions
```

`src/op/operator.cc` 会通过 `TLOpBuilder` 从 TIR `Call` 解析出具体 TileOperator。然后 `src/transform/lower_tile_op.cc` 在 `LowerTileOp` pass 里调用 `tile_op->Lower(...)`。

以 GEMM 为例，`src/op/gemm.cc` 会调用注册在 Python 侧的 `tl.gemm.lower`，再根据 target 和 shape 分派到 CUDA/HIP/CPU 具体实现。CUDA 侧 `src/backend/cuda/op/gemm.cc` 注册 CUDA GEMM 实现，选择 MMA、WGMMA 或 TCGEN5MMA。

### 8. FFI / PackedFunc / object registry

TileLang 大量使用 TVM FFI。Python 侧 `tilelang/transform/__init__.py` 这种函数基本只是 wrapper：

```python
def LayoutInference():
    return _ffi_api.LayoutInference()
```

真正实现多在 C++，通过 `GlobalDef().def("tl.transform.LayoutInference", LayoutInference)` 注册。Layout、Fragment、Gemm 也通过 `@tvm_ffi.register_object(...)` 注册成 TVM object。

这就是 TVM 的典型设计：C++ 实现 IR/pass/codegen，Python 通过 FFI 调用，IR object 可以跨 Python/C++ 传递。

### 9. Host/device split 和 runtime

TileLang 在 lowering 后会拆分 host/device IR：

```python
host_mod = tirx.transform.Filter(_is_host_call)(mod)
device_mod = tirx.transform.Filter(_is_device_call)(mod)
```

codegen 分两类：

- host：`target.build.llvm` 或 `target.build.tilelang_c_host`
- device：`target.build.tilelang_cuda`、`target.build.tilelang_hip`、`target.build.tilelang_metal`

如果用 `tvm_ffi` backend，最后会生成 TVM runtime module，并由 `tilelang/jit/adapter/tvm_ffi.py` 包成可接收 PyTorch tensor 的 callable。

## TVM/TIR 核心设计科普

TVM 的核心思想是：

```text
前端表达计算
-> IR 表达程序
-> pass pipeline 优化/改写 IR
-> Target 决定目标硬件能力
-> codegen 生成目标代码
-> runtime module 执行
```

TIR 是 TVM 里偏低层的 tensor program IR。你可以把它理解成“比 CUDA C 更结构化、比 LLVM IR 更接近张量循环程序”的中间表示。

几个最重要的概念：

- `IRModule`：编译单元，装一组函数。
- `PrimFunc`：低层 primitive function，通常对应一个 kernel 或 host wrapper。
- `PrimExpr`：表达式，比如 index、加减乘除、条件、call。
- `Stmt`：语句，比如 `For`、`IfThenElse`、`BufferStore`、`Evaluate`。
- `Buffer`：张量内存抽象，带 shape、dtype、stride、scope。
- `BufferRegion`：buffer 的访问区域，用于 read/write region 分析。
- `SBlock` / `SBlockRealize`：TensorIR 里的计算块，描述一段计算的读写区域、迭代变量和调度边界。
- `Call` / `call_intrin`：调用 intrinsic 或外部函数。TileLang 的 `tl.tileop.*` 就是靠它挂进 TIR。
- `Target`：目标硬件和 codegen 配置，比如 CUDA arch、ROCm mcpu、host target。
- `Pass` / `PassContext`：IR 变换和变换配置。TileLang 编译基本就是一串 pass。
- `PackedFunc` / FFI：Python/C++/runtime 互调机制。
- `StructuralEqual`：IR 结构相等判断，不依赖 Python 对象 identity。

TileLang 的特殊点在于：它没有直接让用户写标准 TIR schedule，而是定义了更高层的 tile op 和 layout 系统。比如 `T.gemm` 先作为 `tl.tileop.gemm` 留在 TIR 里，等 `LayoutInference`、`LowerTileOp`、target-specific pass 拿到足够上下文后，再决定降成 MMA、WGMMA、TCGEN5MMA 或其他实现。

## 最值得优先读的 TileLang 代码

按优先级：

1. `tilelang/engine/phase.py`

   看完整 lowering pipeline，理解每个 pass 的顺序。

2. `tilelang/engine/lower.py`

   看 `PrimFunc/IRModule -> host/device IR -> codegen`。

3. `tilelang/language/copy_op.py`、`tilelang/language/gemm_op.py`

   看 Python DSL 怎么生成 `tl.tileop.*`。

4. `src/op/operator.h`、`src/op/operator.cc`

   看 tile op 抽象和解析机制。

5. `src/transform/lower_tile_op.cc`

   看 tile op 真正在哪里被降级。

6. `src/op/gemm.cc`、`tilelang/tileop/gemm/__init__.py`、`src/backend/cuda/op/gemm.cc`

   看 GEMM 从 tile op 到后端指令选择。

7. `tilelang/layout/layout.py`、`tilelang/layout/fragment.py`、`src/transform/layout_inference.cc`

   看 layout/fragment inference。

8. `docs/tutorials/debug_tools_for_tilelang.md`

   看如何 dump IR、看中间阶段、调试 lowering。

9. `docs/get_started/targets.md`

   看 TileLang target 语义和 `sm_80/sm_90` 等配置。

## 建议看的 TVM/TIR 官方资料

按这个顺序看：

1. TensorIR overview

   https://tvm.apache.org/docs/deep_dive/tensor_ir/index.html

2. TensorIR abstraction

   https://tvm.apache.org/docs/deep_dive/tensor_ir/abstraction.html

3. TensorIR transformation tutorial

   https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/tir_transformation.html

4. `tvm.tirx` Python API

   https://tvm.apache.org/docs/reference/api/python/tirx/tirx.html

5. `tvm.tirx.transform` API

   https://tvm.apache.org/docs/reference/api/python/tirx/transform.html

6. `tvm.s_tir.transform` API

   https://tvm.apache.org/docs/reference/api/python/s_tir/transform.html

7. TVM pass infra

   https://tvm.apache.org/docs/reference/api/python/transform.html

8. TVM Target API

   https://tvm.apache.org/docs/reference/api/python/target.html

9. TVM runtime / PackedFunc

   https://tvm.apache.org/docs/arch/runtime.html

10. Device/Target interactions

    https://tvm.apache.org/docs/arch/device_target_interactions.html

最后再看源码：

```text
3rdparty/tvm/include/tvm/tirx/
3rdparty/tvm/python/tvm/tirx/
3rdparty/tvm/python/tvm/s_tir/
3rdparty/tvm/src/tirx/
3rdparty/tvm/src/s_tir/
3rdparty/tvm/src/target/
3rdparty/tvm/src/runtime/
```

## 当前最该掌握的三条主线

第一，`PrimFunc/IRModule/Buffer/SBlock` 怎么表达 kernel。

第二，`PassContext + transform pass` 怎么一步步改写 IR。

第三，`Target/TargetContext` 怎么影响 pass 和 codegen。

TileLang 里多数“神秘问题”，包括 layout 推不出来、A100/H100/Blackwell 行为不同、target context 报错、GEMM 选错后端路径，最后都会落到这三条线上。
