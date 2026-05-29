# TileLang Kernel Execution Path

本文记录从 Python 命令运行一个 TileLang kernel 时，代码大致经过哪些层，以及每一层最关键的代码入口。例子可以参考 [examples/quickstart.py](examples/quickstart.py#L8-L68)：`@tilelang.jit` 定义 kernel builder，`matmul(...)` 触发编译，随后 `matmul_relu_kernel(a, b, c)` 触发运行。

## 总览

粗粒度路径可以理解为：

```text
Python DSL
  -> TIR PrimFunc
  -> semantic check
  -> LowerAndLegalize
  -> layout inference
  -> LowerTileOp
  -> OptimizeForTarget
  -> host/device split
  -> backend codegen
  -> runtime adapter
```

源码里的真实路径有几个补充点：

- `host/device split` 实际发生在 `OptimizeForTarget()` 内部的 `AnnotateDeviceRegions()` 和 `SplitHostDevice()`，后面还会继续跑 readonly/shared-memory/API/runtime 相关 pass。
- `@tilelang.jit` 有 lazy 和 eager 两种模式。lazy 模式通常是 Python 函数返回一个内部 `@T.prim_func`；eager 模式则直接 trace DSL builder。
- `tilelang.compile(prim_func, ...)` 可以绕过 Python DSL 阶段，直接从已经构造好的 `PrimFunc` 进入 cache/JIT/lower/codegen。
- 默认 execution backend 会通过 target 自动解析。CUDA/HIP/Metal 默认通常走 `tvm_ffi`，非 GPU fallback 通常走 `cython`。
- `tvm_ffi` backend 会在 `tilelang.lower()` 里同时生成 host runtime module 和 device module；`cython`/`nvrtc` 等 backend 通常只在 lower 阶段拿 device source，后续由 adapter 自己编译或装载。

### 快速全貌

这一节是先看全貌用的最短路径，粒度刚好能把整条链路串起来。后面的章节再展开每一段内部细节。

**总路径**

```text
python script
  -> import tilelang, load libtilelang.so, 注册 FFI/pass/op/codegen
  -> @tilelang.jit 包装 Python DSL 函数
  -> 第一次调用 JITImpl.__call__
  -> Python DSL / T.prim_func 生成 tvm.tir.PrimFunc
  -> cache lookup 或 JITKernel 编译
  -> tilelang.lower
      -> PreLowerSemanticCheck
      -> LowerAndLegalize
          -> LayoutInference
          -> LowerTileOp
      -> OptimizeForTarget
          -> AnnotateDeviceRegions
          -> SplitHostDevice
          -> MakePackedAPI / LowerDeviceKernelLaunch
      -> host_mod / device_mod
  -> backend codegen
  -> adapter 包成 Python callable
  -> kernel(*torch_tensors) 执行
```

**入口层**

`import tilelang` 会加载 TileLang C++ 动态库并导入 `jit`、`engine`、`transform`、`tileop` 等模块，入口在 [tilelang/__init__.py](tilelang/__init__.py#L153)。

典型例子是 [examples/quickstart.py](examples/quickstart.py#L8) 的 `@tilelang.jit`，第一次调用在 [examples/quickstart.py](examples/quickstart.py#L58)，真正执行 kernel 在 [examples/quickstart.py](examples/quickstart.py#L68)。

`@tilelang.jit` 入口在 [tilelang/jit/__init__.py](tilelang/jit/__init__.py#L493)，内部把 Python 函数转成 `JITFunc`，返回 `JITImpl`，关键位置在 [tilelang/jit/__init__.py](tilelang/jit/__init__.py#L543)。

**Python DSL -> TIR PrimFunc**

`T.prim_func` 的入口是 [tilelang/language/eager/builder.py](tilelang/language/eager/builder.py#L1236)。它调用 `mutate(func)` 做 Python AST 改写，入口在 [tilelang/language/eager/ast.py](tilelang/language/eager/ast.py#L647)，再用 `Builder` 和 TVM IRBuilder 生成 `tvm.tir.PrimFunc`。

`T.Kernel(...)` 是 kernel launch DSL 入口：[tilelang/language/kernel.py](tilelang/language/kernel.py#L266)。它调用 `_ffi_api.KernelLaunch`，C++ 侧在 [src/ir.cc](src/ir.cc#L261) 创建 `blockIdx`/`threadIdx` launch frame。

`T.copy` 和 `T.gemm` 这时还没有变成 CUDA 指令，只是生成高层 tile op：

- `T.copy -> tl.tileop.copy`：[tilelang/language/copy_op.py](tilelang/language/copy_op.py#L51)
- `T.gemm -> tl.tileop.gemm`：[tilelang/language/gemm_op.py](tilelang/language/gemm_op.py#L149)

**编译触发**

第一次调用装饰后的对象进入 `JITImpl.__call__`：[tilelang/jit/__init__.py](tilelang/jit/__init__.py#L434)。lazy 模式返回 `JITKernel`，eager 模式会立即执行。

编译链路是：

- `JITImpl.compile`：[tilelang/jit/__init__.py](tilelang/jit/__init__.py#L389)
- `tilelang.jit.compile`：[tilelang/jit/__init__.py](tilelang/jit/__init__.py#L43)
- `cache.cached` 解析 target/backend：[tilelang/cache/__init__.py](tilelang/cache/__init__.py#L30)
- cache miss 时创建 `JITKernel`：[tilelang/cache/kernel_cache.py](tilelang/cache/kernel_cache.py#L351)

`JITKernel` 的核心编译入口是 `_compile_and_create_adapter`：[tilelang/jit/kernel.py](tilelang/jit/kernel.py#L203)，里面调用 `tilelang.lower(...)`：[tilelang/jit/kernel.py](tilelang/jit/kernel.py#L241)。

**Lower Pipeline**

主入口是 [tilelang/engine/lower.py](tilelang/engine/lower.py#L316)，其中 `lower_to_host_device_ir` 做主要 lowering：[tilelang/engine/lower.py](tilelang/engine/lower.py#L275)。

semantic check 在 lowering 前执行：[tilelang/engine/lower.py](tilelang/engine/lower.py#L301)，对应 [tilelang/engine/phase.py](tilelang/engine/phase.py#L125)，主要调用 `NestedLoopChecker` 和 `FragmentLoopChecker`。

`LowerAndLegalize` 在 [tilelang/engine/phase.py](tilelang/engine/phase.py#L144)。这里完成 target bind、负索引合法化、parallel loop 检查、assume 注入、pipeline planning、layout reducer、`LayoutInference`、`LowerTileOp` 等。你列的 `layout inference -> LowerTileOp` 就在这里：[tilelang/engine/phase.py](tilelang/engine/phase.py#L200)。

`OptimizeForTarget` 在 [tilelang/engine/phase.py](tilelang/engine/phase.py#L227)。注意：代码里 host/device split 是 `OptimizeForTarget()` 内部的一步，不是它之后的独立顶层 phase，关键位置在 [tilelang/engine/phase.py](tilelang/engine/phase.py#L278)。

**TileOp Lowering**

Python wrapper `tilelang.transform.LayoutInference`/`LowerTileOp` 只是 FFI 壳：[tilelang/transform/__init__.py](tilelang/transform/__init__.py#L57)。

C++ `LayoutInference` 在 [src/transform/layout_inference.cc](src/transform/layout_inference.cc#L1284)，会收集 buffer use/def，给 block 和 parallel loop 附 layout annotation。

`LowerTileOp` 在 [src/transform/lower_tile_op.cc](src/transform/lower_tile_op.cc#L1397)。核心逻辑是在 `Evaluate(Call)` 里识别 tile op，然后调用 `tile_op->Lower(...)`：[src/transform/lower_tile_op.cc](src/transform/lower_tile_op.cc#L1030)。

tile op 的解析注册机制在 [src/op/operator.h](src/op/operator.h#L168) 和 [src/op/operator.cc](src/op/operator.cc#L32)。

`T.gemm` 比较特殊：C++ `GemmNode::Lower` 会回调 Python 全局函数 `tl.gemm.lower`：[src/op/gemm.cc](src/op/gemm.cc#L180)，Python 入口在 [tilelang/tileop/gemm/__init__.py](tilelang/tileop/gemm/__init__.py#L18)，再选择 MMA/WGMMA/TCGEN05 等实现。

**Codegen 和 Runtime Adapter**

backend codegen 在 [tilelang/engine/lower.py](tilelang/engine/lower.py#L231)。CUDA 会调用 `target.build.tilelang_cuda` 或 `target.build.tilelang_cuda_without_compile`。

CUDA C++ codegen 入口是 [src/backend/cuda/codegen/rt_mod_cuda.cc](src/backend/cuda/codegen/rt_mod_cuda.cc#L94)，里面 `CodeGenTileLangCUDA.AddFunction -> Finish -> tilelang_callback_cuda_compile`。真正打印 CUDA 函数体在 [src/backend/cuda/codegen/codegen_cuda.cc](src/backend/cuda/codegen/codegen_cuda.cc#L5034)。

runtime backend 默认解析在 [tilelang/jit/execution_backend.py](tilelang/jit/execution_backend.py#L66)：CUDA/HIP/Metal 默认 `tvm_ffi`。`tvm_ffi` adapter 在 [tilelang/jit/adapter/tvm_ffi.py](tilelang/jit/adapter/tvm_ffi.py#L39)，最终 Python callable 会准备输出 tensor，然后调用 `runtime.Executable(*tensor_list)`：[tilelang/jit/adapter/tvm_ffi.py](tilelang/jit/adapter/tvm_ffi.py#L204)。

最后 `JITKernel.__call__` 只是转发到 adapter 的 callable：[tilelang/jit/kernel.py](tilelang/jit/kernel.py#L185)。

## 入口层

### 1. import tilelang

入口文件是 [tilelang/__init__.py](tilelang/__init__.py#L153-L211)。

关键动作：

- 加载 TileLang 动态库：`libinfo.find_lib_path("tilelang")` 和 `ctypes.CDLL(...)`。
- 导出用户常用 API：`jit`、`compile`、`par_compile`、`Profiler`、`lower`、`transform`、`language`、`tileop` 等。
- 导入 `.tileop` 时会注册 Python 侧 tile op lowering 入口，例如 GEMM 的 `tl.gemm.lower` 和 `tl.gemm.infer_layout`。

### 2. Python user code

典型 lazy JIT 例子：

```python
@tilelang.jit
def matmul(...):
    @T.prim_func
    def matmul_relu_kernel(...):
        with T.Kernel(...):
            ...
            T.copy(...)
            T.gemm(...)
    return matmul_relu_kernel

kernel = matmul(...)
kernel(a, b, c)
```

对应示例：

- `@tilelang.jit`：[examples/quickstart.py](examples/quickstart.py#L8-L9)
- 内部 `@T.prim_func`：[examples/quickstart.py](examples/quickstart.py#L10-L47)
- 编译返回 kernel object：[examples/quickstart.py](examples/quickstart.py#L57-L58)
- 调用 kernel：[examples/quickstart.py](examples/quickstart.py#L67-L68)

## Python DSL -> TIR PrimFunc

### 1. `@tilelang.jit`

装饰器入口是 [tilelang/jit/__init__.py](tilelang/jit/__init__.py#L493-L557) 的 `jit(...)`。

关键流程：

- `jit(...)` 调用 `prim_func(func, eager_jit=True)`，把 Python 函数包装成 `JITFunc`。
- 返回 `JITImpl(...)`，它负责后续 mode inference、cache key、compile 和 call。

### 2. lazy/eager mode inference

核心在 [tilelang/language/eager/builder.py](tilelang/language/eager/builder.py#L1077-L1195) 的 `JITFunc`。

关键函数：

- `JITFunc._is_lazy_style(...)`：判断是否是 lazy style。如果函数内部含 `@T.prim_func` 或直接返回 `PrimFunc`，走 lazy。
- `JITFunc._build_tir_template(...)`：lazy 时直接调用原函数得到 `PrimFunc`；eager 时用 `Builder` trace DSL body。
- `JITFunc.get_tir(...)`：根据 constexpr/tensor 参数拿到最终 `PrimFunc`。

`T.prim_func` 的入口是 [tilelang/language/eager/builder.py](tilelang/language/eager/builder.py#L1236-L1275) 的 `prim_func(...)`。

### 3. AST transform 和 Builder

Python DSL 被改写成 builder 调用，入口在 [tilelang/language/eager/ast.py](tilelang/language/eager/ast.py#L647-L700) 的 `mutate(...)`。

主要逻辑：

- `DSLMutator` 改写 Python AST。
- `if` 被改写为 `__tb.ctx_if/ctx_then/ctx_else`。
- `for` 被改写为 `__tb.ctx_for(...)`。
- 赋值和带类型标注的赋值被改写为 `__tb.bind(...)` 或 `__tb.assign_slice(...)`。
- `visit_FunctionDef(...)` 会生成一个接受 `__tb` builder 的 closure。

关键代码：

- `DSLMutator` 类：[tilelang/language/eager/ast.py](tilelang/language/eager/ast.py#L256-L502)
- 判断内部 `@T.prim_func`：[tilelang/language/eager/ast.py](tilelang/language/eager/ast.py#L632-L644)
- 构造 `IRGenerator`：[tilelang/language/eager/ast.py](tilelang/language/eager/ast.py#L647-L700)

### 4. DSL op 如何进入 TIR

`T.Kernel(...)` 入口是 [tilelang/language/kernel.py](tilelang/language/kernel.py#L266-L344)。

关键点：

- 检查当前是否存在 `Builder.current()`。
- 规范化 grid/block/thread/cluster attrs。
- 调用 C++ FFI `_ffi_api.KernelLaunch(...)`。

C++ 侧 `KernelLaunch` 在 [src/ir.cc](src/ir.cc#L261-L337)：

- 创建 `blockIdx.x/y/z` 和 `threadIdx.x/y/z` 的 launch frame。
- 包一层 `tilelang_root` block。
- 通过 `TVM_FFI_STATIC_INIT_BLOCK` 注册 `tl.KernelLaunch`。

Tile-level ops 通常先变成 TIR intrinsic call：

- `T.copy(...)`：[tilelang/language/copy_op.py](tilelang/language/copy_op.py#L51-L120)，生成 `tir.call_intrin("handle", Op.get("tl.tileop.copy"), ...)`。
- `T.gemm(...)`：[tilelang/language/gemm_op.py](tilelang/language/gemm_op.py#L149-L198)，生成 `tl.tileop.gemm` call。

这些 call 会在后面的 `LowerTileOp` 阶段被真正展开。

## 编译触发和 cache

### 1. `JITImpl.__call__`

入口是 [tilelang/jit/__init__.py](tilelang/jit/__init__.py#L434-L469)。

关键流程：

1. 如果 mode 是 `auto`，先通过参数判断 lazy/eager。
2. 调用 `self.func.parse_args(...)`，得到 cache key 和 tensor args。
3. 如果当前 `JITImpl` 内部 cache miss，调用 `self.compile(...)`。
4. eager mode 立即执行 kernel；lazy mode 返回 `JITKernel`。

`JITImpl.compile(...)` 在 [tilelang/jit/__init__.py](tilelang/jit/__init__.py#L389-L420)：

- 调用 `self.get_tir(...)` 得到 `PrimFunc`。
- 调用模块级 `compile(...)`。
- 如果设置 `debug_root_path`，会把 generated kernel source 和 TIR script 写出。

### 2. `tilelang.compile`

入口是 [tilelang/jit/__init__.py](tilelang/jit/__init__.py#L43-L120)。

关键动作：

- 断言输入必须是 `tvm.tir.PrimFunc`。
- 合并 `PrimFunc.attrs` 上的 `tilelang_out_idx`、`tilelang_pass_configs`、`tilelang_compile_flags`。
- 调用 `cached(...)` 进入 kernel cache。

### 3. backend 和 target 解析

cache 入口在 [tilelang/cache/__init__.py](tilelang/cache/__init__.py#L30-L86)。

关键动作：

- 从环境变量补默认值：`TILELANG_TARGET`、`TILELANG_EXECUTION_BACKEND`、`TILELANG_VERBOSE`。
- 调用 `determine_target(...)` 规范化 target。
- 调用 `resolve_execution_backend(...)` 得到具体 backend。
- 分发到对应的 `KernelCache`。

target 自动检测在 [tilelang/utils/target.py](tilelang/utils/target.py#L186-L272)。`target="auto"` 时优先检查当前 TVM Target、ROCm/CUDA/Metal 可用性。

execution backend 规则在 [tilelang/jit/execution_backend.py](tilelang/jit/execution_backend.py#L26-L106)：

- CUDA：`tvm_ffi`、`nvrtc`、`cython`
- HIP：`tvm_ffi`、`cython`
- Metal：`tvm_ffi`、`torch`
- CuTeDSL target：`cutedsl`
- `auto` 时 CUDA/HIP/Metal 默认选 `tvm_ffi`，其他 fallback 默认选 `cython`

### 4. KernelCache

入口是 [tilelang/cache/kernel_cache.py](tilelang/cache/kernel_cache.py#L257-L370) 的 `KernelCache.cached(...)`。

关键流程：

1. 如果 cache disabled，直接构造 `JITKernel`。
2. 生成 SHA256 cache key。
3. 查 memory cache。
4. 查 disk cache。
5. miss 时构造 `JITKernel(...)`。
6. 保存到 disk cache，并写入 memory cache。

## JITKernel -> tilelang.lower

`JITKernel` 在 [tilelang/jit/kernel.py](tilelang/jit/kernel.py#L58-L140) 初始化。

关键入口是 [tilelang/jit/kernel.py](tilelang/jit/kernel.py#L203-L330) 的 `_compile_and_create_adapter(...)`：

- 规范化 pass configs 和 compile flags。
- 根据 backend 设置：
  - `enable_host_codegen = execution_backend == "tvm_ffi"`
  - `enable_device_compile = execution_backend == "tvm_ffi"`
- 在 `tvm.transform.PassContext(opt_level=3, config=pass_configs)` 和 target context 下调用 `tilelang.lower(...)`。
- 根据 backend 创建 runtime adapter。

`JITKernel.__call__` 在 [tilelang/jit/kernel.py](tilelang/jit/kernel.py#L185-L201)，它只是转发到 adapter 暴露出来的 `torch_function`。

## Lower pipeline

主入口是 [tilelang/engine/lower.py](tilelang/engine/lower.py#L316-L346) 的 `lower(...)`。

`lower(...)` 做三件事：

1. 调用 `lower_to_host_device_ir(...)` 得到 `host_mod`、`device_mod`、`params`、`target`、`target_host`。
2. 根据 `enable_device_compile` 选择 `device_codegen(...)` 或 `device_codegen_without_compile(...)`。
3. 如果 `enable_host_codegen=True`，继续跑 `host_codegen(...)`，把 device module import 到 host module，并返回带 `rt_mod` 的 `CompiledArtifact`。

`CompiledArtifact` 定义在 [tilelang/engine/param.py](tilelang/engine/param.py#L153-L164)，里面保存：

- `host_mod`
- `device_mod`
- `params`
- `kernel_source`
- `rt_mod`

参数提取在 [tilelang/engine/lower.py](tilelang/engine/lower.py#L181-L188) 的 `extrac_params(...)`，会把 TIR buffer/scalar 参数转成 `KernelParam`。

### 1. semantic check

入口是 [tilelang/engine/lower.py](tilelang/engine/lower.py#L275-L313) 的 `lower_to_host_device_ir(...)`，其中先调用 [tilelang/engine/phase.py](tilelang/engine/phase.py#L125-L142) 的 `PreLowerSemanticCheck(mod)`。

semantic check 当前包含：

- 可选 AST printer。
- `tilelang.analysis.NestedLoopChecker()`。
- `tilelang.analysis.FragmentLoopChecker()`。

这是 validation-only pipeline，不修改 IR。

### 2. LowerAndLegalize

入口是 [tilelang/engine/phase.py](tilelang/engine/phase.py#L144-L224) 的 `LowerAndLegalize(mod, target)`。

关键顺序：

```text
BindTarget
  -> optional LetInline
  -> AddWrapperForSingleBufStore
  -> LegalizeNegativeIndex
  -> optional VerifyParallelLoop
  -> InjectAssumes
  -> Simplify
  -> LayoutReducer
  -> optional ProducerConsumerWarpSpecialized
  -> LowerBlackwell2SM
  -> PipelinePlanning
  -> InjectSoftwarePipeline
  -> Simplify
  -> LayoutInference
  -> optional LayoutVisual
  -> LowerTileOp
  -> LowerL2Persistent
  -> DecoupleTypeCast
  -> LegalizeVectorizedLoop
  -> LegalizeSafeMemoryAccess
  -> LowerAccessPtr
  -> Simplify
  -> HoistNonRestrictParams
```

这里的 `LayoutInference` 和 `LowerTileOp` 是 Python wrapper 调 C++ FFI pass。

### 3. layout inference

Python wrapper 在 [tilelang/transform/__init__.py](tilelang/transform/__init__.py#L57-L65)：

```python
def LayoutInference():
    return _ffi_api.LayoutInference()
```

C++ pass 在 [src/transform/layout_inference.cc](src/transform/layout_inference.cc#L1183-L1301)。

关键动作：

- `LayoutInferencer::Substitute(...)` 先 fuse parallel loops。
- `BufferUseDefCollector` 收集 buffer use/def。
- `VisitStmt_(BlockNode)` 给 block 加 `attr::kLayoutMap` annotation。
- `VisitStmt_(ForNode)` 给 parallel loop 加 `attr::kParallelLoopLayout` 和可选 `attr::kParallelLoopPredicate`。
- pass 名称是 `tl.LayoutInference`，注册为 `tl.transform.LayoutInference`。

### 4. LowerTileOp

Python wrapper 在 [tilelang/transform/__init__.py](tilelang/transform/__init__.py#L68-L76)：

```python
def LowerTileOp():
    return _ffi_api.LowerTileOp()
```

C++ pass 在 [src/transform/lower_tile_op.cc](src/transform/lower_tile_op.cc#L201-L230) 和 [src/transform/lower_tile_op.cc](src/transform/lower_tile_op.cc#L1030-L1081)。

关键动作：

- `LowerTileOpPass::Substitute(...)` 读取 target，建立 buffer/layout remap，遍历函数体。
- 遇到 `Evaluate(Call)` 时调用 `ParseOperator(...)`。
- 如果 call 是 TileLang tile op，就构造 `LowerArgs`，调用 `tile_op->Lower(...)`。
- 低层 lowering 后再继续走 base mutator。

Tile op 解析和注册：

- `LowerArgs` 和 `TileOperatorNode` 定义：[src/op/operator.h](src/op/operator.h#L84-L134)
- `TIR_REGISTER_TL_TILE_OP` 注册宏：[src/op/operator.h](src/op/operator.h#L168-L179)
- `ParseOperator(Call)` 通过 `"TLOpBuilder"` attr 找 builder：[src/op/operator.cc](src/op/operator.cc#L32-L40)

GEMM 是一个特殊点：C++ TileOp 会回调 Python 侧选择实现。

- Python 注册 `tl.gemm.infer_layout` 和 `tl.gemm.lower`：[tilelang/tileop/gemm/__init__.py](tilelang/tileop/gemm/__init__.py#L12-L29)
- Python `Gemm.infer_layout/lower` 选择具体实现类：[tilelang/tileop/gemm/__init__.py](tilelang/tileop/gemm/__init__.py#L125-L143)
- C++ `GemmNode::Lower(...)` 调 Python global func：[src/op/gemm.cc](src/op/gemm.cc#L180-L223)
- C++ `GemmNode::InferLayout(...)` 调 Python global func：[src/op/gemm.cc](src/op/gemm.cc#L225-L260)
- `tl.tileop.gemm` 注册：[src/op/gemm.cc](src/op/gemm.cc#L263-L266)

### 5. OptimizeForTarget 和 host/device split

入口是 [tilelang/engine/phase.py](tilelang/engine/phase.py#L227-L310) 的 `OptimizeForTarget(mod, target)`。

关键顺序可以压缩成：

```text
LowerSharedTmem
  -> IfStmtBinding
  -> PlanAndUpdateBufferAllocationLocation
  -> LowerSharedBarrier
  -> optional FuseMBarrierArriveExpectTx
  -> HoistGlobalBufferAllocations
  -> LowerOpaqueBlock
  -> Simplify
  -> NarrowDataType
  -> FlattenBuffer
  -> ConfigIndexBitwidth
  -> VectorizeLoop
  -> StorageRewrite
  -> LoopUnswitching
  -> UnrollLoop
  -> VerifyMemory
  -> AnnotateEntryFunc
  -> InferFragment
  -> LowerThreadAllreduce
  -> LowerLDGSTG
  -> LowerHopperIntrin
  -> optional ThreadSync("global")
  -> AnnotateDeviceRegions
  -> SplitHostDevice
  -> MarkCudaSyncCalls
  -> AnnotateReadOnlyParams
  -> MergeSharedMemoryAllocations
  -> InjectFenceProxy
  -> ThreadSync("shared")
  -> ThreadSync("shared.dyn")
  -> InjectTcgen05Fence
  -> MergeIfStmt
  -> optional AnnotateWarpGroupRegAlloc
  -> MakePackedAPI
  -> Simplify
  -> LowerDeviceKernelLaunch
  -> PersistThreadblock
```

因此，`host/device split` 在源码里不是 `OptimizeForTarget` 之后单独的一步，而是 `OptimizeForTarget` 的中段：

- `AnnotateDeviceRegions()`：[tilelang/engine/phase.py](tilelang/engine/phase.py#L278-L278)
- `SplitHostDevice()`：[tilelang/engine/phase.py](tilelang/engine/phase.py#L279-L279)

Python wrapper 在 [tilelang/transform/__init__.py](tilelang/transform/__init__.py#L306-L314)。

`OptimizeForTarget(...)` 结束后，[tilelang/engine/lower.py](tilelang/engine/lower.py#L310-L311) 通过 `tir.transform.Filter(...)` 把 module 分成 `host_mod` 和 `device_mod`。

## Backend codegen

### 1. device codegen

入口：

- 编译 device binary/source：[tilelang/engine/lower.py](tilelang/engine/lower.py#L231-L247) 的 `device_codegen(...)`
- 只生成 source module：[tilelang/engine/lower.py](tilelang/engine/lower.py#L250-L272) 的 `device_codegen_without_compile(...)`

CUDA 路径：

- `target.build.tilelang_cuda`
- `target.build.tilelang_cuda_without_compile`

注册点在 [src/backend/cuda/codegen/rt_mod_cuda.cc](src/backend/cuda/codegen/rt_mod_cuda.cc#L94-L170)。

`BuildTileLangCUDA(...)` 会：

1. 创建 `CodeGenTileLangCUDA`。
2. 校验 device global symbols。
3. 遍历 device `PrimFunc`，要求 calling convention 是 `kDeviceKernelLaunch`。
4. 调 `cg.AddFunction(...)` 生成 CUDA C source。
5. 调 `tilelang_callback_cuda_postproc` 做后处理。
6. 调 `tilelang_callback_cuda_compile` 编译为 PTX/CUBIN。
7. 返回 `runtime::CUDAModuleCreate(...)`。

CUDA kernel source 生成入口是 [src/backend/cuda/codegen/codegen_cuda.cc](src/backend/cuda/codegen/codegen_cuda.cc#L5034-L5136) 的 `CodeGenTileLangCUDA::AddFunction(...)`，函数签名打印入口是 [src/backend/cuda/codegen/codegen_cuda.cc](src/backend/cuda/codegen/codegen_cuda.cc#L4952-L5020)。

### 2. host codegen

入口是 [tilelang/engine/lower.py](tilelang/engine/lower.py#L198-L228) 的 `host_codegen(...)`。

关键动作：

- `BindTarget(target_host)`
- `LowerTVMBuiltin`
- `LowerCustomDatatypes`
- `tilelang.transform.LowerIntrin`
- `LowerDeviceStorageAccessInfo`
- `CombineContextCall`
- Metal target 额外跑 `MarkHostMetalContext`
- `target_host=llvm` 时调用 `target.build.llvm`
- `target_host=c` 时调用 `target.build.tilelang_c_host`

`tvm_ffi` backend 会启用 host codegen，并把 device module import 到 host module：

```python
host_mod = host_codegen(...)
host_mod.import_module(codegen_mod)
return CompiledArtifact(..., rt_mod=host_mod)
```

对应代码在 [tilelang/engine/lower.py](tilelang/engine/lower.py#L338-L345)。

## Runtime adapter

adapter 创建发生在 [tilelang/jit/kernel.py](tilelang/jit/kernel.py#L252-L330)。

### 1. TVM FFI backend

类入口：[tilelang/jit/adapter/tvm_ffi.py](tilelang/jit/adapter/tvm_ffi.py#L71-L116)。

关键运行逻辑在 [tilelang/jit/adapter/tvm_ffi.py](tilelang/jit/adapter/tvm_ffi.py#L150-L260)：

- 解析 `KernelParam`，准备 PyTorch dtype/shape。
- 对 `out_idx` 对应的输出自动 `torch.empty(...)`。
- 创建或复用 `runtime.Executable(self.rt_mod)`。
- 执行 `executable(*tensor_list)`。
- 返回输出 tensor 或输出列表。

源码获取入口：

- `get_host_source()`：[tilelang/jit/adapter/tvm_ffi.py](tilelang/jit/adapter/tvm_ffi.py#L299-L303)
- `get_device_source()`：[tilelang/jit/adapter/tvm_ffi.py](tilelang/jit/adapter/tvm_ffi.py#L305-L309)
- `get_kernel_source()`：[tilelang/jit/adapter/tvm_ffi.py](tilelang/jit/adapter/tvm_ffi.py#L311-L316)

### 2. Cython backend

类入口：[tilelang/jit/adapter/cython/adapter.py](tilelang/jit/adapter/cython/adapter.py#L75-L150)。

关键动作：

- `TLWrapper` 根据 target 生成 host wrapper。
- `LibraryGenerator` 编译 shared library。
- `ctypes` load library。
- 创建 `CythonKernelWrapper`。
- `_convert_torch_func(...)` 返回 `cython_wrapper.forward(...)`。

运行入口在 [tilelang/jit/adapter/cython/adapter.py](tilelang/jit/adapter/cython/adapter.py#L347-L360)。

### 3. 其他 backend

`JITKernel._compile_and_create_adapter(...)` 还会按 backend 分发到：

- `NVRTCKernelAdapter`
- `MetalKernelAdapter`
- `CuTeDSLKernelAdapter`

分发代码在 [tilelang/jit/kernel.py](tilelang/jit/kernel.py#L283-L325)。

## 一次完整调用的调用栈轮廓

lazy JIT 模式下，`python examples/quickstart.py` 中 `matmul(...)` 和 `kernel(...)` 大致对应：

```text
tilelang.__init__
  -> load libtilelang and export jit/compile/lower

@tilelang.jit
  -> tilelang.jit.jit
  -> tilelang.language.eager.prim_func(eager_jit=True)
  -> JITFunc
  -> JITImpl

matmul(M, N, K, ...)
  -> JITImpl.__call__
  -> JITFunc._is_lazy_style / parse_args / get_tir
  -> tilelang.jit.compile
  -> tilelang.cache.cached
  -> KernelCache.cached
  -> JITKernel.__init__
  -> JITKernel._compile_and_create_adapter
  -> tilelang.lower
  -> lower_to_host_device_ir
      -> PreLowerSemanticCheck
      -> LowerAndLegalize
          -> LayoutInference
          -> LowerTileOp
      -> OptimizeForTarget
          -> AnnotateDeviceRegions
          -> SplitHostDevice
      -> Filter host/device
  -> device_codegen or device_codegen_without_compile
  -> optional host_codegen
  -> CompiledArtifact
  -> TVMFFIKernelAdapter / CythonKernelAdapter / NVRTCKernelAdapter / ...
  -> return JITKernel

kernel(a, b, c)
  -> JITKernel.__call__
  -> adapter.func
  -> TVM runtime Executable or generated wrapper/library
  -> launch device kernel
```

eager JIT 模式的区别是：`JITImpl.__call__` 在 cache/compile 之后会直接执行 `kernel(*kernel_args.values())`，而不是把 `JITKernel` 返回给用户。

## 调试入口

常用观察点：

- `kernel.get_kernel_source()`：查看最终 device/host source。
- `@tilelang.jit(debug_root_path="...")`：在 `JITImpl.compile(...)` 中写出 kernel source 和 TIR script。
- `pass_configs={PassConfigKey.TL_ENABLE_DUMP_IR: True, ...}`：通过 `DumpIR` dump pass 中间 IR，入口在 [tilelang/jit/kernel.py](tilelang/jit/kernel.py#L235-L248)。
- 直接调用 `tilelang.lower(prim_func, target=..., enable_host_codegen=..., enable_device_compile=...)`：绕过 adapter，只看 lowering/codegen artifact。

## 推荐读源码顺序

1. [examples/quickstart.py](examples/quickstart.py#L8-L68)
2. [tilelang/jit/__init__.py](tilelang/jit/__init__.py#L493-L557)
3. [tilelang/language/eager/builder.py](tilelang/language/eager/builder.py#L1077-L1275)
4. [tilelang/cache/__init__.py](tilelang/cache/__init__.py#L30-L86)
5. [tilelang/jit/kernel.py](tilelang/jit/kernel.py#L203-L330)
6. [tilelang/engine/lower.py](tilelang/engine/lower.py#L275-L346)
7. [tilelang/engine/phase.py](tilelang/engine/phase.py#L125-L310)
8. [src/transform/layout_inference.cc](src/transform/layout_inference.cc#L1183-L1301)
9. [src/transform/lower_tile_op.cc](src/transform/lower_tile_op.cc#L201-L230)
10. [src/op/operator.h](src/op/operator.h#L84-L179)
11. [src/backend/cuda/codegen/rt_mod_cuda.cc](src/backend/cuda/codegen/rt_mod_cuda.cc#L94-L170)
12. [tilelang/jit/adapter/tvm_ffi.py](tilelang/jit/adapter/tvm_ffi.py#L150-L260)
