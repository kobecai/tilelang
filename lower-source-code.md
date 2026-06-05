# TileLang `lower` 源码阅读指南

这份笔记接在 `language-source-code.md` 后面，专门阅读 TileLang 的 `tilelang.lower` 和 backend lowering pipeline。

如果说 `tilelang/language` 负责把用户 Python DSL 捕获成高层 TIRX `PrimFunc`，那么 `tilelang.lower` 负责把这个 `PrimFunc` 继续降到后端可以 codegen、可以被 Python adapter 调用的形态。

核心问题是：

```text
用户写出来的高层 TileLang IR
  T.Kernel / thread_extent
  local.fragment / shared.dyn
  tl.tileop.copy / tl.tileop.gemm / tl.tileop.reduce
  T.Pipelined / num_stages

如何变成：
  host packed API function
  device kernel PrimFunc
  CUDA/HIP/Metal/C source
  runtime module 或 adapter 可执行对象
```

---

## 1. 一句话结论

`tilelang.lower` 不是一个巨大 monolithic compiler pass，而是一个很薄的入口壳：

```text
tilelang/engine/lower.py
  -> PreLowerSemanticCheck
  -> resolve_pipeline(target)
  -> backend pipeline.lower(mod, target)
  -> Filter host/device functions
  -> device codegen / host codegen
  -> CompiledArtifact
```

真正的复杂度分布在 5 个层次：

| 层次 | 代表文件 | 核心职责 |
| --- | --- | --- |
| Python lower 入口 | `tilelang/engine/lower.py` | target 规范化、语义检查、pipeline 分发、host/device 分离、codegen |
| backend pipeline | `tilelang/cuda/pipeline.py` 等 | 决定 pass 顺序，不同 target 有不同 lowering 路线 |
| Python transform wrapper | `tilelang/transform/__init__.py` | 把 Python API 包装到 C++ FFI pass |
| C++ transform pass | `src/transform/*.cc` | LayoutInference、LowerTileOp、SplitHostDevice、MakePackedAPI 等真正改 IR |
| tileop / codegen / adapter | `src/op/`、`src/cuda/`、`tilelang/jit/adapter/` | TileOp 兑现成硬件 intrinsic，生成源码/二进制，并包成 Python callable |

所以读 lower 时要避免一个误区：不要只盯着 `lower.py`。`lower.py` 是路由器，CUDA/ROCm/CPU/Metal pipeline 才是主体。

---

## 2. lower 的输入到底是什么

`tilelang.lower(...)` 的输入类型在 `tilelang/engine/lower.py` 中定义：

```python
def lower(
    func_or_mod: tirx.PrimFunc | tvm.IRModule,
    target: str | Target = "auto",
    target_host: str | Target | None = None,
    runtime_only=False,
    enable_host_codegen=False,
    enable_device_compile=False,
) -> CompiledArtifact:
```

### 2.1 输入一：`tirx.PrimFunc`

最常见输入是 frontend/JIT 生成的一个 `tvm.tirx.PrimFunc`。

这个 `PrimFunc` 通常已经包含：

| 内容 | 来源 | 后续消费者 |
| --- | --- | --- |
| `params` | `T.Tensor` 参数、scalar 参数 | `extrac_params`、MakePackedAPI、adapter |
| `buffer_map` | frontend 把 handle var 映射成 Buffer | 参数提取、SplitHostDevice、MakePackedAPI、adapter 动态 shape 解析 |
| `attrs["global_symbol"]` | `@T.prim_func` / Builder | IRModule key、host/device symbol、kernel source 名字 |
| `with T.Kernel(...)` 生成的 thread/block launch 结构 | `language/kernel.py` | AnnotateDeviceRegions、SplitHostDevice、LowerDeviceKernelLaunch |
| `local.fragment` / `shared.dyn` allocation | `T.alloc_fragment`、`T.alloc_shared` | LayoutInference、LowerTileOp、StorageRewrite、dynamic shared memory launch arg |
| `tl.tileop.copy/gemm/reduce` | `T.copy`、`T.gemm`、`T.reduce` | LayoutInference、LowerTileOp |
| loop annotations | `T.Pipelined`、`T.Parallel` | PipelinePlanning、InjectSoftwarePipeline、LayoutInference、LowerTileOp |

如果输入是 `PrimFunc`，`lower_to_host_device_ir` 会先做：

```python
params = extrac_params(func) if not runtime_only else None
mod = tvm.IRModule({func.attrs["global_symbol"]: func})
```

也就是说，lower 内部统一处理 `IRModule`，单个 `PrimFunc` 只是会被包装成一个单函数 module。

### 2.1.1 问答：lower 输入的 `PrimFunc` 里有没有 TileLang 自定义/封装的 TVM 元素

问题可以拆成两层：

```text
1. lower 入口收到的对象类型是不是 PrimFunc？
2. 这个 PrimFunc 的 body/attrs/annotation 里是否还带 TileLang 自定义语义？
```

答案是：入口对象是 `tvm.tirx.PrimFunc`，但它不是“纯 vanilla TVM TIR”。它通常是 frontend/JIT 已经把 TileLang DSL 语义编码进去的高层 TIRX `PrimFunc`，里面会混有 TIRX 扩展节点、TileLang 注册的 `tl.*` / `tl.tileop.*` call、特殊 storage scope 和 block/loop annotations。

#### 外壳类型：仍然是 `tvm.tirx.PrimFunc`

`tilelang.lower(...)` 的签名是：

```python
def lower(
    func_or_mod: tirx.PrimFunc | tvm.IRModule,
    ...
) -> CompiledArtifact:
```

如果传入的是 `PrimFunc`，`lower_to_host_device_ir` 会先用 `global_symbol` 把它包装成 `IRModule`：

```python
if isinstance(func_or_mod, tirx.PrimFunc):
    func = func_or_mod
    params = extrac_params(func) if not runtime_only else None
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
```

另外，`tilelang/language/eager/builder.py` 中 `PrimFunc` 在运行时并不是 TileLang 自己的新 Python IR 类：

```python
if TYPE_CHECKING:
    class PrimFunc(Generic[_P, _T], tvm.tirx.PrimFunc):
        ...
else:
    PrimFunc = tvm.tirx.PrimFunc
```

所以从对象系统看，lower 入口拿到的是 `tvm.tirx.PrimFunc` 或 `tvm.IRModule`。

#### 内容形态：是带 TileLang 语义的高层 TIRX

虽然外壳是 `PrimFunc`，但这个 `PrimFunc` 通常来自 `@T.prim_func`、`@tilelang.jit`、eager builder 或 lazy-style JIT。frontend 在构造它时，已经把 Python DSL 映射成 TIRX/TIR 结构和 TileLang 扩展 call。

常见内容包括：

| TileLang 写法 | 进入 lower 前的 IR 表达 | 后续主要消费者 |
| --- | --- | --- |
| `T.Kernel(...)` | TIRX launch/thread/block frame 展开的 `thread_extent`、device block、block annotations | `AnnotateDeviceRegions`、`SplitHostDevice`、`LowerDeviceKernelLaunch` |
| `T.gemm(...)` | `Evaluate(Call(op=tl.tileop.gemm, ...))` | `LayoutInference`、`LowerTileOp` |
| `T.copy(...)` | `Evaluate(Call(op=tl.tileop.copy, ...))`，某些 pass 会改写成 `tl.tileop.tma_copy` | `PipelinePlanning`、warp specialization、`LowerTileOp` |
| `T.fill(...)` | `Call(op=tl.tileop.fill, ...)` | `LowerTileOp` |
| `T.reduce(...)` / reducer finalize | `tl.tileop.reduce` / `tl.tileop.finalize_reducer` | `LayoutReducer`、`LowerTileOp` |
| `T.access_ptr(...)` | frontend-only `Call(op=tl.access_ptr, ...)` | `LowerAccessPtr`、部分 vectorize/legalize pass |
| fast math / warp reduce / device assert | `tl.__exp`、`tl.__log`、`tl.warp_reduce_sum`、`tl.device_assert` 等 `tl.*` intrinsic | `LowerIntrin`、target codegen |
| `T.alloc_shared(...)` | TIRX buffer allocation，scope 常见为 `shared.dyn` / `shared` | layout、storage rewrite、shared memory merge、launch 参数 |
| `T.alloc_fragment(...)` | TIRX buffer allocation，scope 为 `local.fragment` | `LayoutInference`、`LowerTileOp`，之后常被 remap 成普通 `local` |
| `T.alloc_barrier(...)` / TMA barrier | `shared.barrier` / `shared.cluster_barrier` scope 和 `barrier_init` annotation | `LowerSharedBarrier`、TMA lowering |
| `T.annotate_layout(...)` | SBlock annotation `layout_map` | `LayoutInference`、`LowerTileOp` |
| `T.Pipelined(...)` | loop annotation `num_stages`、`tl_pipeline_*` | `PipelinePlanning`、`InjectSoftwarePipeline` |
| `T.Parallel(...)` | `ForKind::kParallel`，之后带 `parallel_loop_layout` 等 annotation | `LayoutInference`、`LowerTileOp` |

所以可以把 lower 输入理解成：

```text
tvm.tirx.PrimFunc
  params / buffer_map / attrs
  body: TIRX Stmt tree
    For / SBlock / BufferLoad / BufferStore / Call / Evaluate / AttrStmt ...
    + TileLang op calls: tl.tileop.copy/gemm/reduce/fill/...
    + TileLang intrinsics: tl.access_ptr / tl.__exp / tl.warp_reduce_sum / ...
    + TileLang storage scopes: local.fragment / shared.dyn / shared.barrier / shared.tmem
    + TileLang annotations: layout_map / barrier_init / reducer_info / cluster_dims / ...
```

#### TileLang 的 Python DSL 对象不会原样留在 PrimFunc 里

这里容易混淆：`T.Kernel`、`T.gemm`、`T.alloc_shared` 这些 Python API 是“前端构造工具”，lower 入口看到的不是这些 Python 对象本身。

例如：

`T.Kernel(...)` 在 Python 侧返回 `KernelLaunchFrame`，底层通过 FFI 调到 `tl.KernelLaunch`。C++ 里 `KernelLaunch` 创建一组 TIR builder frame：

```text
blockIdx.x/y/z LaunchThread frame
threadIdx.x/y/z LaunchThread frame
device main SBlock frame
```

退出 `with T.Kernel(...)` 后，最终留在 `PrimFunc.body` 中的是 TIRX 的 thread/block/SBlock/annotation 结构，而不是 Python 的 `KernelLaunchFrame` 对象。

同理：

```python
C = T.alloc_fragment((16, 16), "float32")
S = T.alloc_shared((128, 128), "float16")
```

进入 IR 后是 TIRX buffer allocation，关键 TileLang 信息体现在 buffer scope：

```text
C.scope() == "local.fragment"
S.scope() == "shared.dyn"
```

#### `TileOperatorNode` 是 lowering 时临时解析出来的，不是原始 PrimFunc 中的节点

TileLang C++ 里有一套 `TileOperatorNode` / `TileOperator` 抽象：

```cpp
class TileOperatorNode : public Object {
public:
  virtual Stmt Lower(const LowerArgs& T, arith::Analyzer* analyzer) const = 0;
  virtual LayoutMap InferLayout(const LayoutInferArgs& T, InferLevel level) const = 0;
};
```

这套对象负责承载 `CopyNode`、`GemmNode`、`ReduceNode`、`FillNode` 等 tile op 的 C++ lowering 逻辑。但它通常不是直接存放在输入 `PrimFunc.body` 里的 IR 节点。

输入 `PrimFunc.body` 里更常见的是：

```text
Evaluate(Call(op=tl.tileop.xxx, args, annotations))
```

`LowerTileOp` 运行时会做：

```cpp
Stmt VisitStmt_(const EvaluateNode* op) final {
  auto tile_op = ParseOperator(GetRef<Stmt>(op));
  if (!tile_op.defined()) {
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  auto lowered = tile_op->Lower(LowerArgs{...}, analyzer_);
  return IRMutatorWithAnalyzer::VisitStmt(lowered);
}
```

而 `ParseOperator` 的逻辑是：

```text
Call(op=tl.tileop.xxx, args, annotations)
  -> 查 Op attr map "TLOpBuilder"
  -> 构造 Copy/Gemm/Reduce/Fill/... TileOperator object
  -> 调用 tile_op->Lower(...)
```

也就是说，`TileOperatorNode` 是 `LowerTileOp` 消费 `tl.tileop.*` call 时临时构造出的语义对象；原始 `PrimFunc` 主要保存的是 TVM/TIRX `Call`，其中 `op` 指向 TileLang 注册的 `tl.tileop.*`。

#### Lower pipeline 如何逐步吃掉这些 TileLang 语义

以 CUDA pipeline 为例，前半段会先保留高层 tile-op 语义，让规划类 pass 能看懂它们：

```text
BindTarget
AddWrapperForSingleBufStore
LegalizeNegativeIndex
InjectAssumes
Simplify
LayoutReducer
ProducerConsumerWarpSpecialized
LowerBlackwell2SM
IfStmtBinding
PipelinePlanning
InjectSoftwarePipeline
LayoutInference
LowerTileOp
LowerL2Persistent
DecoupleTypeCast
LegalizeVectorizedLoop
LegalizeSafeMemoryAccess
LowerAccessPtr
...
```

这个顺序很重要：

1. `PipelinePlanning`、warp specialization、`LayoutInference` 需要看到 `tl.tileop.copy/gemm/...` 这种高层语义，才能判断 pipeline stage、TMA/cp.async/WGMMA 路线、fragment/shared layout。
2. `LowerTileOp` 之后，高层 tile op 会被展开成低层 TIR/intrinsic，很多 `local.fragment` buffer 也会根据 layout remap 成普通 local buffer。
3. `LowerAccessPtr` 再把 frontend-only 的 `tl.access_ptr` 降成标准 `tvm_access_ptr` 风格节点。
4. `LowerIntrin` 和 target codegen 最后处理残留的 `tl.*` intrinsic，例如 fast math、warp reduce、mbarrier/TMA/TCGEN 等目标相关 intrinsic。

因此更精确的说法是：

```text
lower 输入是 PrimFunc；
但这是 TileLang frontend 编码过的高层 TIRX PrimFunc；
其中 TileLang 自定义语义主要以 tl.* / tl.tileop.* Call、scope、attrs、annotations 的形式存在；
lower pipeline 再逐步把这些语义降成后端 codegen 能接受的低层 IR。
```

### 2.2 输入二：`tvm.IRModule`

也可以直接传入 `IRModule`。这适合多个 PrimFunc 一起 lowering，或者外部已经构造好 module 的场景。

这种情况下，`params` 不会自动从单个 PrimFunc 提取，除非调用者自己另外维护参数信息。

### 2.3 输入 target 和 target_host

`target` 可以是：

```text
"auto"
"cuda"
"hip"
"c"
"llvm"
"metal"
tvm.target.Target(...)
```

`target="auto"` 会走 `tilelang.utils.target.determine_target`，根据环境选择后端。

`target_host` 如果没有给，`canon_target_host` 会选择：

```text
llvm 可用 -> llvm
否则 -> c
```

最终 lower 会构造：

```python
target_host = tvm.target.Target(target_host)
target = tvm.target.Target(target, target_host)
```

注意：这里的 `target` 是带 host target 的 TVM Target。后续 `AnnotateDeviceRegions` 和 codegen 会频繁用 `target.WithoutHost()` / `target.GetHost()` 分离 host/device 信息。

### 2.4 输入 runtime_only

`runtime_only=True` 时，如果输入是 `PrimFunc`，不会提取 `KernelParam`：

```python
params = extrac_params(func) if not runtime_only else None
```

正常 JIT 编译一般需要 `params`，因为 adapter 要知道哪些参数是 tensor、shape、dtype、输出下标等。`runtime_only` 更像是只关注 lowered runtime module/source 的路径。

### 2.5 输入 pass config

`lower(...)` 本身没有显式 `pass_configs` 参数。pass config 是由调用者在外层打开 `tvm.transform.PassContext` 传入的。

JIT 路径在 `tilelang/jit/kernel.py` 里做：

```python
with tvm.transform.PassContext(opt_level=3, config=pass_configs, instruments=pass_instruments), self.target:
    artifact = tilelang.lower(...)
```

所以每个 pass 内部通过：

```python
tilelang.transform.get_pass_context()
```

或 C++ `PassContext::Current()` 读取配置。

常见 lower 相关配置包括：

| 配置 | 作用 |
| --- | --- |
| `tl.disable_prelower_semantic_check` | 跳过 lower 前语义检查 |
| `tl.disable_data_race_check` | 关闭 parallel loop race check |
| `tl.enable_async_copy` | 是否自动把合适的 global->shared copy 降成 `cp.async` |
| `tl.disable_wgmma` | 禁用 Hopper WGMMA 路线 |
| `tl.disable_warp_specialized` | 禁用 CUDA warp specialization |
| `tl.enable_aggressive_shared_memory_merge` | 更激进地合并 shared memory allocation |
| `tl.disable_shared_memory_reuse` | shared memory 合并但不做 lifetime reuse |
| `tl.layout_visualization_enable` | 打印/导出 layout inference 结果 |
| `tl.ast_print_enable` | 在 PreLowerSemanticCheck 中打印 AST |
| `tirx.disable_vectorize` | 关闭 vectorize loop pass |
| `tl.device_compile_flags` | 传给 nvcc/NVRTC 的额外设备编译参数 |

---

## 3. lower 的输出是什么

`tilelang.lower(...)` 返回 `CompiledArtifact`。它来自 `tilelang/engine/param.py`，在 lower 里构造大致是：

```python
return CompiledArtifact(host_mod, device_mod, params, kernel_source, rt_mod=host_mod_or_none)
```

可以把输出分成 5 件东西：

| 字段 | 含义 | 什么时候用 |
| --- | --- | --- |
| `host_mod` | lowered host-side IRModule，或 host codegen 后 runtime module | TVM FFI backend、debug、wrapper 生成 |
| `device_mod` | lowered device-side IRModule | device codegen、inspect、adapter 包装 |
| `params` | kernel 参数元信息 | adapter 创建输出 tensor、检查输入 dtype/shape、动态 shape 解析 |
| `kernel_source` | device kernel source，例如 CUDA C++ | Cython/NVRTC/cutedsl adapter、debug 输出 |
| `rt_mod` | host codegen 后导入 device module 的 runtime module | `execution_backend="tvm_ffi"` |

### 3.1 `enable_host_codegen=False` 的默认行为

默认不做 host codegen。

原因是大部分 JIT execution backend 有自己的 host wrapper：

| backend | host wrapper 来源 |
| --- | --- |
| `tvm_ffi` | 使用 `lower(..., enable_host_codegen=True, enable_device_compile=True)` 生成 runtime module |
| `cython` | Python/Cython wrapper 自己生成 host launch code |
| `nvrtc` | Python wrapper + CUDA driver/NVRTC 路线 |
| `torch` Metal | Metal adapter 自己处理 |
| `cutedsl` | CuTe DSL adapter 自己处理 |

### 3.2 `enable_device_compile=False` 的默认行为

默认不把 device source 编译成 cubin/hsaco，而是走 `device_codegen_without_compile`。

CUDA 下会调用：

```text
target.build.tilelang_cuda_without_compile
```

这个函数仍然生成 CUDA source，并把它放进 runtime module 的 source map；只是里面的 device binary 是 dummy payload。adapter 后续可以自己用 nvcc/NVRTC 编译。

### 3.3 `enable_host_codegen=True, enable_device_compile=True`

`tvm_ffi` backend 会打开这两个开关。

此时 lower 会：

```text
host_mod lowered IR
device_mod lowered IR
  -> device_codegen(device_mod, target)
  -> host_codegen(host_mod, target_host, target)
  -> host_mod.import_module(codegen_mod)
  -> rt_mod = host_mod
```

最终 adapter 用 `runtime.Executable(rt_mod)` 直接执行。

---

## 4. 从 import 到 FFI/pass/op/codegen 注册

`import tilelang` 的注册链路非常重要。很多 lower 中调用的东西并不是 Python 直接定义的，而是 native library 加载时注册出来的。

### 4.1 Python import 加载 native 库

`tilelang/__init__.py` 中：

```python
lib_path = libinfo.find_lib_path("tilelang")
_LIB, _LIB_PATH = ctypes.CDLL(lib_path), lib_path
```

`libinfo.find_lib_path("tilelang")` 根据平台找：

| 平台 | 文件名 |
| --- | --- |
| Linux | `libtilelang.so` |
| macOS | `libtilelang.dylib` |
| Windows | `tvm_compiler.dll` |

加载 native 库后，C++ 中的 `TVM_FFI_STATIC_INIT_BLOCK()` 会运行，把函数注册进 TVM/tvm_ffi global registry。

### 4.2 C++ pass 注册

比如 `src/transform/layout_inference.cc` 末尾：

```cpp
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.LayoutInference", LayoutInference);
}
```

Python 侧 `tilelang/transform/__init__.py` 的 wrapper：

```python
def LayoutInference():
    return _ffi_api.LayoutInference()
```

这层对应关系是：

```text
Python tilelang.transform.LayoutInference()
  -> tilelang/transform/_ffi_api.py 动态 FFI symbol
  -> global func: tl.transform.LayoutInference
  -> C++ LayoutInference() pass object
```

同理：

```text
tl.transform.LowerTileOp
tl.transform.SplitHostDevice
tl.transform.MakePackedAPI
tl.transform.LowerDeviceKernelLaunch
tl.transform.FlattenBuffer
tl.transform.StorageRewrite
```

都走类似路径。

### 4.3 codegen 注册

CUDA codegen 在 `src/cuda/codegen/rt_mod_cuda.cc` 中注册：

```cpp
refl::GlobalDef()
  .def("target.build.tilelang_cuda", BuildTileLangCUDA)
  .def("target.build.tilelang_cuda_without_compile", BuildTileLangCUDAWithoutCompile);
```

`lower.py` 中调用：

```python
tvm.ffi.get_global_func("target.build.tilelang_cuda")(device_mod, target)
```

或：

```python
tvm.ffi.get_global_func("target.build.tilelang_cuda_without_compile")(device_mod, target)
```

### 4.4 Python tileop callback 注册

GEMM 是一个很典型的跨 Python/C++ lower：

`tilelang/tileop/gemm/__init__.py` 注册：

```python
@tvm_ffi.register_global_func("tl.gemm.infer_layout")
def gemm_infer_layout(...):
    ...

@tvm_ffi.register_global_func("tl.gemm.lower")
def gemm_lower(...):
    ...
```

C++ `src/op/gemm.cc` 中 `GemmNode::InferLayout` / `GemmNode::Lower` 会：

```cpp
Function::GetGlobal("tl.gemm.infer_layout")
Function::GetGlobal("tl.gemm.lower")
```

也就是说，GEMM 的 instruction 选择和 lower 是 C++/Python 协作完成的。

---

## 5. JIT 到 lower 的调用链

用户通常不会直接调用 `tilelang.lower`，而是通过 `@tilelang.jit`。

整体链路是：

```text
@tilelang.jit
  -> tilelang/jit/__init__.py::jit
  -> prim_func(func, eager_jit=True)
  -> JITImpl(...)

第一次调用 JITImpl.__call__
  -> infer lazy/eager mode
  -> parse_args 得到 cache key 和 runtime kernel_args
  -> cache lookup
  -> miss 时 JITImpl.compile
      -> get_tir 生成 PrimFunc
      -> tilelang.jit.compile
      -> cached(...)
      -> JITKernel(...)
      -> JITKernel._compile_and_create_adapter
      -> with PassContext(...), target:
             tilelang.lower(...)
      -> adapter 包成 Python callable

后续调用
  -> hit JITImpl._kernel_cache
  -> eager: kernel(*kernel_args.values())
  -> lazy: return kernel
```

### 5.1 JITImpl 的职责

`tilelang/jit/__init__.py::JITImpl.__call__` 做：

```text
1. 判断 lazy/eager
2. parse_args 生成 cache key
3. 查 Python 进程内 cache
4. 查 frontend persistent cache，lazy 且无 runtime args 时可用
5. miss 则 compile
6. eager mode 立即执行 kernel
7. lazy mode 返回 JITKernel
```

### 5.2 JITKernel 的职责

`tilelang/jit/kernel.py::JITKernel._compile_and_create_adapter` 做：

```text
1. normalize pass configs
2. determine target
3. 根据 execution_backend 决定 codegen 开关
4. 打开 PassContext
5. 调 tilelang.lower
6. 根据 backend 创建 adapter
```

关键开关：

```python
enable_host_codegen = execution_backend == "tvm_ffi"
enable_device_compile = execution_backend == "tvm_ffi"
```

所以 `tvm_ffi` backend 是完整走 TVM runtime module 的路线；`cython`/`nvrtc` 等 backend 更偏向“lower 拿 source，自己包装/编译”。

---

## 6. lower.py：薄入口如何工作

`tilelang/engine/lower.py` 可以按 6 个函数读。

### 6.1 `extrac_params(func)`

这个函数从 `PrimFunc.params` 和 `buffer_map` 提取 kernel 参数元信息。

```python
for var in func.params:
    if var in func.buffer_map:
        tensor_types.append(KernelParam.from_buffer(func.buffer_map[var]))
    else:
        tensor_types.append(KernelParam.from_var(var))
```

它决定 adapter 后续看到的参数列表：

```text
Buffer 参数 -> dtype、shape、stride、scope 等 tensor metadata
Scalar 参数 -> scalar dtype/name
```

这是为什么 lower input 的 `buffer_map` 很重要：没有它，adapter 不知道哪些 handle 参数是 tensor，也无法自动创建 output tensor。

### 6.2 `canon_target_host(target, target_host)`

如果用户没有传 host target：

```text
LLVM runtime 可用 -> llvm
否则 -> c
```

### 6.3 host/device predicate

lower 后的 module 同时包含 host function 和 device function。`lower_to_host_device_ir` 需要用 predicate 分开：

```python
_is_host_call = get_host_call(is_device_c=is_cpu_device_backend(target))
_is_device_call = get_device_call(is_device_c=is_cpu_device_backend(target))

host_mod = tirx.transform.Filter(_is_host_call)(mod)
device_mod = tirx.transform.Filter(_is_device_call)(mod)
```

普通 GPU backend 下，device function 的特征是：

```text
attrs["calling_conv"] == CallingConv.DEVICE_KERNEL_LAUNCH
```

CPU `c` backend 有一点特殊：某些 C target function 也被当成 device side codegen，因此 `is_device_call_c_device` 额外检查 `attrs["target"].kind.name == "c"`。

### 6.4 `lower_to_host_device_ir(...)`

这是最核心的 IR lowering 入口：

```python
PreLowerSemanticCheck(mod)

pipeline = resolve_pipeline(target)
mod = pipeline.lower(mod, target)

host_mod = Filter(host)(mod)
device_mod = Filter(device)(mod)
```

它返回：

```python
(host_mod, device_mod, params, target, target_host)
```

注意：backend pipeline 内部已经跑了 `SplitHostDevice` 和 `LowerDeviceKernelLaunch`。`lower.py` 这里的 `Filter` 不是“执行拆分”，而是从已经混合的 lowered module 中筛选出 host/device 两份。

### 6.5 `device_codegen(...)` 和 `device_codegen_without_compile(...)`

两者都会先做一组 device 端 codegen 前清理：

```python
device_mod = tilelang.transform.LowerIntrin()(device_mod)
device_mod = tirx.transform.Simplify()(device_mod)
device_mod = tilelang.transform.HoistBroadcastValues()(device_mod)
```

然后按 target 分发：

| target | compile 路线 | without_compile 路线 |
| --- | --- | --- |
| CUDA | `target.build.tilelang_cuda` / `tilelang_cutedsl` | `target.build.tilelang_cuda_without_compile` / `tilelang_cutedsl_without_compile` |
| HIP | `target.build.tilelang_hip` | `target.build.tilelang_hip_without_compile` |
| Metal | `target.build.tilelang_metal` | `target.build.tilelang_metal` |
| C | 不走 compile 路线 | `target.build.tilelang_c` |
| LLVM | 不走 compile 路线 | `target.build.llvm` |
| WebGPU | 不走 compile 路线 | `target.build.webgpu` |

### 6.6 `host_codegen(...)`

host 侧 codegen 会先把 host IR 降到目标 runtime 可编译形态：

```python
BindTarget(target_host)
FP8StorageLegalize
BF16StorageLegalize
LowerTVMBuiltin
LowerCustomDatatypes
LowerIntrin
CombineContextCall
```

然后：

| target_host | codegen |
| --- | --- |
| `llvm` | `target.build.llvm` |
| `c` | `target.build.tilelang_c_host` |

Metal 会额外跑 `MarkHostMetalContext`，让 host code 带上 Metal/MPS 同步上下文逻辑。

---

## 7. Pipeline registry：target 如何选 pass 列表

`tilelang/backend/pass_pipeline/pipeline.py` 很小，但它是 multi-backend lowering 的路由核心。

```python
class PassPipeline:
    def __init__(self, name: str, lower: LowerFunc):
        self.name = name
        self._lower = lower

    def lower(self, mod: IRModule, target: Target) -> IRModule:
        return self._lower(mod, target)
```

backend 注册：

```python
register_pipeline(PassPipeline("cuda", CUDAPassPipelineBody))
register_pipeline(PassPipeline("hip", ROCMPassPipelineBody))
register_pipeline(PassPipeline("c", CPUPassPipelineBody))
register_pipeline(PassPipeline("llvm", CPUPassPipelineBody))
register_pipeline(PassPipeline("metal", MetalPassPipelineBody))
```

resolve 很直接：

```python
def resolve_pipeline(target: Target) -> PassPipeline:
    return get_pipeline(target.kind.name)
```

因此 pipeline 名称必须和 TVM target kind 对齐：

```text
cuda -> tilelang/cuda/pipeline.py
hip -> tilelang/rocm/pipeline.py
c / llvm -> tilelang/cpu/pipeline.py
metal -> tilelang/metal/pipeline.py
```

---

## 8. CUDA pipeline 总览

CUDA pipeline 在 `tilelang/cuda/pipeline.py`。建议拆成两层读：

```text
CUDAPassPipelineBodyPrologue
  高层 IR 清理
  pipeline planning
  LayoutInference
  LowerTileOp
  tile-op 后早期合法化

CUDAPassPipelineBody
  CUDA barrier/tmem/tma 处理
  buffer flatten/storage/vectorize
  host/device split
  packed API/kernel launch lowering
  CUDA 后处理
```

### 8.1 Prologue 第一段：绑定 target 和基础清理

```python
mod = tirx.transform.BindTarget(target)(mod)
if should_force_let_inline():
    mod = tilelang.transform.LetInline()(mod)
mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
mod = tilelang.transform.LegalizeNegativeIndex()(mod)
if should_enable_race_check():
    mod = tilelang.transform.VerifyParallelLoop()(mod)
mod = tilelang.transform.InjectAssumes()(mod)
mod = tilelang.transform.Simplify()(mod)
mod = tilelang.transform.LayoutReducer()(mod)
```

这一段的目标是：

```text
让 IR 更规范，让后续分析能依赖 target、assume、简化后的表达式和规范化 layout/reducer 结构。
```

### 8.2 Prologue 第二段：CUDA 高层重写

```python
if allow_warp_specialized(target=target):
    mod = tilelang.cuda.transform.ProducerConsumerWarpSpecialized()(mod)

mod = tilelang.cuda.transform.LowerBlackwell2SM()(mod)
```

这两个 pass 都必须在 `LayoutInference` 和 `LowerTileOp` 前跑。

原因是：

```text
此时 IR 里还保留 T.copy/T.gemm 的高层语义，
还能判断 producer/consumer、TMA/cp.async、Blackwell 2SM TCGEN05 等信息。
```

一旦 `LowerTileOp` 把高层 tile op 降成低层 intrinsic，再反推这些语义会困难很多。

### 8.3 Prologue 第三段：pipeline planning

```python
mod = tilelang.transform.IfStmtBinding()(mod)
mod = tilelang.transform.PipelinePlanning()(mod)
mod = tilelang.transform.InjectSoftwarePipeline()(mod)
mod = tilelang.transform.Simplify()(mod)
```

`IfStmtBinding` 会把某些 guarded sequence 标准化，让 pipeline planner 能看到 copy/gemm 这样的 schedulable statement。

`PipelinePlanning` 读 `T.Pipelined` 或显式 pipeline annotations，把 loop body 中的 statement 分析成 stage/order，并标注 async producer 信息。

`InjectSoftwarePipeline` 根据这些 annotation 重写 loop body，生成实际的软件流水结构。

注意顺序：pipeline planning 在 layout inference 前。这样 layout inference 看到的是最终 pipelined 结构，而不是未展开的高层 loop。

### 8.4 Prologue 第四段：LayoutInference -> LowerTileOp

```python
mod = tilelang.transform.LayoutInference()(mod)
LayoutVisual(mod)
mod = tilelang.transform.LowerTileOp()(mod)
```

这是 lower 最核心的一段。

`LayoutInference` 回答：

```text
每个 fragment/shared buffer 的 logical tile 怎么映射到 thread/lane/local registers/shared memory layout？
每个 T.Parallel loop 应该如何被 thread partition/vectorize？
```

`LowerTileOp` 回答：

```text
T.copy/T.gemm/T.reduce 具体变成哪些低层 loop 或硬件 intrinsic？
```

### 8.5 Prologue 第五段：tile-op 后合法化

```python
mod = tilelang.cuda.transform.LowerL2Persistent()(mod)
mod = tilelang.transform.DecoupleTypeCast()(mod)
mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
mod = tilelang.transform.LowerAccessPtr()(mod)
mod = tilelang.transform.Simplify()(mod)
mod = tilelang.transform.HoistNonRestrictParams()(mod)
```

`LowerAccessPtr` 很关键：frontend 可能留下 TileLang 自己的 `tl.access_ptr` 或 pointer metadata op，这里会降成标准 `tvm_access_ptr`，方便后续 codegen。

### 8.6 Body 第一段：CUDA barrier/tmem/TMA 处理

```python
mod = tilelang.cuda.transform.LowerSharedTmem()(mod)
mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
mod = tilelang.cuda.transform.LowerSharedBarrier()(mod)

has_tma = module_has_tma(mod)
if has_tma:
    mod = tilelang.cuda.transform.FuseMBarrierArriveExpectTx()(mod)
```

`module_has_tma(mod)` 读取的是 `LowerTileOp` 写入的 `tl.has_tma` function attr。

这点很重要：是否真正用了 TMA，不由 frontend 或 pass config 猜，而由 `LowerTileOp` 的实际 lowering 结果决定。

### 8.7 Body 第二段：buffer/storage/vectorize lowering

```python
mod = tilelang.transform.HoistGlobalBufferAllocations()(mod)
mod = tilelang.transform.LowerOpaqueBlock()(mod)
mod = tilelang.transform.Simplify()(mod)
mod = tirx.transform.NarrowDataType(32)(mod)
mod = tilelang.transform.FlattenBuffer()(mod)
mod = tilelang.transform.ConfigIndexBitwidth()(mod)
mod = tirx.transform.Simplify()(mod)
mod = tilelang.transform.VectorizeLoop(...)(mod)
mod = tilelang.transform.StorageRewrite()(mod)
mod = tilelang.transform.LoopUnswitching()(mod)
mod = tilelang.transform.UnrollLoop()(mod)
```

这一段逐步把“结构化 TIRX + 多维 Buffer + 高层 block”降成更接近 codegen 的低层 TIR。

特别注意 `FlattenBuffer`：它会把 body 内部多维 buffer load/store 展平，但故意不展平 `func.buffer_map` 中的参数 buffer，因为参数 buffer 还要被 runtime validation 和 adapter 用来检查用户输入。

### 8.8 Body 第三段：验证、reduction、CUDA intrinsic lowering

```python
mod = tirx.transform.VerifyMemory()(mod)
mod = tirx.transform.AnnotateEntryFunc()(mod)
mod = s_tir.transform.InferFragment()(mod)
mod = tilelang.transform.LowerThreadAllreduce()(mod)
mod = tilelang.cuda.transform.LowerLDGSTG()(mod)
mod = tilelang.cuda.transform.LowerHopperIntrin()(mod)
```

`LowerThreadAllreduce` 被放在这里有历史原因：注释里说明 TL 只用一个 thread dimension，某些 var binding 信息在后面 legalization/simplify 中会丢，所以需要在这里降低 thread-level allreduce。

### 8.9 Body 第四段：host/device split

```python
mod = tilelang.transform.AnnotateDeviceRegions()(mod)
mod = tilelang.transform.SplitHostDevice()(mod)
mod = tilelang.cuda.transform.MarkCudaSyncCalls(have_pdl(target))(mod)
mod = tilelang.transform.AnnotateReadOnlyParams()(mod)
```

`AnnotateDeviceRegions` 先找到 device region，`SplitHostDevice` 再把它抽成独立 device function。

### 8.10 Body 第五段：shared memory merge 和 sync

```python
mod = tilelang.transform.MergeSharedMemoryAllocations(...)(mod)
mod = tilelang.cuda.transform.InjectFenceProxy()(mod)
mod = tilelang.transform.ThreadSync("shared")(mod)
mod = tilelang.transform.ThreadSync("shared.dyn")(mod)
mod = tilelang.cuda.transform.InjectTcgen05Fence()(mod)
mod = tilelang.transform.MergeIfStmt()(mod)
```

`MergeSharedMemoryAllocations` 必须在 `SplitHostDevice` 后，因为合并后的 allocation site 要放在每个 device function 开头。

### 8.11 Body 第六段：ABI 和 launch lowering

```python
mod = tilelang.transform.MakePackedAPI()(mod)
mod = tilelang.transform.Simplify()(mod)
mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
mod = tilelang.cuda.transform.PersistThreadblock()(mod)
```

`MakePackedAPI` 改 host function ABI。

`LowerDeviceKernelLaunch` 改 host 调 device function 的方式，并给 device function 标注 launch metadata。

---

## 9. PreLowerSemanticCheck

`PreLowerSemanticCheck(mod)` 是 backend-independent 的轻量检查。

源码在 `tilelang/engine/semantic_check.py`：

```python
if should_enable_ast_print():
    tilelang.analysis.ASTPrinter()(mod)
tilelang.analysis.NestedLoopChecker()(mod)
tilelang.analysis.FragmentLoopChecker()(mod)
```

它在 target-specific lowering 前运行，所以不应该放 CUDA/HIP 特定逻辑。

可通过 pass config 关闭：

```python
PassConfigKey.TL_DISABLE_PRELOWER_SEMANTIC_CHECK
```

可通过 pass config 打印 AST：

```python
PassConfigKey.TL_AST_PRINT_ENABLE
```

---

## 10. LayoutInference 深入

`LayoutInference` 在 `src/transform/layout_inference.cc`。

它的核心目标不是改成硬件指令，而是推导布局并把布局挂回 IR annotation。

### 10.1 输入 IR 形态

进入 `LayoutInference` 前，IR 仍然保留：

```text
tl.tileop.copy
tl.tileop.gemm
tl.tileop.reduce
local.fragment buffer
shared/shared.dyn buffer
parallel loop
pipeline-expanded structure
```

这些高层信息是 layout inference 的基础。

### 10.2 核心数据结构

`LayoutInferenceResult` 包含：

```cpp
struct LayoutInferenceResult {
  Map<Buffer, Layout> layout_map;
  Map<For, Fragment> for_map;
  Map<For, PrimExpr> predicate_map;
  Map<For, Bool> padding_guard_map;
};
```

含义：

| 字段 | 含义 |
| --- | --- |
| `layout_map` | 每个 Buffer 对应的 Layout，包括 fragment/shared layout |
| `for_map` | parallel loop 对应的 Fragment layout |
| `predicate_map` | ragged partition 或越界时需要的 predicate |
| `padding_guard_map` | inverse lowering 是否需要 padding guard |

### 10.3 BufferUseDefCollector

`BufferUseDefCollector` 是 layout inference 的主分析器。

它会收集：

```text
参与 layout inference 的 TileOperator
每个 operator 读写哪些 BufferRegion
Buffer data Var 的 alias 关系
thread binding / thread bounds
parallel loop 的 layout 约束
floating fragment buffers
explicit annotated layout
```

它不是一次性自顶向下推导，而是做多阶段迭代。

### 10.4 推导阶段

`BufferUseDefCollector::Run()` 的推导过程可以简化成：

```text
0. layout_map = annotated_layout_map
1. floating fragment buffers -> FullyReplicated
2. strict inference
3. strict result 保存为 strict_layout_map
4. common inference，用 BFS 队列传播 use-def 更新
5. free mode 放宽约束继续推导
6. 按同一个 Buffer data Var 做 alias layout 传播
7. 检查所有 local.fragment 都有 layout
8. 生成 For loop layout/predicate/padding info
```

### 10.5 operator 如何参与 layout inference

每个 tile op 都继承/实现类似接口：

```cpp
LayoutMap TileOperator::InferLayout(LayoutInferArgs, InferLevel)
```

例如：

| op | infer 来源 |
| --- | --- |
| copy | target-specific `CopyImpl.infer_layout` |
| gemm | C++ `GemmNode::InferLayout` 反调 Python `tl.gemm.infer_layout` |
| parallel loop | `ParallelOp::InferLayout` |

GEMM 的 layout inference 很有代表性：

```text
GemmNode::InferLayout
  -> Function::GetGlobal("tl.gemm.infer_layout")
  -> Python Gemm.infer_layout
  -> select instruction key
  -> resolve implementation class
  -> GemmMMA/GemmWGMMA/GemmTCGEN05/MFMA/WMMA/Scalar infer_layout
  -> 返回 Buffer -> Layout
```

### 10.6 LayoutInferencer 写回 annotation

`LayoutInferencer::Substitute` 做两件事：

```text
1. 把 block annotation `layout_map` 写成完整 result.layout_map
2. 对 parallel For 写入：
     parallel_loop_layout
     parallel_loop_predicate
     parallel_loop_requires_padding_guard
```

然后运行 `ParallelLoopLayoutValidator` 检查：

```text
所有 parallel loop lowering 前都必须有 layout annotation
嵌套 parallel loop 只允许最外层有 annotation
layout InputDim 必须等于连续嵌套 parallel loop 数量
```

---

## 11. LowerTileOp 深入

`LowerTileOp` 在 `src/transform/lower_tile_op.cc`。

它是真正把高层 TileOp 兑现成低层 IR 的核心 pass。

### 11.1 输入 IR 形态

进入 `LowerTileOp` 时，IR 已经有：

```text
block annotations: layout_map
parallel loop annotations: parallel_loop_layout / predicate
tile op calls: tl.tileop.copy/gemm/reduce/fill/...
target attr
thread_extent
pipeline-expanded loop/body
```

### 11.2 初始化

`LowerTileOpPass::Substitute(PrimFunc f)` 会：

```text
1. 从 func.buffer_map 建立 handle var -> Buffer、data var -> Buffer 映射
2. 读取 target attr
3. VisitStmt 重写 body
4. RemapBufferRewriter 修 padding/safe value map
5. LayoutRemapRewriter 更新 block layout_map
6. 写入 tl.has_tma attr
7. 如有 TMA mbarrier，向 root block 注入 barrier buffer 和 barrier_init annotation
8. CPU target 下把 synthetic fallback thread var 重写成 0
```

### 11.3 buffer remap

LayoutInference 推导出来的 layout 会影响 buffer 的实际 shape 和索引。

`makeBufferWithLayout(buffer, layout, var_remap)` 会：

```text
local.fragment -> local pointer
global buffer -> 保持原 data var
shared buffer -> 根据 layout output shape 改 shape
shared replication -> 如果原 buffer extent 大于 layout extent，前面插 replicate dimension
```

因此 `LowerTileOp` 后，很多 `local.fragment` 不再以 fragment scope 存在，而变成普通 local allocation + 特定 index mapping。

### 11.4 tile op 解析

`VisitStmt_(EvaluateNode)` 中：

```cpp
auto tile_op = ParseOperator(GetRef<Stmt>(op));
if (!tile_op.defined())
    return normal_visit;

auto lowered = tile_op->Lower(LowerArgs{...}, analyzer_);
return VisitStmt(lowered);
```

`ParseOperator` 在 `src/op/operator.cc`：

```text
Evaluate(Call(op=tl.tileop.xxx, args, annotations))
  -> Op attr map "TLOpBuilder"
  -> 构造 Copy/Gemm/Reduce/Fill/... TileOperator object
```

### 11.5 LowerArgs

`LowerTileOp` 传给每个 tile op 的 `LowerArgs` 包含：

| 字段 | 作用 |
| --- | --- |
| `target` | 后端选择 CUDA/HIP/CPU/Metal lowering |
| `thread_bounds` | 当前 threadIdx.x 的范围 |
| `thread_var` | 线程变量 |
| `workspace callback` | tile op 需要临时 shared workspace 时申请 buffer |
| `mbarrier callback` | TMA/cp.async barrier 需要时申请 mbarrier slot |
| `barrier arrive callback` | 更新 barrier arrive count |
| `layout_map` | LayoutInference 结果 |
| `buffer_remap` | 原 buffer 到 lowered buffer 的映射 |
| `bind_var_to_expr` | Bind var 到表达式的映射 |
| `mbar_phase_expr` | pipeline loop 中推导的 mbarrier parity/phase |
| `cluster_size` | cluster dims 乘积，用于 cluster/TMA 相关 lowering |

### 11.6 parallel loop lowering

`LowerTileOp` 还处理 `T.Parallel` 产生的 `ForKind::kParallel`。

逻辑大致是：

```text
如果不是 parallel loop -> 正常 visit
如果是嵌套 parallel loop 的内层且没有 annotation -> 跳过，由外层统一处理
读取 parallel_loop_layout
读取 predicate / padding guard
判断是否有 non-local store，决定是否 thread partition
判断是否有 reducer，决定是否 vectorize
调用 LowerParallelLoop
CUDA 下如果条件满足，注入 PTX cp.async
```

这里体现 TileLang 的设计：parallel loop 的 layout annotation 放在最外层 loop，LowerTileOp 一次性 lower 整个嵌套 parallel nest。

---

## 12. Copy lowering 路线

`T.copy` 在 frontend 中生成 `tl.tileop.copy`。LowerTileOp 解析成 `CopyNode` 后，调用：

```cpp
CopyNode::Lower(const LowerArgs& T, Analyzer* analyzer)
  -> LowerCopyForTarget(*this, T, analyzer)
  -> ResolveCopyImpl(T.target).lower(...)
```

### 12.1 CopyImpl registry

`src/op/copy.cc` 中有 target-specific registry：

```cpp
std::vector<CopyImpl> registry;
RegisterCopyImpl(...);
ResolveCopyImpl(target);
```

不同 backend 会在 native 侧注册自己的 copy implementation。

### 12.2 默认 SIMT copy

如果走普通 copy，`LowerNormalCopy` 会：

```text
1. CopyNode::MakeSIMTLoop 生成 parallel loop
2. ParallelLoopFuser fuse loop nest
3. ParallelOp infer layout
4. LowerParallelLoop 变成 thread-partitioned/vectorized code
```

### 12.3 CUDA copy

CUDA copy 在 `src/cuda/op/copy.cc`。

它会根据 target、scope、annotation 和 region 特征选择：

```text
global -> shared.dyn 且满足条件 -> TMA load 或 cp.async
shared -> global 且满足条件 -> TMA store 或 vectorized store
local/shared/global 普通路径 -> SIMT/vectorized loop
im2col 特殊路径 -> Im2Col lowering
cluster mask / leader scope / barrier annotation -> TMA multicast/leader election/mbarrier
```

一些重要 annotation：

| annotation | 含义 |
| --- | --- |
| `disable_tma` | 禁用 TMA path |
| `is_tma_copy` | 显式 TMA copy |
| `is_async_copy` / `force_cp_async` | 请求 cp.async |
| `eviction_policy` | cache eviction hint |
| `cluster_mask` | cluster multicast mask |
| `barrier` | 显式 mbarrier buffer load |
| `leader_scope_threads` | TMA leader election 范围 |
| `async_copy_no_implicit_commit_wait` | 不自动插 commit/wait |

---

## 13. GEMM lowering 路线

GEMM 是 lower 里最有代表性的跨语言路径。

### 13.1 C++ GemmNode

`src/op/gemm.cc` 解析 frontend 参数：

```text
A/B/C regions
trans_A/trans_B
M/N/K
policy
clear_accum
stride/offset
kPack
wg_wait
mbarrier
c coordinates
scale factor regions for blockscaled GEMM
annotations: is_wgmma / is_tcgen05
```

### 13.2 instruction key 选择

`GemmNode::getGemmInstructionKey(block_size, target)` 会调用 target-specific `GemmImpl` registry。

Python 侧 `Gemm._select_gemm_instruction` 再通过 `_ffi_api.GemmGetGemmInstructionKey` 调回 C++。

典型 instruction key：

| key | target/硬件路径 |
| --- | --- |
| `cuda.mma` | CUDA MMA |
| `cuda.wgmma` | Hopper WGMMA |
| `cuda.tcgen05` | Blackwell TCGEN05 |
| `rocm.mfma` | AMD MFMA |
| `rocm.wmma` | AMD WMMA |
| `cpu.scalar` | CPU scalar fallback |
| `metal.simdgroup` | Metal simdgroup GEMM |

### 13.3 infer_layout

`GemmNode::InferLayout`：

```text
C++ Function::GetGlobal("tl.gemm.infer_layout")
  -> Python Gemm.infer_layout
  -> select gemm instruction
  -> resolve implementation class
  -> impl.infer_layout
  -> 返回 Buffer -> Layout
```

比如 CUDA MMA 的 layout inference 会给：

```text
shared A/B -> swizzled layout
fragment A/B -> mma load layout
fragment C -> mma store layout
```

WGMMA 下 shared layout 要满足 tensor core shared memory access pattern，可能选择 full/half/quarter bank swizzle 或 linear layout。

### 13.4 lower

`GemmNode::Lower`：

```text
C++ Function::GetGlobal("tl.gemm.lower")
  -> Python Gemm.lower
  -> impl.lower(...)
  -> Python 内部定义一个 @T.prim_func macro
  -> _Simplify(...)
  -> 返回 PrimFunc body / SBlockRealize
```

例如 CUDA MMA 路线：

```python
@T.prim_func
def _gemm_ssr() -> None:
    A_local = T.alloc_local(...)
    B_local = T.alloc_local(...)
    if clear_accum:
        T.clear(C_buf)
    for ki in T.serial(...):
        mma_emitter.ldmatrix_a(...)
        mma_emitter.ldmatrix_b(...)
        mma_emitter.mma(...)
```

WGMMA 路线则更直接调用：

```python
mma_emitter.wgmma(A_region, B_region, C_region, clear_accum, wg_wait)
```

这一段说明：LowerTileOp 不是只在 C++ 里展开所有东西。GEMM 的硬件 intrinsic macro 生成很大一部分在 Python tileop implementation 中完成。

---

## 14. PipelinePlanning 和 InjectSoftwarePipeline

`PipelinePlanning` 在 `src/transform/pipeline_planning.cc`。

它处理两类输入：

```text
显式 tl_pipeline_order / tl_pipeline_stage annotations
T.Pipelined(..., num_stages=...) 产生的 num_stages annotation
```

它会分析 loop body 中哪些 statement 是 schedulable，哪些 scalar bind 可以 replay，哪些 copy 是 async producer。

重要概念：

| 概念 | 含义 |
| --- | --- |
| `PipelineStageInfo` | 某个 statement 的 stage、依赖、读写 buffer 信息 |
| `PipelineStageAnalyzer` | 分析 copy last use、producer propagation、scalar dependency |
| `PipelinePlanningBodyAnalyzer` | 分析 loop body 中可调度 statement 和 replayable bind |
| `software_pipeline_order` | 后续 software pipeline pass 使用的执行顺序 |
| `software_pipeline_stage` | 每个 statement 所属 pipeline stage |
| `software_pipeline_async_stages` | 哪些 stage 可使用 async copy |

然后 `InjectSoftwarePipeline` 根据这些 annotation 重写 loop。这样 `LayoutInference` 看到的已经是 pipelined 后的最终结构。

---

## 15. Host/Device 分离

lower 后半段最重要的结构性变化是把一个 frontend PrimFunc 拆成 host function 和 device kernel function。

### 15.1 AnnotateDeviceRegions

`src/transform/annotate_device_regions.cc` 做一件事：

```text
看到 thread_extent / device_scope / pipeline_exec_scope
  -> 包一层 AttrStmt(device_target, tvm::attr::kTarget, ...)
```

CPU `c` target 会特殊处理：如果 host target 存在且 device target 是 `c`，整个 function body 会被包成 device target region。

### 15.2 SplitHostDevice

`src/transform/split_host_device.cc` 看到 `AttrStmt(attr_key=tvm::attr::kTarget)` 后，会调用 `SplitDeviceFunc`。

它做：

```text
1. 从 device body use-def 分析 undefined vars，作为 device function params
2. 为 device params 创建新的 Var，避免和 host Var 共享对象导致 ConvertSSA 混淆
3. Substitute body 中旧 Var -> 新 Var
4. remap buffers
5. 对 CPU/ext_dev 可传播 error code 的 target，device function 返回 int32
6. 插入 DeclBuffer
7. 把 host-side assume 搬到 device body 外层
8. 设置 device attrs:
     target
     noalias
     is_global_func
     non_restrict_params
     cluster_dims
9. 新建 GlobalVar，例如 xxx_kernel
10. device_mod->Add(kernel_symbol_global, device_func)
11. host body 替换成 Call(GlobalVar, old_params)
```

也就是说，`SplitHostDevice` 之后，一个 module 里同时有：

```text
host function:
  负责 packed API、参数解析、调用 device kernel

device function:
  负责真正 kernel body，后续被 device codegen 编成 CUDA/HIP/Metal/C
```

### 15.3 Source kernel 特殊路径

TileLang 支持 `T.CUDASourceCodeKernel` 这类外部 CUDA source kernel。SplitHostDevice 对这种 source kernel 没有普通 DSL body 可分析，所以会从 host PrimFunc signature 和 buffer metadata 重建 device signature。

`tilelang_callback_cuda_validate` 会检查：

```text
code_block_source 中必须有 __global__ kernel
global_symbol 必须匹配 external CUDA source 中的 kernel name
code_block_entry_name 如果存在，也必须匹配 global_symbol
```

---

## 16. MakePackedAPI

`MakePackedAPI` 在 `src/transform/make_packed_api.cc`。

它把 host function 从普通 TIR function 改成 TVM packed function ABI。

### 16.1 触发条件

`RequiresPackedAPI(func)` 大致要求 function 有 `global_symbol` 且带 host target。

如果 target 没有 host，说明可能已经 lowered 过，直接返回。

### 16.2 新签名

MakePackedAPI 会把 host function 参数改成：

```text
self_handle: handle
args: handle
num_args: int32
result: void*
```

并把 calling convention 改成：

```text
CallingConv::kCPackedFunc
global_symbol = tvm_ffi_symbol_prefix + 原 global_symbol
target = target_host
```

### 16.3 参数解析

它会对每个原始 param：

```text
读取 TVMFFIAny type_index
检查 tensor/scalar 类型是否符合预期
从 packed args 取 value
绑定 scalar var
绑定 DLTensor buffer
绑定 shape/stride/offset/device id
```

对 buffer 参数，它特别处理了未使用 input buffer：如果某些 buffer 只用来承载 shape symbol，而 data pointer 没有被访问，就允许 nullable，并在 `BindDLTensors` 中生成“至少一个 carrier 非空”的逻辑。

### 16.4 device context

如果参数绑定中出现 device id，MakePackedAPI 会插入：

```text
device_id attr
device_type attr
必要时 tvm_set_device(device_type, device_id)
```

### 16.5 返回值

host packed function 最终返回 int32 error code：

```text
成功 -> T.ret(0)
```

---

## 17. LowerDeviceKernelLaunch

`LowerDeviceKernelLaunch` 在 `src/transform/lower_device_kernel_launch.cc`。

它把 host function 中对 device function 的普通 `Call(GlobalVar, args)` 改成 runtime kernel launch call。

### 17.1 DeviceInfoCollector

先收集每个 device function 的 `KernelInfo`：

```cpp
struct KernelInfo {
  Target target;
  String global_symbol;
  Array<Var> params;
  Array<String> launch_params;
  Array<PrimExpr> launch_args;
  Map<String, PrimExpr> thread_extent;
  Optional<PrimExpr> dyn_shmem_size;
  Optional<Array<Integer>> cluster_dims;
};
```

来源：

| 信息 | 从哪里收集 |
| --- | --- |
| thread extent | `AttrStmt(thread_extent)` |
| dynamic shared memory | `AllocBuffer` scope 为 `shared.dyn` |
| cluster dims | device PrimFunc attr `cluster_dims` |
| global symbol | PrimFunc attr 或 GlobalVar name |

### 17.2 host call 改写

当 host function 调 device function 时：

```text
Call(GlobalVar kernel, args)
```

会被改成：

```text
tvm_call_packed(
  global_symbol,
  original_args...,
  launch_args...
)
```

launch args 包括：

```text
grid/block/thread extent
dynamic shared memory size
cluster dim x/y/z
programmatic dependent launch flags
cooperative launch flags
```

### 17.3 device function attr 更新

如果某个 device function 被 host 作为 kernel launch 调用，会标注：

```text
calling_conv = DEVICE_KERNEL_LAUNCH
kernel_launch_params = launch_params
global_symbol = info.global_symbol
thread_extent = map
dyn_shared_memory_buf = size, 如果有
cluster_dims = dims, 如果有
```

这就是后面 `lower.py` 的 `Filter(_is_device_call)` 能筛出 device_mod 的原因。

---

## 18. Codegen

### 18.1 CUDA codegen

CUDA codegen 在 `src/cuda/codegen/rt_mod_cuda.cc`。

`BuildTileLangCUDA`：

```text
1. ValidateUniqueDeviceGlobalSymbols
2. tilelang_callback_cuda_validate
3. 对每个 DEVICE_KERNEL_LAUNCH PrimFunc 调 CodeGenTileLangCUDA::AddFunction
4. cg.Finish() 得到 CUDA source
5. tilelang_callback_cuda_postproc 可选后处理
6. tilelang_callback_cuda_compile 调 nvcc 编译 cubin/ptx
7. CUDAModuleCreateWithFallback(binary, fmt, function_info, source_map)
```

`BuildTileLangCUDAWithoutCompile`：

```text
1. 同样生成 CUDA source
2. 不调用 nvcc
3. 用 dummy PTX payload 创建 CUDA module container
4. source_map 中保留真实 CUDA source
```

### 18.2 function info

CUDA module 会提取：

```text
arg types
launch param tags
global symbol
cluster dims
programmatic dependent launch / cooperative launch flags
```

这些信息会进入 runtime module，供 TVM runtime launch 使用。

### 18.3 compile callback

Python `tilelang_callback_cuda_compile` 在 `lower.py` 注册。

它会根据 target compute capability 生成：

```text
-arch=sm_xx
-std=c++20
-I TILELANG_TEMPLATE_PATH
-I CUTLASS_INCLUDE_DIR
--use_fast_math, 如果开启
--ptxas-options=--register-usage-level=N, 如果设置
--ptxas-options=--verbose, 如果设置
tl.device_compile_flags, 如果设置
```

然后调用 `tilelang.contrib.nvcc.compile_cuda`。

HIP 类似，走 `tilelang_callback_hip_compile` 和 `hipcc.compile_hip`。

---

## 19. Adapter：怎么包成 Python callable

lower 输出的 `CompiledArtifact` 还不是用户最终调用的对象。`JITKernel` 会根据 execution backend 创建 adapter。

### 19.1 TVM FFI backend

`TVMFFIKernelAdapter` 接收：

```text
params
result_idx
target
func_or_mod
host_mod
device_mod
rt_mod
device_kernel_source
```

它会：

```text
1. 从 PrimFunc.params/buffer_map 处理动态 shape/stride symbol
2. 创建 runtime.Executable(rt_mod)
3. 调用时根据 result_idx 自动创建输出 tensor
4. 把输入/输出 tensor list 传给 executable
5. 返回指定输出 tensor
```

动态 shape map 的含义：

```text
Var -> (id, buffer_index, dimension, stride_scale)
id = 0: shape
id = 1: stride
id = 2: scalar param
```

sub-byte dtype 会用 `stride_scale` 修正 PyTorch stride 和 kernel logical element stride 的差异。

### 19.2 Cython backend

`CythonKernelAdapter` 会：

```text
1. 用 TLWrapper 根据 host_mod/device_mod/device source 生成 C++ host wrapper
2. LibraryGenerator 编译 shared library
3. ctypes load library
4. CythonKernelWrapper 负责 PyTorch tensor 参数、输出 tensor、动态 shape、ptr map
```

### 19.3 NVRTC backend

`NVRTCKernelAdapter` 会：

```text
1. 用 TLPyWrapper 生成 Python host_func
2. NVRTCLibraryGenerator 编译 device_kernel_source
3. CUDA driver cuLibraryGetKernel 拿 kernel handle
4. 调用时用 PyTorch tensor 指针和当前 stream launch
```

### 19.4 eager/lazy 的最终区别

JITImpl 最后分支：

```python
if self.mode == "eager":
    return kernel(*kernel_args.values())
else:
    return kernel
```

所以：

```text
eager style: 第一次调用 DSL 函数时 compile + 立即执行
lazy style: 第一次调用 factory 时 compile + 返回 JITKernel，用户再调用 kernel(*tensors)
```

---

## 20. 关键 pass 速查表

| pass | 位置 | 作用 |
| --- | --- | --- |
| `PreLowerSemanticCheck` | `tilelang/engine/semantic_check.py` | lower 前 backend-independent 检查 |
| `BindTarget` | TVM/TIRX | 给 PrimFunc 绑定 target attr |
| `VerifyParallelLoop` | `src/transform/verify_parallel_loop.cc` | 检查 parallel loop 读写安全性 |
| `InjectAssumes` | `src/transform/inject_assumes.cc` | 把 assume 变成 pass/analyzer 可利用形态 |
| `LayoutReducer` | `src/transform/layout_reducer.cc` | reducer/layout 规约整理 |
| `IfStmtBinding` | `src/transform/if_stmt_binding.cc` | 标准化 if/bind，暴露 pipeline 可调度语句 |
| `PipelinePlanning` | `src/transform/pipeline_planning.cc` | 分析 software pipeline stage/order/async producer |
| `InjectSoftwarePipeline` | `src/transform/inject_pipeline.cc` | 根据 pipeline annotations 展开流水 |
| `LayoutInference` | `src/transform/layout_inference.cc` | 推导 Buffer/parallel loop layout |
| `LowerTileOp` | `src/transform/lower_tile_op.cc` | 降 `tl.tileop.*` 和 parallel loop |
| `LowerAccessPtr` | `src/transform/lower_access_ptr.cc` | 降 TileLang frontend pointer metadata |
| `FlattenBuffer` | `src/transform/flatten_buffer.cc` | 展平内部多维 buffer access |
| `VectorizeLoop` | `src/transform/vectorize_loop.cc` | loop vectorization |
| `StorageRewrite` | `src/transform/storage_rewrite.cc` | storage reuse/planning/pointer type rewrite |
| `LowerThreadAllreduce` | `src/transform/lower_thread_allreduce.cc` | thread-level allreduce lowering |
| `AnnotateDeviceRegions` | `src/transform/annotate_device_regions.cc` | 标记 device region |
| `SplitHostDevice` | `src/transform/split_host_device.cc` | 抽出 device PrimFunc |
| `AnnotateReadOnlyParams` | `src/transform/annotate_read_only_params.cc` | 标记 read-only params，帮助 CUDA codegen const/read-only cache |
| `MergeSharedMemoryAllocations` | `src/transform/merge_shared_memory_allocations.cc` | 合并/reuse shared memory allocation |
| `ThreadSync` | `src/transform/thread_storage_sync.cc` | 插入 shared memory sync |
| `MakePackedAPI` | `src/transform/make_packed_api.cc` | host function 改 TVM packed ABI |
| `LowerDeviceKernelLaunch` | `src/transform/lower_device_kernel_launch.cc` | host call 改 runtime kernel launch |

CUDA 专属 pass：

| pass | 作用 |
| --- | --- |
| `ProducerConsumerWarpSpecialized` | 在高层 tile-op IR 上做 producer/consumer warp specialization |
| `LowerBlackwell2SM` | Blackwell 2SM TCGEN05 相关预处理 |
| `LowerL2Persistent` | 降 l2 persistent map |
| `LowerSharedTmem` | shared.tmem 初始化和 slot lowering |
| `LowerSharedBarrier` | shared barrier buffer/init lowering |
| `FuseMBarrierArriveExpectTx` | TMA mbarrier arrive/expect-tx 融合 |
| `LowerLDGSTG` | global load/store 降成 LDG/STG intrinsic |
| `LowerHopperIntrin` | Hopper 相关 intrinsic lowering |
| `MarkCudaSyncCalls` | 标记 pdl_sync/pdl_trigger 等 CUDA sync calls |
| `InjectFenceProxy` | TMA/async proxy fence 注入 |
| `InjectTcgen05Fence` | Blackwell TCGEN05 fence 注入 |
| `AnnotateWarpGroupRegAlloc` | warp specialization 下标注 warpgroup register allocation |
| `PersistThreadblock` | persistent threadblock transformation |

---

## 21. IR 形态变化总览

可以用一条“IR 形态演化线”理解 lower：

```text
Frontend PrimFunc
  - 高层 TIRX block/frame
  - tl.tileop.copy/gemm/reduce
  - local.fragment/shared.dyn
  - T.Parallel/T.Pipelined annotations

Pre-lower normalized IR
  - target attr
  - assume attr
  - simplified expressions
  - pipeline annotations normalized

Layout annotated IR
  - block.annotations[layout_map]
  - for.annotations[parallel_loop_layout]
  - for.annotations[parallel_loop_predicate]

Tile-op lowered IR
  - tileop mostly gone
  - fragment -> local
  - shared layout remapped
  - copy/gemm -> loop/intrinsic/mbarrier/TMA/MMA/WGMMA/TCGEN05
  - parallel loops partitioned/vectorized

Low-level TIR
  - flattened buffers
  - storage rewritten
  - vectorized/unrolled/unswitched loops
  - thread allreduce lowered
  - CUDA-specific LDGSTG/Hopper/TMEM/fence lowered

Host/device mixed module
  - host PrimFunc
  - device PrimFunc
  - host call to device kernel

Packed launch ABI module
  - host function: C packed / TVM FFI ABI
  - device function: DEVICE_KERNEL_LAUNCH
  - launch params encoded

Codegen result
  - CUDA/HIP/Metal/C source or binary module
  - CompiledArtifact
  - Python adapter callable
```

---

## 22. 推荐源码阅读路线

### 第一遍：只看 Python 调度骨架

按这个顺序：

```text
1. tilelang/engine/lower.py
2. tilelang/backend/pass_pipeline/pipeline.py
3. tilelang/cuda/pipeline.py
4. tilelang/rocm/pipeline.py / cpu/pipeline.py / metal/pipeline.py 对比
5. tilelang/transform/__init__.py
```

目标是回答：

```text
target 是怎么选 pipeline 的？
pipeline 中 pass 顺序是什么？
哪些 pass 是通用，哪些 pass 是 CUDA-only？
lower 输出 host_mod/device_mod 的时间点在哪里？
```

### 第二遍：读两个核心 pass

重点读：

```text
src/transform/layout_inference.cc
src/transform/lower_tile_op.cc
src/op/operator.cc
src/op/copy.cc
src/op/gemm.cc
tilelang/tileop/gemm/__init__.py
tilelang/cuda/op/gemm/*.py
src/cuda/op/copy.cc
```

目标是回答：

```text
layout 是怎么从 copy/gemm/parallel loop 推出来的？
layout_map 写回哪里？
LowerTileOp 怎么找到 tile op？
copy 和 gemm 为什么 lower 路线不同？
GEMM 为什么 C++ 会反调 Python？
```

### 第三遍：读 ABI 和 codegen

重点读：

```text
src/transform/annotate_device_regions.cc
src/transform/split_host_device.cc
src/transform/make_packed_api.cc
src/transform/lower_device_kernel_launch.cc
src/cuda/codegen/rt_mod_cuda.cc
tilelang/jit/kernel.py
tilelang/jit/adapter/tvm_ffi.py
tilelang/jit/adapter/cython/adapter.py
tilelang/jit/adapter/nvrtc/adapter.py
```

目标是回答：

```text
一个 PrimFunc 如何拆成 host/device 两个 PrimFunc？
host function 如何变成 packed API？
thread_extent/dyn shared memory/cluster dims 如何变成 launch args？
CUDA source 是哪里生成的？
Python callable 是哪里包出来的？
```

---

## 23. 常见误区

### 23.1 lower.py 不等于 lower pipeline

`lower.py` 很薄，只做调度和 codegen。真正 pass 顺序在 backend `pipeline.py`。

### 23.2 LayoutInference 不做硬件 lowering

LayoutInference 只是推导 layout 并写 annotation。真正把 `T.copy/T.gemm` 变成 cp.async/TMA/MMA/WGMMA 的是 LowerTileOp 和 tileop implementation。

### 23.3 LowerTileOp 不是纯 C++

copy 主要走 native target registry；gemm 会 C++ 解析和选择 instruction，然后反调 Python `tl.gemm.lower` 生成 macro body。

### 23.4 SplitHostDevice 后 lower.py 才 Filter

host/device 拆分发生在 pipeline 内的 `SplitHostDevice`。`lower.py` 最后的 `Filter` 只是把同一个 lowered module 里的 host/device function 分成两份。

### 23.5 `enable_device_compile=False` 也会 codegen source

`device_codegen_without_compile` 不是“不 codegen”，而是“不编译成 cubin/hsaco”。CUDA source 仍然会生成，并被 adapter 使用。

### 23.6 `func.buffer_map` 不只是 frontend 细节

buffer_map 在 lower 后半段仍然非常重要：参数提取、SplitHostDevice source kernel signature、MakePackedAPI DLTensor binding、adapter 动态 shape 解析都依赖它。

---

## 24. 建议的源码实验

### 24.1 dump IR 看每个 pass 后形态

使用 pass config：

```python
pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_DUMP_IR: True,
    tilelang.PassConfigKey.TL_DUMP_IR_DIR: "./dump_ir",
}
```

然后对比：

```text
LayoutInference 前后
LowerTileOp 前后
SplitHostDevice 前后
MakePackedAPI 前后
LowerDeviceKernelLaunch 前后
```

### 24.2 打开 layout visualization

```python
pass_configs={
    tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True,
    tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS: "txt",
}
```

观察 fragment/shared layout 如何变化。

### 24.3 对比 GEMM 路线

选一个 matmul kernel，调不同 target 或 pass config：

```text
CUDA SM80 -> MMA
CUDA SM90 -> WGMMA
CUDA SM100 -> TCGEN05
CPU c -> scalar
ROCm -> MFMA/WMMA
```

观察 `tl.gemm.infer_layout` 和 `tl.gemm.lower` 的 implementation class 如何变化。

### 24.4 禁用 async copy / TMA

试：

```python
pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_ASYNC_COPY: False,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}
```

然后看 `T.copy` 是否从 TMA/cp.async 路线退回 SIMT/vectorized loop。

### 24.5 比较 execution backend

同一个 PrimFunc 分别用：

```text
execution_backend="tvm_ffi"
execution_backend="cython"
execution_backend="nvrtc"
```

观察 lower codegen 开关、host wrapper、device compile 位置的差异。

---

## 25. 最后总结

`tilelang.lower` 可以抓住三条主线：

```text
主线一：高层 tile 语义兑现
  LayoutInference -> LowerTileOp -> copy/gemm/reduce target implementation

主线二：低层 TIR 合法化和优化
  FlattenBuffer -> VectorizeLoop -> StorageRewrite -> UnrollLoop -> backend-specific intrinsic/fence/barrier pass

主线三：运行时 ABI 成形
  AnnotateDeviceRegions -> SplitHostDevice -> MakePackedAPI -> LowerDeviceKernelLaunch -> codegen -> adapter
```

理解这三条线后，再看任何具体 pass 就不容易迷路：

```text
这个 pass 是在高层 tile op 还存在时工作，
还是在 tile op 已经 lower 后工作？

它是在决定 layout / instruction，
还是在整理 buffer/storage/vectorization，
还是在为 runtime ABI/codegen 做准备？
```

这也是读 TileLang lowering 的关键心智模型。