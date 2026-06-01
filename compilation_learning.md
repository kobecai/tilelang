# TileLang compilation process and source design

这份笔记的目标是把 TileLang 的编译链路从 Python DSL、TileOp、LayoutInference、LowerTileOp 一直追到 target codegen 和 JIT adapter。内容按当前仓库实现校对，路径以本 checkout 为准。

前提：你已经会写基础 TileLang DSL，能跑 `T.Kernel`、`T.copy`、`T.gemm` 这类小 kernel，理解 `alloc_shared`、`alloc_fragment`、`Pipelined`、`Parallel` 的用途。

## 一、从第 1 周开始的学习路线

阶段目标：从「会用」过渡到「会读」，再过渡到「会改 + 能提小 PR」。

### 第 1 周：GEMM 上下游 + Layout / Fragment 抽象

主题：把 `T.gemm` 的整条路径拉直，从 Python DSL 一路追到 `GemmNode::Lower` 和具体后端实现。

#### Day 1：DSL 入口与 IR 形态

先读 `tilelang/language/gemm_op.py` 里的入口：

```python
def gemm(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    mbar: BarrierType | None = None,
) -> tirx.PrimExpr:
```

这一层主要做三件事：

- 把 `Buffer` / `BufferLoad` / `BufferRegion` 规整成 region 参数。
- 检查 `M/N/K`、stride、offset、transpose 等约束。
- 生成 `tirx.call_intrin("handle", Op.get("tl.tileop.gemm"), ...)`，也就是一个高层 TileOp intrinsic。

它本身不生成 PTX、CUDA、Metal 或 HIP 源码。

当前仓库里 `tilelang.engine.lower.lower_to_host_device_ir` 已经存在。如果目标是看完整 host/device split，优先用它；如果只是想手动观察某两段 pass 的效果，也可以直接调用 `LowerAndLegalize` 和 `OptimizeForTarget`。

CUDA 机器上建议先跑这个最小例子：

```python
import tilelang  # noqa: F401
import tilelang.language as T
from tilelang.engine.lower import lower_to_host_device_ir


@T.prim_func
def gemm_128x128(
    A: T.Tensor((128, 128), "float16"),
    B: T.Tensor((128, 128), "float16"),
    C: T.Tensor((128, 128), "float32"),
):
    with T.Kernel(1, threads=128):
        A_s = T.alloc_shared((128, 128), "float16")
        B_s = T.alloc_shared((128, 128), "float16")
        C_l = T.alloc_fragment((128, 128), "float32")

        T.copy(A, A_s)
        T.copy(B, B_s)
        T.clear(C_l)
        T.gemm(A_s, B_s, C_l)
        T.copy(C_l, C)


print("=== 原始 PrimFunc ===")
print(gemm_128x128.script())

host_mod, device_mod, _, target, target_host = lower_to_host_device_ir(
    gemm_128x128,
    target="cuda -arch=sm_80",
)

print("=== host IR ===")
print(host_mod.script())
print("=== device IR ===")
print(device_mod.script())
```

观察重点：

- 原始 `PrimFunc` 里 `T.gemm` / `T.copy` 仍是 `tl.tileop.*` 高层 intrinsic。
- CUDA lowering 后会看到更低层的 MMA / copy intrinsic，例如 `ptx_mma`、`cp.async` 或对应 TIR 调用。
- `T.copy(A, A_s)` 的具体实现由 target 决定：CUDA 可能走 `cp.async` / TMA / 普通 SIMT copy，Metal 走 Metal 后端 copy，CPU 走标量或向量化 loop。

Mac / Metal 上现在也有 `T.gemm` 实现，路径是 `metal.simdgroup`。它主要支持 shared-shared GEMM：A、B 在 shared/shared.dyn，C 可以在 `local.fragment`、`metal.simdgroup` 或 shared/shared.dyn。可参考这个更接近 Metal 测试的版本：

```python
import tilelang
import tilelang.language as T


@T.prim_func
def metal_gemm(
    A: T.Tensor((128, 128), "float16"),
    B: T.Tensor((128, 128), "float16"),
    C: T.Tensor((128, 128), "float32"),
):
    with T.Kernel(8, 8, threads=128) as (bx, by):
        A_s = T.alloc_shared((16, 16), "float16")
        B_s = T.alloc_shared((16, 16), "float16")
        C_s = T.alloc_shared((16, 16), "float32")

        T.clear(C_s)
        for ko in T.Pipelined(8, num_stages=0):
            T.copy(A[by * 16, ko * 16], A_s)
            T.copy(B[ko * 16, bx * 16], B_s)
            T.gemm(A_s, B_s, C_s)
        T.copy(C_s, C[by * 16, bx * 16])


kernel = tilelang.compile(metal_gemm, target="metal", execution_backend="torch")
print(kernel.get_kernel_source()[:1024])
```

如果本地 native library 是 `USE_CUDA=OFF` 构建，`target="cuda -arch=sm_80"` 可能找不到 CUDA target-specific implementation。这不是 DSL 问题，而是构建产物没有注册 CUDA backend。要看 CUDA GEMM 完整 lowering，需要 CUDA-enabled wheel 或 Linux + CUDA 源码构建。

#### Day 2：TileOp 在 C++ 里的样子

先读 `src/op/operator.h`：

```cpp
enum class InferLevel : uint8_t {
  kFree = 0,
  kCommon = 1,
  kStrict = 2,
};

class TileOperatorNode : public Object {
public:
  virtual Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const = 0;
  virtual LayoutMap InferLayout(const LayoutInferArgs &T,
                                InferLevel level) const = 0;
  virtual TileOperator Clone() const = 0;
  virtual AccessRegions GetAccessRegions() const;
};
```

这是 TileLang 最核心的 tile-level IR 抽象。`T.copy`、`T.gemm`、`T.reduce_*`、`T.fill`、`T.atomic_*` 都会被解析成某个 `TileOperatorNode` 子类。每个子类负责两件事：

1. `InferLayout`：声明这条 op 希望它访问的 buffer 使用什么 `Layout` / `Fragment`。
2. `Lower`：在 layout 已确定后，把 op 降成更低层 TIR。

GEMM 的 C++ lowering 在 `src/op/gemm.cc`：

```cpp
Stmt GemmNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  if (const auto f = Function::GetGlobal("tl.gemm.lower")) {
    PrimExpr mbar_phase = T.mbar_phase_expr;
    if (auto explicit_phase = GetAnnotatedMbarPhaseExpr(annotations_)) {
      mbar_phase = explicit_phase.value();
    }
    auto prim_func =
        Downcast<PrimFunc>((*f)(GetRef<Gemm>(this), T.layout_map, T.target,
                                T.thread_bounds, T.thread_var, mbar_phase));
    ...
  }
}
```

关键点：`GemmNode::Lower` 自己不展开 PTX/WGMMA/TCGEN05 细节，而是通过 FFI 调到 Python：

```text
src/op/gemm.cc::GemmNode::Lower
  -> "tl.gemm.lower"
  -> tilelang/tileop/gemm/__init__.py::gemm_lower
  -> Gemm.lower(...)
  -> _select_gemm_instruction(...)
  -> resolve_gemm_impl(gemm_inst, target)
  -> GemmMMA / GemmMMASm70 / GemmMMASm75 / GemmWGMMA / GemmTCGEN5 / GemmMFMA / GemmWMMA / GemmMetal / GemmScalar
```

注意 `T.copy` 不是这套 Python FFI lowering。Copy 的 target dispatch 在 C++ registry 里完成：`src/op/copy.cc` 调 `ResolveCopyImpl(T.target)`，具体实现注册在 `src/backend/{cuda,rocm,metal,cpu,webgpu}/op/copy.cc`。

#### Day 3：Layout / Fragment 抽象

读 `src/layout/layout.h` 和 `tilelang/layout/fragment.py`。精确心智模型：

| 概念 | 数学对象 | 含义 |
| --- | --- | --- |
| `Layout` | 逻辑坐标 -> 存储坐标 | shared/global/local buffer 怎么排，包括 swizzle、padding、flatten |
| `Fragment` | 逻辑坐标 -> 存储坐标 + thread 坐标 + replicate | register fragment 上每个 thread 持有哪些逻辑元素 |

记一句话：`Layout` 描述空间布局；`Fragment` 描述空间布局加线程分布。`Fragment` 继承自 `Layout`，多了 `forward_thread` 和 `replicate_size`。

实验：

```python
import tilelang  # noqa: F401
from tilelang.layout import Layout, Fragment, make_gemm_fragment_8x8

# 1) 最简单的 row-major layout
L = Layout([128, 128], lambda i, j: i * 128 + j)
print("input_shape  =", list(L.get_input_shape()))
print("output_shape =", list(L.get_output_shape()))
print("forward_idx  =", L.get_forward_index())

# 2) 一个 4x4 逻辑 tile 摊给 4 个 thread
# Fragment.forward_fn 的返回顺序是 (forward_thread, forward_index)
F = Fragment([4, 4], forward_fn=lambda i, j: (i, j))
print("thread expr  =", F.thread)
print("thread size  =", F.get_thread_size())
print("index expr   =", F.get_forward_index())

# 3) Tensor Core 常用的 8x8 fragment
F_tc = make_gemm_fragment_8x8()
print("TC fragment  =", F_tc)
```

GEMM 专用 fragment 构造在 `src/layout/gemm_layouts.cc`。第一次读不必逐个公式硬算，但要知道 `makeGemmFragmentA/B/C` 是按硬件 MMA/WGMMA/TCGEN05 的数据排布规则手写出来的。

#### Day 4：LayoutInference pass

读 `src/transform/layout_inference.cc`。核心结构：

```cpp
struct LayoutInferenceResult {
  Map<Buffer, Layout> layout_map;
  Map<For, Fragment> for_map;
  Map<For, PrimExpr> predicate_map;
};
```

`InferLevel` 的含义：

| Level | 作用 |
| --- | --- |
| `kStrict` | 强约束，典型是 GEMM 这种必须使用硬件要求 layout 的 op |
| `kCommon` | 常规传播，让 copy / parallel 等 op 沿 producer-consumer 关系继承 layout |
| `kFree` | 仍未确定时给默认 layout，让普通 SIMT loop 也能被 lowered |

简化流程：

1. 先以 `kStrict` 跑所有 TileOp，收集不能随便改的 layout。
2. 再以 `kCommon` 做队列传播，layout 更新后通过 `use_list_` 把相关 op 重新入队。
3. 对仍未确定的 fragment / loop，以 `kFree` 尝试补齐默认 layout。
4. 遇到同一个 buffer 推出不兼容 layout，会在 `Get different layout...` 路径报错。

启用 layout 可视化：

```python
kernel = tilelang.compile(
    my_func,
    target="cuda -arch=sm_80",
    pass_configs={
        "tl.layout_visualization_enable": True,
        "tl.layout_visualization_formats": "txt",
    },
)
```

可视化默认格式就是 `txt`，也支持 `png`、`pdf`、`svg` 和 `all`。输出在 TileLang cache 目录下。

#### Day 5：LowerTileOp 与 `makeBufferWithLayout`

读 `src/transform/lower_tile_op.cc`：

```cpp
static Buffer makeBufferWithLayout(const Buffer &buffer, const Layout &layout,
                                   Map<Var, Var> &var_remap) {
  ...
  // convert fragments to normal local buffer
  if (IsFragmentBuffer(buffer)) {
    new_type = PointerType(ptr_type->element_type, "local");
    ...
  }
}
```

这是 `local.fragment` 变成普通 thread-local buffer 的关键位置。理解之后就能解释：

- 用户写 `T.alloc_fragment((128, 128), "float32")` 时拿到的是逻辑 tile。
- `LayoutInference` 决定这个逻辑 tile 如何分摊到 threads。
- `LowerTileOp` 按 `Fragment` 改写 `BufferLoad/Store`，把逻辑坐标变成 thread-local 索引。

#### Day 6-7：复盘与验收

写一份「`T.gemm` 从 DSL 到 backend intrinsic 的全链路」笔记，至少回答：

1. 写 `T.gemm(A_s, B_s, C_l)` 时 DSL 层做了什么？
2. `tl.tileop.gemm` 在哪一步被解析成 `GemmNode`？
3. `GemmNode::InferLayout` 如何通过 `"tl.gemm.infer_layout"` 回到 Python？
4. `GemmNode::Lower` 如何通过 `"tl.gemm.lower"` 选择具体 Python implementation？
5. `sm_80`、`sm_90`、`sm_100`、`metal` 为什么会分叉到不同实现？

CUDA 环境验收：

```bash
python -m pytest testing/python/kernel/test_tilelang_kernel_gemm.py -x
```

Metal 环境可参考：

```bash
python -m pytest testing/python/metal/test_metal_gemm_v2.py -x
```

### 第 2 周：Layout / Fragment 深入 + 端到端 regression test

主题：从「能解释路径」过渡到「能修改它，并提供 regression test」。

#### Day 1-2：GEMM 变体与后端分叉

先读 `tilelang/tileop/gemm/gemm_base.py` 以及各后端实现：

- CUDA：`tilelang/cuda/op/gemm/gemm_mma.py`、`gemm_mma_sm70.py`、`gemm_mma_sm75.py`、`gemm_wgmma.py`、`gemm_tcgen05.py`
- ROCm：`tilelang/rocm/op/gemm/gemm_mfma.py`、`gemm_wmma.py`
- Metal：`tilelang/tileop/gemm/gemm_metal.py`
- CPU：`tilelang/cpu/op/gemm/gemm_scalar.py`

然后自己各写一个 SS / SR / RS / RR 能支持的 kernel，观察：

- `kernel.get_kernel_source()` 的差异。
- `infer_layout` 返回的 A/B/C layout 差异。
- `resolve_gemm_impl(gemm_inst, target)` 在不同 target 上选到哪个 class。

后端分叉表：

| Target / arch | C++ instruction key | Python implementation | 主要低层指令 / 形态 |
| --- | --- | --- | --- |
| CUDA Volta `sm_70` | `cuda.mma` | `GemmMMASm70` | Volta MMA path |
| CUDA Turing `sm_75` | `cuda.mma` | `GemmMMASm75` | Turing MMA path |
| CUDA Ampere `sm_80` | `cuda.mma` | `GemmMMA` | `mma.sync` / `ldmatrix` |
| CUDA Hopper `sm_90` | `cuda.wgmma` 或 `cuda.mma` | `GemmWGMMA` 或 `GemmMMA` | `wgmma.mma_async` / wait group |
| CUDA Blackwell `sm_100+` | `cuda.tcgen05` | `GemmTCGEN5` | `tcgen05.mma`，C 通常在 `shared.tmem` |
| ROCm | `hip.mfma` / `hip.wmma` | `GemmMFMA` / `GemmWMMA` | MFMA / WMMA |
| Metal | `metal.simdgroup` | `GemmMetal` | Metal simdgroup matrix |
| CPU / LLVM / C | `cpu.scalar` | `GemmScalar` | scalar loop |

#### Day 3：Layout 冲突场景

构造一个会让同一个 buffer 被不同 op 推出不兼容 layout 的例子，例如同一个 shared buffer 被两个 GEMM 以不同 transpose 方式消费。目标是把异常栈追到 `src/transform/layout_inference.cc` 里 `Get different layout...` 的路径。

这是适合作为第一批 PR 的方向：改善错误信息，但要以当前代码能拿到的信息为边界。例如从：

```text
Get different layout for <buffer>
 current layout: ...
 previous layout: ...
```

改成包含更明确的 buffer name、infer level、当前 op kind、前后 layout 摘要。不要承诺 file:line，除非你确实把 source span 接进了 TileOp。

#### Day 4：WGMMA / TCGEN05 / Metal simdgroup 对照

只读不写，重点看：

- `src/backend/cuda/op/gemm.cc`：CUDA 侧如何选 `cuda.mma` / `cuda.wgmma` / `cuda.tcgen05`。
- `src/backend/metal/op/gemm.cc`：Metal 如何注册 `metal.simdgroup`。
- `tilelang/cuda/op/gemm/gemm_wgmma.py`：Hopper WGMMA lowering。
- `tilelang/cuda/op/gemm/gemm_tcgen05.py`：Blackwell TCGEN05 lowering 和 `shared.tmem` 约束。
- `tilelang/tileop/gemm/gemm_metal.py`：Metal simdgroup matrix lowering。

产出一份「同一个 `T.gemm` 在 CUDA / ROCm / Metal / CPU 上如何分叉」的短笔记。

#### Day 5-7：完成第一份可 PR 的工件

推荐选题：

- 在 `testing/python/transform/test_tilelang_transform_layout_inference.py` 补一个 layout 冲突的 `pytest.raises` 测试。
- 在 `testing/python/kernel/test_tilelang_kernel_gemm.py` 补一个没有覆盖到的 transpose / dtype / tile shape corner case。
- 在 `testing/python/metal/` 下补一个 Metal GEMM 或 simdgroup store 的边界 case。

验收：

```bash
python -m pytest <你的测试文件> -x
pre-commit run --all-files
```

CUDA-only 行为必须在 CUDA 环境跑；Mac 上只能验证不依赖 CUDA runtime 的 transform 或 Metal/MPS 路径。

### 第 3 周：Pass Pipeline

主题：把 `tilelang/engine/phase.py` 里的 pass 按依赖关系串起来，理解每个 pass 的位置为什么在那里。

#### 两段 pipeline

`lower_to_host_device_ir` 先做 `PreLowerSemanticCheck`，再依次执行：

1. `LowerAndLegalize(mod, target)`
2. `OptimizeForTarget(mod, target)`
3. `Filter(get_host_call(...))` / `Filter(get_device_call(...))`

`LowerAndLegalize` 的主要职责是把 frontend Tile IR 降成更普通的 TIR：

| 顺序 | Pass / 逻辑组 | 作用 |
| --- | --- | --- |
| 1 | `BindTarget` | 把 target 绑定到 `PrimFunc` |
| 2 | `LetInline`（可选） | 受 `tl.force_let_inline` 控制 |
| 3 | `AddWrapperForSingleBufStore`、`LegalizeNegativeIndex`、`InjectAssumes`、`Simplify` | 前端写法规整 |
| 4 | `VerifyParallelLoop`（可选） | 受 data race check 配置控制 |
| 5 | `LayoutReducer` | reducer layout 预处理 |
| 6 | `ProducerConsumerWarpSpecialized`（可选） | 高层 tile-op 上做 warp specialization |
| 7 | `LowerBlackwell2SM` | Blackwell 2SM TCGEN05 相关预处理 |
| 8 | `PipelinePlanning`、`InjectSoftwarePipeline`、`Simplify` | 软件流水线规划与展开 |
| 9 | `metal.MetalFragmentToSimdgroup` | Metal GEMM accumulator 改写，发生在 layout inference 前 |
| 10 | `LayoutInference`、`LayoutVisual` | 推断并可视化 layout / fragment |
| 11 | `LowerTileOp`、`LowerL2Persistent` | 消除高层 TileOp |
| 12 | `DecoupleTypeCast`、`LegalizeVectorizedLoop`、`LegalizeSafeMemoryAccess`、`LowerAccessPtr`、`Simplify` | vectorize / OOB / access ptr 合法化 |
| 13 | `HoistNonRestrictParams` | 参数 annotation 收尾 |

`OptimizeForTarget` 的主要职责是 target-aware 优化、host/device split 和 launch lowering：

| 顺序 | Pass / 逻辑组 | 作用 |
| --- | --- | --- |
| 1 | `LowerSharedTmem`、`IfStmtBinding` | TMEM 和 if binding 预处理 |
| 2 | `PlanAndUpdateBufferAllocationLocation`、`LowerSharedBarrier`、`FuseMBarrierArriveExpectTx`（TMA 时） | buffer allocation 与 barrier |
| 3 | `HoistGlobalBufferAllocations`、`LowerOpaqueBlock`、`Simplify` | allocation / block 清理 |
| 4 | `NarrowDataType`、`FlattenBuffer`、`ConfigIndexBitwidth`、`Simplify` | index 和 buffer flatten |
| 5 | `VectorizeLoop`、`StorageRewrite`、`LoopUnswitching`、`UnrollLoop` | loop / storage 优化 |
| 6 | `RenormalizeSplitPattern`、`RemoveNoOp`、`HoistIfThenElse`、`VerifyMemory`、`AnnotateEntryFunc` | TIR 清理和校验 |
| 7 | `InferFragment`、`LowerThreadAllreduce`、`LowerLDGSTG`、`LowerHopperIntrin` | fragment / allreduce / ISA-specific lowering |
| 8 | `AnnotateDeviceRegions`、`SplitHostDevice`、`MarkCudaSyncCalls`、`AnnotateReadOnlyParams` | host/device 分离 |
| 9 | `MergeSharedMemoryAllocations` | split 后合并 device shared memory allocation |
| 10 | `InjectFenceProxy`、`ThreadSync("shared")`、`ThreadSync("shared.dyn")`、`InjectTcgen05Fence` | fence / sync |
| 11 | `MergeIfStmt`、`AnnotateWarpGroupRegAlloc`（可选） | 收尾优化 |
| 12 | `MakePackedAPI`、`Simplify`、`LowerDeviceKernelLaunch`、`PersistThreadblock` | host API 和 launch lowering |

本周实验：临时在 `phase.py` 的关键 pass 后 `print(mod.script())`，跑一个 32x32 或 64x64 matmul，记录 IR 在哪几个 pass 后变化最大。不要把调试打印提交到 PR。

可做 PR：

- 给某个 transform 测试补更精确断言，而不是只 assert「编译不挂」。
- 给 pass config 的文档或错误信息补当前实际支持的值，例如 layout visualization 支持 `txt,png,pdf,svg,all`。

### 第 4 周：JIT / Adapter / Target

主题：搞清楚 `@tilelang.jit` / `tilelang.compile` 之后发生了什么，不同 `execution_backend` 如何分叉。

主调用链：

```text
tilelang.compile(...)
  -> tilelang.jit.compile(...)
  -> tilelang.cache.cached(...)
  -> resolve_execution_backend(...)
  -> 对应 KernelCache.cached(...)
  -> tilelang.engine.lower.lower(...)
  -> lower_to_host_device_ir(...)
  -> device_codegen_without_compile(...) 或 device_codegen(...)
  -> adapter 包装成 JITKernel
```

`tilelang/jit/execution_backend.py` 当前兼容矩阵：

| target | allowed execution backend |
| --- | --- |
| `cutedsl` target | `cutedsl` |
| CUDA | `tvm_ffi`, `nvrtc`, `cython` |
| HIP | `tvm_ffi`, `cython` |
| Metal | `tvm_ffi`, `torch` |
| C | `cython`, `tvm_ffi` |
| fallback | `cython`, `tvm_ffi` |

`auto` 的默认选择：

- CuTeDSL target -> `cutedsl`
- CUDA / HIP / Metal -> `tvm_ffi`
- 其他 target -> `cython`

Metal 上可对比两个 adapter：

```python
k1 = tilelang.compile(metal_gemm, target="metal", execution_backend="tvm_ffi")
k2 = tilelang.compile(metal_gemm, target="metal", execution_backend="torch")

print("source equal:", k1.get_kernel_source() == k2.get_kernel_source())
print("k1 adapter  :", type(k1.adapter).__name__)
print("k2 adapter  :", type(k2.adapter).__name__)
```

PR 候选：

- `resolve_execution_backend` 的错误信息目前已经包含 target kind 和 allowed backends；如果继续改进，建议补完整 normalized target、requested backend、available backends 与 unavailable reason。
- 给 `determine_target` 或 `resolve_execution_backend` 补 Mac / MPS / Metal 的优先级测试。

### 第 5 周起：独立 PR

优先级从低风险到高风险：

1. 错误信息类：LayoutInference 冲突、backend 不可用、target/backend 不匹配。
2. 回归测试类：`T.copy` strided + boundary、GEMM transpose x dtype、Metal simdgroup 边界。
3. 小型 transform 测试：`LegalizeVectorizedLoop`、`LegalizeSafeMemoryAccess`、`LowerSharedBarrier` 的 corner case。
4. 文档类：安装、target auto selection、pass config 支持项。

第一批不建议碰 `MergeSharedMemoryAllocations`、`ProducerConsumerWarpSpecialized`、`PipelinePlanning` 的主体逻辑。这几块牵连大，review 门槛高，适合读熟 pipeline 后再改。

## 二、源码核心设计与抽象

下面把 TileLang 当作「基于 TVM 的 tile-level compiler」拆成几个层次。

### 2.1 全栈分层

```text
Python DSL 层
  tilelang/language/*
  T.Kernel / T.copy / T.gemm / T.alloc_* / T.Parallel
        |
        v
TIR PrimFunc + TileOp intrinsics
  tir.For / tir.BlockRealize / tir.call_intrin("tl.tileop.*")
        |
        v
TileOperator IR
  src/op/*
  GemmNode / CopyNode / ParallelOp / ReduceNode / FillNode / ...
        |
        v
LayoutInference -> LowerTileOp
  local.fragment / shared layout 被确定并消除
        |
        v
普通 TIR
  loops / BufferLoad / BufferStore / ptx_mma / wgmma / tma / metal simdgroup
        |
        v
OptimizeForTarget -> SplitHostDevice
        |
        v
Host IR + Device IR
        |
        v
target.build.* + JIT adapter
```

关键事实：大多数 `T.xxx` 不是直接生成后端代码，而是先进入 TileOp 抽象。Layout 推断、TileOp lowering、target codegen 是分阶段发生的。

### 2.2 TileOperator 的双接口契约

`src/op/operator.h` 定义了 TileOp 的核心契约：

```cpp
virtual Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const = 0;
virtual LayoutMap InferLayout(const LayoutInferArgs &T,
                              InferLevel level) const = 0;
```

常见映射：

| Python DSL | C++ TileOperator | 主要文件 |
| --- | --- | --- |
| `T.gemm` / `T.wgmma_gemm` / `T.tcgen05_gemm` | `GemmNode` | `src/op/gemm.cc` |
| `T.gemm_sp` | `GemmSPNode` | `src/op/gemm_sp.cc` |
| `T.copy` / `T.async_copy` / `T.tma_copy` | `CopyNode` | `src/op/copy.cc` |
| `T.Parallel(...)` body | `ParallelOp` | `src/op/parallel.cc` |
| `T.reduce_*` | `ReduceNode` / related nodes | `src/op/reduce.cc` |
| `T.fill` / `T.clear` | `FillNode` | `src/op/fill.cc` |
| `T.atomic_*` | atomic nodes | `src/op/atomic_*.cc` |
| `T.region(...)` | `RegionOp` | `src/op/region.cc` |

阅读建议：先读 `operator.h`，再读 `gemm.cc` 和 `copy.cc`。`Gemm` 和 `Copy` 是最好的对照组，因为 GEMM 用 Python FFI 生成低层 IR，而 Copy 主要走 C++ backend registry。

### 2.3 Layout / Fragment

简化模型：

```text
Layout:
  forward_index = f(i0, i1, ..., in)

Fragment:
  forward_index  = f(i0, i1, ..., in, rep)
  forward_thread = g(i0, i1, ..., in, rep)
```

关键操作：

- `Layout::Inverse()`：LowerTileOp 重写 buffer 访问时用。
- `Fragment::BindThreadRange(...)`：把 fragment 的 thread 映射绑定到当前 block thread 范围。
- `Layout::Reshape(...)`：处理不同 dtype / view 共享底层 storage 的场景。

`local.fragment` 是 TileLang 自己的抽象，TVM 原生没有。它必须先经过 `LayoutInference`，再由 `LowerTileOp` 改写成普通 `local` buffer。

### 2.4 `LowerArgs` / `LayoutInferArgs`

`src/op/operator.h` 里这两个 struct 是 TileOp 能拿到的上下文。

`LowerArgs` 包含：

- `target`
- `thread_bounds`
- `thread_var`
- `AddWorkspace`
- `AllocMBarrier`
- `UpdateBarrierArrive`
- `layout_map`
- `buffer_remap`
- `let_var_to_expr`
- `mbar_phase_expr`
- `mbarrier_buffer`
- `cluster_size`

`LayoutInferArgs` 包含：

- `target`
- `thread_bounds`
- `layout_map`
- `analyzer`
- `buffer_oob`
- `buffer_remap`
- `let_var_to_expr`
- `in_pipeline`

后端加新能力时，经常是往这些上下文里加字段，而不是改 TileOperator 的虚函数签名。

### 2.5 Pass pipeline 的切分

`LowerAndLegalize` 和 `OptimizeForTarget` 的边界可以这样记：

- `LowerAndLegalize`：处理 frontend Tile IR，完成 layout inference 和 TileOp lowering。结束后通常不应再有普通的 `tl.tileop.*` 残留。
- `OptimizeForTarget`：处理 buffer flatten、storage rewrite、thread sync、host/device split、packed API、device launch lowering 等 target-aware 工作。

几个顺序约束要特别记住：

- `LayoutInference` 必须在 `LowerTileOp` 前。
- `MetalFragmentToSimdgroup` 必须在 `LayoutInference` 前，因为 Metal simdgroup accumulator 是 opaque 形态。
- `MergeSharedMemoryAllocations` 在 `SplitHostDevice` 后，这样合并发生在 device function 内。
- `InjectTcgen05Fence` 在 `ThreadSync` 后，因为它需要看到 storage sync。

### 2.6 LayoutInference 的图传播模型

`layout_inference.cc` 不是单向 lowering，而是带优先级的不动点传播。

核心数据关系：

- `layout_map`：buffer -> layout / fragment。
- `use_list_`：buffer -> 使用它的 TileOp infer index。
- alias propagation：同一底层 data 的不同 view 需要联动。
- `EnqueueWithPriority(...)`：某个 buffer layout 更新后，把受影响 op 重新放进队列。

读这段时不要一开始陷入所有 case。先用 GEMM + Copy 的小图理解：GEMM 给 A/B/C 强约束，Copy 把约束传回 producer 或传向 consumer，Parallel 在没有强约束时给默认 fragment。

### 2.7 Memory scope 系统

常见 scope：

| Scope | 物理含义 | 常见来源 |
| --- | --- | --- |
| `global` | DRAM / kernel 参数 | function tensor params |
| `shared.dyn` | 可被 shared memory merge 复用的 SMEM | `T.alloc_shared(...)` 默认 |
| `shared` | 不参与动态 shared reuse 的 SMEM | `T.alloc_shared(..., scope="shared")` |
| `shared.barrier` | mbarrier storage | `T.alloc_barrier` / `T.alloc_cluster_barrier` |
| `shared.tmem` | Blackwell tensor memory | `T.alloc_tmem` |
| `local` | thread-private local/register-like storage | `T.alloc_local` |
| `local.fragment` | TileLang fragment 抽象 | `T.alloc_fragment` |
| `local.var` | 标量变量 buffer | `T.alloc_var` |
| `metal.simdgroup` | Metal simdgroup matrix storage | Metal GEMM lowering 内部使用 |

调 shared memory 问题时先看 scope。`T.alloc_shared` 默认是 `shared.dyn`，会参与 `MergeSharedMemoryAllocations`；显式 `scope="shared"` 通常表示不要走动态 shared reuse。

### 2.8 Target 后端注册表模式

GEMM 有两层 dispatch。

C++ 层在 `src/op/gemm.cc` / `src/op/gemm.h` 里定义：

```cpp
struct GemmImpl {
  const char *name;
  GemmTargetPredicate match_target;
  String (*select_inst)(...);
  std::pair<int, int> (*compute_warp_partition)(...);
  bool (*reuse_existing_shared_layout)(String gemm_inst);
  String (*instruction_kind)(String gemm_inst);
};
```

注册点：

- CUDA：`src/backend/cuda/op/gemm.cc`
- ROCm：`src/backend/rocm/op/gemm.cc`
- Metal：`src/backend/metal/op/gemm.cc`
- CPU：`src/backend/cpu/op/gemm.cc`

Python 层在 `tilelang/tileop/gemm/registry.py`：

```python
register_gemm_impl(name, inst_name, predicate, impl_class)
resolve_gemm_impl(gemm_inst, target)
```

典型流程：

1. C++ `GemmImpl::select_inst` 根据 target、shape、dtype、policy 选 instruction key。
2. Python `resolve_gemm_impl(gemm_inst, target)` 根据 instruction key + target predicate 选 implementation class。
3. 对应 Python class 生成具体 TIR。

Copy 的 dispatch 与 GEMM 不同：`src/op/copy.cc` 用 C++ `CopyImpl` registry，具体 target implementation 在 `src/backend/*/op/copy.cc`。

### 2.9 JIT Adapter / Execution Backend

最外层可调用对象是 `tilelang.jit.kernel.JITKernel`。各 backend 的 cache/adapter 负责把 `CompiledArtifact` 包成 Python 可调用对象：

- `tvm_ffi`：TVM PackedFunc / DLPack 路径，CUDA/HIP/Metal 常用默认。
- `cython`：Cython wrapper。
- `nvrtc`：CUDA NVRTC 路径，需要额外依赖。
- `torch`：Metal/MPS 侧常用，用 PyTorch tensor 对接。
- `cutedsl`：CuTeDSL target 专用。

每次新增 target 或 execution backend，都要检查 `tilelang/jit/execution_backend.py`、`tilelang/cache/__init__.py` 和对应 adapter cache。

### 2.10 Python <-> C++ FFI 协作模式

TileLang 不是纯 C++ compiler，也不是纯 Python compiler。常见协作点：

| C++ / codegen 调用 | Python / 注册位置 | 作用 |
| --- | --- | --- |
| `"tl.gemm.infer_layout"` | `tilelang/tileop/gemm/__init__.py` | GEMM layout 推断 |
| `"tl.gemm.lower"` | `tilelang/tileop/gemm/__init__.py` | GEMM 低层 TIR 生成 |
| `"tilelang_callback_cuda_compile"` | `tilelang/engine/lower.py` | CUDA codegen 回调 Python 侧编译 |
| `"target.build.tilelang_cuda"` | `src/backend/cuda/codegen/rt_mod_cuda.cc` | CUDA device module build |
| `"target.build.tilelang_hip"` | `src/backend/rocm/codegen/rt_mod_hip.cc` | HIP device module build |
| `"target.build.tilelang_metal"` | `src/target/codegen_metal.cc` | Metal source build |
| `"target.build.tilelang_c"` / `"target.build.tilelang_c_host"` | `src/target/rt_mod_c.cc` / `src/target/codegen_c_host.cc` | C / host codegen |

看到 `Function::GetGlobal(...)` 或 `tvm.ffi.get_global_func(...)` 时，直接 `rg` 搜字符串，通常能很快找到另一端。

## 三、建议

1. 不要按文件夹顺序读源码。按 `Gemm + Copy` 两条链路横切：DSL -> TileOp -> LayoutInference -> LowerTileOp -> target backend -> adapter。
2. 每次只抓一个具体 kernel。先看原始 `PrimFunc`，再看 `host_mod` / `device_mod`，最后看 generated source。
3. Mac 上优先跑 Metal 和 transform 测试；CUDA 行为放到 CUDA-enabled 环境验证。
4. 第一个 PR 优先选错误信息或测试。这两类改动风险低，能逼你读懂链路，也更容易 review。
5. 读 GEMM 时同步读 Copy。GEMM 展示 Python FFI registry，Copy 展示 C++ backend registry，两者合起来基本覆盖 TileLang 的后端扩展模式。
