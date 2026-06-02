# TileLang `language/` 目录源码阅读指南

这份笔记整理 `tilelang/language` 目录的职责、核心设计、关键代码细节和推荐阅读路线。重点不是只列文件名，而是解释：**用户写的 Python DSL 是如何被捕获、转换成 TIRX `PrimFunc`，并保留高层 tile op 给后续 compiler pass lowering 的**。

## 目录

- [1. 一句话结论](#1-一句话结论)
- [2. 核心心智模型](#2-核心心智模型)
- [2.1 核心精髓：两种 JIT 风格与 AST 改写](#21-核心精髓两种-jit-风格与-ast-改写)
- [3. 从用户代码到 CUDA/HIP/CPU 的整体数据流](#3-从用户代码到-cudahipcpu-的整体数据流)
- [4. 推荐阅读顺序](#4-推荐阅读顺序)
- [5. `__init__.py`：`T` 命名空间的总菜单](#5-__init__pyt-命名空间的总菜单)
- [6. `kernel.py`：`T.Kernel` 与 launch frame 机制](#6-kernelpytkernel-与-launch-frame-机制)
- [7. `eager/ast.py`：Python AST 如何变成 Builder 调用](#7-eagerastpypython-ast-如何变成-builder-调用)
- [8. `eager/builder.py`：真正的 IR 施工队](#8-eagerbuilderpy真正的-ir-施工队)
- [9. `proxy.py`：Tensor、Buffer、ptr 类型代理](#9-proxypytensorbufferptr-类型代理)
- [10. `allocate.py`：内存模型与 scope 字符串](#10-allocatepy内存模型与-scope-字符串)
- [11. `loop.py`：循环抽象与编译策略](#11-looppy循环抽象与编译策略)
- [12. `copy_op.py`：数据移动的高层 TileOp](#12-copy_oppy数据移动的高层-tileop)
- [13. `gemm_op.py`：GEMM 高层 TileOp](#13-gemm_oppygemm-高层-tileop)
- [14. `reduce_op.py`：Reduction 与 macro 展开](#14-reduce_oppyreduction-与-macro-展开)
- [15. `frame.py`：Let 值和 BufferRegion alias 追踪](#15-framepylet-值和-bufferregion-alias-追踪)
- [16. `annotations.py`：编译 hint 注入](#16-annotationspy编译-hint-注入)
- [17. `warpgroup.py`：Warp Specialization scope](#17-warpgrouppywarp-specialization-scope)
- [18. `builtin.py` 与底层 intrinsic 包装](#18-builtinpy-与底层-intrinsic-包装)
- [19. `tir/`、`parser/`、`overrides/`：TVM Script 兼容层](#19-tirparseroverridestvm-script-兼容层)
- [20. 下游边界：`language/` 生成的 IR 在哪里被兑现](#20-下游边界language-生成的-ir-在哪里被兑现)
- [21. 一个 DSL 片段如何落成 IR：逐行解释](#21-一个-dsl-片段如何落成-ir逐行解释)
- [22. 常见误区](#22-常见误区)
- [23. 建议的源码实验](#23-建议的源码实验)
- [24. 阅读时抓住的 6 个关键问题](#24-阅读时抓住的-6-个关键问题)
- [25. 文件职责速查表](#25-文件职责速查表)
- [26. 最后总结](#26-最后总结)
- [附录 A. `KernelLaunchFrame`、`TIRFrame`、`FrameStack` 问答补充](#附录-a-kernellaunchframetirframeframestack-问答补充)
- [附录 B. `register_object`、`_ffi_api` 和 C++ FFI 绑定顺序](#附录-b-register_object_ffi_api-和-c-ffi-绑定顺序)
- [附录 C. `@tilelang.jit`、AST、IRGenerator 补充问答](#附录-c-tilelangjitastirgenerator-补充问答)

---

阅读时可以按三层理解这份文档：

- **主线章节 1-4**：先建立整体心智模型、JIT 风格、数据流和推荐阅读顺序。
- **源码章节 5-20**：按文件/模块解释 `language/` 里每个 API 如何构造 TIR/TIRX，以及下游 pass 在哪里消费这些语义。
- **实践与附录 21-26、A-C**：用例子、误区、实验和问答补齐细节；附录保留多次源码追问中的细节，但不打断主线阅读。

---

## 1. 一句话结论

`tilelang/language` 是 TileLang 的 **Python DSL 前端层**。

它的核心工作是：

```text
把用户写的 Python DSL
    T.Kernel / T.Tensor / T.alloc_shared / T.copy / T.gemm / T.Parallel / T.Pipelined ...
翻译成 TVM/TIRX 的 PrimFunc 和高层 TileLang IR 节点
    block/thread launch frame
    buffer allocation
    For/If/Let/BufferStore
    tl.tileop.copy / tl.tileop.gemm / tl.tileop.reduce
然后交给后续 lowering pipeline 继续编译成 CUDA/HIP/CPU 代码
```

因此，用户写：

```python
T.copy(A, B)
```

背后不是在 Python 里执行数据拷贝，也不是立即 launch GPU kernel，而是在当前 `IRBuilder` 里追加一个高层 TIR 节点，大致是：

```python
tirx.call_intrin(
    "handle",
    tirx.op.Op.get("tl.tileop.copy"),
    src_region,
    dst_region,
    annotations=ann,
)
```

后续 `LayoutInference`、`LowerTileOp`、pipeline/codegen passes 才会把这个高层 tile op 降成 cp.async、TMA、ldmatrix、vectorized store 等低层指令序列。

---

## 2. 核心心智模型

阅读这个目录时，建议把整个系统分成 5 层：

```text
用户 API 层
  tilelang.language as T
  T.Kernel / T.Tensor / T.copy / T.gemm / T.Parallel ...

Python DSL 捕获层
  eager AST rewrite
  TVM script parser compatibility
  @tilelang.jit lazy/eager mode inference

IR Builder / Frame 层
  Builder.current()
  IRBuilder
  TIRFrame / ForFrame / KernelLaunchFrame / LetFrame

高层 TileOp 表达层
  tl.tileop.copy
  tl.tileop.gemm
  tl.tileop.reduce
  layout / scope / pipeline / annotation metadata

下游 lowering 边界
  backend pipeline
  LayoutInference
  LowerTileOp
  FlattenBuffer / StorageRewrite / codegen
```

最重要的一点：**`language/` 的职责不是最终优化，也不是 codegen，而是保留足够多的语义，让后端 pass 能做正确且激进的 lowering。**

---

## 2.1 核心精髓：两种 JIT 风格与 AST 改写

如果只能抓住 `language/` 的一个核心机制，那就是这一段：**TileLang 同时支持 lazy style 和 eager style；lazy style 是“外层 JIT 函数返回一个已构造好的 PrimFunc”，eager style 则是“外层 JIT 函数本身被 AST mutator 改写并通过 Builder trace 成 PrimFunc”。**

这也是 `eager/ast.py` 和 `eager/builder.py` 的存在意义。

### lazy style：返回一个内部 `@T.prim_func`

lazy style 的用户代码通常像这样：

```python
# lazy style: 返回一个内部 @T.prim_func
@tilelang.jit
def make_kernel(M, N):
    @T.prim_func
    def kernel(A: T.Tensor((M, N), T.float32)):
        ...

    return kernel
```

这个风格里，外层 Python 函数更像一个 **kernel factory**。它接收编译期参数，比如 `M/N/block_M/block_N`，然后构造并返回一个内部 `PrimFunc`。

关键点：

- 当前主导出的 `@T.prim_func` 来自 `tilelang/language/eager/builder.py` 的 `prim_func(..., eager_jit=False)`，也会通过 `mutate(func)` 和 `Builder` 构造 PrimFunc。
- `tilelang/language/tir/entry.py` 和 `parser/` 里仍有 TVM script parser 兼容入口，但不是当前 `import tilelang.language as T` 下最后生效的 `T.prim_func` 主入口。
- 外层函数被调用后，直接返回 `PrimFunc`。
- `JITFunc` 会把这个 `PrimFunc` 缓存为一个 `TirTemplate.from_lazy_style(...)`。
- 这种风格适合“先生成 kernel object，再单独调用/查看/benchmark”的流程。

### eager style：直接在 JIT 函数体里写 DSL

eager style 的用户代码通常像这样：

```python
# eager style: 直接在 jit 函数体里写 DSL
@tilelang.jit
def kernel(A, B):
    M, N = T.const("M, N")

    A: T.Tensor((M, N), T.float32)
    B: T.Tensor((M, N), T.float32)
    C = T.empty((M, N), T.float32)

    with T.Kernel(...):
        ...

    return C
```

这个风格里，用户看起来像是在直接写一个 Python 函数，但它不是普通 Python 函数语义。它会被 `eager/ast.py` 改写，然后在 `Builder` 上下文中执行，从而构造 `PrimFunc`。

关键点：

- `T.const(...)` 用来声明 eager JIT 的动态 shape 变量。
- `A: T.Tensor(...)` 这样的 annotation 会被 AST mutator 捕获，并交给 `Builder.bind(...)` 处理。
- `C = T.empty(...)` 声明输出 tensor，不是立即分配 device memory。
- `with T.Kernel(...)` 构造 kernel launch frame。
- `return C` 告诉 JIT wrapper 哪些 tensor 是输出。

### lazy/eager 的关键判定：`JITFunc._is_lazy_style`

核心判定发生在 `tilelang/language/eager/builder.py` 的 `JITFunc._is_lazy_style(...)`。

可以把逻辑简化成：

```text
如果函数体里包含内部 @T.prim_func
        -> lazy
否则尝试调用原始函数
        如果返回 PrimFunc
                -> lazy
        如果调用过程中因为没有 Builder 触发 JITNoBuilderError / EagerJITBuildError
                -> eager
        否则
                -> eager
```

为什么 eager style 会触发 `JITNoBuilderError`？因为 eager style 的原始函数体里会直接调用 `T.const()`、`T.Kernel()`、`T.empty()` 这类必须依赖 `Builder.current()` 的 DSL API。第一次用于 mode inference 的普通调用没有 Builder，所以报错反而成了“这是 eager DSL，需要走 AST/Builder trace”的信号。

这点很精妙：**TileLang 用“原函数是否能直接返回 PrimFunc”来判断 lazy；如果原函数需要 Builder 才能运行，就切到 eager trace 路线。**

### AST 改写设计：不是执行 Python kernel，而是执行改写后的 Python 函数

eager 模式最容易误解。它的核心不是：

```text
执行用户写的 Python kernel
```

而是：

```text
先把用户 Python 函数 AST 改写成 Builder 调用，
再执行这个被改写后的 Python 函数，
执行过程中不断向 IRBuilder 追加 TIR 节点。
```

`eager/ast.py` 里的 `DSLMutator` 会把普通 Python 语句改成 Builder API。

典型改写如下：

| 用户 Python 语法 | 改写后的核心形式 | 作用 |
| --- | --- | --- |
| `if cond: ... else: ...` | `__tb.ctx_if / ctx_then / ctx_else` | 普通 bool 走 Python 分支，`PrimExpr` 生成 TIR `If` |
| `for i in T.serial(...)` | `for tmp in __tb.ctx_for(...): i = __tb.bind(...)` | 进入 TIR ForFrame 并绑定 loop var |
| `with T.Kernel(...)` | `with __tb.ctx_with(T.Kernel(...))` | 进入 KernelLaunchFrame |
| `T.copy(A, B)` | `__tb.eval(T.copy(A, B))` | 把 `tl.tileop.copy` call 追加到 IR |
| `x: T.int32 = expr` | `__tb.bind("x", expr, annot)` | 处理 typed let / scalar bind |
| `A: T.Tensor(...)` | `__tb.bind("A", value, tensor_annot)` | 处理 buffer 参数 annotation |
| 变量读取 `x` | `__tb.rval("x", x)` | 给 Builder 机会处理特殊变量、let、macro、OutTensor |
| `return C` | `__tb.ret(C)` | eager 输出 tensor 收集 |

所以 eager 模式下，“执行 Python”只是构建 IR 的手段。真正有意义的是执行期间产生的 `PrimFunc`。

### Builder 是 AST 改写后的落地点

`DSLMutator` 只负责把语法改成 `__tb.xxx(...)`，真正构造 IR 的是 `Builder`：

```text
DSLMutator
    Python AST -> __tb.ctx_for / __tb.eval / __tb.bind / __tb.ctx_with

Builder
    __tb.ctx_for   -> tirx ForFrame
    __tb.ctx_if    -> tirx If/Then/Else frame
    __tb.ctx_with  -> KernelLaunchFrame / WarpSpecializeFrame 等 context
    __tb.eval      -> evaluate / buffer_store / enter frame
    __tb.bind      -> Var / Buffer / Let / Tensor annotation / OutTensor

IRBuilder
    记录最终 TIR/TIRX AST
```

这一层关系可以用一句话概括：

> `ast.py` 负责“把 Python 语法换成可拦截的 Builder 调用”，`builder.py` 负责“把这些 Builder 调用落成真正的 TIR/TIRX 节点”。

### 两阶段 eager JIT：`phase1` 与 `phase2`

eager JIT 还有一个非常关键的两阶段设计，主要服务动态 shape 和 `T.const(...)`。

```text
phase1: 构造模板 PrimFunc
    - T.const("M, N") 创建 constexpr Var
    - A: T.Tensor((M, N), dtype) 把 constexpr Var 放进 buffer shape/stride
    - TirTemplate.create(...) 扫描 buffer_map，建立 constexpr -> tensor shape/stride 的 matcher

phase2: 根据真实 tensor 参数补全 constexpr
    - 从实际输入 tensor 的 shape/stride 或显式 kwargs 中解析 M/N/K
    - builder.eager_jit_subs = {"M": real_M, "N": real_N, ...}
    - 重新执行 IRGenerator，得到具体 shape 的 PrimFunc
```

这解释了为什么 eager style 可以写：

```python
M, N = T.const("M, N")
A: T.Tensor((M, N), T.float32)
```

而真正调用时又能从实际 tensor shape 推导出 `M/N`。

### 这段机制为什么是 `language/` 的精髓

因为整个 `language/` 的大多数 API 都依赖这条链：

```text
用户 Python 写法
    -> AST mutator 改写成 Builder 调用
    -> Builder 在 IRBuilder 中创建 TIR/TIRX 节点
    -> 高层 DSL API 生成 tl.tileop.* 或 frame/annotation
    -> lowering pipeline 消费这些高层语义
```

如果没有 AST 改写和 Builder，`T.Kernel`、`T.copy`、`T.gemm` 只是普通 Python 函数；有了这层机制，它们才变成 TileLang embedded DSL。

---

## 3. 从用户代码到 CUDA/HIP/CPU 的整体数据流

以 [examples/quickstart.py](examples/quickstart.py) 中的 GEMM 风格为例，用户写的是：

```python
import tilelang
import tilelang.language as T

@tilelang.jit
def matmul(A, B, block_M: int, block_N: int, block_K: int):
    M, N, K = T.const("M, N, K")
    dtype = T.float16
    accum_dtype = T.float32

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)

        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            T.copy(B[ko * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        for i, j in T.Parallel(block_M, block_N):
            C_local[i, j] = T.max(C_local[i, j], 0)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C
```

这段代码经过的核心链路是：

```text
@tilelang.jit
  -> tilelang.jit.jit(...)
  -> tilelang.language.eager.builder.prim_func(..., eager_jit=True)
  -> JITFunc 包装原始 Python 函数
  -> JITImpl 负责 lazy/eager 判断、cache key、compile/call

JITFunc.get_tir(...)
  -> eager 模式下使用 mutate(func) 生成 IRGenerator
  -> Builder.prim_func(...) 创建 IRBuilder 上下文
  -> 执行改写后的 Python 函数体
  -> 每个 T.xxx 调用向 IRBuilder 追加 TIR/TIRX 节点
  -> Builder.get() 得到 PrimFunc

PrimFunc 中保留：
  -> KernelLaunchFrame: blockIdx/threadIdx launch 语义
  -> sblock_alloc_buffer: shared/local/fragment/barrier/tmem allocation
  -> ForFrame: serial/parallel/pipelined loops
  -> tl.tileop.copy / tl.tileop.gemm / tl.tileop.reduce 高层 call_intrin

tilelang.lower(...)
  -> PreLowerSemanticCheck
  -> resolve_pipeline(target)
  -> CUDA/HIP/CPU/Metal backend pipeline
  -> LayoutInference
  -> LowerTileOp
  -> 后续 buffer flatten、vectorize、storage rewrite、intrinsic lowering、host/device split、codegen
```

当前分支中，pipeline 不再集中在旧文档里常见的 `tilelang/engine/phase.py`，而是按 backend 拆开：

| Backend | Pipeline 文件 |
| --- | --- |
| CUDA | `tilelang/cuda/pipeline.py` |
| ROCm/HIP | `tilelang/rocm/pipeline.py` |
| CPU | `tilelang/cpu/pipeline.py` |
| Metal | `tilelang/metal/pipeline.py` |
| Pipeline registry | `tilelang/backend/pass_pipeline/pipeline.py` |

`tilelang/engine/lower.py` 会通过 `resolve_pipeline(target)` 选择对应 backend pipeline。

`resolve_pipeline(target)` 的实现很简单：按 `target.kind.name` 从 registry 中取 `PassPipeline`。因此 pipeline 名字要和 TVM target kind 对齐，比如 `cuda`、`hip`、`metal`、`c`、`llvm`。

---

## 4. 推荐阅读顺序

这不是按文件大小，而是按“理解核心机制”的顺序：

```text
1. __init__.py
   看清 T 命名空间导出了哪些 API

2. examples/quickstart.py
   用一个真实 kernel 建立整体感觉

3. kernel.py
   理解 T.Kernel、KernelLaunchFrame、block/thread binding

4. eager/ast.py + eager/builder.py
   理解 Python DSL 如何被 AST 改写并构造成 PrimFunc

5. proxy.py + allocate.py
   理解 Tensor/Buffer proxy 与 shared/local/fragment scope

6. loop.py
   理解 Parallel、Pipelined、Persistent 的语义与 metadata

7. copy_op.py
   理解 T.copy 如何规范化 region 并生成 tl.tileop.copy

8. gemm_op.py
   理解 T.gemm 如何检查 M/N/K、stride/offset 并生成 tl.tileop.gemm

9. reduce_op.py
   理解 reduction 如何通过 macro 和 fragment 中转隐藏底层细节

10. frame.py
    理解 let value / BufferRegion alias 追踪

11. annotations.py / warpgroup.py / builtin.py / cluster.py / pdl.py
    理解高级 hint、warp specialization 和底层 intrinsic 包装

12. tir/ parser/ overrides/
    理解 TVM script parser 兼容层和 TileLang 特定补丁

13. tilelang/cuda/pipeline.py + src/transform/lower_tile_op.cc
    跳出 language，看高层 tile op 如何真正兑现
```

---

## 5. `__init__.py`：`T` 命名空间的总菜单

文件：`tilelang/language/__init__.py`

这个文件本身逻辑不多，主要做 re-export，但它非常重要，因为它定义了：

```python
import tilelang.language as T
```

之后用户能看到的 API 清单。

可以把它当索引读：

| 来源 | 暴露能力 |
| --- | --- |
| `tvm.tirx.script.parser.*` | 基础 TIR script parser 能力和 TIR 类型/表达式；部分名称会被后续 `.eager` 导入覆盖 |
| `.eager` | eager JIT 的 `prim_func`、`macro`、`const` 等 |
| `.tir.ir` | TIR 基础表达式、数学函数、loop wrappers |
| `.proxy` | `Tensor`、`StridedTensor`、`Buffer`、`ptr`、`make_tensor`、`make_tensor_from_addr`、`SharedBuffer`、`FragmentBuffer`、`LocalBuffer` |
| `.kernel` | `Kernel`、`CUDASourceCodeKernel`、`KernelLaunchFrame`、thread/block binding 查询 |
| `.allocate` | `alloc_shared`、`alloc_fragment`、`alloc_local`、`alloc_global`、`alloc_var`、`alloc_barrier`、`alloc_cluster_barrier`、`alloc_tmem`、`alloc_reducer`、descriptor allocation、`empty` |
| `.loop` | `Parallel`、`Pipelined`、`Persistent`、`serial`、`unroll`、`vectorized` 以及大写 alias |
| `.copy_op` | `copy`、`async_copy`、`tma_copy`、`tma_gather4/scatter4`、`copy_cluster`、`transpose`、`im2col` |
| `.gemm_op` | `gemm`、`wgmma_gemm`、`tcgen05_gemm`、`tcgen05_gemm_blockscaled`、`make_blockscaled_gemm_layout` |
| `.experimental.gemm_sp_op` | sparse GEMM 变体：`gemm_sp`、`wgmma_gemm_sp`、`tcgen05_gemm_sp` |
| `.reduce_op` | `reduce_*`、`finalize_reducer`、`warp_reduce_*` |
| `.fill_op` / `.scan_op` / `.print_op` | `fill/clear`、`cumsum/cummax`、device-side print/assert |
| `.customize` / `.atomic` | atomics、`dp4a`、`reshape`、`view`、`loop_break` |
| `.annotations` | layout/swizzle/L2/restrict/min blocks per SM hints |
| `.builtin` | GPU intrinsic、barrier、shuffle、WGMMA/TCGEN05 helper、load/store intrinsic |
| `.cluster` | cluster barrier、cluster copy/cancel 查询等 |
| `.pdl` | CUDA PDL trigger/sync |
| `.warpgroup` | `T.ws` warp-specialization scope |
| `.symbolics` / `.random` / `.utils` | dynamic/symbolic marker、random API、index 工具 |

此外，`language/__init__.py` 自己还定义了 `import_source(source)`，本质上是把 `pragma_import_c` 挂到当前 statement block，用于注入外部 C/CUDA 代码片段。

读这个文件时不要纠结每个 import 的实现，只要先建立一张分类地图。

---

## 6. `kernel.py`：`T.Kernel` 与 launch frame 机制

文件：`tilelang/language/kernel.py`

这是理解 TileLang DSL 的关键文件之一。

### 6.1 用户语法

用户写：

```python
with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
    ...
```

它的含义不是立即 launch kernel，而是向 TIR 构建一个 kernel launch 作用域。

### 6.2 `Kernel(...)` 做什么

`Kernel(...)` 大致做这些事：

1. 检查当前是否有 `Builder.current()`。
   - 没有 Builder，说明不在 `@tilelang.jit` 或 `@T.prim_func` 上下文里，抛 `JITNoBuilderError`。
2. 规范化 `threads`。
   - GPU kernel 不传 `threads` 时默认用 `128`。
   - `128` -> `[128, 1, 1]`
   - `(64, 2)` -> `[64, 2, 1]`
   - CPU kernel 下允许不传 thread binding。
3. 规范化 `cluster_dims`。
   - `2` -> `[2, 1, 1]`
   - `[1, 1, 1]` 会被视为无 cluster。
4. 收集 attrs。
   - `tilelang.is_cpu_kernel_frame`
   - `pragma_import_c`
   - `cluster_dims`
5. 调用 C++ FFI：

```python
return _ffi_api.KernelLaunch(blocks, threads, attrs)
```

这里的 `prelude` 不是 Python 运行时逻辑，而是以 `pragma_import_c` 形式挂到 kernel block annotation 上，后续 split/codegen 时会注入到生成代码里。

### 6.3 C++ 侧真正创建 frame

C++ 实现在 `src/ir.cc` 的 `KernelLaunch(...)`。

GPU kernel 情况下它会创建 launch-thread frames：

```text
grid_size[0] -> blockIdx.x / bx   # 如果 grid_size 至少 1 维
grid_size[1] -> blockIdx.y / by   # 如果 grid_size 至少 2 维
grid_size[2] -> blockIdx.z / bz   # 如果 grid_size 至少 3 维

block_size[0] -> threadIdx.x / tx
block_size[1] -> threadIdx.y / ty
block_size[2] -> threadIdx.z / tz

最后加一个 tilelang_root SBlockFrame，承载 kernel body 和 attrs
```

CPU kernel 情况下不创建 `threadIdx.*`，而是为每个 grid 维度创建普通 iter var frame，最后同样追加 `tilelang_root` block。这个差异会影响 `KernelLaunchFrame.__enter__` 返回的变量：GPU 返回 block binding，CPU 返回普通 loop var。

也就是说，`T.Kernel` 是 Python DSL 到 TIR launch-thread frame 的桥。

### 6.4 `KernelLaunchFrame.__enter__`

Python 侧注册对象：

```python
@register_object("tl.KernelLaunchFrame")
class KernelLaunchFrame(TIRFrame):
    def __enter__(self) -> Var | list[Var]:
        super().__enter__()
        _get_current_stack().push(self)
        ...
        return _normalize_bindings([...])
```

`__enter__` 做两件重要的事：

1. 调 `TIRFrame.__enter__()`，让 IRBuilder 进入 launch frame。
2. 把当前 frame 放入 thread-local stack，供 `get_thread_binding()`、`get_block_binding()` 查询。

GPU 情况下，`with T.Kernel(...) as (bx, by)` 返回的是 block binding，也就是 `blockIdx.x/y/z` 对应变量。

线程 binding 不通过 `as` 返回，而是用：

```python
tx, ty, tz = T.get_thread_bindings()
```

这个设计体现了 TileLang 的默认抽象层级：用户通常在 block/tile 级别写 kernel，线程映射更多通过 `T.Parallel`、layout inference、tile op lowering 处理。

CPU kernel 情况下，最后一个 `SBlockFrame` 的 annotation 里带有 `tilelang.is_cpu_kernel_frame`。`__enter__` 会排除最后的 `SBlockFrame`，返回前面普通 for frame 的 `vars[0]`；也就是说 CPU 路径没有 `threadIdx.x/y/z` 这三个 frame。

### 6.5 thread-local stack 为什么重要

`kernel.py` 里维护：

```python
_local = threading.local()
```

每个线程有自己的 `kernel_launch_frame_stack`。这样多线程编译不同 kernel 时不会互相污染。

`T.get_thread_binding(dim)` 实际就是：

```python
KernelLaunchFrame.Current().get_thread_binding(dim)
```

所以它不需要用户显式传入当前 kernel frame。

### 6.6 单维 unpack 兼容

`kernel.py` 还给 `Var` 补了 `__iter__` 和 `__len__`：

```python
if not hasattr(Var, "__iter__"):
    def _var_iter(self):
        yield self
    Var.__iter__ = _var_iter
```

这是为了兼容：

```python
with T.Kernel(n) as bx:
    ...

with T.Kernel(n) as (bx,):
    ...
```

单维 kernel 可以返回裸 `Var`，但又能被当成单元素 iterable unpack。

### 6.7 `CUDASourceCodeKernel`

`kernel.py` 还提供 `T.CUDASourceCodeKernel(...)`，用于在 TileLang kernel 里嵌入一段外部 CUDA source 或 source 文件路径。

它的主线和 `T.Kernel` 相似，但多了 source 处理：

1. 检查 `Builder.current()`，不在 Builder 上下文则抛 `JITNoBuilderError`。
2. `_load_cuda_source(...)` 判断参数是文件路径还是 inline CUDA source。
3. 校验 `entry_name`，默认入口名是 `main_kernel`。
4. 把 `code_block_source` 和 `code_block_entry_name` 写入 attrs。
5. 进入 `_ffi_api.KernelLaunch(...)` frame，并发射一个 `tirx.call_extern("int32", entry_name)`。

下游 `tilelang_callback_cuda_validate` 会检查外部 source 至少包含一个 `__global__` kernel，并要求 lowered device `global_symbol` 与 `entry_name` 匹配。这个 API 适合把已经写好的 CUDA kernel 作为 TileLang IR 的一个 device launch block 管起来，而不是让 Python 侧直接 launch CUDA。

---

## 7. `eager/ast.py`：Python AST 如何变成 Builder 调用

文件：`tilelang/language/eager/ast.py`

这是 eager DSL 的核心。它回答一个问题：

> 普通 Python 的 `if`、`for`、`with`、赋值、返回，为什么能变成 TIR？

答案是：**TileLang 先改写 Python AST，再执行改写后的函数。**

### 7.1 `mutate(func)`

`mutate(func)` 会：

1. 拿到函数源码 AST。
2. 收集 nonlocals 和 globals。
3. 用 `DSLMutator` 改写函数体。
4. 编译改写后的 AST。
5. 返回 `IRGenerator`。

`IRGenerator` 持有：

```python
@dataclass
class IRGenerator:
    gen: Callable[[BaseBuilder], Callable]
    source: str
    extra_type_hints: dict[str, Any]
```

`gen(builder)` 会返回一个可以执行的函数，执行时所有 DSL 语义都会走 `builder`。

当前实现还会收集 `extra_type_hints`。这主要服务 eager 函数参数：如果函数体里写了参数 annotation，例如 `A: T.Tensor(...)` 或 `A: T.float32`，mutator 会把这些类型信息保存下来，后续 `prim_func(..., eager_jit=True)` 用它识别哪些参数是 tensor/buffer 参数。

### 7.2 `if` 改写

用户写：

```python
if cond:
    body
else:
    other
```

会被改写成近似：

```python
for br in __tb.ctx_if(cond):
    for _ in __tb.ctx_then(br):
        body
    for _ in __tb.ctx_else(br):
        other
```

这里的 `__tb` 就是 Builder。

这样做的好处是：

- 如果 `cond` 是普通 Python bool，就按普通 Python 控制流执行。
- 如果 `cond` 是 TIR `PrimExpr`，`Builder.ctx_if` 会创建 `tirx.If/Then/Else` frame。

### 7.3 `for` 改写

用户写：

```python
for i, j in T.Parallel(M, N):
    body
```

会被改写成近似：

```python
for tmp in __tb.ctx_for(T.Parallel(M, N)):
    i, j = __tb.bind(...)
    body
```

`Builder.ctx_for` 会检查 loop object 是否是 `ForFrame`，进入对应 frame 后把 loop var yield 给用户变量。

### 7.4 普通表达式改写

用户写：

```python
T.copy(A, B)
```

AST 中是一个 expression statement，会被改写为：

```python
__tb.eval(T.copy(A, B))
```

这点非常关键：

- `T.copy(A, B)` 本身返回一个 `PrimExpr` / `Stmt` / intrinsic call。
- `Builder.eval(...)` 负责把它真正加入 IR。

### 7.5 变量读取改写

`visit_Name` 会把变量读取改成：

```python
__tb.rval("name", node)
```

这给 Builder 一个机会处理 eager JIT 中的特殊变量、macro 变量、let binding、OutTensor 等。

### 7.6 `with T.Kernel(...)` 改写

`visit_With` 会把 context expression 改成：

```python
__tb.ctx_with(T.Kernel(...))
```

如果识别到是 `T.Kernel` context，还会插入：

```python
if __tb.skip_kernel_ctx():
    return
```

用于 eager JIT 的阶段控制和参数推导。

### 7.7 其他语法改写

除了 `if/for/with/return` 主线，当前 `DSLMutator` 还处理了一批容易被忽略的 Python 语法：

| Python 语法 | 改写/处理方式 | 作用 |
| --- | --- | --- |
| `while cond:` | `for _ in __tb.ctx_while(lambda: cond)` | `PrimExpr` 条件生成 TIR `While`；常量真会被视为潜在无限循环 |
| `continue` / `break` | `__tb.ctx_continue()` / `__tb.ctx_break()` | 发射 `tirx.continue_loop()` / `tirx.break_loop()`，并标记后续语句无效 |
| `assert cond, msg` | `__tb.assert_expr(cond, msg)` | Python bool 直接检查，`PrimExpr` 生成 TIR `Assert` frame |
| `a += b` | `__tb.aug_assign(...)` 或 `__tb.aug_assign_slice(...)` | 支持 `local.var`、`Ref`、buffer slice 的更新语义 |
| `a and b` / `a or b` / `not a` | `__tb.boolop(...)` | Python bool 短路；`PrimExpr` 生成 TIR logical op |
| `x if cond else y` | `__tb.ifexp(cond, lambda: x, lambda: y)` | `PrimExpr` 条件生成 `tirx.if_then_else` |
| 语句 span | `__tb.set_fileline(...)` | 给 Builder 记录原始文件/行号，macro 展开时用于更可读的诊断 |

这说明 eager AST 改写不是只覆盖 “TileLang API 调用”，而是尽量把常见 Python 控制流和赋值语义映射到可构造 TIR 的 Builder hook。

---

## 8. `eager/builder.py`：真正的 IR 施工队

文件：`tilelang/language/eager/builder.py`

`Builder` 是 eager DSL 的核心执行对象。AST mutator 只是把 Python 语法改写成 Builder 调用，真正生成 TIR 的是 Builder。

### 8.1 Builder 的核心状态

```python
class Builder(BaseBuilder):
    def __init__(self):
        self.frames = []
        self.ir_builder = IRBuilder()
        self.name_inside_frame = {}
        self.out_idx = []
        self.out_tensor_cnt = 0
        self.constexpr_var = set()
        self.eager_jit = "none"  # phase1 / phase2 / none
        self.eager_jit_subs = {}
        self.func_pass_configs = None
        self.func_compile_flags = None
        self.current_file = "<unknown>"
        self.current_line = 0
        self.current_macro_name = "<unknown-macro>"
```

几个关键点：

- `frames`：追踪当前进入的 TIR frame，比如 PrimFuncFrame、ForFrame、IfFrame、KernelLaunchFrame。
- `ir_builder`：底层 TVM/TIRX IRBuilder。
- `out_idx/out_tensor_cnt`：追踪 eager style 里 `T.empty(...)` 生成的输出 tensor 是否都被 `return`。
- `constexpr_var`：eager JIT 动态 shape 推导用。
- `eager_jit`：分两阶段处理动态 shape。
- `func_pass_configs/func_compile_flags`：函数体内 annotation 最终会写到 PrimFunc attrs，再由 JIT compile 合并。
- `current_file/current_line/current_macro_name`：由 AST mutator 注入的 `set_fileline` 更新，主要用于更清晰的诊断和 macro 调用栈。

### 8.2 thread-local current Builder

```python
@classmethod
def current(cls):
    return getattr(thread_local_storage, "builder", None)
```

很多 DSL API 都依赖当前 Builder：

- `T.Kernel()`
- `T.const()`
- `T.make_tensor()`
- `T.annotate_compile_flags()`
- `T.macro()` 调用

这就是为什么这些 API 必须出现在 `@tilelang.jit` 或 `@T.prim_func` 上下文中。

### 8.3 `prim_func` context

```python
@contextmanager
def prim_func(self, name):
    thread_local_storage.builder = self
    clear_let_values()
    try:
        with self.ir_builder, self.with_frame(tirx.prim_func()):
            tirx.func_name(name)
            yield
    finally:
        clear_let_values()
        del thread_local_storage.builder
```

这个 context 的职责是：

1. 设置当前 thread-local Builder。
2. 进入 TVM IRBuilder。
3. 创建 `PrimFuncFrame`。
4. 执行用户改写后的函数体。
5. 退出后通过 `builder.get()` 取出 `PrimFunc`。

### 8.4 `ctx_if`

```python
def ctx_if(self, cond):
    cond = unwrap_cond(cond)
    if isinstance(cond, PrimExpr):
        with self.with_frame(tirx.If(cond)):
            yield self._has_if_frame
    else:
        yield cond
```

这个设计让同一套 Python `if` 能同时支持：

- 编译期 Python 分支
- 运行期 TIR 分支

### 8.5 `ctx_for`

`ctx_for` 处理三类 loop：

1. `T.serial/T.unroll` 带 step 的 TileLang wrapper。
2. TVM/TIRX 原生 `ForFrame`。
3. 非法对象直接报错。

带 step 的 serial/unroll 会先把 `start/stop/step` 转成真实 trip count：

```text
for i in T.serial(start, stop, step):
    body

实际变成：
for v in T.serial(ceildiv(stop - start, step)):
    i = start + v * step
```

当前实现会显式拒绝常量 `step == 0`；负 step 会按 `ceildiv(start - stop, -step)` 计算 trip count。非静态 step 可以走通，但 Builder 会 warning，因为动态 step 容易让 trip count 和边界行为不直观。

### 8.6 `eval`

`Builder.eval` 负责把 expression statement 落入 IR：

| 输入类型 | 行为 |
| --- | --- |
| `PrimExpr` | `tirx.evaluate(val)` |
| `IRBuilderFrame` | 进入 frame |
| `BufferStore` | 发射 `tirx.buffer_store(...)` |
| `int/bool` | 转成 const 后 evaluate |
| `Buffer/Var/None/str` | 通常忽略 |
| 其他 | warning：返回值未使用 |

这解释了为什么 `T.copy(...)` 一类函数可以返回 intrinsic call，再由 `eval` 统一追加到 IR。

### 8.7 `bind`

`bind` 是最复杂的支撑点之一，处理：

- 普通变量绑定
- 类型标注绑定
- `T.Tensor` 参数声明
- `T.empty` 输出 tensor
- let binding
- macro 局部变量
- eager shape constexpr

一个典型场景：

```python
A: T.Tensor((M, K), T.float16)
```

AST mutator 会把 annotation 交给 `Builder.bind`，Builder 根据 annotation 创建或匹配 TIR buffer。

### 8.8 `macro`

`@T.macro` 不是运行时函数调用，而是 **IR 生成时内联展开**。

```python
@T.macro
def foo(x):
    y = x + 1
    return y
```

调用 `foo(A[i])` 时，会在当前 Builder 上下文中展开 `foo` 的函数体，而不是生成一个 device function call。

`reduce_op.py` 的 `reduce_macro` 就是这种用法。

### 8.9 `return` 与 `T.empty`

eager style 的 `return` 不是任意返回 Python 对象。`Builder.ret(...)` 当前只允许返回由 `T.empty(...)` 声明出来的输出 buffer：

```python
C = T.empty((M, N), dtype=dtype)
return C
```

如果有多个 `T.empty(...)`，它们都必须被返回；否则 `Builder.prim_func(...)` 退出时会报 `Not all tensor allocated from T.empty are returned`。这是因为 eager JIT 需要把输出 tensor 映射成 PrimFunc 的输出 buffer，并把 `tilelang_out_idx` 写入函数 attrs，供外层 `tilelang.jit.compile(...)` 推导返回值。

`T.annotate_pass_configs(...)` 和 `T.annotate_compile_flags(...)` 也走类似路线：Builder 先记录到 `func_pass_configs/func_compile_flags`，构造完成后 `_patch_prim_func_attrs(...)` 写成 `tilelang_pass_configs` / `tilelang_compile_flags`。外层 JIT compile 会读取这些 attrs，并和调用方传入的 `pass_configs/compile_flags` 合并。

### 8.10 lazy/eager JIT 判断

`JITFunc` 负责同时支持两种风格。

lazy style：

```python
@tilelang.jit
def kernel_factory(M, N):
    @T.prim_func
    def kernel(A: T.Tensor((M, N), T.float32)):
        ...
    return kernel
```

eager style：

```python
@tilelang.jit
def kernel(A):
    M = T.const("M")
    A: T.Tensor((M,), T.float32)
    B = T.empty((M,), T.float32)
    with T.Kernel(...):
        ...
    return B
```

判断逻辑：

- 如果函数内部含 `@T.prim_func`，视为 lazy。
- 否则尝试调用原函数，如果返回 `PrimFunc`，视为 lazy。
- 如果调用过程中因为没有 Builder 触发 `JITNoBuilderError`，说明它是 eager-style，需要通过 AST/Builder trace。
- eager style 的 phase1 template 会按非 tensor 编译期参数形成 `p1_key` 缓存；phase2 再从真实 tensor 参数的 shape/stride 或显式 kwargs 形成 `p2_key`，重新执行 `IRGenerator` 生成具体 PrimFunc。

---

## 9. `proxy.py`：Tensor、Buffer、ptr 类型代理

文件：`tilelang/language/proxy.py`

`proxy.py` 提供用户在函数签名和 eager annotation 中使用的类型工厂。

### 9.1 `T.Tensor`

```python
A: T.Tensor((M, K), T.float16)
```

本质是创建一个 TIRX `Buffer`，默认 scope 为 `global`，并自动构造 contiguous strides。

`TensorProxy.__call__` 会把：

```python
T.Tensor((M, K), dtype)
```

转换成类似：

```python
buffer(
    shape=(M, K),
    dtype=dtype,
    strides=(K, 1),
    scope="global",
)
```

如果 shape 是单个 `int` 或 `PrimExpr`，`TensorProxy` 会自动转成一维 shape。`T.StridedTensor(shape, strides, dtype)` 则要求 `len(shape) == len(strides)`，用于显式 ABI stride；这类 stride 也会参与 eager JIT 的 constexpr matcher。

### 9.2 不同 Buffer proxy 的默认 scope

| Proxy | 默认 scope | 用途 |
| --- | --- | --- |
| `T.Tensor` | `global` | 函数参数，全局内存 tensor |
| `T.StridedTensor` | `global` | 显式 stride 的 tensor |
| `T.SharedBuffer` | `shared.dyn` | shared memory buffer annotation |
| `T.FragmentBuffer` | `local.fragment` | fragment/register tile buffer |
| `T.LocalBuffer` | `local` | local memory buffer |

这些 proxy 只是前端工厂，真正决定后端行为的是 TIR Buffer 上携带的 `scope()`。

### 9.3 `T.ptr`

`T.ptr(dtype, storage_scope="global")` 创建 handle 类型 `Var`。

它常用于 pointer table 或手动从地址构造 tensor。

特殊点：

```python
T.Tensor(..., T.ptr)
```

在存储层会被规范化为 `int64`，然后通过 `T.make_tensor(...)` 或 `T.make_tensor_from_addr(...)` 把 loaded address 重新解释为 typed pointer。

当前 `make_tensor` 的分支是：

- 如果传入的是 `Var`，走 `Tensor.from_ptr(...)`，用 `match_buffer` 把 pointer var 绑定成 buffer。
- 如果传入的是地址表达式，会先 `reinterpret("handle", addr)`，再 `bind(...)` 一个带 `PointerType(dtype, storage_scope)` 的指针变量。
- `make_tensor_from_addr(...)` 明确要求在 `Builder.current()` 上下文中使用，因为它要发 TIR bind/buffer 节点。

---

## 10. `allocate.py`：内存模型与 scope 字符串

文件：`tilelang/language/allocate.py`

TileLang 的内存层级主要通过 TIR Buffer 的 scope 字符串表达。

### 10.1 基础 allocation

```python
def alloc_shared(shape, dtype, scope="shared.dyn"):
    return T.sblock_alloc_buffer(shape, dtype, scope=scope)

def alloc_local(shape, dtype, scope="local"):
    return T.sblock_alloc_buffer(shape, dtype, scope=scope)

def alloc_fragment(shape, dtype, scope="local.fragment"):
    return T.sblock_alloc_buffer(shape, dtype, scope=scope)
```

三者主要区别就是 scope。

当前 `alloc_shared` 对 `dtype == "bool"` 有一个特例：scope 会从默认 `shared.dyn` 改成 `shared`。原因是 shared memory merge pass 当前不能很好地 merge bool 类型 shared buffer，使用静态 shared scope 更稳。

`alloc_global(shape, dtype, scope="global")` 也在当前 `T` 命名空间里。它通过 backend API 直接分配全局 workspace，主要用于测试或特殊场景；普通框架集成更推荐在 Torch 等宿主框架侧分配 workspace，再作为参数传入 kernel。

| Scope | 大致硬件含义 | 生命周期/可见性 | 典型用途 |
| --- | --- | --- | --- |
| `global` | device global memory | kernel 外部传入或全局 workspace | 输入输出 tensor |
| `shared.dyn` | dynamic shared memory | CTA 内共享 | global -> shared staging |
| `shared` | static shared memory | CTA 内共享 | bool 等特殊 shared buffer，barrier metadata |
| `local` | local memory / per-thread storage | thread 私有 | 临时 local buffer |
| `local.fragment` | register fragment | thread 私有，layout 描述 tile 分布 | MMA accumulator、ldmatrix 结果 |
| `local.var` | 单元素 scalar buffer | thread 私有 | 可写 scalar 变量 |
| `shared.barrier` | Hopper mbarrier | CTA/cluster sync | TMA / pipeline barrier |
| `shared.cluster_barrier` | cluster barrier | cluster CTA sync | cluster launch 同步 |
| `shared.tmem` | Blackwell TMEM | CTA 共享 Tensor Memory | TCGEN05 accumulator / operand |
| `local.descriptor.*` | descriptor register/local object | thread 私有 | WGMMA/TCGEN05 descriptor |

### 10.2 `alloc_var`

`alloc_var` 比较复杂，因为它兼容多种历史调用方式：

```python
a = T.alloc_var("int32", 1)
a = T.alloc_var("int32", "local.var")
a = T.alloc_var("int32", 1, "local.var")
a = T.alloc_var("int32", init=1)
a = T.alloc_var("int32", "local.var", init=1)
```

最终它会创建 shape 为 `[1]` 的 `local.var` buffer。

如果传了 initializer，当前实现会显式发射：

```python
T.buffer_store(buffer, parsed_init, 0)
```

这是为了避免仅依赖 annotation 初始化时，某些 backend 可能丢 initializer。

### 10.3 barrier / descriptor / tmem

高级 allocation：

| API | Scope | 用途 |
| --- | --- | --- |
| `alloc_barrier` | `shared.barrier` | Hopper/Blackwell mbarrier |
| `alloc_cluster_barrier` | `shared.cluster_barrier` | cluster mbarrier |
| `alloc_tmem` | `shared.tmem` | Blackwell Tensor Memory |
| `alloc_reducer` | `local.fragment` + `reducer_info` attr | `T.Parallel` 内的 thread-private reducer |
| `alloc_wgmma_desc` | `local.descriptor.wgmma` | Hopper WGMMA descriptor |
| `alloc_tcgen05_smem_desc` | `local.descriptor.tcgen05_smem` | Blackwell smem descriptor |
| `alloc_tcgen05_instr_desc` | `local.descriptor.tcgen05_instr` | Blackwell instruction descriptor |

这些对象后续会被 specific lowering pass 或 intrinsic lowering 消费。

`alloc_barrier` 和 `alloc_cluster_barrier` 不只是分配 `uint64` buffer，还会通过 `sblock_attr({"barrier_init": {buffer.data: arrive_count_exprs}})` 把 arrive count 记录到 block attr 中。后续 `LowerSharedBarrier` 这类 pass 会消费这份初始化信息。

`alloc_reducer(shape, dtype, op, replication)` 会把 reducer buffer 放在 `local.fragment`，并写入 `reducer_info` metadata。`op` 当前支持 `sum/max/min`，`replication` 支持 `all/none`。它需要配合 `T.fill(...)` 初始化和 `T.finalize_reducer(...)` 使用。

### 10.4 `T.empty`

`empty` 是 eager-style JIT 的输出 tensor 声明：

```python
C = T.empty((M, N), dtype=dtype)
return C
```

它不会直接分配 device memory，而是创建一个 `OutTensor` 描述，用于 JIT wrapper 在运行时准备输出 tensor，并把它映射到 PrimFunc 的输出 buffer。

当前支持的调用形式包括：

```python
T.empty((M, N), dtype=dtype)
T.empty(M, N, dtype=dtype)
T.empty((M, N), "float16")
```

注意它只能用于 eager-style 输出声明；真正的约束在 `Builder.ret(...)`：由 `T.empty` 创建的输出必须全部被返回。

---

## 11. `loop.py`：循环抽象与编译策略

文件：`tilelang/language/loop.py`

TileLang 的 loop API 不只是语法糖，每种 loop 都携带不同的编译意图。

### 11.1 `T.Parallel`

用户写：

```python
for i, j in T.Parallel(M, N):
    C[i, j] = A[i, j] + B[i, j]
```

语义上表示一个 tile 内的并行 iteration space。它由 `_ffi_api.Parallel(extents, annotations)` 创建 TileLang 自己的 parallel loop frame。

关键参数：

| 参数 | 含义 |
| --- | --- |
| `coalesced_width` | memory coalescing/vectorization hint |
| `loop_layout` | 手动指定 `Fragment` layout，影响每个 thread/lane 拿哪些元素 |
| `prefer_async` | 提示 copy lowering 优先 async 路径 |
| `annotations` | 直接附加到 outermost parallel loop 的 metadata |

`T.Parallel` 的 layout 约束会在 `LayoutInference` 和 `ParallelLoopLayoutValidator` 中处理。

一个重要细节：对嵌套 parallel loop，layout annotation 应该放在最外层 parallel loop 上，因为外层才能描述整个 loop nest 的 iteration mapping。

当前 `T.Parallel(..., loop_layout=layout)` 会把 layout 作为 `"parallel_loop_layout"` annotation 挂到最外层 parallel loop。`LayoutInference` 期间的 `ParallelLoopLayoutValidator` 会检查：

- 嵌套 parallel loop 的 layout 必须覆盖整个 loop nest。
- layout 的 `InputDim` 必须等于 parallel nest 深度。
- inner parallel loop 不应该单独带 layout annotation。
- 如果用户不传 `loop_layout`，compiler 会尝试自动推导并补上合法 layout。

### 11.2 `T.Pipelined`

用户写：

```python
for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
    T.copy(A[...], A_shared)
    T.copy(B[...], B_shared)
    T.gemm(A_shared, B_shared, C_frag)
```

它声明一个 software pipeline。用户描述 producer/consumer 的逻辑顺序，后端 pipeline pass 会重排和插入同步。

在 CUDA pipeline 中，相关 pass 是：

```text
PipelinePlanning
InjectSoftwarePipeline
```

它们会分析：

- 哪些语句是 producer，比如 global -> shared copy。
- 哪些语句是 consumer，比如 GEMM。
- 是否需要 pipeline buffer versioning。
- 是否需要 mbarrier、wait、commit。
- 如何生成 stage 交错执行序列。

`order/stage/sync/group` 参数用于手动 pipeline scheduling。普通阅读可以先理解 `num_stages` 自动 pipeline。

当前源码里还强调一个手动调度细节：`order/stage` 应该对应真正可调度的 pipeline statements，例如 copy、fill、GEMM、reduction、store、wait/commit。由局部 alias 产生的可重放 scalar `Bind` 不应该占用 `order/stage` 条目；pipeline pass 会在需要时自动 replay 这些 bind。旧代码如果把这类 bind 算进 `order/stage`，pass 会尽量兼容并忽略它们。

### 11.3 `T.Persistent`

`Persistent(domain, wave_size, index, group_size)` 表示 persistent kernel / persistent threadblock 风格。

它不是普通 for-loop，而是给后续 `PersistThreadblock` 类 pass 提供语义，让一个 CTA 反复领取 tile work，减少 launch/调度开销。

### 11.4 `serial / unroll / vectorized`

这些是标准 TIR loop kind 的 wrapper。

`loop.py` 中的 `serial/unroll` 额外支持 `step`，Builder 会把带 step 的 loop 转换成 trip count loop，再把 loop var 映射回 `start + v * step`。

`T.Serial/T.Unroll/T.Vectorized` 是对应小写 API 的大写 alias。`unroll(..., unroll_factor=n)` 会把 factor 写成 `pragma_unroll_factor` annotation；`vectorized` 走 TVM/TIRX 原生 vectorized ForFrame。

---

## 12. `copy_op.py`：数据移动的高层 TileOp

文件：`tilelang/language/copy_op.py`

`T.copy` 是 TileLang 最核心的 DSL API 之一。它的重要性在于：**它保留数据移动的高层语义，让后端根据 scope、target、layout、annotation 选择最佳实现。**

### 12.1 输入形式

`T.copy` 接受：

- `tirx.Buffer`
- `tirx.BufferLoad`
- `tirx.BufferRegion`

例如：

```python
T.copy(A, B)
T.copy(A[i, j], B[i, j])
T.copy(A[by * BM, ko * BK], A_shared)
T.copy(C_frag, C[by * BM, bx * BN])
```

### 12.2 `_normalize_copy_regions`

核心预处理函数：

```python
_normalize_copy_regions(src, dst)
```

它做几件事：

1. 如果 src/dst 都是完整 Buffer，则检查 shape structural equal。
2. 从 src/dst 推导 extent。
3. 如果二者都是 scalar `BufferLoad`，直接走 scalar store 快路径。
4. 如果一侧没有 extent，则按另一侧做有限的 broadcast-like 补齐。
5. 用 `legalize_pairwise_extents` 对齐 src/dst extents。
6. 用 `to_buffer_region` 转成 `BufferRegion`。

### 12.3 scalar 快路径

如果用户写的是：

```python
T.copy(A[i], B[i])
```

并且两边都是 scalar `BufferLoad`，`T.copy` 不会生成 `tl.tileop.copy`，而是直接返回：

```python
tirx.BufferStore(dst.buffer, src, dst.indices)
```

这等价于普通赋值。

### 12.4 高层 copy op

一般 tile copy 会生成：

```python
tirx.call_intrin(
    "handle",
    tirx.op.Op.get("tl.tileop.copy"),
    src_region,
    dst_region,
    annotations=ann,
)
```

这个节点在 TIR 中暂时是高层 tile op，占位到 `LowerTileOp`。

### 12.5 annotations

| 参数 | 作用 |
| --- | --- |
| `coalesced_width` | 控制 global memory coalescing/vectorization hint |
| `disable_tma` | 禁用 TMA lowering |
| `eviction_policy` | L2 cache eviction hint：`evict_normal/first/last` |
| `prefer_instruction` | 指定偏好：`tma`、`cp_async`、`sync` 等 |
| `loop_layout` | 给 SIMT copy 生成的 parallel loop 附 layout hint |

当前实现里，显式传入的 `annotations` 字典优先级高于单独 keyword 参数。例如 `annotations={"prefer_instruction": "sync"}` 会覆盖 `prefer_instruction="tma"`。字符串形式的 `prefer_instruction` 会被转换成 `tirx.StringImm`。`loop_layout` 最终写入 annotation key `"parallel_loop_layout"`，供 SIMT copy 生成的 parallel loop 使用；它不适用于 TMA/LDSM/STSM/TMem 这类 lowering。

### 12.6 下游可能 lowering 到什么

具体取决于 target、scope、shape、layout、annotation：

| 源 -> 目的 | 可能 lowering |
| --- | --- |
| global -> shared | TMA load、cp.async、sync SIMT copy |
| shared -> global | TMA store、vectorized store、SIMT store |
| shared -> fragment | ldmatrix、vectorized load、SIMT load |
| fragment -> shared/global | stmatrix、vectorized store、SIMT store |
| scalar load -> scalar store | direct `BufferStore` |
| cluster shared copy | TMA multicast、SM-to-SM copy |

### 12.7 相关变体

| API | 语义 |
| --- | --- |
| `async_copy` | 显式 async copy，通常是 cp.async 语义，不自动插 wait |
| `tma_copy` | 用户管理 barrier 的 TMA producer/store 操作；load 不自动 wait，store 不自动 `tma_store_wait` |
| `tma_gather4/tma_scatter4` | Blackwell gather4/scatter4 TMA tile 操作，用 annotations 表达 rows/col/barrier |
| `copy_cluster` | cluster-aware copy，支持 TMA multicast / SM-to-SM |
| `transpose` | 带转置的数据搬运 |
| `im2col/c2d_im2col` | 卷积类数据重排 |

---

## 13. `gemm_op.py`：GEMM 高层 TileOp

文件：`tilelang/language/gemm_op.py`

`T.gemm` 也是高层 tile op。Python 侧主要做参数规范化和合法性检查，真正选择 MMA/WGMMA/TCGEN05 的地方在 lowering。

当前 API 可以分成两类：

- `T.gemm(...)` 是默认同步接口。Hopper WGMMA 或 Blackwell TCGEN05 lowering 被选中时，compiler 会负责插入对应 wait。
- `T.wgmma_gemm(...)` / `T.tcgen05_gemm(...)` / `T.tcgen05_gemm_blockscaled(...)` 是显式 async 接口。它们要求特定 ISA lowering，不能用时会失败，并且不会自动插入 `warpgroup_wait` 或 `mbarrier_wait_parity`。

### 13.1 统一入口 `_gemm_impl`

以下 API 都走 `_gemm_impl`：

- `gemm`
- `wgmma_gemm`
- `tcgen05_gemm`
- `tcgen05_gemm_blockscaled`
- sparse GEMM 变体在 experimental 下也有类似结构

### 13.2 参数 legalize

`_gemm_impl` 里有一个细节：

```python
def legalize_arguments(arg):
    if isinstance(arg, tirx.Var) and T.has_let_value(arg):
        return T.get_let_value(arg).buffer
    return arg
```

这用于处理 let-bound buffer alias。比如某些 macro 或 bind 让变量间接指向 BufferRegion，GEMM 需要还原成真实 buffer/region。

### 13.3 shape / stride / offset 检查

`_gemm_impl` 会把 A/B/C 统一转成 `BufferRegion`，然后提取：

- shape
- stride
- offset

核心维度关系：

```text
C shape = [M, N]

if not transpose_A:
    A shape last dims = [M, K]
else:
    A shape last dims = [K, M]

if not transpose_B:
    B shape last dims = [K, N]
else:
    B shape last dims = [N, K]
```

检查包括：

```python
assert M_A == M
assert K == K_B
assert N_B == N
```

2CTA 模式下，B 的 N 维可以是 C 的一半：

```python
assert N_B * 2 == N
```

### 13.4 为什么要检查 offset

代码要求：

```python
assert A_offset[-2] == 0
assert B_offset[-2] == 0
```

这说明当前 GEMM lowering 只支持某种规范化 region 形式：矩阵 tile 的 row 起点必须规整，真正可变的偏移主要放在最后一维。这样 C++ lowering 能更直接生成 descriptor / instruction 参数。

### 13.5 生成 `tl.tileop.gemm`

最终生成：

```python
tirx.call_intrin(
    "handle",
    tirx.op.Op.get(op_key),
    A_arg,
    B_arg,
    C_arg,
    transpose_A,
    transpose_B,
    M,
    N,
    K,
    policy,
    clear_accum,
    stride_a,
    stride_b,
    offset_a,
    offset_b,
    k_pack,
    wg_wait,
    mbar_arg,
    C_coords[0],
    C_coords[1],
    annotations=annotations,
)
```

`op_key` 决定语义：

| API | op key | 语义 |
| --- | --- | --- |
| `T.gemm` | `tl.tileop.gemm` | 默认同步 GEMM，高层接口 |
| `T.wgmma_gemm` | `tl.tileop.wgmma_gemm` | Hopper WGMMA explicit async，用户管理 wait |
| `T.tcgen05_gemm` | `tl.tileop.tcgen05_gemm` | Blackwell TCGEN05 explicit async，用户管理 mbarrier wait |
| `T.tcgen05_gemm_blockscaled` | `tl.tileop.gemm` + scale factor args/annotations | Blackwell block-scaled TCGEN05 explicit async |

`tcgen05_gemm(..., use_2cta=True)` 和 blockscaled 2CTA 模式会通过 annotation 请求 true `cta_group::2` lowering，并要求 kernel `cluster_dims` 是 `(2,1,1)` 或 `(1,2,1)`。blockscaled GEMM 还需要 SFA/SFB scale factor 已经在 TMEM 中，并要求显式传入 `mbar`。

### 13.6 `GemmWarpPolicy`

`GemmWarpPolicy` 会影响 warp 如何覆盖 tile。常见策略如 square、row/column 方向展开。它不是 Python 层执行逻辑，而是传给 tile op lowering，用于 instruction/layout 选择。

`make_blockscaled_gemm_layout(C, A, transpose_A=False)` 是 blockscaled 路径的辅助函数，用 A/C 的 shape 和 dtype 创建 C 的 TMEM store layout。用户需要把返回的 layout 通过 `T.annotate_layout({C_tmem: layout})` 挂到对应 buffer 上，否则后续从 TMEM copy/store 时 layout 信息不足。

---

## 14. `reduce_op.py`：Reduction 与 macro 展开

文件：`tilelang/language/reduce_op.py`

`T.reduce` 的实现很能体现 language 层的职责：它不只是发一个 intrinsic，还会根据 memory scope 自动插入中转逻辑。

当前 `reduce(buffer, out, reduce_type, dim, clear, batch=1, nan_propagate=False)` 还会先检查输出 shape：`out` 必须是去掉 `dim` 后的 shape，或者保留该维但 extent 为 1。`dim < 0` 会由各个 `reduce_*` wrapper 转换成正维度。

### 14.1 reduce 为什么要 fragment 中转

GPU 上许多 reduction 最终在 register/fragment 里做，依赖 warp shuffle 或 thread allreduce。shared memory 不能直接表达所有需要的 per-thread fragment layout。

所以 shared -> shared reduction 会被包装成：

```text
1. alloc_fragment(buffer.shape, buffer.dtype) -> red_frag_in
2. alloc_fragment(out.shape, out.dtype) -> red_frag_out
3. copy(shared_in, red_frag_in)
4. tl.tileop.reduce(red_frag_in, red_frag_out)
5. copy(red_frag_out, shared_out)
```

这隐藏了底层 register fragment 的复杂性。

### 14.2 `reduce` 是 macro

代码中：

```python
@macro
def reduce_macro(...):
    ...
```

调用 `reduce_macro(...)` 时不是生成函数调用，而是在当前 IRBuilder 上下文里展开这一段 IR 构造逻辑。

### 14.3 scope 分支

`reduce` 会按输入/输出 scope 分四种情况：

| 输入 | 输出 | 处理方式 |
| --- | --- | --- |
| shared | shared | 输入输出都经 fragment 中转 |
| shared | fragment | 输入 copy 到 fragment，再 reduce 到输出 fragment |
| fragment | shared | reduce 到临时 fragment，再 copy 到 shared |
| fragment | fragment | 直接 `tl.tileop.reduce` |

其他 scope 组合会报错。

`batch > 1` 时，Python 层会把 `"batch"` 写入 annotations，后端可以用 batched AllReduce 减少 barrier 数量。`nan_propagate=True` 只对 CUDA 上的 `max/min/absmax` 这类 float16/bfloat16 reduction 有意义，会要求 lowering 使用 NaN-propagating intrinsic。

### 14.4 `warp_reduce_*`

`warp_reduce_sum/max/min/bitand/bitor` 是更低层的 register value reduction：

```python
tirx.call_intrin(value.dtype, tirx.op.Op.get("tl.warp_reduce_sum"), value)
```

它们不走 fragment 中转，语义更接近 warp shuffle intrinsic。

当前 wrapper 还包括 `reduce_abssum`、`reduce_absmax`、`reduce_bitand`、`reduce_bitor`、`reduce_bitxor`。`finalize_reducer(reducer, batch=1)` 会发射 `tl.tileop.finalize_reducer`，通常和 `alloc_reducer` 配合使用，把 `T.Parallel` 内部累积的 per-thread partial results 做最终归并。

---

## 15. `frame.py`：Let 值和 BufferRegion alias 追踪

文件：`tilelang/language/frame.py`

`frame.py` 维护另一个 thread-local stack，用于追踪 let binding。

### 15.1 为什么需要追踪 let value

用户可能写：

```python
base: T.int32 = i * BK
```

或者 macro 中创建中间变量。TIR 里这可能成为一个 bind/let var。某些高层 op 后面还需要知道这个 var 背后的真实值，尤其是 BufferRegion alias。

例如 `gemm_op.py` 中会检查：

```python
if isinstance(arg, tirx.Var) and T.has_let_value(arg):
    return T.get_let_value(arg).buffer
```

### 15.2 `register_let_value`

旧路径中 `LetFrame.__enter__` 会自动更新 stack。现在部分 tirx 会发 flat Bind，所以提供：

```python
register_let_value(var, value)
```

显式记录 var -> value 映射。

### 15.3 BufferLoad 到 BufferRegion 的转换

`LetFrame.__enter__` 中有一个细节：如果 let value 是 `BufferLoad`，并且索引里有 vector lane，它会转换成 `BufferRegion`：

```python
BufferRegion(self.value.buffer, [Range(x.base, x.lanes) for x in indices])
```

这让后续 tile op 能把 vectorized/block load 当成 region 来处理。

---

## 16. `annotations.py`：编译 hint 注入

文件：`tilelang/language/annotations.py`

annotation API 通常不生成运行时代码，而是把 metadata 挂到 block/function/buffer 上，供后续 pass 或 codegen 使用。

常见 API：

| API | 作用 |
| --- | --- |
| `use_swizzle` | threadblock swizzle/rasterization hint，提高 L2 locality |
| `annotate_layout` | 为 buffer 手动指定 `Layout/Fragment`，减少或覆盖自动 layout inference |
| `annotate_safe_value` | 给越界或 guarded access 指定安全值 |
| `annotate_l2_hit_ratio` | 给 global buffer 的 L2 cache behavior 提供 hint |
| `annotate_restrict_buffers` | 控制 buffer alias/restrict 标记 |
| `annotate_min_blocks_per_sm` | 影响 launch bounds / register pressure |

阅读时要抓住：annotation 的价值不在 Python 层，而在后续 lowering/codegen。

---

## 17. `warpgroup.py`：Warp Specialization scope

文件：`tilelang/language/warpgroup.py`

`T.ws(...)` 用于表达 warp-group specialization，常见于 Hopper TMA + WGMMA producer/consumer 模式。

用户可能写：

```python
with T.ws(0):
    T.tma_copy(...)

with T.ws(1):
    T.wgmma_gemm(...)
```

核心设计：

1. 通过 `get_thread_bindings()` 拿当前 `threadIdx.x/y/z`。
2. 计算 flatten thread id。
3. 根据 warp group id 创建条件 scope。
4. 生成 `_ffi_api.WarpSpecialize(...)` frame。

后续 CUDA pipeline 中的 `ProducerConsumerWarpSpecialized` pass 会进一步把这种高层 producer/consumer 结构改写成更贴近 Hopper 硬件的 named barrier、proxy fence、TMA/WGMMA 协作模式。

---

## 18. `builtin.py` 与底层 intrinsic 包装

文件：`tilelang/language/builtin.py`

这个文件很大，但阅读优先级应该靠后。它主要做底层 intrinsic 的 Python wrapper。

大类包括：

| 类别 | 示例 |
| --- | --- |
| load/store intrinsic | `__ldg`、`ldg32/64/128/256`、`stg32/64/128/256` |
| barrier/sync | `sync_threads`、`sync_warp`、`mbarrier_arrive`、`mbarrier_wait_parity` |
| warp shuffle/vote | `shfl_sync`、`shfl_xor`、`ballot_sync`、`any_sync`、`all_sync` |
| WGMMA helpers | `warpgroup_arrive`、`warpgroup_commit_batch`、`warpgroup_wait` |
| descriptor helpers | `initialize_wgmma_descriptor`、`increase_descriptor_offset` |
| TCGEN05 helpers | `tcgen05_mma_arrive`、`tcgen05_cp_warpx4`、descriptor init |
| PTX/HIP low-level ops | `ptx_mma_sm70`、`ds_read_tr16_b64` 等 |

多数函数最终都是：

```python
tirx.call_intrin(...)
tirx.call_extern(...)
evaluate(...)
```

读懂主线前先别从这里开始，否则容易被大量硬件指令名淹没。

---

## 19. `tir/`、`parser/`、`overrides/`：TVM Script 兼容层

### 19.1 `tir/`

目录：`tilelang/language/tir`

这里提供更接近 TVM/TIR script 的入口和基础 IR wrapper。

`tir/entry.py` 中的 `prim_func` 基本走 TVM parser：

```python
parse(func, utils.inspect_function_capture(func), check_well_formed=...)
```

注意当前 `tilelang/language/__init__.py` 中 `T.prim_func` 最后会被 `.eager` 导入覆盖，所以主路径是 `eager/builder.py` 的 `prim_func`。这里的 `tir/entry.py` 更适合作为 TVM script parser 兼容入口来读，而不是当前 eager/lazy JIT 主线的入口。

`tir/ir.py` 重新导出或包了一批 TIR 表达式/loop 工具，如：

- `serial`
- `parallel`
- `vectorized`
- `unroll`
- `thread_binding`
- `ceildiv`
- `max_value/min_value`
- 数学和 bitwise op

注意：`tir/ir.py` 的 `serial` 更接近原生 TIR loop wrapper；`loop.py` 的 `serial` 是 TileLang 扩展入口，支持 step 和额外 annotation 处理。

### 19.2 `parser/`

目录：`tilelang/language/parser`

这是 TileLang 早期/兼容 TVM script parser 的一层，包含：

- `entry.py`：`prim_func`、`macro`、`BufferProxy`、`PtrProxy`
- `parser.py`：对 TVM script parser visit 方法的注册/扩展
- `operation.py`：parser 操作支持

当前 `language/__init__.py` 中已经直接导入 `tvm.tirx.script.parser`，并有注释说明希望未来完全兼容 upstream；同时 `.eager` 导入覆盖了主命名空间里的 `prim_func/macro/const` 等 eager 入口。所以 `parser/` 不是初学主线，更多是历史兼容和 TVM script 扩展背景。

### 19.3 `overrides/`

目录：`tilelang/language/overrides`

这里是对 upstream parser/buffer 行为的补丁。

例如 `overrides/buffer.py` 会 patch `tirx.Buffer.__getitem__`，当用户索引维度不匹配时给更友好的错误信息。

`overrides/parser.py` 覆盖 `Assign/AugAssign/AnnAssign`，支持：

- chained writes
- 写入 `local.var` buffer
- 更符合 TileLang DSL 的 assignment 行为

---

## 20. 下游边界：`language/` 生成的 IR 在哪里被兑现

`language/` 生成的是高层 TIR/TIRX，不是最终代码。要理解核心运作原理，必须知道它和下游的边界。

### 20.1 `tilelang/engine/lower.py`

`lower_to_host_device_ir(...)` 做：

1. 如果输入是 `PrimFunc`，包成 `IRModule`。
2. 解析 target 和 target_host。
3. 跑 `PreLowerSemanticCheck`。
4. 根据 target 选择 backend pipeline：

```python
pipeline = resolve_pipeline(target)
mod = pipeline.lower(mod, target)
```

5. split host/device IRModule。

### 20.2 CUDA pipeline 的核心片段

文件：`tilelang/cuda/pipeline.py`

核心顺序大致是：

```text
BindTarget
LetInline / wrapper / negative index legalization
VerifyParallelLoop
InjectAssumes
Simplify
LayoutReducer
ProducerConsumerWarpSpecialized
LowerBlackwell2SM
IfStmtBinding
PipelinePlanning
InjectSoftwarePipeline
Simplify
LayoutInference
LowerTileOp
LowerL2Persistent
DecoupleTypeCast
LegalizeVectorizedLoop
LegalizeSafeMemoryAccess
LowerAccessPtr
Simplify
HoistNonRestrictParams
LowerSharedTmem
PlanAndUpdateBufferAllocationLocation
LowerSharedBarrier
FuseMBarrierArriveExpectTx
HoistGlobalBufferAllocations
LowerOpaqueBlock
FlattenBuffer
ConfigIndexBitwidth
VectorizeLoop
StorageRewrite
LoopUnswitching
UnrollLoop
VerifyMemory
InferFragment
LowerThreadAllreduce
LowerLDGSTG
LowerHopperIntrin
...
```

阅读时最关键的是：

```text
LayoutInference
LowerTileOp
```

`LayoutInference` 推导 fragment/shared/parallel loop layout。

`LowerTileOp` 把 `tl.tileop.copy/gemm/reduce` 等高层 op 降成低层 TIR/intrinsic。

当前 CUDA pipeline 的重要分界是：

- `ProducerConsumerWarpSpecialized`、`LowerBlackwell2SM`、`PipelinePlanning`、`InjectSoftwarePipeline` 都在 `LayoutInference` 前运行，让 layout inference 看到较最终的高层结构。
- `LowerTileOp` 之后才进入更偏存储/代码形态的 pass，例如 TMEM/barrier lowering、allocation placement、buffer flatten、vectorize、storage rewrite、host/device split。
- `LowerTileOp` 会设置类似 `tl.has_tma` 的函数 attr，后续 pipeline 会据此决定是否运行 `FuseMBarrierArriveExpectTx` 等 TMA 相关处理。

### 20.3 C++ `LowerTileOp`

入口：`src/transform/lower_tile_op.cc`

大致职责：

1. 遍历 `PrimFunc` body。
2. 找到 `tl.tileop.*` call。
3. 根据 target、buffer scope、layout、annotation 构造低层 IR。
4. 记录是否使用 TMA 等信息，比如 `tl.has_tma` function attr。
5. 注入自动分配的 mbarrier、barrier init metadata 等。

所以 Python 的 `T.copy/T.gemm/T.reduce` 只负责留下足够 rich 的高层语义，真正硬件相关选择在 C++/backend pass 中发生。

---

## 21. 一个 DSL 片段如何落成 IR：逐行解释

用户代码：

```python
@T.prim_func
def matmul(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
        A_s = T.alloc_shared((BM, BK), "float16")
        B_s = T.alloc_shared((BK, BN), "float16")
        C_f = T.alloc_fragment((BM, BN), "float32")

        T.clear(C_f)

        for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
            T.copy(A[by * BM, ko * BK], A_s)
            T.copy(B[ko * BK, bx * BN], B_s)
            T.gemm(A_s, B_s, C_f)

        T.copy(C_f, C[by * BM, bx * BN])
```

逐行理解：

| DSL | language 层产物 | 后续 pass 关注点 |
| --- | --- | --- |
| `@T.prim_func` | 生成 `PrimFunc` | function attrs、buffer_map |
| `A: T.Tensor(...)` | global scope TIR Buffer | 参数 ABI、shape/stride |
| `T.Kernel(...)` | `KernelLaunchFrame`，block/thread launch frame，`tilelang_root` block | device launch lowering、host/device split |
| `alloc_shared` | `sblock_alloc_buffer(..., scope="shared.dyn")` | shared memory planning/merge/rewrite |
| `alloc_fragment` | `sblock_alloc_buffer(..., scope="local.fragment")` | layout inference、fragment lowering |
| `T.Pipelined(...)` | 带 pipeline metadata 的 loop frame | pipeline planning / software pipeline injection |
| `T.copy(A..., A_s)` | `tl.tileop.copy` | TMA/cp.async/SIMT/ldmatrix lowering |
| `T.gemm(A_s, B_s, C_f)` | `tl.tileop.gemm` | MMA/WGMMA/TCGEN05 lowering |
| `T.copy(C_f, C...)` | `tl.tileop.copy` | store lowering、safe access、vectorization |

函数体执行完后，得到的是一个 `PrimFunc`。这时还没有最终 CUDA C++ 源码，只有 TIR/TIRX AST。

---

## 22. 常见误区

### 22.1 `T.copy` 是否真的复制数据？

不是。它生成 IR 节点。真正拷贝发生在最终编译出的 device kernel 运行时。

### 22.2 DSL 是 lazy 的吗？

需要分层理解：

- Python 函数在 JIT/Builder 上下文中会被执行。
- `T.copy/T.gemm` 调用当场生成并追加 IR 节点。
- 但高层 tile op 的具体硬件实现是 deferred 到 lowering pass。

所以：**DSL 构建 IR 是 eager 的；硬件指令选择是 deferred 的。**

### 22.3 `T.Kernel` 返回的是 thread id 吗？

不是。`with T.Kernel(...) as bx` 返回 block binding。thread binding 用：

```python
tx = T.get_thread_binding(0)
tx, ty, tz = T.get_thread_bindings()
```

### 22.4 `local.fragment` 是普通 local memory 吗？

不是普通 local memory。它表示一个 fragment tile，通常映射到 per-thread register fragment，需要 layout 描述每个 thread/lane 持有哪些元素。

### 22.5 scope 字符串只是注释吗？

不是。scope 是后端行为的关键开关。`shared.dyn`、`local.fragment`、`local.var`、`shared.barrier` 会影响 memory planning、layout inference、tile op lowering 和 codegen。

### 22.6 `T.Pipelined(num_stages=3)` 是否只是 unroll？

不是。它是 software pipeline 声明。后续 pass 会分析 producer/consumer、插入 barrier/wait/commit、做 buffer versioning 和 stage 交错。

---

## 23. 建议的源码实验

为了真正掌握核心机制，建议做几个小实验。

### 23.1 打印 PrimFunc script

写一个最小 kernel，然后：

```python
kernel = matmul.get_tir(...)
print(kernel.script())
```

观察里面是否出现：

- `thread_binding`
- `blockIdx.x/y`
- `T.alloc_buffer(..., scope="shared.dyn")`
- `tl.tileop.copy`
- `tl.tileop.gemm`

### 23.2 对比 scalar copy 和 tile copy

写：

```python
T.copy(A[i], B[i])
```

和：

```python
T.copy(A, B)
```

观察前者是否变成 `BufferStore`，后者是否保留 `tl.tileop.copy`。

### 23.3 改 scope 看 lowering 行为

把：

```python
A_s = T.alloc_shared(...)
C_f = T.alloc_fragment(...)
```

改成不同 scope，观察 `T.copy/T.gemm` 是否报错或生成不同 IR。

### 23.4 改 `prefer_instruction`

尝试：

```python
T.copy(A_tile, A_s, prefer_instruction="sync")
T.copy(A_tile, A_s, prefer_instruction="cp_async")
T.copy(A_tile, A_s, prefer_instruction="tma")
```

看 `LowerTileOp` 和生成 kernel source 的变化。

### 23.5 看 debug 输出

`@tilelang.jit(debug_root_path="...")` 会把 TIR script 和 kernel source 写出。对照阅读 `language/` 和 pipeline 非常有帮助。

---

## 24. 阅读时抓住的 6 个关键问题

读任何一个 `language/` 文件时，问这 6 个问题：

1. 这个 API 是用户直接调用的吗，还是内部辅助？
2. 它是立即发 TIR frame，还是返回高层 `call_intrin`？
3. 它是否依赖 `Builder.current()`？
4. 它给 IR 附加了哪些 metadata：scope、layout、annotations、attrs、pipeline config？
5. 它生成的节点后续由哪个 pass 消费：`LayoutInference`、`LowerTileOp`、pipeline pass、intrinsic lowering、codegen？
6. 它是否只是 wrapper，真正逻辑在 C++ FFI 或 backend pass 中？

这比逐行硬读更有效。

---

## 25. 文件职责速查表

| 文件/目录 | 主要职责 | 阅读优先级 |
| --- | --- | --- |
| `__init__.py` | 汇总导出 `T` 命名空间 | 高 |
| `kernel.py` | `T.Kernel`、launch frame、thread/block binding 查询 | 高 |
| `eager/ast.py` | Python AST 改写为 Builder 调用 | 高 |
| `eager/builder.py` | IRBuilder wrapper、JITFunc、macro、const、PrimFunc 构造 | 高 |
| `proxy.py` | Tensor/Buffer/ptr 类型代理 | 高 |
| `allocate.py` | shared/local/fragment/barrier/tmem allocation | 高 |
| `loop.py` | Parallel/Pipelined/Persistent/serial/unroll/vectorized | 高 |
| `copy_op.py` | 高层 data movement tile op | 高 |
| `gemm_op.py` | 高层 GEMM tile op | 高 |
| `reduce_op.py` | reduction macro 和 reduce tile op | 中高 |
| `frame.py` | let value 和 BufferRegion alias 追踪 | 中高 |
| `annotations.py` | 编译 hint 注入 | 中 |
| `warpgroup.py` | warp specialization scope | 中 |
| `fill_op.py` / `scan_op.py` | fill/clear、prefix scan 类 TileOp | 中 |
| `print_op.py` | device-side print/assert wrapper | 中低 |
| `builtin.py` | 底层 GPU intrinsic wrapper | 中低，后读 |
| `math_intrinsics.py` / `fastmath.py` | 数学 intrinsic wrapper | 中低 |
| `customize.py` / `atomic.py` | atomic、reshape/view、定制 intrinsic | 中低 |
| `cluster.py` | cluster barrier/cancel/query API | 中低 |
| `pdl.py` | PDL trigger/sync API | 中低 |
| `dtypes.py` / `symbolics.py` / `random.py` | dtype 对象、dynamic/symbolic marker、随机数 API | 中低 |
| `experimental/` | sparse GEMM 等实验性 TileOp | 中低 |
| `tir/` | TVM/TIR wrapper 和 parser entry | 中 |
| `parser/` | TVM script parser 兼容/扩展 | 中低 |
| `overrides/` | parser/buffer 行为补丁 | 中低 |
| `ast/` | TIR builder 兼容层/FFI buffer helper | 中低 |

---

## 26. 最后总结

`tilelang/language` 的核心设计可以压缩成一句话：

> 用 Python 语法构造带有 TileLang 高层语义的 TIR，而不是用 Python 执行 GPU 计算。

它最有价值的地方，是把用户熟悉的 Python 写法转换成包含丰富 metadata 的 IR：

- launch 语义：`T.Kernel`
- 内存层次：`T.Tensor`、`T.alloc_shared`、`T.alloc_fragment`
- 并行和流水：`T.Parallel`、`T.Pipelined`
- 高层数据移动：`T.copy`
- 高层张量计算：`T.gemm`
- layout/scope/annotation/pipeline 信息

然后让后端 compiler pass 基于这些信息做：

- layout inference
- tile op lowering
- software pipeline injection
- memory planning
- vectorization
- hardware intrinsic selection
- final codegen

所以阅读这个目录时，不要只看函数名，也不要把 `T.copy/T.gemm` 当普通 Python 函数理解。要始终追问：**它往 IR 里放了什么语义？这个语义后面由哪个 pass 消费？**

抓住这条线，`tilelang/language` 就不是一堆零散 API，而是一套完整的 Python embedded DSL 前端。

---

## 附录 A. `KernelLaunchFrame`、`TIRFrame`、`FrameStack` 问答补充

这个附录记录一次围绕 `tilelang/language/kernel.py` 的源码阅读问题，重点解释几个容易混淆的概念：`TIRFrame`、`KernelLaunchFrame`、`FrameStack`、`self.frames`、`SBlockFrame` 和 Python slice 语义。

### A.1 `TIRFrame` 是什么

`TIRFrame` 可以理解成 TVM/TIRX script builder 里的“语法作用域帧”。

用户写 TileLang DSL 时，很多代码并不是立即执行计算，而是在构造 TIR AST。例如：

```python
with T.Kernel(...) as bx:
    ...
```

进入 `with` 时，builder 需要知道“现在正在构造一个 kernel launch 作用域”；退出 `with` 时，builder 要把这个作用域收起来，组装成对应的 TIR 结构。`TIRFrame` 就是这类 builder-time context object 的基类。

所以 `TIRFrame` 主要服务于 **构造 TIR 的过程**：

```text
进入 frame
    -> 建立 IRBuilder 上下文
    -> 收集 body 里的 statements / 子 frame

退出 frame
    -> 把收集到的内容组装成 TIR node
    -> 返回给上层 builder
```

最终真正留下来并被 lowering/codegen 使用的是 TIR tree，例如 `For`、`Block`、`SeqStmt`、`AttrStmt`、`PrimFunc` 等。`TIRFrame` / `KernelLaunchFrame` 本身更像构造期脚手架，构造出 TIR tree 之后，语义上就不需要继续参与运行时执行。

### A.2 `KernelLaunchFrame` 是什么

`KernelLaunchFrame` 是 TileLang 为 `T.Kernel(...)` 定制的一个 `TIRFrame` 子类：

```python
@register_object("tl.KernelLaunchFrame")
class KernelLaunchFrame(TIRFrame):
    ...
```

它代表一个 kernel launch 的外层作用域。用户写：

```python
with T.Kernel(grid_x, grid_y, threads=(128, 2)) as (bx, by):
    ...
```

`T.Kernel(...)` 最终通过 `_ffi_api.KernelLaunch(blocks, threads, attrs)` 创建一个 `KernelLaunchFrame`。这个 frame 内部的 `self.frames` 持有 launch 骨架所需的一组直接子 frame，大致可以理解成：

```text
KernelLaunchFrame
  self.frames:
    blockIdx.x frame
    blockIdx.y frame
    blockIdx.z frame        # 如果对应维度被创建
    threadIdx.x frame
    threadIdx.y frame
    threadIdx.z frame
    SBlockFrame             # kernel body + kernel attrs 所在的 statement block
```

更精确地说：`KernelLaunchFrame` 不是“把 kernel body 里所有嵌套 TIRFrame 都平铺保存起来”的对象，而是一个表示 kernel launch 外壳的复合 frame。它的 `self.frames` 是这个 launch 的直接组成部分：block/thread 绑定，以及承载 body 的 `SBlockFrame`。

### A.3 `FrameStack` 和 `self.frames` 不是一回事

`kernel.py` 里定义的 `FrameStack` 是 TileLang 自己写的一个小栈，本质是 `deque` 包装：

```python
class FrameStack:
    def push(self, item): ...
    def pop(self): ...
    def top(self): ...
```

它的作用是记录“当前线程里，正在进入哪个 `KernelLaunchFrame`”。

进入 kernel launch 时：

```python
def __enter__(self):
    super().__enter__()
    _get_current_stack().push(self)
```

退出时：

```python
def __exit__(self, ptype, value, trace):
    stack = _get_current_stack()
    if stack.top() is self:
        stack.pop()
    super().__exit__(ptype, value, trace)
```

这个栈服务于：

```python
KernelLaunchFrame.Current()
```

也就是让 `T.get_thread_binding()`、`T.get_block_binding()` 这类 API 不需要用户显式传当前 kernel frame，就能找到当前上下文。

而 `self.frames` 来自父类 `TIRFrame` / builder 体系。它不是 `FrameStack`，而是当前这个 `KernelLaunchFrame` 内部的直接子 frame 列表。

两者区别可以这样记：

```text
FrameStack:
    管“当前线程现在处在哪个 KernelLaunchFrame 里”
    是 thread-local 的当前上下文栈

self.frames:
    管“这个 KernelLaunchFrame 自己由哪些子 frame 组成”
    是当前 frame 的 launch 骨架结构
```

### A.4 为什么 `__exit__` 要判断 `stack.top() is self`

正常情况下，`with T.Kernel(...)` 的进入和退出是严格配对的：

```text
__enter__ -> push(self)
__exit__  -> pop(self)
```

但 `__exit__` 没有直接 `stack.pop()`，而是写成：

```python
if stack.top() is self:
    stack.pop()
```

这是为了保证当前 frame 只弹出自己。如果因为嵌套 frame、异常路径或 builder 状态错乱导致栈顶已经不是当前对象，直接 `pop()` 会误删别的 frame，让上下文更乱。

这里用 `is` 而不是 `==`，是因为要比较对象身份：退出哪个 `with`，就只能清理同一个 `KernelLaunchFrame` 实例。

### A.5 `SBlockFrame` 为什么不展开成 x/y/z 三维

这里最容易混淆的是两个 “block” 不是一个概念。

```text
blockIdx.x / blockIdx.y / blockIdx.z:
    CUDA launch 的 grid/block 维度索引
    每一维都有独立 iter_var

SBlockFrame:
    TVM/TIR 里的 statement block / scope block
    用来承载 kernel body 和 annotations
```

所以 `threadIdx.x/y/z` 或 `blockIdx.x/y/z` 是索引维度，天然可以按 x/y/z 展开；但 `SBlockFrame` 是语句作用域，不是索引空间。

可以把 `SBlockFrame` 类比成 CUDA kernel body 外面那层大括号：

```cuda
__global__ void kernel(...) {
    // TileLang DSL body 最终构造出的 statements 在这里
}
```

它不表示 `block.x`、`block.y`、`block.z`，而是表示一个承载 body 和属性的 TIR block。

在 `KernelLaunchFrame.__enter__` 中，最后一个 frame 会被检查为 `SBlockFrame`：

```python
last_block_frame = self.frames[-1]
assert isinstance(last_block_frame, SBlockFrame)
```

然后从它的 `annotations` 里读取 kernel 级别属性：

```python
maybe_cpu = last_block_frame.annotations.get("tilelang.is_cpu_kernel_frame", False)
```

例如 CPU kernel 标记、`pragma_import_c`、`cluster_dims` 这类信息都挂在这个 block scope 上。

### A.6 `get_block_bindings()` 取的是什么

`get_block_bindings()` 的实现是：

```python
def get_block_bindings(self) -> list[Var]:
    return [frame.iter_var.var for frame in self.frames[0:-4]]
```

这里读取的就是 `blockIdx.*` 对应的 frame 绑定变量。

对 GPU kernel launch，代码约定最后 4 个 frame 是：

```text
threadIdx.x frame
threadIdx.y frame
threadIdx.z frame
SBlockFrame
```

因此：

```python
self.frames[0:-4]
```

表示“从开头取到倒数第 4 个之前”，也就是排除最后 4 个，只留下前面的 `blockIdx.*` frame。

如果要取最后 4 个，Python slice 语法才是：

```python
self.frames[-4:]
```

举例：

```python
frames = [
    "blockIdx.x",
    "blockIdx.y",
    "threadIdx.x",
    "threadIdx.y",
    "threadIdx.z",
    "SBlockFrame",
]

frames[0:-4]
# ["blockIdx.x", "blockIdx.y"]

frames[-4:]
# ["threadIdx.x", "threadIdx.y", "threadIdx.z", "SBlockFrame"]
```

所以 `get_block_bindings()` 用 `self.frames[0:-4]` 是为了取 block binding，而不是取最后 4 个。

### A.7 `get_thread_bindings()` 为什么是 `self.frames[-4:-1]`

`get_thread_bindings()` 的实现是：

```python
def get_thread_bindings(self) -> list[Var]:
    return [frame.iter_var.var for frame in self.frames[-4:-1]]
```

这个 slice 表示：

```text
从倒数第 4 个开始，取到倒数第 1 个之前
```

也就是：

```text
threadIdx.x frame
threadIdx.y frame
threadIdx.z frame
```

不包含最后那个 `SBlockFrame`。

单独取某一维 thread binding 时：

```python
def get_thread_binding(self, dim: int = 0) -> Var:
    return self.frames[-4 + dim].iter_var.var
```

对应关系是：

```text
dim = 0 -> self.frames[-4] -> threadIdx.x
dim = 1 -> self.frames[-3] -> threadIdx.y
dim = 2 -> self.frames[-2] -> threadIdx.z
```

### A.8 `with T.Kernel(...) as ...` 返回 block binding

GPU 情况下，`KernelLaunchFrame.__enter__` 返回的是 block binding，而不是 thread binding：

```python
return _normalize_bindings([frame.iter_var.var for frame in self.frames[0:-4]])
```

所以：

```python
with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
    ...
```

这里的 `bx`、`by` 是 `blockIdx.x`、`blockIdx.y` 绑定变量。

thread binding 要通过 API 查询：

```python
tx = T.get_thread_binding(0)
tx, ty, tz = T.get_thread_bindings()
```

单维 kernel 下，`_normalize_bindings` 会把单元素 list 转成裸 `Var`：

```python
with T.Kernel(n, threads=128) as bx:
    ...
```

同时 `kernel.py` 给 `Var` 补了 `__iter__` 和 `__len__`，所以也兼容：

```python
with T.Kernel(n, threads=128) as (bx,):
    ...
```

### A.9 CPU kernel 的特殊路径

GPU kernel 的 frame 末尾约定是 `threadIdx.x/y/z + SBlockFrame`，所以 block binding 用 `self.frames[0:-4]`。

CPU kernel 不创建 thread binding，`KernelLaunchFrame.__enter__` 会走另一条路径：

```python
if maybe_cpu:
    return _normalize_bindings([frame.vars[0] for frame in self.frames[0:-1]])
```

也就是排除最后的 `SBlockFrame`，返回普通 for frame 的 loop var。

这也解释了为什么 GPU 和 CPU 分支切片不同：GPU 多了三个 thread binding frame。

### A.10 最终心智模型

可以把这组关系压缩成一张图：

```text
thread-local FrameStack
  top -> 当前正在构造的 KernelLaunchFrame

KernelLaunchFrame  # 一个 T.Kernel(...) launch 作用域
  self.frames      # launch 的直接子 frame
    blockIdx.* frame(s)
    threadIdx.x frame
    threadIdx.y frame
    threadIdx.z frame
    SBlockFrame    # kernel body + annotations

TIR tree
  frame 退出后真正生成的 IR 结果
  后续 lowering/codegen 消费的是这里的 IR node
```

因此这轮讨论的结论是：

```text
FrameStack 是“当前 kernel launch 上下文栈”。
self.frames 是“当前 KernelLaunchFrame 的直接子 frame 列表”。
get_block_bindings() 从 self.frames[0:-4] 取 blockIdx.*。
get_thread_bindings() 从 self.frames[-4:-1] 取 threadIdx.x/y/z。
SBlockFrame 是 kernel body 的 TIR statement block，不是三维索引。
TIRFrame / KernelLaunchFrame 是构造 TIR 时的辅助结构，最终产物是 TIR tree。
```

---

## 附录 B. `register_object`、`_ffi_api` 和 C++ FFI 绑定顺序

这个附录补充 `tilelang/language/kernel.py` 里的这一行：

```python
@register_object("tl.KernelLaunchFrame")
class KernelLaunchFrame(TIRFrame):
        ...
```

问题的核心是：`register_object` 不是 Python 标准库里的东西，而是 TVM/TVM-FFI 提供的 Python-C++ 对象系统注册接口。它负责把 C++ 侧的 FFI object type key，绑定到 Python 侧的包装类。

### B.1 `register_object` 来自哪里

`kernel.py` 里写的是：

```python
from tvm.ffi import register_object
```

但在这个仓库里，`tvm.ffi` 基本只是转发到 `tvm_ffi`：

```python
# 3rdparty/tvm/python/tvm/ffi.py
from tvm_ffi import *
```

真正实现位于：

```text
3rdparty/tvm/3rdparty/tvm-ffi/python/tvm_ffi/registry.py
```

核心逻辑可以简化成：

```python
def register_object(type_key: str | None = None, *, init: bool = True):
        def _register(cls, object_name):
                type_index = core._object_type_key_to_index(object_name)
                if type_index is None:
                        raise ValueError(f"Cannot find object type index for {object_name}")
                info = core._register_object_by_index(type_index, cls)
                setattr(cls, "__tvm_ffi_type_info__", info)
                return cls
```

也就是说，`@register_object("tl.KernelLaunchFrame")` 做的不是“创建一个 Python 类”，而是：

```text
拿字符串 "tl.KernelLaunchFrame"
    -> 去 C++/FFI 类型系统里查 type index
    -> 把这个 type index 绑定到 Python class KernelLaunchFrame
```

如果 C++ 侧没有先注册 `"tl.KernelLaunchFrame"`，这里会找不到 object type index，然后报错。

### B.2 C++ 侧如何声明这个对象类型

C++ 侧对应代码在 `src/ir.cc`：

```cpp
class KernelLaunchFrameNode : public TIRFrameNode {
public:
    Array<TIRFrame> frames;

    static void RegisterReflection() {
        namespace refl = reflection;
        refl::ObjectDef<KernelLaunchFrameNode>().def_ro(
                "frames", &KernelLaunchFrameNode::frames);
    }

    TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.KernelLaunchFrame",
                                                                        KernelLaunchFrameNode, TIRFrameNode);
};
```

这里最重要的是：

```cpp
TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.KernelLaunchFrame", ...)
```

它声明了这个 C++ Object 的 type key。Python 里的装饰器必须写同一个字符串：

```python
@register_object("tl.KernelLaunchFrame")
```

所以 Python 和 C++ 不是靠文件名、类名自动匹配，而是靠这个 type key 字符串精确关联。

### B.3 `KernelLaunch` 函数如何暴露给 Python

对象类型注册是一条线，函数注册是另一条线。

C++ 侧 `KernelLaunch(...)` 是一个普通 C++ 函数：

```cpp
KernelLaunchFrame KernelLaunch(const Array<PrimExpr> &grid_size,
                                                             const Optional<Array<PrimExpr>> &block_size_opt,
                                                             const Map<String, Any> &attrs) {
    ObjectPtr<KernelLaunchFrameNode> n = make_object<KernelLaunchFrameNode>();
    ...
    return KernelLaunchFrame(n);
}
```

然后通过 FFI global registry 暴露出去：

```cpp
TVM_FFI_STATIC_INIT_BLOCK() {
    namespace refl = reflection;
    refl::GlobalDef()
            .def("tl.Parallel", ParallelFor)
            .def("tl.Pipelined", PipelinedFor)
            .def("tl.Persistent", PersistentFor)
            .def("tl.KernelLaunch", KernelLaunch);
}
```

这行：

```cpp
.def("tl.KernelLaunch", KernelLaunch)
```

注册的是“可调用函数”。它和 `@register_object("tl.KernelLaunchFrame")` 不是同一件事。

可以这样区分：

```text
GlobalDef().def("tl.KernelLaunch", KernelLaunch)
    注册 C++ 函数，让 Python 能调用 _ffi_api.KernelLaunch(...)

TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.KernelLaunchFrame", ...)
+ @register_object("tl.KernelLaunchFrame")
    注册对象类型映射，让 C++ 返回的对象能包装成 Python KernelLaunchFrame
```

### B.4 `_ffi_api.py` 怎么把 C++ 函数变成 Python 函数

TileLang 有一个很小的文件：

```python
# tilelang/_ffi_api.py
import tvm_ffi

tvm_ffi.init_ffi_api("tl", __name__)
```

`init_ffi_api("tl", __name__)` 会扫描 FFI global registry 里所有以 `tl.` 开头的函数。

比如 C++ 注册了：

```text
tl.KernelLaunch
tl.Parallel
tl.Pipelined
```

那么 Python module `tilelang._ffi_api` 里就会被自动挂上：

```python
_ffi_api.KernelLaunch
_ffi_api.Parallel
_ffi_api.Pipelined
```

简化后的逻辑是：

```python
def init_ffi_api(namespace: str, target_module_name: str | None = None):
        prefix = namespace
        target_module = sys.modules[target_module_name]

        for name in list_global_func_names():
                if not name.startswith(prefix):
                        continue

                fname = name[len(prefix) + 1:]
                if "." in fname:
                        continue

                f = get_global_func(name)
                setattr(target_module, fname, f)
```

所以 `kernel.py` 里能写：

```python
from tilelang import _ffi_api

return _ffi_api.KernelLaunch(blocks, threads, attrs)
```

本质上是在调用 C++ 注册的 `tl.KernelLaunch`。

### B.5 import 顺序：从 `import tilelang` 到 `T.Kernel(...)`

一次典型 import 的顺序可以理解成：

```text
1. import tilelang

2. 进入 tilelang/__init__.py
     - import tvm
     - 找到 libtilelang.dylib / libtilelang.so / tvm_compiler.dll
     - 用 ctypes.CDLL(...) 加载 TileLang C++ 动态库

3. 动态库被加载
     - C++ 静态初始化块执行
     - TVM_FFI_STATIC_INIT_BLOCK 注册 tl.* 函数
     - C++ object type key 也进入 FFI 类型系统

4. tilelang/__init__.py 继续 import tilelang.language

5. tilelang/language/__init__.py import .kernel

6. tilelang/language/kernel.py 执行
     - from tvm.ffi import register_object
     - from tilelang import _ffi_api

7. tilelang/_ffi_api.py 执行
     - tvm_ffi.init_ffi_api("tl", __name__)
     - 把 C++ 注册的 tl.KernelLaunch 暴露成 _ffi_api.KernelLaunch

8. kernel.py 执行类定义装饰器
     - @register_object("tl.KernelLaunchFrame")
     - 查 C++ FFI type key
     - 把 type index 绑定到 Python class KernelLaunchFrame
```

这里的关键顺序是：动态库必须先加载，C++ 侧的 `tl.*` 函数和 object type key 才会出现在 FFI registry 里。然后 Python 的 `_ffi_api` 和 `register_object` 才能根据这些已注册信息进行绑定。

### B.6 当用户写 `with T.Kernel(...)` 时发生什么

完整调用链可以压缩成：

```text
with T.Kernel(grid, threads=128) as bx:
        ...

T.Kernel(...)
    -> Python 函数 tilelang.language.kernel.Kernel
    -> 检查 Builder.current()
    -> 规范化 threads / cluster_dims / attrs
    -> 调 _ffi_api.KernelLaunch(blocks, threads, attrs)

_ffi_api.KernelLaunch(...)
    -> 调 C++ global function "tl.KernelLaunch"
    -> C++ 创建 KernelLaunchFrameNode
    -> 填入 blockIdx/threadIdx frames 和 tilelang_root block
    -> 返回 C++ KernelLaunchFrame 对象

返回 Python
    -> tvm_ffi 看到对象 type key 是 "tl.KernelLaunchFrame"
    -> 查到 Python class KernelLaunchFrame
    -> 包装成 Python KernelLaunchFrame 实例

Python with 语义
    -> 调 KernelLaunchFrame.__enter__
    -> super().__enter__ 进入底层 TIRFrame scope
    -> push 到 Python thread-local FrameStack
    -> 返回 block binding，例如 bx/by/bz

退出 with
    -> 调 KernelLaunchFrame.__exit__
    -> 从 FrameStack pop 当前 frame
    -> super().__exit__ 退出底层 TIRFrame scope
```

所以 `@register_object("tl.KernelLaunchFrame")` 的作用可以一句话概括为：

```text
它让 C++ 返回的 tl.KernelLaunchFrame FFI 对象，在 Python 里变成 tilelang.language.kernel.KernelLaunchFrame 实例，
从而拥有 Python 侧定义的 __enter__ / __exit__ / get_thread_binding 等行为。
```

### B.7 两种 registry 的心智模型

最后可以把这套机制拆成两张 registry 表：

```text
Global function registry
    key: "tl.KernelLaunch"
    value: C++ 函数 KernelLaunch
    Python 入口: _ffi_api.KernelLaunch(...)

Object type registry
    key: "tl.KernelLaunchFrame"
    value: C++ KernelLaunchFrameNode type info + Python KernelLaunchFrame class
    Python 入口: @register_object("tl.KernelLaunchFrame")
```

两者一起工作，才得到最终效果：

```text
Python 能调用 C++ 函数，且 C++ 返回的对象能恢复成正确的 Python 包装类。
```

这也是 TileLang 里很多 API 的共同模式：

```text
Python DSL 函数很薄
    -> 参数整理
    -> 调 _ffi_api.xxx
    -> C++ 创建/变换 IR object
    -> Python register_object 提供包装类和便捷方法
```

## 附录 C. `@tilelang.jit`、AST、IRGenerator 补充问答

这个附录整理两个常见追问：

1. `@tilelang.jit` 到底怎么运作，跟 `tilelang/language/` 目录相关的调用链是什么。
2. Python AST、`IRGenerator`、`IRBuilder` 分别是什么，它们之间是什么关系。

### C.1 `@tilelang.jit` 的职责

`@tilelang.jit` 自己不是 parser，也不是 IR builder。它更像一个 **JIT 包装层**：

- 在装饰阶段把原函数包装成 `JITImpl`
- 在第一次调用时判断这是 `lazy style` 还是 `eager style`
- 通过 `tilelang/language/eager/` 里的机制把用户函数变成 `PrimFunc`
- 再把 `PrimFunc` 交给后续 compile / cache / backend pipeline

压缩成一句话：

```text
@tilelang.jit 负责调度；
tilelang/language/eager 负责把 Python DSL 变成 PrimFunc。
```

### C.2 装饰阶段的调用链

当用户写：

```python
@tilelang.jit
def foo(...):
    ...
```

装饰阶段的关键调用链是：

```text
tilelang.jit.jit(...)
    -> tilelang.language.eager.builder.prim_func(func, eager_jit=True)
    -> tilelang.language.eager.ast.mutate(func)
    -> 返回 JITFunc
    -> 用 JITFunc 构造 JITImpl
```

这里有两个关键点：

- `mutate(func)` 会把原始 Python 函数的 AST 改写成一个 IR 生成函数。
- `prim_func(..., eager_jit=True)` 这一步不会立刻生成 `PrimFunc`，而是先返回 `JITFunc`，把真正的 IR 构建延迟到函数第一次调用时。

### C.3 调用阶段：先判定 lazy/eager，再生成 PrimFunc

当用户第一次调用 `foo(...)` 时，主链路是：

```text
JITImpl.__call__
    -> JITImpl._infer_jit_mode()
    -> JITFunc._is_lazy_style()
    -> JITFunc.parse_args()
    -> JITImpl.compile()
    -> JITImpl.get_tir()
    -> JITFunc.get_tir()
```

真正跟 `tilelang/language/` 相关的是 `get_tir()` 这段。

#### lazy style

如果用户函数内部返回一个 `@T.prim_func`：

```python
@tilelang.jit
def make_kernel(M, N):
    @T.prim_func
    def kernel(...):
        ...
    return kernel
```

则：

```text
外层函数执行
    -> 内层 @T.prim_func 直接构造 PrimFunc
    -> 外层返回 PrimFunc
    -> JITImpl 再编译这个 PrimFunc
```

这里的 `@T.prim_func` 默认走 `tilelang.language.eager.builder.prim_func(..., eager_jit=False)`。  
也就是说，虽然用户感觉这是“lazy 风格”，但语言层的 `PrimFunc` 构造仍然是经过 `eager/builder.py` 的。

#### eager style

如果用户直接在 `@tilelang.jit` 的函数体里写：

```python
@tilelang.jit
def kernel(A, B):
    M, N = T.const("M N")
    A: T.Tensor((M, N), "float16")
    with T.Kernel(...):
        ...
```

则它会走两阶段 eager JIT：

```text
phase1
    -> Builder(eager_jit="phase1")
    -> 执行改写后的 IRGenerator
    -> 收集 constexpr / tensor 参数 / 模板 PrimFunc

phase2
    -> 从真实 tensor shape/stride 提取 M/N/...
    -> Builder(eager_jit="phase2")
    -> 再执行一次 IRGenerator
    -> 得到最终 PrimFunc
```

所以 eager 模式下，用户函数体其实会被“以构造 IR 的方式执行两次”，而不是按普通 Python 运行时语义执行一次。

### C.4 跟 `tilelang/language/` 目录最相关的调用链

如果只保留语言层相关模块，可以把主链总结成：

```text
原始 Python 函数
    -> tilelang.language.eager.ast.mutate
    -> IRGenerator
    -> tilelang.language.eager.builder.Builder
    -> tilelang.language.kernel / proxy / allocate / loop / builtin ...
    -> Builder 内部驱动 IRBuilder / tirx frame
    -> PrimFunc
```

一个典型的 eager kernel 子链看起来是：

```text
with T.Kernel(...)
    -> tilelang.language.kernel.Kernel(...)
    -> _ffi_api.KernelLaunch(...)
    -> 返回 KernelLaunchFrame
    -> Builder.ctx_with(...)
    -> Builder.with_frame(...)
    -> KernelLaunchFrame.__enter__()
```

因此：

- `kernel.py` 负责 launch frame 语义
- `proxy.py` 负责 Tensor/Buffer 代理类型
- `allocate.py` 负责 allocation API
- `loop.py` 负责循环 frame API
- `eager/ast.py` 负责把 Python 语法改写成 Builder 调用
- `eager/builder.py` 负责把这些调用落成真实 TIR/TIRX IR

### C.5 Python AST 是什么

AST 是 Abstract Syntax Tree，抽象语法树。  
它表示的是“这段 Python 代码的结构”，不是执行结果。

例如：

```python
def foo(x):
    y = x + 1
    return y
```

在 AST 层面，它会是这样的结构元素：

- `FunctionDef`
- `Assign`
- `Name("y")`
- `BinOp(x, Add, 1)`
- `Return`

TileLang 先拿到这棵 AST，然后做改写。  
这个动作在：

```text
tilelang.language.eager.ast.mutate(func)
```

它的作用不是“读源码用于展示”，而是：

- 识别 `if / for / with / return / 赋值`
- 把这些 Python 语句改写成对 `Builder` 的调用
- 让原本看起来像 Python 的 DSL 代码，最终变成“构建 IR 的 Python 代码”

### C.6 `IRGenerator` 是什么

`IRGenerator` 不是 IR，也不是 builder。  
它是 AST 改写之后得到的一个 **IR 生成器包装对象**。

可以把它理解成：

```text
“一段已经被 AST mutator 改写过、执行时会调用 Builder API 的 Python 函数”
```

它的核心字段是：

```python
IRGenerator(
    gen=Callable[[BaseBuilder], Callable[..., ...]],
    source=...,
    extra_type_hints=...,
)
```

其中：

- `gen(builder)` 会返回一个闭包函数
- 这个闭包在执行时，不再是普通 Python 语义，而是会调用 `builder.bind`、`builder.ctx_for`、`builder.ctx_with`、`builder.ret` 等方法

所以 `IRGenerator` 的作用是：

- 保存改写后的函数逻辑
- 延迟到真正构建 IR 时再执行
- 执行时把语句导向 `Builder`

### C.7 `IRBuilder` 是什么

`IRBuilder` 是更底层的 TVM/TIRX IR 构造器。  
TileLang 的 `Builder` 内部持有它：

```text
Builder
    └── self.ir_builder = IRBuilder()
```

真正往 TIR/TIRX 里写节点的是这层：

- `Builder.prim_func(...)` 打开 `IRBuilder` 上下文
- `Builder.ctx_if / ctx_for / bind / ret / eval ...` 内部调用 `tirx.*`
- `Builder.get()` 最后从 `IRBuilder` 里取出 `PrimFunc`

所以三者职责分别是：

- AST：原始 Python 函数的语法结构
- `IRGenerator`：AST 改写后的“生成函数包装器”
- `IRBuilder`：底层真实的 IR 构造器

### C.8 `IRBuilder` 接收的是 `IRGenerator` 吗

不是。

更准确的关系是：

```text
Python 函数
    -> AST
    -> IRGenerator
    -> Builder
    -> IRBuilder
    -> PrimFunc
```

也就是说：

- `IRGenerator` 执行时接收的是 `Builder`
- `Builder` 再去驱动 `IRBuilder`

不是：

```text
IRBuilder(IRGenerator)
```

而是：

```text
IRGenerator --执行--> Builder --使用--> IRBuilder
```

### C.9 一个最短心智模型

如果只记一句话，记这个：

```text
@tilelang.jit 负责包装和调度；
AST 改写负责把 Python DSL 变成 Builder 调用；
Builder 再借助 IRBuilder 把这些调用落成 PrimFunc。
```
