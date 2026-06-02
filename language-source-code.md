# TileLang `language/` 目录源码阅读指南

这份笔记整理 `tilelang/language` 目录的职责、核心设计、关键代码细节和推荐阅读路线。重点不是只列文件名，而是解释：**用户写的 Python DSL 是如何被捕获、转换成 TIRX `PrimFunc`，并保留高层 tile op 给后续 compiler pass lowering 的**。

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

## 核心精髓：两种 JIT 风格与 AST 改写

如果只能抓住 `language/` 的一个核心机制，那就是这一段：**TileLang 同时支持 lazy style 和 eager style；lazy style 基本走 TVM script parser，eager style 则靠 TileLang 自己的 AST mutator 把 Python 函数改写成 Builder 调用。**

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

- 内部 `@T.prim_func` 基本走 TVM/TIR script parser 路线。
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
| `tvm.tirx.script.parser.*` | 基础 TIR script parser 能力，兼容 `@T.prim_func`、TIR 类型/表达式 |
| `.eager` | eager JIT 的 `prim_func`、`macro`、`const` 等 |
| `.tir.ir` | TIR 基础表达式、数学函数、loop wrappers |
| `.proxy` | `Tensor`、`Buffer`、`ptr`、`make_tensor`、`SharedBuffer`、`FragmentBuffer` |
| `.kernel` | `Kernel`、`CUDASourceCodeKernel`、`KernelLaunchFrame`、thread/block binding 查询 |
| `.allocate` | `alloc_shared`、`alloc_fragment`、`alloc_local`、`alloc_var`、`alloc_barrier`、`alloc_tmem` |
| `.loop` | `Parallel`、`Pipelined`、`Persistent`、`serial`、`unroll`、`vectorized` |
| `.copy_op` | `copy`、`async_copy`、`tma_copy`、`copy_cluster`、`transpose`、`im2col` |
| `.gemm_op` | `gemm`、`wgmma_gemm`、`tcgen05_gemm`、blockscaled GEMM |
| `.reduce_op` | `reduce_sum/max/min`、`finalize_reducer`、`warp_reduce_*` |
| `.customize` | atomics、`dp4a`、`reshape`、`view`、`loop_break` |
| `.annotations` | layout/swizzle/L2/restrict/min blocks per SM hints |
| `.builtin` | GPU intrinsic、barrier、shuffle、WGMMA/TCGEN05 helper、load/store intrinsic |
| `.cluster` | cluster barrier、cluster copy/cancel 查询等 |
| `.pdl` | CUDA PDL trigger/sync |
| `.warpgroup` | `T.ws` warp-specialization scope |

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

### 6.3 C++ 侧真正创建 frame

C++ 实现在 `src/ir.cc` 的 `KernelLaunch(...)`。

GPU kernel 情况下它会创建：

```text
grid_size[0] -> blockIdx.x / bx
grid_size[1] -> blockIdx.y / by
grid_size[2] -> blockIdx.z / bz

block_size[0] -> threadIdx.x / tx
block_size[1] -> threadIdx.y / ty
block_size[2] -> threadIdx.z / tz

最后加一个 tilelang_root block
```

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
        self.constexpr_var = set()
        self.eager_jit = "none"  # phase1 / phase2 / none
        self.eager_jit_subs = {}
        self.func_pass_configs = None
        self.func_compile_flags = None
```

几个关键点：

- `frames`：追踪当前进入的 TIR frame，比如 PrimFuncFrame、ForFrame、IfFrame、KernelLaunchFrame。
- `ir_builder`：底层 TVM/TIRX IRBuilder。
- `constexpr_var`：eager JIT 动态 shape 推导用。
- `eager_jit`：分两阶段处理动态 shape。

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

### 8.9 lazy/eager JIT 判断

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
| `alloc_wgmma_desc` | `local.descriptor.wgmma` | Hopper WGMMA descriptor |
| `alloc_tcgen05_smem_desc` | `local.descriptor.tcgen05_smem` | Blackwell smem descriptor |
| `alloc_tcgen05_instr_desc` | `local.descriptor.tcgen05_instr` | Blackwell instruction descriptor |

这些对象后续会被 specific lowering pass 或 intrinsic lowering 消费。

### 10.4 `T.empty`

`empty` 是 eager-style JIT 的输出 tensor 声明：

```python
C = T.empty((M, N), dtype)
return C
```

它不会直接分配 device memory，而是创建一个 `OutTensor` 描述，用于 JIT wrapper 在运行时准备输出 tensor，并把它映射到 PrimFunc 的输出 buffer。

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

### 11.3 `T.Persistent`

`Persistent(domain, wave_size, index, group_size)` 表示 persistent kernel / persistent threadblock 风格。

它不是普通 for-loop，而是给后续 `PersistThreadblock` 类 pass 提供语义，让一个 CTA 反复领取 tile work，减少 launch/调度开销。

### 11.4 `serial / unroll / vectorized`

这些是标准 TIR loop kind 的 wrapper。

`loop.py` 中的 `serial/unroll` 额外支持 `step`，Builder 会把带 step 的 loop 转换成 trip count loop，再把 loop var 映射回 `start + v * step`。

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
| `tma_copy` | 用户管理 barrier 的 TMA producer 操作 |
| `copy_cluster` | cluster-aware copy，支持 TMA multicast / SM-to-SM |
| `transpose` | 带转置的数据搬运 |
| `im2col/c2d_im2col` | 卷积类数据重排 |

---

## 13. `gemm_op.py`：GEMM 高层 TileOp

文件：`tilelang/language/gemm_op.py`

`T.gemm` 也是高层 tile op。Python 侧主要做参数规范化和合法性检查，真正选择 MMA/WGMMA/TCGEN05 的地方在 lowering。

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

### 13.6 `GemmWarpPolicy`

`GemmWarpPolicy` 会影响 warp 如何覆盖 tile。常见策略如 square、row/column 方向展开。它不是 Python 层执行逻辑，而是传给 tile op lowering，用于 instruction/layout 选择。

---

## 14. `reduce_op.py`：Reduction 与 macro 展开

文件：`tilelang/language/reduce_op.py`

`T.reduce` 的实现很能体现 language 层的职责：它不只是发一个 intrinsic，还会根据 memory scope 自动插入中转逻辑。

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

### 14.4 `warp_reduce_*`

`warp_reduce_sum/max/min/bitand/bitor` 是更低层的 register value reduction：

```python
tirx.call_intrin(value.dtype, tirx.op.Op.get("tl.warp_reduce_sum"), value)
```

它们不走 fragment 中转，语义更接近 warp shuffle intrinsic。

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

当前 `language/__init__.py` 中已经直接导入 `tvm.tirx.script.parser`，并有注释说明希望未来完全兼容 upstream，所以这块不是初学主线。

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
| `builtin.py` | 底层 GPU intrinsic wrapper | 中低，后读 |
| `math_intrinsics.py` / `fastmath.py` | 数学 intrinsic wrapper | 中低 |
| `customize.py` / `atomic.py` | atomic、reshape/view、定制 intrinsic | 中低 |
| `cluster.py` | cluster barrier/cancel/query API | 中低 |
| `pdl.py` | PDL trigger/sync API | 中低 |
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