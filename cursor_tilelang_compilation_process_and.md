# TileLang compilation process and source design
_Exported on 5/22/2026 at 15:56:14 GMT+8 from Cursor (3.0.13)_

---

**User**

@TILELANG_AI_INFRA_LEARNING_PLAN.md:159-168 我正在开始学习tilelang的完整编译过程，给出详细学习指南，从第3-4周开始。我已经写过tilelang DSL了。另外，给我独立讲解一下源码的核心设计和抽象，帮助我更好的阅读源码

---

**Cursor**

让我先全面探索 TileLang 源码结构，给出最准确的学习指南和源码核心设计讲解。

我用了几分钟把第 3-4 周开始的关键文件和抽象都对照了一遍源码，下面给你一份比 `TILELANG_AI_INFRA_LEARNING_PLAN.md` 更具体、可执行的学习路线，以及一份独立的「源码核心设计 + 抽象拆解」笔记。两部分可以独立使用：前者按周推进，后者按抽象层切片。

> **运行环境约定**：下文所有 Python 代码片段都假设你用 `pip install tilelang`（当前最新 0.1.9）安装的正式包，不依赖任何只存在于 main 分支上的内部 API。当源码（HEAD）已有更方便的入口但 pip 包还没发出时，会在代码块上方明确标注。源码里的 `tilelang/engine/lower.py::lower_to_host_device_ir` 就是一个典型例子——它在 pip 0.1.9 里**还没有**，所以下面统一改用 pass 级入口 `LowerAndLegalize` / `OptimizeForTarget`。

---

## 一、第 3-4 周开始的详细学习路线

前提：你已经会写 TileLang DSL，能跑 `T.Kernel + T.copy + T.gemm` 这类小 kernel，懂 `alloc_shared / alloc_fragment / Pipelined / Parallel` 的含义。

阶段目标从「会用」过渡到「会读」，最后过渡到「会改 + 会提 PR」。

### 第 3 周：GEMM 上下游 + Layout / Fragment 抽象

主题：把 `T.gemm` 的整条路径拉直，从 Python DSL 一路追到 C++ `GemmNode::Lower` 和具体后端 emitter。

**Day 1 - DSL 入口与 IR 形态**

任务：把同一个 GEMM kernel 在不同阶段的 IR 打出来，建立 mental model。

读这几处（这是 GEMM 在 DSL 层的形态）:

```149:199:tilelang/language/gemm_op.py
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
) -> tir.PrimExpr:
```

注意上面这一层只做两件事：参数 shape/stride check + 包成 `tir.call_intrin("tl.tileop.gemm", ...)`。它本身**不生成 PTX，也不生成 CUDA 源**。

实验脚本分两条路径走：

> **兼容性提示**：
> - pip 上目前发布的 `tilelang` 还没有 `tilelang.engine.lower.lower_to_host_device_ir`（那是 main 分支后加的）。下面统一用 `LowerAndLegalize` + `OptimizeForTarget` 这两个 pip 0.1.x 上就有的公共入口直接跑 pass pipeline。
> - 你的本地 dev 构建是 `USE_CUDA=OFF` 的（只编了 CUDA stubs，没编 backend 实现），所以**在你的 Mac 上 `target="cuda -arch=sm_80"` 会在 `LayoutInference` 那一步就挂掉**，错误信息是 `tl.copy requires a target-specific implementation, but no copy implementation is registered for Target(...kind='cuda'...)`。要看 T.gemm 的完整 lowering，**必须在 USE_CUDA=ON 的环境里跑**（一般是远端 Linux + CUDA 机器）。在 Mac 上 IR pass 可以走 `target="metal"`，但 `T.gemm` 只有 CUDA/HIP/Tcgen05 后端有实现，所以 Mac Metal 路径要换成手写 SIMT loop。

**Mac 上能跑通的版本（target="metal"，不写 T.gemm）：**

```python
import tilelang  # noqa: F401  # 触发 tilelang native lib 加载
import tilelang.language as T
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget


@T.prim_func
def matmul_simt(
    A: T.Tensor((128, 128), "float32"),
    B: T.Tensor((128, 128), "float32"),
    C: T.Tensor((128, 128), "float32"),
):
    with T.Kernel(2, 2, threads=128) as (bx, by):
        A_s = T.alloc_shared((64, 32), "float32", scope="shared")
        B_s = T.alloc_shared((32, 64), "float32", scope="shared")
        C_l = T.alloc_fragment((64, 64), "float32")
        T.clear(C_l)
        for ko in T.Pipelined(4, num_stages=0):
            T.copy(A[by * 64, ko * 32], A_s)
            T.copy(B[ko * 32, bx * 64], B_s)
            for i, j in T.Parallel(64, 64):
                for k in T.Serial(32):
                    C_l[i, j] += A_s[i, k] * B_s[k, j]
        T.copy(C_l, C[by * 64, bx * 64])


print("=== 原始 PrimFunc (DSL 直接产出) ===")
print(matmul_simt.script())

target = tvm.target.Target("metal")
mod = tvm.IRModule.from_expr(matmul_simt.with_attr("global_symbol", "main"))
with target:
    mod = LowerAndLegalize(mod, target)
    mod = OptimizeForTarget(mod, target)

print("=== Lower + Optimize 后的 IR（host + device 在同一个 mod 里）===")
print(mod.script())
```

> 这里跑完之后整个 `mod` 里只有一个 `main` PrimFunc——`SplitHostDevice` 已经把 device 区域标好了，但在 Metal 这种「host 端没有独立 launcher 函数」的 target 上不会再切出第二个 PrimFunc。如果换成 USE_CUDA=ON 的环境跑下面 CUDA 版本，你会看到 `main`（host）+ `main_kernel`（device）两个 PrimFunc，这才是「host/device split」的典型样子。

**远端 CUDA 机器（USE_CUDA=ON 的 wheel / 源码构建）上的版本（含 `T.gemm`）：**

```python
import tilelang  # noqa: F401
import tilelang.language as T
from tilelang import tvm
from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget


@T.prim_func
def gemm_128x128(
    A: T.Tensor((128, 128), "float16"),
    B: T.Tensor((128, 128), "float16"),
    C: T.Tensor((128, 128), "float16"),
):
    with T.Kernel(1, threads=128) as (_,):
        A_s = T.alloc_shared((128, 128), "float16")
        B_s = T.alloc_shared((128, 128), "float16")
        C_l = T.alloc_fragment((128, 128), "float16")
        T.copy(A, A_s)
        T.copy(B, B_s)
        T.clear(C_l)
        T.gemm(A_s, B_s, C_l)
        T.copy(C_l, C)


print(gemm_128x128.script())

target = tvm.target.Target("cuda -arch=sm_80")
mod = tvm.IRModule.from_expr(
    gemm_128x128.with_attr("global_symbol", "gemm_128x128")
)
with target:
    mod = LowerAndLegalize(mod, target)
    mod = OptimizeForTarget(mod, target)

print(mod.script())  # 应该能看到 ptx_mma / cp_async 等低层 intrinsic
```

如果你也想模拟老 `lower_to_host_device_ir` 那种「host_mod + device_mod 两个独立 IRModule」的返回，CUDA 版本上追加一段：

```python
from tvm import tir
from tilelang.engine.lower import (
    is_cpu_device_backend,
    get_host_call,
    get_device_call,
)

is_device_c = is_cpu_device_backend(target)
host_mod = tir.transform.Filter(get_host_call(is_device_c=is_device_c))(mod)
device_mod = tir.transform.Filter(get_device_call(is_device_c=is_device_c))(mod)

print("=== device IR ===")
print(device_mod.script())
```

观察重点：

- 原始 PrimFunc 里 `T.gemm` / `T.copy` 都只是 `T.call_intrin("tl.tileop.*", ...)`。
- 跑完 pass 之后：CUDA 路径会出现 `T.ptx_mma / T.warpgroup_* / cp_async`；Metal 路径会出现普通 SIMT loop + `simdgroup_*`（如果走 Metal MMA）/ 否则纯 `threadIdx.x` 循环。
- `T.copy(A, A_s)` 在不同 target 下会变成 `cp.async` / `tma_copy` / 普通 SIMT loop——同一行源码在 IR 后期分叉。

**Day 2 - TileOp 在 C++ 里的样子**

读两段：

```127:158:src/op/operator.h
class TileOperatorNode : public Object {
public:
  virtual Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const = 0;

  virtual LayoutMap InferLayout(const LayoutInferArgs &T,
                                InferLevel level) const = 0;

  virtual TileOperator Clone() const = 0;

  virtual AccessRegions GetAccessRegions() const {
```

这是整个 tile-level IR 抽象的基类。任何 `T.copy / T.gemm / T.reduce / T.fill / T.atomic_*` 都对应一个 `TileOperatorNode` 子类，每个子类只对两件事负责：

1. **`InferLayout`**：告诉系统「这条 op 对它访问的 buffer 期望什么样的 Layout/Fragment」。
2. **`Lower`**：在 Layout 已经确定之后，把自己降成更低层的 TIR（loops、ptx_mma、cp.async 等）。

再看 GEMM 的 Lower 实际上是怎么走的：

```180:223:src/op/gemm.cc
Stmt GemmNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  if (const auto f = ffi::Function::GetGlobal("tl.gemm.lower")) {
    PrimExpr mbar_phase = T.mbar_phase_expr;
    if (auto explicit_phase = GetAnnotatedMbarPhaseExpr(annotations_)) {
      mbar_phase = explicit_phase.value();
    }
    // NOTE(wt): Decide the instruction key and compute warp partition on Python
    // side.
    auto prim_func = Downcast<PrimFunc>(
        (*f)(tvm::ffi::GetRef<Gemm>(this), T.layout_map, T.target,
             T.thread_bounds, T.thread_var, mbar_phase));
```

注意：C++ 的 GemmNode 自己**几乎不做任何 PTX 选择**，它把整个 lowering 通过 FFI 调用到 Python (`tl.gemm.lower`)。所以阅读路径是：

C++ `GemmNode::Lower`
→ Python `tilelang.tileop.gemm.registry.resolve_gemm_impl(gemm_inst, target)`
→ Python `GemmMMA / GemmWGMMA / GemmTCGEN05 / GemmMMA_SM70` 之一
→ 各自的 `TensorCoreIntrinEmitter` 真正生成 `ptx_mma / wgmma / tcgen05.mma` 序列。

读这段就能看到 Python 侧的实际 lowering：

```40:67:tilelang/cuda/op/gemm/gemm_mma.py
    def infer_layout(self, target: Target, thread_nums: int):
        mma_emitter = self._make_mma_emitter(target, thread_nums)
        if self.is_gemm_ss():
            return {
                self.A: make_swizzled_layout(self.A),
                self.B: make_swizzled_layout(self.B),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
        elif self.is_gemm_sr():
            return {
                self.A: make_swizzled_layout(self.A),
                self.B: mma_emitter.make_mma_load_layout(self.B, matrix="B"),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
```

这里你能直接看到「为什么 `T.gemm` 需要 `A` 在 shared、`C` 在 fragment」——因为 `infer_layout` 是根据 SS / SR / RS / RR 分类来返回 layout 字典的。

**Day 3 - Layout / Fragment 抽象**

读这两段，搞清楚 Layout 和 Fragment 是什么数学对象：

```118:127:src/layout/layout.h
class Layout : public ObjectRef {
public:
  TVM_DLL Layout(Array<IterVar> forward_var, Array<PrimExpr> forward_index);
  TVM_DLL Layout(Array<PrimExpr> input_size, Array<PrimExpr> forward_index);
```

```129:188:src/layout/layout.h
class FragmentNode : public LayoutNode {
public:
  FragmentNode() = default;
  FragmentNode(Array<PrimExpr> input_size, Array<PrimExpr> forward_index,
               PrimExpr forward_thread, PrimExpr replicate_size);

  PrimExpr GetForwardThread() const { return forward_thread_; }
```

精确的心智模型：

| 概念 | 数学对象 | 含义 |
| --- | --- | --- |
| `Layout` | `f : 逻辑坐标 → 存储坐标` | shared/global 上数据怎么排（含 swizzle / padding） |
| `Fragment` | `g : 逻辑坐标 → (存储坐标, thread 坐标, replicate)` | register 上每个 thread 持有哪些逻辑坐标的副本 |

记一句口诀：**Layout 描述空间布局，Fragment 描述空间布局 + 线程分布**。Fragment 继承自 Layout，多了一个 `forward_thread`。

实验（注意 `Fragment.forward_fn` 的返回值顺序是 `(forward_thread, forward_index)`，不是反过来的；见 `tilelang/layout/fragment.py` 的构造函数）：

```python
import tilelang  # noqa: F401  # 触发 tilelang lib 加载
from tilelang.layout import Layout, Fragment, make_gemm_fragment_8x8

# 1) 最简单的 Layout：128x128 数据，按 row-major 平铺
L = Layout([128, 128], lambda i, j: i * 128 + j)
print("input_shape  =", list(L.get_input_shape()))
print("output_shape =", list(L.get_output_shape()))
print("forward_idx  =", L.get_forward_index())

# 2) 手写一个最朴素的 Fragment：4x4 个逻辑元素摊给 4 个 thread
#    forward_fn(*vars) -> (forward_thread, forward_index)
F = Fragment([4, 4], forward_fn=lambda i, j: (i, j))
print("thread expr  =", F.thread)             # forward_thread = i
print("thread size  =", F.get_thread_size())  # 共 4 个 thread
print("index expr   =", F.get_forward_index())  # forward_index = j

# 3) 真正用在 Tensor Core ldmatrix/stmatrix 上的 8x8 fragment
F_tc = make_gemm_fragment_8x8()
print("TC fragment  =", F_tc)
```

再去看实际 GEMM 用的 fragment 构造函数：

```222:269:src/layout/layout.h
Fragment makeGemmFragmentC(const int block_m, const int block_n,
                           const int warp_m, const int warp_n,
                           const int element_size);
```

`makeGemmFragmentA/B/C` 在 `src/layout/gemm_layouts.cc` 里展开成具体的 forward_index/forward_thread 表达式——这部分配 Volta/Ampere/Hopper/Blackwell 有不同实现，是阅读 TileLang「为什么不同 SM 走不同路径」的关键。

**Day 4 - Layout Inference Pass**

读这段，看 BFS 是怎么把 layout 从 producer 传到 consumer 的：

```98:103:src/transform/layout_inference.cc
struct LayoutInferenceResult {
  Map<Buffer, Layout> layout_map;
  Map<For, Fragment> for_map;
  Map<For, PrimExpr> predicate_map;
};
```

核心算法（三级 infer）：

```64:82:src/op/operator.h
enum class InferLevel : uint8_t {
  kFree = 0,
  kCommon = 1,
  kStrict = 2,
};
```

理解为「每一轮 BFS 都用同一级 InferLevel 对当前队列的所有 op 调 `InferLayout`，从 kStrict 开始（只有真正强制要求 layout 的 op 才返回结果，比如 `T.gemm`），传播完后再退到 kCommon、最后 kFree」。这是为什么手写一个不带 layout 注解的 SIMT loop，也能自动推出 fragment layout。

实验：自己写两个 kernel，一个有 `T.gemm`、一个只有 `T.copy + T.Parallel`，然后启用 layout 可视化。`tilelang.compile` 在 pip 版本里就有 `pass_configs=` 参数，不需要手动开 `PassContext`。

Mac 本地版本（target="metal"，SIMT 累加，纯 IR + Metal codegen，没有 nvcc 依赖）：

```python
import tilelang
import tilelang.language as T


@T.prim_func
def my_func(
    A: T.Tensor((128, 128), "float32"),
    B: T.Tensor((128, 128), "float32"),
    C: T.Tensor((128, 128), "float32"),
):
    with T.Kernel(2, 2, threads=128) as (bx, by):
        A_s = T.alloc_shared((64, 32), "float32", scope="shared")
        B_s = T.alloc_shared((32, 64), "float32", scope="shared")
        C_l = T.alloc_fragment((64, 64), "float32")
        T.clear(C_l)
        for ko in T.Pipelined(4, num_stages=0):
            T.copy(A[by * 64, ko * 32], A_s)
            T.copy(B[ko * 32, bx * 64], B_s)
            for i, j in T.Parallel(64, 64):
                for k in T.Serial(32):
                    C_l[i, j] += A_s[i, k] * B_s[k, j]
        T.copy(C_l, C[by * 64, bx * 64])


kernel = tilelang.compile(
    my_func,
    target="metal",
    pass_configs={
        "tl.layout_visualization_enable": True,
        "tl.layout_visualization_formats": "txt",
    },
)
print(kernel.get_kernel_source()[:512])  # 看一眼 MSL 头
```

CUDA 环境（USE_CUDA=ON 的机器）上的版本，主要差别是把 target 换成 `"cuda -arch=sm_80"`、kernel 里用 `T.gemm`：

```python
@T.prim_func
def my_func(
    A: T.Tensor((128, 128), "float16"),
    B: T.Tensor((128, 128), "float16"),
    C: T.Tensor((128, 128), "float16"),
):
    with T.Kernel(1, threads=128):
        A_s = T.alloc_shared((128, 128), "float16")
        B_s = T.alloc_shared((128, 128), "float16")
        C_l = T.alloc_fragment((128, 128), "float16")
        T.copy(A, A_s); T.copy(B, B_s); T.clear(C_l)
        T.gemm(A_s, B_s, C_l)
        T.copy(C_l, C)

kernel = tilelang.compile(
    my_func, target="cuda -arch=sm_80",
    pass_configs={
        "tl.layout_visualization_enable": True,
        "tl.layout_visualization_formats": "txt",
    },
)
```

跑完后看 `~/.tilelang/cache/...` 下的 `.txt` layout 可视化文件，对照 `Fragment` 的 `forward_thread` 表达式。Metal 版本只能看到 `T.copy` / `T.Parallel` 推出的 fragment；要看 `T.gemm` 强约束推出的 fragment 必须走 CUDA 路径。

**Day 5 - LowerTileOp 与 makeBufferWithLayout**

读这段：

```36:83:src/transform/lower_tile_op.cc
static Buffer makeBufferWithLayout(const Buffer &buffer, const Layout &layout,
                                   Map<Var, Var> &var_remap) {
  const auto *ptr_type =
      TVM_TYPE_AS(buffer->data->type_annotation, PointerTypeNode);
  Type new_type;
  // convert fragments to normal local buffer
  if (IsFragmentBuffer(buffer)) {
    new_type = PointerType(ptr_type->element_type, "local");
```

这段是 `local.fragment → local` 的转换，也是 register tile 真正落到 SSA / `__local__` 数组的地方。读懂之后你就明白：

- 用户写 `T.alloc_fragment((128, 128), "float16")` 时拿到的是 128×128 个**逻辑元素**；
- LowerTileOp 之后会变成每个 thread 持有 `128 * 128 / threads` 个寄存器元素；
- 这个转换完全由前面 `LayoutInference` 推出的 Fragment 决定。

**Day 6-7 - 复盘 + 写读书笔记**

任务：写一份「`T.gemm` 从 DSL 到 PTX 的全链路」笔记，至少回答下面五个问题。

1. 我写 `T.gemm(A_s, B_s, C_l)` 时，DSL 层做了什么？（→ 参数 check + `tir.call_intrin("tl.tileop.gemm", ...)`）
2. 这个 intrinsic 在哪一步被认为是 TileOperator？（→ `LowerTileOp` 里 `ParseOperator(call)` 把它构造成 `GemmNode`，见 `src/op/operator.h`）
3. `GemmNode` 的 `InferLayout` 返回了哪些 buffer 的 layout？（→ 答案在 `tilelang/cuda/op/gemm/gemm_mma.py::infer_layout`）
4. `GemmNode::Lower` 通过哪个 FFI 函数回到 Python？（→ `"tl.gemm.lower"`，在 `tilelang/tileop/gemm/__init__.py` 注册）
5. 在 sm_80 vs sm_90 上为什么走不同 lowering？（→ `tilelang.tileop.gemm.registry.resolve_gemm_impl(gemm_inst, target)`，`gemm_inst` 来自 C++ `ResolveGemmImpl(target).select_inst(...)`，target 不同就匹配不同 `GemmImpl`）

验收标准：远端 CUDA 机器上跑通

```bash
python -m pytest testing/python/kernel/test_tilelang_kernel_gemm.py -x
```

并能解释其中至少 3 个 test case 的 lowering 链路。

---

### 第 4 周：Layout / Fragment 深入 + 端到端跑通一个 regression test

主题：从「能解释路径」过渡到「能修改它，并提供 regression test」。

**Day 1-2 - 三种 GEMM 变体 (SS/SR/RS) 的层次对比**

读：

```39:57:tilelang/tileop/gemm/gemm_base.py
    def is_gemm_ss(self) -> bool:
        """Return True if both A and B are in shared memory (SS variant)."""
        return is_shared(self.A) and is_shared(self.B)

    def is_gemm_sr(self) -> bool:
        """Return True if A is in shared memory and B is in registers (SR variant)."""
        return is_shared(self.A) and is_fragment(self.B)
```

并且自己各写一个 SS / SR / RS 的 kernel（A、B 一个在 shared 一个在 fragment 等组合），看：

- 生成的 CUDA 源代码差异（`kernel.get_kernel_source()`）。
- `make_mma_load_layout / make_mma_store_layout` 实际生成的 Fragment forward_thread 表达式。
- 在 Hopper target 上跑同样代码，会切到 `gemm_wgmma.py` 而不是 `gemm_mma.py`。

**Day 3 - Layout 冲突场景**

写一个会触发 layout 冲突的 kernel（比如同一个 shared buffer 被两个 `T.gemm` 以不同 transpose 消费），把异常栈拉到 `src/transform/layout_inference.cc:268` 的 `LOG(FATAL) << "Get different layout"` 那一句，理解错误信息的产生路径。

这是非常好的 PR 选题：把这一行错误信息从

```
Get different layout for buf_A
 current layout: ...
 previous layout: ...
```

改成

```
Layout conflict on buf_A while inferring T.gemm at <file:line>:
  expected: ...
  inferred earlier by: <op_kind> at <file:line>
  reason: incompatible thread/element mapping for matrices of dtype=<>, transpose=<>
```

（这个 PR 不需要硬件验证，Mac 上能完整测试。）

**Day 4 - WGMMA / TCGEN05 路径区分**

只读不写。重点对照表：

| 变量 | sm_80 (Ampere) | sm_90 (Hopper) | sm_100 (Blackwell) |
| --- | --- | --- | --- |
| GEMM 类 | `GemmMMA` | `GemmWGMMA` | `GemmTCGEN05` |
| 指令 | `mma.sync.aligned` | `wgmma.mma_async` | `tcgen05.mma` |
| C 存储 | register fragment | register fragment | TMEM (`alloc_tmem`) |
| 同步 | implicit barrier | `wgmma.wait_group` | `mbarrier_wait_parity` |
| Pass 触发 | 无 | `producer_consumer_ws.cc` + `inject_pipeline.cc` | `lower_blackwell_2sm.cc` + `inject_tcgen05_fence.cc` |

把 `tilelang/cuda/op/gemm/gemm_wgmma.py` 和 `tilelang/cuda/op/gemm/gemm_tcgen05.py` 与 `gemm_mma.py` 对照读一遍，写一份「三代 GEMM 在 TileLang 里如何分叉」的笔记。

**Day 5-7 - 完成第一份可 PR 的工件**

选项 A（推荐）：选一个 `testing/python/kernel/test_tilelang_kernel_gemm.py` 里没覆盖的 corner case（例如 transpose 组合 / 非典型 tile shape / dtype），加一个 minimal regression test。

选项 B：在 `testing/python/transform/test_tilelang_transform_layout_inference.py` 里补一个 layout 冲突的 `pytest.raises` test，确保上面 Day 3 的错误路径以后有 regression 保护。

不论选哪个，验收：

- Mac 上 `python -m pytest <你的测试文件> -x` 走过 codegen 路径（即使运行被 skip）；
- 远端 CUDA 机器上跑同样测试，能确实跑出结果；
- `pre-commit run --all-files` 全绿。

---

### 第 5 周：Pass Pipeline（从 compiler 视角）

主题：把 `tilelang/engine/phase.py` 里的 30 多个 pass 排出来，理解每个 pass 解决什么问题。

精读这两个函数（已经在文档里贴过名字，但这次按一条线读）：

```144:224:tilelang/engine/phase.py
def LowerAndLegalize(mod: IRModule, target: Target) -> IRModule:
```

```227:310:tilelang/engine/phase.py
def OptimizeForTarget(mod: IRModule, target: Target) -> IRModule:
```

阅读顺序建议（按依赖链分组，不是文件顺序）：

| 分组 | Pass | 你需要理解的事 |
| --- | --- | --- |
| **入口** | `BindTarget` | Target 信息怎么挂到 PrimFunc.attrs |
| **DSL 规整** | `LegalizeNegativeIndex` + `InjectAssumes` + `VerifyParallelLoop` | 把 frontend 写法规整成 TVM 友好形态 |
| **流水线** | `ProducerConsumerWarpSpecialized` + `PipelinePlanning` + `InjectSoftwarePipeline` | WS / 软件流水的核心 3 个 pass |
| **Layout** | `LayoutReducer` + `LayoutInference` | reducer layout 优先确定，再走全图推 |
| **Lower** | `LowerTileOp` + `LowerL2Persistent` | TileOp → 普通 TIR |
| **Safety** | `LegalizeSafeMemoryAccess` + `LegalizeVectorizedLoop` + `DecoupleTypeCast` | 边界条件 / OOB / dtype 处理 |
| **Buffer / Memory** | `PlanAndUpdateBufferAllocationLocation` + `FlattenBuffer` + `StorageRewrite` + `MergeSharedMemoryAllocations` | shared memory 复用、寄存器 packing |
| **Sync** | `ThreadSync("shared")` ×2 + `InjectFenceProxy` + `InjectTcgen05Fence` | 屏障 / fence |
| **Codegen 前** | `SplitHostDevice` + `MakePackedAPI` + `LowerDeviceKernelLaunch` + `PersistThreadblock` | host/device 分离 + 持久化 |

可行的实验：在 `LowerAndLegalize` 每个 pass 之后插一个 `print(mod.script())` 临时调试（fork 一个本地 branch 改源码即可），跑一个 32×32 的 matmul，记录 IR 在哪一步「形变」最大。

第 5 周的产出目标：

- 找到 `testing/python/transform/test_tilelang_transform_layout_inference.py` 或 `test_tilelang_transform_pipeline_planning.py` 里某个测试只 assert「编译不挂」的地方，加一条更精确的断言（例如断言生成的 IR 里 `cp.async` 出现次数 / `mbarrier_arrive` 在哪个 stage 触发）。这是非常容易合入的 PR 类型。

---

### 第 6 周：JIT / Adapter / Target

主题：搞清楚 `@tilelang.jit` 装饰器之后到底发生了什么；不同 execution_backend 的路径分叉。

读这一条调用链：

1. `@tilelang.jit` → `tilelang.jit.compile(func, ...)`（`tilelang/jit/__init__.py`）
2. → `tilelang.cache.cached(...)`（决定是不是 reuse 之前编好的 `.so` / cubin）
3. → `tilelang.engine.lower.lower(...)`（执行 phase.py 两条 pipeline）
4. → `tilelang/jit/adapter/{tvm_ffi,nvrtc,cython,torch,cutedsl}/*` 之一（用 lower 结果包成可调 Python 对象）

Mac 上能完整实验的部分（注意 Metal 上没有 tensor core，所以 kernel 里**不能写 `T.gemm`**，要用手写的 SIMT 累加循环；下面这段在 `pip install tilelang` + macOS / Apple Silicon 上能直接跑通）：

```python
import tilelang
import tilelang.language as T


def make_matmul_kernel(M=128, N=128, K=128, block_M=64, block_N=64, block_K=32):
    dtype = T.float32

    @T.prim_func
    def matmul_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), dtype)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                for i, j in T.Parallel(block_M, block_N):
                    for k in T.Serial(block_K):
                        C_local[i, j] += A_shared[i, k] * B_shared[k, j]
            T.copy(C_local, C[by * block_M, bx * block_N])

    return matmul_kernel


my_func = make_matmul_kernel()

# 对比两个 execution_backend 在 Metal 上的产物
k1 = tilelang.compile(my_func, target="metal", execution_backend="tvm_ffi")
k2 = tilelang.compile(my_func, target="metal", execution_backend="torch")
# 外层 JITKernel 是同一个壳子，差别在内部 adapter 上
print("source equal :", k1.get_kernel_source() == k2.get_kernel_source())  # True
print("k1.adapter   :", type(k1.adapter).__name__)  # TVMFFIKernelAdapter
print("k2.adapter   :", type(k2.adapter).__name__)  # MetalKernelAdapter
```

PR 选题候选：

- 把 `tilelang/jit/execution_backend.py::allowed_backends_for_target` 的错误信息改成「当前 target=<>, 当前 backend=<>, 允许的 backend 是 <list>」（现在的错误信息有时漏掉 target 名）。
- 给 `determine_target` 加 Mac 上的 MPS 探测优先级测试。

---

### 第 7 周开始：独立 PR

`TILELANG_AI_INFRA_LEARNING_PLAN.md` 里第 7-8 周的部分已经写得比较细了，这里只做一个补充的「具体可挑」清单。基于你目前已经过 3-6 周的能力梯度，下面这些 PR 对你性价比最高：

1. **错误信息类**（推荐先做）
   - 上面提到的 `LowerTileOp` / `LayoutInference` 冲突错误。
   - `tilelang/jit/execution_backend.py` 中 `allowed_backends_for_target` 的报错。
2. **回归测试类**
   - 某个 `T.copy` 的 strided + boundary case。
   - 某个 GEMM 的 transpose × dtype 组合。
3. **小型 transform 类**
   - 给 `LegalizeVectorizedLoop` / `LegalizeSafeMemoryAccess` 添加边界 case 的单元测试。
4. **文档类**
   - `docs/get_started/Installation.md` 补 Mac 流程。
   - `docs/get_started/targets.md` 补 `auto` selection 优先级。

不建议第一批触碰：`MergeSharedMemoryAllocations`、`ProducerConsumerWarpSpecialized`、`PipelinePlanning` 这三个 pass 牵连最大，调试代价远高于审阅门槛。

---

## 二、源码核心设计与抽象

下面这部分是**独立讲解**，可以脱离学习计划单独食用。我把 TileLang 当作「一个用 TVM 作为底座的 tile-level compiler」拆成 9 个抽象层来看。

### 2.1 全栈分层（一定要先建立这张图）

```text
┌─────────────────────────────────────────────────────────────────┐
│ Python DSL 层  (tilelang/language/*)                            │
│   - T.Kernel / T.copy / T.gemm / T.alloc_* / T.Parallel ...     │
│   - 本质：写一段 @T.prim_func，里头调用 tir.call_intrin            │
└─────────────────────────────────────────────────────────────────┘
                       │ tvmscript 解析
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ TIR PrimFunc + TileOp Intrinsics  (混合 IR)                      │
│   tir.For + tir.BlockRealize + tir.call_intrin("tl.tileop.gemm")│
└─────────────────────────────────────────────────────────────────┘
                       │ LowerAndLegalize  (tilelang/engine/phase.py)
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ TileOperator IR  (src/op/*)                                     │
│   GemmNode / CopyNode / ParallelOp / ReduceNode / ...           │
│   每个都有 InferLayout + Lower 两个虚函数                          │
└─────────────────────────────────────────────────────────────────┘
                       │ LayoutInference → LowerTileOp
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ 普通 TIR (loops, BufferLoad/Store, mma/cp.async/tma intrinsic)  │
└─────────────────────────────────────────────────────────────────┘
                       │ OptimizeForTarget → SplitHostDevice
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│ Host TIR ─────┬───── Device TIR                                  │
│               │                                                  │
│        target.build.* (LLVM / nvcc / hipcc / metal / webgpu)    │
│               │                                                  │
│        Adapter (tvm_ffi / cython / nvrtc / torch / cutedsl)     │
└─────────────────────────────────────────────────────────────────┘
```

读源码时的关键事实：**任何 `T.xxx` 都不会直接生成 CUDA 代码。它们先变成 TileOperator IR，让 layout 推断在 IR 上跑完，再 Lower 成普通 TIR，最后才进 TVM 后端**。这是 TileLang 区别于 Triton 的根本设计。

### 2.2 TileOperator —— 最重要的双接口抽象

`TileOperatorNode`（`src/op/operator.h:127-152`）只暴露两个虚函数：

```cpp
virtual Stmt    Lower      (const LowerArgs &T,       arith::Analyzer *analyzer) const = 0;
virtual LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) const = 0;
```

这是 TileLang 的**核心契约**。每一个 `T.xxx` 在 C++ 层都对应一个 `TileOperatorNode` 子类：

| Python DSL | TileOperator 子类 | 文件 |
| --- | --- | --- |
| `T.gemm / T.wgmma_gemm / T.tcgen05_gemm` | `GemmNode` | `src/op/gemm.cc` |
| `T.gemm_sp` | `GemmSPNode` | `src/op/gemm_sp.cc` |
| `T.copy / T.async_copy / T.tma_copy` | `CopyNode` | `src/op/copy.cc` |
| `T.c2d_im2col` | `Conv2DIm2ColOpNode` | `src/op/copy.cc` |
| `T.Parallel(...)` 体 | `ParallelOp` | `src/op/parallel.cc` |
| `T.reduce_*` | `ReduceNode` | `src/op/reduce.cc` |
| `T.fill / T.clear` | `FillNode` | `src/op/fill.cc` |
| `T.atomic_*` | `AtomicAddNode / AtomicReduceNode` | `src/op/atomic_*.cc` |
| `T.region(...)` | `RegionOp` | `src/op/region.cc` |

理解 TileOperator 的关键观察：

1. **InferLayout 不修改 IR**，只把推断结果写回到 `LayoutMap`。
2. **Lower 不做 layout 决策**，它读 `LowerArgs::layout_map` 当输入。这意味着 layout inference 必须完整跑完再进 lowering。
3. **真正后端选择不在 TileOperator 里**。例如 `GemmNode::Lower` 只是 `ffi::Function::GetGlobal("tl.gemm.lower")` 派发到 Python，再由 Python registry 选 `GemmMMA / GemmWGMMA / GemmTCGEN05`。这种「C++ 节点 + Python registry」的设计让加新后端不用动 C++。
4. **`access_regions_` 是 op 的副作用声明**，让 TIR scheduler 知道哪些 buffer 被读 / 写。

阅读建议：第一次读源码不要读 `gemm.cc / copy.cc` 的实现，而是先读 `operator.h` 和 `parallel.h`，把契约本身搞清楚。

### 2.3 Layout / Fragment —— 索引变换的核心数据结构

```cpp
// 简化版心智模型
Layout   : forward_index   = f(i0, i1, ..., in-1)
Fragment : forward_index   = f(i0, ..., in-1, rep)
           forward_thread  = g(i0, ..., in-1, rep)
```

- `Layout` 是「一个 N 维逻辑索引 → 一个或多个存储索引」的纯函数变换。它表达「这个 buffer 在存储里到底怎么排」，例如 swizzle 之后某个元素的物理 offset。
- `Fragment` 多了 `forward_thread`：每个逻辑元素**还会被映射到一个或几个 thread 上**。`replicate_size` 表示同一个逻辑元素被多少个 thread 同时持有（对 broadcast / accumulator init 很关键）。

记三个关键操作：

- `Layout::Inverse()`：根据存储 offset 反推逻辑坐标，是 `LowerTileOp` 重写 BufferLoad/Store 的核心工具。
- `Fragment::BindThreadRange(...)`：把 thread 维度 bind 到具体的 `[tx_lo, tx_hi)` 区间，绑定之后才能在 lowering 里产生真正的 `if (tid >= ...)` 代码。
- `Layout::Reshape(..., rescale_num, rescale_den)`：处理 fp32→i8 这种**同一物理存储不同 dtype alias** 的场景（在 quantize 路径里非常常见）。

具体的 GEMM fragment 构造在 `src/layout/gemm_layouts.cc` 里。第一次读不必逐个看，但要意识到 `makeGemmFragmentA / makeGemmFragmentB / makeGemmFragmentC` 是按 Ampere 的 mma.sync 排列规则手动算出来的——这就是「为什么 TileLang 编出来的 ldmatrix / stmatrix 模式总是和 cutlass 一致」。

### 2.4 LowerArgs / LayoutInferArgs —— 两个上下文对象

`src/op/operator.h:84-123` 定义了两个最重要的上下文 struct。读懂它们就知道每个 TileOperator 的虚函数能拿到什么：

```cpp
struct LowerArgs {
  Target target;                                  // 当前编译 target
  Range thread_bounds;                            // 当前 block 内的 thread 范围
  Var thread_var;                                 // threadIdx.x 对应的 Var
  AddWorkspaceCallback AddWorkspace;              // 临时申请 global workspace
  AllocMBarrierCallback AllocMBarrier;            // 在 shared.barrier 里申请 mbarrier
  LayoutMap layout_map;                           // 上一阶段推完的所有 buffer layout
  Map<Buffer, Buffer> buffer_remap;               // fragment buffer 已被改名 (data var)
  Map<Var, PrimExpr> let_var_to_expr;             // LetStmt 展开表
  PrimExpr mbar_phase_expr;                       // mbarrier phase
  Optional<Buffer> *mbarrier_buffer;              // shared.barrier 指针
  int cluster_size;                               // cluster dim 乘积
};

struct LayoutInferArgs {
  Target target;
  Range thread_bounds;
  LayoutMap layout_map;
  arith::Analyzer *analyzer;
  bool buffer_oob;
  Map<Buffer, Buffer> buffer_remap;
  Map<Var, PrimExpr> let_var_to_expr;
  bool in_pipeline;                               // 是否在 num_stages > 0 的 loop 里
};
```

这两个 struct 是 TileLang 增量演化的「上下文增长点」。每次后端加新功能（cluster、TMA、PDL、blockscaled MMA），通常做法就是往这俩 struct 里加字段，而不是改 TileOperator 接口本身。

### 2.5 Pass Pipeline 的两阶段切分

`tilelang/engine/phase.py` 把整条 pipeline 切成两段：

- **`LowerAndLegalize`**（target-agnostic 大部分）：负责消除 TileOp、推 layout、规整 vectorize / OOB。这段结束后，IR 已经是「普通 TIR」，几乎没有 `tl.tileop.*` 残留。
- **`OptimizeForTarget`**（target-aware）：负责 buffer flatten、storage rewrite、thread sync、host/device split、packed API。这段后期会用到 Target 的 ISA 信息（如 `have_tma(target)`、`have_pdl(target)`）。

把 `phase.py` 当成「TileLang 的 release 顺序合约」来读：所有新 pass 必须明确「插在哪两个 pass 之间」。学完之后你就明白为什么有些 pass 顺序看起来奇怪：

- `MergeSharedMemoryAllocations` 必须在 `SplitHostDevice` 之后（不然 host 里也会被错误合并）。
- `LowerThreadAllreduce` 必须在 `Simplify` 之前（var binding 信息在 Legalize 后会丢）。
- `InjectTcgen05Fence` 必须在 `ThreadSync` 之后（要看见 `tvm_storage_sync`）。

### 2.6 Layout Inference 的三级 BFS 算法

`src/transform/layout_inference.cc` 是 TileLang 里**最反 TVM 直觉**的一段代码。它不是单向 lowering，而是一个**带优先级的 BFS 不动点**。

```cpp
enum class InferLevel : uint8_t {
  kStrict = 2,   // 只有 hard-constraint op 返回 (T.gemm 这种)
  kCommon = 1,   // 合理的默认推断 (T.copy 沿着 producer/consumer 传)
  kFree   = 0,   // 谁都没意见时给一个 default layout (Parallel SIMT loop)
};
```

执行顺序：

1. 用 kStrict 跑一遍所有 op，得到「绝对不能动」的 layout 集合 `strict_layout_map`。
2. 用 kCommon 跑 BFS，让 strict 推出的 layout 传播给 Copy / Parallel 这类**有偏好但不强制**的 op。
3. 对仍未确定的 op，逐个用 kFree 强制给一个 default layout，继续传播。

这个算法是 TileLang「用户可以不写 layout 注解，但写了也能尊重」的根本原因。

读它的时候，理解三件事：

- **`use_list_`**：buffer → 哪些 op 用它（双向图）。
- **`EnqueueWithPriority(...)`**：layout 一旦更新就把这些 op 重新塞回队列。
- **alias propagation**：同一个 `data` 但不同 `dtype/shape` 的 buffer（fp32↔fp16 view）也要联动更新。

### 2.7 内存 Scope 系统

TileLang 在 TVM 原生 scope 之上扩展了一套自己的 scope 名字，写代码时要严格区分：

| scope | 物理位置 | 谁能访问 | 在 IR 里的形态 |
| --- | --- | --- | --- |
| `global` | DRAM | host + device | tensor params 默认 |
| `shared` / `shared.dyn` | SMEM | block 内所有 thread | `T.alloc_shared` |
| `shared.tmem` | Blackwell TMEM | warp group | `T.alloc_tmem` |
| `shared.barrier` | SMEM 上的 mbarrier 表 | block 内所有 thread | `T.alloc_barrier / alloc_cluster_barrier` |
| `local` | register | 单 thread | `T.alloc_local` |
| `local.fragment` | register（Layout 由 inference 决定） | block 内多 thread 协作（MMA fragment） | `T.alloc_fragment` |
| `local.var` | register（标量） | 单 thread | `T.alloc_var` |

`shared.dyn` vs `shared` 的关键区别：`shared.dyn` 由 `MergeSharedMemoryAllocations` 重用 offset，可以多个 buffer 共享同一片 SMEM；`shared` 不参与合并。debug 时如果遇到 SMEM 溢出，第一反应是看 buffer scope 是不是写错成 `shared`。

`local.fragment` 是 TileLang 自己引入的，TVM 原生没有。它的 lowering 由 `LowerTileOp` 的 `makeBufferWithLayout` 完成——本质是「在 layout 推断完之后，把 fragment 改写成 thread-private 的 local 数组」。

### 2.8 Target 后端注册表模式

GEMM 和 Copy 在 C++ 都用同一种模式：

```cpp
// src/op/gemm.h
struct GemmImpl {
  const char *name;
  GemmTargetPredicate match_target;          // 哪个 target 命中
  String (*select_inst)(...);                // 具体选哪条 ISA (cuda.mma / cuda.wgmma / cuda.tcgen05 / hip.mfma)
  std::pair<int,int> (*compute_warp_partition)(...);
  bool (*reuse_existing_shared_layout)(String gemm_inst);
  String (*instruction_kind)(String gemm_inst);
};
void RegisterGemmImpl(GemmImpl impl);   // 加新后端只用调它
```

实际注册点：

- `src/backend/cuda/op/gemm.cc` 注册 CUDA 的 mma/wgmma/tcgen05。
- `src/backend/rocm/op/gemm.cc` 注册 HIP 的 mfma/wmma。
- `src/backend/metal/op/gemm.cc` 注册 Metal 的 simdgroup matmul。

注意：C++ 的 `GemmImpl` 只决定 **指令选择 + warp 分区**。真正的 IR 生成代码已经被 punt 到 Python（见 2.2 中提到的 `tl.gemm.lower` FFI 调用）。这种**「C++ 选指令、Python 生成代码」**的拆分是 TileLang 后端可扩展性的关键。

加新后端时 typical 路径：

1. C++ 写一个新的 `GemmImpl` 并 `RegisterGemmImpl`。
2. Python 写一个新的 `GemmBase` 子类，实现 `infer_layout` 和 `lower`。
3. Python 调用 `register_gemm_impl(name, inst_name, predicate, impl_class)` 在 registry 注册。

Copy 完全是同样的套路（`CopyImpl + RegisterCopyImpl`）。

### 2.9 JIT Adapter / Execution Backend

最外层的「kernel 怎么被 Python 调起来」抽象在 `tilelang/jit/adapter/base.py`：

```python
class BaseKernelAdapter(ABC):
    @abstractmethod
    def _convert_torch_func(self) -> callable: ...
```

各 backend 子类负责把 `CompiledArtifact (host_mod + device_mod + params)` 包成可调对象：

- `tvm_ffi`：用 TVM 自己的 PackedFunc + DLPack（默认，性能稳定，所有 target 通用）。
- `cython`：编一个 Cython 包装层调 ctypes，启动开销最低。
- `nvrtc`：跳过 TVM 的 host 代码生成，直接用 NVRTC 编译 device 部分（cubin），host 由 cython 包。
- `torch`：直接接 PyTorch tensor，主要用在 Metal 上和 MPS 集成。
- `cutedsl`：CUDA only，用 CUTLASS Python 端的 cute dsl 作为编译器（不走 TVM 的 device codegen）。

`allowed_backends_for_target`（`tilelang/jit/execution_backend.py`）是一张**target × backend 的兼容矩阵**，每次新加 target 或 backend 都要更新它。

### 2.10 Python ↔ C++ FFI 协作模式

最后一个值得单独点出的设计点：TileLang 的 C++ 代码大量使用 `ffi::Function::GetGlobal("...")` 把工作甩到 Python。例子：

| C++ 调用 | Python 实现位置 | 干什么 |
| --- | --- | --- |
| `"tl.gemm.lower"` | `tilelang/tileop/gemm/__init__.py` | GEMM IR 生成 |
| `"tl.gemm.infer_layout"` | 同上 | GEMM layout 推断 |
| `"tl.copy.lower"` | `tilelang/tileop/copy/...` | Copy 的具体 ISA 选择 |
| `"target.build.tilelang_cuda"` | TVM target build hook | 调 nvcc |
| `"tilelang_callback_cuda_compile"` | `tilelang/engine/lower.py` | 拼 nvcc 命令行 |

这种「C++ 占类型系统 + Python 写策略」的混合架构是 TileLang 区别于纯 TVM / 纯 Triton 的最大特征。在你阅读 C++ 代码时，看到 `ffi::Function::GetGlobal` 不要慌，**立刻去 Python 这边搜对应字符串**，路径会瞬间明朗。

---

## 三、给你的建议

1. **不要按文件夹顺序读源码**。按上面的 9 个抽象层切片来读，每一层都看一两个具体例子（`Gemm` + `Copy` 是最好的对照组）。
2. **抽象 → 实现 → 测试**三件套并读。每次找一个 TileOperator 子类：先看 `operator.h` 里的契约，再看子类实现，最后看 `testing/python/transform/` 或 `testing/python/kernel/` 里的对应测试。
3. **Mac 上能跑的部分尽量在本地跑**，把 `tilelang.transform.PassContext` 里的几个 debug 开关用熟，特别是：
   - `tl.layout_visualization_enable=True` 可视化 fragment。
   - `tl.ast_print_enable=True` 看 PreLowerSemanticCheck 阶段 AST。
   - `tl.disable_warp_specialized=True` 关掉 WS 对比基线。
4. **第一个 PR 优先选错误信息或测试**。这两类 PR 风险低、review 快、能逼你真正理解链路。
5. **GEMM 是绝佳的读源码 anchor**。所有抽象（TileOperator / Layout / Fragment / Registry / FFI / Adapter）在 GEMM 路径上都出现一次。把 `T.gemm` 链路读透，剩下 80% 的源码模式都是这条链路的变体。

如果你已经准备好开始第 3 周 Day 1，那就照上面 GEMM 路径走起来；过程中遇到具体源码看不懂的地方，告诉我哪一行，我可以帮你拆。

