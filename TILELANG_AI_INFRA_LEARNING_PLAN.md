# TileLang AI Infra Learning and Contribution Plan

> 目标读者：熟悉 CUDA 和 Triton，想进入 AI Infra / kernel compiler 方向，并以给 TileLang 贡献 PR 为近期目标。
>
> 本文基于本地仓库 `main` 分支、`VERSION=0.1.9`、commit `c2a9d987`（2026-05-21 校对）的源码、`docs/`、`CMakeLists.txt` 整理。NPU 方向的外部信息以 2026-05-15 查看 `tile-ai/tilelang-ascend` 官方仓库为参考。
>
> 文中所有路径、API、pass 名、测试文件名都已对照当前源码验证；如发现与上游有出入，以 `main` 分支为准。

## 1. 总体判断

TileLang 对你比较适合作为从 CUDA/Triton 过渡到 AI Infra 的学习项目，因为它处在三个层次的交界处：

- 上层是 Python DSL，写法比 TVM TIR 更接近 kernel 作者的思维。
- 中层是 TVM TIR pass pipeline，能看到 lowering、layout inference、software pipeline、memory flatten、thread sync 等 compiler infra 逻辑。
- 下层是 CUDA/ROCm/Metal 后端和 GEMM tile operator，对应 Tensor Core、WGMMA、TCGEN05、MFMA 等硬件指令。

如果目标是贡献 PR，不建议一开始就做国产 NPU 后端。更现实的路线是：

1. 先在主仓库熟悉 TileLang DSL、TIR lowering、测试体系。
2. 用 MacBook 做源码阅读、Python/C++ 修改、文档和大部分单元测试。
3. 租 CUDA/ROCm/Ascend 机器做 correctness、codegen 和性能验证。
4. 从小型、可验证、容易 review 的 PR 切入，再逐步进入 backend/operator/pass。

## 2. MacBook 是否适合

适合。MacBook 可以承担大部分开发工作，尤其是 Apple Silicon 机器还可以跑 Metal 后端的部分测试。

Mac 上适合做：

- 阅读和修改 Python DSL：`tilelang/language/*`
- 阅读和修改 JIT/target/cache/autotune：`tilelang/jit/*`、`tilelang/utils/*`、`tilelang/autotuner/*`
- 阅读和修改 pass / operator 逻辑：`src/transform/*`、`src/op/*`
- 写文档、examples、issue regression tests
- 跑 Metal codegen / runtime 的小规模测试：`testing/python/metal/*`
- 做 NPU/Ascend 方向的接口理解、文档整理、测试设计

Mac 上不适合作为最终验证环境：

- CUDA Tensor Core / WGMMA / TCGEN05 性能调优
- ROCm MFMA / WMMA correctness 和性能验证
- Ascend NPU 后端真实编译、运行、profile
- 任何依赖真实 NVIDIA/AMD/Ascend runtime 的 CI 级验证

推荐本地开发命令（macOS / Apple Silicon）：

```bash
python -m venv .venv
source .venv/bin/activate

# 构建依赖（cython、cmake、ninja、tvm-ffi、z3 等），不包含测试依赖
pip install -r requirements-dev.txt

# 在 Apple 上 CMakeLists.txt 默认就启用 Metal 后端，无需显式 -DUSE_METAL=ON；
# 这里把 generator 固定为 Ninja，给 ccache 留出缓存空间。
cmake -S . -B build -G Ninja
cmake --build build --parallel

# 让 Python 在仓库根目录就能 import 到本地 tilelang，并通过
# build/lib 加载新编出的 libtilelang.dylib（见 tilelang/env.py）。
export PYTHONPATH=$(pwd):$PYTHONPATH
python -c "import tilelang; print(tilelang.__version__)"

# 跑测试前再装一遍测试依赖；Metal 测试用 requirements-test-metal.txt。
pip install -r requirements-test-metal.txt
python -m pytest testing/python/metal/ -x
```

如果只想快速安装当前仓库（含完整构建链路）：

```bash
pip install . -v
```

依赖文件的分工（避免混淆）：

| 文件 | 用途 |
| --- | --- |
| `requirements.txt` | 运行时依赖（apache-tvm-ffi、torch、numpy、z3 等） |
| `requirements-dev.txt` | 构建依赖（cython、cmake、ninja、pre-commit、scikit-build-core）+ 运行时 |
| `requirements-lint.txt` | 格式化和 lint：`pre-commit`、`ruff`、`clang-format`、`codespell` |
| `requirements-test.txt` | 通用测试依赖（pytest、einops、pandas 等）|
| `requirements-test-cuda.txt` / `requirements-test-rocm.txt` / `requirements-test-metal.txt` | 各后端测试增量依赖 |

注意：本仓库开发时不要使用 `pip install -e .`。从仓库根目录运行 Python 时，当前目录已经会被放到 `sys.path`，editable install 反而容易和源码树本地的 `tilelang/` 包发生 import 混淆（仓库内置的 `.agents/skills/tilelang-build/SKILL.md` 也明确反对 editable install）。

> 注：上游 `docs/get_started/Installation.md` 中仍出现了 `pip install -e .` 的写法，那是为通用 Linux/Windows 场景准备的。对当前这套 Mac + 仓库内开发流，按照上面 `cmake + PYTHONPATH` 的写法最稳。

## 3. 先建立的源码地图

建议按这条线读，不要一开始钻进后端大文件：

1. `README.md`
   - 建立 TileLang 的 DSL 形态、典型 GEMM/attention 示例、支持设备范围。

2. `docs/get_started/Installation.md`
   - 看构建方式、CMake 选项、运行环境变量。

3. `docs/get_started/targets.md`
   - 理解 `cuda`、`hip`、`metal`、`llvm`、`cutedsl`、`webgpu` 等 target 字符串。

4. `tilelang/language/__init__.py`
   - DSL API 的总入口。除了 `T.Kernel`、`T.alloc_shared`、`T.alloc_fragment`、`T.copy`、`T.gemm`、`T.Pipelined`、`T.Parallel`，还要顺手扫一遍这些常用名字：
     - 内存：`alloc_local`、`alloc_global`、`alloc_var`、`alloc_barrier`、`alloc_cluster_barrier`、`alloc_tmem`、`alloc_reducer`、`alloc_wgmma_desc`、`alloc_tcgen05_smem_desc`、`alloc_tcgen05_instr_desc`
     - GEMM 家族：`gemm`、`wgmma_gemm`、`tcgen05_gemm`、`tcgen05_gemm_blockscaled`、`gemm_sp`
     - Copy 家族：`copy`、`async_copy`、`tma_copy`、`transpose`、`c2d_im2col`
     - 控制流：`Parallel`、`Pipelined`、`Persistent`、`Serial` / `serial`、`Unroll` / `unroll`、`Vectorized` / `vectorized`
     - Reduce / Atomic / Fill：`reduce_sum/max/min/abssum/absmax/bitand/bitor/bitxor`、`cumsum`、`finalize_reducer`、`warp_reduce_*`、`atomic_add/addx2/addx4/max/min/load/store`、`fill`、`clear`
     - 注解 / 调试：`annotate_layout`、`use_swizzle`、`annotate_safe_value`、`annotate_l2_hit_ratio`、`annotate_restrict_buffers`、`annotate_min_blocks_per_sm`、`print`、`device_assert`
     - Cluster / Warp 专门化：`ws`（warp specialize 入口）、`cluster_arrive`、`cluster_sync`、`block_rank_in_cluster`
     - PDL / Random：`pdl_trigger`、`pdl_sync`、`rng_init`、`rng_rand`

5. `tilelang/language/kernel.py`
   - 看 `Kernel`、`KernelLaunchFrame`、`get_thread_binding(s)`、`get_block_binding(s)`、`CUDASourceCodeKernel`，覆盖 launch grid、thread binding、cluster dims、raw CUDA source kernel 的封装。

6. `tilelang/language/allocate.py`
   - 看 TileLang 如何表达 shared memory、fragment/local memory、TMEM、barrier、wgmma/tcgen05 descriptor。

7. `tilelang/language/copy_op.py`
   - 看 `T.copy`、`T.async_copy`、`T.tma_copy`、`T.transpose`、`T.c2d_im2col` 如何映射到 tile op，以及 `coalesced_width` 等可选参数。

8. `tilelang/language/gemm_op.py`
   - 看 `T.gemm`、`T.wgmma_gemm`、`T.tcgen05_gemm`、`T.tcgen05_gemm_blockscaled` 的参数检查、`GemmWarpPolicy`、tile op 生成。Sparse 版本在 `tilelang/language/experimental/gemm_sp.py`。

9. `tilelang/jit/__init__.py`、`tilelang/jit/kernel.py`
   - 看 `@tilelang.jit` 如何从 Python 函数变成 PrimFunc，再调用 lowering 和 adapter。

10. `tilelang/engine/lower.py`、`tilelang/engine/phase.py`
    - 看真正的编译流水线：semantic check、legalize、layout inference、lower tile op、target optimization、host/device split。

11. `src/op/*`
    - 看 C++ tile operator 基类、copy/gemm operator、operator registry。

12. `src/transform/*`
    - 看 layout inference、lower tile op、pipeline planning、shared memory merge 等核心 compiler pass。

13. `tilelang/cuda/op/gemm/*`、`src/backend/cuda/op/gemm.cc`
    - 看 GEMM 指令选择和 CUDA Tensor Core / WGMMA / TCGEN05 lowering。

14. `tilelang/rocm/op/gemm/*`、`src/backend/rocm/op/gemm.cc`
    - 对比 AMD MFMA / WMMA 后端。

15. `testing/python/*`
    - 最后回到测试，用测试反推行为边界，这是找 PR 机会的主要入口。
    - 子目录大致分工：`language/` 校验 DSL 行为，`kernel/` 校验整端到端 kernel，`transform/` 校验单个 pass，`metal/` 校验 Metal codegen，`autotuner/`、`amd/` 等是各自方向的专门测试。

16. `examples/*`
    - 真实场景的参考实现，比 `testing/` 里碎片化的测试更利于建立大局观。建议挑这些读：
      - `examples/gemm/`、`examples/gemm_fp8/`、`examples/gemm_int4/`：从最朴素到 fp8/int4 量化的 GEMM 演进。
      - `examples/flash_attention/`、`examples/flash_attention_sm100/`、`examples/flash_decoding/`：FlashAttention 的不同版本和 Hopper/Blackwell 变体。
      - `examples/gemm_sm100/`、`examples/blockscaled_gemm_sm100/`：Blackwell `tcgen05` 路径的端到端写法。
      - `examples/dequantize_gemm/`、`examples/fusedmoe/`、`examples/dynamic_shape/`：和 LLM serving 直接相关的 case。
      - `examples/eager_jit/`、`examples/autodd/`：eager / autotune 接口的使用。

## 4. 核心编译链路

TileLang 的主链路可以按下面理解：

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

几个关键点：

- `T.copy`、`T.gemm` 这类高级操作不是立即变成 CUDA 代码，而是先进入 TileOp 表示。
- `Layout` / `Fragment` 负责描述 fragment 内数据如何映射到 thread、lane、local register。
- `LayoutInference` 会尝试在 producer/consumer 之间传播 layout，减少用户手工写 layout 的负担。
- `LowerTileOp` 把 TileOp 降成更底层的 TIR / intrinsic。
- CUDA 后端根据 target 架构选择 MMA、WGMMA、TCGEN05 等实现。
- JIT adapter 负责把编译好的 kernel 接到 PyTorch tensor / TVM runtime / NVRTC / Cython 等执行路径。

你熟悉 CUDA 和 Triton，建议重点对比：

| 主题 | Triton | TileLang |
| --- | --- | --- |
| 编程模型 | block program / `tl.load` / `tl.dot` | TIR-style block/thread + `T.copy` / `T.gemm` |
| 抽象层级 | 用户直接写 program-level tensor op | 用户写 DSL，编译器用 TVM pass 逐层 lower |
| layout 控制 | 通常隐含在 program shape / dot lowering | 显式 `Layout` / `Fragment`，也支持推导 |
| 同步 / 流水 | `tl.async_copy` + 隐式 barrier | `T.async_copy` / `T.tma_copy` + `T.Pipelined` + `ProducerConsumerWarpSpecialized` pass |
| 后端扩展 | Triton compiler backend | TVM target + TileOp lowering + codegen（CUDA/HIP/Metal/WebGPU/C/CuTe DSL）|
| 学习收益 | 高效写 kernel | 更接近 compiler infra 和 backend 工程 |

实际的 pass 顺序定义在 `tilelang/engine/phase.py`：

- `LowerAndLegalize`（target-agnostic 主路径）依次跑：`BindTarget` → `LetInline`（可选）→ `AddWrapperForSingleBufStore` → `LegalizeNegativeIndex` → `VerifyParallelLoop`（默认开）→ `InjectAssumes` → `Simplify` → `LayoutReducer` → `ProducerConsumerWarpSpecialized`（CUDA + TMA）→ `LowerBlackwell2SM` → `PipelinePlanning` → `InjectSoftwarePipeline` → `Simplify` → **`LayoutInference`** → **`LowerTileOp`** → `LowerL2Persistent` → `DecoupleTypeCast` → `LegalizeVectorizedLoop` → `LegalizeSafeMemoryAccess` → `LowerAccessPtr` → `Simplify` → `HoistNonRestrictParams`。
- `OptimizeForTarget` 再做 target 相关的事：`LowerSharedTmem` → `IfStmtBinding` → `PlanAndUpdateBufferAllocationLocation` → `LowerSharedBarrier` →（TMA 时）`FuseMBarrierArriveExpectTx` → `HoistGlobalBufferAllocations` → `LowerOpaqueBlock` → `NarrowDataType(32)` → `FlattenBuffer` → `ConfigIndexBitwidth` → `VectorizeLoop` → `StorageRewrite` → `LoopUnswitching` → `UnrollLoop` → `VerifyMemory` → `AnnotateEntryFunc` → `InferFragment` → `LowerThreadAllreduce` → `LowerLDGSTG` → `LowerHopperIntrin` →（可选）global `ThreadSync` → `AnnotateDeviceRegions` → **`SplitHostDevice`** → `MarkCudaSyncCalls` → `AnnotateReadOnlyParams` → `MergeSharedMemoryAllocations` → `InjectFenceProxy` → shared `ThreadSync` ×2 → `InjectTcgen05Fence` → `MergeIfStmt` →（WS 时）`AnnotateWarpGroupRegAlloc` → **`MakePackedAPI`** → `LowerDeviceKernelLaunch` → `PersistThreadblock`。

读源码时，把上面这条线放在手边对照，比单文件深挖更容易建立全景。

### 4.1 让 TileLang 把中间产物吐出来

调试 / 学习时常用的 inspection 套路：

```python
import tilelang
import tilelang.language as T

@tilelang.jit(target="cuda -arch=sm_80")
def matmul(M, N, K):
    @T.prim_func
    def gemm(A: T.Tensor((M, K), "float16"),
             B: T.Tensor((K, N), "float16"),
             C: T.Tensor((M, N), "float16")):
        ...
    return gemm

kernel = matmul(1024, 1024, 1024)
print(kernel.get_kernel_source())  # 生成的 CUDA / Metal / HIP 源码
print(kernel.prim_func)            # 最终下沉到的 PrimFunc
```

更细粒度的中间形态可以通过 `tilelang.engine.lower.lower_to_host_device_ir(...)` 拿到 host / device IRModule，或者在 `tilelang.transform.PassContext` 里打开下面这些开关：

- `tl.layout_visualization_enable=True` / `tl.layout_visualization_formats="txt,png"`：可视化 fragment / shared layout。
- `tl.ast_print_enable=True`：在 `PreLowerSemanticCheck` 阶段 dump AST。
- `tl.disable_warp_specialized=True`：关闭 WS，方便对比 baseline 流水。
- `tl.disable_prelower_semantic_check=True`：临时跳过 semantic check 看更底层错误。
- `tl.force_let_inline=True`：在 lower 之前把 let 展开掉，便于断点阅读。

## 5. 八周学习计划

### 第 0 周：环境和阅读基线

目标：能在 MacBook 上 import TileLang、跑 Metal 或 codegen 测试、知道失败原因。

任务：

- 建好 Python venv。
- 完成本地 CMake build。
- 跑 `testing/python/metal/`。
- 阅读 `README.md`、`docs/get_started/Installation.md`、`docs/get_started/targets.md`。
- 记录本机能跑和不能跑的测试范围。

验收标准：

```bash
python -c "import tilelang; print(tilelang.__version__)"
python -m pytest testing/python/metal/ -x
```

如果没有 Apple Silicon 或没有 MPS，可跳过 runtime 测试，优先跑 codegen / transform 类测试。

### 第 1-2 周：写 TileLang DSL

目标：能独立写小 kernel，并理解 DSL 到 TIR 的形态。

练习：

- vector add
- contiguous copy
- strided copy
- row-wise reduction
- RMSNorm
- naive matmul
- shared memory tiled matmul

重点源码：

- `tilelang/language/kernel.py`
- `tilelang/language/allocate.py`
- `tilelang/language/copy_op.py`
- `tilelang/language/loop.py`
- `testing/python/language/*`

建议动作：

- 每写一个 kernel，都打印或保存 PrimFunc script。
- 每写一个 kernel，都看生成的 kernel source。
- 遇到行为不符合预期时，先在 `testing/python/language/` 下找类似测试。

远端 CUDA 机器验证命令：

```bash
pip install . -v
python -m pytest testing/python/language/test_tilelang_language_copy.py -x
```

### 第 3-4 周：理解 GEMM、Layout、Fragment

目标：能看懂 TileLang 的 GEMM lowering 路径，知道 layout 在哪里产生、传播和消费。

重点源码：

- `tilelang/language/gemm_op.py`
- `tilelang/layout/layout.py`
- `tilelang/layout/fragment.py`
- `src/transform/layout_inference.cc`
- `src/transform/lower_tile_op.cc`
- `src/op/gemm.cc`
- `tilelang/tileop/gemm/*`
- `tilelang/cuda/op/gemm/*`

练习：

- 改写一个 naive matmul 为 tiled matmul。
- 对比 `T.gemm` 和手写 inner loop。
- 看同一个 GEMM 在 `cuda -arch=sm_80`、`cuda -arch=sm_90` 下走的路径差异。
- 读 `testing/python/kernel/test_tilelang_kernel_gemm.py`，挑 1-2 个 case 做断点式阅读。

验收标准：

- 能解释 `shared`、`local.fragment`、`local` 的差异。
- 能解释为什么 WGMMA 对 tile shape、warp group、shared layout 有要求。
- 能从一个 `T.gemm` 调用追到 Python lowering 实现和 C++ instruction selection。

远端 CUDA 机器验证命令：

```bash
python -m pytest testing/python/kernel/test_tilelang_kernel_gemm.py -x
```

### 第 5 周：理解 Pass Pipeline

目标：从 compiler infra 角度理解 TileLang，不只停留在 DSL 使用。

重点源码：

- `tilelang/engine/lower.py`
- `tilelang/engine/phase.py`
- `src/transform/*`
- `testing/python/transform/*`

建议按顺序读这些 pass：

1. `PreLowerSemanticCheck`
2. `LowerAndLegalize`
3. `PipelinePlanning` + `InjectSoftwarePipeline`
4. `LayoutInference`
5. `LowerTileOp`
6. `LegalizeSafeMemoryAccess` + `LegalizeVectorizedLoop`
7. `OptimizeForTarget`
8. `FlattenBuffer` + `StorageRewrite`
9. `ThreadSync` + `InjectFenceProxy`
10. `SplitHostDevice`
11. `AnnotateEntryFunc`
12. `MakePackedAPI`

`testing/python/transform/` 下相对独立、好读的几个测试文件：

- `test_tilelang_transform_layout_inference.py`
- `test_tilelang_transform_lower_tile_op.py`
- `test_tilelang_transform_pipeline_planning.py`
- `test_tilelang_transform_Inject_software_pipeline.py`
- `test_tilelang_transform_thread_sync.py`
- `test_tilelang_transform_legalize_safe_memory_access.py`
- `test_tilelang_transform_legalize_vectorized_loop.py`
- `test_tilelang_transform_flatten_buffer.py`
- `test_tilelang_transform_split_host_device.py`
- `test_tilelang_transform_simplify.py`

练习：

- 找一个简单 copy kernel，观察 lower 前后 TIR。
- 找一个 GEMM kernel，观察 `LowerTileOp` 前后的变化。
- 关掉 `tl.disable_warp_specialized` 默认值，对比有 / 无 WS 的 pipeline IR。
- 给一个已有 transform test 增加更明确的断言，而不是只检查能编译。

验收标准：

```bash
python -m pytest testing/python/transform/test_tilelang_transform_lower_tile_op.py -x
python -m pytest testing/python/transform/test_tilelang_transform_layout_inference.py -x
python -m pytest testing/python/transform/test_tilelang_transform_pipeline_planning.py -x
```

### 第 6 周：理解 Runtime 和 Target

目标：知道 TileLang 如何选择目标、编译、缓存、调用 kernel。

重点源码：

- `tilelang/utils/target.py`
- `tilelang/jit/__init__.py`
- `tilelang/jit/kernel.py`
- `tilelang/jit/execution_backend.py`
- `tilelang/jit/adapter/*`
- `tilelang/env.py`
- `tilelang/libinfo.py`

`tilelang/jit/execution_backend.py` 里 `allowed_backends_for_target` 给出的真实组合（不要凭印象）：

| target kind | 允许的 execution_backend |
| --- | --- |
| `cuda`（非 CuTe DSL）| `tvm_ffi`、`nvrtc`、`cython` |
| `cuda`（含 `cutedsl` key）| `cutedsl`（唯一） |
| `hip` | `tvm_ffi`、`cython` |
| `metal` | `tvm_ffi`、`torch` |
| `c`（CPU C 后端）| `cython`、`tvm_ffi` |
| 其它 | `cython`、`tvm_ffi` |

`auto` 在 cuda/metal/hip 上默认选 `tvm_ffi`，在 CuTe DSL target 上选 `cutedsl`，在其它 target 上选 `cython`。注意 `"dlpack"` 是 `"tvm_ffi"` 的历史别名。

练习：

- 在 Mac 上比较 `execution_backend="auto"` 与 `"torch"` 在 Metal 上的差异（`torch` 适合 PyTorch tensor 直传）。
- 在 CUDA 机器上比较 `"tvm_ffi"`、`"nvrtc"`、`"cython"` 三条路径，看 kernel cache 落在哪里、第一次和第二次调用的耗时差。
- 看 target detection 在 Mac、CUDA、ROCm 机器上分别走哪条分支（`tilelang/utils/target.py::determine_target`）。
- 尝试为 target validation 增加一个更友好的错误信息或测试。

验收标准：

- 能解释 `@tilelang.jit` 调用时，何时 lazy compile，何时 eager compile（提示：`tilelang.compile(...)` vs 第一次调用 `JITKernel`）。
- 能解释为什么 target 字符串推荐使用 `"cuda -arch=sm_80"` 这种形式（提示：cache 键 / TVM 解析 / 与 `nvcc.get_target_arch` 的关系）。
- 能从 `tilelang.utils.target.describe_supported_targets()` 列出所有 base name。

### 第 7 周：第一批 PR

目标：提交小而完整、review 成本低的 PR。

优先级从高到低：

1. 文档 PR
   - 修正 Mac 开发流程、target 使用说明、测试命令。
   - 优点：风险低，容易合并。

2. 测试 PR
   - 为已存在但覆盖不足的 DSL 行为补 regression test。
   - 优点：熟悉项目边界，review 重点明确。

3. 错误信息 PR
   - 改善 shape mismatch、target mismatch、unsupported dtype、unsupported scope 等报错。
   - 优点：对新用户价值大，通常不改核心行为。

4. 小型 transform PR
   - 给某个 pass 增加缺失断言、清理边界 case、修复明确 bug。
   - 优点：逐渐进入 compiler infra。

5. Metal 小 PR
   - 补 Metal codegen test 或修正 Apple Silicon 相关文档。
   - 优点：可在 Mac 本地验证。

不建议第一批 PR 做：

- 新增完整 NPU backend。
- 大规模重构 pass pipeline。
- 大改 GEMM instruction selection。
- 没有真实 GPU/NPU 验证的性能优化。

### 第 8 周及以后：进入 NPU / Ascend 方向

目标：从“会用 TileLang”进入“理解新硬件后端如何接入 TileLang”。

当前主仓库主要是 CUDA / ROCm / Metal / CPU/WebGPU 等方向；Ascend NPU 方向需要关注独立仓库：

- https://github.com/tile-ai/tilelang-ascend

截至 2026-05-15，该仓库 README 描述它支持两条技术路线：

- Ascend C & PTO
- AscendNPU IR

建议读 Ascend 方向时重点对照主仓库：

| 主仓库概念 | Ascend 方向要找的对应物 |
| --- | --- |
| `shared` / `local.fragment` | L1 / UB / L0 等 NPU memory scope |
| `T.copy` | GM/L1/UB/L0 之间的数据搬运 |
| `T.Pipelined` | Ascend 上的软件流水 |
| `T.Parallel` | Vector 侧并行和自动向量化 |
| `T.gemm` | Cube / Matmul 指令或 PTO lowering |
| CUDA thread/block | Ascend core / block / task mapping |
| barrier / mbarrier | Ascend 同步原语 |

进入 NPU 后端前，至少先完成这些准备：

- 能在主仓库看懂 `T.copy` 的 TileOp lowering。
- 能在主仓库看懂 `T.gemm` 的 instruction selection。
- 能写一个 regression test 证明 DSL 行为。
- 能读懂一个 generated source，不管是 CUDA、Metal 还是 Ascend C。
- 能区分“DSL 设计问题”“lowering 问题”“runtime 问题”“硬件调度问题”。

## 6. 实操路线：本地 Mac + 远端机器

### 本地 Mac 工作流

适合每天使用：

```bash
source .venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH
cmake --build build --parallel
python -m pytest testing/python/metal/ -x
pre-commit run --all-files
```

如果修改了 Python 文件，通常不需要重装；如果修改 C++，重新 build：

```bash
cmake --build build --parallel
```

### 远端 CUDA 机器工作流

适合 PR 前验证：

```bash
git clone --recursive https://github.com/<your-name>/tilelang.git
cd tilelang
pip install . -v
python -c "import tilelang; print(tilelang.__version__)"
python -m pytest testing/python/language/test_tilelang_language_copy.py -x
python -m pytest testing/python/transform/test_tilelang_transform_lower_tile_op.py -x
python -m pytest testing/python/kernel/test_tilelang_kernel_gemm.py -x
```

如果只是验证某个小 PR，不要一上来跑全量测试。先跑相关测试文件，再按影响范围扩大。

### 远端 NPU 机器工作流

建议等你进入 Ascend 仓库后再固化脚本。初期只需要做到：

- 能安装 Ascend toolkit / CANN / PyTorch NPU 相关依赖。
- 能跑官方 `tilelang-ascend` 的 hello world 或 GEMM 示例。
- 能定位 generated source。
- 能知道失败发生在 Python DSL、lowering、compiler、runtime、driver 哪一层。

不要把第一次 NPU 调试和第一次 PR 绑定在一起。先做能稳定复现的本地 PR，再进入硬件相关改动。

## 7. 推荐的第一批练习题

按顺序做，每个练习都尽量写成一个可运行脚本或测试：

1. Vector Add
   - 目标：熟悉 `T.Kernel`、block/thread、`T.Parallel`。

2. Contiguous Copy
   - 目标：熟悉 `T.copy` 和 global memory 访问。

3. Strided Copy
   - 目标：理解 region、stride、boundary check。

4. Row Sum
   - 目标：理解 reduction、fragment/local accumulation。

5. RMSNorm
   - 目标：连接真实 LLM operator。

6. Naive Matmul
   - 目标：对照 CUDA/Triton 的 matmul 思维。

7. Shared Memory Matmul
   - 目标：理解 `alloc_shared`、`alloc_fragment`、pipeline。

8. `T.gemm` Matmul
   - 目标：进入 tile op 和 backend lowering。

9. Layout Visualization / Inspection
   - 目标：理解 `Layout` / `Fragment` 如何影响 register 和 lane mapping。

10. Issue Regression Test
    - 目标：把 GitHub issue 或已有 corner case 写成测试，这是最像真实 PR 的练习。

## 8. 可贡献 PR 选题清单

### 文档类

- 给 `docs/get_started/Installation.md` 补充 Mac/Metal 开发路径。
- 给 `docs/get_started/targets.md` 补充 `auto` 在 Mac/CUDA/ROCm 下的选择优先级说明。
- 给 examples 增加“如何打印 TIR / kernel source”的说明。
- 给 NPU/Ascend 学习者写一篇主仓库到 Ascend 仓库的概念映射文档。

### 测试类

- `T.copy` 的 strided / boundary / dtype case。
- `T.Pipelined` 的 `num_stages=0/1/N` 行为。
- target validation 的错误路径。
- Metal codegen 的基本 dtype case。
- layout inference 冲突时的 regression test。

### 错误信息类

- shape mismatch 报错中加入 A/B/C shape、expected shape、operator 名。
- unsupported target 报错中列出 `describe_supported_targets()` 的可选值。
- dtype 不支持时说明当前 target、当前 op、支持 dtype。
- device mismatch 时说明 DLPack device code 和 target backend。

### 小型 compiler 类

- 为某个 pass 增加更精确的 unit test。
- 清理 `LowerTileOp` 或 `LayoutInference` 中可复现的边界 case。
- 为 target helper 增加测试。
- 改善 debug dump / logging 的可读性。

### 暂缓的方向

- 新增完整后端。
- 大规模改 GEMM scheduling。
- 没有硬件验证的性能优化。
- 需要同时改 Python DSL、C++ pass、runtime adapter 的大 PR。

## 9. PR 前检查清单

每次提交 PR 前，按影响范围选择检查。

环境一次性准备（lint 工具与版本以仓库 `requirements-lint.txt` / `.pre-commit-config.yaml` 为准）：

```bash
pip install -r requirements-lint.txt
pre-commit install
```

文档或 Python-only 小改：

```bash
pre-commit run --all-files            # 跑全套：ruff、ruff-format、clang-format、codespell、pymarkdown 等
python -m compileall tilelang         # 至少保证 Python 语法过得去
```

如果只想跑某个 hook，可以：

```bash
pre-commit run ruff --all-files
pre-commit run ruff-format --all-files
pre-commit run clang-format --all-files
pre-commit run codespell --all-files
```

DSL / transform 改动：

```bash
python -m pytest testing/python/language/ -x
python -m pytest testing/python/transform/ -x
```

Metal 改动：

```bash
python -m pytest testing/python/metal/ -x
```

CUDA kernel / GEMM 改动：

```bash
python -m pytest testing/python/language/test_tilelang_language_copy.py -x
python -m pytest testing/python/kernel/test_tilelang_kernel_gemm.py -x
```

如果只修改一处行为，先跑最小相关测试，再扩大范围。PR 描述里要写清楚：

- 改了什么。
- 为什么这样改。
- 哪些测试跑过。
- 哪些测试因为没有硬件没跑。
- 是否影响 CUDA/ROCm/Metal/Ascend。

## 10. 每周产出节奏

建议保持每周一个可见产出：

| 周期 | 产出 |
| --- | --- |
| 第 0 周 | 本地环境记录、能跑的测试列表 |
| 第 1 周 | 3 个小 kernel 脚本 |
| 第 2 周 | 1 个 DSL 行为测试 |
| 第 3 周 | 1 篇 GEMM lowering 阅读笔记 |
| 第 4 周 | 1 个 layout / gemm 相关 regression test |
| 第 5 周 | 1 个 transform test 或错误信息 PR |
| 第 6 周 | 1 个 target / runtime 相关小 PR |
| 第 7 周 | 第一个正式 PR |
| 第 8 周 | 开始读 `tilelang-ascend`，整理 NPU 对照表 |

## 11. 个人能力补齐清单

你已经熟悉 CUDA 和 Triton，下一步最值得补的是：

- TVM TIR 基础：PrimFunc、Buffer、Block、Schedule、PassContext。
- MLIR / compiler pass 基础：IR rewrite、analysis vs transform、canonicalization。
- GPU memory hierarchy 对照：NVIDIA shared/register/TMEM，AMD LDS/VGPR，Ascend GM/L1/UB/L0。
- TensorCore 指令族：MMA、WGMMA、TCGEN05；AMD MFMA；Ascend Cube。
- Kernel correctness 方法：随机测试、边界 shape、dtype coverage、reference implementation。
- 性能分析方法：roofline、memory bandwidth、occupancy、register pressure、pipeline stall。

## 12. Triton / CUDA 视角的术语对照

给从 Triton/CUDA 切过来的人留一张速查表，避免每次读 TileLang 源码都要重新建立心智模型。

| 概念 | Triton 里的对应物 | CUDA 里的对应物 | TileLang 里的关键词 |
| --- | --- | --- | --- |
| 程序入口 | `@triton.jit` 函数 | `__global__` kernel | `@T.prim_func` + `with T.Kernel(...)` + `@tilelang.jit` |
| Grid / launch | `grid = (...)` 参数 | `<<<grid, block>>>` | `T.Kernel(bx, by, ..., threads=...)` |
| Block 内并行循环 | 隐式（program 维度）| threadIdx 显式索引 | `T.Parallel(...)` |
| 同步软件流水 | `tl.load(.., other=, mask=)` + 自动 stage | 手写 mbarrier / async pipeline | `T.Pipelined(..., num_stages=N)` + `InjectSoftwarePipeline` pass |
| Persistent kernel | 自己写 `while`/`for` | grid-stride loop | `T.Persistent(...)` + `PersistThreadblock` pass |
| Shared memory | `tl.zeros((..,), dtype=)` 在 program 内 | `__shared__` | `T.alloc_shared((..), dtype)` |
| Register / fragment | 隐含在 dot 中 | MMA fragment / `.reg` | `T.alloc_fragment((..), dtype)`、`Layout` / `Fragment` |
| 异步 global → shared | Triton 在 `tl.load` / `tl.store` 中隐式调度 async copy | `cp.async` / TMA | `T.async_copy`、`T.tma_copy` |
| Tensor Core MMA | `tl.dot` | `mma.sync` / `wgmma` / `tcgen05.mma` | `T.gemm`、`T.wgmma_gemm`、`T.tcgen05_gemm`、`T.tcgen05_gemm_blockscaled` |
| Warp specialization | 手写 producer/consumer | named barrier / proxy fence | `T.ws(...)` + `ProducerConsumerWarpSpecialized` pass |
| Cluster / DSMEM | 不支持 | Hopper `cluster_*` intrinsic | `T.alloc_cluster_barrier`、`cluster_arrive`、`cluster_sync`、`block_rank_in_cluster` |
| TMEM（Blackwell）| 不支持 | `tcgen05.alloc.tmem` | `T.alloc_tmem`、`alloc_tcgen05_smem_desc`、`InjectTcgen05Fence` |
| Layout 控制 | 用户基本看不到 | `ldmatrix` / `stmatrix` 模式 | `tilelang.layout.Layout`、`Fragment`、`LayoutInference` pass |
| 一次性内核源码注入 | `triton.jit(extern=...)` | 手写 `.cu` 调起 | `T.CUDASourceCodeKernel` |
| Reduction | `tl.sum/max/min` | warp shuffle | `T.reduce_*`、`T.warp_reduce_*`、`finalize_reducer` |
| Atomic | `tl.atomic_add` 等 | `atomicAdd` 系列 | `T.atomic_add/max/min/load/store`、`atomic_addx2/x4` |
| Codegen | Triton GPU dialect → LLVM | nvcc / ptxas | TVM target → `target.build.tilelang_cuda/hip/metal` |
| 缓存 | `~/.triton/cache` | nvcc 缓存 / kernel module | 默认 `~/.tilelang/cache`（见 `docs/get_started/Installation.md`），各 adapter 自己的 cache 实现在 `tilelang/jit/adapter/*/kernel_cache.py` |

简单记忆：Triton 里大多数底层细节是编译器替你做的，TileLang 把这些细节都"暴露但默认推导"——你可以不管，但只要想管就有一个明确的名字可以挂上去。

## 13. 最推荐的切入点

如果你想尽快开始贡献，建议第一条线选这个：

1. 在 Mac 上读 `testing/python/language/test_tilelang_language_copy.py`。
2. 找一个未覆盖的 `T.copy` 边界 case。
3. 写一个最小 regression test。
4. 在 Mac 上跑 transform/codegen 能跑的部分。
5. 租 CUDA 机器跑相关 language test。
6. 提交一个只包含测试或小修复的 PR。

原因很简单：`T.copy` 是所有后端都会遇到的基础能力，足够重要；同时它比 GEMM/WGMMA/TCGEN05 更容易 review，也更适合作为第一次 PR。

第二条线是错误信息：

1. 找一个你自己学习时踩到的 confusing error。
2. 追到抛错位置。
3. 补一个测试固定错误路径。
4. 改成更明确的报错。
5. 提 PR。

这类 PR 对开源项目很有价值，也能逼你真正读懂编译链路。

