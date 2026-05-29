# TVM/TIR learning examples for TileLang

这个目录对应仓库根目录的 `TILELANG_TVM_TIR_OVERVIEW.md`，目标不是跑最快 kernel，而是用小脚本把 TileLang 依赖的 TVM/TIR 概念拆开观察。

建议从仓库根目录运行：

```bash
python examples/tvm/01_ir_module_primfunc.py
python examples/tvm/02_schedule_sblock.py
python examples/tvm/03_pass_context_target.py --target llvm
python examples/tvm/04_tilelang_tile_ops.py
python examples/tvm/05_tir_transformation_tutorial.py --target llvm
```

如果你已经装好完整 TileLang 依赖、构建好本地库，并且目标后端可用，可以继续探索：

```bash
python examples/tvm/03_pass_context_target.py --target cuda
python examples/tvm/04_tilelang_tile_ops.py --lower --target cuda
```

## 文件顺序

1. `01_ir_module_primfunc.py`

   观察 `IRModule`、`PrimFunc`、`Buffer`、`SBlock/Block`、read/write region 和 `structural_equal`。这是理解 TileLang kernel 最终如何进入 TVM 编译体系的起点。

2. `02_schedule_sblock.py`

   用标准 TensorIR matmul 演示 `Schedule` 如何围绕 `SBlock/Block` 做 `split/reorder`。TileLang 用户平时不直接写这套 schedule，但 `s_tir`、layout inference 和 fragment 推导都建立在这些边界上。

3. `03_pass_context_target.py`

   演示 `PassContext`、`Target`、`Target.current()` 和一组可用的 TIR pass。TileLang 的 `LowerAndLegalize`、`OptimizeForTarget` 本质也是 `IRModule -> pass -> IRModule`。

4. `04_tilelang_tile_ops.py`

   用 TileLang DSL 生成一个带 `T.copy`、`T.Parallel`、`alloc_shared`、`alloc_fragment` 的 `PrimFunc`，并列出几个 TileLang 通过 TVM FFI 注册的 pass/codegen hook。加 `--lower` 后会进入 TileLang lowering 并打印 host/device IR。

5. `05_tir_transformation_tutorial.py`

   复刻 TVM 官方 TensorIR transformation 教程里的 `mm_relu`、`evaluate`、`split/reorder`、`reverse_compute_at`、`decompose_reduction` 和 `trace/show` 步骤。脚本保留官方默认 `llvm` target，但通过本目录的 `common.import_tvm()` 导入 TileLang bundled TVM，适合在远端 GPU 开发机的特殊环境里学习。

## 环境说明

这些脚本优先通过 `from tilelang import tvm` 使用仓库内的 TVM fork，因为当前 TileLang 使用 `tirx` / `s_tir` 分层。当前 Python 环境如果缺 `torch`、未构建 TileLang native library，脚本会给出提示；先按项目构建方式安装依赖，例如：

```bash
pip install .
```

不要用这些脚本做性能 benchmark。它们故意保持小尺寸和高可读性，适合你修改 shape、target、pass 顺序，然后对比打印出来的 IR。

`05_tir_transformation_tutorial.py` 默认会构建并计时官方教程里的函数；如果只想看 schedule 步骤和 IR 打印，可以先运行：

```bash
python examples/tvm/05_tir_transformation_tutorial.py --no-evaluate
```
