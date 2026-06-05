# TileLang JIT Lowering 与 Cache 源码笔记

这份笔记整理本次会话里对 `@tilelang.jit`、`JITImpl.__call__`、
`tilelang.lower`、kernel cache 和线程/进程模型的源码解读。

主要结论：

- 一个 `@tilelang.jit` 装饰出来的函数通常对应一个 `JITImpl` 对象。
- `JITImpl.__call__` 不直接调用 `tilelang.lower`，而是先做模式推断、参数解析和对象级 cache 查找。
- cache miss 时，`JITImpl.compile()` 会生成 `PrimFunc`，再经 `tilelang.cache.cached(...)` 创建或加载 `JITKernel`。
- 真正直接调用 `tilelang.lower(...)` 的地方在 `JITKernel._compile_and_create_adapter(...)`。
- `tilelang.lower` 的输入不是原始 Python DSL 函数，也不是 torch tensor，而是已经生成好的 `tvm.tirx.PrimFunc` 或 `IRModule`。
- cache 有多层：`JITFunc.p1_cache`、`JITImpl._kernel_cache`、进程内 backend `KernelCache._memory_cache`、磁盘 kernel cache、lazy frontend cache。
- cache 不是线程局部的。线程局部主要用于 DSL builder 等前端构造状态，不用于 kernel cache。

## 一条完整调用链

用户代码：

```python
@tilelang.jit
def kernel(...):
    ...
```

装饰后，函数名 `kernel` 指向的已经不是原始 Python 函数，而是一个
`JITImpl` 对象。第一次调用的大致路径是：

```text
python script
  -> import tilelang
     -> load libtilelang.so
     -> register FFI / pass / op / codegen

@tilelang.jit
  -> tilelang.jit.jit(...)
  -> prim_func(func, eager_jit=True)
  -> JITFunc
  -> JITImpl

JITImpl.__call__(*args, **kwargs)
  -> infer lazy/eager mode
  -> JITFunc.parse_args(...)
     -> may check/build JITFunc.p1_cache
        -> value is TirTemplate, not runnable kernel
  -> check JITImpl._kernel_cache
  -> cache miss
  -> JITImpl.compile(...)
     -> JITImpl.get_tir(...)
     -> JITFunc.get_tir(...)
     -> generated tvm.tirx.PrimFunc
     -> tilelang.jit.compile(prim_func, ...)
     -> tilelang.cache.cached(prim_func, ...)
     -> backend KernelCache.cached(...)
     -> check process memory cache
     -> check disk cache
     -> cache miss
     -> JITKernel(...)
        -> JITKernel._compile_and_create_adapter(...)
        -> tilelang.lower(prim_func, ...)
        -> backend codegen
        -> create adapter
  -> store JITImpl._kernel_cache[key] = JITKernel
  -> eager: kernel(*runtime_tensor_args)
  -> lazy: return JITKernel
```

源码位置：

- `tilelang/jit/__init__.py`: `jit`、`JITImpl`、`JITImpl.__call__`、`JITImpl.compile`
- `tilelang/language/eager/builder.py`: `JITFunc`、`TirTemplate`、DSL builder
- `tilelang/jit/kernel.py`: `JITKernel`、`_compile_and_create_adapter`
- `tilelang/engine/lower.py`: `lower`、`lower_to_host_device_ir`
- `tilelang/cache/__init__.py`: backend cache dispatch 和 frontend cache key
- `tilelang/cache/kernel_cache.py`: 进程内和磁盘 kernel cache

## `JITImpl.__call__` 不直接调用 `tilelang.lower`

`JITImpl.__call__` 主要做控制流和 cache 查找：

```python
if self.mode == "auto":
    self.mode = self._infer_jit_mode(*args, **kwargs)
    self.func.set_mode(self.mode)

key, kernel_args = self.func.parse_args(*args, **kwargs)
kernel = self._kernel_cache.get(key, None)
if kernel is None:
    ...
    kernel = self.compile(*args, **kwargs)
    ...
    self._kernel_cache[key] = kernel

if self.mode == "eager":
    return kernel(*kernel_args.values())
else:
    return kernel
```

这里的 `self.compile(...)` 才会进入后续编译链路。

`tilelang.lower(...)` 的直接调用点在 `tilelang/jit/kernel.py` 的
`JITKernel._compile_and_create_adapter(...)`：

```python
with tvm.transform.PassContext(opt_level=3, config=pass_configs, instruments=pass_instruments), self.target:
    artifact = tilelang.lower(
        tilelang_func,
        target=target,
        target_host=target_host,
        enable_host_codegen=enable_host_codegen,
        enable_device_compile=enable_device_compile,
    )
```

这里的 `tilelang_func` 已经是 `PrimFunc`。

## `tilelang.lower` 的输入是什么

JIT 路径传给 `tilelang.lower` 的主输入是：

```python
tvm.tirx.PrimFunc
```

不是：

- 原始 Python 函数
- Python DSL AST
- torch tensor
- 已经编译好的 CUDA source

`tilelang.lower` 也支持直接接收 `tvm.IRModule`。如果输入是 `PrimFunc`，
`lower_to_host_device_ir` 会先包成 `IRModule`：

```python
params = extrac_params(func) if not runtime_only else None
mod = tvm.IRModule({func.attrs["global_symbol"]: func})
```

`params` 是 runtime kernel 参数描述，来自：

- `func.params`
- `func.buffer_map`

每个参数会被转成 `KernelParam`，后续 adapter 用它决定 Python tensor/标量
参数如何绑定到 runtime 调用。

## lazy 和 eager 两种 JIT 模式

`@tilelang.jit` 内部先把原始函数包成 `JITFunc`：

```python
pf = prim_func(func, eager_jit=True)
return JITImpl(func=pf, ...)
```

### lazy style

lazy style 是用户函数显式返回一个 `PrimFunc`：

```python
@tilelang.jit
def foo(M, N):
    @T.prim_func
    def kernel(...):
        ...
    return kernel
```

`JITFunc._is_lazy_style(...)` 会调用原始函数。如果返回值是 `PrimFunc`，
就认为是 lazy style，并保存一个 lazy `TirTemplate`。

lazy 模式下，调用 `foo(M, N)` 通常返回一个 `JITKernel`；用户之后再用
runtime tensor 调用它。

### eager style

eager style 使用 TileLang DSL builder：

```python
@tilelang.jit
def foo(A, B, C):
    A: T.Tensor(...)
    with T.Kernel(...):
        ...
```

这种函数不显式返回 `PrimFunc`。TileLang 会通过 `Builder` trace 函数体，
构造一个 `PrimFunc` 模板，再根据 runtime tensor 的 shape/stride 做
phase2 具体化。

eager 模式下，调用 `foo(A, B, C)` 会在 cache miss 时编译，然后立即执行
compiled kernel。

## 参数解析和 `JITFunc.p1_cache`

`JITImpl.__call__` 会调用：

```python
key, kernel_args = self.func.parse_args(*args, **kwargs)
```

`self.func` 是 `JITFunc`。它内部还有一个 `p1_cache`：

```python
self.p1_cache: dict[Any, TirTemplate] = {}
```

这不是最终 kernel cache，而是前端 TIR 模板 cache。

### phase1 key

`JITFunc._parse_phase1_key(...)` 会做两件事：

1. 把 positional args 合并进 kwargs。
2. 从 kwargs 里摘出 tensor args。

源码逻辑大致是：

```python
kwargs.update({k: v for k, v in zip(self.arg_names, args)})
tensor_args = {}
for k in self.tensor_args:
    if k in kwargs:
        tensor_args[k] = kwargs.pop(k)
    elif k in self.tensor_args_defaults:
        tensor_args[k] = self.tensor_args_defaults[k]
p1_key = tuple(sorted(kwargs.items()))
```

所以 `p1_key` 主要由非 tensor 参数组成，比如：

- block size
- dtype 字符串
- 显式 constexpr 参数
- 其他编译期 Python 参数

`tensor_args` 是 runtime tensor 参数，比如 torch tensor。

### `p1_cache` 的 key/value

`JITFunc.p1_cache` 的 key 是：

```python
p1_key = tuple(sorted(kwargs.items()))
```

value 是：

```python
TirTemplate
```

`TirTemplate` 里保存：

- `prim_func`: phase1 构造出来的模板 `PrimFunc`
- `matcher`: eager 模式下 constexpr 变量到 tensor shape/stride 的匹配规则
- `constexprs`: 需要 phase2 代入的变量集合
- `is_lazy_style`: 是否 lazy
- `ir_gen`: eager 模式下重新生成 phase2 `PrimFunc` 的 IR generator

### phase2 key

eager 模式下，如果 `PrimFunc` 模板里有 constexpr 变量来自 tensor shape 或
stride，`TirTemplate._parse_phase2_key(...)` 会从 runtime tensor 中读出这些
值：

```python
if ty == "shape":
    result.append(kwargs[k].shape[i])
elif ty == "stride":
    result.append(kwargs[k].stride()[i])
```

最后得到：

```python
p2_key = tuple(...)
```

`JITImpl._kernel_cache` 使用的最终 key 是：

```python
(p1_key, p2_key)
```

如果没有 tensor args，则是：

```python
(p1_key, None)
```

这意味着 eager 模式下，同一个 Python 函数、同一组编译期参数，但不同
tensor shape/stride，通常会对应不同的 compiled kernel。

## `JITImpl._kernel_cache`

每个 `JITImpl` 对象都有自己的 `_kernel_cache`：

```python
self._kernel_cache: dict[tuple, Kernel] = {}
```

### key

key 来自：

```python
key, kernel_args = self.func.parse_args(*args, **kwargs)
```

通常形态是：

```python
(p1_key, p2_key)
```

其中：

- `p1_key`: 非 tensor 编译期参数
- `p2_key`: 从 tensor shape/stride 或显式 constexpr 推导出的动态形状 key

`p2_key` 不包含 `target`、`target_host`、`execution_backend`、`pass_configs`
或 `compile_flags`。这些编译选项对单个 `JITImpl` 来说通常是对象属性，
对底层跨对象/跨进程 cache 来说则进入 `KernelCache` key。

### value

value 是：

```python
JITKernel
```

它已经持有：

- `prim_func`
- `artifact`
- `adapter`
- `torch_function = adapter.func`

所以命中 `JITImpl._kernel_cache` 后，通常不会再进入：

- `JITImpl.compile`
- `tilelang.cache.cached`
- `JITKernel.__init__`
- `tilelang.lower`

### 生命周期

`JITImpl._kernel_cache` 是：

- 对象级 cache
- 进程内 cache
- 不落盘
- 不跨 `JITImpl` 对象共享
- 多线程共享同一个 `JITImpl` 时也共享
- 当前源码里没有锁

如果你对同一个原始函数调用两次 `tilelang.jit(...)`，会得到两个不同的
`JITImpl` 对象，它们的 `_kernel_cache` 不共享。

## `tilelang.cache.cached` 和 backend cache dispatch

`JITImpl.compile(...)` 会调用顶层：

```python
tilelang.jit.compile(prim_func, ...)
```

后者再调用：

```python
tilelang.cache.cached(...)
```

`tilelang/cache/__init__.py` 里有 backend dispatch 表：

```python
_dispatch_map = {
    "tvm_ffi": TVMFFIKernelCache(),
    "cython": CythonKernelCache(),
    "nvrtc": NVRTCKernelCache(),
    "cutedsl": CuTeDSLKernelCache(),
    "torch": TorchKernelCache(),
}
```

`_resolve_cache_dispatch(...)` 会：

1. 从参数或环境变量解析 target。
2. 从参数或环境变量解析 execution backend。
3. 调 `determine_target(...)` 规范化 target。
4. 调 `resolve_execution_backend(...)` 把 `"auto"` 映射到实际 backend。
5. 返回对应 backend 的 `KernelCache` 实例。

所以同一进程内，不同 backend 有不同的 cache 实例和不同的 `_memory_cache`。

## `KernelCache._memory_cache`

`KernelCache` 是 backend 级的进程内 cache。它有：

```python
_lock = threading.Lock()
_memory_cache = {}
```

### key

`KernelCache._generate_key(...)` 生成一个 SHA256 字符串。

它先取：

```python
func_binary = func.script(show_meta=True).encode()
```

然后构造：

```python
key_data = {
    "func": sha256(func_binary).hexdigest(),
    "out_idx": ...,
    "args_repr": tuple(repr(arg) for arg in args),
    "target": str(target),
    "target_host": str(target_host) if target_host else None,
    "execution_backend": execution_backend,
    "pass_configs": pass_configs,
    "compile_flags": compile_flags,
    **self._get_base_key(),
}
```

最后：

```python
key_string = json.dumps(key_data, sort_keys=True)
key = sha256(key_string.encode()).hexdigest()
```

`_get_base_key()` 至少包含：

- TileLang version
- platform machine

开发构建中还可能包含 native library stamp，也就是 `libtilelang.so`、
`libtvm_runtime.so`、`libtvm_compiler.so` 等库文件内容 hash。这样 C++ pass
或 codegen 变了，即使 Python `PrimFunc` 文本没变，也能让 cache key 变化。

macOS 上还会包含 torch version。

### value

`KernelCache._memory_cache[key]` 的 value 是：

```python
JITKernel
```

如果是从磁盘加载的，也会通过：

```python
JITKernel.from_database(...)
```

重建出一个可调用的 `JITKernel`。

### 查找顺序

`KernelCache.cached(...)` 的顺序是：

```text
if cache disabled:
  compile directly with JITKernel(...)

generate kernel key
check _memory_cache under lock
if memory hit:
  return JITKernel

load from disk outside lock
if disk hit:
  insert _memory_cache under lock
  return JITKernel

compile JITKernel outside lock
save to disk under lock
tag kernel with cache key/path
insert _memory_cache under lock
return JITKernel
```

注意：编译本身不在 `_lock` 保护范围内。因此多个线程同时 miss 时，仍然可能
重复编译同一个 kernel。

## cache 访问顺序和短路能力

这里要区分“模板 cache”和“可运行 kernel cache”。

`JITFunc.p1_cache` 是模板 cache：

```text
key: p1_key
value: TirTemplate
```

它的 value 不是可运行 kernel。因此命中 `p1_cache` 后，调用不会结束，仍然
要继续计算 `p2_key`，再继续查 `JITImpl._kernel_cache`。它只能省掉
“重新构造 TIR 模板”这一步。

后面的几层是 kernel cache 或 kernel artifact cache：

```text
JITImpl._kernel_cache
  value: JITKernel
  hit 后可以直接执行或返回，不再访问 KernelCache._memory_cache / disk cache / tilelang.lower

KernelCache._memory_cache
  value: JITKernel
  hit 后可以返回 JITKernel，不再访问 disk cache / tilelang.lower

Disk kernel cache
  value: 可重建 JITKernel 的 artifact
  hit 后会 from_database 重建 JITKernel，不再调用 tilelang.lower/codegen
```

所以不能简单说“三层 cache 每次都要访问”。更准确的顺序是：

```text
JITImpl.__call__
  -> JITFunc.parse_args(...)
     -> 可能访问 JITFunc.p1_cache
        hit:
          复用 TirTemplate
          继续往下
        miss:
          构造 TirTemplate
          写入 p1_cache
          继续往下

  -> 查 JITImpl._kernel_cache
     hit:
       直接得到可运行 JITKernel
       停止访问更底层 cache

     miss:
       JITImpl.compile(...)
         -> JITFunc.get_tir(...)
         -> tilelang.cache.cached(...)
            -> 查 KernelCache._memory_cache
               hit:
                 返回 JITKernel
                 不访问 disk cache

               miss:
                 查 disk cache
                   hit:
                     from_database 重建 JITKernel
                     写入 KernelCache._memory_cache
                     返回

                   miss:
                     JITKernel(...)
                       -> tilelang.lower(...)
                       -> codegen / adapter
                     写 disk cache
                     写 KernelCache._memory_cache
                     返回

       写入 JITImpl._kernel_cache
       执行或返回 JITKernel
```

因此：

```text
p1_cache 命中:
  只跳过模板构造，还要继续访问后续 kernel cache。

JITImpl._kernel_cache 命中:
  已经拿到可运行 JITKernel，直接短路后续编译链路。

KernelCache._memory_cache 命中:
  跳过磁盘读取和 lower/codegen。

disk cache 命中:
  跳过 tilelang.lower 和 backend codegen，但仍需要从 artifact 重建 JITKernel。
```

换句话说，只有“产物已经是 JITKernel”的 cache 命中，才能直接短路到执行；
`JITFunc.p1_cache` 命中只减少前端模板构造成本。

## 磁盘 kernel cache

磁盘 cache 根目录来自环境变量：

```text
TILELANG_CACHE_DIR
```

默认是：

```text
~/.tilelang/cache
```

临时文件目录来自：

```text
TILELANG_TMP_DIR
```

默认在 cache 目录下的 `tmp`。

### 目录命名

磁盘 cache 使用 namespace：

```text
<TILELANG_CACHE_DIR>/<version-platform>/kernels/<kernel_key>/
```

如果 `_get_base_key()` 里包含 native library stamp，namespace/key 会受 native
library 内容影响。

还有 frontend cache 目录：

```text
<TILELANG_CACHE_DIR>/<version-platform>/frontend/
```

以及 staging 目录：

```text
<TILELANG_CACHE_DIR>/<version-platform>/.staging/
```

### kernel cache 的 value 存什么

基础 `KernelCache` 认为一个完整 cache entry 至少要有：

```text
device_kernel.cu
host_kernel.cu
kernel_lib.so
params.pkl
```

其中：

- `device_kernel.cu`: device kernel source
- `host_kernel.cu`: host wrapper source
- `kernel_lib.so`: 编译出的 shared library 或 backend 对应二进制
- `params.pkl`: `KernelParam` 列表，用 `cloudpickle` 保存

另外还有可选文件：

- `prim_func.pkl`: 保存 `PrimFunc`，给 frontend cache 跨进程重建 adapter 用
- `resource_usage.json`: HIP/ROCm kernel resource usage 记录

不同 backend 会改写部分文件名或增加文件。

### backend 差异

`tvm_ffi` backend:

- `kernel_lib_path = "executable.so"`
- 保存的是 TVM executable
- host source 通过 `kernel.adapter.get_host_source()` 保存

`nvrtc` backend:

- `kernel_lib_path = "kernel.cubin"`
- 额外保存 `kernel.py`
- required files 包含 `kernel.py`

`cutedsl` backend:

- `kernel_lib_path = "kernel.py"`
- `device_kernel_path = "device_kernel.py"`
- `host_kernel_path = "host_kernel.py"`
- 额外保存 `launcher_lib.so`
- 可能保存 `launcher.cpp`

`cython` 和 `torch` 当前直接继承基础 `KernelCache` 行为。

### 从磁盘 value 重建 `JITKernel`

磁盘 hit 后，`KernelCache._load_kernel_from_disk(...)` 会：

1. 检查 required files 是否完整。
2. 读取 `params.pkl`。
3. 构造 `CachedTextSource(path=...)` 指向 host/device source。
4. 调：

   ```python
   JITKernel.from_database(
       func=func,
       host_kernel_source=...,
       device_kernel_source=...,
       kernel_lib_path=...,
       params=kernel_params,
       target=target,
       target_host=target_host,
       out_idx=out_idx,
       execution_backend=execution_backend,
       pass_configs=pass_configs,
       compile_flags=compile_flags,
   )
   ```

5. `JITKernel.from_database(...)` 再通过 adapter 的 `from_database` 路径重建
   Python callable。

这条路径不会重新跑 `tilelang.lower`。

## lazy frontend cache

`JITImpl.__call__` 里还有一层 frontend cache，但只在特定条件下启用：

```python
if self.mode == "lazy" and not kernel_args:
    frontend_key_data = self._frontend_cache_key_data(key)
    kernel = load_frontend_cached(...)
```

也就是说：

- 只用于 lazy mode
- 且 `parse_args` 没留下 runtime tensor args

### frontend key

`JITImpl._frontend_cache_key_data(...)` 包含：

```python
{
    "function": func_name,
    "qualname": func_qualname,
    "module": func_module,
    "source": self.func_source,
    "signature": str(self.signature),
    "key": repr(key),
    "mode": self.mode,
}
```

`tilelang.cache._make_frontend_cache_key(...)` 再把这些 frontend 数据和编译
选项合并：

```python
{
    "frontend": normalized_frontend_key_data,
    "out_idx": out_idx,
    "target": str(target),
    "target_host": str(target_host) if target_host else None,
    "execution_backend": execution_backend,
    "pass_configs": pass_configs,
    "compile_flags": compile_flags,
}
```

最后 hash 成 frontend key。

### frontend value

frontend cache 文件是 JSON，内容只有：

```json
{"kernel_key": "..."}
```

它不是直接保存 kernel，而是保存：

```text
frontend_key -> kernel_key
```

然后再用 `kernel_key` 去正常的磁盘 kernel cache 目录里加载 compiled artifact。

这层的作用是：新的 Python 进程里，即使还没重新 elaborate Python DSL，也可以
通过 source/signature/参数 key 找到之前保存的 `prim_func.pkl` 和 compiled
kernel artifact。

## `TILELANG_DISABLE_CACHE` 的影响

`KernelCache.cached(...)` 开头会检查：

```python
if not env.is_cache_enabled():
    return JITKernel(...)
```

环境变量：

```text
TILELANG_DISABLE_CACHE=1
```

会禁用 `KernelCache` 的进程内 `_memory_cache` 和磁盘 cache 路径。

但要注意：这不等于禁用 `JITImpl._kernel_cache`。如果调用仍然经过同一个
`JITImpl.__call__`，第一次 compile 后，`JITImpl` 仍会执行：

```python
self._kernel_cache[key] = kernel
```

所以 `TILELANG_DISABLE_CACHE` 更准确地说是禁用 backend `KernelCache` 和磁盘
cache，不是禁用所有 JIT wrapper 内部复用。

## 线程模型与隔离

### 哪些状态是线程局部的

TileLang 前端 DSL builder 使用了 `threading.local()`：

```python
thread_local_storage = threading.local()
```

`Builder.current()` 读取当前线程的 builder：

```python
builder = getattr(thread_local_storage, "builder", None)
```

所以 DSL trace、macro/frame 构造这类前端构造状态是线程局部的。

源码里还可以看到其他线程局部状态，例如：

- language kernel frame
- let value state
- HIP resource recorder
- autotuner capture state

### 哪些 cache 不是线程局部的

以下 cache 都不是 thread-local：

- `JITFunc.p1_cache`
- `JITImpl._kernel_cache`
- backend `KernelCache._memory_cache`
- 磁盘 cache
- frontend cache

如果多个线程共享同一个 decorated function，也就是共享同一个 `JITImpl`，
它们就共享同一个：

- `JITImpl.mode`
- `JITImpl._kernel_cache`
- `JITFunc.p1_cache`

### GIL 为什么不等于逻辑线程安全

CPython GIL 可以让单个 `dict.get` 或 `dict.__setitem__` 不至于破坏 dict
内部结构。但这个 cache miss 路径不是原子操作：

```python
kernel = self._kernel_cache.get(key, None)
if kernel is None:
    kernel = self.compile(*args, **kwargs)
    self._kernel_cache[key] = kernel
```

两个线程可能这样交错：

```text
Thread A: get(key) -> None
Thread B: get(key) -> None
Thread A: compile(...)
Thread B: compile(...)
Thread A: set cache[key]
Thread B: set cache[key]
```

最常见结果是重复编译，最后 `_kernel_cache[key]` 留下其中一个 `JITKernel`。

这通常不会把 dict 写坏，但不能保证：

- 同一个 key 只编译一次
- 第一次 mode 推断只有一个线程执行
- `JITFunc.p1_cache` 只构造一次
- native lowering/codegen 全链路都按单线程方式运行

另外，`tilelang.lower` 和 backend codegen 会进入 TVM/TileLang C++、编译器、
driver 或文件系统路径，这些地方可能释放 GIL。因此不能把 GIL 当成整个 JIT
编译链路的同步机制。

### `KernelCache._memory_cache` 的锁范围

backend `KernelCache` 有 `_lock`，但锁保护的是：

- memory cache 查找
- memory cache 插入
- 部分磁盘保存过程

它不保护整个编译过程。源码逻辑是：

```text
check memory cache under lock
load disk cache outside lock
compile JITKernel outside lock
save to disk under lock
insert memory cache under lock
```

所以 backend `KernelCache` 也不是 single-flight cache。多个线程同时 miss 时，
仍可能重复编译。它更像是：

- 防止 `_memory_cache` 读写结构竞争
- 尽量避免重复插入
- 通过磁盘原子写避免半成品 cache 被其他进程读取

### 进程模型

不同 Python 进程之间：

- 不共享 `JITImpl._kernel_cache`
- 不共享 `JITFunc.p1_cache`
- 不共享 `KernelCache._memory_cache`
- 可以共享磁盘 cache，只要 `TILELANG_CACHE_DIR` 和 namespace 一致

磁盘写入没有全局文件锁，也不是跨进程 single-flight。两个进程同时 miss 时，
都可能编译；写盘时通过 staging directory 和 atomic rename 处理竞争。

## 磁盘写入的原子性

`KernelCache._save_kernel_to_disk(...)` 不会直接往最终 cache 目录写完整内容。
它先创建 staging 目录：

```text
<cache namespace>/.staging/<kernel_key>_<pid>_<uuid>/
```

所有文件先写到 staging 目录。普通文件写入时，也会先写临时文件，再：

```python
os.replace(temp_path, path)
```

staging 文件齐全后，检查完整性：

```python
missing_files = self._get_missing_complete_cache_files(staging_path)
```

然后：

```python
os.rename(staging_path, cache_path)
```

如果 rename 时发现目标目录已经存在，说明其他进程可能赢了这次 race，当前
进程会删除自己的 staging 目录。

这保证其他线程/进程不会看到半写入的完整 cache entry。但它不保证只有一个
线程/进程执行编译。

## `tilelang.lower` 内部做什么

`tilelang.lower(...)` 首先进入：

```python
lower_to_host_device_ir(...)
```

主要步骤：

1. 如果输入是 `PrimFunc`，包成 `IRModule`。
2. 解析 target 和 target host。
3. 构造 host/device function predicate。
4. 执行 backend-independent semantic check：

   ```python
   PreLowerSemanticCheck(mod)
   ```

5. 根据 target kind 找 backend pipeline：

   ```python
   pipeline = resolve_pipeline(target)
   mod = pipeline.lower(mod, target)
   ```

6. 分离 host 和 device IR：

   ```python
   host_mod = tirx.transform.Filter(_is_host_call)(mod)
   device_mod = tirx.transform.Filter(_is_device_call)(mod)
   ```

然后 `lower(...)` 继续做 device codegen：

```python
codegen_mod = device_codegen(device_mod, target) \
    if enable_device_compile \
    else device_codegen_without_compile(device_mod, target)
kernel_source = codegen_mod.inspect_source()
```

如果 `enable_host_codegen=True`，还会：

```python
host_mod = host_codegen(host_mod, target_host, target=target)
host_mod.import_module(codegen_mod)
return CompiledArtifact(host_mod, device_mod, params, kernel_source, rt_mod=host_mod)
```

`tvm_ffi` backend 会设置：

```python
enable_host_codegen = True
enable_device_compile = True
```

其他 backend 通常走 `device_codegen_without_compile(...)`，再由 adapter 负责
后续编译或包装。

## backend pass pipeline

概念上可以把 pass 分成：

```text
PreLowerSemanticCheck
LowerAndLegalize
  -> LayoutInference
  -> LowerTileOp
OptimizeForTarget
  -> AnnotateDeviceRegions
  -> SplitHostDevice
  -> MakePackedAPI
  -> LowerDeviceKernelLaunch
```

但当前 Python 源码里不是在 `engine/lower.py` 里硬编码
`LowerAndLegalize(...)` 和 `OptimizeForTarget(...)` 两个函数。

实际是每个 backend 注册自己的 `PassPipeline`：

- CUDA: `tilelang/cuda/pipeline.py`
- ROCm/HIP: `tilelang/rocm/pipeline.py`
- CPU: `tilelang/cpu/pipeline.py`
- Metal: `tilelang/metal/pipeline.py`
- WebGPU/common fallback: `tilelang/backend/common.py`

CUDA pipeline 里能看到这些关键阶段：

- `BindTarget`
- `AddWrapperForSingleBufStore`
- `LegalizeNegativeIndex`
- `InjectAssumes`
- `Simplify`
- `LayoutReducer`
- CUDA-specific warp specialization / Blackwell lowering
- `PipelinePlanning`
- `InjectSoftwarePipeline`
- `LayoutInference`
- `LowerTileOp`
- CUDA-specific tile op / barrier / Hopper / LDG STG lowering
- `FlattenBuffer`
- `VectorizeLoop`
- `StorageRewrite`
- `UnrollLoop`
- `InferFragment`
- `LowerThreadAllreduce`
- `AnnotateDeviceRegions`
- `SplitHostDevice`
- `MergeSharedMemoryAllocations`
- `ThreadSync`
- `MakePackedAPI`
- `LowerDeviceKernelLaunch`
- CUDA-specific `PersistThreadblock`

所以用户流程图在概念上是对的；源码实现上，后半段是 target-specific pipeline。

## adapter 如何变成 Python callable

`tilelang.lower` 返回：

```python
CompiledArtifact(host_mod, device_mod, params, kernel_source, rt_mod=...)
```

`JITKernel._compile_and_create_adapter(...)` 根据 execution backend 创建 adapter：

- `TVMFFIKernelAdapter`
- `CythonKernelAdapter`
- `NVRTCKernelAdapter`
- `MetalKernelAdapter`
- `CuTeDSLKernelAdapter`

然后：

```python
self.adapter = adapter
self.torch_function = adapter.func
```

`JITKernel.__call__` 只是转发：

```python
return self.torch_function(*args, **kwds)
```

因此最终 Python callable 是 adapter 生成的 `func`。

## 补充：编译期参数、运行期参数和 JIT 编译选项

这里主要区分三类信息：

- 编译期参数
- 运行期参数
- JIT 编译选项

在前面说：

```text
p1_key: 非 tensor 编译期参数
p2_key: 从 tensor shape/stride 或显式 constexpr 推导出的动态形状 key
```

更准确地说，`p1_key` 指的是：

```text
用户 @tilelang.jit Python 函数参数里，除 tensor 参数以外、留在 kwargs 里的那些参数。
```

这些参数通常在生成 TIR/kernel 之前就要确定，所以称为 DSL 层面的“编译期参数”。
这里的“编译期”描述的是使用时机：它们决定生成什么 TIR/kernel。

### 编译期参数是什么

例如：

```python
@tilelang.jit
def matmul(A, B, C, block_M=128, block_N=128, block_K=32, dtype="float16"):
    ...
```

这里：

```text
A, B, C
```

是 tensor 参数。

而：

```text
block_M, block_N, block_K, dtype
```

是用户 JIT 函数的普通 Python 参数。它们通常会影响生成出来的 TIR/kernel，
比如：

- `block_M` / `block_N`: 决定 tile 形状、loop 范围、shared memory 大小
- `block_K`: 决定 K 方向分块、pipeline stage、load/store 结构
- `dtype`: 决定 buffer dtype、intrinsic 选择、codegen 类型
- `num_warps` / `threads`: 决定线程布局、thread binding、launch config
- `use_tma` / `use_wgmma` / `enable_xxx`: 决定走哪套 lowering 或 intrinsic 路径

所以：

```python
matmul(A, B, C, block_M=128)
matmul(A, B, C, block_M=64)
```

一般会得到不同的 `p1_key`，也可能生成不同 `PrimFunc` 和不同 kernel。

源码上，`JITFunc._parse_phase1_key(...)` 会把 tensor 参数摘出来，剩余 kwargs
组成：

```python
p1_key = tuple(sorted(kwargs.items()))
```

所以 `p1_key` 的来源是用户函数调用参数中的非 tensor 部分。

### 运行期参数是什么

运行期参数是 kernel 执行时才传进去的数据，通常就是 tensor 对象本身：

```python
matmul(A, B, C, block_M=128)
```

这里 `A`、`B`、`C` 的 data pointer/storage 是运行期数据。它们不会作为
Python 对象本身进入 `p1_key`。否则每换一个 tensor 对象都会触发重新编译，
这显然不合理。

命中或编译得到 `JITKernel` 后，eager 模式最终执行：

```python
kernel(*kernel_args.values())
```

也就是把 runtime tensor 参数传给已经编译好的 callable。

一句话：

```text
编译期参数决定“生成什么 kernel”；
运行期参数决定“这个 kernel 这次处理哪块数据”。
```

### p2_key 为什么来自 tensor shape/stride

虽然 tensor 对象本身是运行期参数，但 tensor 的 shape/stride 可能影响
kernel 生成。

例如 DSL 中有 constexpr 变量依赖 tensor shape：

```python
M, N = T.const("M N")
A: T.Tensor((M, N), "float16")
```

如果第一次传入：

```text
A.shape == (1024, 1024)
```

第二次传入：

```text
A.shape == (2048, 1024)
```

那么生成的 loop 边界、buffer shape、index 计算或 launch config 可能不同。
这类从 tensor 元信息推出来的值进入 `p2_key`。

所以：

```python
matmul(A1, B1, C1, block_M=128)
matmul(A2, B2, C2, block_M=128)
```

如果 `A1/B1/C1` 和 `A2/B2/C2` 的 shape/stride 一样：

```text
p1_key 一样
p2_key 一样
=> 复用同一个 compiled kernel
=> 执行时传不同 tensor data pointer
```

如果 shape 或 stride 变了：

```text
p1_key 可能一样
p2_key 变了
=> 可能编译另一个 kernel
```

### JIT 编译选项在哪里

除了 `p1_key` / `p2_key` 之外，底层 `KernelCache` key 还会包含
`@tilelang.jit(...)` 装饰器参数或环境变量解析出来的 JIT 编译选项，例如：

```python
@tilelang.jit(
    out_idx=[2],
    target="cuda",
    target_host="llvm",
    execution_backend="tvm_ffi",
    verbose=True,
    pass_configs={...},
    compile_flags=["--use_fast_math"],
)
def matmul(...):
    ...
```

这些选项控制“怎么编译”和“用哪个 backend 包装执行”：

- `target`: 目标后端，比如 `cuda`、`hip`、`llvm`、`metal`
- `target_host`: host 侧目标，比如 `llvm` 或 `c`
- `execution_backend`: 执行包装方式，比如 `tvm_ffi`、`cython`、`nvrtc`、`torch`、`cutedsl`
- `out_idx`: 哪些参数作为输出返回
- `pass_configs`: 传给 TVM/TileLang `PassContext` 的配置
- `compile_flags`: 传给 device compiler 的额外 flags
- `verbose`: 控制日志

这些选项不进入 `p1_key`，而是进入更底层的 `KernelCache` key：

```python
key_data = {
    "func": sha256(func.script(show_meta=True)).hexdigest(),
    "out_idx": ...,
    "target": str(target),
    "target_host": str(target_host),
    "execution_backend": execution_backend,
    "pass_configs": pass_configs,
    "compile_flags": compile_flags,
    ...
}
```

### 三类参数的关系

可以用一个例子汇总：

```python
@tilelang.jit(target="cuda", execution_backend="tvm_ffi")
def matmul(A, B, C, block_M=128, block_N=128):
    ...

matmul(A, B, C, block_M=128)
```

拆开看：

```text
runtime tensor 参数:
  A, B, C

DSL 编译期参数，也就是 p1_key 主要来源:
  block_M=128
  block_N=128

tensor 元信息动态 key，也就是 p2_key:
  A.shape / A.stride
  B.shape / B.stride
  C.shape / C.stride
  以及由这些推导出的 constexpr

JIT 编译选项:
  target="cuda"
  execution_backend="tvm_ffi"
  out_idx
  pass_configs
  compile_flags
```

因此更精确的表述是：

```text
p1_key:
  来自用户 JIT 函数调用参数中的非 tensor 参数。
  它们通常是 DSL 层面的编译期配置。

p2_key:
  来自 tensor shape/stride 或显式 constexpr 的动态形状 key。

KernelCache key:
  在 PrimFunc 内容之外，还额外包含 target/backend/pass_configs/compile_flags 等 JIT 编译选项。
```

## 小结

`JITImpl` 本身确实有 kernel cache，但它只是每个 decorated function 对象里的
普通 dict。更底层还有 backend `KernelCache`，负责进程内和磁盘 cache。

可以把几层 key/value 总结成：

```text
JITFunc.p1_cache
  key: p1_key = tuple(sorted(non_tensor_kwargs.items()))
  value: TirTemplate
  hit: 只跳过模板构造，不能直接执行，还要继续查 kernel cache

JITImpl._kernel_cache
  key: (p1_key, p2_key)
  value: JITKernel
  hit: 可直接执行或返回，不再访问底层 KernelCache

KernelCache._memory_cache
  key: sha256(PrimFunc.script(show_meta=True), target, backend, configs, version, lib stamp, ...)
  value: JITKernel
  hit: 不再访问 disk cache，也不调用 tilelang.lower

Disk kernel cache
  key: same kernel_key as KernelCache._memory_cache
  value: source files + executable/cubin/shared library + params.pkl + optional PrimFunc/resource metadata
  hit: from_database 重建 JITKernel，不调用 tilelang.lower/codegen

Frontend cache
  key: sha256(function source/signature/name + JIT key + compile options)
  value: {"kernel_key": "..."}
  hit: 找到 kernel_key 后仍要加载对应 disk artifact
```

线程上，前端 builder 状态是 thread-local；这些 cache 不是 thread-local。
GIL 能保护单个 Python dict 操作的内存安全，但不能让 cache miss、compile、
store 这组逻辑变成原子操作。因此并发首次调用时，重复编译是合理预期；当前
源码没有实现 per-key single-flight 编译锁。
