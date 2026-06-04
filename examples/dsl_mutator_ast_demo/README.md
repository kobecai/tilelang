# DSLMutator AST 改写示例

这个目录是一个完全独立的教学示例，用 Python 标准库演示
`tilelang/language/eager/ast.py::DSLMutator` 的核心思想。

它不导入 TileLang、TVM、CUDA，也不依赖仓库里的其他代码。目标是把
Python AST 改写的过程讲清楚，方便直接运行和观察。

## 这个示例展示什么

`DSLMutator` 的作用是拿到一个 Python DSL 函数的源码，然后把普通 Python
语法改写成对 builder 对象的调用。在 TileLang 里，这个 builder 是
`BaseBuilder` 的子类，比如 `tilelang.language.eager.builder.Builder`。

这个 toy demo 只保留少量语法，展示同一种结构：

- `x = value` 改写成 `x = __tb.bind("x", value)`
- `for i in range(n)` 改写成 `for __0 in __tb.ctx_for(range(n))`
- `if cond` 改写成 `for __1 in __tb.ctx_if(cond)`，再配合 then/else hook
- `return value` 改写成 `return __tb.ret(value)`

## 用到的 Python 标准库和 API

- `inspect.getsource()`：读取被装饰函数的源码
- `inspect.getsourcelines()`、`inspect.getsourcefile()`、`inspect.getfile()`：
  在需要保留原始行号和文件名时使用
- `ast.parse()`：把源码解析成 AST
- `ast.NodeTransformer`：遍历并替换 AST 节点，例如 `ast.If`、`ast.For`、
  `ast.Assign`、`ast.Return`
- `ast.fix_missing_locations()`：给新生成的 AST 节点补齐行号/列号信息
- `ast.unparse()`：把改写后的 AST 打印回 Python 源码，方便调试
- `compile()`：把 AST 或源码编译成 code object
- `exec()`：执行 code object，让其中的 `def` 语句真正创建 function object
- `eval()`：在需要从原函数环境里求值某些表达式时使用，TileLang 里会用到

## AST 改写流程

整个过程可以理解成 source-to-source transformation，只不过中间表示是
Python AST：

```text
Python function
  -> inspect 读取函数源码
  -> ast.parse 生成 AST
  -> ast.NodeTransformer 把普通 Python 语法替换成 builder 调用
  -> ast.fix_missing_locations 修复新 AST 节点的源码位置信息
  -> compile 生成 Python code object
  -> exec 执行 code object，得到改写后的 function object
```

改写后的代码仍然是合法 Python。区别在于，原本由 Python 语义直接处理的赋值、
循环、分支、return，现在会先进入名为 `__tb` 的 builder 对象。

普通赋值：

```python
acc = acc + i
```

会改写成：

```python
acc = __tb.bind("acc", acc + i)
```

普通分支：

```python
if acc > limit:
    acc = acc - 1
else:
    acc = acc + 2
```

会改写成 builder 控制的结构：

```python
for __0 in __tb.ctx_if(acc > limit):
    for _ in __tb.ctx_then(__0):
        acc = __tb.bind("acc", acc - 1)
    for _ in __tb.ctx_else(__0):
        acc = __tb.bind("acc", acc + 2)
```

所以改写后的函数仍然可以由 Python 解释器执行。Python 负责执行这段合法的
控制流骨架；`__tb` builder 决定 `bind`、`ctx_for`、`ctx_if`、`ctx_then`、
`ctx_else`、`ret` 的具体含义。

## 为什么 `compile()` 后还要 `exec()` 才能得到 function object

这是 Python 底层执行模型里一个容易混淆的点：`compile()` 不会直接创建函数对象。

比如这段源码：

```python
def foo(x):
    return x + 1
```

执行：

```python
code = compile(source, "<demo>", "exec")
```

得到的是一个 code object。这个 code object 表示“一段模块级代码应该如何执行”。
它里面包含了“执行到 `def foo...` 时如何创建函数”的指令，但此时 `foo` 这个
function object 还没有被创建。

只有当你执行：

```python
namespace = {}
exec(code, namespace)
```

Python 解释器才会真正运行这段模块级 code object。运行到 `def foo...` 语句时，
解释器会创建一个 function object，并把它绑定到命名空间里：

```python
namespace["foo"]
```

所以流程不是：

```text
compile -> function object
```

而是：

```text
compile -> module-level code object
exec(code object) -> 执行 def 语句 -> 创建并绑定 function object
```

可以把 `compile()` 理解成“把源码/AST 变成可执行说明书”，而 `exec()` 才是
“照着说明书实际执行一遍”。`def` 本身是一条可执行语句；只有执行它，函数对象
才会出现。

这也是为什么 demo 里的 `rewrite_function()` 会这样做：

```python
compiled = compile(tree, filename="<toy_ast_mutator>", mode="exec")
locals_ns = {}
exec(compiled, globals_ns, locals_ns)
rewritten_func = locals_ns[func.__name__]
```

`locals_ns[func.__name__]` 里拿到的就是执行 `def toy_kernel...` 后创建出来的
新函数对象。

## code object 和 function object 的区别

可以把它们分成两层：

- code object 是“代码本身的编译结果”
- function object 是“可以被调用的函数对象”

比如：

```python
def foo(x):
    return x + 1
```

Python 编译这段代码时，会涉及两类 code object：

1. 外层模块代码的 code object
2. `foo` 函数体自己的 code object

但刚编译完时，不一定已经有 `foo` 这个函数对象。函数对象是在执行
`def foo...` 这条语句时创建的。

### code object 是什么

code object 是 Python 编译器生成的不可变对象，里面保存的是“怎么执行这段代码”
的信息，例如：

- 字节码指令
- 常量表
- 变量名
- 参数个数
- 文件名
- 行号信息
- 函数体里用到的名字

可以把它理解成“模块体或函数体的指令蓝图”。它本身不是普通函数，不能像
`foo(1)` 那样直接调用。

例子：

```python
source = "def foo(x):\n    return x + 1\n"

code = compile(source, "<demo>", "exec")
print(type(code))
```

输出类型是：

```python
<class 'code'>
```

这里的 `code` 表示“执行这段模块级代码”。它还不是 `foo`。

### function object 是什么

function object 是运行时对象，也就是平时说的“函数”。它除了持有函数体的
code object，还带着运行时上下文：

- `__code__`：函数体的 code object
- `__globals__`：函数执行时使用的全局命名空间
- `__defaults__`：默认参数
- `__closure__`：闭包变量
- `__name__`：函数名
- `__annotations__`：类型注解

所以可以粗略理解为：

```text
function object = code object + 执行环境 + 函数元信息
```

继续上面的例子：

```python
namespace = {}
exec(code, namespace)

foo = namespace["foo"]

print(type(foo))
print(type(foo.__code__))
print(foo(10))
```

会得到类似结果：

```python
<class 'function'>
<class 'code'>
11
```

这里能看到两者的联系：`foo` 是 function object；`foo.__code__` 是它内部持有的
函数体 code object。

### `exec(code, namespace)` 到底做了什么

这段代码的关键是 `namespace`：

```python
namespace = {}
exec(code, namespace)

foo = namespace["foo"]

print(type(foo))
print(type(foo.__code__))
print(foo(10))
```

`exec(code, namespace)` 的意思是：执行 `code` 这段模块级 code object，并把执行
过程中创建出来的名字放进 `namespace` 这个字典。

假设 `code` 来自这段源码：

```python
source = """
a = 123

def foo(x):
    return x + 1
"""

code = compile(source, "<demo>", "exec")
```

刚执行完 `compile()` 时，`code` 只是“编译好的模块级代码”。它还没有真的运行，
所以此时不会有 `a`，也不会有 `foo`。

```python
namespace = {}
print(namespace)
```

这时 `namespace` 还是空的：

```python
{}
```

执行：

```python
exec(code, namespace)
```

Python 会在 `namespace` 这个命名空间里运行 `code`。运行过程大致相当于：

```python
# 在 namespace 里执行
a = 123

def foo(x):
    return x + 1
```

执行完以后，`namespace` 里就多了这些用户定义的名字：

```python
namespace["a"]    # 123
namespace["foo"]  # <function foo ...>
```

所以后面才能写：

```python
foo = namespace["foo"]
```

这一步不是在“重新定义函数”，而是从字典里把刚刚由 `exec()` 创建并绑定进去的
函数对象取出来。

完整观察代码可以写成：

```python
namespace = {}
print("exec 前:", sorted(k for k in namespace if not k.startswith("__")))

exec(code, namespace)

print("exec 后 a:", namespace["a"])
print("exec 后 foo:", namespace["foo"])
print("foo 的类型:", type(namespace["foo"]))
print("foo.__code__ 的类型:", type(namespace["foo"].__code__))
print("foo(10):", namespace["foo"](10))
```

你会看到：

```text
exec 前: []
exec 后 a: 123
exec 后 foo: <function foo ...>
foo 的类型: <class 'function'>
foo.__code__ 的类型: <class 'code'>
foo(10): 11
```

一句话总结：`compile()` 负责生成 code object；`exec(code, namespace)` 负责
真正执行这个 code object，并把执行期间定义出来的变量、函数、类等名字放进
`namespace`。

### 为什么 `def` 很关键

`def` 在 Python 里不是纯声明，而是一条会被执行的语句。概念上，这段代码：

```python
def foo(x):
    return x + 1
```

执行时可以粗略理解成：

```python
foo_code = <函数体的 code object>
foo = function(foo_code, globals, name="foo")
```

真实 CPython 实现更复杂，但这个模型足够解释这里的问题。

因此，`compile(source, ..., "exec")` 只是得到“模块代码的 code object”。
`exec(code, namespace)` 才会执行模块代码。执行过程中遇到 `def foo...`，
解释器才创建 `foo` 这个 function object，并放进 `namespace`。

一句话总结：

```text
code object: 编译后的指令蓝图，描述“怎么执行”
function object: 可调用的运行时函数，包装了 code object 和执行环境
```

## 改写后的结果是不是 Callable

是。经过 `compile()` 和 `exec()` 后，改写后的函数是一个真正的 Python
function object，所以它是 callable。

在这个 demo 里，`rewrite_function()` 把这个函数放进 `RewrittenFunction`
容器里。`@toy_jit` wrapper 调用它时是这样的：

```python
rewritten.function(tb, *args, **kwargs)
```

在 TileLang 里，`mutate()` 返回的是 `IRGenerator`，不是裸函数。它的 `gen`
字段是 callable，形状大概是：

```python
Callable[[BaseBuilder], Callable[..., Any]]
```

也就是说，第一层调用传入 builder，返回一个使用该 builder 的改写后 callable。
TileLang 的 builder 不只是计算普通 Python 值，它会记录 DSL 操作并构造 IR。

## 运行方式

从仓库根目录运行：

```bash
python examples/dsl_mutator_ast_demo/toy_ast_mutator.py
python -m unittest discover -s examples/dsl_mutator_ast_demo -v
```

或者进入当前目录运行：

```bash
python toy_ast_mutator.py
python -m unittest -v
```
