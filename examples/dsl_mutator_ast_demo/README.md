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

## `NodeTransformer.visit()` 如何调用各个 `visit_xxx` 方法

demo 里的核心调用是：

```python
tree = ToyDSLMutator().visit(tree)
```

这里的 `visit()` 不是 `ToyDSLMutator` 自己定义的，而是继承自
`ast.NodeTransformer`。它会根据当前 AST 节点的类型，动态拼出要调用的方法名：

```python
method_name = "visit_" + node.__class__.__name__
```

比如：

- 当前节点是 `ast.FunctionDef`，就尝试调用 `visit_FunctionDef(node)`
- 当前节点是 `ast.For`，就尝试调用 `visit_For(node)`
- 当前节点是 `ast.If`，就尝试调用 `visit_If(node)`
- 当前节点是 `ast.Assign`，就尝试调用 `visit_Assign(node)`

可以把它粗略理解成下面这个简化版：

```python
def visit(self, node):
        method_name = "visit_" + node.__class__.__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
```

所以不需要手动写：

```python
mutator.visit_FunctionDef(...)
mutator.visit_For(...)
mutator.visit_Assign(...)
```

只要从根节点调用一次：

```python
mutator.visit(tree)
```

`NodeTransformer` 就会在递归遍历 AST 时，自动把不同类型的节点分发给对应的
`visit_xxx` 方法。

## 是否必须实现所有 AST 节点的 `visit_xxx`

不需要。一个 mutator 只需要实现自己关心、想改写的节点类型。

这个 demo 只实现了：

```python
visit_FunctionDef
visit_Assign
visit_For
visit_If
visit_Expr
visit_Return
```

也就是说，它只特殊处理函数定义、赋值、`for`、`if`、表达式语句和 `return`。
像 `Module`、`BinOp`、`Call`、`Name`、`Constant`、`Compare` 这些节点没有专门的
`visit_xxx` 方法，就会走默认的 `generic_visit()`。

Python AST 节点类型由 Python 语法定义，是有限的一批，但不止几个。常见的有：

```text
Module
FunctionDef
ClassDef
Return
Assign
AnnAssign
AugAssign
For
While
If
With
Try
Expr
Call
Name
Constant
BinOp
UnaryOp
Compare
BoolOp
Attribute
Subscript
List
Tuple
Dict
Lambda
Import
ImportFrom
```

还有更细的节点或 AST 辅助对象，比如 `Add`、`Sub`、`Gt`、`Load`、`Store` 等。
实际写 DSL mutator 时，通常不会全部处理，只会处理 DSL 语义需要拦截的那部分。

## `generic_visit()` 对未自定义节点做什么

`generic_visit(node)` 可以理解成默认递归器。它对当前节点本身不做特殊改写，
但会继续访问当前节点里面的子节点：

```text
generic_visit(node)
    遍历 node 的每个字段
    如果字段里有 AST 子节点，就对子节点调用 self.visit(child)
    如果 child 被改写了，就把新的 child 放回去
    最后返回当前 node
```

所以它不是完全什么都不做，而是：当前节点默认保留，里面的孩子仍然有机会被
自定义的 `visit_xxx` 改写。

例如原代码：

```python
def toy_kernel(n, limit):
        acc = 0
        return acc + limit
```

AST 大致是：

```text
Module
    FunctionDef
        arguments
        Assign
        Return
            BinOp
                Name("acc")
                Add
                Name("limit")
```

这个 demo 没有实现 `visit_Module`、`visit_arguments`、`visit_BinOp`、
`visit_Name`、`visit_Add`，但是实现了 `visit_FunctionDef`、`visit_Assign`、
`visit_Return`。

遍历流程可以理解成：

```text
visit(Module)
    没有 visit_Module
    调用 generic_visit(Module)

    generic_visit(Module) 进入 body
        visit(FunctionDef)
            有 visit_FunctionDef，改写函数签名并继续遍历函数体

            visit(Assign)
                有 visit_Assign，把 acc = 0 改写成 __tb.bind(...)

            visit(Return)
                有 visit_Return
                Return 里的 BinOp 没有自定义 visit_BinOp，所以 BinOp 原样保留
                最后把 return value 包成 __tb.ret(value)
```

最后改写结果大致是：

```python
def toy_kernel(__tb, n, limit):
        range = __tb.override('range')
        acc = __tb.bind('acc', 0)
        return __tb.ret(acc + limit)
```

注意这里的：

```python
acc + limit
```

对应的是 `BinOp`。因为 demo 没有实现 `visit_BinOp`，所以这个表达式本身保持原样；
它只是被外层的 `visit_Return` 包进了 `__tb.ret(...)`。

再看一个赋值表达式的例子：

```python
x = foo(y + 1)
```

AST 大致是：

```text
Assign
    target: Name("x")
    value: Call
        func: Name("foo")
        args:
            BinOp
                Name("y")
                Add
                Constant(1)
```

因为 demo 有 `visit_Assign`，外层赋值会被改写：

```python
x = __tb.bind('x', foo(y + 1))
```

但 demo 没有 `visit_Call`、`visit_BinOp`、`visit_Name`、`visit_Constant`，所以：

```python
foo(y + 1)
```

这部分表达式本身保持原样。

一句话总结：`generic_visit()` 对当前节点默认保留原样，但会继续深入它的子节点，
让那些有自定义 `visit_xxx` 的子节点被改写。

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
