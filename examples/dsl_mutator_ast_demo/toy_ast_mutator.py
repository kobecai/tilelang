from __future__ import annotations

import ast
import copy
import functools
import inspect
import textwrap
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


class TraceBuilder:
    """A tiny builder that records the rewritten function's runtime behavior."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def override(self, name: str) -> Any:
        self.events.append(f"override {name}")
        if name == "range":
            return range
        raise KeyError(f"no override registered for {name!r}")

    def ctx_for(self, iterable: Iterable[Any]) -> Iterable[Any]:
        values = list(iterable)
        self.events.append(f"ctx_for {values!r}")
        return values

    def ctx_if(self, cond: Any) -> Iterable[Any]:
        self.events.append(f"ctx_if {cond!r}")
        yield cond

    def ctx_then(self, cond: Any) -> Iterable[None]:
        self.events.append(f"ctx_then {cond!r}")
        if cond:
            yield None

    def ctx_else(self, cond: Any) -> Iterable[None]:
        self.events.append(f"ctx_else {cond!r}")
        if not cond:
            yield None

    def bind(self, name: str, value: Any) -> Any:
        self.events.append(f"bind {name} = {value!r}")
        return value

    def eval(self, value: Any) -> Any:
        self.events.append(f"eval {value!r}")
        return value

    def ret(self, value: Any) -> Any:
        self.events.append(f"ret {value!r}")
        return value


class QuoteRewriter(ast.NodeTransformer):
    """Replace placeholder names and `pass` blocks in small AST templates."""

    def __init__(self, names: dict[str, ast.AST], passes: list[list[ast.stmt]] | None = None) -> None:
        self.names = names
        self.passes = list(passes or [])

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id not in self.names:
            return node
        replacement = copy.deepcopy(self.names[node.id])
        return ast.copy_location(replacement, node)

    def visit_Pass(self, node: ast.Pass) -> ast.AST | list[ast.stmt]:
        if not self.passes:
            return node
        return self.passes.pop(0)


def quote(template: str, *, passes: list[list[ast.stmt]] | None = None, **names: ast.AST) -> list[ast.stmt]:
    tree = ast.parse(textwrap.dedent(template))
    rewritten = QuoteRewriter(names, passes).visit(tree)
    assert isinstance(rewritten, ast.Module)
    return rewritten.body


class ToyDSLMutator(ast.NodeTransformer):
    """Small educational subset of TileLang's DSLMutator."""

    def __init__(self) -> None:
        self.tmp_counter = 0

    def get_tmp(self) -> str:
        name = f"__{self.tmp_counter}"
        self.tmp_counter += 1
        return name

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.decorator_list.clear()
        node.args.args.insert(0, ast.arg(arg="__tb"))
        for arg in node.args.args:
            arg.annotation = None
        node.returns = None

        node = self.generic_visit(node)
        node.body = quote("range = __tb.override('range')") + node.body
        return node

    def visit_Assign(self, node: ast.Assign) -> list[ast.stmt]:
        node = self.generic_visit(node)
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise NotImplementedError("this demo only rewrites simple name assignments")

        target = node.targets[0]
        return quote(
            f"target = __tb.bind('{target.id}', value)",
            target=target,
            value=node.value,
        )

    def visit_For(self, node: ast.For) -> list[ast.stmt]:
        node = self.generic_visit(node)
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError("this demo only rewrites `for name in ...` loops")

        tmp = self.get_tmp()
        tmp_value = ast.Name(id=tmp, ctx=ast.Load())
        bind_loop_var = quote(
            f"target = __tb.bind('{node.target.id}', value)",
            target=node.target,
            value=tmp_value,
        )
        return quote(
            f"for {tmp} in __tb.ctx_for(iterable):\n"
            "    pass\n",
            iterable=node.iter,
            passes=[bind_loop_var + node.body],
        )

    def visit_If(self, node: ast.If) -> list[ast.stmt]:
        node = self.generic_visit(node)
        tmp = self.get_tmp()

        if not node.orelse:
            return quote(
                f"for {tmp} in __tb.ctx_if(cond):\n"
                f"    for _ in __tb.ctx_then({tmp}):\n"
                "        pass\n",
                cond=node.test,
                passes=[node.body],
            )

        return quote(
            f"for {tmp} in __tb.ctx_if(cond):\n"
            f"    for _ in __tb.ctx_then({tmp}):\n"
            "        pass\n"
            f"    for _ in __tb.ctx_else({tmp}):\n"
            "        pass\n",
            cond=node.test,
            passes=[node.body, node.orelse],
        )

    def visit_Expr(self, node: ast.Expr) -> list[ast.stmt]:
        node = self.generic_visit(node)
        return quote("__tb.eval(value)", value=node.value)

    def visit_Return(self, node: ast.Return) -> list[ast.stmt]:
        node = self.generic_visit(node)
        if node.value is None:
            return quote("return __tb.ret(None)")
        return quote("return __tb.ret(value)", value=node.value)


@dataclass(frozen=True)
class RewrittenFunction:
    original_source: str
    rewritten_source: str
    function: Callable[..., Any]


def rewrite_function(func: Callable[..., Any]) -> RewrittenFunction:
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    tree = ToyDSLMutator().visit(tree)
    ast.fix_missing_locations(tree)

    rewritten_source = ast.unparse(tree)
    compiled = compile(tree, filename=f"<toy_ast_mutator:{func.__name__}>", mode="exec")
    globals_ns = dict(func.__globals__)
    locals_ns: dict[str, Any] = {}
    exec(compiled, globals_ns, locals_ns)
    return RewrittenFunction(source, rewritten_source, locals_ns[func.__name__])


def toy_jit(func: Callable[..., Any]) -> Callable[..., Any]:
    rewritten = rewrite_function(func)

    @functools.wraps(func)
    def wrapper(*args: Any, builder: TraceBuilder | None = None, **kwargs: Any) -> Any:
        tb = builder if builder is not None else TraceBuilder()
        result = rewritten.function(tb, *args, **kwargs)
        wrapper.last_builder = tb
        return result

    wrapper.original_source = rewritten.original_source
    wrapper.rewritten_source = rewritten.rewritten_source
    wrapper.rewritten_function = rewritten.function
    wrapper.last_builder = None
    return wrapper


def plain_reference(n: int, limit: int) -> int:
    acc = 0
    for i in range(n):
        acc = acc + i
        if acc > limit:
            acc = acc - 1
        else:
            acc = acc + 2
    return acc


@toy_jit
def toy_kernel(n: int, limit: int) -> int:
    acc = 0
    for i in range(n):
        acc = acc + i
        if acc > limit:
            acc = acc - 1
        else:
            acc = acc + 2
    return acc


def main() -> None:
    builder = TraceBuilder()
    result = toy_kernel(4, 3, builder=builder)

    print("=== original source ===")
    print(toy_kernel.original_source)
    print("=== rewritten source ===")
    print(toy_kernel.rewritten_source)
    print("=== result ===")
    print(f"plain_reference(4, 3) -> {plain_reference(4, 3)}")
    print(f"toy_kernel(4, 3)      -> {result}")
    print("=== builder trace ===")
    for event in builder.events:
        print(event)


if __name__ == "__main__":
    main()
