# DSL mutator AST demo

This directory is a standalone, standard-library-only demo of the core idea
behind `tilelang/language/eager/ast.py::DSLMutator`.

It does not import TileLang, TVM, CUDA, or any other project code. The goal is
to make Python AST rewriting easy to inspect.

## What it demonstrates

`DSLMutator` takes the source of a Python DSL function and rewrites normal
Python syntax into calls on a builder object. In TileLang, that builder is a
`BaseBuilder` subclass such as `tilelang.language.eager.builder.Builder`.

This toy demo shows the same shape with fewer cases:

- `x = value` becomes `x = __tb.bind("x", value)`
- `for i in range(n)` becomes `for __0 in __tb.ctx_for(range(n))`
- `if cond` becomes `for __1 in __tb.ctx_if(cond)` plus then/else builder hooks
- `return value` becomes `return __tb.ret(value)`

The important Python APIs are:

- `inspect.getsource()` to read the decorated function source
- `ast.parse()` to build a syntax tree
- `ast.NodeTransformer` to replace syntax nodes
- `ast.unparse()` to print the rewritten Python
- `compile()` and `exec()` to turn the rewritten tree back into a function

## Run it

From the repository root:

```bash
python examples/dsl_mutator_ast_demo/toy_ast_mutator.py
python -m unittest discover -s examples/dsl_mutator_ast_demo -v
```

Or from this directory:

```bash
python toy_ast_mutator.py
python -m unittest -v
```

