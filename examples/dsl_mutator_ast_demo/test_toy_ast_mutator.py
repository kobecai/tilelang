from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from toy_ast_mutator import TraceBuilder, plain_reference, toy_kernel  # noqa: E402


class ToyASTMutatorTests(unittest.TestCase):
    def test_rewritten_function_matches_regular_python(self) -> None:
        builder = TraceBuilder()

        result = toy_kernel(5, 4, builder=builder)

        self.assertEqual(result, plain_reference(5, 4))

    def test_rewritten_source_shows_builder_calls(self) -> None:
        source = toy_kernel.rewritten_source

        self.assertIn("def toy_kernel(__tb, n, limit):", source)
        self.assertIn("__tb.bind('acc'", source)
        self.assertIn("__tb.ctx_for", source)
        self.assertIn("__tb.ctx_if", source)
        self.assertIn("__tb.ctx_then", source)
        self.assertIn("__tb.ctx_else", source)
        self.assertIn("__tb.ret", source)
        self.assertNotIn("@toy_jit", source)

    def test_trace_records_builder_control_flow(self) -> None:
        builder = TraceBuilder()

        toy_kernel(3, 2, builder=builder)
        events = "\n".join(builder.events)

        self.assertIn("override range", events)
        self.assertIn("ctx_for [0, 1, 2]", events)
        self.assertIn("ctx_if False", events)
        self.assertIn("ctx_if True", events)
        self.assertIn("bind acc =", events)
        self.assertIn("ret", events)


if __name__ == "__main__":
    unittest.main()

