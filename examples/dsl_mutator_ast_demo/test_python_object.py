from __future__ import annotations

import textwrap
import types
import unittest


SOURCE = textwrap.dedent(
    """
    a = 123

    def foo(x):
        return x + 1
    """
)


class PythonObjectTests(unittest.TestCase):
    def test_exec_code_object_populates_namespace(self) -> None:
        code = compile(SOURCE, "<demo>", "exec")
        self.assertIsInstance(code, types.CodeType)

        namespace: dict[str, object] = {}
        self.assertNotIn("a", namespace)
        self.assertNotIn("foo", namespace)

        exec(code, namespace)

        self.assertEqual(namespace["a"], 123)
        self.assertIsInstance(namespace["foo"], types.FunctionType)

        foo = namespace["foo"]
        self.assertEqual(foo(10), 11)
        self.assertIsInstance(foo.__code__, types.CodeType)


def main() -> None:
    code = compile(SOURCE, "<demo>", "exec")

    namespace: dict[str, object] = {}
    print("exec 前 namespace 里的用户定义名字:", sorted(k for k in namespace if not k.startswith("__")))

    exec(code, namespace)

    foo = namespace["foo"]
    print("exec 后 namespace['a']:", namespace["a"])
    print("exec 后 namespace['foo']:", foo)
    print("namespace['foo'] 的类型:", type(foo))
    print("namespace['foo'].__code__ 的类型:", type(foo.__code__))
    print("namespace['foo'](10):", foo(10))


if __name__ == "__main__":
    main()
