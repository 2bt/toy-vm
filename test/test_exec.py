#!/usr/bin/env python3
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "build" / "toy"

HEADER = """
func print(msg: int* @ 18):
    asm "    int #1"

var digit: int[2]

func print_n(n):
    digit[0] = n + 48
    digit[1] = 0
    print(digit.ptr)
"""


class ExecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(
            ["cmake", "--build", "build"],
            cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
        )

    def run_toy(self, source):
        with tempfile.TemporaryDirectory(dir=ROOT / "test") as directory:
            src = Path(directory) / "main.toy"
            src.write_text(HEADER + textwrap.dedent(source))
            compiled = subprocess.run(
                [sys.executable, ROOT / "compile.py", src],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            assembled = subprocess.run(
                [sys.executable, ROOT / "assemble.py", src.with_suffix(".asm")],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(assembled.returncode, 0, assembled.stderr)
            result = subprocess.run(
                [TOY, "--once", directory],
                cwd=ROOT, capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        return result.stdout

    def test_print_string(self):
        self.assertEqual(self.run_toy("""
            func tick():
                print("ok")
        """), "ok\n")

    def test_call_and_add(self):
        self.assertEqual(self.run_toy("""
            func add(a, b):
                return a + b
            func tick():
                print_n(add(2, 3))
        """), "5\n")

    def test_pointer_deref(self):
        self.assertEqual(self.run_toy("""
            func tick():
                var p = digit.ptr
                p.* = 6
                print_n(p.*)
        """), "6\n")

    def test_while_sum(self):
        self.assertEqual(self.run_toy("""
            func tick():
                var n = 0
                var i = 0
                while i < 4:
                    n += i
                    i += 1
                print_n(n)
        """), "6\n")

    def test_for_sum(self):
        self.assertEqual(self.run_toy("""
            func tick():
                var n = 0
                for i = 1, 4:
                    n += i
                print_n(n)
        """), "6\n")

    def test_for_continue(self):
        self.assertEqual(self.run_toy("""
            func tick():
                var n = 0
                for i = 0, 5:
                    if i == 2:
                        continue
                    n += 1
                print_n(n)
        """), "4\n")

    def test_while_update_is_conditional(self):
        self.assertEqual(self.run_toy("""
            func tick():
                var i = 2
                var k = 0
                while i:
                    if k == 0:
                        k = 1
                    else:
                        i -= 1
                print_n(k)
                print_n(i)
        """), "1\n0\n")

    def test_if_join_when_condition_is_true(self):
        self.assertEqual(self.run_toy("""
            func tick():
                var x = 0
                var y = 0
                var z = 0
                if x == 0:
                    y = 5
                z = 9
                print_n(y)
                print_n(z)
        """), "5\n9\n")

    def test_if_join_when_condition_is_false(self):
        self.assertEqual(self.run_toy("""
            func tick():
                var x = 1
                var y = 0
                var z = 0
                if x == 0:
                    y = 5
                z = 9
                print_n(y)
                print_n(z)
        """), "0\n9\n")

    def compile_text(self, source):
        with tempfile.TemporaryDirectory(dir=ROOT / "test") as directory:
            src = Path(directory) / "main.toy"
            src.write_text(textwrap.dedent(source))
            return subprocess.run(
                [sys.executable, ROOT / "compile.py", src],
                cwd=ROOT, capture_output=True, text=True,
            )

    def test_self_call_is_unknown(self):
        compiled = self.compile_text("""
            func tick():
                tick()
        """)
        self.assertNotEqual(compiled.returncode, 0)
        self.assertIn("unknown function", compiled.stderr)

    def test_unknown_variable(self):
        compiled = self.compile_text("func tick():\n    x = 1\n")
        self.assertNotEqual(compiled.returncode, 0)
        self.assertIn("unknown variable", compiled.stderr)

    def test_constant_divide_by_zero(self):
        for expr in ("1 / 0", "1 % 0", "1 / (2 - 2)", "1 % (2 - 2)"):
            compiled = self.compile_text(f"""
                func tick():
                    var n = {expr}
            """)
            self.assertNotEqual(compiled.returncode, 0, expr)
            self.assertIn("division by zero", compiled.stderr)

    def test_constant_math_matches_vm(self):
        self.assertEqual(self.run_toy("""
            func tick():
                var a = -3
                var b = 2
                print_n((-3 % 2) == (a % b))
                print_n((-3 % 2) == -1)
                print_n((-3 / 2) == (a / b))
                print_n((0 or 4) == 1)
                print_n((5 and 3) == 1)
                print_n((2 or 0) == 1)
        """), "1\n1\n1\n1\n1\n1\n")

    def test_explicit_output_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            src = d / "main.toy"
            asm = d / "out.asm"
            code = d / "out.code"
            src.write_text("func tick():\n    return\n")
            compiled = subprocess.run(
                [sys.executable, ROOT / "compile.py", src, asm],
                cwd=directory, capture_output=True, text=True,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            self.assertTrue(asm.is_file())
            assembled = subprocess.run(
                [sys.executable, ROOT / "assemble.py", asm, code],
                cwd=directory, capture_output=True, text=True,
            )
            self.assertEqual(assembled.returncode, 0, assembled.stderr)
            self.assertGreater(code.stat().st_size, 0)

    def compile_asm(self, source):
        with tempfile.TemporaryDirectory(dir=ROOT / "test") as directory:
            src = Path(directory) / "main.toy"
            src.write_text(textwrap.dedent(source))
            compiled = subprocess.run(
                [sys.executable, ROOT / "compile.py", src],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            return src.with_suffix(".asm").read_text()

    def test_inline_single_call(self):
        asm = self.compile_asm("""
            func add1(n):
                return n + 1
            func tick():
                var x = add1(4)
        """)
        self.assertNotIn("jsr add1", asm)
        self.assertNotIn("\nadd1:", asm)

    def test_keep_function_called_twice(self):
        asm = self.compile_asm("""
            func add1(n):
                return n + 1
            func tick():
                var x = add1(1)
                var y = add1(2)
        """)
        self.assertIn("jsr add1", asm)

    def test_inline_asm_wrapper_at_every_call(self):
        asm = self.compile_asm("""
            func bump(n):
                asm "    add bump.n, #1"
            func tick():
                bump(1)
                bump(2)
        """)
        self.assertNotIn("jsr bump", asm)
        self.assertNotIn("\nbump:", asm)
        self.assertEqual(asm.count("add bump.n, #1"), 2)

    def test_inlined_branch_still_runs(self):
        self.assertEqual(self.run_toy("""
            func add1(n):
                if n < 0:
                    return 0
                return n + 1
            func tick():
                print_n(add1(4))
        """), "5\n")

    def test_assembler_can_run_twice(self):
        script = textwrap.dedent("""
            import sys
            from argparse import Namespace
            from pathlib import Path
            sys.path.insert(0, {root!r})
            import assemble
            for name in ("a", "b"):
                src = Path(name + ".asm")
                out = Path(name + ".code")
                src.write_text("_R = 0\\n.code\\n    ret\\n")
                assemble.asm(Namespace(src=src, out=out, print=False))
        """).format(root=str(ROOT))
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=directory, capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

if __name__ == "__main__":
    unittest.main()
