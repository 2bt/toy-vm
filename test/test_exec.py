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
    asm "    int #3"

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

    def test_unknown_variable(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "test") as directory:
            src = Path(directory) / "main.toy"
            src.write_text("func tick():\n    x = 1\n")
            compiled = subprocess.run(
                [sys.executable, ROOT / "compile.py", src],
                cwd=ROOT, capture_output=True, text=True,
            )
        self.assertNotEqual(compiled.returncode, 0)
        self.assertIn("unknown variable", compiled.stderr)


if __name__ == "__main__":
    unittest.main()
