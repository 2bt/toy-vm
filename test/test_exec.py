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
func print(msg: int* @ 2):
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

    def test_nested_calls_distinct(self):
        # Arg-position calls attribute to the callee being prepared so formals
        # don't overlap callees; LTR into formals is then safe.
        self.assertEqual(self.run_toy("""
            func add(a, b):
                return a + b
            func mul(a, b):
                return a * b
            func tick():
                print_n(add(mul(2, 3), mul(1, 1)))
        """), "7\n")

    def test_call_preserves_live_temps(self):
        # Left side leaves a live _T; call setup must not reuse it (tmp_i=0).
        # print_n is single-digit: (9-2) - f(1+1) == 5.
        self.assertEqual(self.run_toy("""
            var x = 9
            var y = 2
            var a = 1
            var b = 1
            func f(n):
                return n
            func tick():
                print_n((x - y) - f(a + b))
                print_n(f(0))
        """), "5\n0\n")

    def test_call_reuses_temps_across_setup(self):
        # f(x-y) - f(a+b): after each call, setup temps are dead. Save/restore
        # of tmp_i should need only _T0 for the first call result (not _T1).
        with tempfile.TemporaryDirectory(dir=ROOT / "test") as directory:
            src = Path(directory) / "main.toy"
            src.write_text(HEADER + textwrap.dedent("""
                var x = 9
                var y = 2
                var a = 1
                var b = 1
                func f(n):
                    return n
                func tick():
                    print_n(f(x - y) - f(a + b))
                    print_n(f(0))
            """))
            compiled = subprocess.run(
                [sys.executable, ROOT / "compile.py", src],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            asm = src.with_suffix(".asm").read_text()
            self.assertIn("tick._T0", asm)
            self.assertNotIn("tick._T1", asm)

    def test_nested_same_function_forbidden(self):
        compiled = self.compile_text("""
            func add(a, b):
                return a + b
            func tick():
                add(add(5, 1), add(1, 1))
        """)
        self.assertNotEqual(compiled.returncode, 0)
        self.assertIn("recursive call", compiled.stderr)

    def test_quasi_recursion_forbidden(self):
        compiled = self.compile_text("""
            func a(x):
                return x
            func b(x):
                return a(x)
            func tick():
                a(b(a(1)))
        """)
        self.assertNotEqual(compiled.returncode, 0)
        self.assertIn("recursive call", compiled.stderr)

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

    def test_struct_assign(self):
        self.assertEqual(self.run_toy("""
            struct V:
                x
                y
            var a: V
            var b: V
            func tick():
                a.x = 3
                a.y = 7
                b = a
                print_n(b.x)
                print_n(b.y)
        """), "3\n7\n")

    def test_array_assign(self):
        self.assertEqual(self.run_toy("""
            var a: int[3]
            var b: int[3]
            func tick():
                a[0] = 1
                a[1] = 2
                a[2] = 3
                b = a
                print_n(b[0])
                print_n(b[1])
                print_n(b[2])
        """), "1\n2\n3\n")

    def test_struct_assign_via_pointer(self):
        self.assertEqual(self.run_toy("""
            struct V:
                x
                y
            var a: V
            var b: V
            func tick():
                a.x = 4
                a.y = 5
                var p = &b
                p.* = a
                print_n(b.x)
                print_n(b.y)
        """), "4\n5\n")

    def test_struct_assign_indexed(self):
        self.assertEqual(self.run_toy("""
            struct V:
                x
                y
            var vs: V[3]
            func tick():
                vs[0].x = 1
                vs[0].y = 2
                vs[2] = vs[0]
                print_n(vs[2].x)
                print_n(vs[2].y)
        """), "1\n2\n")

    def test_struct_assign_nested_field(self):
        self.assertEqual(self.run_toy("""
            struct Inner:
                a
                b
            struct Outer:
                i: Inner
                x
            var u: Outer
            var v: Outer
            func tick():
                u.i.a = 8
                u.i.b = 9
                u.x = 1
                v.i = u.i
                print_n(v.i.a)
                print_n(v.i.b)
                print_n(v.x)
        """), "8\n9\n0\n")

    def test_struct_assign_pointer_to_pointer(self):
        self.assertEqual(self.run_toy("""
            struct V:
                x
                y
            var a: V
            var b: V
            func tick():
                a.x = 6
                a.y = 7
                var src = &a
                var dst = &b
                dst.* = src.*
                print_n(b.x)
                print_n(b.y)
        """), "6\n7\n")

    def test_struct_self_assign(self):
        self.assertEqual(self.run_toy("""
            struct V:
                x
                y
            var a: V
            func tick():
                a.x = 2
                a.y = 3
                a = a
                print_n(a.x)
                print_n(a.y)
        """), "2\n3\n")

    def test_struct_assign_emits_word_moves(self):
        asm = self.compile_asm("""
            struct V:
                x
                y
                z
            var a: V
            var b: V
            func tick():
                b = a
        """)
        self.assertGreaterEqual(asm.count("mov b"), 3)
        self.assertIn("mov b+1, a+1", asm)
        self.assertIn("mov b+2, a+2", asm)

    def test_init_list_struct(self):
        self.assertEqual(self.run_toy("""
            struct V:
                x
                y
            var b: V
            func tick():
                b = { 3, 7 }
                print_n(b.x)
                print_n(b.y)
        """), "3\n7\n")

    def test_local_struct(self):
        self.assertEqual(self.run_toy("""
            struct V:
                x
                y
            func tick():
                var b: V = { 3, 7 }
                print_n(b.x)
                print_n(b.y)
        """), "3\n7\n")

    def test_local_array(self):
        self.assertEqual(self.run_toy("""
            func tick():
                var a: int[3] = { 1, 2, 3 }
                print_n(a[0])
                print_n(a[1])
                print_n(a[2])
        """), "1\n2\n3\n")

    def test_local_struct_copy(self):
        self.assertEqual(self.run_toy("""
            struct V:
                x
                y
            var g: V = { 4, 5 }
            func tick():
                var b: V = g
                print_n(b.x)
                print_n(b.y)
        """), "4\n5\n")

    def test_cannot_return_struct(self):
        compiled = self.compile_text("""
            struct V:
                x
                y
            func f() -> V:
                var b: V
                return b
            func tick():
                f()
        """)
        self.assertNotEqual(compiled.returncode, 0)
        self.assertIn("cannot return aggregate", compiled.stderr)

    def test_struct_arg(self):
        self.assertEqual(self.run_toy("""
            struct V:
                x
                y
            func show(v: V):
                print_n(v.x)
                print_n(v.y)
            func tick():
                var a: V = { 3, 7 }
                show(a)
        """), "3\n7\n")

    def test_struct_arg_init_list(self):
        self.assertEqual(self.run_toy("""
            struct V:
                x
                y
            func show(v: V):
                print_n(v.x)
                print_n(v.y)
            func tick():
                show({ 3, 7 })
        """), "3\n7\n")

    def test_array_arg_init_list(self):
        self.assertEqual(self.run_toy("""
            func show(a: int[3]):
                print_n(a[0])
                print_n(a[1])
                print_n(a[2])
            func tick():
                show({ 1, 2, 3 })
        """), "1\n2\n3\n")

    def test_init_list_trailing_comma(self):
        self.assertEqual(self.run_toy("""
            struct V:
                x
                y
            var b: V
            func tick():
                b = { 3, 7, }
                print_n(b.x)
                print_n(b.y)
        """), "3\n7\n")

    def test_init_list_array(self):
        self.assertEqual(self.run_toy("""
            var a: int[3]
            func tick():
                a = { 1, 2, 3 }
                print_n(a[0])
                print_n(a[1])
                print_n(a[2])
        """), "1\n2\n3\n")

    def test_init_list_expressions(self):
        self.assertEqual(self.run_toy("""
            struct V:
                x
                y
            var b: V
            var x = 4
            var y = 3
            func tick():
                b = { x + 1, y * 2 }
                print_n(b.x)
                print_n(b.y)
        """), "5\n6\n")

    def test_init_list_nested(self):
        self.assertEqual(self.run_toy("""
            struct Inner:
                a
                b
            struct Outer:
                i: Inner
                x
            var o: Outer
            func tick():
                o = { { 8, 9 }, 1 }
                print_n(o.i.a)
                print_n(o.i.b)
                print_n(o.x)
        """), "8\n9\n1\n")

    def test_init_list_aggregate_slot(self):
        self.assertEqual(self.run_toy("""
            struct Inner:
                a
                b
            struct Outer:
                i: Inner
                x
            var u: Outer
            var o: Outer
            func tick():
                u.i.a = 2
                u.i.b = 3
                o = { u.i, 1 }
                print_n(o.i.a)
                print_n(o.i.b)
                print_n(o.x)
        """), "2\n3\n1\n")

    def test_init_list_via_pointer(self):
        self.assertEqual(self.run_toy("""
            struct V:
                x
                y
            var b: V
            func tick():
                var p = &b
                p.* = { 4, 5 }
                print_n(b.x)
                print_n(b.y)
        """), "4\n5\n")

    def test_init_list_indexed(self):
        self.assertEqual(self.run_toy("""
            struct V:
                x
                y
            var vs: V[3]
            func tick():
                vs[2] = { 1, 2 }
                print_n(vs[2].x)
                print_n(vs[2].y)
        """), "1\n2\n")

    def test_init_list_pads_short(self):
        self.assertEqual(self.run_toy("""
            struct V:
                x
                y
            var b: V
            func tick():
                b = { 3 }
                print_n(b.x)
                print_n(b.y)
        """), "3\n0\n")

    def test_init_list_too_many(self):
        compiled = self.compile_text("""
            struct V:
                x
                y
            var b: V
            func tick():
                b = { 1, 2, 3 }
        """)
        self.assertNotEqual(compiled.returncode, 0)
        self.assertIn("wrong number of initializers", compiled.stderr)

    def test_init_list_onto_scalar(self):
        compiled = self.compile_text("""
            var n
            func tick():
                n = { 1 }
        """)
        self.assertNotEqual(compiled.returncode, 0)
        self.assertIn("type error", compiled.stderr)

    def test_init_list_type_mismatch(self):
        compiled = self.compile_text("""
            struct V:
                x
                y
            var b: V
            var p: int*
            func tick():
                b = { p, 1 }
        """)
        self.assertNotEqual(compiled.returncode, 0)
        self.assertIn("type error", compiled.stderr)

    def test_assembler_mov_nxt_nxt(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "test") as directory:
            src = Path(directory) / "t.asm"
            out = Path(directory) / "code"
            src.write_text(textwrap.dedent("""\
                _R = 0
                a = 10
                b = 20
                .code
                    mov b, a
                    mov b+1, a+1
                    mov b+2, a+2
                    ret
            """))
            result = subprocess.run(
                [sys.executable, ROOT / "assemble.py", src, out, "--print"],
                cwd=ROOT, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            # first mov keeps both operands; the next two should be opcode-only
            self.assertIn("mov ++%, ++%", result.stdout)
            self.assertEqual(result.stdout.count("mov ++%, ++%"), 2)
            # mov abs,abs (3) + two nxt,nxt (1 each) + ret (1)
            self.assertIn("codes: 6", result.stdout)

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
