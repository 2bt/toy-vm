#!/usr/bin/env python3
import atexit
import re
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import compile as C  # noqa: E402


def write_smolt(source: str) -> Path:
    d = Path(tempfile.mkdtemp(dir=ROOT / "test"))
    atexit.register(shutil.rmtree, d, True)
    p = d / "main.smolt"
    p.write_text(textwrap.dedent(source))
    return p


def parse_smolt(source: str) -> C.Parser:
    path = write_smolt(source)
    parser = C.Parser()
    parser.program(path)
    return parser


def compile_smolt(source: str):
    parser = parse_smolt(source)
    asm = C.Codegen().compile(parser)
    return parser, asm


def compile_fails(source: str) -> str:
    """Return stderr/message text; fail the test if compile succeeds."""
    path = write_smolt(source)
    try:
        parser = C.Parser()
        parser.program(path)
        C.Codegen().compile(parser)
    except SystemExit as e:
        return str(e)
    raise AssertionError("expected compile to fail")


class ScopeTests(unittest.TestCase):
    def test_same_scope_redeclare_errors(self):
        msg = compile_fails("""
            func tick():
                var x = 1
                var x = 2
        """)
        self.assertIn("already exists", msg)

    def test_shadowing_allowed_and_outer_restored(self):
        # Behavioral: after inner block, outer x is used again.
        parser, asm = compile_smolt("""
            func tick():
                var x = 1
                if 1:
                    var x = 2
                x = 3
        """)
        self.assertIn("tick.x", parser.vars)
        self.assertIn("tick.x.1", parser.vars)
        # assignment after if must refer to tick.x (unique without .n)
        self.assertRegex(asm, r"mov tick\.x, #3")

    def test_bound_for_index_cannot_assign(self):
        msg = compile_fails("""
            func tick():
                for i = 0, 3:
                    i = 1
        """)
        self.assertIn("bound", msg)

    def test_bound_for_index_cannot_addr(self):
        msg = compile_fails("""
            func tick():
                for i = 0, 3:
                    var p = &i
        """)
        self.assertIn("bound", msg)

    def test_for_index_reusable_in_later_for(self):
        parser, _ = compile_smolt("""
            func tick():
                for i = 0, 2:
                    var z = i
                for i = 0, 2:
                    var z = i
        """)
        # two loop indices minted
        self.assertIn("tick.i", parser.vars)
        self.assertIn("tick.i.1", parser.vars)

    def test_sibling_blocks_reuse_offset_and_frame_size(self):
        parser, asm = compile_smolt("""
            func tick():
                if 1:
                    var a: int[4]
                else:
                    var b: int[4]
        """)
        self.assertEqual(parser.vars["tick.a"].offset, parser.vars["tick.b"].offset)
        self.assertEqual(parser.funcs["tick"].frame_size, 4)
        # same absolute address in asm
        ma = re.search(r"^tick\.a = (\d+)", asm, re.M)
        mb = re.search(r"^tick\.b = (\d+)", asm, re.M)
        self.assertIsNotNone(ma)
        self.assertIsNotNone(mb)
        self.assertEqual(ma.group(1), mb.group(1))

    def test_nested_shadow_does_not_reuse_with_outer(self):
        parser, _ = compile_smolt("""
            func tick():
                var x = 1
                if 1:
                    var x = 2
        """)
        self.assertNotEqual(
            parser.vars["tick.x"].offset,
            parser.vars["tick.x.1"].offset,
        )
        self.assertEqual(parser.funcs["tick"].frame_size, 2)

    def test_missing_indent_errors(self):
        msg = compile_fails("""
            func tick():
                if 0:
                var x = 1
        """)
        self.assertIn("expected an indented block", msg)

    def test_empty_branch_before_elif_allowed(self):
        compile_smolt("""
            func tick():
                var c = 1
                if c == 0:
                elif c == 1: c = 2
        """)

    def test_var_named_like_const_errors(self):
        msg = compile_fails("""
            const K = 5
            func tick():
                var K = 9
        """)
        self.assertIn("is a constant", msg)

    def test_error_messages(self):
        cases = [
            ("fucn tick():\n    return\n", "expected include, const, var, struct or func"),
            ("func tick():\n    var x = !1\n", "unexpected '!'"),
            ("func tick():\n    var x = 1\n        x = 2\n", "unexpected indent"),
            ("struct V:\n    x\nvar v: V\nfunc tick():\n    var x = v + 1\n", "can't apply '+' to V and int"),
            ("var a: int[3]\nvar b: int[2]\nfunc tick():\n    a = b\n", "expected int[3], got int[2]"),
            ("struct V:\n    x\nvar v: V\nfunc tick():\n    if v: return\n", "condition must be int or pointer, not V"),
            ("func tick():\n    1 + 2\n", "expected an assignment or a function call"),
        ]
        for source, expected in cases:
            with self.subTest(expected=expected):
                self.assertIn(expected, compile_fails(source))

    def test_include_lib_prefix_resolves_to_repo_lib(self):
        _, asm = compile_smolt("""
            include "lib/api.smolt"
            var w = SCREEN_W
            func tick():
                w += 1
        """)
        self.assertIn("320", asm)

    def test_unreferenced_globals_dropped(self):
        _, asm = compile_smolt("""
            var unused_str = "never read"
            var unused_bss: int[50]
            var A = "reached only through TABLE"
            var TABLE: int*[] = {A}
            var used_bss: int[3]
            var mov = 1
            var ff = 2
            var keep = 3
            func unused():
                used_bss[0] = 1
            func tick():
                var p = TABLE[0]
                asm "    mov [keep], #$ff"
        """)
        labels = set(re.findall(r"(?m)^(\w+)(?::| =)", asm))
        self.assertTrue({"A", "TABLE", "keep"} <= labels)
        self.assertFalse({"unused_str", "unused_bss", "used_bss", "mov", "ff"} & labels)


if __name__ == "__main__":
    unittest.main()
