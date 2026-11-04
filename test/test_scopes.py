#!/usr/bin/env python3
import re
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import compile as C  # noqa: E402


def write_toy(source: str) -> Path:
    d = Path(tempfile.mkdtemp(dir=ROOT / "test"))
    p = d / "main.toy"
    p.write_text(textwrap.dedent(source))
    return p


def parse_toy(source: str) -> C.Parser:
    path = write_toy(source)
    parser = C.Parser()
    parser.program(path)
    return parser


def compile_toy(source: str):
    parser = parse_toy(source)
    asm = C.Codegen().compile(parser)
    return parser, asm


def compile_fails(source: str) -> str:
    """Return stderr/message text; fail the test if compile succeeds."""
    path = write_toy(source)
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
        parser, asm = compile_toy("""
            func tick():
                var x = 1
                if 1:
                    var x = 2
                x = 3
        """)
        self.assertIn("tick.x", parser.var_type)
        self.assertIn("tick.x.1", parser.var_type)
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
        parser, _ = compile_toy("""
            func tick():
                for i = 0, 2:
                    var z = i
                for i = 0, 2:
                    var z = i
        """)
        # two loop indices minted
        self.assertIn("tick.i", parser.var_type)
        self.assertIn("tick.i.1", parser.var_type)

    def test_sibling_blocks_reuse_offset_and_frame_size(self):
        parser, asm = compile_toy("""
            func tick():
                if 1:
                    var a: int[4]
                else:
                    var b: int[4]
        """)
        self.assertEqual(parser.var_offset["tick.a"], parser.var_offset["tick.b"])
        self.assertEqual(parser.funcs["tick"].frame_size, 4)
        # same absolute address in asm
        ma = re.search(r"^tick\.a = (\d+)", asm, re.M)
        mb = re.search(r"^tick\.b = (\d+)", asm, re.M)
        self.assertIsNotNone(ma)
        self.assertIsNotNone(mb)
        self.assertEqual(ma.group(1), mb.group(1))

    def test_nested_shadow_does_not_reuse_with_outer(self):
        parser, _ = compile_toy("""
            func tick():
                var x = 1
                if 1:
                    var x = 2
        """)
        self.assertNotEqual(
            parser.var_offset["tick.x"],
            parser.var_offset["tick.x.1"],
        )
        self.assertEqual(parser.funcs["tick"].frame_size, 2)


if __name__ == "__main__":
    unittest.main()
