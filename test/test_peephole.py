#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import compile as C  # noqa: E402


def peep(lines, *tmps):
    return C.peephole(
        ["f:"] + [f"    {l}" for l in lines] + ["    ret"],
        {t: 1 for t in tmps},
    )


def body(asm_lines):
    out = []
    for l in asm_lines:
        if l.startswith("    ") and l != "    ret":
            out.append(l.strip())
    return out


class PeepholeTests(unittest.TestCase):
    def test_deref_mul_not_forwarded(self):
        # mov T,[s]; mov T,[T]; mul T,#256; mov D,T must keep the chain
        got = body(peep([
            "mov T, [s]",
            "mov T, [T]",
            "mul T, #256",
            "mov D, T",
        ], "T"))
        self.assertEqual(got, [
            "mov T, [s]",
            "mov T, [T]",
            "mul T, #256",
            "mov D, T",
        ])

    def test_short_fold_does_not_write_tmp(self):
        # mov T,#3; add T,T must not become add T,#3
        got = body(peep([
            "mov T, #3",
            "add T, T",
            "mov D, T",
        ], "T"))
        self.assertEqual(got[0], "mov T, #3")
        self.assertIn("add T, T", got)

    def test_short_fold_into_other_dest(self):
        got = body(peep([
            "mov T, #5",
            "add D, T",
        ], "T"))
        self.assertEqual(got, ["add D, #5"])

    def test_forward_simple_mul(self):
        got = body(peep([
            "mov T, [s]",
            "mul T, #256",
            "mov D, T",
        ], "T"))
        self.assertEqual(got, [
            "mov D, [s]",
            "mul D, #256",
        ])

    def test_inplace_negate(self):
        got = body(peep([
            "mov T, n",
            "mul T, #-1",
            "mov n, T",
        ], "T"))
        self.assertEqual(got, ["mul n, #-1"])

    def test_inplace_increment(self):
        got = body(peep([
            "mov T, D",
            "add T, #1",
            "mov D, T",
        ], "T"))
        self.assertEqual(got, ["add D, #1"])

    def test_forward_aborts_when_dst_reread(self):
        # mov T,D; mul T,#2; add T,D; mov D,T  →  3*D, not 4*D
        got = body(peep([
            "mov T, D",
            "mul T, #2",
            "add T, D",
            "mov D, T",
        ], "T"))
        self.assertEqual(got, [
            "mov T, D",
            "mul T, #2",
            "add T, D",
            "mov D, T",
        ])

    def test_invert_branch(self):
        got = C.peephole([
            "f:",
            "    jne _A",
            "    jmp _B",
            "_A:",
            "    ret",
        ], {})
        self.assertIn("    jeq _B", got)
        self.assertNotIn("    jne _A", got)
        self.assertNotIn("    jmp _B", got)
        # _A is only reached by fallthrough; unused-label cleanup may drop it


if __name__ == "__main__":
    unittest.main()
