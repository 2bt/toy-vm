#!/usr/bin/env python3
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import assemble  # noqa: E402  (reads OPCODE_TABLE from src/vm.cpp)


class OpcodeTableTests(unittest.TestCase):
    def test_web_table_matches_vm_cpp(self):
        web = (ROOT / "web" / "index.html").read_text()
        web = web[web.index("const OPCODE_TABLE"):]
        web = web[:web.index("return t;")]
        table = []
        for args in re.findall(r"^\s*op\(([^)]*)\);", web, re.M):
            m = [a.strip().lower() for a in args.split(",")]
            table.append(tuple(m + ["nil"] * (3 - len(m))))
        self.assertEqual(table, assemble.OPCODE_TABLE)


if __name__ == "__main__":
    unittest.main()
