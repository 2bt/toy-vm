#!/usr/bin/env python3
import sys, re
import ast
from pathlib import Path
import argparse
from collections import namedtuple
from types import SimpleNamespace

"""
idea: transform ast

    var a: int[4]
    a[2] = 4    -->     *(a + 2) = 4

    var q: int*
    q = &a[3]   -->     q = a + 3
    *q = 4
    q += 1

    mov q, #a + 3
    mov [q], #4
    add q, 1

    &a[3]


    &:
    + works on lvalues: VarRef, Index, Deref, Field
    + inc ptr

    *:
    + works on pointers
    + ptr > 0 --> dec ptr

"""




Token    = namedtuple("Token",    "k v loc")

Type     = namedtuple("Type",     "base ptr array")
TyStruct = namedtuple("TyStruct", "fields size")
TyField  = namedtuple("TyField",  "type offset")

Imm      = namedtuple("Imm",      "value")
VarRef   = namedtuple("VarRef",   "name")
Index    = namedtuple("Index",    "base idx")
Field    = namedtuple("Field",    "base name")
AddrOf   = namedtuple("AddrOf",   "lv")
Deref    = namedtuple("Deref",    "expr")
BinOp    = namedtuple("BinOp",    "op a b")
Call     = namedtuple("Call",     "name args")

Var      = namedtuple("Var",      "name type addr")
Func     = namedtuple("Func",     "name args type body")

Assign   = namedtuple("Assign",   "lhs op rhs")
While    = namedtuple("While",    "cond body")
Break    = namedtuple("Break",    "")
Continue = namedtuple("Continue", "")
If       = namedtuple("If",       "cond then els")
Ret      = namedtuple("Ret",      "expr")
Asm      = namedtuple("Asm",      "asm")


KEYWORDS = {
    "include", "var", "const", "func", "struct",
    "if", "then", "elif", "else", "while", "do", "end", "break", "continue",
    "return", "asm", "or", "and",
}


token_regex = re.compile(r"""
    [ \t]*(?P<nl>      \n                          )|
    [ \t]*(?P<comment> \#.*                        )|
    [ \t]*(?P<num>     \$[0-9A-Fa-f]+|[0-9]+       )|
    [ \t]*(?P<id>      [A-Za-z_][A-Za-z0-9_]*      )|
    [ \t]*(?P<sym>     ==|!=|<=|>=|\|=|&=|\+=|-=|\*=|/=|%=|[+\-*/%{}()<>\[\]=,.:&|@])|
    [ \t]*(?P<string>  "(?:[^"]|\\.)*"             )|
    [ \t]*(?P<other>   [^ \t]+                     )
""", re.VERBOSE)

def tokenize(path):
    src = path.read_text(encoding="utf-8")
    out = []
    i, line, = 0, 1
    while m := token_regex.match(src, i):
        i = m.end()
        k = m.lastgroup
        v = m.group(k)
        if k == "comment": continue
        if k == "nl": line += 1; continue
        if k == "string": line += v.count("\n")
        if k == "id" and v in KEYWORDS: k = v
        out.append(Token(k, v, (path, line)))
    out.append(Token("eof", "", (path, line)))
    return out


PRECEDENCE = {
    "or": 1,
    "and": 2,
    "<": 3,
    ">": 3,
    "<=": 3,
    ">=": 3,
    "!=": 3,
    "==": 3,
    "|": 4,
    "&": 5,
    "+": 6,
    "-": 6,
    "*": 7,
    "/": 7,
    "%": 7,
}
ARITH_OPS = {
    "|": "ore",
    "&": "and",
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    "%": "mod",
}
ASSIGN_OPS = { "=": "mov" } | { f"{k}=": v for k, v in ARITH_OPS.items() }
CMP_TO_JMP = {
    "<":  ("jlt", "jge"),
    "<=": ("jle", "jgt"),
    ">":  ("jgt", "jle"),
    ">=": ("jge", "jlt"),
    "==": ("jeq", "jne"),
    "!=": ("jne", "jeq"),
}
JMP_SWAP = {
    "jlt": "jgt",
    "jle": "jge",
    "jgt": "jlt",
    "jge": "jle",
    "jeq": "jeq",
    "jne": "jne",
}


class Parser:
    def __init__(self):
        self.consts   = {}
        self.decls    = {}
        self.funcs    = {}
        self.structs  = {}
        # for includes
        self.included = set()
        self.stack    = []

    def error(self, msg, loc=None):
        file, nr = loc or self.peek().loc
        line = file.read_text().split("\n")[nr - 1]
        sys.exit(f"{msg}\n{file.name}:{nr}:{line}")

    def peek(self, k=None, v=None):
        tok = self.head.toks[self.head.i]
        if k and tok.k != k: return None
        if v and tok.v != v: return None
        return tok


    def eat(self, k=None, v=None):
        tok = self.peek(k, v)
        if not tok:
            need = f"{k or ''} {v or ''}".strip()
            self.error(f"expected {need}")
        self.head.i += 1
        return tok

    def const_expr(self):
        n = self.expr()
        if not isinstance(n, Imm): self.error("const expression expected")
        return n.value

    def type(self):
        if not self.peek("sym", ":"): return Type("int", 0, None)
        self.eat()
        base = self.eat("id").v
        ptr = 0
        while self.peek("sym", "*"): self.eat(); ptr += 1
        array = None
        if self.peek("sym", "["):
            self.eat()
            array = self.const_expr()
            self.eat("sym", "]")
        return Type(base, ptr, array)

    def type_size(self, t):
        if t.ptr > 0: return 1
        n = t.array or 1
        e = 1 if t.base == "int" else self.structs[t.base].size
        return n * e

    def add_decl(self, prefix=""):
        name = prefix + self.eat("id").v
        if name in self.decls: self.error(f"variable '{name}' already exists")
        type = self.type()
        addr = None
        if self.peek("sym", "@"):
            self.eat()
            addr = self.const_expr()
        self.decls[name] = Var(name, type, addr)
        return name

    def program(self, path):
        self.included.add(path)
        self.head = SimpleNamespace(toks = tokenize(path), i = 0)

        while any(self.peek(k) for k in ["include", "const", "var", "func", "struct"]):
            tok = self.eat()
            if tok.k == "include":
                incl = path.parent / ast.literal_eval(self.eat("string").v)
                if incl in self.included: continue
                self.stack.append(self.head)
                self.program(incl)
                self.head = self.stack.pop()

            elif tok.k == "const":
                name = self.eat("id").v
                self.eat("sym", "=")
                self.consts[name] = self.const_expr()

            elif tok.k == "var":
                self.add_decl()

            elif tok.k == "struct":
                name = self.eat("id").v
                if name in self.structs: self.error(f"struct '{name}' already exists")
                fields = {}
                offset = 0
                while not self.peek("end"):
                    n = self.eat("id").v
                    t = self.type()
                    fields[n] = TyField(t, offset)
                    offset += self.type_size(t)
                self.eat("end")
                self.structs[name] = TyStruct(fields, offset)

            elif tok.k == "func":
                name = self.eat("id").v
                if name in self.funcs: self.error(f"function '{name}' already exists")
                self.local_var_prefix = f"_{name}_"
                self.eat("sym", "(")
                args = []
                if not self.peek("sym", ")"):
                    args.append(self.add_decl(self.local_var_prefix))
                    while self.peek("sym", ","):
                        self.eat()
                        args.append(self.add_decl(self.local_var_prefix))
                self.eat("sym", ")")
                type = self.type()
                body = self.block()
                self.eat("end")
                self.funcs[name] = Func(name, args, type, body)

            else: assert False
        self.eat("eof")

    def block(self):
        stmts = []
        while not any(self.peek(k) for k in ("end", "elif", "else", "eof")):
            stmts.append(self.stmt())
        return stmts

    def after_if(self):
        cond = self.expr()
        self.eat("then")
        then = self.block()
        if self.peek("elif"):
            self.eat()
            return If(cond, then, [self.after_if()])
        if self.peek("else"):
            self.eat()
            els = self.block()
            self.eat("end")
            return If(cond, then, els)
        self.eat("end")
        return If(cond, then, [])

    def stmt(self):
        if self.peek("while"):
            self.eat()
            cond = self.expr()
            self.eat("do")
            body = self.block()
            self.eat("end")
            return While(cond, body)
        if self.peek("break"):
            self.eat()
            return Break()
        if self.peek("continue"):
            self.eat()
            return Break()
        if self.peek("if"):
            self.eat()
            return self.after_if()
        if self.peek("return"):
            self.eat()
            expr = self.expr()
            return Ret(expr)
        if self.peek("asm"):
            self.eat()
            return Asm(self.eat("string").v)

        lhs = self.primary()
        if isinstance(lhs, Call): return lhs

        op = self.peek("sym")
        if isinstance(lhs, (VarRef, Deref, Index, Field)) and op.v in ASSIGN_OPS:
            op = self.eat().v
            rhs = self.expr()
            return Assign(lhs, op, rhs)

        self.error("invalid statement")


    def expr(self, minp=1):
        left = self.unary()
        while tok := self.peek():
            p = PRECEDENCE.get(tok.v, 0)
            if p < minp: break
            self.eat()
            right = self.expr(p + 1)
            op = tok.v
            if isinstance(left, Imm) and isinstance(right, Imm):
                left = Imm(int(eval(f"{left.value} {op} {right.value}")))
            else:
                left = BinOp(op, left, right)
        return left

    def unary(self):
        if self.peek("sym", "&"):
            self.eat()
            return AddrOf(self.primary())
        if self.peek("sym", "("):
            self.eat()
            e = self.expr()
            self.eat("sym", ")")
            return e
        if self.peek("sym", "-"):
            self.eat()
            u = self.unary()
            if isinstance(u, Imm): return Imm(-u.value)
            return BinOp("*", u, Imm(-1))
        return self.primary()

    def primary(self):
        # number
        if self.peek("num"):
            v = self.eat().v
            if v.startswith("$"): n = int(v[1:], 16)
            else: n = int(v)
            return Imm(n)

        # constant
        name = self.eat("id").v
        if name in self.consts:
            return Imm(self.consts[name])

        # function call
        if self.peek("sym", "("):
            self.eat()
            args = []
            if not self.peek("sym", ")"):
                args.append(self.expr())
                while self.peek("sym", ","): self.eat(); args.append(self.expr())
            self.eat("sym", ")")
            f = self.funcs.get(name)
            if not f: self.error(f"unknown function '{name}'")
            if len(args) != len(f.args): self.error("wrong number of function arguments")
            return Call(name, args)

        # resolve name
        if name not in self.decls:
            old = name
            name = self.local_var_prefix + name
            if name not in self.decls: self.error(f"unknown variable '{old}'")

        node = VarRef(name)
        while True:
            if self.peek("sym", "["):
                self.eat()
                node = Index(node, self.expr())
                self.eat("sym", "]")
            elif self.peek("sym", "@"):
                self.eat()
                node = Deref(node)
            elif self.peek("sym", "."):
                self.eat()
                node = Field(node, self.eat("id").v)
            else: return node





class Codegen:

    def emit(self, s):
        self.lines.append(s)

    def newtmp(self, *ts):
        for t in ts:
            if t in self.tmp_vars: return t
            if t.startswith("[") and t[1:-1] in self.tmp_vars: return t[1:-1]
        t = f"_T{self.tmp_i}"
        self.tmp_i += 1
        if t not in self.tmp_vars: self.tmp_vars[t] = ()
        return t

    def label(self, base):
        self.label_i += 1
        return f"_{base}_{self.label_i}"


    def emit_label(self, l):
        # remove useless jump
        i = len(self.lines) - 1
        while i > 0 and (m := re.match(f"    j.. ([^ ]+)", self.lines[i])):
            if m[1] == l: self.lines.pop(i)
            i -= 1
        self.emit(f"{l}:")


    def get_type(self, expr):
        i = id(expr)
        if t := self.expr_types.get(i): return t
        if isinstance(expr, Imm): t = ("int", 0, None)
        elif isinstance(expr, VarRef): t = self.decls[expr.name].type
        elif isinstance(expr, Index):
            s = self.get_type(expr.base)
            assert s.array
            assert s.ptr == 0
            return s.base


        self.expr_types[i] = t
        return t


    def branch(self, node, T, F):
        if isinstance(node, Imm):
            if node.value: self.emit(f"    jmp {T}")
            else: self.emit(f"    jmp {F}")

        elif isinstance(node, VarRef):
            self.emit(f"    cmp {node.name}, #0")
            self.emit(f"    jeq {F}")
            self.emit(f"    jne {T}")

        elif isinstance(node, BinOp) and node.op == "&":
            a = self.value(node.a)
            b = self.value(node.b)
            if a.startswith("#"): a, b = b, a
            self.emit(f"    tst {a}, {b}")
            self.emit(f"    jeq {F}")
            self.emit(f"    jne {T}")

        elif isinstance(node, BinOp) and node.op in CMP_TO_JMP:
            a = self.value(node.a)
            b = self.value(node.b)
            jt, jf = CMP_TO_JMP[node.op]
            if a.startswith("#"):
                a, b = b, a
                jt = JMP_SWAP[jt]
                jf = JMP_SWAP[jf]
            self.emit(f"    cmp {a}, {b}")
            self.emit(f"    {jf} {F}")
            self.emit(f"    {jt} {T}")

        elif isinstance(node, BinOp) and node.op == "and":
            N = self.label("and_next")
            self.branch(node.a, N, F)
            self.emit_label(N)
            self.branch(node.b, T, F)

        elif isinstance(node, BinOp) and node.op == "or":
            N = self.label("or_next")
            self.branch(node.a, T, N)
            self.emit_label(N)
            self.branch(node.b, T, F)

        else:
            _ = self.value(node)
            self.emit(f"    jne {T}")
            self.emit(f"    jeq {F}")


    def value(self, node):
        if isinstance(node, Imm): return f"#{node.value}"
        if isinstance(node, VarRef): return node.name
        if isinstance(node, Index):
            # t = self.get_type(node.base)
            base = node.base
            if isinstance(base, VarRef):
                type = self.decls[base.name].type
                assert type.array, "must be array"
                elem_size = self.ast.type_size(Type(type.base, type.ptr, None))
                if isinstance(node.idx, Imm):
                    return f"{base.name}+{node.idx.value * elem_size}"
                v = self.value(node.idx)
                t = self.newtmp(v)
                if t != v: self.emit(f"    mov {t}, {v}")
                if elem_size > 0: self.emit(f"    mul {t}, #{elem_size}")
                self.emit(f"    add {t}, #{lv.name}")
                return f"[{t}]"

            else: assert False, f"TODO: support Deref, Field (type is {type(node).__name__})"

        # if isinstance(node, Field):
        #     t = self.get_type(node.base)
        #     assert t.base in self.structs
        #     assert t.ptr == 0
        #     assert t.array == None
        #     v = self.value(node.base)
        #     field = self.structs[t.base].fields[node.name]
        #     print("######", field)
        #     exit(1)

        if isinstance(node, AddrOf):
            if isinstance(node.lv, Index):
                v = self.value(node.lv)
                if v.startswith("["): return v[1:-1]
                return f"#{v}"

            else: assert False

        if isinstance(node, Deref):
            v = self.value(node.expr)
            if not v.startswith("["): return f"[{v}]"
            t = self.newtmp(v)
            if t != v: self.emit(f"    mov {t}, {v}")
            return f"[{t}]"


        if isinstance(node, BinOp):
            if op := ARITH_OPS.get(node.op):
                a = self.value(node.a)
                b = self.value(node.b)
                t = self.newtmp(a, b)
                if t == a:
                    self.emit(f"    {op} {t}, {b}")
                elif t == b and node.op not in {"-", "/", "%"}:
                    self.emit(f"    {op} {t}, {a}")
                else:
                    self.emit(f"    mov {t}, {a}")
                    self.emit(f"    {op} {t}, {b}")
                return t
            if node.op in CMP_TO_JMP or node.op in {"and", "or"}:
                t = self.newtmp()
                T = self.label("true")
                F = self.label("false")
                D = self.label("done")
                self.branch(node, T, F)
                self.emit_label(F)
                self.emit(f"    mov {t}, #0")
                self.emit(f"    jmp {D}")
                self.emit_label(T)
                self.emit(f"    mov {t}, #1")
                self.emit_label(D)
                return t
            assert False, "unsupported binop {node.op}"

        if isinstance(node, Call): return self.call(node)

        assert False, f"unsupported expression {node}"


    def call(self, node):
        name, args = node.name, node.args
        f = self.funcs[name]
        for a, b in zip(args, f.args):
            t = self.value(a)
            self.emit(f"    mov {b}, {t}")
        self.emit(f"    jsr {f.name}")
        return "_R"


    def stmt(self, node):
        self.tmp_i = 0 # reuse temporary variables

        if isinstance(node, Assign):
            assert isinstance(node.lhs, (VarRef, Index, Deref, Field))
            dst = self.value(node.lhs)
            v = self.value(node.rhs)
            op = ASSIGN_OPS[node.op]
            if dst != v or node.op != "=": self.emit(f"    {op} {dst}, {v}")

        elif isinstance(node, While):
            L = self.label("while")
            T = self.label("while_true")
            E = self.label("end_while")
            self.emit_label(L)
            self.branch(node.cond, T, E)
            self.emit_label(T)
            self.loop_stack.append((L, E))
            for st in node.body: self.stmt(st)
            self.loop_stack.pop()
            self.emit(f"    jmp {L}")
            self.emit_label(E)

        elif isinstance(node, Continue):
            l = self.loop_stack[-1][0]
            self.emit(f"    jmp {l}")

        elif isinstance(node, Break):
            l = self.loop_stack[-1][1]
            self.emit(f"    jmp {l}")

        elif isinstance(node, If):
            T = self.label("if_true")
            F = self.label("if_false")
            E = self.label("end_if")
            self.branch(node.cond, T, F)
            self.emit_label(T)
            for st in node.then: self.stmt(st)
            if node.els:
                self.emit(f"    jmp {E}")
                self.emit_label(F)
                for st in node.els: self.stmt(st)
                self.emit_label(E)
            else:
                self.emit_label(F)

        elif isinstance(node, Ret):
            v = self.value(node.expr)
            self.emit(f"    mov _R, {v}")
            self.emit(f"    ret")

        elif isinstance(node, Call):
            self.call(node)

        elif isinstance(node, Asm):
            self.lines += ast.literal_eval(node.asm.replace("\n","\\n")).rstrip().split("\n")

        else:
            sys.exit("unsupported statement")


    def compile(self, ast):
        self.expr_types = {}
        self.lines      = []
        self.tmp_i      = 0
        self.label_i    = 0
        self.decls      = ast.decls
        self.funcs      = ast.funcs
        self.structs    = ast.structs

        self.tmp_vars   = {} # use dict to keep original order
        self.loop_stack = []
        self.ast        = ast

        # functions
        for f in self.funcs.values():
            self.emit("")
            self.emit(f"    ; function {f.name}")
            self.emit(f"{f.name}:")
            for st in f.body:
                self.stmt(st)
            self.emit("    ret")

        lines, self.lines = self.lines, []

        addr = 30 # TODO
        for name in ["_R"] + list(self.tmp_vars):
            self.emit(f"{name} = {addr}")
            addr += 1

        for v in self.decls.values():
            if v.addr != None:
                self.emit(f"{v.name} = {v.addr}")
            else:
                self.emit(f"{v.name} = {addr}")
                addr += ast.type_size(v.type)

        return "\n".join(self.lines + lines) + "\n"



def main(args):
    parser = Parser()
    parser.program(Path(args.src))

    asm = Codegen().compile(parser)
    out = args.out or Path(args.src).with_suffix(".asm")
    with open(out, "w") as f:
        f.write(asm)

    print("lines:", asm.count("\n"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("out", nargs="?")
    args = parser.parse_args()
    main(args)