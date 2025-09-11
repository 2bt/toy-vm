#!/usr/bin/env python3
import sys, re
import ast
from pathlib import Path
import argparse
from collections import namedtuple


Token   = namedtuple("Token",  "k v file line")

Imm     = namedtuple("Imm",    "value")
VarRef  = namedtuple("VarRef", "name")
Index   = namedtuple("Index",  "name idx")
Neg     = namedtuple("Neg",    "expr")
BinOp   = namedtuple("BinOp",  "op a b")
Call    = namedtuple("Call",   "name args")

Var     = namedtuple("Var",    "name addr")
VarArr  = namedtuple("VarArr", "name size addr")
Func    = namedtuple("Func",   "name args body")

Assign  = namedtuple("Assign", "lhs op rhs")
While   = namedtuple("While",  "cond body")
If      = namedtuple("If",     "cond then else_")
Ret     = namedtuple("Ret",    "expr")
Asm     = namedtuple("Asm",    "asm")

KEYWORDS = set("const var func end while do if then else elif and or return asm".split())

token_regex = re.compile(r"""
    [ \t]*(?P<nl>      \n                          )|
    [ \t]*(?P<comment> \#.*                        )|
    [ \t]*(?P<num>     \$[0-9A-Fa-f]+|[0-9]+       )|
    [ \t]*(?P<id>      [A-Za-z_][A-Za-z0-9_]*      )|
    [ \t]*(?P<sym>     ==|!=|<=|>=|\|=|&=|\+=|-=|\*=|/=|%=|[+\-*/%()<>\[\]=,:&|@])|
    [ \t]*(?P<string>  "(?:[^"]|\\.)[^"]*"         )|
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
        out.append(Token(k, v, path, line))
    out.append(Token("eof", "", path, line))
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
        self.included = set()
        self.inc_stack = []
        self.consts    = {}
        self.decls     = {}
        self.funcs     = {}

    def error(self, msg):
        t = self.peek()
        sys.exit(f"{t.file}:{t.line}: parse error: {msg}")

    def peek(self, k=None, v=None):
        tok = self.toks[self.i]
        if k and tok.k != k: return None
        if v and tok.v != v: return None
        return tok


    def eat(self, k=None, v=None):
        tok = self.peek(k, v)
        if not tok:
            here = self.toks[self.i]
            need = f"{k or ''} {v or ''}".strip()
            self.error(f"expected {need}")
        self.i += 1
        return tok

    def const_expr(self):
        n = self.expr()
        if not isinstance(n, Imm):
            self.error("const expression expected")
        return n.value

    def var_address(self):
        if self.peek("sym", "@"):
            self.eat()
            return self.const_expr()
        return None


    def add_decl(self, v):
        if v.name in self.decls:
            self.error(f"variable '{v.name}' already exists")
        self.decls[v.name] = v


    def program(self, path):

        self.toks = tokenize(path)
        self.i = 0


        while self.peek("const") or self.peek("var") or self.peek("func"):
            tok = self.eat()
            name = self.eat("id").v
            if tok.k == "const":
                self.eat("sym", "=")
                self.consts[name] = self.const_expr()

            elif tok.k == "var":
                if self.peek("sym", "["):
                    self.eat()
                    n = self.const_expr()
                    self.eat("sym", "]")
                    self.add_decl(VarArr(name, n, self.var_address()))
                else:
                    self.add_decl(Var(name, self.var_address()))

            else:
                # function definition
                self.local_var_prefix = f"_{name}_"
                self.eat("sym", "(")
                args = []
                def add_arg():
                    arg = self.local_var_prefix + self.eat("id").v
                    self.add_decl(Var(arg, self.var_address()))
                    args.append(arg)

                if not self.peek("sym", ")"):
                    add_arg()
                    while self.peek("sym", ","):
                        self.eat()
                        add_arg()
                self.eat("sym", ")")
                body = self.block()
                self.eat("end")

                if name in self.funcs:
                    sys.exit(f"{tok.line}: function already exists")
                self.funcs[name] = Func(name, args, body)

        self.eat("eof")
        return self.decls, self.funcs

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

        # assignment or call
        lhs = self.primary()
        if self.peek("sym") and self.peek().v in ASSIGN_OPS:
            op = self.eat().v
            rhs = self.expr()
            return Assign(lhs, op, rhs)

        if isinstance(lhs, Call):
            return lhs

        self.error("expected assignment or call")


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
        if self.peek("sym", "("):
            self.eat()
            e = self.expr()
            self.eat("sym", ")")
            return e
        if self.peek("sym", "-"):
            self.eat()
            u = self.unary()
            if isinstance(u, Imm):
                return Imm(-u.value)
            return Neg(u)
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
                while self.peek("sym", ","):
                    self.eat()
                    args.append(self.expr())
            self.eat("sym", ")")

            f = self.funcs.get(name)
            if not f:
                self.error("unknown function")
            if len(args) != len(f.args):
                self.error("wrong number of arguments")


            return Call(name, args)

        if name not in self.decls:
            old = name
            name = self.local_var_prefix + name
            if name not in self.decls:
                self.error(f"unknown variable '{old}'")

        # indexing
        if self.peek("sym", "["):
            self.eat()
            idx = self.expr()
            self.eat("sym", "]")
            return Index(name, idx)

        return VarRef(name)





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

    def addr_of_lvalue(self, node):
        if isinstance(node, VarRef): return node.name
        if isinstance(node, Index):
            # base = self.vars[node.name]
            v = self.value(node.idx)
            t = self.newtmp(v)
            if t != v: self.emit(f"    mov {t}, {v}")
            self.emit(f"    add {t}, #{node.name}")
            return f"[{t}]"
        sys.exit("unsupported lvalue")


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
        """Return a location holding the value of node."""
        if isinstance(node, Imm): return f"#{node.value}"
        if isinstance(node, VarRef): return node.name
        if isinstance(node, Index):
            # base = self.decls[node.name]
            v = self.value(node.idx)
            t = self.newtmp(v)
            if t != v: self.emit(f"    mov {t}, {v}")
            self.emit(f"    add {t}, #{node.name}")
            return f"[{t}]"
        if isinstance(node, Neg):
            v = self.value(node.expr)
            t = self.newtmp(v)
            if t != v: self.emit(f"    mov {t}, {v}")
            self.emit(f"    mul {t}, #-1")
            return t

        if isinstance(node, BinOp):
            if op := ARITH_OPS.get(node.op):
                a = self.value(node.a)
                b = self.value(node.b)
                t = self.newtmp(a, b)
                if t == a:
                    self.emit(f"    {op} {t}, {b}")
                elif t == b and node.op not in {"/", "%"}:
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

            sys.exit(f"unsupported binop {node.op}")

        if isinstance(node, Call):
            return self.call(node)

        sys.exit("unsupported expression node")


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
            dst = self.addr_of_lvalue(node.lhs)
            v = self.value(node.rhs)
            op = ASSIGN_OPS[node.op]
            if dst != v: self.emit(f"    {op} {dst}, {v}")

        elif isinstance(node, While):
            L = self.label("while")
            T = self.label("while_true")
            E = self.label("end_while")
            self.emit_label(L)
            self.branch(node.cond, T, E)
            self.emit_label(T)
            for st in node.body: self.stmt(st)
            self.emit(f"    jmp {L}")
            self.emit_label(E)

        elif isinstance(node, If):
            T = self.label("if_true")
            F = self.label("if_false")
            E = self.label("end_if")
            self.branch(node.cond, T, F)
            self.emit_label(T)
            for st in node.then: self.stmt(st)
            if node.else_:
                self.emit(f"    jmp {E}")
                self.emit_label(F)
                for st in node.else_:
                    self.stmt(st)
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


    def compile(self, decls, funcs):
        self.lines     = []
        self.tmp_i     = 0
        self.label_i   = 0
        self.decls     = decls
        self.funcs     = funcs
        self.tmp_vars  = {} # use dict to keep original order

        # functions
        for f in funcs.values():
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
                if isinstance(v, VarArr): addr += v.size
                else: addr += 1

        return "\n".join(self.lines + lines) + "\n"



def main(args):
    decls, funcs = Parser().program(Path(args.src))

    asm = Codegen().compile(decls, funcs)
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