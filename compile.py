#!/usr/bin/env python3
import sys, re
import ast
from pathlib import Path
import argparse
from collections import namedtuple
from types import SimpleNamespace


Token    = namedtuple("Token",    "k v loc")

Type     = namedtuple("Type",     "base ptr array")
TyStruct = namedtuple("TyStruct", "fields size")
TyField  = namedtuple("TyField",  "type offset")

Imm      = namedtuple("Imm",      "value")
VarRef   = namedtuple("VarRef",   "name")
AddrOf   = namedtuple("AddrOf",   "lv")
Deref    = namedtuple("Deref",    "expr")
BinOp    = namedtuple("BinOp",    "op a b")
Call     = namedtuple("Call",     "name args")

Var      = namedtuple("Var",      "type addr data")
Func     = namedtuple("Func",     "args type body")

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

TYPE_INT = Type("int", 0, None)

def is_int(t): return t == TYPE_INT
def is_struct(t): return t.array == None and t.ptr == 0 and t.base != "int"
def is_ptr(t): return t.array == None and t.ptr > 0
def is_array(t): return t.array != None
def is_cond(t): return is_int(t) or is_ptr(t)

def addr_of(e):
    if isinstance(e, Deref): return e.expr
    return AddrOf(e)



token_regex = re.compile(r"""
    [ \t]*(?P<nl>      \n                          )|
    [ \t]*(?P<comment> \#.*                        )|
    [ \t]*(?P<num>     \$[0-9A-Fa-f]+|[0-9]+       )|
    [ \t]*(?P<id>      [A-Za-z][A-Za-z0-9_]*       )|
    [ \t]*(?P<sym>     ==|!=|<=|>=|\|=|&=|\+=|-=|\*=|/=|%=|->|
                       [+\-*/%{}()<>\[\]=,.:&|@]   )|
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
        self.consts     = {}
        self.decls      = {}
        self.funcs      = {}
        self.structs    = {}
        self.loop       = 0
        self.var_prefix = None
        # for includes
        self.included   = set()
        self.stack      = []

    def error(self, msg, tok=None):
        file, nr = (tok or self.last_tok).loc
        line = file.read_text().split("\n")[nr - 1]
        sys.exit(f"{file.name}:{nr}: {msg}\n{line}")

    def peek(self, k=None, v=None):
        tok = self.head.toks[self.head.i]
        if k and tok.k != k: return None
        if v and tok.v != v: return None
        return tok


    def eat(self, k=None, v=None):
        self.last_tok = self.head.toks[self.head.i]
        tok = self.peek(k, v)
        if not tok:
            need = f"{k or ''} {v or ''}".strip()
            self.error(f"expected {need}")
        self.head.i += 1
        return tok

    def const_expr(self):
        n, _ = self.expr()
        if not isinstance(n, Imm): self.error("const expression expected")
        return n.value

    def type(self):
        if not self.peek("sym", ":"): return TYPE_INT
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
        if t.base == "int": return n
        if t.base not in self.structs: self.error(f"unknown struct '{t.base}'")
        return n * self.structs[t.base].size

    def add_decl(self):
        name = self.eat("id").v
        if self.var_prefix:
            name = self.var_prefix + name
        if name in self.decls: self.error(f"variable '{name}' already exists")
        type = self.type()
        addr = None
        if self.peek("sym", "@"):
            self.eat()
            addr = self.const_expr()

        data = None
        if not self.var_prefix and self.peek("sym", "="):
            # parse data
            self.eat()
            def null_obj(t):
                if is_array(t):
                    return null_obj(Type(t.base, t.ptr, None)) * t.array
                if is_struct(t):
                    data = []
                    for f in self.structs[t.base].fields.values():
                        data += null_obj(f.type)
                    return data
                return [0]
            def parse(t):
                if is_array(t):
                    self.eat("sym", "{")
                    data = []
                    l = t.array
                    t = Type(t.base, t.ptr, None)
                    n = 0
                    while not self.peek("sym", "}"):
                        if is_int(t) and self.peek("string"):
                            string = ast.literal_eval(self.eat("string").v)
                            data += list(map(ord, string))
                            n += len(string)
                        else:
                            data += parse(t)
                            n += 1
                        if n > l: self.error("too too many initializers")
                        if not self.peek("sym", ","): break
                        self.eat()
                    self.eat("sym", "}")
                    data += null_obj(t) * (l - n)
                    return data
                if is_struct(t):
                    self.eat("sym", "{")
                    data = []
                    parsing = True
                    for f in self.structs[t.base].fields.values():
                        if self.peek("sym", "}"): parsing = False
                        if parsing:
                            data += parse(f.type)
                            if not self.peek("sym", ","): parsing = False
                            else: self.eat()
                        else:
                            data += null_obj(f.type)
                    self.eat("sym", "}")
                    return data
                return [self.const_expr()]
            data = parse(type)

        self.decls[name] = Var(type, addr, data)
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
                self.var_prefix = f"_{name}_"
                self.eat("sym", "(")
                args = []
                if not self.peek("sym", ")"):
                    args.append(self.add_decl())
                    while self.peek("sym", ","):
                        self.eat()
                        args.append(self.add_decl())
                self.eat("sym", ")")
                self.return_type = self.type()
                body = self.block()
                self.eat("end")
                self.funcs[name] = Func(args, self.return_type, body)
                self.var_prefix = None

            else: assert False
        self.eat("eof")

    def block(self):
        stmts = []
        while not any(self.peek(k) for k in ("end", "elif", "else", "eof")):
            stmts.append(self.stmt())
        return stmts

    def after_if(self):
        cond, t = self.expr()
        if not is_cond(t): self.error("bad type")
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
            cond, t = self.expr()
            if not is_cond(t): self.error("bad type")
            self.eat("do")
            self.loop += 1
            body = self.block()
            self.loop -= 1
            self.eat("end")
            return While(cond, body)
        if self.peek("break"):
            self.eat()
            if self.loop == 0: self.error("break outside loop")
            return Break()
        if self.peek("continue"):
            self.eat()
            if self.loop == 0: self.error("continue outside loop")
            return Continue()
        if self.peek("if"):
            self.eat()
            return self.after_if()
        if self.peek("return"):
            self.eat()
            expr, type = self.expr()
            if type != self.return_type: self.error("return type mismatch")
            return Ret(expr)
        if self.peek("asm"):
            self.eat()
            return Asm(self.eat("string").v)

        a, ta = self.primary()
        if isinstance(a, Call): return a

        if isinstance(a, (VarRef, Deref)) and self.peek("sym") and self.peek().v in ASSIGN_OPS:
            op = self.eat().v
            b, tb = self.expr()
            if op == "=" and (ta == tb or is_ptr(ta) and is_int(tb) and b == Imm(0)): pass
            elif op in ("+=", "-=") and is_ptr(ta) and is_int(tb):
                # pointer arithmetic
                elem_size = self.type_size(Type(ta.base, ta.ptr - 1, None))
                if elem_size > 1:
                    if isinstance(b, Imm): b = Imm(b.value * elem_size)
                    else: b = BinOp("*", b, Imm(elem_size))
            else:
                if not (is_int(ta) and is_int(tb)): self.error("type mismatch")
            return Assign(a, op, b)

        self.error("invalid statement")


    def expr(self, minp=1):
        a, ta = self.unary()
        while tok := self.peek():
            p = PRECEDENCE.get(tok.v, 0)
            if p < minp: break
            self.eat()
            b, tb = self.expr(p + 1)
            op = tok.v
            if isinstance(a, Imm) and isinstance(b, Imm):
                # fold constant
                a = Imm(int(eval(f"{a.value} {op} {b.value}")))
                ta = TYPE_INT
                continue
            if op in CMP_TO_JMP and ta == tb and (is_int(ta) or is_ptr(ta)): ta = TYPE_INT
            elif op in ("and", "or") and is_cond(ta) and is_cond(tb):        ta = TYPE_INT
            elif op in ("+", "-") and is_ptr(ta) and is_int(tb):
                # pointer arithmetic
                elem_size = self.type_size(Type(ta.base, ta.ptr - 1, None))
                if elem_size > 1:
                    if isinstance(b, Imm): b = Imm(b.value * elem_size)
                    else: b = BinOp("*", b, Imm(elem_size))
            else:
                if not (is_int(ta) and is_int(tb)): self.error("type mismatch", tok)
            a = BinOp(op, a, b)
        return a, ta

    def unary(self):
        if self.peek("sym", "&"):
            self.eat()
            a, ta = self.primary()
            if not isinstance(a, (VarRef, Deref)): self.error("& operand must be lvalue")
            if is_array(ta): self.error("address of array not allowed")
            return addr_of(a), Type(ta.base, ta.ptr + 1, None)
        if self.peek("sym", "("):
            self.eat()
            e, t = self.expr()
            self.eat("sym", ")")
            return e, t
        if self.peek("sym", "-"):
            self.eat()
            a, t = self.unary()
            if not is_int(t): self.error("bad type")
            if isinstance(a, Imm): return Imm(-a.value), TYPE_INT
            return BinOp("*", a, Imm(-1)), TYPE_INT
        return self.primary()

    def primary(self):
        # number
        if self.peek("num"):
            v = self.eat().v
            if v.startswith("$"): n = int(v[1:], 16)
            else: n = int(v)
            return Imm(n), TYPE_INT

        # constant
        name = self.eat("id").v
        if name in self.consts:
            return Imm(self.consts[name]), TYPE_INT

        # function call
        if self.peek("sym", "("):
            f = self.funcs.get(name)
            if not f: self.error(f"unknown function '{name}'")
            self.eat()
            args = []
            types = []
            if not self.peek("sym", ")"):
                e, t = self.expr()
                args.append(e)
                types.append(t)
                while self.peek("sym", ","):
                    self.eat()
                    e, t = self.expr()
                    args.append(e)
                    types.append(t)
            self.eat("sym", ")")
            if len(args) != len(f.args): self.error("wrong number of function arguments")
            for t, n in zip(types, f.args):
                if t != self.decls[n].type: self.error("type mismatch")
            return Call(name, args), f.type

        # resolve name
        local_name = self.var_prefix + name
        if local_name in self.decls: name = local_name
        elif name not in self.decls: self.error(f"unknown variable '{name}'")

        node = VarRef(name)
        type = self.decls[name].type
        while True:
            if self.peek("sym", "["):
                self.eat()
                idx, tidx = self.expr()
                self.eat("sym", "]")
                if not is_array(type): self.error("cannot index non-array")
                if not is_int(tidx): self.error("index must be int")
                type = Type(type.base, type.ptr, None)
                elem_size = self.type_size(type)
                if idx != Imm(0):
                    if elem_size > 1:
                        if isinstance(idx, Imm): idx = Imm(idx.value * elem_size)
                        else: idx = BinOp("*", idx, Imm(elem_size))
                    node = Deref(BinOp("+", addr_of(node), idx))

            elif self.peek("sym", "@"):
                self.eat()
                if not is_ptr(type): self.error("non-pointer deref")
                node = Deref(node)
                type = Type(type.base, type.ptr - 1, None)

            elif self.peek("sym", "->"):
                self.eat()
                if not is_ptr(type): self.error("non-pointer deref")
                type = Type(type.base, type.ptr - 1, None)
                if not is_struct(type): self.error("request for member of non-struct")
                field = self.eat("id").v
                struct = self.structs[type.base]
                f = struct.fields.get(field)
                if not f: self.error(f"struct '{type.base}' has no field '{field}'")
                type = f.type
                if f.offset > 0:node = BinOp("+", node, Imm(f.offset))
                node = Deref(node)

            elif self.peek("sym", "."):
                self.eat()
                if not is_struct(type): self.error("request for member of non-struct")
                field = self.eat("id").v
                struct = self.structs[type.base]
                f = struct.fields.get(field)
                if not f: self.error(f"struct '{type.base}' has no field '{field}'")
                type = f.type
                if f.offset > 0: node = Deref(BinOp("+", addr_of(node), Imm(f.offset)))
            else: return node, type

def split_add(e):
    match e:
        case Imm(v): return (None, None, v)
        case AddrOf(VarRef(n)): return (None, n, 0)
        case AddrOf(Deref(e)): assert False, "WOOT"
        case BinOp("+", a, b):
            d1, s1, k1 = split_add(a)
            d2, s2, k2 = split_add(b)
            if d1 and d2: d = BinOp("+", d1, d2)
            else: d = d1 or d2
            if s1 and s2: assert False, "WOOT"
            return (d, s1 or s2, k1 + k2)
    return (e, None, 0)


class Codegen:

    def newtmp(self, *ts):
        for t in ts:
            if t in self.tmp_vars: return t
            if t.startswith("[") and t[1:-1] in self.tmp_vars: return t[1:-1]
        t = f"_{self.current_func}_T{self.tmp_i}"
        self.tmp_i += 1
        self.tmp_vars[t] = ()
        return t

    def label(self, base):
        self.label_i += 1
        return f"_{base}_{self.label_i}"

    def emit(self, s):
        self.lines.append(s)

    def emit_label(self, l):
        # remove useless jump
        i = len(self.lines) - 1
        while i > 0 and (m := re.match(f"    j[^m]. ([^ ]+)", self.lines[i])):
            if m[1] == l: self.lines.pop(i)
            i -= 1
        self.emit(f"{l}:")


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
        if isinstance(node, AddrOf):
            v = self.value(node.lv)
            if v.startswith("["): return v[1:-1]
            return f"#{v}"

        if isinstance(node, Deref):
            dyn, sym, off = split_add(node.expr)
            if not dyn: return f"{sym}+{off}"
            a = self.value(dyn)
            a = f"[{a}]"
            if sym: a += f"+{sym}"
            if off: a += f"+{off}"
            return a

        if isinstance(node, BinOp):
            if node.op == "+":
                dyn, sym, off = split_add(node)
                if not dyn: return f"#{sym}+{off}"

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
            assert False, f"unsupported binop {node.op}"

        if isinstance(node, Call): return self.call(node)

        assert False, f"unsupported expression {node}"


    def call(self, node):
        f = self.funcs[node.name]
        for a, b in zip(node.args, f.args):
            t = self.value(a)
            self.emit(f"    mov {b}, {t}")
        self.emit(f"    jsr {node.name}")
        return "_R"


    def stmt(self, node):
        self.tmp_i = 0 # reuse temporary variables

        if isinstance(node, Assign):
            assert isinstance(node.lhs, (VarRef, Deref))
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
            assert False, f"unsupported statement {node}"


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
        for name, f in self.funcs.items():
            self.current_func = name
            self.emit("")
            self.emit(f"    ; function {name}")
            self.emit(f"{name}:")
            for st in f.body:
                self.stmt(st)
            self.emit("    ret")


        # variables and data
        lines, self.lines = self.lines, []

        # API variables
        addr = 0
        for name, v in self.decls.items():
            if v.addr != None:
                assert v.data == None
                self.emit(f"{name} = {v.addr}")
                addr = max(addr, v.addr + 1)

        # temp variables
        for name in ["_R"] + list(self.tmp_vars):
            self.emit(f"{name} = {addr}")
            addr += 1

        # BSS
        for name, v in self.decls.items():
            if v.addr == None and not v.data:
                self.emit(f"{name} = {addr}")
                addr += ast.type_size(v.type)

        # data
        self.emit(f"")
        self.emit(f"    .data {addr}")
        for name, v in self.decls.items():
            if v.data:
                assert v.addr == None
                self.emit(f"{name}:")
                ln = "   "
                for d in v.data:
                    lo = ln
                    ln += f" {d},"
                    if len(ln) > 80:
                        self.emit(lo[:-1])
                        ln = f"    {d},"
                self.emit(ln[:-1])

        # code
        self.emit(f"")
        self.emit(f"    .code")
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