#!/usr/bin/env python3
import sys, re
import ast
from pathlib import Path
import argparse
from collections import namedtuple
from dataclasses import dataclass, field

@dataclass
class Head:
    toks: list
    i = 0

Token    = namedtuple("Token",    "k v loc")
Type     = namedtuple("Type",     "base ptr array")
TyStruct = namedtuple("TyStruct", "fields size")
TyField  = namedtuple("TyField",  "type offset")

Imm      = namedtuple("Imm",      "value")
VarRef   = namedtuple("VarRef",   "name")
AddrOf   = namedtuple("AddrOf",   "lv")
Deref    = namedtuple("Deref",    "expr")
BinOp    = namedtuple("BinOp",    "op a b")
Assign   = namedtuple("Assign",   "op a b")
Not      = namedtuple("Not",      "expr")
Call     = namedtuple("Call",     "name args")
Var      = namedtuple("Var",      "type addr data")
While    = namedtuple("While",    "cond body")
For      = namedtuple("For",      "init cond next body")
Break    = namedtuple("Break",    "")
Continue = namedtuple("Continue", "")
If       = namedtuple("If",       "cond then els")
Return   = namedtuple("Return",   "expr")
Asm      = namedtuple("Asm",      "asm")
NoOp     = namedtuple("NoOp",     "")

@dataclass
class Func:
    name: str
    type: Type   = None
    locals: dict = field(default_factory=dict)
    args: list   = field(default_factory=list)
    body: list   = None
    calls: set   = field(default_factory=set)


KEYWORDS = {
    "include", "var", "const", "func", "struct", "if", "then", "elif", "else",
    "while", "for", "do", "end", "break", "continue", "return", "asm", "or",
    "and", "not",
}

token_regex = re.compile(r"""
    [ \t]*(?P<nl>      \n                          )|
    [ \t]*(?P<comment> \#.*                        )|
    [ \t]*(?P<num>     \$[0-9A-Fa-f]+|[0-9]+       )|
    [ \t]*(?P<id>      [A-Za-z][A-Za-z0-9_]*       )|
    [ \t]*(?P<sym>     ==|!=|<=|>=|\|=|&=|\+=|-=|\*=|/=|%=|
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
        if k == "string":
            line += v.count("\n")
            v = ast.literal_eval(v)
        if k == "id" and v in KEYWORDS: k = v
        out.append(Token(k, v, (path, line)))
    out.append(Token("eof", "", (path, line)))
    return out


INT = Type("int", 0, None)

def is_int(t): return t == INT
def is_struct(t): return t.array == None and t.ptr == 0 and t.base != "int"
def is_ptr(t): return t.array == None and t.ptr > 0
def is_array(t): return t.array != None
def is_cond(t): return is_int(t) or is_ptr(t)

def addr_of(e):
    if isinstance(e, Deref): return e.expr
    return AddrOf(e)

def has_call(e):
    match e:
        case Imm(_): return False
        case VarRef(_): return False
        case AddrOf(_): return False
        case Call(_, _): return True
        case Not(e): return has_call(e)
        case Deref(e): return has_call(e)
        case BinOp(_, a, b): return has_call(a) or has_call(b)
    assert False, e

def split_add(e):
    match e:
        case Imm(v): return (None, None, v)
        case AddrOf(VarRef(n)): return (None, n, 0)
        case BinOp("+", a, b):
            d1, s1, k1 = split_add(a)
            d2, s2, k2 = split_add(b)
            if d1 and d2: d = BinOp("+", d1, d2)
            else: d = d1 or d2
            if s1 and s2: assert False, "WOOT"
            return (d, s1 or s2, k1 + k2)
    return (e, None, 0)


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
        self.globals    = {}
        self.funcs      = {}
        self.structs    = {}
        self.loop       = 0
        self.current_func = None
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

    def const_int_or_addr(self):
        n, t = self.expr()
        dyn, sym, off = split_add(n)
        if dyn: self.error("const expression expected")
        if sym: return f"{sym}+{off}" if off else sym
        return off

    def type(self, auto_array_size=False):
        if not self.peek("sym", ":"): return INT
        self.eat()
        base = self.eat("id").v
        ptr = 0
        while self.peek("sym", "*"): self.eat(); ptr += 1
        array = None
        if self.peek("sym", "["):
            self.eat()
            if auto_array_size and self.peek("sym", "]"):
                array = -1
            else:
                array = self.const_expr()
                if array <= 0: self.error("invalid array size")
            self.eat("sym", "]")
        return Type(base, ptr, array)

    def type_size(self, t):
        if t.ptr > 0: return 1
        n = t.array or 1
        if t.base == "int": return n
        return n * self.get_struct(t.base).size

    def global_var(self):
        name = self.eat("id").v
        if name in self.globals: self.error(f"variable '{name}' already exists")
        type = self.type(True)
        addr = None
        data = None
        if self.peek("sym", "@"):
            self.eat()
            addr = self.const_expr()

        elif self.peek("sym", "="):
            self.eat()

            def parse():
                if self.peek("sym", "{"):
                    self.eat()
                    data = []
                    while not self.peek("sym", "}"):
                        data.append(parse())
                        if not self.peek("sym", ","): break
                        self.eat()
                    self.eat("sym", "}")
                    return data
                if self.peek("string"):
                    return list(map(ord, self.eat("string").v)) + [0]
                return self.const_int_or_addr()

            data = parse()
            if type.array == -1:
                if not isinstance(data, list):
                    self.error("data doesn't match type")
                type = Type(type.base, type.ptr, len(data))

            def null_obj(t):
                if is_array(t):
                    return null_obj(Type(t.base, t.ptr, None)) * t.array
                if is_struct(t):
                    data = []
                    for f in self.get_struct(t.base).fields.values():
                        data += null_obj(f.type)
                    return data
                return [0]

            def unroll(t, d):
                res = []
                if is_array(t) and isinstance(d, list):
                    l = type.array
                    t = Type(t.base, t.ptr, None)
                    if len(d) > l: self.error("too too many initializers")
                    for x in d: res += unroll(t, x)
                    res += null_obj(t) * (l - len(d))
                    return res
                if is_struct(t) and isinstance(d, list):
                    res = []
                    fields = self.get_struct(t.base).fields
                    if len(d) > len(fields): self.error("too too many initializers")
                    for i, f in enumerate(fields.values()):
                        if i < len(d): res += unroll(f.type, d[i])
                        else: res += null_obj(f.type)
                    return res
                if is_ptr(t) and isinstance(d, str) or d == 0:
                    return [d]
                if is_int(t) and isinstance(d, int):
                    return [d]
                self.error("data doesn't match type")

            data = unroll(type, data)

        self.globals[name] = Var(type, addr, data)

    def get_struct(self, name):
        if name not in self.structs: self.error(f"unknown struct '{name}'")
        return self.structs[name]

    def program(self, path):
        self.included.add(path)
        # self.head = SimpleNamespace(toks = tokenize(path), i = 0)
        self.head = Head(tokenize(path))

        while any(self.peek(k) for k in ["include", "const", "var", "func", "struct"]):
            tok = self.eat()
            if tok.k == "include":
                incl = path.parent / self.eat("string").v
                if incl in self.included: continue
                self.stack.append(self.head)
                try: self.program(incl)
                except FileNotFoundError: self.error("file not found")
                self.head = self.stack.pop()

            elif tok.k == "const":
                name = self.eat("id").v
                self.eat("sym", "=")
                self.consts[name] = self.const_expr()

            elif tok.k == "var":
                self.global_var()

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
                func = Func(self.eat("id").v)
                if func.name in self.funcs: self.error(f"function '{func.name}' already exists")
                self.funcs[func.name] = func

                def func_arg():
                    name = self.eat("id").v
                    if name in func.locals: self.error(f"variable '{name}' already exists")
                    type = self.type()
                    if is_array(type): self.error("local arrays not allowed")
                    if is_struct(type): self.error("local structs not allowed")
                    addr = None
                    if self.peek("sym", "@"):
                        self.eat()
                        addr = self.const_expr()
                    func.locals[name] = Var(type, addr, None)
                    func.args.append(name)

                self.eat("sym", "(")
                if not self.peek("sym", ")"):
                    func_arg()
                    while self.peek("sym", ","):
                        self.eat()
                        func_arg()

                self.eat("sym", ")")
                func.type = self.type()
                self.current_func = func
                func.body = self.block()
                self.current_func = None
                self.eat("end")

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
        if self.peek("for"):
            self.eat()
            init = self.stmt()
            if not isinstance(init, Assign) or init.op != "=": self.error("invalid statement")
            self.eat("sym", ",")
            cond, t = self.expr()
            if not is_cond(t): self.error("bad type")
            self.eat("sym", ",")
            next = self.stmt()
            if not isinstance(next, Assign): self.error("invalid statement")
            self.eat("do")
            self.loop += 1
            body = self.block()
            self.loop -= 1
            self.eat("end")
            return For(init, cond, next, body)
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
            # TODO: add void return type to functions
            if self.peek() and self.peek().k in KEYWORDS:
                return Return(None)
            expr, type = self.expr()
            if not (type == self.current_func.type or (is_ptr(self.current_func.type) and expr == Imm(0))):
                self.error("return type mismatch")
            return Return(expr)
        if self.peek("asm"):
            self.eat()
            return Asm(self.eat("string").v)
        if self.peek("var"):
            self.eat()
            name = self.eat("id").v
            if name in self.current_func.locals: self.error(f"variable '{name}' already exists")
            a = VarRef(f"{self.current_func.name}.{name}")
            ta = None
            if self.peek("sym", ":"):
                ta = self.type()
                if is_array(ta): self.error("local arrays not allowed")
                if is_struct(ta): self.error("local structs not allowed")
                self.current_func.locals[name] = Var(ta, None, None)
            if not self.peek("sym", "="):
                self.current_func.locals[name] = Var(ta or INT, None, None)
                return NoOp()
            self.eat("sym", "=")
            b, tb = self.expr()
            if ta != None and ta != tb: self.error("type mismatch") # XXX check for x: Foo* = 0
            if is_array(tb): self.error("local arrays not allowed")
            if is_struct(tb): self.error("local structs not allowed")
            self.current_func.locals[name] = Var(tb, None, None)
            return Assign("=", a, b)


        a, ta = self.primary()
        if isinstance(a, Call): return a

        if isinstance(a, (VarRef, Deref)) and self.peek("sym") and self.peek().v in ASSIGN_OPS:
            op = self.eat().v
            b, tb = self.expr()
            if op == "=" and (ta == tb or is_ptr(ta) and is_int(tb) and b == Imm(0)):
                if is_array(ta): self.error("array assign not allowed")
                if is_struct(ta): self.error("struct assign not allowed")
            elif op in ("+=", "-=") and is_ptr(ta) and is_int(tb):
                # pointer arithmetic
                elem_size = self.type_size(Type(ta.base, ta.ptr - 1, None))
                if elem_size > 1:
                    if isinstance(b, Imm): b = Imm(b.value * elem_size)
                    else: b = BinOp("*", b, Imm(elem_size))
            else:
                if not (is_int(ta) and is_int(tb)): self.error("type mismatch")
            return Assign(op, a, b)

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
                ta = INT
                continue
            if op in CMP_TO_JMP and ta == tb and (is_int(ta) or is_ptr(ta)): ta = INT
            elif op in ("and", "or") and is_cond(ta) and is_cond(tb):        ta = INT
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
            if isinstance(a, Imm): return Imm(-a.value), INT
            return BinOp("*", a, Imm(-1)), INT
        if self.peek("not"):
            self.eat()
            a, t = self.unary()
            if not is_cond(t): self.error("bad type")
            if isinstance(a, Imm): return Imm(int(not a.value)), INT
            return Not(a), INT
        return self.primary()

    def primary(self):
        # number
        if self.peek("num"):
            v = self.eat().v
            if v.startswith("$"): n = int(v[1:], 16)
            else: n = int(v)
            return Imm(n), INT

        # string literal -> int*
        if self.peek("string"):
            data = list(map(ord, self.eat().v)) + [0]
            name = f"_str_{len(self.globals)}"
            self.globals[name] = Var(Type("int", 0, len(data)), None, data)
            node = AddrOf(VarRef(name))
            type = Type("int", 1, None)

        else:
            # constant
            name = self.eat("id").v
            if name in self.consts: return Imm(self.consts[name]), INT

            # function call
            if self.peek("sym", "("):
                func = self.funcs.get(name)
                if not func: self.error(f"unknown function '{name}'")
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
                if len(args) != len(func.args): self.error("wrong number of function arguments")
                for t, n in zip(types, func.args):
                    if t != func.locals[n].type: self.error("type mismatch")
                self.current_func.calls.add(func.name)
                return Call(name, args), func.type

            # variable
            if self.current_func and name in self.current_func.locals:
                node, type = VarRef(f"{self.current_func.name}.{name}"), self.current_func.locals[name].type
            else:
                if name not in self.globals: self.error(f"unknown variable '{name}'")
                node, type = VarRef(name), self.globals[name].type

        while True:
            if self.peek("sym", "["):
                self.eat()
                if not is_array(type): self.error("cannot index non-array")
                idx, tidx = self.expr()
                if not is_int(tidx): self.error("index must be int")
                self.eat("sym", "]")
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

            elif self.peek("sym", "."):
                self.eat()
                field = self.eat("id").v
                if is_array(type):
                    if field == "len": return Imm(type.array), INT
                    if field == "ptr":
                        node = AddrOf(node)
                        type = Type(type.base, type.ptr + 1, None)
                        continue
                    if field == "limit":
                        offset = self.type_size(Type(type.base, type.ptr, None)) * type.array
                        node = BinOp("+", AddrOf(node), Imm(offset))
                        type = Type(type.base, type.ptr + 1, None)
                        continue

                if is_ptr(type): # auto pointer deref
                    node = Deref(node)
                    type = Type(type.base, type.ptr - 1, None)
                if not is_struct(type): self.error("request for member of non-struct")
                f = self.get_struct(type.base).fields.get(field)
                if not f: self.error(f"struct '{type.base}' has no field '{field}'")
                type = f.type
                if f.offset > 0: node = Deref(BinOp("+", addr_of(node), Imm(f.offset)))

            else: return node, type


def peephole(lines, tmp_vars):
    for _ in range(2):
        # remove unused labels
        used_labels = set()
        for l in lines:
            if m := re.match(r"    j.. (_[^ ]+)", l): used_labels.add(m[1])
        new_lines = []
        for l in lines:
            if m := re.match(r"(_[^: ]+):$", l):
                if m[1] not in used_labels: continue
            new_lines.append(l)
        lines = []
        # remove dead code
        dead = False
        for l in new_lines:
            if dead and re.match(r"(_[^: ]+):$", l): dead = False
            if not dead: lines.append(l)
            if l == "    ret": dead = True

    def process(block):
        new_block = []
        while block:
            match block:
                case [("mov", a, b), ("cmp", c, d), *rest] if a in tmp_vars and a == c:
                    new_block.append(("cmp", b, d))
                    block = rest
                case [(op, a, b), ("cmp", c, "#0"), *rest] if a == c:
                    new_block.append((op, a, b))
                    block = rest
                case [("mov", a, b), (op, c, d), *rest] if a in tmp_vars and a == d:
                    new_block.append((op, c, b))
                    block = rest
                case [("mov", a, b), *foo] if a in tmp_vars:
                    q = []
                    while True:
                        match block:
                            case [(op, c, d), *rest] if c == a:
                                block = rest
                                q.append((op, d))
                                continue
                            case [("mov", c, d), *rest] if d == a and not any(c == x for _, x in q):
                                block = rest
                                for op, x in q: new_block.append((op, c, x))
                                break
                        new_block.append(("mov", a, b))
                        block = foo
                        break
                case [b, *rest]:
                    new_block.append(b)
                    block = rest
        return new_block

    new_lines = []
    block     = []
    operands  = set()
    for line in lines:
        if m := re.match(r"    ([^ ]{3}) ([^, ]+), ([^ ]+)", line):
            block.append(m.groups())
            continue
        if m := re.match(r"    (j..) ([^ ]+)", line):
            block.append(m.groups())
            continue
        if block:
            pa = None
            for o in process(process(block)):
                match o:
                    case op, a, b:
                        operands |= {a, b}
                        if op != "mov" and a == pa: a = "%" # previous destination address
                        else: pa = a
                        new_lines.append(f"    {op} {a}, {b}")
                    case j, l:
                        new_lines.append(f"    {j} {l}")
            block = []
        new_lines.append(line)

    # remove unused tmp vars
    for t in list(tmp_vars.keys()):
        if t not in operands: del tmp_vars[t]

    return new_lines



class Codegen:

    def newtmp(self, *ts):
        for t in ts:
            if t in self.tmp_vars: return t
            if t.startswith("[") and t[1:-1] in self.tmp_vars: return t[1:-1]
        t = f"{self.current_func.name}._T{self.tmp_i}"
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
        while i > 0 and (m := re.match(f"    jmp {l}", self.lines[i])):
            self.lines.pop(i)
            i -= 1
        while i > 0 and (m := re.match(r"    j[^m]. ([^ ]+)", self.lines[i])):
            if m[1] == l: self.lines.pop(i)
            i -= 1
        self.lines.append(f"{l}:")

    def branch(self, node, T, F):
        if isinstance(node, Imm):
            if node.value: self.emit(f"    jmp {T}")
            else: self.emit(f"    jmp {F}")

        elif isinstance(node, VarRef):
            self.emit(f"    cmp {node.name}, #0")
            self.emit(f"    jeq {F}")
            self.emit(f"    jne {T}")

        elif isinstance(node, Not):
            self.branch(node.expr, F, T)

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
            a = self.value(node)
            self.emit(f"    cmp {a}, #0")
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
            if a.startswith("["):
                t, a = a, self.newtmp()
                self.emit(f"    mov {a}, {t}")
            a = f"[{a}]"
            if sym: a += f"+{sym}"
            if off: a += f"+{off}"
            return a

        if isinstance(node, Not):
            t = self.newtmp()
            T = self.label("true")
            E = self.label("done")
            self.emit(f"    mov {t}, #0")
            self.branch(node.expr, E, T)
            self.emit_label(T)
            self.emit(f"    mov {t}, #1")
            self.emit_label(E)
            return t

        if isinstance(node, BinOp):
            if node.op == "+":
                dyn, sym, off = split_add(node)
                if not dyn: return f"#{sym}+{off}"
            if op := ARITH_OPS.get(node.op):
                a = self.value(node.a)
                b = self.value(node.b)
                if node.op in {"-", "/", "%"}:
                    t = self.newtmp(a)
                    if t != a: self.emit(f"    mov {t}, {a}")
                    self.emit(f"    {op} {t}, {b}")
                else:
                    t = self.newtmp(a, b)
                    if t == a: self.emit(f"    {op} {t}, {b}")
                    elif t == b: self.emit(f"    {op} {t}, {a}")
                    else:
                        self.emit(f"    mov {t}, {a}")
                        self.emit(f"    {op} {t}, {b}")
                return t
            if node.op in CMP_TO_JMP or node.op in {"and", "or"}:
                t = self.newtmp()
                T = self.label("true")
                E = self.label("done")
                self.emit(f"    mov {t}, #1")
                self.branch(node, E, T)
                self.emit_label(T)
                self.emit(f"    mov {t}, #0")
                self.emit_label(E)
                return t

            assert False, f"unsupported binop {node.op}"

        if isinstance(node, Call):
            self.call(node)
            t = self.newtmp()
            self.emit(f"    mov {t}, _R")
            return t

        assert False, f"unsupported expression {node}"


    def call(self, node):
        func = self.funcs[node.name]
        staged = []
        for e in node.args:
            t = None
            if has_call(e):
                v = self.value(e)
                t = self.newtmp(v)
                if t != v: self.emit(f"    mov {t}, {v}")
            staged.append(t)
        for a, e, v in zip(func.args, node.args, staged):
            if not v: v = self.value(e)
            self.emit(f"    mov {func.name}.{a}, {v}")
        self.emit(f"    jsr {node.name}")



    def stmt(self, node):
        self.tmp_i = 0 # reuse temporary variables

        if isinstance(node, Assign):
            assert isinstance(node.a, (VarRef, Deref))
            a = self.value(node.a)
            b = self.value(node.b)
            op = ASSIGN_OPS[node.op]
            if a != b or node.op != "=": self.emit(f"    {op} {a}, {b}")

        elif isinstance(node, While):
            L = self.label("while")
            T = self.label("while_true")
            E = self.label("while_end")
            self.emit_label(L)
            self.branch(node.cond, T, E)
            self.emit_label(T)
            self.loop_stack.append((L, E))
            for st in node.body: self.stmt(st)
            self.loop_stack.pop()
            self.emit(f"    jmp {L}")
            self.emit_label(E)

        elif isinstance(node, For):
            L = self.label("for")
            T = self.label("for_true")
            N = self.label("for_next")
            E = self.label("for_end")
            self.stmt(node.init)
            self.emit_label(L)
            self.branch(node.cond, T, E)
            self.emit_label(T)
            self.loop_stack.append((N, E))
            for st in node.body: self.stmt(st)
            self.loop_stack.pop()
            self.emit_label(N)
            self.stmt(node.next)
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
            E = self.label("if_end")
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

        elif isinstance(node, Return):
            if node.expr:
                v = self.value(node.expr)
                self.emit(f"    mov _R, {v}")
            self.emit(f"    ret")

        elif isinstance(node, Call):
            self.call(node)

        elif isinstance(node, Asm):
            self.lines += node.asm.rstrip().split("\n")

        elif isinstance(node, NoOp): pass

        else:
            assert False, f"unsupported statement {node}"


    def compile(self, ast):
        self.lines      = []
        self.label_i    = 0
        self.globals    = ast.globals
        self.funcs      = ast.funcs
        self.structs    = ast.structs
        self.loop_stack = []
        self.ast        = ast

        # functions
        lines = []
        for name, f in self.funcs.items():
            self.current_func = f
            self.tmp_vars     = {} # use dict to keep original order
            self.lines        = []
            self.emit("")
            self.emit(f"    ; function {name}")
            self.emit(f"{name}:")
            for st in f.body: self.stmt(st)
            self.emit("    ret")
            # optimize asm

            lines += peephole(self.lines, self.tmp_vars)

            # add temporary variables to locals
            f.locals.update({t.split(".", 1)[1]: Var(INT, None, None) for t in self.tmp_vars})

        # variables with fixed addresses
        addr = 0
        self.lines = []
        for name, v in self.globals.items():
            if v.addr != None:
                assert v.data == None
                self.emit(f"{name} = {v.addr}")
                addr = max(addr, v.addr + ast.type_size(v.type))
        for name, f in self.funcs.items():
            for k, v in f.locals.items():
                if v.addr != None:
                    assert v.data == None
                    self.emit(f"{name}.{k} = {v.addr}")
                    addr = max(addr, v.addr + ast.type_size(v.type))

        # special register for return value
        self.emit(f"_R = {addr}")
        addr += 1

        cache = {}
        def alloc_vars(name):
            if name in cache: return cache[name]
            func = self.funcs[name]
            a = addr
            for f in func.calls: a = max(a, alloc_vars(f))
            for k, v in func.locals.items():
                assert not v.data
                if v.addr == None:
                    self.emit(f"{name}.{k} = {a}")
                    a += ast.type_size(v.type)
            cache[name] = a
            return a
        addr = max(alloc_vars(name) for name in self.funcs)

        # global variables
        for name, v in self.globals.items():
            if v.addr == None and not v.data:
                self.emit(f"{name} = {addr}")
                addr += ast.type_size(v.type)

        # data
        self.emit(f"")
        self.emit(f"    .data {addr}")
        for name, v in self.globals.items():
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
    path = args.out or Path(args.src).with_suffix(".asm")
    path.write_text(asm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("out", nargs="?")
    args = parser.parse_args()
    main(args)
