#!/usr/bin/env python3
import argparse
import pathlib
import re
import sys
from collections import namedtuple

OPCODE_TABLE = []
def load_opcode_table():
    path = pathlib.Path(__file__).resolve().parent.parent / "src" / "vm.cpp"
    with path.open() as f:
        while l := f.readline():
            if "OPCODE_TABLE" in l: break
        while l := f.readline():
            l = l.strip()
            if not l or l.startswith("//"): continue
            if not l.startswith("{"): break
            m = list(map(str.lower, re.findall(r'[A-Z]+', l)))
            m += ["nil"] * (3 - len(m))
            OPCODE_TABLE.append(tuple(m))
        if not OPCODE_TABLE:
            sys.exit("no opcode table")

load_opcode_table()

Operand = namedtuple("Operand", "mode value offset", defaults=(0, 0))
Inst = namedtuple("Inst", "op target a b nr")  # nr: source line, for errors
NIL  = Operand("nil")


token_regex = re.compile(r"""
    \s*(?P<comment> ;.*                    )|
    \s*(?P<name>    [A-Za-z_][A-Za-z0-9_.]*)|
    \s*(?P<number>  [0-9]+|\$[A-Fa-f0-9]*  )|
    \s*(?P<sym>     [\[\]():,\.#=+\-*/%]   )|
    \s*(?P<other>   [^\s]+                 )
""", re.VERBOSE)


def tokenize(line):
    tokens = []
    pos = 0
    while m := token_regex.match(line, pos):
        pos = m.end()
        t = m.lastgroup
        if t == "comment": break
        v = m.group(m.lastgroup)
        tokens.append((t, v))
    return tokens


variables = {}
line      = None
line_nr   = 0
src_name  = "?"


def error(msg, nr=None):
    sys.exit(f"{src_name}:{nr or line_nr}: {msg}" + ("" if nr else f"\n{line}"))

def bad_line():
    error("bad line")

def expect(ts, t, v):
    match ts:
        case [(tt, vv), *rest] if tt == t and vv == v: return rest
    bad_line()


PRECEDENCE = {
    "+": 1,
    "-": 1,
    "*": 2,
    "/": 2,
    "%": 2,
}

def eval_expression(ts, min_p=1):
    match ts:
        case [("sym", "-"), *rest]:
            a, ts = eval_expression(rest, 99)
            a = -a
        case [("number", v), *rest]:
            if v.startswith("$"): a = int(v[1:], 16)
            else: a = int(v)
            ts = rest
        case [("sym", "("), *rest]:
            a, ts = eval_expression(rest)
            ts = expect(ts, "sym", ")")
        case [("name", v), *rest] if v in variables:
            a = variables[v]
            ts = rest
        case _: bad_line()

    while 1:
        match ts:
            case [("sym", o), *rest] if PRECEDENCE.get(o, 0) >= min_p:
                b, ts = eval_expression(rest, PRECEDENCE[o] + 1)
                match o:
                    case "+": a += b
                    case "-": a -= b
                    case "*": a *= b
                    case "/": a //= b
                    case "%": a %= b
            case _: break
    return a, ts


def encode(out: bytearray, v: int):
    # encode int28 into up to 4 bytes
    # negative numbers take up 4 bytes
    if not -2**27 <= v < 2**27: sys.exit(f"{src_name}: constant {v} doesn't fit in 28 bits")
    v &= 0xfffffff
    while v >= 0x80:
        out.append((v & 0x7f) | 0x80)
        v >>= 7
    out.append(v)


def optimize(insts, labels):
    n = len(insts)
    preds = [[] for _ in range(n + 1)]  # n is the exit
    preds[0].append(-1)  # the host enters at 0 with unknown VM state
    for i, inst in enumerate(insts):
        op = inst.op
        dest = None
        if inst.target:
            if inst.target not in labels:
                error(f"undefined label '{inst.target}'", inst.nr)
            dest = labels[inst.target]
        if op == "ret":
            continue
        if op == "jmp":
            preds[dest].append(i)
        elif op == "jsr":
            # Call enters the callee, then execution resumes at the next instruction.
            preds[dest].append(i)
            preds[i + 1].append(i)
        elif op.startswith("j"):
            preds[i + 1].append(i)
            preds[dest].append(i)
        else:
            preds[i + 1].append(i)

    def merge(slots, outgoing):
        changed = True
        while changed:
            changed = False
            for i in range(n):
                incoming = None
                for p in preds[i]:
                    other = () if p < 0 else outgoing(p, slots)
                    if incoming is None: incoming = other
                    elif other is None: continue
                    elif incoming != other: incoming = ()
                if slots[i] != incoming:
                    slots[i] = incoming
                    changed = True

    # VM dest pointer `a` and source cursor `c` reaching each instruction
    incoming_a = [None] * n
    def outgoing_a(p, slots):
        inst = insts[p]
        if inst.op == "jsr": return ()
        if not inst.op.startswith("j") and inst.a.mode != "nil": return inst.a
        return slots[p]
    merge(incoming_a, outgoing_a)

    incoming_c = [None] * n
    def outgoing_c(p, slots):
        inst = insts[p]
        if inst.op == "jsr": return ()
        if inst.b.mode in ("abs", "ind", "idx"): return inst.b
        return slots[p]
    merge(incoming_c, outgoing_c)

    def is_nxt(have, o):
        if have == Operand("abs", o.value - 1) and o.mode == "abs": return True
        if have == Operand("idx", o.value, o.offset - 1) and o.mode == "idx": return True
        if have == Operand("ind", o.value) and o.mode == "idx" and o.offset == 1: return True
        return False

    optimized = []
    for i, inst in enumerate(insts):
        have_a = incoming_a[i]
        have_c = incoming_c[i]
        a, b = inst.a, inst.b
        if have_a:
            if inst.op == "mov" and is_nxt(have_a, a):
                a = a._replace(mode="nxt")
            elif inst.op != "mov" and have_a == a:
                a = a._replace(mode="pda")
        if inst.op == "mov" and a.mode == "nxt" and have_c and is_nxt(have_c, b):
            if (inst.op, "nxt", "nxt") in OPCODE_TABLE:
                b = b._replace(mode="nxt")
        if b.mode == "imm":
            for value, mode in ((0, "zro"), (1, "one"), (-1, "neg")):
                if b.value == value and (inst.op, a.mode, mode) in OPCODE_TABLE:
                    b = b._replace(mode=mode)
        optimized.append(inst._replace(a=a, b=b))
    return optimized


def asm(args):
    global variables, line_nr, line, src_name
    src_name  = args.src.name
    variables = {}
    line_nr   = 0
    line      = None
    labels    = {}
    insts     = []
    section   = "code"
    data_base = None
    data      = []

    for l in args.src.open():
        line = l.rstrip()
        line_nr += 1
        ts = tokenize(line)

        match ts:
            # code section
            case [("sym", "."), ("name", "code")]:
                section = "code"
                continue

            # data section
            case [("sym", "."), ("name", "data"), ("number", base)]:
                section = "data"
                if data_base is not None: bad_line()
                data_base = int(base)
                continue

            # variable
            case [("name", name), ("sym", "="), *rest]:
                if name in variables: bad_line()
                variables[name], rest = eval_expression(rest)
                if rest: bad_line()
                continue

            # label
            case [("name", name), ("sym", ":"), *rest]:
                ts = rest
                if section == "code":
                    if name in labels:
                        sys.exit(f"{line_nr}: label '{name}' already used")
                    labels[name] = len(insts)
                if section == "data":
                    if name in variables: bad_line()
                    variables[name] = len(data) + data_base

        if section == "data":
            while ts:
                v, ts = eval_expression(ts)
                data.append(v)
                if ts: ts = expect(ts, "sym", ",")
            continue

        # opcode
        match ts:
            case []:
                continue
            case [("name", op), *rest]:
                ts = rest
            case _: bad_line()

        is_jump = op.startswith("j")
        target  = None
        a, b    = NIL, NIL

        # operand a
        match ts:
            case []: pass

            # label
            case [("name", label)] if is_jump:
                ts = []
                b = Operand("imm")  # label is actually operand b
                target = label

            # immediate
            case [("sym", "#"), *rest]:
                # immediate as operand a: only "int #n"
                value, ts = eval_expression(rest)
                b = Operand("imm", value)

            # indirect
            case [("sym", "["), *rest]:
                value, ts = eval_expression(rest)
                ts = expect(ts, "sym", "]")
                a = Operand("ind", value)
                match ts:
                    case [("sym", "+"), *rest]:
                        offset, ts = eval_expression(rest)
                        a = Operand("idx", value, offset)

            # absolute
            case rest:
                value, ts = eval_expression(rest)
                a = Operand("abs", value)

        # operand b
        match ts:
            # immediate
            case [("sym", ","), ("sym", "#"), *rest]:
                value, ts = eval_expression(rest)
                b = Operand("imm", value)

            # indirect
            case [("sym", ","), ("sym", "["), *rest]:
                value, ts = eval_expression(rest)
                ts = expect(ts, "sym", "]")
                b = Operand("ind", value)
                match ts:
                    case [("sym", "+"), *rest]:
                        offset, ts = eval_expression(rest)
                        b = Operand("idx", value, offset)

            # absolute
            case [("sym", ","), *rest]:
                value, ts = eval_expression(rest)
                b = Operand("abs", value)

        if ts: bad_line()
        insts.append(Inst(op, target, a, b, line_nr))

    insts = optimize(insts, labels)
    bin_cmds = []
    jumps = []
    for i, inst in enumerate(insts):
        op, target, a, b, nr = inst
        if target: jumps.append((i, target))
        try:
            opc = OPCODE_TABLE.index((op, a.mode, b.mode))
        except ValueError:
            error(f"no '{op}' instruction with operands {a.mode}, {b.mode}", nr)
        bin_cmd = [opc]
        if a.mode in ("abs", "ind", "idx"): bin_cmd.append(a.value)
        if a.mode == "idx": bin_cmd.append(a.offset)
        if b.mode in ("abs", "ind", "idx", "imm"): bin_cmd.append(b.value)
        if b.mode == "idx": bin_cmd.append(b.offset)
        bin_cmds.append(bin_cmd)

    # fix labels
    def get_label_addr(label):
        return sum(len(cmd) for cmd in bin_cmds[:labels[label]])
    for pos, label in jumps:
        bin_cmd = bin_cmds[pos]
        bin_cmd[1] = get_label_addr(label)

    # write binary
    code = []
    for cmd in bin_cmds: code += cmd

    path = args.out or args.src.parent / "code"
    out = bytearray()
    encode(out, data_base or 0)
    encode(out, len(data))
    for x in data: encode(out, x)
    encode(out, len(code))
    for x in code: encode(out, x)
    path.write_bytes(out)

    # print assembled code
    if args.print:
        pos = 0
        for bin in bin_cmds:
            q = " ".join(map(str, bin))
            bs = bytearray()
            for x in bin: encode(bs, x)
            bs = bs.hex(" ").upper()
            l = f"{pos:6} : {q:<20} {bs:<27}"
            pos += len(bin)
            o, a, b = OPCODE_TABLE[bin.pop(0)]
            l += f" {o}"
            if o.startswith("j"):
                l += f" {bin.pop(0)}"
            else:
                if a == "abs": l += f" {bin.pop(0)}"
                if a == "ind": l += f" [{bin.pop(0)}]"
                if a == "idx": l += f" [{bin.pop(0)}]+{bin.pop(0)}"
                if a == "pda": l += " %"
                if a == "nxt": l += " ++%"
                if a != "nil" and b != "nil": l += ","
                if b == "abs": l += f" {bin.pop(0)}"
                if b == "ind": l += f" [{bin.pop(0)}]"
                if b == "idx": l += f" [{bin.pop(0)}]+{bin.pop(0)}"
                if b == "imm": l += f" #{bin.pop(0)}"
                if b == "zro": l += " #0"
                if b == "one": l += " #1"
                if b == "neg": l += " #-1"
                if b == "nxt": l += " ++%"
            print(l)
    print("codes:", len(code))
    print("size:", len(out))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=pathlib.Path)
    parser.add_argument("out", nargs="?", type=pathlib.Path)
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args()
    asm(args)
