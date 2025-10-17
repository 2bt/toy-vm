#!/usr/bin/env python3
import argparse
import pathlib
import re
import sys

OPCODE_TABLE = []
def load_opcode_table():
    with open("src/vm.cpp") as f:
        while l := f.readline():
            if "OPCODE_TABLE" in l: break
        while l := f.readline():
            l = l.strip()
            if not l or l.startswith("//"): continue
            if not l.startswith("{"): break
            m = list(map(str.lower, re.findall(r'[A-Z]+', l)))
            m += ["nil"] * (3 - len(m))
            OPCODE_TABLE.append(tuple(m))
            o, a, b = m
        if not OPCODE_TABLE:
            sys.exit("no opcode table")

load_opcode_table()

OPTIMIZE = True


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
nr        = 0
line      = None


def bad_line():
    sys.exit(f"{nr}: bad line:\n{line}")

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
    assert -2**27 <= v < 2**27
    v &= 0xfffffff
    while v >= 0x80:
        out.append((v & 0x7f) | 0x80)
        v >>= 7
    out.append(v)


def asm(args):
    labels    = {}
    jumps     = []
    bin_cmds  = []
    section   = "code"
    data_base = None
    data      = []

    prev_a = None, None, None

    for l in open(args.src):
        global line
        global nr
        line = l.rstrip()
        nr += 1
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
                        sys.exit(f"{nr}: label '{name}' already used")
                    labels[name] = len(bin_cmds)
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

        is_jump  = op.startswith("j")
        a_mode   = "nil"
        a_value  = 0
        a_offset = 0
        b_mode   = "nil"
        b_value  = 0
        b_offset = 0

        # operand a
        match ts:
            case []: pass

            # label
            case [("name", label)] if is_jump:
                ts = []
                b_mode = "imm" # label is actually operand b
                jumps.append((len(bin_cmds), label))

            # immediate
            case [("sym", "#"), *rest]:
                b_mode = "imm" # XXX: only INT uses immediate
                b_value, ts = eval_expression(rest)

            # indirect
            case [("sym", "["), *rest]:
                a_mode = "ind"
                a_value, ts = eval_expression(rest)
                ts = expect(ts, "sym", "]")
                # offset
                match ts:
                    case [("sym", "+"), *rest]:
                        a_mode = "idx"
                        a_offset, ts = eval_expression(rest)

            # previous address
            case [("sym", "%"), *rest]:
                a_mode = "pda"
                ts = rest

            # next address
            case [("sym", "+"), ("sym", "+"), ("sym", "%"), *rest]:
                a_mode = "nxt"
                ts = rest

            # absolute
            case rest:
                a_mode = "abs"
                a_value, ts = eval_expression(rest)

        # operand b
        match ts:
            # immediate
            case [("sym", ","), ("sym", "#"), *rest]:
                b_mode = "imm"
                b_value, ts = eval_expression(rest)

            # indirect
            case [("sym", ","), ("sym", "["), *rest]:
                b_mode = "ind"
                b_value, ts = eval_expression(rest)
                ts = expect(ts, "sym", "]")
                # offset
                match ts:
                    case [("sym", "+"), *rest]:
                        b_mode = "idx"
                        b_offset, ts = eval_expression(rest)

            # absolute
            case [("sym", ","), *rest]:
                b_mode = "abs"
                b_value, ts = eval_expression(rest)

        if ts: bad_line()

        a_mode_opt = a_mode
        if OPTIMIZE:
            a = a_mode, a_value, a_offset
            if is_jump or op == "ret":
                prev_a = None, None, None
            elif op != "int":
                if op == "mov":
                    prev_a_mode, prev_a_value, prev_a_offset = prev_a
                    if prev_a_mode == a_mode:
                        if a_mode == "abs" and prev_a_value == a_value - 1:
                            a_mode_opt = "nxt"
                        elif a_mode == "idx" and prev_a_value == a_value and prev_a_offset == a_offset - 1:
                            a_mode_opt = "nxt"
                else:
                    if a == prev_a: a_mode_opt = "pda"
                prev_a = a

            if b_mode == "imm":
                if b_value == 0 and (op, a_mode_opt, "zro") in OPCODE_TABLE: b_mode = "zro"
                if b_value == 1 and (op, a_mode_opt, "one") in OPCODE_TABLE: b_mode = "one"

        try:
            opc = OPCODE_TABLE.index((op, a_mode_opt, b_mode))
        except ValueError:
            print(op, a_mode_opt, b_mode)
            bad_line()
        bin_cmd = [opc]
        if a_mode_opt in ("abs", "ind", "idx"): bin_cmd.append(a_value)
        if a_mode_opt == "idx": bin_cmd.append(a_offset)
        if b_mode in ("abs", "ind", "idx", "imm"): bin_cmd.append(b_value)
        if b_mode == "idx": bin_cmd.append(b_offset)
        bin_cmds.append(bin_cmd)

    # fix labels
    def get_label_addr(label):
        return sum(len(cmd) for cmd in bin_cmds[:labels[label]])
    for pos, label in jumps:
        if label not in labels:
            sys.exit(f"undefined label '{label}'")
        bin_cmd = bin_cmds[pos]
        bin_cmd[1] = get_label_addr(label)

    # write binary
    code = []
    for cmd in bin_cmds: code += cmd

    path = args.out or pathlib.Path(args.src).parent / "code"
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
            print(l)
    print("codes:", len(code))
    print("size:", len(out))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("out", nargs="?")
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args()
    asm(args)
