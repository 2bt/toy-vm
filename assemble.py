#!/usr/bin/python
import argparse
import re
import pathlib
import struct


OPCODE_TABLE = []

def load_opcode_table():
    f = open("src/vm.cpp")
    while l := f.readline():
        if "OPCODE_TABLE" in l: break
    while l := f.readline():
        l = l.strip()
        if not l: continue
        if not l.startswith("{"): break
        m = list(map(str.lower, re.findall(r'[A-Z]+', l)))
        m += ["nil"] * (3 - len(m))
        OPCODE_TABLE.append(tuple(m))
        o, a, b = m
    if not OPCODE_TABLE:
        print("no opcode table")
        exit(1)

load_opcode_table()


token_regex = re.compile(r"""
    \s*(?P<comment> ;.*                    )|
    \s*(?P<name>    [A-Za-z_][A-Za-z0-9_]* )|
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
    print(f"{nr}: bad line:")
    print(line)
    exit(1)


def expect(ts, t, v):
    match ts:
        case [(tt, vv), *rest] if tt == t and vv == v:
            return rest
        case _: bad_line()


PRECEDENCE = {
    "+": 1,
    "-": 1,
    "*": 2,
    "/": 2,
    "%": 2,
}

def eval_expression(ts, min_p=0):
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
            case [("sym", o), *rest] if o in PRECEDENCE and PRECEDENCE[o] >= min_p:
                b, ts = eval_expression(rest, PRECEDENCE[o] + 1)
                match o:
                    case "+": a += b
                    case "-": a -= b
                    case "*": a *= b
                    case "/": a //= b
                    case "%": a %= b
            case _: break
    return a, ts


DATA_OFFSET = 100


def asm(args):
    labels   = {}
    jumps    = []
    data     = []
    bin_cmds = []
    section  = "code"

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

            # code section
            case [("sym", "."), ("name", "data")]:
                section = "data"
                continue

            # variable
            case [("name", name), ("sym", "="), *rest]:
                variables[name], rest = eval_expression(rest)
                if rest: bad_line()
                continue

            # label
            case [("name", name), ("sym", ":"), *rest]:
                ts = rest
                if section == "code":
                    if name in labels:
                        print(f"{nr}: label '{name}' already used")
                        exit(1)
                    labels[name] = len(bin_cmds)
                if section == "data":
                    variables[name] = len(data) + DATA_OFFSET

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
        mode_a  = "nil"
        value_a = 0
        mode_b  = "nil"
        value_b = 0

        # operand a
        match ts:
            # label
            case [("name", label)] if is_jump:
                ts = []
                mode_b = "imm" # label is actually operand b
                jumps.append((len(bin_cmds), label))

            # immediate
            case [("sym", "#"), *rest]:
                mode_b = "imm" # XXX: only INT uses immediate
                value_b, ts = eval_expression(rest)

            # indirect
            case [("sym", "["), *rest]:
                mode_a = "ind"
                value_a, ts = eval_expression(rest)
                ts = expect(ts, "sym", "]")

            # absolute
            case []: pass
            case rest:
                mode_a = "abs"
                value_a, ts = eval_expression(rest)

        # operand b
        match ts:
            # immediate
            case [("sym", ","), ("sym", "#"), *rest]:
                mode_b = "imm"
                value_b, ts = eval_expression(rest)

            # indirect
            case [("sym", ","), ("sym", "["), *rest]:
                mode_b = "ind"
                value_b, ts = eval_expression(rest)
                ts = expect(ts, "sym", "]")

            # absolute
            case [("sym", ","), *rest]:
                mode_b = "abs"
                value_b, ts = eval_expression(rest)

        if ts: bad_line()

        try: opc = OPCODE_TABLE.index((op, mode_a, mode_b))
        except ValueError: bad_line()
        bin_cmd = [opc]
        if mode_a != "nil": bin_cmd.append(value_a)
        if mode_b != "nil": bin_cmd.append(value_b)
        bin_cmds.append(bin_cmd)


    # fix labels
    def get_label_addr(label):
        return sum(len(cmd) for cmd in bin_cmds[:labels[label]])
    for pos, label in jumps:
        if label not in labels:
            print(f"undefined label '{label}'")
            exit(1)
        bin_cmd = bin_cmds[pos]
        bin_cmd[1] = get_label_addr(label)


    # print assembled code
    pos = 0
    for bin_cmd in bin_cmds:
        o, a, b = OPCODE_TABLE[bin_cmd[0]]
        q = " ".join(map(str, bin_cmd))
        l = f"{pos:6} : {q:<16} {o}"
        if o.startswith("j"):
            l += f" {bin_cmd[1]}"
        else:
            if a == "abs": l += f" {bin_cmd[1]}"
            if a == "ind": l += f" [{bin_cmd[1]}]"
            if a != "nil" and b != "nil": l += ","
            if b == "abs": l += f" {bin_cmd[-1]}"
            if b == "ind": l += f" [{bin_cmd[-1]}]"
            if b == "imm": l += f" #{bin_cmd[-1]}"
        print(l)
        pos += len(bin_cmd)

    # write binary
    out = args.out or pathlib.Path(args.src).parent / "code"
    code = []
    for cmd in bin_cmds: code += cmd
    with open(out, "wb") as f:
        f.write(struct.pack(f"i", len(data)))
        f.write(struct.pack(f"{len(data)}i", *data))
        f.write(struct.pack(f"i", len(code)))
        f.write(struct.pack(f"{len(code)}i", *code))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("out", nargs="?")
    args = parser.parse_args()
    asm(args)
