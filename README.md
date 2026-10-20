# toy-vm

A tiny fantasy-console–style virtual machine.
Write games in a C-like language (`.toy`), compile to bytecode, and run them on either the native SDL backend or the browser (JavaScript) backend.

**[Play in the browser!](https://2bt.github.io/toy-vm/)**

Games live in a folder with:

* `code` → compiled VM bytecode
* `sprite.png` → sprite sheet

Game Controls:

* Arrows → movement
* Z / X / C → buttons
* Esc → quit


## Build & Run

Dependencies: C++17, SDL2, SDL2_image, CMake, Python 3

```bash
./run.sh game-breakout
```

This will:

1. Build the VM (`toy`)
2. Compile `main.toy` → `main.asm`
3. Assemble `main.asm` → `code`
4. Run the game in the VM

Native host: `build/toy [--once] <game-dir>`. Browser host: `web/index.html` (same bytecode + ABI).


## Machine model

* **Word:** signed 32-bit integer. Pointers are word addresses into `mem`.
* **Memory:** flat `mem[0 … 2²⁴)` (words). Code lives in a separate instruction stream (`pc` indexes that, not `mem`).
* **Calls:** `jsr` / `ret` use a host-side return stack. No recursion in Toy (locals are not re-entrant).
* **Entry:** each frame the host updates `input` and runs from `pc = 0`; the compiler’s entry is `func tick():`.

### Host ABI (low memory)

Pinned by games (see e.g. `game-breakout/prelude.toy`) via `var` / formals `@ addr`:

| Addr | Role |
|------|------|
| `0` | `ret` — host writes interrupt results (e.g. `rand`, `sqrt`) |
| `1` | `input` — button bitfield (`1<<0` up … `1<<5` B) |
| `2…` | interrupt argument block (overlaid per call) |

Interrupts (`asm "int #N"`):

| N | Name | Args @ 2+ | Notes |
|---|------|-----------|--------|
| 1 | print | `msg` (`int*`) | zero-terminated chars; newline after |
| 2 | sprite | `x y w h s t flip` | blit from `sprite.png` |
| 3 | sound | `i pitch pw vol wave` | voice `i` |
| 4 | rand | `lo hi` | result → `mem[0]` |
| 5 | sqrt | `a` | result → `mem[0]` |

Screen is 320×180.


## Assembler & bytecode

`compile.py` emits text assembly; `assemble.py` writes the `code` blob.

**Assembly** (`;` comments):

* `.code` — instructions (default)
* `.data BASE` — words loaded into `mem` starting at `BASE` (compiler places globals here)
* `name = expr` — assemble-time constant
* `label:` — code or data label
* Ops: `mov` `add` `sub` `mul` `div` `mod` `and` `ore` `cmp`, jumps (`jmp` `jsr` `ret`, `jlt`…`jne`), `int #N`
* Operands: `addr`, `#imm`, `[addr]`, `[base]+off` (dest, src). Opcode encoding picks a concrete addressing mode from `OPCODE_TABLE` in `src/vm.cpp`.

**`code` file layout** (all fields are signed 28-bit values, LEB-style 7-bit bytes):

1. `data_base`, `data_len`, then `data_len` words → `mem[data_base…]`
2. `code_len`, then `code_len` instruction words (opcodes + immediates)

Each instruction is an opcode index into `OPCODE_TABLE` (mnemonic + operand modes), followed by the absolute/indirect/index/immediate words that mode needs.


## Toy language

Indentation-based, C-flavored. Entry point is `func tick():` (called every frame).
`#` starts a line comment.

### Types

* `int` — word-sized integer (default when a type is omitted)
* `T*` — pointer (multiple `*` allowed)
* `T[N]` — fixed array; `T[]` only on `var` with an initializer (size inferred)
* `struct` — fields are `int` by default, or annotated (`box: Box`)

Untyped `var x = …` takes the type of the right-hand side (structs/arrays are **copied**).
A typed destination can coerce: arrays and lvalues decay (`&`) into pointers when the destination wants `T*`.

### Declarations

```toy
include "other.toy"

const N = 10

struct Point:
    x
    y

var origin: Point = {0, 0}
var mem @ 100             # fixed address

func add1(x) int:
    return x + 1
```

* `func name(args) RetType:` — return type is optional (defaults to `int`). No aggregate returns.
* Formals: `name` or `name: Type`. Optional `@ addr` pins a formal/global to a fixed address (host ABI / MMIO).
* `asm "..."` — emit raw assembly in a function body.

### Control flow

```toy
if cond:
    ...
elif other:
    ...
else:
    ...

while cond:
    ...

for i = 0, n:       # i runs while i < n
    ...
for p in items:     # p is a pointer walking the array
    ...
```

`break` / `continue` work in loops.
Ternary: `cond ? a : b`.

### Expressions

Operators (low → high binding): `or`, `and`, comparisons, `|`, `&` (bitwise), `+` `-`, `*` `/` `%`, then postfix / unary.

* Unary: `-`, `not`, `&` (address of lvalue)
* Assignments: `=`, `+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`
* Literals: decimals, `$hex`, `'c'`, `"string"` (zero-terminated `int` array; adjacent string literals concatenate), `{1, 2, 3}` (`int[N]` value, no trailing 0; decays/interns like strings when an address is needed)
* `true` / `false` / `null` are ordinary constants if you define them

### Pointers and arrays

```toy
var p: Box* = boxes[4]   # auto-& when destination is a pointer
p.x = 1                  # auto-deref on field access
p.* = other              # whole-value assign through pointer
q = p as int*            # pointer → pointer cast (reinterpret)
a[i]                     # index array or pointer
a.ptr / a.len / a.limit  # array address, length, one-past-end pointer
```

No recursion (direct or mutual). Locals are not re-entrant.
