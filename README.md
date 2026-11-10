# Smolt

A tiny fantasy-console–style virtual machine.
Write games in a C-like language (`.smolt`), compile to bytecode, and run them on either the native SDL backend or the browser (JavaScript) backend.

![Outpost Oberon](media/outpost-oberon.gif)

**[Play in the browser!](https://2bt.github.io/smolt/)**

| | |
|---|---|
| Display | 320×180, 60 fps, true colour (no palette) |
| Graphics | one blit call: any rectangle of `sprite.png`, with H/V flip |
| Sound | 8 voices: triangle, saw, pulse (width 0–1023), noise; pitch in ⅛ semitones (456 = A4), volume 0–256 |
| Input | d-pad, A, B, menu (keyboard, gamepad, or touch in the browser) |
| CPU | 32-bit words, memory-to-memory instructions, up to 1M instructions per frame |
| Memory | 2²⁴ words of data; code in a separate instruction stream |
| Storage | one save slot per game |
| Language | Smolt: indentation-based, C-flavoured, compiled to assembly |
| Hosts | native (C++/SDL2) and browser (JavaScript), same bytecode |

Games live in a folder with:

* `code` → compiled VM bytecode
* `sprite.png` → sprite sheet

Shared code lives in `lib/`. An include path starting with `lib/` always refers to it (e.g. `include "lib/api.smolt"`); any other path is relative to the including file.

* `api.smolt` → host ABI: input, interrupts, constants
* `util.smolt` → input helpers, math, strings, text drawing, boxes
* `audio.smolt` → music and sound-effect players

Game Controls:

* Arrows → movement
* Z / X / C → buttons
* Esc → quit
* Return → screenshot (`screenshot-<time>.png`, 320×180)
* R → start/stop recording frames to `recording-<time>/`; `./gif.sh recording-<time>` turns them into a gif
* Gamepad: d-pad / left stick, A or Y → A, B or X → B, Start → menu


## Build & Run

Dependencies: C++20, SDL2, SDL2_image, CMake, Python 3

```bash
./run.sh game-breakout
```

This will:

1. Build the VM (`smolt`)
2. Compile `main.smolt` → `main.asm`
3. Assemble `main.asm` → `code`
4. Run the game in the VM

Native host: `build/smolt [--once] <game-dir>`. Browser host: `web/index.html` (same bytecode + ABI).


## Machine model

* **Word:** signed 32-bit integer. Pointers are word addresses into `mem`.
* **Memory:** flat `mem[0 … 2²⁴)` (words). Code lives in a separate instruction stream (`pc` indexes that, not `mem`).
* **Calls:** `jsr` / `ret` use a host-side return stack. No recursion in Smolt (locals are not re-entrant).
* **Entry:** each frame the host updates `input` and runs from `pc = 0`; the compiler’s entry is `func tick():`.

### Host ABI (low memory)

Pinned in `lib/api.smolt` via `var` / formals `@ addr`:

| Addr | Role |
|------|------|
| `0` | `ret` — host writes interrupt results (e.g. `rand`, `sqrt`) |
| `1` | `input` — button bitfield (`1<<0` up … `1<<5` B) |
| `2…` | interrupt argument blocks (disjoint per interrupt) |

Interrupts (`asm "int #N"`):

| N | Name | Args | Notes |
|---|------|------|--------|
| 1 | print | `msg` @ 2 (`int*`) | zero-terminated chars; newline after |
| 2 | sprite | `x y w h s t flip` @ 3–9 | blit from `sprite.png` |
| 3 | sound | `i pitch pw vol wave` @ 10–14 | voice `i` |
| 4 | rand | `lo hi` @ 15–16 | result → `mem[0]` |
| 5 | sqrt | `a` @ 17 | result → `mem[0]` |
| 6 | save | `addr len` @ 20–21 | write `len` words from `addr` to the game's save slot; words written → `mem[0]` |
| 7 | load | `addr len` @ 20–21 | read up to `len` words into `addr`; words read → `mem[0]` (0 = no save) |

Screen is 320×180.

Each game has one save slot: `<game-dir>/save` (raw words) natively, `localStorage` in the browser.


## Assembler & bytecode

`tools/compile.py main.smolt` emits text assembly (`main.asm`) and assembles it into the `code` blob (`--asm-only` skips that step); `tools/assemble.py` can also be run on its own.

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


## Smolt language

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

```smolt
include "other.smolt"

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

```smolt
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
Ternary: `cond ? a : b`. GNU form `cond ?: b` reuses `cond`'s value (evaluated once).

### Expressions

Operators (low → high binding): `or`, `and`, comparisons, `|`, `&` (bitwise), `+` `-`, `*` `/` `%`, then postfix / unary.

* Unary: `-`, `not`, `&` (address of lvalue)
* Assignments: `=`, `+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`
* Literals: decimals, `$hex`, `'c'`, `"string"` (zero-terminated `int` array; may span lines; adjacent string literals concatenate), `{1, 2, 3}` (`int[N]` value, no trailing 0; decays/interns like strings when an address is needed)
* `true` / `false` / `null` are ordinary constants if you define them

### Pointers and arrays

```smolt
var p: Box* = boxes[4]   # auto-& when destination is a pointer
p.x = 1                  # auto-deref on field access
p.* = other              # whole-value assign through pointer
q = p as int*            # pointer → pointer cast (reinterpret)
a[i]                     # index array or pointer
a.ptr / a.len / a.limit  # array address, length, one-past-end pointer
```

No recursion (direct or mutual). Locals are not re-entrant.


## License

MIT, see [LICENSE](LICENSE).
