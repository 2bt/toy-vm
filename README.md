# toy-vm

A tiny fantasy-console–style virtual machine.
Write games in a C-like language (`.toy`), compile to bytecode, and run them on either the native SDL backend or the browser (JavaScript) backend.

**[Play Breakout in the browser](https://2bt.github.io/toy-vm/)**

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
* Literals: decimals, `$hex`, `'c'`, `"string"` (zero-terminated `int` array; adjacent string literals concatenate)
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
