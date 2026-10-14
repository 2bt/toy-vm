# Runtime init lists: `x = { ... }`

## Goal

Allow assigning an aggregate (struct or fixed array) from a brace list of full expressions:

```
pad = { 0, y, PAD_START_H, 0 }
o   = { {8, 9}, 1 }
o   = { other.i, 1 }
p.* = { 1, 2 }
```

Builds on existing word-wise `Copy` / field `Assign`. Does not replace the compile-time global initializer `var x: T = { ... }` (const ints only).

## Syntax & semantics

- `{...}` is allowed only as the RHS of `=` onto an aggregate lvalue (global struct/array, indexed element, or `p.*`).
- Each list element is a full expression, or a nested `{...}`.
- Slot count must match the target exactly (struct fields in declaration order, or array length). Too few or too many is an error. No zero-fill / partial init.
- Slot typing:
  - scalar slot ← int or pointer expression (normal assign type rules)
  - aggregate slot ← nested `{...}` **or** an expression of the same aggregate type (lowered to `Copy`)
- Evaluation order: left to right, each slot stored immediately (normal `Assign` / `Copy`). No RHS staging temps. Overlap such as `a = { a.y, a.x }` is the programmer's problem.

## Out of scope

- Partial lists / zero-fill
- `{...}` outside assign RHS (returns, call args, etc.)
- Unifying with global const `var x = { ... }`
- Local struct/array variables or their init
- Safe swap / overlap detection

## AST & lowering

- New node: `InitList(items)` — list of expression nodes or nested `InitList`s. Untyped when parsed.
- Parse `{...}` only when an assign RHS begins with `{` (not as a general primary), so it cannot appear in other expression positions.
- On `lvalue = InitList` with aggregate LHS type `T`, recursively lower:

  `lower_init(dst, T, InitList) -> sequence of Assign | Copy`

  - **struct:** zip declaration-order fields with items; build each field lvalue (same addressing as `.field` today); lower that slot
  - **array:** zip indices `0 .. len-1` with items; build each element lvalue; lower that slot
  - **scalar expected + expr:** type-check and emit `Assign("=", dst, expr)`
  - **aggregate expected + InitList:** recurse
  - **aggregate expected + aggregate expr:** `Copy(dst, expr, type_size)`
  - length or type mismatch → compile error
- The statement is that sequence (a small `Block`/`Seq` node, or flatten into the function body). Codegen for `Assign` / `Copy` stays as today.

## Tests

**Exec:**

- struct: `b = { 3, 7 }`
- array: `a = { 1, 2, 3 }`
- nested braces: `o = { {8, 9}, 1 }`
- aggregate slot from value: `o = { u.i, 1 }`
- via pointer / index: `p.* = { ... }`, `vs[2] = { ... }`
- expressions in slots: `b = { x + 1, y * 2 }`

**Compile errors:** wrong length; `{...}` onto scalar; type mismatch in a slot.
