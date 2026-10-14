# Runtime Init-List Assign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow `aggregate = { expr, ... }` (nested braces and whole-aggregate slots) by parsing an `InitList` and desugaring it to existing `Assign` / `Copy` nodes.

**Architecture:** Parse `{...}` only on the assign RHS into an untyped `InitList`. Recursively match against the LHS aggregate type and emit a `Block` of `Assign`/`Copy` statements. Codegen only needs to walk `Block`; scalar and copy emission stay unchanged. No RHS staging temps — stores are left-to-right.

**Tech Stack:** Python compiler (`compile.py`), existing unittest harness (`test/test_exec.py`), VM via `build/toy --once`.

## Global Constraints

- Only as RHS of `=` onto an aggregate lvalue (not returns/args/locals).
- Exact slot count (no zero-fill).
- Nested `{...}` or same-type aggregate expression per aggregate slot.
- Global const `var x: T = { ... }` path unchanged.
- Left-to-right stores; overlap is undefined / programmer's problem.

## File map

| File | Role |
|------|------|
| `compile.py` | `InitList`, `Block`, parse, `lower_init`, assign hook, codegen for `Block` |
| `test/test_exec.py` | Exec + compile-error tests |
| `todo.md` | Mark init-list item done when finished |

---

### Task 1: Struct init list (`b = { 3, 7 }`)

**Files:**
- Modify: `compile.py` (AST nodes, parse, lower, assign path, codegen)
- Test: `test/test_exec.py`

**Interfaces:**
- Produces: `InitList(items)`, `Block(stmts)`, `Parser.init_list()`, `Parser.lower_init(dst, ty, init) -> list`, codegen handles `Block`

- [ ] **Step 1: Write the failing test**

Add to `test/test_exec.py`:

```python
def test_init_list_struct(self):
    self.assertEqual(self.run_toy("""
        struct V:
            x
            y
        var b: V
        func tick():
            b = { 3, 7 }
            print_n(b.x)
            print_n(b.y)
    """), "3\n7\n")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 test/test_exec.py ExecTests.test_init_list_struct -v`

Expected: FAIL (compile error — `{` not accepted as assign RHS)

- [ ] **Step 3: Add AST nodes**

In `compile.py` near `Copy` / `NoOp`:

```python
InitList = namedtuple("InitList", "items")  # items: InitList | (expr, type)
Block    = namedtuple("Block",    "stmts")
```

- [ ] **Step 4: Parse `InitList`**

Add methods on `Parser`:

```python
def init_list(self):
    # peek().v == "{", token kind "sym"
    self.eat()
    items = []
    if not (self.peek() and self.peek().v == "}"):
        items.append(self.init_list_item())
        while self.peek() and self.peek().v == ",":
            self.eat()
            items.append(self.init_list_item())
    if not (self.peek() and self.peek().v == "}"):
        self.error("expected }")
    self.eat()
    return InitList(items)

def init_list_item(self):
    if self.peek() and self.peek().v == "{":
        return self.init_list()
    e, t = self.expr()
    return (e, t)
```

- [ ] **Step 5: Lvalue helpers + `lower_init`**

Reuse the same addressing rules as `primary()` (field offset 0 keeps base node; index 0 keeps base node):

```python
def field_lvalue(self, base, offset):
    if offset == 0: return base
    return Deref(BinOp("+", addr_of(base), Imm(offset)))

def elem_lvalue(self, base, index, elem_ty):
    if index == 0: return base
    off = self.type_size(elem_ty) * index
    return Deref(BinOp("+", addr_of(base), Imm(off)))

def lower_init(self, dst, ty, init):
    if not isinstance(init, InitList):
        self.error("expected { ... }")
    stmts = []
    if is_struct(ty):
        fields = list(self.get_struct(ty.base).fields.values())
        if len(init.items) != len(fields):
            self.error("wrong number of initializers")
        for item, f in zip(init.items, fields):
            slot = self.field_lvalue(dst, f.offset)
            stmts += self.lower_init_slot(slot, f.type, item)
    elif is_array(ty):
        elem = Type(ty.base, ty.ptr, None)
        if len(init.items) != ty.array:
            self.error("wrong number of initializers")
        for i, item in enumerate(init.items):
            slot = self.elem_lvalue(dst, i, elem)
            stmts += self.lower_init_slot(slot, elem, item)
    else:
        self.error("type error")
    return stmts

def lower_init_slot(self, dst, ty, item):
    if is_array(ty) or is_struct(ty):
        if isinstance(item, InitList):
            return self.lower_init(dst, ty, item)
        e, tb = item
        self.check_types_for_assign(ty, tb, e)
        return [Copy(dst, e, self.type_size(ty))]
    # scalar
    if isinstance(item, InitList):
        self.error("type error")
    e, tb = item
    self.check_types_for_assign(ty, tb, e)
    return [Assign("=", dst, e)]
```

- [ ] **Step 6: Hook assign path**

In `Parser.stmt`, where `op == "="` after parsing LHS, **before** calling `self.expr()` for the RHS:

```python
if op == "=" and self.peek() and self.peek().v == "{":
    if not (is_array(ta) or is_struct(ta)):
        self.error("type error")
    init = self.init_list()
    self.eat("eol")
    return Block(self.lower_init(a, ta, init))
```

Keep the existing `b, tb = self.expr()` path for non-brace RHS (including `Copy`).

- [ ] **Step 7: Codegen `Block`**

In `Codegen.stmt`:

```python
elif isinstance(node, Block):
    for st in node.stmts:
        self.stmt(st)
```

- [ ] **Step 8: Run test to verify it passes**

Run: `python3 test/test_exec.py ExecTests.test_init_list_struct -v`

Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add compile.py test/test_exec.py
git commit -m "Parse struct init lists and lower to field assigns."
```

---

### Task 2: Array init list + expressions in slots

**Files:**
- Modify: `test/test_exec.py`
- (Implementation already in `lower_init` from Task 1 if array branch is complete)

**Interfaces:**
- Consumes: `lower_init` array branch from Task 1

- [ ] **Step 1: Write failing tests**

```python
def test_init_list_array(self):
    self.assertEqual(self.run_toy("""
        var a: int[3]
        func tick():
            a = { 1, 2, 3 }
            print_n(a[0])
            print_n(a[1])
            print_n(a[2])
    """), "1\n2\n3\n")

def test_init_list_expressions(self):
    self.assertEqual(self.run_toy("""
        struct V:
            x
            y
        var b: V
        var x = 4
        var y = 5
        func tick():
            b = { x + 1, y * 2 }
            print_n(b.x)
            print_n(b.y)
    """), "5\n10\n")
```

- [ ] **Step 2: Run tests**

Run: `python3 test/test_exec.py ExecTests.test_init_list_array ExecTests.test_init_list_expressions -v`

Expected: PASS if Task 1 array/scalar paths are correct; otherwise fix `lower_init` / `elem_lvalue` until PASS.

- [ ] **Step 3: Commit**

```bash
git add test/test_exec.py compile.py
git commit -m "Test array and expression slots in init lists."
```

---

### Task 3: Nested braces + aggregate slot from value

**Files:**
- Modify: `test/test_exec.py`
- Modify: `compile.py` only if nested/`Copy` slot path is incomplete

**Interfaces:**
- Consumes: `lower_init_slot` aggregate branches

- [ ] **Step 1: Write failing tests**

```python
def test_init_list_nested(self):
    self.assertEqual(self.run_toy("""
        struct Inner:
            a
            b
        struct Outer:
            i: Inner
            x
        var o: Outer
        func tick():
            o = { { 8, 9 }, 1 }
            print_n(o.i.a)
            print_n(o.i.b)
            print_n(o.x)
    """), "8\n9\n1\n")

def test_init_list_aggregate_slot(self):
    self.assertEqual(self.run_toy("""
        struct Inner:
            a
            b
        struct Outer:
            i: Inner
            x
        var u: Outer
        var o: Outer
        func tick():
            u.i.a = 2
            u.i.b = 3
            o = { u.i, 1 }
            print_n(o.i.a)
            print_n(o.i.b)
            print_n(o.x)
    """), "2\n3\n1\n")
```

- [ ] **Step 2: Run tests**

Run: `python3 test/test_exec.py ExecTests.test_init_list_nested ExecTests.test_init_list_aggregate_slot -v`

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add test/test_exec.py compile.py
git commit -m "Support nested and aggregate-valued init-list slots."
```

---

### Task 4: Pointer and indexed destinations

**Files:**
- Modify: `test/test_exec.py`

**Interfaces:**
- Consumes: existing LHS `primary()` lvalues + `lower_init`

- [ ] **Step 1: Write failing tests**

```python
def test_init_list_via_pointer(self):
    self.assertEqual(self.run_toy("""
        struct V:
            x
            y
        var b: V
        func tick():
            var p = &b
            p.* = { 4, 5 }
            print_n(b.x)
            print_n(b.y)
    """), "4\n5\n")

def test_init_list_indexed(self):
    self.assertEqual(self.run_toy("""
        struct V:
            x
            y
        var vs: V[3]
        func tick():
            vs[2] = { 1, 2 }
            print_n(vs[2].x)
            print_n(vs[2].y)
    """), "1\n2\n")
```

- [ ] **Step 2: Run tests**

Run: `python3 test/test_exec.py ExecTests.test_init_list_via_pointer ExecTests.test_init_list_indexed -v`

Expected: PASS (LHS already builds `Deref` / indexed lvalues; `lower_init` only needs those as `dst`)

- [ ] **Step 3: Commit**

```bash
git add test/test_exec.py
git commit -m "Test init lists through pointers and array elements."
```

---

### Task 5: Compile errors

**Files:**
- Modify: `test/test_exec.py`

**Interfaces:**
- Consumes: error paths in assign hook + `lower_init`

- [ ] **Step 1: Write failing tests**

```python
def test_init_list_wrong_length(self):
    compiled = self.compile_text("""
        struct V:
            x
            y
        var b: V
        func tick():
            b = { 1 }
    """)
    self.assertNotEqual(compiled.returncode, 0)
    self.assertIn("wrong number of initializers", compiled.stderr)

def test_init_list_onto_scalar(self):
    compiled = self.compile_text("""
        var n
        func tick():
            n = { 1 }
    """)
    self.assertNotEqual(compiled.returncode, 0)
    self.assertIn("type error", compiled.stderr)

def test_init_list_type_mismatch(self):
    compiled = self.compile_text("""
        struct V:
            x
            y
        var b: V
        var p: int*
        func tick():
            b = { p, 1 }
    """)
    self.assertNotEqual(compiled.returncode, 0)
    self.assertIn("type error", compiled.stderr)
```

- [ ] **Step 2: Run tests**

Run: `python3 test/test_exec.py ExecTests.test_init_list_wrong_length ExecTests.test_init_list_onto_scalar ExecTests.test_init_list_type_mismatch -v`

Expected: PASS (adjust error strings in `compile.py` only if messages differ — keep them matching the asserts above)

- [ ] **Step 3: Full suite**

Run: `python3 test/test_exec.py -q`

Expected: OK (all tests)

- [ ] **Step 4: Mark todo + commit**

In `todo.md`, change the init-list line to checked:

```
+ [x] init via x = { ... } (full expressions per slot)
```

```bash
git add test/test_exec.py compile.py todo.md
git commit -m "Reject bad init lists; mark init-list todo done."
```

---

## Spec coverage check

| Spec requirement | Task |
|------------------|------|
| Assign-RHS only `{...}` | 1 (hook before `expr()`) |
| Exact arity | 1 + 5 |
| Nested braces | 3 |
| Aggregate slot = value → Copy | 3 |
| Left-to-right, no temps | 1 (`Assign`/`Copy` order) |
| Global const path unchanged | (untouched) |
| Exec tests listed in spec | 1–4 |
| Compile errors | 5 |
