# Entity Coords + `level_move_box_*` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `sx`/`sy` the subpixel top-left of each entity AABB, then give enemies `level_move_box_x` / `level_move_box_y` so they stop hand-rolling left/right/floor resolution.

**Architecture:** Two phases. Phase 1 redefines the coordinate model everywhere (hero + enemies), deletes `entity_update_box_pos`, and rewrites probes/draw/camera for the new origin. Phase 2 adds axis move helpers next to `check_collision` and switches patrol/spider resolution to them; hero keeps multi-sensor collision.

**Tech Stack:** Toy (`.toy`), `./compile.py game-outpost-oberon/main.toy` or `./run.sh game-outpost-oberon`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-10-22-entity-coords-and-move-box-design.md`
- Do **not** put hero on `level_move_box_*`.
- Draw offsets stay **inline** in sprite calls (no `spr_ox` fields / named offset tables).
- No automated `.toy` unit-test harness — verify via compile + playtest.
- Prefer not to `git add` under `docs/` (gitignored); game source / `code` only at commit steps.
- Commit only when the human asks, or at explicit task commit steps.

## File structure

| File | Role |
|------|------|
| `game-outpost-oberon/game.toy` | Hero spawn/collision/draw/camera; `check_collision*`; **new** `level_move_box_x` / `level_move_box_y` (must live here — they call `check_collision`, and `level.toy` is included *before* those helpers exist) |
| `game-outpost-oberon/level.toy` | Delete `entity_update_box_pos` |
| `game-outpost-oberon/enemy.toy` | Spawn, patrol/spider probes → new origin; then switch resolution to move helpers; draw offsets |

### Origin conversion (old → new)

Old model (`entity_update_box_pos`):

```text
box.x = ox + 8 - w/2
box.y = oy + 16 - h
```

New: `(ox, oy) = (box.x, box.y)`. Any probe that used old origin offsets `(ox+a, oy+b)` becomes:

```text
(box.x + a - (8 - w/2), box.y + b - (16 - h))
```

| Entity | w×h | Subtract from old (a,b) |
|--------|-----|-------------------------|
| Hero   | 6×11 | (5, 5) |
| Patrol | 8×17 | (4, −1) → x−4, y+1 |
| Spider | 12×11 | (2, 5) |

---

### Task 1: Coordinate unify (hero + enemies, no move helpers yet)

**Files:**
- Modify: `game-outpost-oberon/game.toy` (hero spawn, update collision, draw, camera)
- Modify: `game-outpost-oberon/enemy.toy` (spawn, patrol/spider probes + sync, draw)
- Modify: `game-outpost-oberon/level.toy` (delete `entity_update_box_pos`)

**Interfaces:**
- Consumes: `check_collision`, `check_collision_down`, `SP`, `box_center_*`
- Produces: `sx`/`sy` = subpixel box top-left; `box.x = sx/SP`, `box.y = sy/SP` after movement; no `entity_update_box_pos`

- [ ] **Step 1: Delete `entity_update_box_pos` and replace every call with box sync**

In `level.toy`, remove the function and its comment (lines ~209–213).

Everywhere that called `entity_update_box_pos(e.sx, e.sy, e.box)` or hero equivalent, use:

```toy
box.x = sx / SP
box.y = sy / SP
```

(for the relevant entity). Call sites today: `hero_place_at_start`, end of `hero_update` collision, `enemy_spawn`, end of `enemy_update_patrol`, end of `enemy_update_spider`.

- [ ] **Step 2: Hero spawn + camera + draw**

`hero_place_at_start` — place the **box** on the spawn tile (old visual: box at `spawn + (5,5)`):

```toy
func hero_place_at_start():
    hero.sx = (level.spawn_x + 5) * SP
    hero.sy = (level.spawn_y + 5) * SP + SP - 1
    hero.box.x = hero.sx / SP
    hero.box.y = hero.sy / SP
    if check_collision_down(hero.box.x, hero.box.y + hero.box.h,
                            hero.box.y + hero.box.h - 1):
        hero.frame = 0
        hero.on_ground = true
    else:
        hero.frame = 4
        hero.on_ground = false
    # ... reset vx/vy/facing/countdowns unchanged ...
```

Camera — follow box center (old origin was near mid-cell; center keeps framing close):

```toy
func camera_center_on_hero():
    camera.box.x = box_center_x(hero.box) - SCREEN_W / 2
    camera.box.y = box_center_y(hero.box) - SCREEN_H / 2

func camera_follow_hero():
    var hx = box_center_x(hero.box)
    var hy = box_center_y(hero.box)
    camera.box.x = clamp(camera.box.x, hx + CAM_MARGIN_H - SCREEN_W, hx - CAM_MARGIN_H)
    var ty = hy - SCREEN_H / 2
    camera.box.y = clamp(camera.box.y, ty, hy - CAM_MARGIN_T)
    if hero.on_ground and ty > 0: camera.box.y += (ty - camera.box.y) / 8

func camera_return_toward_spawn():
    var tx = clamp(box_center_x(hero.box) - SCREEN_W / 2, 0, level.w * TS - SCREEN_W)
    var ty = clamp(box_center_y(hero.box) - SCREEN_H / 2, 0, level.h * TS - SCREEN_H)
    # ... rest unchanged (dx/dy/speed) ...
```

`hero_draw` — old sprite at origin; box was origin+(5,5):

```toy
camera_sprite(hero.box.x - 5, hero.box.y - 5, 16, 16,
              hero.frame * 16,
              128 + row * 16,
              hero.facing_left)
```

- [ ] **Step 3: Rewrite hero collision probes (subtract (5,5) from old offsets)**

In `hero_update`, keep the same structure; `x`/`y` are now box top-left pixels (`hero.sx/SP`, `hero.sy/SP`). Replace the probe block with:

```toy
    hero.sx += hero.vx
    var x = hero.sx / SP
    var y = hero.sy / SP

    var o
    o = max(check_collision(x + 1, y +  2, DIR_LEFT),
            check_collision(x - 1, y +  8, DIR_LEFT))
    if o:
        x += o
        hero.sx = x * SP
        hero.vx = 0
    o = min(check_collision(x + 4, y +  2, DIR_RIGHT),
            check_collision(x + 6, y +  8, DIR_RIGHT))
    if o:
        x += o
        hero.sx = x * SP
        if hero.vx > 0: hero.sx += SP - 1
        hero.vx = 0

    # jump / gravity unchanged ...

    var py = hero.sy / SP
    hero.sy += hero.vy
    y = hero.sy / SP
    o = min(check_collision_down(x + 0, y + 10, py + 10),
            check_collision_down(x + 5, y + 10, py + 10))
    if o:
        if not hero.on_ground: play_sound(SOUND_LAND)
        hero.on_ground = true
        y += o
        hero.sy = y * SP + SP - 1
        hero.vy = 0
    else:
        hero.on_ground = false

    o = check_collision(x + 3 - hero.facing_left, y + 0, DIR_UP)
    if o:
        y += o
        hero.sy = y * SP
        hero.vy = 8

    # shoot muzzle (old x+4/x+11, y+9 → new)
    # ...
        var bx = hero.facing_left ? x - 1 : x + 6
        hero_bullet_spawn(bx, y + 4, hero.facing_left)

    # after anim:
    hero.box.x = hero.sx / SP
    hero.box.y = hero.sy / SP
```

Dying path that only does `hero.sy += hero.vy` should still sync box if anything reads it while dying (particles use `hero.box` — sync there too, or sync once per frame before particles):

```toy
        hero.sy += hero.vy
        hero.box.y = hero.sy / SP
```

- [ ] **Step 4: Enemy spawn + draw for new origin**

`enemy_spawn` — tile `(x,y)` was old origin (tile top-left). Box was at `(x + 8 - w/2, y + 16 - h)`:

```toy
        e.box.w = ...  # set w/h first as today
        e.box.h = ...
        e.sx = (x + 8 - e.box.w / 2) * SP
        e.sy = (y + 16 - e.box.h) * SP
        e.box.x = e.sx / SP
        e.box.y = e.sy / SP
```

`enemy_draw_all`:

```toy
        if e.type == ENEMY_TYPE_PATROL:
            # old: (sx/SP, sy/SP - 16); box = origin + (4, -1)
            camera_sprite(e.box.x - 4, e.box.y - 15, 16, 32,
                          256 + e.frame * 16,
                          0 + (e.hit_countdown > 0) * 32,
                          e.facing_left)
        elif e.type == ENEMY_TYPE_SPIDER:
            # old: (sx/SP, sy/SP); box = origin + (2, 5)
            camera_sprite(e.box.x - 2, e.box.y - 5, 16, 16,
                          256 + e.frame * 16,
                          80 + (e.hit_countdown > 0) * 16,
                          e.facing_left)
```

- [ ] **Step 5: Rewrite enemy resolution probes (temporary; Task 3–4 replace)**

**Patrol** walk block — old face at `x+4` / `x+11`, `y+8`, ledge `y+16`; subtract (4, −1):

```toy
        e.sx += e.facing_left ? -6 : 6
        x = e.sx / SP
        y = e.sy / SP

        var face_x = e.facing_left ? x + 0 : x + 7
        var wall_dir = e.facing_left ? DIR_LEFT : DIR_RIGHT
        var o = check_collision(face_x, y + 9, wall_dir)
        var turn = -1
        if o or not check_collision_down(face_x, y + 17, y + 16):
            x += o
            e.sx = x * SP
            turn = 1
        # ... rest of turn/pace logic unchanged ...

    e.box.x = e.sx / SP
    e.box.y = e.sy / SP
```

**Spider** x/y collision — old probes minus (2, 5):

```toy
    e.sx += e.vx
    var x = e.sx / SP
    var y = e.sy / SP

    var o
    var u
    o = max(check_collision(x + 2, y + 3, DIR_LEFT),
            check_collision(x + 1, y + 7, DIR_LEFT))
    u = min(check_collision(x + 9, y + 3, DIR_RIGHT),
            check_collision(x + 10, y + 7, DIR_RIGHT))
    if o or u:
        x += o + u
        e.sx = x * SP
        # ... bounce / turn logic unchanged ...

    e.vy += GRAVITY
    e.vy = min(e.vy, SP * 4)
    var py = e.sy / SP
    e.sy += e.vy
    y = e.sy / SP
    var was_air = not e.on_ground
    o = min(check_collision_down(x + 3, y + 10, py + 10),
            check_collision_down(x + 8, y + 10, py + 10))
    if o:
        e.on_ground = true
        y += o
        e.sy = y * SP + SP - 1
        e.vy = 0
        # ... land pause unchanged ...
    else:
        e.on_ground = false

    o = check_collision(x + 6, y + 1, DIR_UP)
    if o:
        y += o
        e.sy = y * SP
        e.vy = 8

    e.box.x = e.sx / SP
    e.box.y = e.sy / SP
```

Wall/ledge **AI** probes already use `e.box.*` — leave them. Facing compare `e.sx > hero.sx` still works (both top-left).

Dying drift (`e.sx += e.vx` etc.) — sync box if particles need it:

```toy
            e.sx += e.vx
            e.sy += e.vy
            e.box.x = e.sx / SP
            e.box.y = e.sy / SP
```

- [ ] **Step 6: Compile + playtest phase 1**

Run: `./compile.py game-outpost-oberon/main.toy`  
Expected: success, no reference to `entity_update_box_pos`.

Run: `./run.sh game-outpost-oberon`  
Expected:
1. Hero spawn standing on floor; sprite aligned with debug box overlay if enabled.
2. Run/jump/land/`^` platforms feel like before.
3. Camera still tracks hero without a large jump at level start.
4. Patrols walk, turn at walls/ledges, shoot; spiders sleep/chase/jump; sprites sit on their boxes.

- [ ] **Step 7: Commit (optional)**

```bash
git add game-outpost-oberon/game.toy game-outpost-oberon/enemy.toy game-outpost-oberon/level.toy game-outpost-oberon/code
git commit -m "Make entity sx/sy the subpixel top-left of the hitbox."
```

---

### Task 2: Add `level_move_box_x` / `level_move_box_y`

**Files:**
- Modify: `game-outpost-oberon/game.toy` (immediately after `check_collision_down`)

**Interfaces:**
- Consumes: `check_collision`, `check_collision_down`, `DIR_*`, `SP`, `Box`
- Produces:
  - `func level_move_box_x(sx: int*, box: Box*, dx) int` → `0 | DIR_LEFT | DIR_RIGHT`
  - `func level_move_box_y(sy: int*, box: Box*, dy) int` → `0 | DIR_UP | DIR_DOWN`

Toy: lvalues decay to pointers, so callers write `level_move_box_x(e.sx, e.box, e.vx)`.

- [ ] **Step 1: Implement both helpers**

```toy
# Resolve axis motion for enemy boxes. Mutates sx/sy and box.x/box.y.
# Probes: two points on the leading face (simple; not hero-grade).

func level_move_box_x(sx: int*, box: Box*, dx) int:
    sx.* += dx
    box.x = sx.* / SP
    if dx == 0: return 0

    var x = box.x
    var y = box.y
    var y0 = y + box.h / 4
    var y1 = y + box.h * 3 / 4
    var o = 0
    var hit = 0

    if dx < 0:
        o = max(check_collision(x, y0, DIR_LEFT),
                check_collision(x, y1, DIR_LEFT))
        if o:
            x += o
            sx.* = x * SP
            box.x = x
            hit = DIR_LEFT
    else:
        o = min(check_collision(x + box.w - 1, y0, DIR_RIGHT),
                check_collision(x + box.w - 1, y1, DIR_RIGHT))
        if o:
            x += o
            sx.* = x * SP
            if dx > 0: sx.* += SP - 1
            box.x = sx.* / SP
            hit = DIR_RIGHT
    return hit


func level_move_box_y(sy: int*, box: Box*, dy) int:
    var py = sy.* / SP
    sy.* += dy
    box.y = sy.* / SP
    if dy == 0: return 0

    var x = box.x
    var y = box.y
    var x0 = x + box.w / 4
    var x1 = x + box.w * 3 / 4
    var o = 0
    var hit = 0

    if dy > 0:
        var foot = y + box.h - 1
        var pfoot = py + box.h - 1
        o = min(check_collision_down(x0, foot, pfoot),
                check_collision_down(x1, foot, pfoot))
        if o:
            y += o
            sy.* = y * SP + SP - 1
            box.y = sy.* / SP
            hit = DIR_DOWN
    else:
        o = max(check_collision(x0, y, DIR_UP),
                check_collision(x1, y, DIR_UP))
        if o:
            y += o
            sy.* = y * SP
            box.y = y
            hit = DIR_UP
    return hit
```

- [ ] **Step 2: Compile**

Run: `./compile.py game-outpost-oberon/main.toy`  
Expected: success (helpers unused is fine).

- [ ] **Step 3: Commit (optional)**

```bash
git add game-outpost-oberon/game.toy game-outpost-oberon/code
git commit -m "Add level_move_box_x/y for enemy axis resolution."
```

---

### Task 3: Patrol uses `level_move_box_*`

**Files:**
- Modify: `game-outpost-oberon/enemy.toy` (`enemy_update_patrol`)

**Interfaces:**
- Consumes: `level_move_box_x`, `level_move_box_y` (y unused if patrol is grounded-only — still call y with `dy=0` only if needed; patrol today has no gravity — **only call `level_move_box_x` for walk**)
- Produces: patrol walk/turn via helper return + separate ledge probe

- [ ] **Step 1: Replace walk resolution**

Patrol has no `vy` gravity today — only horizontal walk. Replace the walk `sx +=` / `check_collision` / snap block with:

```toy
        var dx = e.facing_left ? -6 : 6
        var hit = level_move_box_x(e.sx, e.box, dx)

        var face_x = e.facing_left ? e.box.x : e.box.x + e.box.w - 1
        var foot_y = e.box.y + e.box.h
        var turn = -1
        if hit or not check_collision_down(face_x, foot_y, foot_y - 1):
            turn = 1
        else:
            e.run_tick += 1
            e.frame = 1 + e.run_tick / 4 % 4
            e.pace_countdown -= 1
            if e.pace_countdown == 0: turn = rand(0, 4)
        if turn >= 0:
            e.wait_countdown = rand(50, 90)
            e.wait_turn = turn > 0
            e.frame = 0
```

Remove the trailing `e.box.x = e.sx / SP` sync in the walk path — `level_move_box_x` already wrote `box.x`. Still sync both axes once at end of `enemy_update_patrol` if idle branches never move (harmless):

```toy
    e.box.x = e.sx / SP
    e.box.y = e.sy / SP
```

Or only sync when not walking; simplest: keep one sync at end for y (x already correct after walk).

- [ ] **Step 2: Compile + playtest patrols**

Run: `./run.sh game-outpost-oberon`  
Expected: patrols pace, stop/turn at walls and ledges, muzzle/LOS still work.

- [ ] **Step 3: Commit (optional)**

```bash
git add game-outpost-oberon/enemy.toy game-outpost-oberon/code
git commit -m "Resolve patrol walking with level_move_box_x."
```

---

### Task 4: Spider uses `level_move_box_*`

**Files:**
- Modify: `game-outpost-oberon/enemy.toy` (`enemy_update_spider`)

**Interfaces:**
- Consumes: `level_move_box_x`, `level_move_box_y`
- Produces: spider physics via helpers; AI wall/ledge probes unchanged (`e.box`)

- [ ] **Step 1: Replace x/y collision block**

After AI sets `e.vx` / jump `e.vy`, replace the hand-rolled collision with:

```toy
    var hit_x = level_move_box_x(e.sx, e.box, e.vx)
    if hit_x:
        if e.on_ground:
            e.facing_left = e.vx > 0
            e.vx = 0
            e.no_turn_countdown = rand(30, 50)
        else:
            e.vx = -e.vx
            e.wait_turn = true

    e.vy += GRAVITY
    e.vy = min(e.vy, SP * 4)
    var was_air = not e.on_ground
    var hit_y = level_move_box_y(e.sy, e.box, e.vy)
    if hit_y == DIR_DOWN:
        e.on_ground = true
        e.vy = 0
        if was_air and e.wait_turn:
            e.wait_countdown = rand(10, 40)
            e.wait_turn = false
    elif hit_y == DIR_UP:
        e.vy = 8
        e.on_ground = false
    else:
        e.on_ground = false

    if e.wait_countdown == 0:
        e.run_tick += 1
        e.frame = 1 + e.run_tick / 4 % 4
```

Delete the old probe block and the final `e.box.x/y = …` sync (helpers own both axes when both are called each frame).

- [ ] **Step 2: Compile + playtest spiders**

Run: `./run.sh game-outpost-oberon`  
Expected:
1. Spiders land on floors and `^` jump-throughs.
2. Wall bump while grounded turns / pauses; mid-air bump reverses `vx`.
3. Ceiling bump sets `vy = 8`.
4. Jump AI (wall/ledge/attack) still fires; sprite stays glued to box.

- [ ] **Step 3: Commit (optional)**

```bash
git add game-outpost-oberon/enemy.toy game-outpost-oberon/code
git commit -m "Resolve spider movement with level_move_box_x/y."
```

---

### Task 5: Final sweep

**Files:**
- Modify: `game-outpost-oberon/enemy.toy` / `game.toy` only if greps find leftovers

- [ ] **Step 1: Grep for old patterns**

```bash
rg -n "entity_update_box_pos|sx / SP \\+|sy / SP - 16" game-outpost-oberon/
```

Expected: no `entity_update_box_pos`; no old patrol draw `sy/SP - 16` unless intentionally kept (should be gone).

- [ ] **Step 2: Full playtest checklist**

Run: `./run.sh game-outpost-oberon`  
Checklist:
- [ ] Hero move / jump / land / shoot
- [ ] Hero death / respawn / camera return
- [ ] Patrol walk / shoot / die
- [ ] Spider sleep / chase / jump / die
- [ ] Lava kill still works (tile sample at `box.x`, `box.y`)
- [ ] Debug box overlay matches feet on tiles

- [ ] **Step 3: Commit (optional)**

```bash
git add game-outpost-oberon/game.toy game-outpost-oberon/enemy.toy game-outpost-oberon/level.toy game-outpost-oberon/code
git commit -m "Finish entity coord unify and enemy move-box helpers."
```

---

## Spec coverage

| Spec item | Task |
|-----------|------|
| `sx`/`sy` = subpixel box top-left | 1 |
| Delete `entity_update_box_pos` | 1 |
| Draw via `box` + inline offsets | 1 |
| Camera follows box center | 1 |
| Hero keeps multi-sensor collision | 1 (never calls helpers) |
| `level_move_box_x/y` same interface, mutate axis + box | 2 |
| Return `0` / axis `DIR_*` only | 2 |
| Jump-through via `dy` in `level_move_box_y` | 2, 4 |
| Patrol/spider use helpers | 3, 4 |
| Ledge/AI probes stay separate | 3, 4 |
