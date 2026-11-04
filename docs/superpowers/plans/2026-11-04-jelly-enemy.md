# Jelly Enemy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `ENEMY_TYPE_JELLY` — a 1 HP bobbing blob-jelly that fires single aimed shots (cannon-lite) and occupies air space in Outpost Oberon.

**Architecture:** Extend the existing enemy pipeline in `enemy.toy` (spawn / update / draw / score) with a jelly branch. Reuse `ENEMY_BULLET_TYPE_AIMED` and `PARTICLE_TYPE_CHARGE`. Bob with a triangle-wave Y offset around a stored `home_sy` (no `sin` syscall). Keep a `Particle*` on the enemy and rewrite charge position each frame while aiming. Sprites: new 16×16 purple frames (+ white flash row) on `sprite.png`.

**Tech Stack:** Toy language (`.toy`), `compile.py` / `assemble.py`, SDL host via `./run.sh game-outpost-oberon`, `sprite.png` edited with Python/Pillow.

**Spec:** `docs/superpowers/specs/2026-11-04-jelly-enemy-design.md`

## Global Constraints

- Type name: `ENEMY_TYPE_JELLY`; level tile: `j`
- HP 1; score 200; no facing/flip; body hitbox only (~10×10)
- Aimed singles only (no burst); cannon-lite latch ~15–20; aim ~25 frames; cooldown ~120
- Charge particle must follow bob while aiming
- Reuse `SFX_CHARGE` / `SFX_CANNON_SHOOT` unless playtest says otherwise
- No new bullet types; no upward hero shooting

## File map

| File | Responsibility |
|---|---|
| `game-outpost-oberon/sprite.png` | Jelly frames (normal + hit flash) |
| `game-outpost-oberon/enemy.toy` | Type, spawn, update, draw, score, charge follow |
| `game-outpost-oberon/level.toy` | At least one test `j` placement |
| `game-outpost-oberon/particle.toy` | No API change required (spawn already returns `Particle*`) |

---

### Task 1: Jelly sprites on the sheet

**Files:**
- Modify: `game-outpost-oberon/sprite.png`
- (Optional helper, delete after use): `/tmp/paint_jelly_sprites.py`

**Interfaces:**
- Produces: 5 rubber-arm frames at `(256 + frame*16, 144)` size 16×16, color `(64, 49, 141, 255)`; identical shapes in white `(232, 232, 232, 255)` at y `160` for hit flash
- Consumes: nothing from later tasks

- [ ] **Step 1: Paint frames into `sprite.png`**

Write and run a one-shot script (Pillow). Body is fixed; arms use the rubber lag/overshoot offsets from the brainstorm companion:

```python
#!/usr/bin/env python3
"""Paint ENEMY_TYPE_JELLY frames into game-outpost-oberon/sprite.png"""
from PIL import Image
import math

PATH = "game-outpost-oberon/sprite.png"
PURPLE = (64, 49, 141, 255)
WHITE = (232, 232, 232, 255)
BASE_X, BASE_Y, FLASH_Y = 256, 144, 160
N = 5

BODY = [
    "................",
    ".....######.....",
    "....########....",
    "...##########...",
    "...##########...",
    "....########....",
]

def rubber_arms(t):
    """t in [0,1). Returns 5 rows of arm pixels (length-5 tentacles)."""
    L0, R0 = 4, 11
    rows = []
    for i in range(5):
        lag = i / 4
        delayed = math.sin((t - lag * 0.35) * math.pi * 2) * 2
        overshoot = math.sin((t - lag * 0.35) * math.pi * 2) * lag * 1.2
        off = delayed + overshoot
        xl = max(1, min(6, round(L0 + off)))
        xr = max(9, min(14, round(R0 - off)))
        row = ["."] * 16
        row[xl] = "#"
        row[xr] = "#"
        rows.append("".join(row))
    return rows

def paint_frame(img, ox, oy, color, t):
    rows = BODY + rubber_arms(t)
    while len(rows) < 16:
        rows.append("................")
    for y, row in enumerate(rows[:16]):
        for x, ch in enumerate(row):
            if ch == "#":
                img.putpixel((ox + x, oy + y), color)

def main():
    img = Image.open(PATH).convert("RGBA")
    # clear destination slots
    for f in range(N):
        for y in range(16):
            for x in range(16):
                img.putpixel((BASE_X + f * 16 + x, BASE_Y + y), (0, 0, 0, 0))
                img.putpixel((BASE_X + f * 16 + x, FLASH_Y + y), (0, 0, 0, 0))
    for f in range(N):
        t = f / N
        paint_frame(img, BASE_X + f * 16, BASE_Y, PURPLE, t)
        paint_frame(img, BASE_X + f * 16, FLASH_Y, WHITE, t)
    img.save(PATH)
    print(f"wrote {N} jelly frames at y={BASE_Y}/{FLASH_Y}")

if __name__ == "__main__":
    main()
```

Run from repo root:

```bash
python3 /tmp/paint_jelly_sprites.py
```

Expected: `wrote 5 jelly frames at y=144/160`

- [ ] **Step 2: Visually confirm**

Open `game-outpost-oberon/sprite.png` and check purple blob+arms at the right edge under the cannon row, plus white duplicates one row down. No overlap with cannon (`y=112`) or bugger (`y=80`).

- [ ] **Step 3: Commit**

```bash
git add game-outpost-oberon/sprite.png
git commit -m "$(cat <<'EOF'
Add jelly enemy sprite frames (purple + hit flash).

EOF
)"
```

---

### Task 2: Spawn, draw, bob (no combat yet)

**Files:**
- Modify: `game-outpost-oberon/enemy.toy`
- Modify: `game-outpost-oberon/level.toy` (one test `j`)

**Interfaces:**
- Consumes: sprite frames from Task 1 at `(256+frame*16, 144)` / flash `160`
- Produces: `ENEMY_TYPE_JELLY = 4`; spawn from tile `j`; `enemy_jelly_update` bob + arm anim; draw path; score `200` on death

- [ ] **Step 1: Add type constant and Enemy fields**

Near the top of `enemy.toy`, after the other type constants:

```
const ENEMY_TYPE_JELLY   = 4
```

In `struct Enemy`, add (after existing fields is fine):

```
    home_sy
    charge: Particle*
```

- [ ] **Step 2: Spawn jelly from tile `j`**

In `enemy_init_all`:

```
elif t == 'j': enemy_spawn(ENEMY_TYPE_JELLY, x, y)
```

In `enemy_spawn`, add branch:

```
elif type == ENEMY_TYPE_JELLY:
    e.box.w = 10
    e.box.h = 10
    e.hp    = 1
    e.shoot_countdown = rand(40, 100)
    e.charge = null
```

After the shared `e.sx` / `e.sy` placement block (the non-cannon default), if type is jelly, set home and optional vertical centering tweak:

```
if type == ENEMY_TYPE_JELLY:
    e.home_sy = e.sy
    e.charge  = null
```

(Keep the default tile centering; bob will move Y.)

- [ ] **Step 3: Wire update / draw / score stubs**

In `enemy_update_all` dying score:

```
e.type == ENEMY_TYPE_JELLY  ? 200 :
```

(insert in the existing ternary chain before the final `0`)

In the normal update dispatch:

```
elif e.type == ENEMY_TYPE_JELLY: enemy_jelly_update(e)
```

In `enemy_draw_all`:

```
elif e.type == ENEMY_TYPE_JELLY:
    camera_sprite(e.box.x - 3, e.box.y - 3, 16, 16,
                  256 + e.frame * 16,
                  144 + (e.hit_countdown > 0) * 16,
                  0)
```

(No flip — third arg `0`.)

- [ ] **Step 4: Implement bob + arm anim only**

Append to `enemy.toy`:

```
# ---------------------------------------------------------------------------
# jelly
# ---------------------------------------------------------------------------
const JELLY_BOB_AMP   = 12
const JELLY_BOB_HALF  = 40
const JELLY_ARM_FRAMES = 5
const JELLY_ARM_RATE   = 4

func enemy_jelly_clear_charge(e: Enemy*):
    e.charge = null

func enemy_jelly_update(e: Enemy*):
    # triangle-wave bob around home_sy
    e.run_tick += 1
    var p = e.run_tick % (JELLY_BOB_HALF * 2)
    var yoff = 0
    if p < JELLY_BOB_HALF:
        yoff = p * JELLY_BOB_AMP * 2 / JELLY_BOB_HALF - JELLY_BOB_AMP
    else:
        yoff = (JELLY_BOB_HALF * 2 - p) * JELLY_BOB_AMP * 2 / JELLY_BOB_HALF - JELLY_BOB_AMP
    e.sy = e.home_sy + yoff * SP
    e.box.x = e.sx / SP
    e.box.y = e.sy / SP

    # rubber arm frames (decorative)
    e.frame = e.run_tick / JELLY_ARM_RATE % JELLY_ARM_FRAMES
```

Combat hooks come in Task 3 — do not fire yet.

- [ ] **Step 5: Place a test `j` in LEVEL2**

In `LEVEL2_DATA`, in the open shaft near spawn (air cell the hero can shoot through horizontally). Example: on the row under the `SSS` platform row, replace one space a few tiles to the right of spawn with `j` (keep string length identical — every row must stay the same width).

Concrete edit — find this LEVEL2 row (the mostly-empty corridor under spawn):

```
"00000000000000000000000000                        00000000000000000000000000000000000000000000000000000000"
```

Change one interior space to `j` (column ~40-ish, still in the open gap), preserving total character count of the row.

- [ ] **Step 6: Compile**

```bash
./compile.py game-outpost-oberon/main.toy && ./assemble.py game-outpost-oberon/main.asm
```

Expected: exit 0, no type errors about `Particle*` / `home_sy`.

- [ ] **Step 7: Playtest bob**

```bash
./run.sh game-outpost-oberon
```

Reach LEVEL2 (or temporarily point game start at level 2 if faster). Confirm jelly bobs vertically, arms animate, no flip, contact still kills, one hero bullet kills it for 200 points.

- [ ] **Step 8: Commit**

```bash
git add game-outpost-oberon/enemy.toy game-outpost-oberon/level.toy
git commit -m "$(cat <<'EOF'
Spawn and bob jelly enemy from tile j.

EOF
)"
```

---

### Task 3: Cannon-lite aimed shot + charge follow

**Files:**
- Modify: `game-outpost-oberon/enemy.toy` (`enemy_jelly_update`, helpers)

**Interfaces:**
- Consumes: `enemy_bullet_spawn(ENEMY_BULLET_TYPE_AIMED, …)`, `particle_spawn(PARTICLE_TYPE_CHARGE, …)`, `level_has_los`, `SFX_CHARGE`, `SFX_CANNON_SHOOT`
- Produces: full jelly combat matching the spec

- [ ] **Step 1: Add combat constants**

Next to the other jelly consts:

```
const JELLY_LOS_LATCH     = 18
const JELLY_AIM_FRAMES    = 25
const JELLY_SHOT_AT       = 5
const JELLY_COOLDOWN      = 120
const JELLY_BULLET_SPEED  = 18
```

- [ ] **Step 2: Replace `enemy_jelly_update` with full behavior**

Keep bob + arm anim first, then combat. Full function:

```
func enemy_jelly_update(e: Enemy*):
    # --- motion (same as Task 2) ---
    e.run_tick += 1
    var p = e.run_tick % (JELLY_BOB_HALF * 2)
    var yoff = 0
    if p < JELLY_BOB_HALF:
        yoff = p * JELLY_BOB_AMP * 2 / JELLY_BOB_HALF - JELLY_BOB_AMP
    else:
        yoff = (JELLY_BOB_HALF * 2 - p) * JELLY_BOB_AMP * 2 / JELLY_BOB_HALF - JELLY_BOB_AMP
    e.sy = e.home_sy + yoff * SP
    e.box.x = e.sx / SP
    e.box.y = e.sy / SP
    e.frame = e.run_tick / JELLY_ARM_RATE % JELLY_ARM_FRAMES

    var cx = box_center_x(e.box)
    var cy = box_center_y(e.box)
    var hx = box_center_x(hero.box)
    var hy = box_center_y(hero.box)
    var loc = (hero.state == HERO_STATE_NORMAL
        and box_encloses(camera.box, e.box)
        and level_has_los(cx, cy, hx, hy))

    # keep charge glued to body while aiming
    if e.charge:
        if e.charge.ttl <= 0 or e.charge.type != PARTICLE_TYPE_CHARGE:
            e.charge = null
        else:
            e.charge.sx = cx * SP
            e.charge.sy = cy * SP
            if e.charge.ttl < 3: e.charge.ttl = 3

    if e.aim_countdown > 0:
        if not loc:
            enemy_jelly_clear_charge(e)
            e.aim_countdown = 0
            e.pace_countdown = 0
            e.shoot_countdown = 30
            return

        if e.aim_countdown == JELLY_AIM_FRAMES:
            e.charge = particle_spawn(PARTICLE_TYPE_CHARGE, cx, cy)
            sfx_play(SFX_CHARGE)

        if e.aim_countdown == JELLY_SHOT_AT:
            var dx = hx - cx
            var dy = hy - cy
            var len = sqrt(dx * dx + dy * dy)
            if len < 1: len = 1
            var b = enemy_bullet_spawn(ENEMY_BULLET_TYPE_AIMED, cx - 1, cy - 1)
            if b:
                b.vx = dx * JELLY_BULLET_SPEED / len
                b.vy = dy * JELLY_BULLET_SPEED / len
                sfx_play(SFX_CANNON_SHOOT)
            enemy_jelly_clear_charge(e)

        e.aim_countdown -= 1
        if e.aim_countdown == 0:
            e.shoot_countdown = JELLY_COOLDOWN
            e.pace_countdown = 0

    elif e.shoot_countdown > 0:
        e.shoot_countdown -= 1
        e.pace_countdown = 0

    elif loc:
        e.pace_countdown += 1
        if e.pace_countdown >= JELLY_LOS_LATCH:
            e.aim_countdown = JELLY_AIM_FRAMES
            e.pace_countdown = 0
    else:
        e.pace_countdown = 0
```

- [ ] **Step 3: Clear charge on death**

In `enemy_kill`, after setting dying state:

```
if e.type == ENEMY_TYPE_JELLY: enemy_jelly_clear_charge(e)
```

- [ ] **Step 4: Compile**

```bash
./compile.py game-outpost-oberon/main.toy && ./assemble.py game-outpost-oberon/main.asm
```

Expected: exit 0.

- [ ] **Step 5: Playtest combat**

```bash
./run.sh game-outpost-oberon
```

Checklist:

1. Hold LOS → short pause → charge appears on jelly body and **moves with the bob**
2. One aimed bullet fires (not a burst); then ~2s quiet
3. Break LOS mid-charge → charge stops / aim cancels
4. Still 1 HP; score 200

- [ ] **Step 6: Commit**

```bash
git add game-outpost-oberon/enemy.toy
git commit -m "$(cat <<'EOF'
Add jelly cannon-lite aimed fire with following charge.

EOF
)"
```

---

### Task 4: Spec commit + smoke pass

**Files:**
- Add (if not already committed): `docs/superpowers/specs/2026-11-04-jelly-enemy-design.md`
- Add: `docs/superpowers/plans/2026-11-04-jelly-enemy.md`

- [ ] **Step 1: Commit docs**

```bash
git add docs/superpowers/specs/2026-11-04-jelly-enemy-design.md docs/superpowers/plans/2026-11-04-jelly-enemy.md
git commit -m "$(cat <<'EOF'
Document jelly enemy design and implementation plan.

EOF
)"
```

- [ ] **Step 2: Final smoke**

```bash
./compile.py game-outpost-oberon/main.toy && ./assemble.py game-outpost-oberon/main.asm && ./run.sh game-outpost-oberon
```

Confirm jelly still present, bobs, shoots once per cycle, dies in one hit.

---

## Spec coverage (self-review)

| Spec item | Task |
|---|---|
| `ENEMY_TYPE_JELLY`, tile `j`, HP 1, score 200 | 2 |
| Blob purple + rubber arms + flash row | 1 |
| Triangle bob, fixed X, `home_sy` | 2 |
| No facing | 2 (draw flip 0) |
| Body hitbox ~10×10 | 2 |
| Cannon-lite latch / single aimed / cooldown | 3 |
| Charge particle follow + ttl bump | 3 |
| Reuse aimed bullet + charge SFX | 3 |
| Clear charge on cancel/kill | 3 |
| Test placement | 2 |
| Out of scope (dropper, burst, new bullets) | honored |

No placeholders left in steps. Field names (`home_sy`, `charge`) consistent across tasks.
