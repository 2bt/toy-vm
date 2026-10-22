# Outpost Oberon: entity coords + `level_move_box_*`

## Goal

Make entity position easy to reason about: `sx`/`sy` are the subpixel top-left of the AABB. Then simplify enemy tile collision with shared `level_move_box_x` / `level_move_box_y` helpers. Hero keeps its multi-sensor collision.

## Success criteria

- No `entity_update_box_pos`; no “sprite origin → box” conversion.
- `box.x` / `box.y` always mean the same thing as `sx / SP` / `sy / SP`.
- Drawing uses `box.x`/`box.y` plus hardcoded offsets inside each sprite call.
- Enemy horizontal/vertical resolution goes through the move helpers; enemy update no longer probes left and right (or floor/ceiling) by hand for resolution.
- Hero feel (multi-probe collision) unchanged in intent; probes rewritten for the new origin.
- Camera framing stays sensible (follow box center after the origin shift).

## Non-goals

- Putting hero on `level_move_box_*` (enemies only).
- Dropping subpixel `sx`/`sy` in favor of pixel-only boxes.
- Named sprite-offset constants or `spr_ox` fields — offsets stay inline in draw calls.
- Reworking bullet / particle / camera internals beyond what the new hero origin requires.
- A general physics engine or Entity base type.

## Phase 1 — Coordinate unify

### Model

| Field | Meaning |
|-------|---------|
| `sx`, `sy` | Subpixel top-left of the collision box |
| `box.x`, `box.y` | Pixel top-left (`sx / SP`, `sy / SP`) |
| `box.w`, `box.h` | Size in pixels (unchanged) |

Delete `entity_update_box_pos` from `level.toy`.

### Hero

- After custom collision each frame: `hero.box.x = hero.sx / SP`, `hero.box.y = hero.sy / SP`.
- Rewrite collision probes relative to top-left (current magic numbers assumed the old sprite origin: +8 horizontal center of a 16px cell, +16 toward feet).
- Spawn: set `sx`/`sy` so the box sits correctly on the spawn tile (not the old origin convention).
- Draw: `camera_sprite(hero.box.x ± ox, hero.box.y ± oy, …)` with hardcoded offsets.
- Camera: follow box center (or equivalent) so the view does not jump when the origin meaning changes.

### Enemies (phase 1 only)

- Same spawn/draw/box sync changes as hero so the game still runs before helpers land.
- Temporary: keep existing resolution logic but relative to the new origin; replace with helpers in phase 2.

### Draw offsets

Per call site only (hero, patrol, spider). No shared offset table. Values chosen so sprites match current on-screen placement relative to the hitbox.

## Phase 2 — `level_move_box_x` / `level_move_box_y`

### Interface

Same shape for both axes (exact pointer/`*` syntax follows what Toy needs so `sx`/`sy` can be mutated):

```text
level_move_box_x(sx, box, dx) → 0 | DIR_LEFT | DIR_RIGHT
level_move_box_y(sy, box, dy) → 0 | DIR_UP   | DIR_DOWN
```

- Apply `dx`/`dy` to the subpixel coordinate.
- Resolve against tiles using the box edges (simple leading-side probes; no hero-style dual-sensor care).
- Write back subpixel pos **and** the matching `box` axis (`box.x` or `box.y`). Callers do not sync that axis by hand.
- `level_move_box_y` handles jump-through platforms (`^`) using `dy` (same role as today’s `py` in `check_collision_down`: only land when moving downward across the platform).

### Return contract

| Helper | Returns |
|--------|---------|
| `level_move_box_x` | `0`, `DIR_LEFT`, or `DIR_RIGHT` |
| `level_move_box_y` | `0`, `DIR_UP`, or `DIR_DOWN` |

AI uses the return value to react (spider bounce/turn on wall; landing via `DIR_DOWN`). Ledge / look-ahead checks for patrol and spider jump decisions stay separate probes; helpers only resolve the move.

### Enemy call pattern

1. Compute `vx` / `vy` (AI, gravity, accel).
2. `hit_x = level_move_box_x(e.sx, e.box, e.vx)` — updates `e.sx` and `e.box.x`.
3. `hit_y = level_move_box_y(e.sy, e.box, e.vy)` — updates `e.sy` and `e.box.y`.
4. React to `hit_x` / `hit_y`.

Patrol walking uses the same helper with `dx` of ±6 (or via `vx`).

### Hero

Does **not** call these helpers. Keeps multi-sensor collision from phase 1.

## Out of scope / later

- Migrating hero onto `level_move_box_*` if the helpers prove good enough.
- Sharing probe helpers between hero and move-box internals.

## Risks

- Wrong draw offsets → sprites float or sink relative to hitboxes; verify visually per entity type.
- Spawn tile placement off by the old +8/+16 convention → entities stuck in floor/ceiling at level start.
- Jump-through mishandled in `level_move_box_y` → spiders/patrols fall through or snag on `^`.
- Camera still keyed off old origin → framing drift until center follow is updated.
