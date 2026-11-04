# Jelly enemy design

Air threat for Outpost Oberon: a bobbing blob-jelly that fills the gap left by ground patrols, melee buggers, and wall cannons.

## Role

| Property | Value |
|---|---|
| Type name | `ENEMY_TYPE_JELLY` |
| Level tile | `j` |
| HP | 1 (glass cannon) |
| Score | 200 |
| Contact damage | Yes (existing hero↔enemy box check) |
| Facing / flip | None — sprite is symmetric |

## Look

- Single-color silhouette in **bugger purple** `#40318d` / `(64, 49, 141)`; white hit-flash on a second sprite row (same pattern as other enemies).
- **Blob body** with two long stub arms.
- Arm animation: **rubber** motion — arms stay long; base leads and tips lag/overshoot (~5–8 frames). Arms are decorative only.
- Hitbox ≈ body only (~10×10), not the full arm reach.
- Sprite size: 16×16 frames on `sprite.png` (new row; leave room for arm frames + flash row).

## Motion

- Fixed X at spawn (tile-centered like other enemies).
- Smooth vertical bob around spawn Y. Amplitude ~8–16 px so the body crosses the hero’s horizontal shot line.
- No `sin` in the VM API today — implement bob with a **triangle wave** (or small lookup) driven by a phase counter so the motion reads as smooth sine without new syscalls. Prefer triangle unless a tiny LUT is clearer in-code.
- Store home Y at spawn (reuse an unused `Enemy` field such as keeping spawn `sy` separately, or derive phase-only offset from a stored home). Do not accumulate drift.

## Combat (cannon-lite)

- Aimed singles only: reuse `ENEMY_BULLET_TYPE_AIMED`. **No burst.**
- Requires hero normal + jelly on-camera + LOS from jelly center to hero.
- Short LOS latch (~15–20 frames, lighter than cannon’s 25).
- On latch complete: aim window of ~25 frames — spawn/follow charge at start, fire **one** aimed bullet near the end (~frame 5 of the window, mirroring patrol’s late muzzle), then cooldown ~120 frames. Cancel aim and clear charge if LOS drops mid-window.
- Bullet aim uses same normalize-by-`sqrt` approach as cannon.
- No `facing_left` updates for drawing; optional face tracking is unnecessary.

### Charge particle follow

Charge particles are fixed-position today (`PARTICLE_TYPE_CHARGE`, `ttl = 10`). While aiming:

1. Spawn charge at jelly center; keep a `Particle*` (or particle index) on the jelly.
2. Each jelly update, if the pointer is still live, rewrite `sx`/`sy` to the current center so the charge rides the bob.
3. If the aim window is longer than 10 frames, bump `ttl` (or respawn) so the telegraph doesn’t die mid-aim.
4. Clear the pointer when the shot fires or aim cancels (LOS lost / inactive).

## Wiring

- `enemy_init_all`: tile `j` → `enemy_spawn(ENEMY_TYPE_JELLY, …)`.
- `enemy_update_all` / `enemy_draw_all` / dying score branch: add jelly cases.
- Prefer reusing existing `Enemy` fields (`aim_countdown`, `shoot_countdown`, `pace_countdown` for latch, `frame` / `run_tick` for arms + bob phase) plus one particle pointer/index for charge follow.
- New SFX only if existing `SFX_CANNON_SHOOT` / `SFX_CHARGE` feel wrong in playtest; default to reusing those.

## Placement note

Place `j` in shafts/corridors where the bob band overlaps jump/shot height. Avoid putting it permanently above the highest platform the hero can shoot from.

## Out of scope

- Horizontal air patrol / diving dropper
- Burst fire
- New bullet types
- Upward hero shooting
- Facing / flip variants
