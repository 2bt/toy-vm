# Sound interrupt (replace memory-mapped voices)

## Goal

Stop memory-mapping APU voices into VM RAM. Games set a voice through a `sound` interrupt that assigns pitch, pulse width, volume, and wave in one call. Host keeps a shadow voice buffer and still feeds the APU once per frame via `next_frame`.

## Interrupt API

| # | Name | Notes |
|---|------|--------|
| 1 | `print` | unchanged |
| 2 | `sprite` | unchanged |
| 3 | `sound` | **new** (inserted below sprite) |
| 4 | `rand` | was 3 |
| 5 | `sqrt` | was 4 |

Prelude wrapper:

```
func sound(i @ 2, pitch @ 3, pw @ 4, vol @ 5, wave @ 6):
    asm "    int #3"
```

One call sets all four voice parameters for voice index `i`.

## Memory map

Remove `voices` from `MemMap`. Layout:

- `ret` @ 0
- `input` @ 1
- interrupt argument union starting @ 2 (no gap)

All interrupt helpers move arg bases from `@ 18` to `@ 2` (including `print`, `sprite`, `rand`, `sqrt`).

Union arm for sound: `{ i, pitch, pw, vol, wave }`.

## Host / APU

- Host-owned `std::array<APU::VoiceInput, 4> voices{}` (not in VM memory).
- On `INT_SOUND`: if `i` in `0..3`, copy pitch/pw/vol/wave into `voices[i]`; ignore out-of-range `i`.
- Each frame: `apu.next_frame(voices)` after `vm.run` (unchanged timing).
- APU mix/config logic unchanged.

## Breakout game

**prelude.toy**

- Platform constants (`SCREEN_*`, `INPUT_*`), `var input @ 1`, interrupt wrappers only.
- No `struct Voice`, no `voices`, no `WAVE_*`.

**sound.toy**

- Owns `WAVE_*`, `struct Voice`, `var voices: Voice[4]` (ordinary RAM, not mapped).
- Rename effect starter `sound` → `play_sound`; update `main.toy` call sites.
- Command streams keep shadow `Voice*` state plus a voice index.
- Once per frame, after finishing that stream’s command processing for the frame, push with `sound(voice_index, pitch, pw, vol, wave)` so the host snapshot matches the shadow voice (same effective timing as today’s end-of-frame `next_frame`).

## Tests

- `test/test_exec.py` embedded prelude: interrupt arg addresses `@ 18` → `@ 2`.
- Update `rand`/`sqrt` interrupt numbers if those wrappers appear in tests later.

## Out of scope

- APU synthesis / mix changes
- Fixing prelude-vs-APU wave enum numbering mismatch (values stay as games use them today)
- Per-field sound setters
