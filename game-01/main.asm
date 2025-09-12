_R = 30
_T0 = 31
_T1 = 32
btn = 0
_sprite_x = 1
_sprite_y = 2
_sprite_w = 3
_sprite_h = 4
_sprite_s = 5
_sprite_t = 6
_sprite_f = 7
_rand_lo = 8
_rand_hi = 9
i = 33
ball_x = 34
ball_y = 1034
ball_vx = 2034
ball_vy = 3034
x = 4034
y = 4035
_update_ball_n = 4036

    ; function preamble
preamble:
    jmp init
    jmp update
    ret

    ; function sprite
sprite:
    int #0
    ret

    ; function rand
rand:
    int #1
    mov _R, 10
    ret

    ; function init
init:
    mov i, #0
_while_1:
    cmp i, #1000
    jge _end_while_3
_while_true_2:
    mov _T0, i
    add _T0, #ball_x
    mov _rand_lo, #0
    mov _rand_hi, #319
    jsr rand
    mov _T1, _R
    mul _T1, #256
    mov [_T0], _T1
    mov _T0, i
    add _T0, #ball_y
    mov _rand_lo, #0
    mov _rand_hi, #179
    jsr rand
    mov _T1, _R
    mul _T1, #256
    mov [_T0], _T1
    mov _T0, i
    add _T0, #ball_vx
    mov _rand_lo, #-100
    mov _rand_hi, #100
    jsr rand
    mov [_T0], _R
    mov _T0, i
    add _T0, #ball_vy
    mov _rand_lo, #-100
    mov _rand_hi, #100
    jsr rand
    mov [_T0], _R
    add i, #1
    jmp _while_1
_end_while_3:
    mov x, #100
    mov y, #100
    ret

    ; function update_ball
update_ball:
    mov _T0, _update_ball_n
    add _T0, #ball_x
    mov _T1, _update_ball_n
    add _T1, #ball_vx
    add [_T0], [_T1]
    mov _T0, _update_ball_n
    add _T0, #ball_y
    mov _T1, _update_ball_n
    add _T1, #ball_vy
    add [_T0], [_T1]
    mov _T0, _update_ball_n
    add _T0, #ball_x
    cmp [_T0], #0
    jlt _if_true_4
_or_next_7:
    mov _T1, _update_ball_n
    add _T1, #ball_x
    cmp [_T1], #81920
    jlt _if_false_5
_if_true_4:
    mov _T0, _update_ball_n
    add _T0, #ball_vx
    mul [_T0], #-1
_if_false_5:
    mov _T0, _update_ball_n
    add _T0, #ball_y
    cmp [_T0], #0
    jlt _if_true_8
_or_next_11:
    mov _T1, _update_ball_n
    add _T1, #ball_y
    cmp [_T1], #46080
    jlt _if_false_9
_if_true_8:
    mov _T0, _update_ball_n
    add _T0, #ball_vy
    mul [_T0], #-1
_if_false_9:
    mov _T0, _update_ball_n
    add _T0, #ball_x
    mov _T0, [_T0]
    div _T0, #256
    mov _sprite_x, _T0
    mov _T1, _update_ball_n
    add _T1, #ball_y
    mov _T1, [_T1]
    div _T1, #256
    mov _sprite_y, _T1
    mov _sprite_w, #1
    mov _sprite_h, #1
    mov _sprite_s, #48
    mov _sprite_t, #0
    mov _sprite_f, #0
    jsr sprite
    ret

    ; function update
update:
    mov i, #0
_while_12:
    cmp i, #1000
    jge _end_while_14
_while_true_13:
    mov _update_ball_n, i
    jsr update_ball
    add i, #1
    jmp _while_12
_end_while_14:
    tst btn, #4
    jeq _if_false_16
_if_true_15:
    sub x, #2
_if_false_16:
    tst btn, #8
    jeq _if_false_19
_if_true_18:
    add x, #2
_if_false_19:
    tst btn, #1
    jeq _if_false_22
_if_true_21:
    sub y, #2
_if_false_22:
    tst btn, #2
    jeq _if_false_25
_if_true_24:
    add y, #2
_if_false_25:
    mov _sprite_x, x
    mov _sprite_y, y
    mov _sprite_w, #32
    mov _sprite_h, #32
    mov _sprite_s, #0
    mov _sprite_t, #0
    mov _sprite_f, #0
    jsr sprite
    ret
