
    ; interrupts
    INT_SPRITE = 0


    ; special memory locations
    BTN         = 0

    SPRITE_X    = 1
    SPRITE_Y    = 2
    SPRITE_W    = 3
    SPRITE_H    = 4
    SPRITE_S    = 5
    SPRITE_T    = 6


    ; button map
    BTN_UP    = 1
    BTN_DOWN  = 2
    BTN_LEFT  = 4
    BTN_RIGHT = 8
    BTN_A     = 16
    BTN_B     = 32


.data

sx:     0
sy:     0



.code
        jmp init
        jmp update


; initialize game
init:

        hlt



; update game
update:

        tst BTN, #BTN_UP
        jeq not_up
        sub sy, #1
not_up:
        tst BTN, #BTN_DOWN
        jeq not_down
        add sy, #1
not_down:
        tst BTN, #BTN_LEFT
        jeq not_left
        sub sx, #1
not_left:
        tst BTN, #BTN_RIGHT
        jeq not_right
        add sx, #1
not_right:


        ; circle
        mov SPRITE_S, #0
        mov SPRITE_T, #0
        mov SPRITE_W, #32
        mov SPRITE_H, #32
        mov SPRITE_X, #100
        mov SPRITE_Y, #100
        int #INT_SPRITE

        ; cursor
        mov SPRITE_S, #32
        mov SPRITE_T, #0
        mov SPRITE_W, #8
        mov SPRITE_H, #8
        mov SPRITE_X, sx
        mov SPRITE_Y, sy
        int #INT_SPRITE


        hlt