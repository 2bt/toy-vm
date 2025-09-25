#include <cstdint>
#include <string>
#include <random>
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include "vm.hpp"



enum {
    // io memory locations
    IO_BTN,

    IO_SPRITE_X,
    IO_SPRITE_Y,
    IO_SPRITE_W,
    IO_SPRITE_H,
    IO_SPRITE_S,
    IO_SPRITE_T,
    IO_SPRITE_FLAGS,

    IO_RAND_LO,
    IO_RAND_HI,
    IO_RAND_RESULT,

    // interrupts
    INT_SPRITE = 0,
    INT_RAND,

    // button bits
    BTN_UP = 0,
    BTN_DOWN,
    BTN_LEFT,
    BTN_RIGHT,
    BTN_A,
    BTN_B,

    // screen
    SCREEN_W = 320,
    SCREEN_H = 180,
};


SDL_Texture* load_texture(SDL_Renderer* renderer, std::string file) {
    SDL_Surface* surf = IMG_Load(file.c_str());
    if (!surf) {
        printf("ERROR: cannot load '%s'\n", file.c_str());
        return nullptr;
    }
    SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, surf);
    SDL_FreeSurface(surf);
    if (!tex) {
        printf("ERROR: SDL_CreateTextureFromSurface: %s\n", SDL_GetError());
        return nullptr;
    }
    return tex;
}


VM            vm;
SDL_Renderer* renderer;
SDL_Texture*  sprite_tex;


void interrupt(int32_t n) {
    if (n == INT_SPRITE) {
        if (!renderer) return;
        int32_t x = vm.mem_at(IO_SPRITE_X);
        int32_t y = vm.mem_at(IO_SPRITE_Y);
        int32_t w = vm.mem_at(IO_SPRITE_W);
        int32_t h = vm.mem_at(IO_SPRITE_H);
        int32_t s = vm.mem_at(IO_SPRITE_S);
        int32_t t = vm.mem_at(IO_SPRITE_T);
        int32_t f = vm.mem_at(IO_SPRITE_FLAGS);
        SDL_Rect src = { s, t, w, h };
        SDL_Rect dst = { x, y, w, h };
        SDL_RenderCopyEx(renderer, sprite_tex, &src, &dst, 0, nullptr, SDL_RendererFlip(f));
    }
    else if (n == INT_RAND) {
        int32_t lo = vm.mem_at(IO_RAND_LO);
        int32_t hi = vm.mem_at(IO_RAND_HI);
        static std::random_device rd;
        static std::mt19937 gen(rd());
        vm.mem_at(IO_RAND_RESULT) = std::uniform_int_distribution<>(lo, hi)(gen);
    }
    else {
        printf("ERROR: unknown interrupt %d\n", n);
        exit(1);
    }
}



int main(int argc, char** argv) {
    if (argc != 2) {
        printf("usage: %s game-dir\n", argv[0]);
        return 1;
    }
    std::string dir = argv[1];
    dir += "/";

    if (!vm.load(dir + "code")) return 1;
    vm.run(0, interrupt); // init

    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow(
        "vm",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        SCREEN_W * 3, SCREEN_H * 3,
        SDL_WINDOW_RESIZABLE);

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_PRESENTVSYNC);
    SDL_RenderSetLogicalSize(renderer, SCREEN_W, SCREEN_H);
    sprite_tex = load_texture(renderer, dir + "sprite.png");


    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
            default: break;
            case SDL_QUIT:
                running = false;
                break;
            case SDL_KEYDOWN:
                if (event.key.keysym.scancode == SDL_SCANCODE_ESCAPE) running = false;
                break;
            }
        }

        // input
        const Uint8* ks = SDL_GetKeyboardState(nullptr);
        uint8_t button_bits = 0;
        button_bits |= !!ks[SDL_SCANCODE_LEFT ] << BTN_LEFT;
        button_bits |= !!ks[SDL_SCANCODE_RIGHT] << BTN_RIGHT;
        button_bits |= !!ks[SDL_SCANCODE_UP   ] << BTN_UP;
        button_bits |= !!ks[SDL_SCANCODE_DOWN ] << BTN_DOWN;
        button_bits |= !!ks[SDL_SCANCODE_X    ] << BTN_A;
        button_bits |= !!ks[SDL_SCANCODE_Z    ] << BTN_B;
        button_bits |= !!ks[SDL_SCANCODE_C    ] << BTN_B;
        vm.mem_at(IO_BTN) = button_bits;

        SDL_RenderClear(renderer);
        vm.run(2, interrupt); // update
        SDL_RenderPresent(renderer);
    }

    SDL_DestroyTexture(sprite_tex);
    SDL_Quit();

    return 0;
}
