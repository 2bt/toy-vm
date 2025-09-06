#include <cstdint>
#include <fstream>
#include <string>
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include "vm.hpp"


enum {
    SCREEN_W = 320,
    SCREEN_H = 180,

    // memory locations
    BTN = 0,
    SPRITE_X,
    SPRITE_Y,
    SPRITE_W,
    SPRITE_H,
    SPRITE_S,
    SPRITE_T,

    DATA_OFFSET = 100,


    BTN_UP = 0,
    BTN_DOWN,
    BTN_LEFT,
    BTN_RIGHT,
    BTN_A,
    BTN_B,

    INT_SPRITE = 0,
};


template <class T>
bool read(std::istream& f, T& t) {
    return !!f.read((char*) &t, sizeof(T));
}


SDL_Texture* load_texture(SDL_Renderer* renderer, std::string file) {
    SDL_Surface* surf = IMG_Load(file.c_str());
    if (!surf) {
        printf("ERROR: IMG_Load: %s\n", SDL_GetError());
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


int main(int argc, char** argv) {
    if (argc != 2) {
        printf("usage: %s game-dir\n", argv[0]);
    }
    std::string dir = argv[1];
    dir += "/";

    VM vm;

    std::ifstream file(dir + "code", std::ios::binary);
    if (!file.is_open()) {
        printf("ERROR: no file 'code'\n");
        return 1;
    }

    int32_t data_size;

    read(file, data_size);
    file.read((char*) &vm.mem_at(DATA_OFFSET), data_size * sizeof(int32_t));


    int32_t code_size;
    read(file, code_size);
    std::vector<int32_t> code(code_size);
    file.read((char*) code.data(), code_size * sizeof(int32_t));

    vm.set_code(std::move(code));

    // init
    vm.run(0);

    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("vm",
                                          SDL_WINDOWPOS_UNDEFINED,
                                          SDL_WINDOWPOS_UNDEFINED,
                                          SCREEN_W * 3,
                                          SCREEN_H * 3,
                                          SDL_WINDOW_RESIZABLE);

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_PRESENTVSYNC);
    SDL_RenderSetLogicalSize(renderer, SCREEN_W, SCREEN_H);
    // SDL_Texture* screen_tex = SDL_CreateTexture(
    //     renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING,
    //     SCREEN_W, SCREEN_H);


    SDL_Texture* sprite_tex = load_texture(renderer, dir + "sprite.png");


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
        const Uint8* ks = SDL_GetKeyboardState(nullptr);

        uint8_t button_bits = 0;
        button_bits |= !!ks[SDL_SCANCODE_LEFT ] << BTN_LEFT;
        button_bits |= !!ks[SDL_SCANCODE_RIGHT] << BTN_RIGHT;
        button_bits |= !!ks[SDL_SCANCODE_UP   ] << BTN_UP;
        button_bits |= !!ks[SDL_SCANCODE_DOWN ] << BTN_DOWN;
        button_bits |= !!ks[SDL_SCANCODE_X    ] << BTN_A;
        button_bits |= !!ks[SDL_SCANCODE_Z    ] << BTN_B;
        button_bits |= !!ks[SDL_SCANCODE_C    ] << BTN_B;
        vm.mem_at(BTN) = button_bits;

        SDL_RenderClear(renderer);

        vm.run(2, [&](int32_t n){
            if (n == INT_SPRITE) {
                SDL_Rect src = {
                    vm.mem_at(SPRITE_S),
                    vm.mem_at(SPRITE_T),
                    vm.mem_at(SPRITE_W),
                    vm.mem_at(SPRITE_H),
                };
                SDL_Rect dst = {
                    vm.mem_at(SPRITE_X),
                    vm.mem_at(SPRITE_Y),
                    vm.mem_at(SPRITE_W),
                    vm.mem_at(SPRITE_H),
                };
                SDL_RenderCopy(renderer, sprite_tex, &src, &dst);
            }

        });


        SDL_RenderPresent(renderer);
    }

    SDL_DestroyTexture(sprite_tex);
    // SDL_DestroyTexture(screen_tex);
    SDL_Quit();

    return 0;
}
