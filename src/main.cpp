#include "vm.hpp"
#include "apu.hpp"
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <random>


enum {
    SCREEN_W = 320,
    SCREEN_H = 180,

    INPUT_UP = 0,
    INPUT_DOWN,
    INPUT_LEFT,
    INPUT_RIGHT,
    INPUT_A,
    INPUT_B,

    INT_PRINT = 1,
    INT_SPRITE,
    INT_RAND,
    INT_SQRT,
};


struct MemMap {
    int32_t                        ret;
    int32_t                        input;
    std::array<APU::VoiceInput, 4> voices;

    // interrupts
    union {
        struct {
            int32_t x;
            int32_t y;
            int32_t w;
            int32_t h;
            int32_t s;
            int32_t t;
            int32_t flags;
        } sprite;
        struct {
            int32_t lo;
            int32_t hi;
        } rand;
        struct {
            int32_t a;
            int32_t b;
            int32_t c;
        } math;
        struct {
            int32_t msg;
        } print;
    };
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
MemMap&       mem = *(MemMap*)vm.mem.data();
APU           apu;
SDL_Renderer* renderer;
SDL_Texture*  sprites;



void interrupt_once(int32_t n) {
    if (n == INT_PRINT) {
        for (int32_t addr = mem.print.msg; addr >= 0 && addr < VM::MEM_SIZE; ++addr) {
            int32_t c = vm.mem[addr];
            if (c == 0) break;
            putchar(c);
        }
        putchar('\n');
    }
    else {
        printf("ERROR: unknown interrupt %d\n", n);
        exit(1);
    }
}

void interrupt(int32_t n) {
    if (n == INT_SPRITE) {
        if (!renderer) return;
        SDL_Rect src = { mem.sprite.s, mem.sprite.t, mem.sprite.w, mem.sprite.h };
        SDL_Rect dst = { mem.sprite.x, mem.sprite.y, mem.sprite.w, mem.sprite.h };
        SDL_RenderCopyEx(renderer, sprites, &src, &dst, 0, nullptr, SDL_RendererFlip(mem.sprite.flags));
    }
    else if (n == INT_RAND) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        mem.ret = std::uniform_int_distribution<>(mem.rand.lo, mem.rand.hi)(gen);
    }
    else if (n == INT_SQRT) {
        mem.ret = std::sqrt(mem.math.a);
    }
    else interrupt_once(n);
}



int main(int argc, char** argv) {
    bool once = false;
    bool bad = false;
    std::string dir;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--once") once = true;
        else if (dir.empty()) dir = arg;
        else bad = true;
    }
    if (bad || dir.empty()) {
        printf("usage: %s [--once] app-dir\n", argv[0]);
        return 1;
    }

    if (!vm.load(dir + "/code")) return 1;
    if (once) {
        vm.run(interrupt_once);
        return 0;
    }

    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO);
    SDL_Window* window = SDL_CreateWindow(
        "toy-vm",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        SCREEN_W * 3, SCREEN_H * 3,
        SDL_WINDOW_RESIZABLE);

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_PRESENTVSYNC);
    SDL_RenderSetLogicalSize(renderer, SCREEN_W, SCREEN_H);
    sprites = load_texture(renderer, dir + "/sprite.png");

    SDL_AudioSpec spec = { APU::MIXRATE, AUDIO_F32, 1, 0, 64, 0, 0, &APU::callback, &apu };
    SDL_OpenAudio(&spec, nullptr);
    SDL_PauseAudio(0);

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
        uint8_t const* ks = SDL_GetKeyboardState(nullptr);
        mem.input = 0;
        mem.input |= !!ks[SDL_SCANCODE_LEFT ] << INPUT_LEFT;
        mem.input |= !!ks[SDL_SCANCODE_RIGHT] << INPUT_RIGHT;
        mem.input |= !!ks[SDL_SCANCODE_UP   ] << INPUT_UP;
        mem.input |= !!ks[SDL_SCANCODE_DOWN ] << INPUT_DOWN;
        mem.input |= !!ks[SDL_SCANCODE_X    ] << INPUT_A;
        mem.input |= !!ks[SDL_SCANCODE_Z    ] << INPUT_B;
        mem.input |= !!ks[SDL_SCANCODE_C    ] << INPUT_B;

        SDL_RenderClear(renderer);
        vm.run(interrupt);
        apu.next_frame(mem.voices);
        SDL_RenderPresent(renderer);
    }

    SDL_DestroyTexture(sprites);
    SDL_Quit();

    return 0;
}
