#include "vm.hpp"
#include "apu.hpp"
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <random>
#include <cstdio>
#include <vector>


enum {
    SCREEN_W = 320,
    SCREEN_H = 180,

    BTN_UP = 0,
    BTN_DOWN,
    BTN_LEFT,
    BTN_RIGHT,
    BTN_A,
    BTN_B,
    BTN_MENU,

    INT_PRINT = 1,
    INT_SPRITE,
    INT_SOUND,
    INT_RAND,
    INT_SQRT,
};


struct MemMap {
    int32_t ret;
    int32_t input;

    // interrupts
    union {
        struct {
            int32_t msg;
        } print;
        struct {
            int32_t x;
            int32_t y;
            int32_t w;
            int32_t h;
            int32_t s;
            int32_t t;
            int32_t flip;
        } sprite;
        struct {
            int32_t i;
            int32_t pitch;
            int32_t pw;
            int32_t vol;
            int32_t wave;
        } sound;
        struct {
            int32_t lo;
            int32_t hi;
        } rand;
        struct {
            int32_t a;
            int32_t b;
            int32_t c;
        } math;
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


VM                                       vm;
MemMap&                                  mem = *(MemMap*)vm.mem.data();
std::array<APU::VoiceInput, APU::VOICES> voice_inputs{};
APU                                      apu;
SDL_Renderer*                            renderer;
SDL_Texture*                             sprites;



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
        SDL_RenderCopyEx(renderer, sprites, &src, &dst, 0, nullptr, SDL_RendererFlip(mem.sprite.flip));
    }
    else if (n == INT_SOUND) {
        int32_t i = mem.sound.i;
        if (i >= 0 && i < (int32_t)voice_inputs.size()) {
            voice_inputs[i] = { mem.sound.pitch, mem.sound.pw, mem.sound.vol, mem.sound.wave };
        }
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



static bool write_wav_f32(char const* path, std::vector<float> const& samples, int rate) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    uint32_t data_bytes = (uint32_t)(samples.size() * sizeof(float));
    uint32_t riff_size  = 36 + data_bytes;
    uint16_t audio_fmt  = 3; // IEEE float
    uint16_t channels   = 1;
    uint16_t bits       = 32;
    uint32_t byte_rate  = rate * channels * (bits / 8);
    uint16_t block_align = channels * (bits / 8);
    uint32_t fmt_size   = 16;
    auto w4 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
    auto w2 = [&](uint16_t v) { fwrite(&v, 2, 1, f); };
    fwrite("RIFF", 1, 4, f); w4(riff_size);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); w4(fmt_size);
    w2(audio_fmt); w2(channels); w4((uint32_t)rate); w4(byte_rate); w2(block_align); w2(bits);
    fwrite("data", 1, 4, f); w4(data_bytes);
    fwrite(samples.data(), 1, data_bytes, f);
    fclose(f);
    return true;
}


int main(int argc, char** argv) {
    bool once = false;
    bool bad = false;
    std::string dir;
    std::string record_path;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--once") once = true;
        else if (arg == "--record" && i + 1 < argc) record_path = argv[++i];
        else if (dir.empty()) dir = arg;
        else bad = true;
    }
    if (bad || dir.empty()) {
        printf("usage: %s [--once] [--record out.wav] app-dir\n", argv[0]);
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
    if (!record_path.empty()) apu.begin_capture();

    constexpr Uint32 FRAME_MS = 1000 / 60;
    Uint32           last_tick = SDL_GetTicks();
    bool             running = true;
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

        Uint32 now = SDL_GetTicks();
        Uint32 elapsed = now - last_tick;
        if (elapsed < FRAME_MS) SDL_Delay(FRAME_MS - elapsed);
        last_tick = SDL_GetTicks();

        // input
        uint8_t const* ks = SDL_GetKeyboardState(nullptr);
        mem.input = 0;
        mem.input |= !!ks[SDL_SCANCODE_LEFT  ] << BTN_LEFT;
        mem.input |= !!ks[SDL_SCANCODE_RIGHT ] << BTN_RIGHT;
        mem.input |= !!ks[SDL_SCANCODE_UP    ] << BTN_UP;
        mem.input |= !!ks[SDL_SCANCODE_DOWN  ] << BTN_DOWN;
        mem.input |= !!ks[SDL_SCANCODE_X     ] << BTN_A;
        mem.input |= !!ks[SDL_SCANCODE_Z     ] << BTN_B;
        mem.input |= !!ks[SDL_SCANCODE_C     ] << BTN_B;
        mem.input |= !!ks[SDL_SCANCODE_RETURN] << BTN_MENU;

        SDL_RenderClear(renderer);
        vm.run(interrupt);
        apu.next_frame(voice_inputs);
        SDL_RenderPresent(renderer);
    }

    if (!record_path.empty()) {
        SDL_PauseAudio(1);
        SDL_CloseAudio();
        auto samples = apu.end_capture();
        if (write_wav_f32(record_path.c_str(), samples, APU::MIXRATE))
            printf("wrote %s (%zu samples, %.2fs)\n",
                   record_path.c_str(), samples.size(),
                   samples.size() / (double)APU::MIXRATE);
        else
            printf("ERROR: cannot write '%s'\n", record_path.c_str());
    }

    SDL_DestroyTexture(sprites);
    SDL_Quit();

    return 0;
}
