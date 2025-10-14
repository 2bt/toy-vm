#include "vm.hpp"
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <cmath>
#include <mutex>
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

    INT_SPRITE = 0,
    INT_RAND,
    INT_SQRT,
};


struct MemMap {
    struct Voice {
        int32_t pitch;
        int32_t pw;
        int32_t vol;
        int32_t wave;
    };
    int32_t              ret;
    int32_t              input;
    std::array<Voice, 4> voices;

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
    };
};



class APU {
public:
    static constexpr int MIXRATE = 48000;

    static void callback(void* u, uint8_t* stream, int bytes) {
        constexpr int SAMPLES_PER_FRAME = MIXRATE / 60;
        APU& apu = *(APU*)u;
        float* buffer = (float*)stream;
        int    length = bytes / sizeof(float);
        while (length > 0) {
            static int sample = 0;
            if (sample == 0) apu.config_voices();
            int step = std::min(SAMPLES_PER_FRAME - sample, length);
            apu.mix(buffer, step);
            buffer += step;
            length -= step;
            sample += step;
            if (sample == SAMPLES_PER_FRAME) sample = 0;
        }
    }

    void next_frame(std::array<MemMap::Voice, 4>const& voices) {
        std::lock_guard<std::mutex> lock(m_mem_mtx);
        m_mem_voices = voices;
    }

private:
    enum { SAW = 1, TRI, PULSE, NOISE };

    void config_voices() {
        std::lock_guard<std::mutex> lock(m_mem_mtx);
        for (size_t i = 0; i < m_voices.size(); ++i) {
            Voice&               v = m_voices[i];
            MemMap::Voice const& m = m_mem_voices[i];
            v.pitch = std::exp2f((m.pitch * (1.0f / 8.0f) - 57.0f) * (1.0f / 12.0f)) * (440.0f / MIXRATE);
            v.pw    = (m.pw % 256) * (1.0f / 256.0f);
            v.vol   = std::clamp(m.vol * (1.0f / 256.0f), 0.0f, 1.0f) * 0.5f;
            v.wave  = m.wave;
            if (v.wave == NOISE) v.pitch *= 8.0f;
        }
    }

    void mix(float* out, size_t num_samples) {
        memset(out, 0, sizeof(float) * num_samples);
        for (Voice& v : m_voices) {
            for (size_t i = 0; i < num_samples; ++i) {
                v.phase += v.pitch;
                v.phase -= int(v.phase);
                float x = 0.0f;
                if (v.wave == SAW)   x = v.phase * 2.0f - 1.0f;
                if (v.wave == TRI)   x = 1.0f - 2.0f * std::abs(2.0f * v.phase - 1.0f);
                if (v.wave == PULSE) x = v.phase <= v.pw ? -1.0f : 1.0f;
                if (v.wave == NOISE) {
                    if (v.phase < v.pitch) {
                        uint32_t& s = v.shift;
                        s = (s << 1) | (((s >> 22) ^ (s >> 17)) & 1);
                        // uint8_t n = ((s >> 22) & 1) << 7 |
                        //             ((s >> 20) & 1) << 6 |
                        //             ((s >> 16) & 1) << 5 |
                        //             ((s >> 13) & 1) << 4 |
                        //             ((s >> 11) & 1) << 3 |
                        //             ((s >>  7) & 1) << 2 |
                        //             ((s >>  4) & 1) << 1 |
                        //             ((s >>  2) & 1) << 0;
                        uint8_t n = _pext_u32(s, 0b10100010010100010010100);
                        v.noise = n * (1.0f / 128.0f) - 1.0f;
                    }
                    x = v.noise;
                }
                out[i] += x * v.vol;
            }
        }
    }

    struct Voice {
        float    pitch{};
        float    pw{};
        float    vol{};
        int      wave{};
        float    phase{};
        float    noise{};
        uint32_t shift = 1234567;
    };
    std::array<Voice, 4>         m_voices{};
    std::array<MemMap::Voice, 4> m_mem_voices{};
    std::mutex                   m_mem_mtx{};
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

    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO);
    SDL_Window* window = SDL_CreateWindow(
        "toy-vm",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        SCREEN_W * 3, SCREEN_H * 3,
        SDL_WINDOW_RESIZABLE);

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_PRESENTVSYNC);
    SDL_RenderSetLogicalSize(renderer, SCREEN_W, SCREEN_H);
    sprites = load_texture(renderer, dir + "sprite.png");

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
        vm.run(2, interrupt); // update
        apu.next_frame(mem.voices);
        SDL_RenderPresent(renderer);
    }

    SDL_DestroyTexture(sprites);
    SDL_Quit();

    return 0;
}
