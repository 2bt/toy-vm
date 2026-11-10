#include "vm.hpp"
#include "apu.hpp"
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <random>
#include <cstdio>
#include <ctime>
#include <vector>
#include <filesystem>


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
    INT_SAVE,
    INT_LOAD,
};


struct MemMap {
    int32_t ret;
    int32_t input;

    // interrupt args — disjoint so nested int calls don't clobber
    struct { int32_t msg; } print;                                          // @2
    struct { int32_t x, y, w, h, s, t, flip; } sprite;                      // @3
    struct { int32_t i, pitch, pw, vol, wave; } sound;                      // @10
    struct { int32_t lo, hi; } rand;                                        // @15
    struct { int32_t a, pad[2]; } sqrt;                                     // @17 (pad: save stays at @20)
    struct { int32_t addr, len; } save;                                     // @20 save + load
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


VM               vm;
MemMap&          mem = *(MemMap*)vm.mem.data();
APU::VoiceInputs voice_inputs{};
APU              apu;
SDL_Renderer*    renderer;
SDL_Texture*     sprites;
SDL_Texture*     canvas; // SCREEN_W x SCREEN_H render target, scaled to the window
std::string      save_path;



void interrupt(int32_t n) {
    if (n == INT_SPRITE) {
        if (!renderer) return; // --once
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
    else if (n == INT_PRINT) {
        for (int32_t addr = mem.print.msg; addr >= 0 && addr < VM::MEM_SIZE; ++addr) {
            int32_t c = vm.mem[addr];
            if (c == 0) break;
            putchar(c);
        }
        putchar('\n');
    }
    else if (n == INT_RAND) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        mem.ret = mem.rand.hi < mem.rand.lo ? mem.rand.lo : std::uniform_int_distribution<>(mem.rand.lo, mem.rand.hi)(gen);
    }
    else if (n == INT_SQRT) {
        mem.ret = mem.sqrt.a > 0 ? std::sqrt(mem.sqrt.a) : 0;
    }
    else if (n == INT_SAVE || n == INT_LOAD) {
        // one save slot per game: raw words in <game-dir>/save; ret = words written/read
        int32_t addr = mem.save.addr, len = mem.save.len;
        mem.ret = 0;
        if (addr < 0 || len < 0 || addr > VM::MEM_SIZE - len) return;
        if (FILE* f = fopen(save_path.c_str(), n == INT_SAVE ? "wb" : "rb")) {
            mem.ret = n == INT_SAVE ? fwrite(&vm.mem[addr], 4, len, f) : fread(&vm.mem[addr], 4, len, f);
            fclose(f);
        }
    }
    else {
        printf("ERROR: unknown interrupt %d\n", n);
        exit(1);
    }
}



// save the canvas (native resolution) as png; call while canvas is the render target
static void save_canvas(char const* path) {
    SDL_Surface* surf = SDL_CreateRGBSurfaceWithFormat(0, SCREEN_W, SCREEN_H, 32, SDL_PIXELFORMAT_RGBA32);
    if (surf && SDL_RenderReadPixels(renderer, nullptr, SDL_PIXELFORMAT_RGBA32, surf->pixels, surf->pitch) == 0) {
        if (IMG_SavePNG(surf, path) != 0) printf("ERROR: cannot write '%s': %s\n", path, IMG_GetError());
    } else {
        printf("ERROR: screenshot failed: %s\n", SDL_GetError());
    }
    if (surf) SDL_FreeSurface(surf);
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
    save_path = dir + "/save";
    if (once) {
        vm.run(interrupt);
        return 0;
    }

    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_GAMECONTROLLER);
    SDL_Window* window = SDL_CreateWindow(
        "Smolt",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        SCREEN_W * 3, SCREEN_H * 3,
        SDL_WINDOW_RESIZABLE
    );

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_PRESENTVSYNC);
    SDL_RenderSetLogicalSize(renderer, SCREEN_W, SCREEN_H);
    sprites = load_texture(renderer, dir + "/sprite.png");
    canvas  = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_TARGET, SCREEN_W, SCREEN_H);
    SDL_SetTextureBlendMode(canvas, SDL_BLENDMODE_NONE);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255); // opaque clear, so screenshots aren't transparent

    SDL_AudioSpec spec = { APU::MIXRATE, AUDIO_F32, 1, 0, 64, 0, 0, &APU::callback, &apu };
    SDL_OpenAudio(&spec, nullptr);
    SDL_PauseAudio(0);
    if (!record_path.empty()) apu.begin_capture();

    Uint32           start_tick = SDL_GetTicks();
    Uint32           frame = 0;       // frames since start_tick, for exact 60 fps pacing
    bool             running = true;
    bool             take_screenshot = false;
    std::string      recording_dir;   // non-empty while recording frames (toggle with R)
    int              recording_frame = 0;
    std::vector<SDL_GameController*> pads;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
            default: break;
            case SDL_QUIT:
                running = false;
                break;
            case SDL_KEYDOWN:
                if (event.key.repeat) break;
                if (event.key.keysym.scancode == SDL_SCANCODE_ESCAPE) running = false;
                else if (event.key.keysym.scancode == SDL_SCANCODE_F12) take_screenshot = true;
                else if (event.key.keysym.scancode == SDL_SCANCODE_R) {
                    if (recording_dir.empty()) {
                        recording_dir   = "recording-" + std::to_string(time(nullptr));
                        recording_frame = 0;
                        std::filesystem::create_directory(recording_dir);
                        printf("recording to %s/\n", recording_dir.c_str());
                    } else {
                        printf("recorded %d frames\n", recording_frame);
                        recording_dir.clear();
                    }
                }
                break;
            case SDL_CONTROLLERDEVICEADDED: // also fires for pads present at startup
                if (SDL_GameController* gc = SDL_GameControllerOpen(event.cdevice.which)) pads.push_back(gc);
                break;
            case SDL_CONTROLLERDEVICEREMOVED:
                if (SDL_GameController* gc = SDL_GameControllerFromInstanceID(event.cdevice.which)) {
                    std::erase(pads, gc);
                    SDL_GameControllerClose(gc);
                }
                break;
            }
        }

        Uint32 due = ++frame * 1000 / 60;
        Uint32 now = SDL_GetTicks() - start_tick;
        if (now < due) SDL_Delay(due - now);
        else if (now > due + 100) start_tick = SDL_GetTicks(), frame = 0; // fell behind: don't rush to catch up

        // input
        uint8_t const* ks = SDL_GetKeyboardState(nullptr);
        mem.input = 0;
        mem.input |= !!(ks[SDL_SCANCODE_LEFT ] | ks[SDL_SCANCODE_A]) << BTN_LEFT;
        mem.input |= !!(ks[SDL_SCANCODE_RIGHT] | ks[SDL_SCANCODE_D]) << BTN_RIGHT;
        mem.input |= !!(ks[SDL_SCANCODE_UP   ] | ks[SDL_SCANCODE_W]) << BTN_UP;
        mem.input |= !!(ks[SDL_SCANCODE_DOWN ] | ks[SDL_SCANCODE_S]) << BTN_DOWN;
        mem.input |= !!ks[SDL_SCANCODE_X     ] << BTN_A;
        mem.input |= !!ks[SDL_SCANCODE_Z     ] << BTN_B;
        mem.input |= !!ks[SDL_SCANCODE_C     ] << BTN_B;
        mem.input |= !!ks[SDL_SCANCODE_RETURN] << BTN_MENU;
        for (SDL_GameController* gc : pads) {
            auto btn = [&](SDL_GameControllerButton b) { return SDL_GameControllerGetButton(gc, b); };
            auto axis = [&](SDL_GameControllerAxis a) { return SDL_GameControllerGetAxis(gc, a); };
            constexpr int DEAD = 16000;
            mem.input |= (btn(SDL_CONTROLLER_BUTTON_DPAD_LEFT ) || axis(SDL_CONTROLLER_AXIS_LEFTX) < -DEAD) << BTN_LEFT;
            mem.input |= (btn(SDL_CONTROLLER_BUTTON_DPAD_RIGHT) || axis(SDL_CONTROLLER_AXIS_LEFTX) >  DEAD) << BTN_RIGHT;
            mem.input |= (btn(SDL_CONTROLLER_BUTTON_DPAD_UP   ) || axis(SDL_CONTROLLER_AXIS_LEFTY) < -DEAD) << BTN_UP;
            mem.input |= (btn(SDL_CONTROLLER_BUTTON_DPAD_DOWN ) || axis(SDL_CONTROLLER_AXIS_LEFTY) >  DEAD) << BTN_DOWN;
            mem.input |= (btn(SDL_CONTROLLER_BUTTON_A) || btn(SDL_CONTROLLER_BUTTON_Y)) << BTN_A;
            mem.input |= (btn(SDL_CONTROLLER_BUTTON_B) || btn(SDL_CONTROLLER_BUTTON_X)) << BTN_B;
            mem.input |= btn(SDL_CONTROLLER_BUTTON_START) << BTN_MENU;
        }

        SDL_SetRenderTarget(renderer, canvas);
        SDL_RenderClear(renderer);
        vm.run(interrupt);
        apu.next_frame(voice_inputs);
        voice_inputs = {};
        if (take_screenshot) {
            take_screenshot = false;
            char path[64];
            snprintf(path, sizeof path, "screenshot-%ld.png", (long)time(nullptr));
            save_canvas(path);
            printf("wrote %s\n", path);
        }
        if (!recording_dir.empty()) {
            char path[64];
            snprintf(path, sizeof path, "/%05d.png", recording_frame++);
            save_canvas((recording_dir + path).c_str());
        }
        SDL_SetRenderTarget(renderer, nullptr);
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, canvas, nullptr, nullptr);
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

    SDL_DestroyTexture(canvas);
    SDL_DestroyTexture(sprites);
    SDL_Quit();

    return 0;
}
