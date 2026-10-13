#pragma once

#include <array>
#include <mutex>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <immintrin.h>


class APU {
public:

    struct VoiceInput {
        int32_t pitch;
        int32_t pw;
        int32_t vol;
        int32_t wave;
    };

    static constexpr int MIXRATE = 48000;
    static constexpr int VOICES  = 4;

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

    void next_frame(std::array<VoiceInput, VOICES>const& voice_inputs) {
        std::lock_guard<std::mutex> lock(m_mtx);
        m_voice_inputs = voice_inputs;
    }

private:
    enum { SAW = 1, TRI, PULSE, NOISE };

    void config_voices() {
        std::lock_guard<std::mutex> lock(m_mtx);
        for (size_t i = 0; i < m_voices.size(); ++i) {
            Voice&            v = m_voices[i];
            VoiceInput const& m = m_voice_inputs[i];
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
    std::array<Voice, VOICES>      m_voices{};
    std::array<VoiceInput, VOICES> m_voice_inputs{};
    std::mutex                     m_mtx{};
};
