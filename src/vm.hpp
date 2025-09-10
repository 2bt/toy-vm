#pragma once
#include <cstdint>
#include <array>
#include <vector>
#include <functional>


class VM {
public:
    enum {
        MEM_SIZE = 1 << 20,
        STEP_LIMIT = 1000000,
    };

    void set_code(std::vector<int32_t> c) { code = std::move(c); }
    void run(int32_t start, std::function<void(int32_t)> interrupt = [](int32_t){});
    int32_t& mem_at(int32_t addr);

private:
    int32_t next();

    std::vector<int32_t>          code;
    int32_t                       pc  = 0;
    std::array<int32_t, MEM_SIZE> mem = {};
};
