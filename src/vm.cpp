#include "vm.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <fstream>


enum Operation {
    MOV, ADD, SUB, MUL, DIV, MOD, AND, ORE, CMP, TST,
    JLT, JLE, JGT, JGE, JEQ, JNE, JMP, JSR,
    RET,
    INT,
};

enum AddressMode {
    NIL, ABS, IND, IMM, IDX,
};

struct Opcode {
    Operation   o;
    AddressMode a;
    AddressMode b;
};

constexpr Opcode OPCODE_TABLE[] = {

    { MOV, ABS, ABS },
    { MOV, ABS, IND },
    { MOV, ABS, IDX },
    { MOV, ABS, IMM },
    { MOV, IND, ABS },
    { MOV, IND, IND },
    { MOV, IND, IDX },
    { MOV, IND, IMM },
    { MOV, IDX, ABS },
    { MOV, IDX, IND },
    { MOV, IDX, IDX },
    { MOV, IDX, IMM },

    { ADD, ABS, ABS },
    { ADD, ABS, IND },
    { ADD, ABS, IDX },
    { ADD, ABS, IMM },
    { ADD, IND, ABS },
    { ADD, IND, IND },
    { ADD, IND, IDX },
    { ADD, IND, IMM },
    { ADD, IDX, ABS },
    { ADD, IDX, IND },
    { ADD, IDX, IDX },
    { ADD, IDX, IMM },

    { SUB, ABS, ABS },
    { SUB, ABS, IND },
    { SUB, ABS, IDX },
    { SUB, ABS, IMM },
    { SUB, IND, ABS },
    { SUB, IND, IND },
    { SUB, IND, IDX },
    { SUB, IND, IMM },
    { SUB, IDX, ABS },
    { SUB, IDX, IND },
    { SUB, IDX, IDX },
    { SUB, IDX, IMM },

    { MUL, ABS, ABS },
    { MUL, ABS, IND },
    { MUL, ABS, IDX },
    { MUL, ABS, IMM },
    { MUL, IND, ABS },
    { MUL, IND, IND },
    { MUL, IND, IDX },
    { MUL, IND, IMM },
    { MUL, IDX, ABS },
    { MUL, IDX, IND },
    { MUL, IDX, IDX },
    { MUL, IDX, IMM },

    { DIV, ABS, ABS },
    { DIV, ABS, IND },
    { DIV, ABS, IDX },
    { DIV, ABS, IMM },
    { DIV, IND, ABS },
    { DIV, IND, IND },
    { DIV, IND, IDX },
    { DIV, IND, IMM },
    { DIV, IDX, ABS },
    { DIV, IDX, IND },
    { DIV, IDX, IDX },
    { DIV, IDX, IMM },

    { MOD, ABS, ABS },
    { MOD, ABS, IND },
    { MOD, ABS, IDX },
    { MOD, ABS, IMM },
    { MOD, IND, ABS },
    { MOD, IND, IND },
    { MOD, IND, IDX },
    { MOD, IND, IMM },
    { MOD, IDX, ABS },
    { MOD, IDX, IND },
    { MOD, IDX, IDX },
    { MOD, IDX, IMM },

    { AND, ABS, ABS },
    { AND, ABS, IND },
    { AND, ABS, IDX },
    { AND, ABS, IMM },
    { AND, IND, ABS },
    { AND, IND, IND },
    { AND, IND, IDX },
    { AND, IND, IMM },
    { AND, IDX, ABS },
    { AND, IDX, IND },
    { AND, IDX, IDX },
    { AND, IDX, IMM },

    { ORE, ABS, ABS },
    { ORE, ABS, IND },
    { ORE, ABS, IDX },
    { ORE, ABS, IMM },
    { ORE, IND, ABS },
    { ORE, IND, IND },
    { ORE, IND, IDX },
    { ORE, IND, IMM },
    { ORE, IDX, ABS },
    { ORE, IDX, IND },
    { ORE, IDX, IDX },
    { ORE, IDX, IMM },

    { CMP, ABS, ABS },
    { CMP, ABS, IND },
    { CMP, ABS, IDX },
    { CMP, ABS, IMM },
    { CMP, IND, ABS },
    { CMP, IND, IND },
    { CMP, IND, IDX },
    { CMP, IND, IMM },
    { CMP, IDX, ABS },
    { CMP, IDX, IND },
    { CMP, IDX, IDX },
    { CMP, IDX, IMM },

    { TST, ABS, ABS },
    { TST, ABS, IND },
    { TST, ABS, IDX },
    { TST, ABS, IMM },
    { TST, IND, ABS },
    { TST, IND, IND },
    { TST, IND, IDX },
    { TST, IND, IMM },
    { TST, IDX, ABS },
    { TST, IDX, IND },
    { TST, IDX, IDX },
    { TST, IDX, IMM },

    { JLT, NIL, IMM },
    { JLE, NIL, IMM },
    { JGT, NIL, IMM },
    { JGE, NIL, IMM },
    { JEQ, NIL, IMM },
    { JNE, NIL, IMM },
    { JMP, NIL, IMM },
    { JSR, NIL, IMM },
    { RET },

    { INT, NIL, IMM },
};

template <class T>
bool read(std::istream& f, T& t) {
    return !! f.read((char*) &t, sizeof(T));
}

bool VM::load(std::string const& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        printf("ERROR: cannot open file '%s'\n", path.c_str());
        return false;
    }

    int32_t base;
    int32_t size;

    read(file, base);
    read(file, size);
    file.read((char*) &mem[base], size * sizeof(int32_t));

    read(file, size);
    code.resize(size);
    file.read((char*) code.data(), size * sizeof(int32_t));

    return true;
}


int32_t VM::next() {
    if (pc < 0 || pc >= int32_t(code.size())) {
        printf("ERROR: bad pc %d\n", pc);
        exit(1);
    }
    return code[pc++];
}

int32_t& VM::mem_at(int32_t addr) {
    if (addr < 0 || addr >= MEM_SIZE) {
        printf("ERROR: bad mem addr %d\n", addr);
        exit(1);
    }
    return mem[addr];
}

void VM::run(int32_t start, std::function<void(int32_t)> interrupt) {
    pc = start;
    std::vector<int32_t> stack;
    int32_t* a   = nullptr;
    int32_t  b   = 0;
    int32_t  res = 0;
    for (size_t steps = 0; steps < STEP_LIMIT; ++steps) {
        auto oc = OPCODE_TABLE[next()];
        if (oc.a == ABS) a = &mem_at(next());
        if (oc.a == IND) a = &mem_at(mem_at(next()));
        if (oc.a == IDX) { b = mem_at(next()); a = &mem_at(b + next()); };
        if (oc.b == IMM) b = next();
        if (oc.b == ABS) b = mem_at(next());
        if (oc.b == IND) b = mem_at(mem_at(next()));
        if (oc.b == IDX) { b = mem_at(next()); b = mem_at(b + next()); };
        switch (oc.o) {
        case MOV: res = *a = b; break;
        case ADD: res = *a += b; break;
        case SUB: res = *a -= b; break;
        case MUL: res = *a *= b; break;
        case DIV: res = *a /= b ?: 1; break;
        case MOD: res = *a %= b ?: 1; break;
        case AND: res = *a &= b; break;
        case ORE: res = *a |= b; break;
        case CMP: res = *a - b; break;
        case TST: res = *a & b; break;
        case JLT: if (res <  0) pc = b; break;
        case JLE: if (res <= 0) pc = b; break;
        case JGT: if (res >  0) pc = b; break;
        case JGE: if (res >= 0) pc = b; break;
        case JEQ: if (res == 0) pc = b; break;
        case JNE: if (res != 0) pc = b; break;
        case JMP: pc = b; break;
        case JSR: stack.push_back(pc); pc = b; break;
        case RET:
            if (stack.empty()) return;
            pc = stack.back();
            stack.pop_back();
            break;
        case INT: interrupt(b); break;
        default: assert(0);
        }
    }
    printf("ERROR: step limit reached\n");
    exit(1);
}

