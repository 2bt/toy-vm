#include "vm.hpp"
#include <cstdlib>
#include <cstdio>
#include <cassert>


enum Operation {
    HLT,
    MOV, ADD, SUB, MUL, DIV, MOD, AND, ORE, CMP, TST,
    NEG,
    JLT, JLE, JGT, JGE, JEQ, JNE, JMP, JSR,
    RET,
    INT,
};

enum AddressMode {
    NIL, ABS, IND, IMM,
};

struct Opcode {
    Operation   o;
    AddressMode a;
    AddressMode b;
};

constexpr Opcode OPCODE_TABLE[] = {
    { HLT },

    { MOV, ABS, ABS },
    { MOV, ABS, IND },
    { MOV, ABS, IMM },
    { MOV, IND, ABS },
    { MOV, IND, IND },
    { MOV, IND, IMM },

    { ADD, ABS, ABS },
    { ADD, ABS, IND },
    { ADD, ABS, IMM },
    { ADD, IND, ABS },
    { ADD, IND, IND },
    { ADD, IND, IMM },

    { SUB, ABS, ABS },
    { SUB, ABS, IND },
    { SUB, ABS, IMM },
    { SUB, IND, ABS },
    { SUB, IND, IND },
    { SUB, IND, IMM },

    { MUL, ABS, ABS },
    { MUL, ABS, IND },
    { MUL, ABS, IMM },
    { MUL, IND, ABS },
    { MUL, IND, IND },
    { MUL, IND, IMM },

    { DIV, ABS, ABS },
    { DIV, ABS, IND },
    { DIV, ABS, IMM },
    { DIV, IND, ABS },
    { DIV, IND, IND },
    { DIV, IND, IMM },

    { MOD, ABS, ABS },
    { MOD, ABS, IND },
    { MOD, ABS, IMM },
    { MOD, IND, ABS },
    { MOD, IND, IND },
    { MOD, IND, IMM },

    { AND, ABS, ABS },
    { AND, ABS, IND },
    { AND, ABS, IMM },
    { AND, IND, ABS },
    { AND, IND, IND },
    { AND, IND, IMM },

    { ORE, ABS, ABS },
    { ORE, ABS, IND },
    { ORE, ABS, IMM },
    { ORE, IND, ABS },
    { ORE, IND, IND },
    { ORE, IND, IMM },

    { CMP, ABS, ABS },
    { CMP, ABS, IND },
    { CMP, ABS, IMM },
    { CMP, IND, ABS },
    { CMP, IND, IND },
    { CMP, IND, IMM },

    { TST, ABS, ABS },
    { TST, ABS, IND },
    { TST, ABS, IMM },
    { TST, IND, ABS },
    { TST, IND, IND },
    { TST, IND, IMM },


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
    int32_t res = 0;
    size_t step_counter = 0;
    while (step_counter < STEP_LIMIT) {
        auto oc = OPCODE_TABLE[next()];
        if (oc.o == HLT) break;

        int32_t* a = nullptr;
        int32_t  b = 0;
        if (oc.a == ABS) a = &mem_at(next());
        if (oc.a == IND) a = &mem_at(mem_at(next()));
        if (oc.b == ABS) b = mem_at(next());
        if (oc.b == IND) b = mem_at(mem_at(next()));
        if (oc.b == IMM) b = next();

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
        case RET: pc = stack.back(); stack.pop_back(); break;
        case INT: interrupt(b); break;
        default: assert(0);
        }

        ++step_counter;
    }

    // printf("%4zu |", step_counter);
    // for (size_t i = 0; i < 10; ++i) printf("%6d", mem[i]);
    // printf("\n");

    if (step_counter >= STEP_LIMIT) {
        printf("ERROR: step limit reached\n");
        exit(1);
    }
}

