# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fantasy-console style virtual machine for writing simple games in a C-like language (`.toy` files). The toolchain consists of:

1. **Compiler** (`compile.py`) - Compiles `.toy` source files to assembly (`.asm`)
2. **Assembler** (`assemble.py`) - Assembles `.asm` files to bytecode (`code` files)
3. **Virtual Machine** (`toy`) - Executes bytecode with graphics and input

## Build & Run Commands

### Main Development Workflow
```bash
# Build VM and run a complete game (compile + assemble + run)
./run.sh game-breakout

# Build the VM only
cmake --build build

# Compile individual .toy source to assembly
./compile.py game-name/main.toy

# Assemble .asm to bytecode
./assemble.py game-name/main.asm

# Run compiled game
./toy game-name
```

### Dependencies
- C++20 compiler
- CMake
- SDL2 and SDL2_image
- Python 3

## Architecture

### Core Components
- **VM Core** (`src/vm.cpp`, `src/vm.hpp`) - Virtual machine with 1MB memory, executes bytecode instructions
- **Main Entry** (`src/main.cpp`) - SDL2-based frontend handling graphics, input, and audio
- **Compiler** (`compile.py`) - Full C-like language compiler with lexer, parser, and code generator
- **Assembler** (`assemble.py`) - Text-based assembler that reads opcode definitions directly from VM source

### Language Design (.toy files)
- C-like syntax with `var`, `func`, `if`, `while`, `for`, `struct`, etc.
- Compiles to stack-based bytecode
- Supports inline assembly with `asm { ... }` blocks
- Current limitations noted in `todo.md` (no forward function declarations, limited struct support)

### Game Structure
Games are organized in directories containing:
- `main.toy` - Source code
- `code` - Compiled bytecode (generated)
- `main.asm` - Assembly output (generated)
- `sprite.png` - Sprite sheet

### Key Design Notes
- The assembler dynamically loads opcode table from `src/vm.cpp` comments, ensuring consistency
- VM uses a simple stack-based architecture with 32-bit integers
- Graphics and input are handled through SDL2 with a fixed console-like display
- Python toolchain uses ruff for linting (configured in `pyproject.toml`)