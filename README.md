# toy-vm

A tiny fantasy-console–style virtual machine.
Write simple games in a C-like language (`.toy`), compile to bytecode, and run them with sprites and input through a custom VM.

Games live in a folder with:

* `code` → compiled VM bytecode
* `sprite.png` → sprite sheet


## Build & Run

Dependencies: C++17, SDL2, SDL2_image, CMake, Python 3

```bash
./run.sh game-breakout
```

This will:

1. Build the VM (`toy`)
2. Compile `main.toy` → `main.asm`
3. Assemble `main.asm` → `code`
4. Run the game in the VM


## Controls

* Arrows → movement
* Z / X / C → buttons
* Esc → quit


## Roadmap

* [x] Breakout example
* [x] Audio support
* [ ] More demos

