#!/bin/sh
# build the VM, compile a game and run it: ./run.sh [game name or dir] [smolt options]
set -e
app="${1:-outpost-oberon}"
[ -d "$app" ] || app="games/$app"   # ./run.sh breakout == ./run.sh games/breakout
[ $# -gt 0 ] && shift
[ -f "$app/main.smolt" ] || { echo "no game at $app (expected $app/main.smolt)"; exit 1; }
once=; [ -f "$app/sprite.png" ] || once=--once
[ -d build ] || cmake -S . -B build
cmake --build build
./tools/compile.py "$app/main.smolt" --print
./build/smolt "$app" $once "$@"
