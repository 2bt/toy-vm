app="${1:-./game-breakout}"
[ $# -gt 0 ] && shift
once=; [ -f "$app/sprite.png" ] || once=--once

cmake --build build && \
./compile.py  $app/main.smolt && \
./assemble.py $app/main.asm --print && \
./build/smolt $app $once "$@"
