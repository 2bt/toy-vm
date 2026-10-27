app="${1:-./game-breakout}"
[ $# -gt 0 ] && shift
once=; [ -f "$app/sprite.png" ] || once=--once

cmake --build build && \
./compile.py  $app/main.toy && \
./assemble.py $app/main.asm --print && \
./build/toy $app $once "$@"
