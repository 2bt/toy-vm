app="./game-breakout"
once=
for arg in "$@"; do
    if [ "$arg" = "--once" ]; then once=--once
    else app="$arg"
    fi
done
[ -f "$app/sprite.png" ] || once=--once

cmake --build build && \
./compile.py  $app/main.toy && \
./assemble.py $app/main.asm --print && \
./build/toy $app $once
