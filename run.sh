app="${1:-./game-breakout}"
cmake --build build && \
./compile.py  $app/main.toy && \
./assemble.py $app/main.asm --print && \
once=; [ -f "$app/sprite.png" ] || once=--once
./build/toy $app $once
