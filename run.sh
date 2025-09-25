app="${1:-./game-breakout}"
cmake --build build && \
./compile.py  $app/main.toy && \
./assemble.py $app/main.asm && \
./toy $app
