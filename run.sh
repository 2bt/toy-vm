app="${1:-./game-breakout}"
[ $# -gt 0 ] && shift
once=; [ -f "$app/sprite.png" ] || once=--once

cmake --build build && \
./tools/compile.py $app/main.smolt --print && \
./build/smolt $app $once "$@"
