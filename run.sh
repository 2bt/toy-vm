app="${1:-breakout}"
[ -d "$app" ] || app="games/$app"   # ./run.sh breakout == ./run.sh games/breakout
[ $# -gt 0 ] && shift
once=; [ -f "$app/sprite.png" ] || once=--once

cmake --build build && \
./tools/compile.py $app/main.smolt --print && \
./build/smolt $app $once "$@"
