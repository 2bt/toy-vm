#!/bin/sh
# usage: ./gif.sh recording-DIR [out.gif]
# 60 fps frames -> 30 fps gif, 2x nearest-neighbour upscale
set -e
out=${2:-$1.gif}
ffmpeg -loglevel error -y -framerate 60 -i "$1/%05d.png" \
    -vf "fps=30,scale=iw*2:ih*2:flags=neighbor,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=none" \
    "$out"
gifsicle -O3 --batch "$out"
echo "wrote $out ($(du -h "$out" | cut -f1))"
