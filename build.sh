#!/bin/sh
# compile every games/*/main.smolt to code and write games/index.json for the web host
set -e
cd "$(dirname "$0")"
list=
for dir in games/*/; do
    id=$(basename "$dir")
    [ -f "$dir/main.smolt" ] || continue
    tools/compile.py "$dir/main.smolt" > /dev/null
    list="$list${list:+, }\"$id\""
    echo "built $dir"
done
printf '[%s]\n' "$list" > games/index.json
