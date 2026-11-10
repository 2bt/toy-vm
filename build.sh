#!/bin/sh
# compile every games/*/main.smolt to code and write games/index.json for the web host
set -e
cd "$(dirname "$0")"
list=
for dir in games/*/; do
    id=$(basename "$dir")
    [ -f "$dir/main.smolt" ] || continue
    tools/compile.py "$dir/main.smolt" > /dev/null
    title=$(echo "$id" | sed 's/-/ /g; s/\b\(.\)/\u\1/g')  # outpost-oberon -> Outpost Oberon
    list="$list${list:+,
}  {\"id\": \"$id\", \"title\": \"$title\"}"
    echo "built $dir"
done
printf '[\n%s\n]\n' "$list" > games/index.json
