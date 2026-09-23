#!/usr/bin/env bash
set -euo pipefail

for backlight in /sys/class/backlight/*; do
    [[ -r "$backlight/max_brightness" && -w "$backlight/brightness" ]] || continue
    cat "$backlight/max_brightness" >"$backlight/brightness"
done
