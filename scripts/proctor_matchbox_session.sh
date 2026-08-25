#!/usr/bin/env bash
set -euo pipefail

export XDG_CURRENT_DESKTOP=Proctor
export XDG_SESSION_DESKTOP=proctor-matchbox
export XDG_SESSION_TYPE=x11

xset s off
xset -dpms
xsetroot -solid "#10243e"

exec matchbox-window-manager \
    -use_titlebar no \
    -use_cursor yes \
    -use_lowlight no \
    -use_desktop_mode plain \
    -use_super_modal yes \
    -kbdconfig /etc/proctor/matchbox-kbdconfig
