#!/usr/bin/env bash
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "Execute como root: sudo bash scripts/install_matchbox_session.sh" >&2
    exit 1
fi

RUN_USER="${SUDO_USER:-proctor}"
RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_SCRIPT="/usr/local/bin/proctor-matchbox-session"
SESSION_DESKTOP="/usr/share/xsessions/proctor-matchbox.desktop"
ACCOUNT_FILE="/var/lib/AccountsService/users/$RUN_USER"
GDM_CONFIG="/etc/gdm3/custom.conf"

apt-get install -y -qq matchbox-window-manager x11-xserver-utils xbindkeys wmctrl

install -m 755 "$PROJECT_DIR/scripts/proctor_matchbox_session.sh" "$SESSION_SCRIPT"
install -d -m 755 /etc/proctor /etc/X11/xorg.conf.d /var/lib/AccountsService/users

cat >/etc/proctor/matchbox-kbdconfig <<'EOF'
# Intencionalmente vazio. O arquivo padrão do Matchbox abre terminais e menus.
EOF
chmod 644 /etc/proctor/matchbox-kbdconfig

cat >"$SESSION_DESKTOP" <<EOF
[Desktop Entry]
Name=Proctor Matchbox
Comment=Sessão X11 dedicada para avaliações
Exec=$SESSION_SCRIPT
TryExec=$SESSION_SCRIPT
Type=Application
DesktopNames=Proctor
X-GDM-SessionRegisters=true
EOF
chmod 644 "$SESSION_DESKTOP"

cat >/etc/X11/xorg.conf.d/90-proctor-kiosk.conf <<'EOF'
Section "ServerFlags"
    Option "DontVTSwitch" "true"
    Option "DontZap" "true"
    Option "DontZoom" "true"
EndSection
EOF

[[ -f "$ACCOUNT_FILE" ]] && cp -a "$ACCOUNT_FILE" "$ACCOUNT_FILE.proctor-backup"
cat >"$ACCOUNT_FILE" <<EOF
[User]
Session=proctor-matchbox
XSession=proctor-matchbox
SystemAccount=false
EOF
chmod 600 "$ACCOUNT_FILE"

cat >"$RUN_HOME/.dmrc" <<'EOF'
[Desktop]
Session=proctor-matchbox
EOF
chown "$RUN_USER:$RUN_USER" "$RUN_HOME/.dmrc"
chmod 600 "$RUN_HOME/.dmrc"

cp -a "$GDM_CONFIG" "$GDM_CONFIG.proctor-backup"
if grep -q '^AutomaticLoginEnable=' "$GDM_CONFIG"; then
    sed -i 's/^AutomaticLoginEnable=.*/AutomaticLoginEnable=True/' "$GDM_CONFIG"
else
    sed -i '/^\[daemon\]/a AutomaticLoginEnable=True' "$GDM_CONFIG"
fi
if grep -q '^AutomaticLogin=' "$GDM_CONFIG"; then
    sed -i "s/^AutomaticLogin=.*/AutomaticLogin=$RUN_USER/" "$GDM_CONFIG"
else
    sed -i "/^\[daemon\]/a AutomaticLogin=$RUN_USER" "$GDM_CONFIG"
fi

echo "Sessão Matchbox instalada para $RUN_USER."
echo "Reinicie o GDM ou a NUC para ativar."
