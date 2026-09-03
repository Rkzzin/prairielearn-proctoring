#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="proctor.service"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-$(whoami)}"
PYTHON_BIN="$PROJECT_DIR/venv/bin/python"
PORT="$("$PYTHON_BIN" - <<'PY'
from src.core.config import AppConfig
print(AppConfig().api_port)
PY
)"
DATA_DIR="$("$PYTHON_BIN" - <<'PY'
from src.core.config import AppConfig
print(AppConfig().data_dir)
PY
)"
DISPLAY_NAME="$("$PYTHON_BIN" - <<'PY'
from src.core.config import AppConfig
print(AppConfig().recorder.display)
PY
)"
RUN_UID="$(id -u "$RUN_USER")"
XAUTHORITY_PATH="/run/user/$RUN_UID/gdm/Xauthority"
UNIT_PATH="/etc/systemd/system/$SERVICE_NAME"
SUDOERS_PATH="/etc/sudoers.d/proctor-dashboard-reboot"
TMP_UNIT="$(mktemp)"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python da virtualenv não encontrado em $PYTHON_BIN" >&2
  exit 1
fi

# PROCTOR_APP_DATA_DIR (default /opt/proctor/data) costuma ficar fora do
# diretorio do projeto, em caminho que so root pode criar (/opt e' 755
# root:root). Sem isto, a primeira sessao falha ao criar
# {data_dir}/sessions/{id}/... com Permission denied assim que a
# identificacao termina, e o auto-start faz rollback pro "waiting student"
# sem nunca chegar a gravar nada.
mkdir -p "$DATA_DIR"
chown -R "$RUN_USER":"$RUN_USER" "$DATA_DIR"

# O comando autenticado do dashboard só pode reiniciar a estação após um
# `git pull --ff-only` bem-sucedido; não concede sudo genérico ao serviço.
printf '%s ALL=(root) NOPASSWD: /usr/bin/systemctl reboot\n' "$RUN_USER" >"$SUDOERS_PATH"
chmod 440 "$SUDOERS_PATH"
visudo -cf "$SUDOERS_PATH"

cat >"$TMP_UNIT" <<EOF
[Unit]
Description=Proctor Station Session Manager
After=display-manager.service network-online.target
Wants=display-manager.service network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$PROJECT_DIR
Environment=PYTHONUNBUFFERED=1
Environment=DISPLAY=$DISPLAY_NAME
Environment=XAUTHORITY=$XAUTHORITY_PATH
ExecStartPre=/bin/bash -c 'for _ in {1..60}; do /usr/bin/xdpyinfo -display $DISPLAY_NAME >/dev/null 2>&1 && exit 0; sleep 1; done; exit 1'
ExecStart=$PYTHON_BIN -m uvicorn src.api.server:app --host 0.0.0.0 --port $PORT
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

install -m 644 "$TMP_UNIT" "$UNIT_PATH"
rm -f "$TMP_UNIT"

systemctl daemon-reload
systemctl enable --now "$SERVICE_NAME"
systemctl status "$SERVICE_NAME" --no-pager
