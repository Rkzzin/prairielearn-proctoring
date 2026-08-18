#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="proctor-dashboard.service"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-$(whoami)}"
PYTHON_BIN="$PROJECT_DIR/venv/bin/python"
# Default pensado para rodar atras de um nginx (mesma maquina ou compartilhado
# com outro servico) fazendo TLS/virtual host — a app so escuta local. Para
# expor a app direto (sem nginx, ex: teste rapido), passe
# PROCTOR_DASHBOARD_HOST=0.0.0.0.
HOST="${PROCTOR_DASHBOARD_HOST:-127.0.0.1}"
PORT="${PROCTOR_DASHBOARD_PORT:-8010}"
UNIT_PATH="/etc/systemd/system/$SERVICE_NAME"
TMP_UNIT="$(mktemp)"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Execute com sudo: sudo bash scripts/install_dashboard_service.sh" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python da virtualenv nao encontrado em $PYTHON_BIN" >&2
  exit 1
fi

cat >"$TMP_UNIT" <<EOF
[Unit]
Description=Proctor Station Teacher Dashboard
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$PROJECT_DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=$PYTHON_BIN -m uvicorn src.dashboard.app:create_app --factory --host $HOST --port $PORT
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

install -m 644 "$TMP_UNIT" "$UNIT_PATH"
rm -f "$TMP_UNIT"

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"

if command -v lsof >/dev/null 2>&1 && lsof -i ":$PORT" -P -n >/dev/null 2>&1; then
  echo "Porta $PORT ja esta em uso. Pare o dashboard manual antes de iniciar $SERVICE_NAME." >&2
  systemctl status "$SERVICE_NAME" --no-pager || true
  exit 2
fi

systemctl start "$SERVICE_NAME"
systemctl status "$SERVICE_NAME" --no-pager
