#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="proctor.service"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-$(whoami)}"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
PORT="$("$PYTHON_BIN" - <<'PY'
from src.core.config import AppConfig
print(AppConfig().api_port)
PY
)"
UNIT_PATH="/etc/systemd/system/$SERVICE_NAME"
TMP_UNIT="$(mktemp)"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python da virtualenv não encontrado em $PYTHON_BIN" >&2
  exit 1
fi

cat >"$TMP_UNIT" <<EOF
[Unit]
Description=Proctor Station Session Manager
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$PROJECT_DIR
Environment=PYTHONUNBUFFERED=1
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
