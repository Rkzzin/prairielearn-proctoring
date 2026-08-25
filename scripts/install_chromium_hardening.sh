#!/usr/bin/env bash
# Instala policies gerenciadas do Chromium para impedir rotas internas perigosas
# durante o uso da estacao de prova. O bloqueio e restrito a paginas internas e
# esquemas sensiveis para nao quebrar navegacao normal de manutencao.

set -euo pipefail

if [ "${EUID}" -ne 0 ]; then
    printf 'Uso: sudo bash %s\n' "$0" >&2
    exit 1
fi

POLICY_NAME="proctor-browser-hardening.json"
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
POLICY_DIRS=(
    "/etc/chromium/policies/managed"
    "/etc/chromium-browser/policies/managed"
    "/var/snap/chromium/current/policies/managed"
)

install_policy() {
    local dir="$1"
    mkdir -p "$dir"
    cat > "${dir}/${POLICY_NAME}" <<'JSON'
{
  "AutofillAddressEnabled": false,
  "AutofillCreditCardEnabled": false,
  "BrowserSignin": 0,
  "DeveloperToolsAvailability": 2,
  "IncognitoModeAvailability": 2,
  "PasswordManagerEnabled": false,
  "SyncDisabled": true,
  "URLBlocklist": [
    "about:*",
    "chrome://*",
    "chrome-untrusted://*",
    "data:*",
    "devtools://*",
    "file://*",
    "javascript:*",
    "view-source:*"
  ]
}
JSON
    chown root:root "${dir}/${POLICY_NAME}"
    chmod 0644 "${dir}/${POLICY_NAME}"
    printf '[install-chromium-hardening] policy instalada: %s\n' "${dir}/${POLICY_NAME}"
}

for dir in "${POLICY_DIRS[@]}"; do
    install_policy "$dir"
done

install -o root -g root -m 0755 \
    "${PROJECT_DIR}/scripts/apply_chromium_policy.py" \
    /usr/local/sbin/proctor-apply-chromium-policy
cat >/etc/sudoers.d/proctor-browser-policy <<'EOF'
proctor ALL=(root) NOPASSWD: /usr/local/sbin/proctor-apply-chromium-policy
EOF
chmod 0440 /etc/sudoers.d/proctor-browser-policy
visudo -cf /etc/sudoers.d/proctor-browser-policy >/dev/null

printf '[install-chromium-hardening] concluido\n'
