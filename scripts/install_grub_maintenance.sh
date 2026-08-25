#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
    echo "Execute como root." >&2
    exit 1
fi

if [[ -z ${GRUB_PASSWORD_HASH:-} ]]; then
    echo "Defina GRUB_PASSWORD_HASH com um hash gerado por grub-mkpasswd-pbkdf2." >&2
    exit 1
fi

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

install -m 0755 "${PROJECT_DIR}/scripts/proctor_boot_mode.sh" /usr/local/sbin/proctor-boot-mode
install -m 0755 "${PROJECT_DIR}/scripts/grub_proctor_boot_modes" /etc/grub.d/09_proctor_boot_modes
install -m 0644 "${PROJECT_DIR}/systemd/proctor-boot-mode.service" /etc/systemd/system/proctor-boot-mode.service

cat >/etc/grub.d/01_proctor_users <<EOF
#!/bin/sh
cat <<'GRUB_EOF'
set superusers="proctor-admin"
password_pbkdf2 proctor-admin ${GRUB_PASSWORD_HASH}
GRUB_EOF
EOF
chmod 0755 /etc/grub.d/01_proctor_users

install -d -m 0755 /etc/default/grub.d
cat >/etc/default/grub.d/90-proctor-boot.cfg <<'EOF'
GRUB_DEFAULT=proctor-kiosk
GRUB_TIMEOUT_STYLE=menu
GRUB_TIMEOUT=8
EOF

install -d -m 0755 /etc/systemd/system/proctor.service.d
cat >/etc/systemd/system/proctor.service.d/maintenance.conf <<'EOF'
[Unit]
ConditionKernelCommandLine=!proctor.maintenance=1
EOF

systemctl daemon-reload
systemctl enable proctor-boot-mode.service
/usr/local/sbin/proctor-boot-mode
update-grub

echo "Modo de manutenção GNOME instalado."
