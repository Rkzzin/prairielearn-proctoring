#!/usr/bin/env bash
# Remove artefatos instalados por prototipos antigos do modo prova, sem apagar
# dados de sessoes nem o usuario de manutencao.

set -euo pipefail

EXAM_USER="${PROCTOR_LEGACY_EXAM_USER:-exam}"
STAMP="$(date +%Y%m%d_%H%M%S)"

log() {
    printf '[cleanup-legacy-exam-mode] %s\n' "$*"
}

require_root() {
    if [ "${EUID}" -ne 0 ]; then
        printf 'Uso: sudo bash %s\n' "$0" >&2
        exit 1
    fi
}

remove_path() {
    local path
    for path in "$@"; do
        if [ -e "$path" ] || [ -L "$path" ]; then
            rm -rf "$path"
            log "removido: $path"
        else
            log "ausente: $path"
        fi
    done
}

disable_unit() {
    local unit="$1"
    if systemctl list-unit-files "$unit" >/dev/null 2>&1 || systemctl list-units "$unit" >/dev/null 2>&1; then
        systemctl disable --now "$unit" >/dev/null 2>&1 || true
        log "systemd desativado: $unit"
    else
        log "systemd ausente: $unit"
    fi
}

disable_legacy_gdm_autologin() {
    local conf="/etc/gdm3/custom.conf"

    if [ ! -f "$conf" ]; then
        log "GDM custom.conf ausente"
        return
    fi

    if ! grep -Eq "^[#[:space:]]*AutomaticLogin[[:space:]]*=[[:space:]]*${EXAM_USER}[[:space:]]*$" "$conf"; then
        log "GDM sem autologin legado para ${EXAM_USER}"
        return
    fi

    cp -a "$conf" "${conf}.proctor-legacy-backup.${STAMP}"
    sed -i -E \
        -e "s/^([[:space:]]*)AutomaticLoginEnable[[:space:]]*=.*/\1AutomaticLoginEnable=false/" \
        -e "/^[#[:space:]]*AutomaticLogin[[:space:]]*=[[:space:]]*${EXAM_USER}[[:space:]]*$/d" \
        "$conf"
    log "autologin legado do GDM desativado; backup em ${conf}.proctor-legacy-backup.${STAMP}"
}

remove_legacy_exam_user() {
    if ! id "$EXAM_USER" >/dev/null 2>&1; then
        log "usuario legado ausente: ${EXAM_USER}"
        return
    fi

    loginctl terminate-user "$EXAM_USER" >/dev/null 2>&1 || true
    pkill -KILL -u "$EXAM_USER" >/dev/null 2>&1 || true

    if command -v deluser >/dev/null 2>&1; then
        deluser --remove-home "$EXAM_USER" >/dev/null 2>&1 || userdel -r "$EXAM_USER" >/dev/null 2>&1 || true
    else
        userdel -r "$EXAM_USER" >/dev/null 2>&1 || true
    fi

    if id "$EXAM_USER" >/dev/null 2>&1; then
        passwd -l "$EXAM_USER" >/dev/null 2>&1 || true
        chage -E 0 "$EXAM_USER" >/dev/null 2>&1 || true
        log "usuario legado nao pode ser removido; conta bloqueada: ${EXAM_USER}"
    else
        log "usuario legado removido: ${EXAM_USER}"
    fi
}

purge_legacy_packages() {
    if [ "${PROCTOR_PURGE_LEGACY_PACKAGES:-true}" != "true" ]; then
        log "purge de pacotes legados desativado por PROCTOR_PURGE_LEGACY_PACKAGES"
        return
    fi

    if ! command -v dpkg-query >/dev/null 2>&1 || ! command -v apt-get >/dev/null 2>&1; then
        log "apt/dpkg ausente; pulando purge de pacotes legados"
        return
    fi

    if dpkg-query -W -f='${Status}' openbox 2>/dev/null | grep -q "install ok installed"; then
        DEBIAN_FRONTEND=noninteractive apt-get purge -y openbox >/dev/null 2>&1 || log "falha ao remover pacote legado: openbox"
        DEBIAN_FRONTEND=noninteractive apt-get autoremove -y >/dev/null 2>&1 || true
        log "pacote legado removido: openbox"
    else
        log "pacote legado ausente: openbox"
    fi
}

main() {
    require_root

    log "iniciando limpeza dos artefatos legados de modo prova"

    disable_unit "proctor-exam-mode.path"
    disable_unit "proctor-exam-mode.service"
    disable_unit "proctor-seat-poc.service"

    remove_path \
        /etc/systemd/system/proctor-exam-mode.path \
        /etc/systemd/system/proctor-exam-mode.service \
        /etc/systemd/system/proctor-exam-mode.tmpfiles \
        /etc/systemd/system/proctor-seat-poc.service \
        /etc/tmpfiles.d/proctor-exam-mode.conf \
        /usr/local/sbin/proctor-exam-mode-switch \
        /usr/local/bin/proctor-exam-session \
        /usr/local/bin/proctor-exam-openbox-session \
        /usr/share/xsessions/proctor-exam.desktop \
        /etc/sudoers.d/proctor-exam-mode \
        /run/proctor/exam-mode \
        /run/proctor/seat-poc \
        "/var/lib/AccountsService/users/${EXAM_USER}" \
        "/home/${EXAM_USER}/.dmrc" \
        "/home/${EXAM_USER}/.xbindkeys.noauto" \
        "/home/${EXAM_USER}/.config/openbox/autostart" \
        /etc/chromium/policies/managed/proctor-exam-base.json \
        /etc/chromium-browser/policies/managed/proctor-exam-base.json \
        /var/snap/chromium/current/policies/managed/proctor-exam-base.json

    disable_legacy_gdm_autologin
    remove_legacy_exam_user
    purge_legacy_packages

    systemctl daemon-reload
    log "limpeza concluida; reinicie a NUC se algum autologin legado ainda estiver ativo"
}

main "$@"
