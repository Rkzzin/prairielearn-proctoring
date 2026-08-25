#!/usr/bin/env bash
set -euo pipefail

GDM_CONFIG=/etc/gdm3/custom.conf
ACCOUNT_CONFIG=/var/lib/AccountsService/users/proctor

if grep -qw 'proctor.maintenance=1' /proc/cmdline; then
    automatic_login=False
    session=ubuntu-xorg
else
    automatic_login=True
    session=proctor-matchbox
fi

cat >"${GDM_CONFIG}" <<EOF
[daemon]
AutomaticLogin=proctor
AutomaticLoginEnable=${automatic_login}
WaylandEnable=false

[security]

[xdmcp]

[chooser]

[debug]
EOF

install -d -m 0755 "$(dirname "${ACCOUNT_CONFIG}")"
cat >"${ACCOUNT_CONFIG}" <<EOF
[User]
Session=${session}
XSession=${session}
SystemAccount=false
EOF
