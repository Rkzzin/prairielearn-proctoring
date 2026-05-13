"""Allowlist de navegação para o Chromium controlado.

A extensão é estática e versionada em ``allowlist_extension``. Em runtime
gravamos apenas um ``config.json`` pequeno com os domínios da prova atual.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

_HOST_RE = re.compile(r"^[a-z0-9][a-z0-9.-]*[a-z0-9]$|^[a-z0-9]$")
_IPV4_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")

EXTENSION_DIR = Path(__file__).resolve().parent / "allowlist_extension"
EXTENSION_CONFIG_NAME = "config.json"


@dataclass(frozen=True)
class AllowedSite:
    host: str
    label: str
    url: str


@dataclass(frozen=True)
class AllowlistConfig:
    start_url: str
    sites: list[AllowedSite] = field(default_factory=list)
    strict_subresources: bool = False
    blocked_message: str = "Este site nao esta autorizado para esta prova."

    def to_payload(self) -> dict[str, object]:
        return {
            "version": 1,
            "startUrl": self.start_url,
            "strictSubresources": self.strict_subresources,
            "blockedMessage": self.blocked_message,
            "sites": [asdict(site) for site in self.sites],
        }


def normalize_host(value: str) -> str | None:
    """Normaliza uma entrada de allowlist para hostname.

    Aceita ``example.com``, ``https://example.com/path`` e ``*.example.com``.
    Caminhos e query strings sao descartados no MVP porque a regra atual e por
    dominio, nao por URL especifica.
    """
    raw = value.strip().lower()
    if not raw:
        return None
    if raw.startswith("*."):
        raw = raw[2:]

    if "://" in raw:
        host = urlsplit(raw).hostname
    else:
        host = urlsplit(f"//{raw}").hostname

    if host is None:
        return None
    host = host.strip(".")
    if not host or "*" in host or "/" in host or " " in host:
        return None
    if _IPV4_RE.fullmatch(host):
        parts = host.split(".")
        if all(0 <= int(part) <= 255 for part in parts):
            return host
        return None
    if not _HOST_RE.fullmatch(host):
        return None
    if ".." in host:
        return None
    return host


def host_from_url(url: str) -> str | None:
    return normalize_host(url)


def build_allowlist_config(
    *,
    start_url: str,
    allowlist: list[str],
    strict_subresources: bool = False,
) -> AllowlistConfig:
    hosts: list[str] = []
    start_host = host_from_url(start_url)
    if start_host:
        hosts.append(start_host)
        hosts.extend(_implicit_hosts_for_start_host(start_host))
    for item in allowlist:
        host = normalize_host(item)
        if host:
            hosts.append(host)

    unique_hosts = sorted(dict.fromkeys(hosts))
    sites = [
        AllowedSite(host=host, label=host, url=_default_url_for_host(host))
        for host in unique_hosts
    ]
    return AllowlistConfig(
        start_url=start_url,
        sites=sites,
        strict_subresources=strict_subresources,
    )


def write_extension_config(
    config: AllowlistConfig,
    *,
    extension_dir: Path | str | None = None,
) -> Path:
    target_dir = Path(extension_dir or EXTENSION_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / EXTENSION_CONFIG_NAME
    target.write_text(
        json.dumps(config.to_payload(), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def build_chromium_policies(
    config: AllowlistConfig,
    *,
    extension_id: str | None = None,
) -> dict[str, object]:
    allowed_patterns = ["chrome-extension://*"]
    for site in config.sites:
        allowed_patterns.extend(_policy_patterns_for_host(site.host))

    policies: dict[str, object] = {
        "URLBlocklist": ["*"],
        "URLAllowlist": sorted(dict.fromkeys(allowed_patterns)),
        "DeveloperToolsAvailability": 2,
        "IncognitoModeAvailability": 1,
        "SyncDisabled": True,
        "BrowserSignin": 0,
        "PasswordManagerEnabled": False,
        "AutofillAddressEnabled": False,
        "AutofillCreditCardEnabled": False,
        "ClearBrowsingDataOnExitList": [
            "browsing_history",
            "download_history",
            "cookies_and_other_site_data",
            "cached_images_and_files",
            "password_signin",
            "autofill",
            "site_settings",
            "hosted_app_data",
        ],
    }
    if extension_id:
        policies["ExtensionInstallBlocklist"] = ["*"]
        policies["ExtensionInstallAllowlist"] = [extension_id]
    return policies


def write_chromium_policies(
    policies: dict[str, object],
    *,
    output_path: Path | str,
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(policies, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def _default_url_for_host(host: str) -> str:
    scheme = "http" if host.startswith("127.") or host == "localhost" else "https"
    return f"{scheme}://{host}"


def _implicit_hosts_for_start_host(host: str) -> list[str]:
    """Inclui hosts conhecidos de redirects do provedor da prova.

    PrairieLearn publica a URL curta em prairielearn.org, mas o login real
    redireciona para us.prairielearn.com. Sem esse host implícito a allowlist
    bloqueia a própria prova antes do aluno conseguir autenticar.
    """
    if host == "prairielearn.org" or host.endswith(".prairielearn.org"):
        return ["us.prairielearn.com"]
    return []


def _policy_patterns_for_host(host: str) -> list[str]:
    schemes = ("http", "https")
    patterns = [f"{scheme}://{host}/*" for scheme in schemes]
    if not _IPV4_RE.fullmatch(host) and host != "localhost":
        patterns.extend(f"{scheme}://*.{host}/*" for scheme in schemes)
    return patterns
