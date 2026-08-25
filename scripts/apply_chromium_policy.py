#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


POLICY_PATHS = (
    Path("/etc/chromium/policies/managed/proctor-browser-hardening.json"),
    Path("/etc/chromium-browser/policies/managed/proctor-browser-hardening.json"),
    Path("/var/snap/chromium/current/policies/managed/proctor-browser-hardening.json"),
)


def validate_policy(policy: object) -> dict[str, object]:
    if not isinstance(policy, dict) or policy.get("URLBlocklist") != ["*"]:
        raise ValueError("URLBlocklist deve bloquear tudo")
    allowlist = policy.get("URLAllowlist")
    if not isinstance(allowlist, list) or not allowlist:
        raise ValueError("URLAllowlist deve conter ao menos uma entrada")
    if any(
        not isinstance(pattern, str)
        or not pattern.startswith(("http://", "https://", "chrome-extension://"))
        for pattern in allowlist
    ):
        raise ValueError("URLAllowlist contem um padrao invalido")
    return policy


def write_policy(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> int:
    try:
        policy = validate_policy(json.load(sys.stdin))
        payload = json.dumps(policy, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        for path in POLICY_PATHS:
            write_policy(path, payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Falha ao instalar policy do Chromium: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
