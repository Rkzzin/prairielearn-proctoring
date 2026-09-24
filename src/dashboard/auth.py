"""Autenticação usuário/senha do dashboard.

PBKDF2-HMAC-SHA256 (stdlib, sem dependência nova), comparação em tempo
constante. Sem SQL manual em lugar nenhum — o armazenamento em
`DashboardStore` usa apenas queries parametrizadas.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os
import secrets

_ITERATIONS = 390_000

#: Nome do cookie de sessão do painel. HttpOnly + SameSite=Lax (setado na
#: hora do login em app.py) — não é JS-acessível nem enviado em navegação
#: cross-site, só nas requisições do próprio dashboard.
SESSION_COOKIE_NAME = "proctor_dashboard_session"


def generate_session_token() -> str:
    """Token de sessão opaco — só o hash (`hash_password`) fica no Postgres.

    Reaproveita PBKDF2 em vez de comparar o token em claro contra o banco,
    igual já é feito pro token de estação (`station_tokens`).
    """
    return secrets.token_urlsafe(32)


def hash_password(password: str, *, salt: bytes | None = None) -> str:
    salt = salt if salt is not None else os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, digest_hex = stored_hash.split("$", 1)
        salt = bytes.fromhex(salt_hex)
    except ValueError:
        return False
    expected = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
    return hmac.compare_digest(expected.hex(), digest_hex)


def hash_session_token(token: str) -> str:
    """SHA-256 do token de sessão — determinístico (sem salt), pra permitir

    lookup por igualdade no Postgres. Diferente de `hash_password`: aqui o
    "segredo" é a alta entropia do token gerado por `generate_session_token`
    (32 bytes de `secrets.token_urlsafe`), não uma senha escolhida por humano
    que precisaria de salt/iterações contra força bruta offline.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def parse_basic_auth(header_value: str | None) -> tuple[str, str] | None:
    """Extrai (usuário, senha) de um header ``Authorization: Basic ...``."""
    if not header_value or not header_value.startswith("Basic "):
        return None
    encoded = header_value[len("Basic ") :]
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return None
    username, sep, password = decoded.partition(":")
    if not sep or not username:
        return None
    return username, password
