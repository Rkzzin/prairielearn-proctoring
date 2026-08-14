from __future__ import annotations

from src.dashboard.auth import hash_password, parse_basic_auth, verify_password


def test_hash_and_verify_roundtrip():
    stored = hash_password("senha-forte")
    assert verify_password("senha-forte", stored)
    assert not verify_password("senha-errada", stored)


def test_hash_password_uses_random_salt_by_default():
    assert hash_password("senha-forte") != hash_password("senha-forte")


def test_verify_password_rejects_malformed_stored_hash():
    assert not verify_password("qualquer", "sem-separador")


def test_parse_basic_auth_decodes_credentials():
    import base64

    header = "Basic " + base64.b64encode(b"prof:senha-forte").decode()
    assert parse_basic_auth(header) == ("prof", "senha-forte")


def test_parse_basic_auth_rejects_missing_or_malformed_header():
    assert parse_basic_auth(None) is None
    assert parse_basic_auth("Bearer token") is None
    assert parse_basic_auth("Basic not-base64!!") is None
