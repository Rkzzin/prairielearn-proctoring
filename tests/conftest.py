"""Fixtures compartilhadas de teste.

`psycopg` só é importado dentro da fixture (não no topo do módulo) para não
exigir o extra `dashboard` de quem roda a suíte só com `[station,dev]`
instalado — só os testes que usam `dashboard_database_url` precisam dele.
"""

from __future__ import annotations

import os
import uuid

import pytest

_TEST_DATABASE_DSN = os.environ.get(
    "PROCTOR_TEST_DATABASE_URL",
    "postgresql://postgres:postgres@127.0.0.1:5432/postgres",
)


def _replace_dbname(dsn: str, dbname: str) -> str:
    base, _, _current_dbname = dsn.rpartition("/")
    return f"{base}/{dbname}"


@pytest.fixture
def dashboard_database_url() -> str:
    """DSN de um banco Postgres novo e isolado, dropado ao final do teste.

    Requer um Postgres acessível via `PROCTOR_TEST_DATABASE_URL` (ou o
    default local `postgresql://postgres:postgres@127.0.0.1:5432/postgres`)
    — ver `docker-compose.yml` para subir um localmente.
    """
    import psycopg

    dbname = f"test_{uuid.uuid4().hex}"
    with psycopg.connect(_TEST_DATABASE_DSN, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{dbname}"')

    dsn = _replace_dbname(_TEST_DATABASE_DSN, dbname)
    yield dsn

    with psycopg.connect(_TEST_DATABASE_DSN, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE "{dbname}" WITH (FORCE)')
