#!/usr/bin/env python3
"""CLI para emitir (ou rotacionar) o token de autenticação de uma estação.

O dashboard não usa mais a senha do professor pra autenticar heartbeat/sessão
das NUCs — cada estação tem seu próprio token, guardado só como hash no
Postgres. Rodar este script na máquina do dashboard (mesmo `.env`/venv que o
serviço usa) toda vez que uma NUC nova entrar em produção, ou que se suspeite
que um token vazou.

Uso:

  Emitir (ou rotacionar) o token de uma estação:
    python scripts/issue_station_token.py nuc-01 --label "Sala 3, estação 1"

O token é impresso em texto puro **uma única vez** — copiar imediatamente
para o `.env` da NUC correspondente (`PROCTOR_DASHBOARD_STATION_TOKEN`); só
o hash fica gravado, não é recuperável depois. Rodar de novo para o mesmo
`station_id` substitui o token anterior — ele para de autenticar
imediatamente (é assim que se revoga).

Variáveis de ambiente relevantes:
  PROCTOR_DASHBOARD_DATABASE_URL   DSN do Postgres do dashboard (obrigatório)
"""

from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import AppConfig
from src.dashboard.auth import hash_password
from src.dashboard.store import DashboardStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("station_id", help="Identificador da estação (mesmo valor de PROCTOR_DASHBOARD_STATION_ID na NUC)")
    parser.add_argument("--label", default=None, help="Anotação livre pra identificar a estação (ex: 'Sala 3, estação 1')")
    args = parser.parse_args()

    app_config = AppConfig()
    store = DashboardStore(app_config.dashboard.database_url, app_config=app_config)

    token = secrets.token_urlsafe(32)
    store.set_station_token_hash(args.station_id, hash_password(token), label=args.label)

    print(f"Token emitido para '{args.station_id}':")
    print(token)
    print()
    print("Copie agora — não fica recuperável depois. No .env da NUC correspondente:")
    print(f"  PROCTOR_DASHBOARD_STATION_ID={args.station_id}")
    print(f"  PROCTOR_DASHBOARD_STATION_TOKEN={token}")


if __name__ == "__main__":
    main()
