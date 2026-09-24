#!/usr/bin/env python3
"""CLI para cadastrar/atualizar/remover/listar usuários do dashboard.

Cadastro é fechado por design (o pedido do Corsi foi "sem signup"): a única
forma de um usuário existir é alguém com acesso à máquina do dashboard
rodar este script. Não existe rota HTTP de cadastro.

Uso:
  Cadastrar (ou trocar a senha de) um usuário:
    python scripts/manage_dashboard_user.py add felipehl --display-name "Felipe Henrique" --password

  Listar usuários cadastrados:
    python scripts/manage_dashboard_user.py list

  Remover um usuário (encerra as sessões de cookie dele também):
    python scripts/manage_dashboard_user.py remove felipehl

A senha nunca é passada como argumento de linha de comando (ficaria no
histórico do shell) — o script sempre pede via `getpass`.

Variáveis de ambiente relevantes:
  PROCTOR_DASHBOARD_DATABASE_URL   DSN do Postgres do dashboard (obrigatório)
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import AppConfig
from src.dashboard.auth import hash_password
from src.dashboard.store import DashboardStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Cadastra um usuário novo ou troca a senha/nome de um existente")
    add_parser.add_argument("username", help="Login (identificador único, ex: felipehl)")
    add_parser.add_argument("--display-name", required=True, help="Nome de exibição (ex: 'Felipe Henrique')")

    list_parser = subparsers.add_parser("list", help="Lista os usuários cadastrados")  # noqa: F841

    remove_parser = subparsers.add_parser("remove", help="Remove um usuário (e suas sessões ativas)")
    remove_parser.add_argument("username")

    args = parser.parse_args()

    app_config = AppConfig()
    store = DashboardStore(app_config.dashboard.database_url, app_config=app_config)

    if args.command == "add":
        password = getpass.getpass(f"Senha para '{args.username}': ")
        confirm = getpass.getpass("Confirme a senha: ")
        if password != confirm:
            print("As senhas não coincidem.", file=sys.stderr)
            raise SystemExit(1)
        if not password:
            print("Senha não pode ser vazia.", file=sys.stderr)
            raise SystemExit(1)
        store.upsert_dashboard_user(args.username, hash_password(password), args.display_name)
        print(f"Usuário '{args.username}' ({args.display_name}) cadastrado/atualizado.")
    elif args.command == "list":
        users = store.list_dashboard_users()
        if not users:
            print("Nenhum usuário cadastrado.")
            return
        for user in users:
            print(f"{user['username']}\t{user['display_name']}")
    elif args.command == "remove":
        removed = store.delete_dashboard_user(args.username)
        if removed:
            print(f"Usuário '{args.username}' removido.")
        else:
            print(f"Usuário '{args.username}' não encontrado.", file=sys.stderr)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
