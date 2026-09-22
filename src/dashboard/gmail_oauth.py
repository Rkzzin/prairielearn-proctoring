"""OAuth 2.0 e SMTP XOAUTH2 para a conta Gmail do dashboard."""

from __future__ import annotations

import base64
import secrets
import smtplib
import ssl
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage
from typing import Any
from urllib.parse import urlencode

import httpx
from cryptography.fernet import Fernet, InvalidToken

_AUTHORIZATION_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
_GMAIL_SCOPE = "https://mail.google.com/"
_ACCOUNT_EMAIL = "corsiferrao@gmail.com"


class GmailOAuthError(RuntimeError):
    """Erro apresentável na vinculação ou entrega pelo Gmail."""


class GmailOAuth:
    def __init__(self, *, store, dashboard_config, http_client_factory=None, smtp_factory=None):
        self._store = store
        self._config = dashboard_config
        self._http_client_factory = http_client_factory or httpx.Client
        self._smtp_factory = smtp_factory or smtplib.SMTP

    @property
    def is_configured(self) -> bool:
        return bool(
            self._config.google_oauth_client_id
            and self._config.google_oauth_client_secret
            and self._config.google_oauth_redirect_uri
            and self._config.google_oauth_encryption_key
        )

    def connection_status(self) -> dict[str, object]:
        connection = self._store.get_google_oauth_connection()
        return {
            "configured": self.is_configured,
            "connected": connection is not None,
            "email": connection.get("email") if connection else None,
            "connected_at": connection.get("connected_at") if connection else None,
        }

    def authorization_url(self) -> str:
        self._require_config()
        state = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii")
        self._store.create_google_oauth_state(
            state,
            datetime.now(UTC) + timedelta(minutes=10),
        )
        query = urlencode(
            {
                "client_id": self._config.google_oauth_client_id,
                "redirect_uri": self._config.google_oauth_redirect_uri,
                "response_type": "code",
                "scope": f"openid email {_GMAIL_SCOPE}",
                "access_type": "offline",
                "prompt": "consent",
                "login_hint": _ACCOUNT_EMAIL,
                "state": state,
            }
        )
        return f"{_AUTHORIZATION_URL}?{query}"

    def connect(self, *, code: str, state: str) -> None:
        self._require_config()
        if not self._store.consume_google_oauth_state(state):
            raise GmailOAuthError("A solicitação de autenticação expirou ou já foi utilizada.")
        token = self._post_token(
            {
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": self._config.google_oauth_redirect_uri,
            }
        )
        refresh_token = token.get("refresh_token")
        access_token = token.get("access_token")
        if not isinstance(refresh_token, str) or not isinstance(access_token, str):
            raise GmailOAuthError("O Google não retornou as credenciais necessárias. Tente novamente.")
        email = self._get_account_email(access_token)
        if email != _ACCOUNT_EMAIL:
            raise GmailOAuthError(f"Autorize exatamente a conta {_ACCOUNT_EMAIL}.")
        self._store.save_google_oauth_connection(
            {
                "email": email,
                "connected_at": datetime.now(UTC).isoformat(),
                "refresh_token": self._fernet.encrypt(refresh_token.encode("utf-8")).decode("ascii"),
            }
        )

    def send(self, message: EmailMessage) -> str:
        connection = self._store.get_google_oauth_connection()
        if connection is None:
            raise GmailOAuthError("Conecte a conta Gmail antes de enviar relatórios.")
        encrypted_token = connection.get("refresh_token")
        if not isinstance(encrypted_token, str):
            raise GmailOAuthError("A credencial Gmail armazenada é inválida.")
        try:
            refresh_token = self._fernet.decrypt(encrypted_token.encode("ascii")).decode("utf-8")
        except (InvalidToken, UnicodeDecodeError) as exc:
            raise GmailOAuthError("Não foi possível ler a credencial Gmail armazenada.") from exc
        access_token = self._post_token(
            {"grant_type": "refresh_token", "refresh_token": refresh_token}
        ).get("access_token")
        if not isinstance(access_token, str):
            raise GmailOAuthError("O Google não retornou um token de acesso para envio.")
        smtp = self._smtp_factory("smtp.gmail.com", 587, timeout=20)
        try:
            smtp.ehlo()
            smtp.starttls(context=ssl.create_default_context())
            smtp.ehlo()
            auth = base64.b64encode(
                f"user={_ACCOUNT_EMAIL}\x01auth=Bearer {access_token}\x01\x01".encode()
            ).decode("ascii")
            code, response = smtp.docmd("AUTH", f"XOAUTH2 {auth}")
            if code != 235:
                raise GmailOAuthError(f"O Gmail recusou a autenticação SMTP: {response.decode(errors='replace')}")
            refused = smtp.send_message(message)
            if refused:
                raise GmailOAuthError("O Gmail recusou um ou mais destinatários.")
            return message.get("Message-ID", "")
        finally:
            try:
                smtp.quit()
            except smtplib.SMTPException:
                pass

    @property
    def _fernet(self) -> Fernet:
        self._require_config()
        try:
            return Fernet(self._config.google_oauth_encryption_key.encode("ascii"))
        except (AttributeError, ValueError) as exc:
            raise GmailOAuthError("A chave de criptografia OAuth do Gmail é inválida.") from exc

    def _post_token(self, payload: dict[str, str]) -> dict[str, Any]:
        self._require_config()
        request = {
            "client_id": self._config.google_oauth_client_id,
            "client_secret": self._config.google_oauth_client_secret,
            **payload,
        }
        with self._http_client_factory(timeout=20) as client:
            response = client.post(_TOKEN_URL, data=request)
        if response.status_code != 200:
            raise GmailOAuthError("O Google recusou a autorização. Verifique as credenciais OAuth.")
        return response.json()

    def _get_account_email(self, access_token: str) -> str:
        with self._http_client_factory(timeout=20) as client:
            response = client.get(_USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"})
        if response.status_code != 200:
            raise GmailOAuthError("Não foi possível confirmar a conta Google autorizada.")
        email = response.json().get("email")
        return email.strip().lower() if isinstance(email, str) else ""

    def _require_config(self) -> None:
        if not self.is_configured:
            raise GmailOAuthError("Configure as credenciais OAuth do Google no ambiente do dashboard.")
