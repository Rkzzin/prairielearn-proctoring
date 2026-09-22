from __future__ import annotations

from email.message import EmailMessage
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

from cryptography.fernet import Fernet

from src.dashboard.gmail_oauth import GmailOAuth


class Store:
    def __init__(self):
        self.states = set()
        self.connection = None

    def create_google_oauth_state(self, state, _expires_at):
        self.states.add(state)

    def consume_google_oauth_state(self, state):
        if state not in self.states:
            return False
        self.states.remove(state)
        return True

    def save_google_oauth_connection(self, payload):
        self.connection = payload

    def get_google_oauth_connection(self):
        return self.connection


class Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class HttpClient:
    def __init__(self, responses):
        self.responses = responses
        self.requests = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def post(self, url, *, data):
        self.requests.append(("POST", url, data))
        return self.responses.pop(0)

    def get(self, url, *, headers):
        self.requests.append(("GET", url, headers))
        return self.responses.pop(0)


class SMTP:
    def __init__(self, *_args, **_kwargs):
        self.auth = None
        self.sent = None

    def ehlo(self):
        pass

    def starttls(self, *, context):
        assert context is not None

    def docmd(self, command, argument):
        self.auth = (command, argument)
        return 235, b"accepted"

    def send_message(self, message):
        self.sent = message
        return {}

    def quit(self):
        pass


def _config():
    return SimpleNamespace(
        google_oauth_client_id="client-id",
        google_oauth_client_secret="client-secret",
        google_oauth_redirect_uri="https://dashboard.example/api/notification-settings/gmail/callback",
        google_oauth_encryption_key=Fernet.generate_key().decode("ascii"),
    )


def test_gmail_oauth_connects_only_expected_account_and_sends_with_xoauth2():
    store = Store()
    http = HttpClient(
        [
            Response({"access_token": "initial-token", "refresh_token": "refresh-token"}),
            Response({"email": "corsiferrao@gmail.com"}),
            Response({"access_token": "send-token"}),
        ]
    )
    smtp = SMTP()
    oauth = GmailOAuth(
        store=store,
        dashboard_config=_config(),
        http_client_factory=lambda **_kwargs: http,
        smtp_factory=lambda *_args, **_kwargs: smtp,
    )

    authorization_url = oauth.authorization_url()
    state = parse_qs(urlsplit(authorization_url).query)["state"][0]
    assert parse_qs(urlsplit(authorization_url).query)["login_hint"] == ["corsiferrao@gmail.com"]

    oauth.connect(code="authorization-code", state=state)
    assert store.connection["email"] == "corsiferrao@gmail.com"
    assert "refresh-token" not in str(store.connection)

    message = EmailMessage()
    message["From"] = "corsiferrao@gmail.com"
    message["To"] = "teacher@example.edu"
    message.set_content("test")
    oauth.send(message)

    assert smtp.auth[0] == "AUTH"
    assert smtp.auth[1].startswith("XOAUTH2 ")
    assert smtp.sent is message
