"""Relatórios por e-mail de avaliações finalizadas, enviados pelo AWS SES."""

from __future__ import annotations

import html
import logging
import smtplib
import ssl
import threading
from collections import Counter
from email.message import EmailMessage
from email.utils import make_msgid
from types import SimpleNamespace
from urllib.parse import quote
from zoneinfo import ZoneInfo

import boto3

from src.dashboard.integrity_score import compute_integrity_score
from src.dashboard.models import EventSeverity, EventSnapshotRecord, SessionEmailReport
from src.dashboard.store import DashboardStore

logger = logging.getLogger(__name__)

_REPORT_TIMEZONE = ZoneInfo("America/Sao_Paulo")
_MAX_INLINE_IMAGE_BYTES = 7 * 1024 * 1024

_EVENT_LABELS = {
    "GAZE_WARNING": "Olhar desviado detectado",
    "GAZE_ALERT": "Alerta de olhar desviado",
    "ABSENCE_WARNING": "Aluno não detectado pela câmera",
    "ABSENCE_ALERT": "Alerta de ausência",
    "MULTI_FACE_ALERT": "Múltiplas faces detectadas",
    "DIFFERENT_USER_ALERT": "Usuário diferente detectado",
    "ELECTRONIC_DEVICE_DETECTED": "Celular ou notebook detectado",
    "BROWSER_EXIT_ALERT": "Navegador protegido foi fechado",
    "UNRECOGNIZED_AUTHENTICATION": "Tentativa de autenticação não reconhecida",
}


class SessionReportMailer:
    def __init__(
        self,
        *,
        store: DashboardStore,
        ses_client_factory=None,
        gmail_username: str | None = None,
        gmail_app_password: str | None = None,
        smtp_factory=None,
    ):
        self._store = store
        self._ses_client_factory = ses_client_factory or (
            lambda region: boto3.client("ses", region_name=region)
        )
        self._gmail_username = gmail_username
        self._gmail_app_password = gmail_app_password
        self._smtp_factory = smtp_factory or smtplib.SMTP
        self._lock = threading.Lock()
        self._running: set[str] = set()

    def prepare(self, session_id: str) -> bool:
        """Persiste a intenção antes das imagens para sobreviver a reinícios."""
        settings = self._store.get_notification_settings()
        if not self._valid_settings(settings):
            return False
        return self._store.queue_email_report(
            session_id,
            settings,
            status="waiting_snapshots",
        )

    def enqueue(self, session_id: str, *, force: bool = False) -> bool:
        if not force and self._store.activate_waiting_email_report(session_id):
            self._start(session_id)
            return True
        settings = self._store.get_notification_settings()
        if not self._valid_settings(settings):
            return False
        queued = self._store.queue_email_report(
            session_id,
            settings,
            force=force,
        )
        if queued:
            self._start(session_id)
        return queued

    def resume_pending(self) -> None:
        for session_id in self._store.pending_email_report_ids():
            self._start(session_id)

    def _valid_settings(self, settings) -> bool:
        return settings.enabled and self._has_delivery_config(settings)

    def _has_delivery_config(self, settings) -> bool:
        configured = bool(settings.sender_email and settings.recipient_emails)
        if settings.delivery_provider == "gmail":
            return configured and bool(
                self._gmail_username
                and self._gmail_app_password
                and settings.sender_email == self._gmail_username
            )
        return configured

    def _start(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._running:
                return
            self._running.add(session_id)
        threading.Thread(
            target=self._run,
            args=(session_id,),
            name=f"session-email-{session_id[:24]}",
            daemon=True,
        ).start()

    def _run(self, session_id: str) -> None:
        report = None
        try:
            report = self._store.claim_email_report(session_id)
            session = self._store.get_session(session_id)
            if report is None or session is None:
                return
            snapshots = self._select_snapshots(
                self._store.list_event_snapshots(session_id),
                report.image_link_limit,
            )
            inline_images = self._load_inline_images(snapshots)
            message = self._build_message(report, session, snapshots, inline_images)
            message_id = self._send_message(report, message)
            self._store.finish_email_report(
                report,
                message_id=message_id,
            )
        except Exception as exc:
            logger.exception("Falha ao enviar relatório da sessão %s", session_id)
            if report is not None:
                self._store.finish_email_report(report, error=str(exc)[:500])
        finally:
            with self._lock:
                self._running.discard(session_id)

    def send_test(self, settings) -> None:
        if not self._has_delivery_config(settings):
            raise RuntimeError("Configure e conecte o provedor de e-mail antes do teste.")
        message = EmailMessage()
        message["Subject"] = "Teste de e-mail do Proctoring"
        message["From"] = settings.sender_email
        message["To"] = ", ".join(settings.recipient_emails)
        message["Message-ID"] = make_msgid(domain="proctoring.local")
        message.set_content("O envio de e-mail do dashboard está configurado corretamente.")
        report = SimpleNamespace(
            delivery_provider=settings.delivery_provider,
            ses_region=settings.ses_region,
            sender_email=settings.sender_email,
            recipient_emails=settings.recipient_emails,
        )
        self._send_message(report, message)

    def _send_message(self, report, message: EmailMessage) -> str:
        if report.delivery_provider == "gmail":
            return self._send_gmail(message)
        response = self._ses_client_factory(report.ses_region).send_raw_email(
            Source=report.sender_email,
            Destinations=report.recipient_emails,
            RawMessage={"Data": message.as_bytes()},
        )
        return str(response.get("MessageId") or "")

    def _send_gmail(self, message: EmailMessage) -> str:
        if not self._gmail_username or not self._gmail_app_password:
            raise RuntimeError("Configure a conta Gmail e a senha de app no ambiente do dashboard.")
        smtp = self._smtp_factory("smtp.gmail.com", 587, timeout=20)
        try:
            smtp.ehlo()
            smtp.starttls(context=ssl.create_default_context())
            smtp.ehlo()
            smtp.login(self._gmail_username, self._gmail_app_password)
            refused = smtp.send_message(message)
            if refused:
                raise RuntimeError("O Gmail recusou um ou mais destinatários.")
            return message.get("Message-ID", "")
        finally:
            try:
                smtp.quit()
            except smtplib.SMTPException:
                pass

    @staticmethod
    def _select_snapshots(
        snapshots: list[EventSnapshotRecord],
        limit: int,
    ) -> list[EventSnapshotRecord]:
        ready = [snapshot for snapshot in snapshots if snapshot.status == "ready"]
        ready.sort(
            key=lambda snapshot: (
                0 if snapshot.severity == EventSeverity.CRITICAL else 1,
                snapshot.event_timestamp,
            )
        )
        return ready if limit == 0 else ready[:limit]

    def _load_inline_images(
        self,
        snapshots: list[EventSnapshotRecord],
    ) -> dict[str, bytes]:
        images: dict[str, bytes] = {}
        total_bytes = 0
        for snapshot in snapshots:
            try:
                image = self._store.read_event_snapshot_image(snapshot)
            except Exception:
                logger.warning(
                    "Falha ao carregar imagem inline do evento %s",
                    snapshot.event_key,
                    exc_info=True,
                )
                continue
            if not image or total_bytes + len(image) > _MAX_INLINE_IMAGE_BYTES:
                continue
            images[snapshot.event_key] = image
            total_bytes += len(image)
        return images

    @staticmethod
    def _build_message(
        report: SessionEmailReport,
        session,
        snapshots,
        inline_images: dict[str, bytes] | None = None,
    ) -> EmailMessage:
        inline_images = inline_images or {}
        base_url = report.public_dashboard_url.rstrip("/")
        session_url = f"{base_url}/sessions/{quote(session.session_id, safe='')}"
        all_flagged = [
            event
            for event in session.events
            if event.severity in {EventSeverity.WARNING, EventSeverity.CRITICAL}
        ]
        critical_count = sum(
            event.severity == EventSeverity.CRITICAL for event in all_flagged
        )
        warning_count = len(all_flagged) - critical_count
        omitted_count = max(0, len(all_flagged) - len(snapshots))
        counts = Counter(event.event_type for event in all_flagged)
        integrity = compute_integrity_score(session)
        student_name = session.student.student_name if session.student else "Não identificado"
        student_id = session.student.student_id if session.student else "-"
        started_at = session.started_at.astimezone(_REPORT_TIMEZONE)
        started_at_label = started_at.strftime("%d/%m/%Y às %H:%M:%S")

        rows = []
        text_rows = []
        related_images = []
        for index, snapshot in enumerate(snapshots, start=1):
            label = _EVENT_LABELS.get(snapshot.event_type, snapshot.event_type)
            image_url = (
                f"{session_url}/event-snapshots/{quote(snapshot.event_key, safe='')}"
            )
            image = inline_images.get(snapshot.event_key)
            if image:
                cid = f"snapshot-{index}@proctoring"
                image_html = (
                    f'<a href="{html.escape(image_url)}">'
                    f'<img src="cid:{cid}" alt="{html.escape(label)}" '
                    'style="display:block;max-width:480px;width:100%;height:auto"></a><br>'
                    f'<a href="{html.escape(image_url)}">Abrir imagem</a>'
                )
                related_images.append((cid, f"snapshot-{index}.jpg", image))
            else:
                image_html = f'<a href="{html.escape(image_url)}">Ver imagem</a>'
            relative_seconds = max(
                0,
                int((snapshot.event_timestamp - session.started_at).total_seconds()),
            )
            minutes, seconds = divmod(relative_seconds, 60)
            relative_time = f"{minutes:02d}:{seconds:02d}"
            rows.append(
                "<tr>"
                f"<td>{html.escape(relative_time)}</td>"
                f"<td>{html.escape(label)}</td>"
                f"<td>{html.escape(snapshot.severity.value)}</td>"
                f"<td>{image_html}</td>"
                "</tr>"
            )
            text_rows.append(
                f"- {relative_time} | {label} | {snapshot.severity.value} | {image_url}"
            )

        summary_rows = "".join(
            f"<li>{html.escape(_EVENT_LABELS.get(event_type, event_type))}: {count}</li>"
            for event_type, count in sorted(counts.items())
        ) or "<li>Nenhum alerta registrado</li>"
        html_body = f"""
        <html><body style="font-family:Arial,sans-serif;color:#1d2a33">
          <h1>Resumo da avaliação</h1>
          <p><strong>Aluno:</strong> {html.escape(student_name)} ({html.escape(student_id)})</p>
          <p><strong>Turma:</strong> {html.escape(session.turma)}<br>
             <strong>Avaliação:</strong> {html.escape(session.assessment)}<br>
             <strong>Estação:</strong> {html.escape(session.station_id)}<br>
             <strong>Início da prova:</strong> {html.escape(started_at_label)}<br>
             <strong>Índice de integridade:</strong> {integrity.score} ({html.escape(integrity.band)})</p>
          <p><strong>{critical_count}</strong> crítico(s) e
             <strong>{warning_count}</strong> alerta(s).</p>
          <h2>Resumo dos alertas</h2><ul>{summary_rows}</ul>
          <p><a href="{html.escape(session_url)}" style="display:inline-block;padding:12px 18px;background:#287681;color:white;text-decoration:none;border-radius:6px">Abrir revisão completa</a></p>
          <h2>Imagens selecionadas</h2>
          <table style="border-collapse:collapse;width:100%" border="1" cellpadding="8">
            <thead><tr><th>Momento</th><th>Alerta</th><th>Severidade</th><th>Imagem</th></tr></thead>
            <tbody>{''.join(rows) or '<tr><td colspan="4">Nenhuma imagem disponível</td></tr>'}</tbody>
          </table>
          {f'<p>Mais {omitted_count} alerta(s) estão disponíveis na revisão completa.</p>' if omitted_count else ''}
        </body></html>
        """
        text_body = "\n".join(
            [
                "Resumo da avaliação",
                f"Aluno: {student_name} ({student_id})",
                f"Turma: {session.turma}",
                f"Avaliação: {session.assessment}",
                f"Estação: {session.station_id}",
                f"Início da prova: {started_at_label}",
                f"Índice de integridade: {integrity.score} ({integrity.band})",
                f"Alertas: {critical_count} críticos, {warning_count} warnings",
                *text_rows,
                f"Alertas adicionais na revisão: {omitted_count}",
                f"Revisão completa: {session_url}",
            ]
        )

        message = EmailMessage()
        message["Subject"] = f"Relatório da avaliação - {session.assessment} - {student_name}"
        message["From"] = report.sender_email
        message["To"] = report.sender_email
        message.set_content(text_body)
        message.add_alternative(html_body, subtype="html")
        html_part = message.get_payload()[-1]
        for cid, filename, image in related_images:
            html_part.add_related(
                image,
                maintype="image",
                subtype="jpeg",
                cid=f"<{cid}>",
                filename=filename,
                disposition="inline",
            )
        return message
