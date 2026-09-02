from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from datetime import datetime


# Paleta acadêmica: azul institucional identifica a plataforma; as cores de
# estado nunca são usadas como decoração, só para comunicar resultado/risco.
BG = "#F8FAFC"
PANEL = "#FFFFFF"
PANEL_BORDER = "#CBD5E1"
TEXT = "#0F172A"
MUTED = "#475569"
PRIMARY = "#1E3A5F"
ACTION = "#2563EB"
ACTION_ACTIVE = "#1D4ED8"
ACCENT = "#B45309"
SUCCESS = "#15803D"
DANGER = "#DC2626"
DANGER_ACTIVE = "#B91C1C"
DANGER_BG = "#FFF7F7"
PENDING = "#B45309"
DISABLED_BG = "#E2E8F0"
DISABLED_TEXT = "#64748B"
FONT = "DejaVu Sans"


def _button(parent, *, text: str, command, variant: str = "primary", **kwargs):
    import tkinter as tk

    font = kwargs.pop("font", (FONT, 18, "bold"))

    styles = {
        "primary": {
            "bg": ACTION,
            "fg": "#FFFFFF",
            "activebackground": ACTION_ACTIVE,
            "activeforeground": "#FFFFFF",
            "highlightbackground": ACTION_ACTIVE,
        },
        "secondary": {
            "bg": PANEL,
            "fg": PRIMARY,
            "activebackground": "#E9EEF5",
            "activeforeground": PRIMARY,
            "highlightbackground": PRIMARY,
        },
        "danger": {
            "bg": DANGER,
            "fg": "#FFFFFF",
            "activebackground": DANGER_ACTIVE,
            "activeforeground": "#FFFFFF",
            "highlightbackground": DANGER_ACTIVE,
        },
    }
    return tk.Button(
        parent,
        text=text,
        command=command,
        relief="solid",
        borderwidth=1,
        highlightthickness=1,
        cursor="hand2",
        font=font,
        **styles[variant],
        **kwargs,
    )


def _status_summary(payload: dict) -> tuple[str, str]:
    checks = payload.get("checks", [])
    if payload.get("state") == "BLOCKED":
        return "Avaliação pausada", "fail"
    if any(check.get("state") == "fail" for check in checks):
        return "Atenção necessária", "fail"
    if checks and all(check.get("state") == "ok" for check in checks):
        return "Monitoramento ativo", "ok"
    return "Preparando o ambiente", "pending"


def _format_remaining_time(seconds: int | None) -> str | None:
    if seconds is None:
        return None
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _format_clock_time(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%H:%M:%S")


def _send_stop_request(stop_url: str) -> tuple[bool, str | None]:
    request = urllib.request.Request(
        stop_url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            if 200 <= response.status < 300:
                return True, None
            return False, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError) as exc:
        return False, str(exc)


def _send_start_request(start_url: str) -> tuple[bool, str | None]:
    request = urllib.request.Request(
        start_url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            if 200 <= response.status < 300:
                return True, None
            return False, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read())
            return False, str(payload.get("detail") or f"HTTP {exc.code}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False, f"HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError) as exc:
        return False, str(exc)


def _controls_mode(stop_url: str, status_url: str) -> int:
    import tkinter as tk

    root = tk.Tk()
    root.title("Controles da avaliação")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    try:
        root.attributes("-type", "dock")
    except tk.TclError:
        pass
    root.configure(bg=PANEL)

    frame = tk.Frame(root, bg=PANEL, padx=8, pady=8)
    frame.pack()

    def place_controls() -> None:
        root.update_idletasks()
        button_width = max(root.winfo_reqwidth(), root.winfo_width())
        button_height = max(root.winfo_reqheight(), root.winfo_height())
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        inset = 4
        x = max(screen_width - button_width - inset, 0)
        y = max(screen_height - button_height - inset, 0)
        root.geometry(f"{button_width}x{button_height}+{x}+{y}")

    ewmh_applied = False

    def keep_controls_above() -> None:
        nonlocal ewmh_applied
        if not root.winfo_exists():
            return
        root.attributes("-topmost", True)
        root.lift()
        if not ewmh_applied:
            window_id = f"0x{root.winfo_id():x}"
            env = os.environ.copy()
            try:
                result = subprocess.run(
                    ["wmctrl", "-i", "-r", window_id, "-b", "add,above,sticky"],
                    env=env,
                    capture_output=True,
                    timeout=1,
                    check=False,
                )
                ewmh_applied = result.returncode == 0
            except (OSError, subprocess.SubprocessError):
                pass
        root.after(250, keep_controls_above)

    root.after(0, place_controls)
    root.after(0, keep_controls_above)

    def show_dialog(title: str, message: str, *, confirm: bool = False) -> bool:
        dialog = tk.Toplevel(root)
        dialog.title(title)
        dialog.transient(root)
        dialog.attributes("-topmost", True)
        dialog.configure(bg=BG)
        dialog.resizable(False, False)

        result = {"value": False}

        container = tk.Frame(dialog, bg=BG, padx=28, pady=24)
        container.pack()

        title_label = tk.Label(
            container,
            text=title,
            fg=TEXT,
            bg=BG,
            font=(FONT, 28, "bold"),
        )
        title_label.pack(pady=(0, 12))

        message_label = tk.Label(
            container,
            text=message,
            fg=MUTED,
            bg=BG,
            justify="center",
            font=(FONT, 18),
        )
        message_label.pack(pady=(0, 20))

        buttons = tk.Frame(container, bg=BG)
        buttons.pack()

        def close(value: bool) -> None:
            result["value"] = value
            dialog.destroy()

        if confirm:
            cancel_button = _button(
                buttons,
                text="Cancelar",
                command=lambda: close(False),
                padx=16,
                pady=8,
                variant="secondary",
            )
            cancel_button.pack(side="left", padx=(0, 10))

            confirm_button = _button(
                buttons,
                text="Finalizar agora",
                command=lambda: close(True),
                padx=16,
                pady=8,
                variant="danger",
            )
            confirm_button.pack(side="left")
        else:
            ok_button = _button(
                buttons,
                text="Fechar",
                command=lambda: close(False),
                padx=16,
                pady=8,
                variant="secondary",
            )
            ok_button.pack()

        dialog.update_idletasks()
        dialog_width = dialog.winfo_reqwidth()
        dialog_height = dialog.winfo_reqheight()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        dialog_x = max((screen_width - dialog_width) // 2, 0)
        dialog_y = max((screen_height - dialog_height) // 2, 0)
        dialog.geometry(f"+{dialog_x}+{dialog_y}")
        dialog.grab_set()
        dialog.focus_force()
        dialog.protocol("WM_DELETE_WINDOW", lambda: close(False))
        root.wait_window(dialog)
        return result["value"]

    def on_stop() -> None:
        confirmed = show_dialog(
            "Finalizar avaliação",
            "Seu progresso será finalizado e enviado. Deseja encerrar a avaliação agora?",
            confirm=True,
        )
        if not confirmed:
            return
        ok, error = _send_stop_request(stop_url)
        if ok:
            root.destroy()
        else:
            show_dialog(
                "Não foi possível finalizar",
                "A avaliação continua ativa. Tente novamente em alguns instantes."
                + (f"\n\nDetalhe: {error}" if error else ""),
            )

    _add_status_box(
        frame,
        status_url=status_url,
        bg=PANEL,
        compact=True,
        on_update=lambda _payload: root.after_idle(place_controls),
    )
    button = _button(
        frame,
        text="FINALIZAR AVALIAÇÃO",
        command=on_stop,
        padx=12,
        pady=9,
        font=(FONT, 12, "bold"),
        variant="danger",
    )
    button.pack(fill="x")
    root.mainloop()
    return 0


def _add_camera_preview(
    parent,
    *,
    preview_url: str,
    bg: str,
    max_size: tuple[int, int] = (480, 360),
) -> None:
    import tkinter as tk
    from io import BytesIO

    from PIL import Image, ImageTk

    tk.Label(
        parent,
        text="ENQUADRAMENTO DA CÂMERA",
        fg=ACCENT,
        bg=bg,
        font=(FONT, 15, "bold"),
    ).pack(pady=(0, 8))
    label = tk.Label(parent, text="Preparando a câmera...", fg=MUTED, bg=bg, font=(FONT, 18))
    label.pack(pady=(0, 20))

    def refresh() -> None:
        try:
            with urllib.request.urlopen(preview_url, timeout=0.5) as response:
                image = Image.open(BytesIO(response.read())).convert("RGB")
            image.thumbnail(max_size)
            photo = ImageTk.PhotoImage(image)
            label.configure(image=photo, text="", relief="solid", bd=2, highlightbackground=PANEL_BORDER)
            label.image = photo
        except (OSError, urllib.error.URLError, TimeoutError):
            label.configure(image="", text="Não foi possível exibir a câmera. Aguarde um instante.", relief="flat", bd=0)
            label.image = None
        parent.after(500, refresh)

    parent.after(0, refresh)


def _add_status_box(
    parent,
    *,
    status_url: str,
    bg: str,
    compact: bool = False,
    on_update=None,
) -> None:
    import tkinter as tk

    box_bg = "#E9EEF5"
    box = tk.Frame(
        parent,
        bg=box_bg,
        highlightbackground=PANEL_BORDER,
        highlightthickness=1,
        padx=10 if compact else 16,
        pady=6 if compact else 12,
    )
    box.pack(pady=(4, 8) if compact else (8, 18), fill="x")
    heading = "STATUS DA AVALIAÇÃO" if compact else "VERIFICAÇÃO DO AMBIENTE"
    tk.Label(
        box,
        text=heading,
        fg=ACCENT,
        bg=box_bg,
        font=(FONT, 11 if compact else 14, "bold"),
    ).pack(anchor="w")
    timer_label = tk.Label(
        box,
        text="",
        fg=TEXT,
        bg=box_bg,
        font=(FONT, 16 if compact else 19, "bold"),
    )
    if compact:
        summary = tk.Label(
            box,
            text="Preparando o ambiente",
            fg=PENDING,
            bg=box_bg,
            font=(FONT, 12, "bold"),
        )
        summary.pack(anchor="w", pady=(2, 0))
    else:
        summary = None
        rows_frame = tk.Frame(box, bg=box_bg)
        rows_frame.pack(fill="x", pady=(6, 0))
    rows: dict[str, tk.Label] = {}
    for index, (key, label) in enumerate((
        ("session", "Sessão liberada"),
        ("webcam", "Webcam detectada"),
        ("student", "Aluno identificado"),
        ("presence", "Aluno presente"),
        ("faces", "Rosto único"),
        ("gaze", "Olhar dentro do permitido"),
        ("chromium", "Chromium protegido"),
    )):
        if compact:
            break
        row = tk.Label(
            rows_frame,
            text=f"●  {label}",
            fg=MUTED,
            bg=box_bg,
            anchor="w",
            font=(FONT, 16, "bold"),
        )
        row.grid(row=index // 2, column=index % 2, sticky="w", padx=(0, 28), pady=2)
        rows[key] = row

    colors = {"ok": SUCCESS, "fail": DANGER, "pending": PENDING}

    def refresh() -> None:
        try:
            with urllib.request.urlopen(status_url, timeout=0.5) as response:
                payload = json.loads(response.read())
            if on_update is not None:
                on_update(payload)
            if summary is not None:
                text, state = _status_summary(payload)
                summary.configure(text=f"●  {text}", fg=colors[state])
            remaining = _format_remaining_time(payload.get("seconds_remaining"))
            if remaining is None:
                timer_label.pack_forget()
            else:
                timer_label.configure(
                    text=f"Tempo restante  {remaining}    |    Horário atual  {_format_clock_time()}"
                )
                if not timer_label.winfo_manager():
                    timer_label.pack(anchor="w", pady=(4, 1))
            for check in payload.get("checks", []):
                row = rows.get(check.get("key"))
                if row is not None:
                    row.configure(text=f"●  {check.get('label')}", fg=colors.get(check.get("state"), "#7e8d84"))
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if summary is not None:
                summary.configure(text="●  Reconectando ao monitoramento", fg=DANGER)
            elif "webcam" in rows:
                rows["webcam"].configure(text="●  Verificando câmera...", fg=DANGER)
        parent.after(500, refresh)

    parent.after(0, refresh)


def _violation_report_message(reason: str) -> str | None:
    if reason.strip().upper() in {"ABSENCE", "MULTI_FACE", "DIFFERENT_USER"}:
        return "Esta ocorrência foi registrada automaticamente para a equipe responsável."
    return None


def _show_preview_during_block(reason: str) -> bool:
    return reason.strip().upper() not in {"ABSENCE", "GAZE", "MULTI_FACE", "DIFFERENT_USER"}


def _blocked_reason_message(reason: str) -> str:
    return {
        "ABSENCE": "Ausência detectada. Volte para a frente da câmera.",
        "MULTI_FACE": "Mais de um rosto detectado. Apenas o aluno pode permanecer no enquadramento.",
        "GAZE": "Olhar fora do permitido. Olhe para a tela e para a câmera.",
        "BROWSER_EXIT": "O navegador protegido foi encerrado.",
        "DIFFERENT_USER": "Usuário diferente detectado. O aluno autenticado deve retornar.",
    }.get(reason.strip().upper(), "Olhe para a câmera para retomar a prova.")


def _blocked_mode(
    reason: str,
    preview_url: str,
    status_url: str,
    student_id: str,
    stop_url: str,
    timeout_sec: float,
) -> int:
    import tkinter as tk

    root = tk.Tk()
    root.title("Avaliação pausada")
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.configure(bg=DANGER_BG)

    container = tk.Frame(
        root,
        bg=PANEL,
        highlightbackground="#F1B5B5",
        highlightthickness=2,
        padx=48,
        pady=36,
    )
    container.place(relx=0.5, rely=0.5, anchor="center")

    tk.Label(
        container,
        text="AVALIAÇÃO PAUSADA",
        fg=DANGER,
        bg=PANEL,
        font=(FONT, 17, "bold"),
    ).pack(pady=(0, 6))
    tk.Label(
        container,
        text="Sua atenção é necessária",
        fg=TEXT,
        bg=PANEL,
        font=(FONT, 42, "bold"),
    ).pack(pady=(0, 18))

    if student_id:
        student_card = tk.Frame(
            container,
            bg="#F8FAFC",
            highlightbackground=PANEL_BORDER,
            highlightthickness=1,
            padx=24,
            pady=12,
        )
        student_card.pack(fill="x", pady=(0, 16))
        tk.Label(
            student_card,
            text="USUÁRIO DO ALUNO",
            fg=PRIMARY,
            bg="#F8FAFC",
            font=(FONT, 14, "bold"),
        ).pack()
        tk.Label(
            student_card,
            text=student_id,
            fg=TEXT,
            bg="#F8FAFC",
            font=(FONT, 26, "bold"),
        ).pack(pady=(4, 0))

    subtitle = tk.Label(
        container,
        text=_blocked_reason_message(reason),
        fg=TEXT,
        bg=PANEL,
        wraplength=820,
        justify="center",
        font=(FONT, 25, "bold"),
    )
    subtitle.pack(pady=(0, 16))

    tk.Label(
        container,
        text="Corrija a situação indicada para que a avaliação seja retomada automaticamente.",
        fg=MUTED,
        bg=PANEL,
        wraplength=800,
        justify="center",
        font=(FONT, 18),
    ).pack(pady=(0, 10))

    _add_status_box(container, status_url=status_url, bg=PANEL)

    report_message = _violation_report_message(reason)
    if report_message:
        report_label = tk.Label(
            container,
            text=report_message,
            fg="#9F2D2D",
            bg=PANEL,
            font=(FONT, 16),
            wraplength=700,
            justify="center",
        )
        report_label.pack(pady=(0, 16))

    if _show_preview_during_block(reason):
        _add_camera_preview(container, preview_url=preview_url, bg=PANEL, max_size=(360, 270))

    if reason:
        reason_label = tk.Label(
            container,
            text=f"Código do bloqueio: {reason}",
            fg=MUTED,
            bg=PANEL,
            font=(FONT, 14),
        )
        reason_label.pack()

    cancellation_feedback = tk.Label(
        container,
        text="",
        fg=DANGER,
        bg=PANEL,
        font=(FONT, 15, "bold"),
    )
    cancellation_feedback.pack(pady=(12, 0))

    def request_cancellation() -> None:
        dialog = tk.Toplevel(root)
        dialog.title("Cancelar avaliação")
        dialog.transient(root)
        dialog.attributes("-topmost", True)
        dialog.configure(bg=BG)
        dialog.resizable(False, False)

        result = {"confirmed": False}
        content = tk.Frame(dialog, bg=BG, padx=36, pady=30)
        content.pack()
        tk.Label(
            content,
            text="Cancelar esta avaliação?",
            fg=TEXT,
            bg=BG,
            font=(FONT, 28, "bold"),
        ).pack(pady=(0, 12))
        tk.Label(
            content,
            text="A sessão será encerrada e os registros coletados serão enviados normalmente.",
            fg=MUTED,
            bg=BG,
            wraplength=680,
            justify="center",
            font=(FONT, 18),
        ).pack(pady=(0, 24))
        actions = tk.Frame(content, bg=BG)
        actions.pack()

        def close(confirmed: bool) -> None:
            result["confirmed"] = confirmed
            dialog.destroy()

        _button(
            actions,
            text="Voltar para a avaliação",
            command=lambda: close(False),
            padx=22,
            pady=12,
            variant="secondary",
        ).pack(side="left", padx=(0, 12))
        _button(
            actions,
            text="Sim, cancelar avaliação",
            command=lambda: close(True),
            padx=22,
            pady=12,
            variant="danger",
        ).pack(side="left")

        dialog.update_idletasks()
        x = max((root.winfo_screenwidth() - dialog.winfo_reqwidth()) // 2, 0)
        y = max((root.winfo_screenheight() - dialog.winfo_reqheight()) // 2, 0)
        dialog.geometry(f"+{x}+{y}")
        dialog.grab_set()
        dialog.focus_force()
        dialog.protocol("WM_DELETE_WINDOW", lambda: close(False))
        root.wait_window(dialog)

        if not result["confirmed"]:
            return
        cancel_button.configure(state="disabled", text="Cancelando avaliação...")
        ok, error = _send_stop_request(stop_url)
        if ok:
            root.destroy()
            return
        cancel_button.configure(state="normal", text="Cancelar avaliação")
        cancellation_feedback.configure(
            text="Não foi possível cancelar. Tente novamente."
            + (f" Detalhe: {error}" if error else "")
        )

    cancel_button = _button(
        container,
        text="Cancelar avaliação",
        command=request_cancellation,
        padx=28,
        pady=14,
        variant="danger",
    )
    cancel_button.pack(pady=(12, 0))

    timeout_label = tk.Label(
        container,
        fg=DANGER,
        bg=PANEL,
        font=(FONT, 15, "bold"),
    )
    timeout_label.pack(pady=(14, 0))
    deadline = root.tk.call("clock", "seconds") + max(1, int(timeout_sec))

    def update_timeout() -> None:
        remaining = max(0, deadline - root.tk.call("clock", "seconds"))
        if remaining <= 0:
            timeout_label.configure(text="Tempo esgotado. Encerrando a avaliação...")
            cancel_button.configure(state="disabled")
            return
        timeout_label.configure(text=f"A avaliação será cancelada em {remaining}s se a situação não for corrigida.")
        root.after(250, update_timeout)

    root.after(0, update_timeout)

    root.mainloop()
    return 0


def _waiting_mode(message: str, start_url: str) -> int:
    import tkinter as tk

    root = tk.Tk()
    root.title("Preparando avaliação")
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.configure(bg=BG)

    container = tk.Frame(
        root,
        bg=PANEL,
        highlightbackground=PANEL_BORDER,
        highlightthickness=1,
        padx=64,
        pady=52,
    )
    container.place(relx=0.5, rely=0.5, anchor="center")

    if message:
        tk.Label(
            container,
            text=message,
            fg=TEXT,
            bg=PANEL,
            wraplength=900,
            justify="center",
            font=(FONT, 38, "bold"),
        ).pack()
    else:
        tk.Label(
            container,
            text="AVALIAÇÃO PRESENCIAL",
            fg=PRIMARY,
            bg=PANEL,
            font=(FONT, 15, "bold"),
        ).pack(pady=(0, 12))
        tk.Label(
            container,
            text="Pronto para iniciar?",
            fg=TEXT,
            bg=PANEL,
            font=(FONT, 42, "bold"),
        ).pack(pady=(0, 10))
        tk.Label(
            container,
            text="Ao continuar, a câmera será ativada para confirmar sua identidade.",
            fg=MUTED,
            bg=PANEL,
            wraplength=620,
            justify="center",
            font=(FONT, 18),
        ).pack(pady=(0, 28))
        feedback = tk.Label(
            container,
            text="",
            fg=DANGER,
            bg=PANEL,
            wraplength=800,
            justify="center",
            font=(FONT, 16, "bold"),
        )

        def restore_button(error: str | None) -> None:
            if not root.winfo_exists():
                return
            start_button.configure(state="normal", text="Iniciar prova", bg=ACTION, fg="#FFFFFF")
            feedback.configure(text=error or "Não foi possível iniciar. Tente novamente.")
            feedback.pack(pady=(18, 0))

        def send_start() -> None:
            ok, error = _send_start_request(start_url)
            if not ok:
                try:
                    root.after(0, restore_button, error)
                except tk.TclError:
                    pass

        def begin_identification() -> None:
            start_button.configure(
                state="disabled",
                text="Verificando identidade...",
                bg=DISABLED_BG,
                fg=DISABLED_TEXT,
            )
            feedback.pack_forget()
            threading.Thread(target=send_start, name="request-exam-start", daemon=True).start()

        start_button = _button(
            container,
            text="Iniciar prova",
            command=begin_identification,
            padx=54,
            pady=18,
            font=(FONT, 26, "bold"),
            variant="primary",
        )
        start_button.pack()

    root.mainloop()
    return 0


def _confirmation_mode(
    student_id: str,
    student_name: str,
    timeout_sec: float,
    confirm_url: str,
    cancel_url: str,
    preview_url: str,
    status_url: str,
) -> int:
    import tkinter as tk

    root = tk.Tk()
    root.title("Confirmação da avaliação")
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.configure(bg=BG)

    container = tk.Frame(
        root,
        bg=PANEL,
        highlightbackground=PANEL_BORDER,
        highlightthickness=1,
        padx=42,
        pady=28,
    )
    container.place(relx=0.5, rely=0.5, anchor="center")

    tk.Label(
        container,
        text="IDENTIDADE CONFIRMADA",
        fg=SUCCESS,
        bg=PANEL,
        font=(FONT, 15, "bold"),
    ).pack(pady=(0, 8))
    tk.Label(
        container,
        text="Confira seus dados antes de iniciar",
        fg=TEXT,
        bg=PANEL,
        wraplength=1000,
        justify="center",
        font=(FONT, 36, "bold"),
    ).pack(pady=(0, 14))
    identity_card = tk.Frame(
        container,
        bg="#E9EEF5",
        highlightbackground="#B7C9DE",
        highlightthickness=1,
        padx=24,
        pady=12,
    )
    identity_card.pack(fill="x", pady=(0, 12))
    tk.Label(
        identity_card,
        text=f"{student_name}\nIdentificação: {student_id}",
        fg=TEXT,
        bg="#E9EEF5",
        justify="center",
        font=(FONT, 24, "bold"),
    ).pack()

    _add_camera_preview(
        container,
        preview_url=preview_url,
        bg=PANEL,
        max_size=(220, 165),
    )

    notice = (
        "Para garantir uma avaliação justa, câmera, áudio ambiente, tela e atividade do teclado "
        "serão monitorados durante a realização.\n\n"
        "REGRAS DA AVALIAÇÃO\n"
        "1. Permaneça visível durante toda a avaliação.\n"
        "2. Realize a atividade individualmente.\n"
        "3. Não utilize celular ou materiais não autorizados."
    )
    tk.Label(
        container,
        text=notice,
        fg=MUTED,
        bg=PANEL,
        wraplength=1050,
        justify="left",
        font=(FONT, 16),
    ).pack(pady=(0, 8))

    confirmed = tk.BooleanVar(value=False)
    confirm_button: tk.Button
    remaining_label = tk.Label(container, fg=MUTED, bg=PANEL, font=(FONT, 15))

    checks_ready = False

    def set_confirm_enabled() -> None:
        enabled = confirmed.get() and checks_ready
        confirm_button.configure(
            state="normal" if enabled else "disabled",
            bg=ACTION if enabled else DISABLED_BG,
            fg="#FFFFFF" if enabled else DISABLED_TEXT,
        )

    def update_release_state(payload: dict) -> None:
        nonlocal checks_ready
        checks_ready = bool(payload.get("ready"))
        set_confirm_enabled()

    status_box = tk.Frame(container, bg=PANEL)
    status_box.pack(fill="x")
    _add_status_box(
        status_box,
        status_url=status_url,
        bg=PANEL,
        on_update=update_release_state,
    )

    acknowledgement = tk.Frame(
        container,
        bg="#F8FAFC",
        highlightbackground=PANEL_BORDER,
        highlightthickness=1,
        padx=18,
        pady=14,
        cursor="hand2",
    )
    acknowledgement.pack(fill="x", pady=(0, 18))
    checkbox = tk.Canvas(
        acknowledgement,
        width=52,
        height=52,
        bg="#F8FAFC",
        highlightthickness=0,
        cursor="hand2",
    )
    checkbox.pack(side="left", padx=(0, 16))
    acknowledgement_text = tk.Label(
        acknowledgement,
        text=(
            "Confirmo que meus dados estão corretos, li as regras da avaliação "
            "e estou ciente de que devo cumpri-las."
        ),
        fg=TEXT,
        bg="#F8FAFC",
        justify="left",
        wraplength=900,
        font=(FONT, 16, "bold"),
        cursor="hand2",
    )
    acknowledgement_text.pack(side="left", fill="x", expand=True)

    def draw_checkbox() -> None:
        checkbox.delete("all")
        checkbox.create_rectangle(
            4,
            4,
            48,
            48,
            fill=SUCCESS if confirmed.get() else PANEL,
            outline=SUCCESS if confirmed.get() else PRIMARY,
            width=3,
        )
        if confirmed.get():
            checkbox.create_line(13, 27, 22, 36, 40, 16, fill="#FFFFFF", width=5, capstyle="round", joinstyle="round")

    def toggle_acknowledgement(_event=None) -> None:
        confirmed.set(not confirmed.get())
        draw_checkbox()
        set_confirm_enabled()

    for widget in (acknowledgement, checkbox, acknowledgement_text):
        widget.bind("<Button-1>", toggle_acknowledgement)
    draw_checkbox()

    buttons = tk.Frame(container, bg=PANEL)
    buttons.pack()

    finished = False

    def respond(url: str) -> None:
        nonlocal finished
        if finished:
            return
        finished = True
        _send_stop_request(url)
        root.destroy()

    _button(
        buttons,
        text="Não são meus dados",
        command=lambda: respond(cancel_url),
        padx=24,
        pady=12,
        variant="secondary",
    ).pack(side="left", padx=(0, 12))
    confirm_button = _button(
        buttons,
        text="Confirmar e iniciar avaliação",
        command=lambda: respond(confirm_url),
        state="disabled",
        padx=24,
        pady=12,
        variant="primary",
    )
    confirm_button.pack(side="left")
    set_confirm_enabled()
    remaining_label.pack(pady=(18, 0))

    deadline = root.tk.call("clock", "seconds") + max(1, int(timeout_sec))

    def update_countdown() -> None:
        remaining = max(0, deadline - root.tk.call("clock", "seconds"))
        remaining_label.configure(text=f"Esta confirmação expira em {remaining}s")
        if remaining <= 0:
            respond(cancel_url)
            return
        root.after(250, update_countdown)

    root.protocol("WM_DELETE_WINDOW", lambda: respond(cancel_url))
    root.after(0, update_countdown)
    root.mainloop()
    return 0


def _guard_mode(height: int) -> int:
    import tkinter as tk

    root = tk.Tk()
    root.title("Proctor Guard")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.configure(bg="#0b0f12", cursor="none")

    screen_width = root.winfo_screenwidth()
    guard_height = max(1, height)
    root.geometry(f"{screen_width}x{guard_height}+0+0")

    root.mainloop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Overlay da estação de prova")
    parser.add_argument("--mode", choices=["controls", "blocked", "waiting", "confirmation", "guard"], required=True)
    parser.add_argument("--stop-url", default="http://127.0.0.1:8000/session/stop")
    parser.add_argument("--reason", default="")
    parser.add_argument("--message", default="Vamos preparar sua avaliação")
    parser.add_argument("--preview-url", default="http://127.0.0.1:8000/camera-preview.jpg")
    parser.add_argument("--status-url", default="http://127.0.0.1:8000/exam-checks")
    parser.add_argument("--student-id", default="")
    parser.add_argument("--student-name", default="")
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--confirm-url", default="http://127.0.0.1:8000/pre-exam/confirmation/accept")
    parser.add_argument("--cancel-url", default="http://127.0.0.1:8000/pre-exam/confirmation/cancel")
    parser.add_argument("--start-url", default="http://127.0.0.1:8000/pre-exam/start")
    parser.add_argument("--guard-height", type=int, default=32)
    args = parser.parse_args(argv)

    os.environ.setdefault("DISPLAY", os.environ.get("DISPLAY", ":0"))

    if args.mode == "controls":
        return _controls_mode(args.stop_url, args.status_url)
    if args.mode == "waiting":
        return _waiting_mode(args.message, args.start_url)
    if args.mode == "confirmation":
        return _confirmation_mode(
            args.student_id,
            args.student_name,
            args.timeout_sec,
            args.confirm_url,
            args.cancel_url,
            args.preview_url,
            args.status_url,
        )
    if args.mode == "guard":
        return _guard_mode(args.guard_height)
    return _blocked_mode(
        args.reason,
        args.preview_url,
        args.status_url,
        args.student_id,
        args.stop_url,
        args.timeout_sec,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
