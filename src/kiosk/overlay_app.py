from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime


BG = "#10243e"
PANEL = "#183451"
PANEL_BORDER = "#315574"
TEXT = "#f8f5ee"
MUTED = "#c4d0da"
ACCENT = "#efb64d"
SUCCESS = "#58d68d"
DANGER = "#ff7878"
PENDING = "#efbd62"


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
        with urllib.request.urlopen(request, timeout=5) as response:
            if 200 <= response.status < 300:
                return True, None
            return False, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError) as exc:
        return False, str(exc)


def _controls_mode(stop_url: str, status_url: str) -> int:
    import tkinter as tk

    root = tk.Tk()
    root.title("Controles da avaliação")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.configure(bg=BG)

    frame = tk.Frame(root, bg=BG, padx=10, pady=10)
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

    root.after(0, place_controls)

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
            fg="white",
            bg=BG,
            font=("Helvetica", 28, "bold"),
        )
        title_label.pack(pady=(0, 12))

        message_label = tk.Label(
            container,
            text=message,
            fg=MUTED,
            bg=BG,
            justify="center",
            font=("Helvetica", 21),
        )
        message_label.pack(pady=(0, 20))

        buttons = tk.Frame(container, bg=BG)
        buttons.pack()

        def close(value: bool) -> None:
            result["value"] = value
            dialog.destroy()

        if confirm:
            cancel_button = tk.Button(
                buttons,
                text="Cancelar",
                command=lambda: close(False),
                bg=PANEL,
                fg="white",
                activebackground="#4a5459",
                activeforeground="white",
                relief="flat",
                padx=16,
                pady=8,
                font=("Helvetica", 18, "bold"),
            )
            cancel_button.pack(side="left", padx=(0, 10))

            confirm_button = tk.Button(
                buttons,
                text="Finalizar agora",
                command=lambda: close(True),
                bg="#b84f4f",
                fg="white",
                activebackground="#9b471e",
                activeforeground="white",
                relief="flat",
                padx=16,
                pady=8,
                font=("Helvetica", 18, "bold"),
            )
            confirm_button.pack(side="left")
        else:
            ok_button = tk.Button(
                buttons,
                text="Fechar",
                command=lambda: close(False),
                bg=PANEL,
                fg="white",
                activebackground="#4a5459",
                activeforeground="white",
                relief="flat",
                padx=16,
                pady=8,
                font=("Helvetica", 18, "bold"),
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
        bg=BG,
        compact=True,
        on_update=lambda _payload: root.after_idle(place_controls),
    )
    action_border = tk.Frame(frame, bg="#ff8d8d", padx=2, pady=2)
    action_border.pack(fill="x", pady=(2, 0))
    button = tk.Button(
        action_border,
        text="FINALIZAR AVALIAÇÃO",
        command=on_stop,
        bg="#d95757",
        fg="white",
        activebackground="#bd4141",
        activeforeground="white",
        relief="flat",
        borderwidth=0,
        cursor="hand2",
        padx=24,
        pady=14,
        font=("Helvetica", 20, "bold"),
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
        font=("Helvetica", 17, "bold"),
    ).pack(pady=(0, 8))
    label = tk.Label(parent, text="Preparando a câmera...", fg=MUTED, bg=bg, font=("Helvetica", 22))
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

    box = tk.Frame(parent, bg=PANEL, highlightbackground=PANEL_BORDER, highlightthickness=1, padx=16, pady=12)
    box.pack(pady=(8, 18), fill="x")
    heading = "STATUS DA AVALIAÇÃO" if compact else "VERIFICAÇÃO DO AMBIENTE"
    tk.Label(box, text=heading, fg=ACCENT, bg=PANEL, font=("Helvetica", 16, "bold")).pack(anchor="w")
    timer_label = tk.Label(
        box,
        text="",
        fg=TEXT,
        bg=PANEL,
        font=("Helvetica", 28 if compact else 22, "bold"),
    )
    if compact:
        summary = tk.Label(box, text="Preparando o ambiente", fg=PENDING, bg=PANEL, font=("Helvetica", 20, "bold"))
        summary.pack(anchor="w", pady=(4, 0))
    else:
        summary = None
        rows_frame = tk.Frame(box, bg=PANEL)
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
            fg="#7e8d84",
            bg=PANEL,
            anchor="w",
            font=("Helvetica", 20, "bold"),
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
                    timer_label.pack(anchor="w", pady=(8, 2))
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
    if reason.strip().upper() in {"ABSENCE", "MULTI_FACE"}:
        return "Esta ocorrência foi registrada automaticamente para a equipe responsável."
    return None


def _show_preview_during_block(reason: str) -> bool:
    return reason.strip().upper() not in {"ABSENCE", "GAZE", "MULTI_FACE"}


def _blocked_reason_message(reason: str) -> str:
    return {
        "ABSENCE": "Ausência detectada. Volte para a frente da câmera.",
        "MULTI_FACE": "Mais de um rosto detectado. Apenas o aluno pode permanecer no enquadramento.",
        "GAZE": "Olhar fora do permitido. Olhe para a tela e para a câmera.",
        "BROWSER_EXIT": "O navegador protegido foi encerrado.",
    }.get(reason.strip().upper(), "Olhe para a câmera para retomar a prova.")


def _blocked_mode(reason: str, preview_url: str, status_url: str) -> int:
    import tkinter as tk

    root = tk.Tk()
    root.title("Avaliação pausada")
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.configure(bg="#241a24")

    container = tk.Frame(root, bg="#241a24", padx=48, pady=36)
    container.place(relx=0.5, rely=0.5, anchor="center")

    title = tk.Label(
        container,
        text="Avaliação pausada",
        fg=TEXT,
        bg="#241a24",
        font=("Helvetica", 58, "bold"),
    )
    title.pack(pady=(0, 16))

    subtitle = tk.Label(
        container,
        text=_blocked_reason_message(reason),
        fg="#f2d6d0",
        bg="#241a24",
        wraplength=820,
        justify="center",
        font=("Helvetica", 32),
    )
    subtitle.pack(pady=(0, 16))

    tk.Label(
        container,
        text="A avaliação será retomada automaticamente assim que a situação for regularizada.",
        fg=MUTED,
        bg="#241a24",
        wraplength=800,
        justify="center",
        font=("Helvetica", 24),
    ).pack(pady=(0, 10))

    _add_status_box(container, status_url=status_url, bg="#241a24")

    report_message = _violation_report_message(reason)
    if report_message:
        report_label = tk.Label(
            container,
            text=report_message,
            fg="#e8b6ae",
            bg="#241a24",
            font=("Helvetica", 22),
            wraplength=700,
            justify="center",
        )
        report_label.pack(pady=(0, 16))

    if _show_preview_during_block(reason):
        _add_camera_preview(container, preview_url=preview_url, bg="#241a24", max_size=(360, 270))

    if reason:
        reason_label = tk.Label(
            container,
            text=f"Código do bloqueio: {reason}",
            fg="#9fa9b2",
            bg="#241a24",
            font=("Helvetica", 17),
        )
        reason_label.pack()

    root.mainloop()
    return 0


def _waiting_mode(message: str, preview_url: str, status_url: str) -> int:
    import tkinter as tk

    root = tk.Tk()
    root.title("Preparando avaliação")
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.configure(bg=BG)

    container = tk.Frame(root, bg=BG, padx=48, pady=32)
    container.place(relx=0.5, rely=0.5, anchor="center")

    eyebrow = tk.Label(
        container,
        text="AMBIENTE DE AVALIAÇÃO",
        fg=ACCENT,
        bg=BG,
        font=("Helvetica", 24, "bold"),
    )
    eyebrow.pack(pady=(0, 16))

    title = tk.Label(
        container,
        text=message,
        fg=TEXT,
        bg=BG,
        wraplength=900,
        justify="center",
        font=("Helvetica", 58, "bold"),
    )
    title.pack(pady=(0, 18))

    subtitle = tk.Label(
        container,
        text="Olhe para a câmera e permaneça no centro do enquadramento.\nA próxima etapa aparecerá quando sua identidade for reconhecida.",
        fg=MUTED,
        bg=BG,
        wraplength=780,
        justify="center",
        font=("Helvetica", 32),
    )
    subtitle.pack()

    _add_status_box(container, status_url=status_url, bg=BG)

    _add_camera_preview(container, preview_url=preview_url, bg=BG, max_size=(420, 315))

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

    container = tk.Frame(root, bg=BG, padx=48, pady=24)
    container.place(relx=0.5, rely=0.5, anchor="center")

    tk.Label(
        container,
        text="CONFIRMAÇÃO DO ALUNO",
        fg=ACCENT,
        bg=BG,
        font=("Helvetica", 20, "bold"),
    ).pack(pady=(0, 8))
    tk.Label(
        container,
        text="Confira seus dados antes de começar",
        fg=TEXT,
        bg=BG,
        wraplength=1000,
        justify="center",
        font=("Helvetica", 52, "bold"),
    ).pack(pady=(0, 14))
    identity_card = tk.Frame(container, bg=PANEL, highlightbackground=PANEL_BORDER, highlightthickness=1, padx=24, pady=12)
    identity_card.pack(fill="x", pady=(0, 12))
    tk.Label(
        identity_card,
        text=f"{student_name}\nIdentificação: {student_id}",
        fg=TEXT,
        bg=PANEL,
        justify="center",
        font=("Helvetica", 30, "bold"),
    ).pack()

    _add_camera_preview(
        container,
        preview_url=preview_url,
        bg=BG,
        max_size=(240, 180),
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
        bg=BG,
        wraplength=1050,
        justify="left",
        font=("Helvetica", 22),
    ).pack(pady=(0, 8))

    confirmed = tk.BooleanVar(value=False)
    confirm_button: tk.Button
    remaining_label = tk.Label(container, fg=MUTED, bg=BG, font=("Helvetica", 18))

    checks_ready = False

    def set_confirm_enabled() -> None:
        confirm_button.configure(state="normal" if confirmed.get() and checks_ready else "disabled")

    def update_release_state(payload: dict) -> None:
        nonlocal checks_ready
        checks_ready = bool(payload.get("ready"))
        set_confirm_enabled()

    status_box = tk.Frame(container, bg=BG)
    status_box.pack(fill="x")
    _add_status_box(
        status_box,
        status_url=status_url,
        bg=BG,
        on_update=update_release_state,
    )

    acknowledgement = tk.Frame(
        container,
        bg=PANEL,
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
        bg=PANEL,
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
        bg=PANEL,
        justify="left",
        wraplength=900,
        font=("Helvetica", 22, "bold"),
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
            fill=SUCCESS if confirmed.get() else BG,
            outline=SUCCESS if confirmed.get() else MUTED,
            width=3,
        )
        if confirmed.get():
            checkbox.create_line(13, 27, 22, 36, 40, 16, fill=BG, width=5, capstyle="round", joinstyle="round")

    def toggle_acknowledgement(_event=None) -> None:
        confirmed.set(not confirmed.get())
        draw_checkbox()
        set_confirm_enabled()

    for widget in (acknowledgement, checkbox, acknowledgement_text):
        widget.bind("<Button-1>", toggle_acknowledgement)
    draw_checkbox()

    buttons = tk.Frame(container, bg=BG)
    buttons.pack()

    finished = False

    def respond(url: str) -> None:
        nonlocal finished
        if finished:
            return
        finished = True
        _send_stop_request(url)
        root.destroy()

    tk.Button(
        buttons,
        text="Não são meus dados",
        command=lambda: respond(cancel_url),
        bg=PANEL,
        fg="white",
        activebackground="#4a5459",
        activeforeground="white",
        relief="flat",
        padx=24,
        pady=12,
        font=("Helvetica", 22, "bold"),
    ).pack(side="left", padx=(0, 12))
    confirm_button = tk.Button(
        buttons,
        text="Começar avaliação",
        command=lambda: respond(confirm_url),
        state="disabled",
        bg="#24734b",
        fg="white",
        activebackground="#1f633e",
        activeforeground="white",
        relief="flat",
        padx=24,
        pady=12,
        font=("Helvetica", 22, "bold"),
    )
    confirm_button.pack(side="left")
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
    parser.add_argument("--guard-height", type=int, default=32)
    args = parser.parse_args(argv)

    os.environ.setdefault("DISPLAY", os.environ.get("DISPLAY", ":0"))

    if args.mode == "controls":
        return _controls_mode(args.stop_url, args.status_url)
    if args.mode == "waiting":
        return _waiting_mode(args.message, args.preview_url, args.status_url)
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
    return _blocked_mode(args.reason, args.preview_url, args.status_url)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
