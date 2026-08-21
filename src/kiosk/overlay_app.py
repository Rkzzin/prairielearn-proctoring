from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request


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


def _controls_mode(stop_url: str) -> int:
    import tkinter as tk

    root = tk.Tk()
    root.title("Proctor Controls")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.configure(bg="#1d2a33")

    frame = tk.Frame(root, bg="#1d2a33", padx=10, pady=10)
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
        dialog.configure(bg="#1d1f21")
        dialog.resizable(False, False)

        result = {"value": False}

        container = tk.Frame(dialog, bg="#1d1f21", padx=28, pady=24)
        container.pack()

        title_label = tk.Label(
            container,
            text=title,
            fg="white",
            bg="#1d1f21",
            font=("Helvetica", 16, "bold"),
        )
        title_label.pack(pady=(0, 12))

        message_label = tk.Label(
            container,
            text=message,
            fg="#f0e4d3",
            bg="#1d1f21",
            justify="center",
            font=("Helvetica", 12),
        )
        message_label.pack(pady=(0, 20))

        buttons = tk.Frame(container, bg="#1d1f21")
        buttons.pack()

        def close(value: bool) -> None:
            result["value"] = value
            dialog.destroy()

        if confirm:
            cancel_button = tk.Button(
                buttons,
                text="Cancelar",
                command=lambda: close(False),
                bg="#3b4348",
                fg="white",
                activebackground="#4a5459",
                activeforeground="white",
                relief="flat",
                padx=16,
                pady=8,
                font=("Helvetica", 11, "bold"),
            )
            cancel_button.pack(side="left", padx=(0, 10))

            confirm_button = tk.Button(
                buttons,
                text="Encerrar agora",
                command=lambda: close(True),
                bg="#bb5a2a",
                fg="white",
                activebackground="#9b471e",
                activeforeground="white",
                relief="flat",
                padx=16,
                pady=8,
                font=("Helvetica", 11, "bold"),
            )
            confirm_button.pack(side="left")
        else:
            ok_button = tk.Button(
                buttons,
                text="Fechar",
                command=lambda: close(False),
                bg="#3b4348",
                fg="white",
                activebackground="#4a5459",
                activeforeground="white",
                relief="flat",
                padx=16,
                pady=8,
                font=("Helvetica", 11, "bold"),
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
            "Encerrar prova",
            "Deseja realmente encerrar a prova?",
            confirm=True,
        )
        if not confirmed:
            return
        ok, error = _send_stop_request(stop_url)
        if ok:
            root.destroy()
        else:
            show_dialog(
                "Falha ao encerrar",
                "Não foi possível encerrar a prova pela API local."
                + (f"\n\nDetalhe: {error}" if error else ""),
            )

    button = tk.Button(
        frame,
        text="Encerrar prova",
        command=on_stop,
        bg="#bb5a2a",
        fg="white",
        activebackground="#9b471e",
        activeforeground="white",
        relief="flat",
        padx=18,
        pady=10,
        font=("Helvetica", 12, "bold"),
    )
    button.pack()
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

    label = tk.Label(parent, text="Carregando câmera...", fg="#d7e4d5", bg=bg, font=("Helvetica", 13))
    label.pack(pady=(0, 20))

    def refresh() -> None:
        try:
            with urllib.request.urlopen(preview_url, timeout=0.5) as response:
                image = Image.open(BytesIO(response.read())).convert("RGB")
            image.thumbnail(max_size)
            photo = ImageTk.PhotoImage(image)
            label.configure(image=photo, text="", relief="solid", bd=2)
            label.image = photo
        except (OSError, urllib.error.URLError, TimeoutError):
            label.configure(image="", text="Câmera indisponível", relief="flat", bd=0)
            label.image = None
        parent.after(150, refresh)

    parent.after(0, refresh)


def _violation_report_message(reason: str) -> str | None:
    if reason.strip().upper() in {"ABSENCE", "MULTI_FACE"}:
        return "Essa violação foi registrada e reportada."
    return None


def _show_preview_during_block(reason: str) -> bool:
    return reason.strip().upper() not in {"ABSENCE", "GAZE", "MULTI_FACE"}


def _blocked_mode(reason: str, preview_url: str) -> int:
    import tkinter as tk

    root = tk.Tk()
    root.title("Sessão bloqueada")
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.configure(bg="#1d1f21")

    container = tk.Frame(root, bg="#1d1f21")
    container.place(relx=0.5, rely=0.5, anchor="center")

    title = tk.Label(
        container,
        text="Sessão bloqueada",
        fg="white",
        bg="#1d1f21",
        font=("Helvetica", 28, "bold"),
    )
    title.pack(pady=(0, 16))

    subtitle = tk.Label(
        container,
        text="Olhe para a câmera para retomar a prova.",
        fg="#f0e4d3",
        bg="#1d1f21",
        font=("Helvetica", 18),
    )
    subtitle.pack(pady=(0, 16))

    report_message = _violation_report_message(reason)
    if report_message:
        report_label = tk.Label(
            container,
            text=report_message,
            fg="#ffb4a9",
            bg="#1d1f21",
            font=("Helvetica", 16, "bold"),
            wraplength=700,
            justify="center",
        )
        report_label.pack(pady=(0, 16))

    if _show_preview_during_block(reason):
        _add_camera_preview(container, preview_url=preview_url, bg="#1d1f21")

    if reason:
        reason_label = tk.Label(
            container,
            text=f"Motivo: {reason}",
            fg="#f2c9ad",
            bg="#1d1f21",
            font=("Helvetica", 14),
        )
        reason_label.pack()

    root.mainloop()
    return 0


def _waiting_mode(message: str, preview_url: str) -> int:
    import tkinter as tk

    root = tk.Tk()
    root.title("Aguardando aluno")
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.configure(bg="#0f1f1a")

    container = tk.Frame(root, bg="#0f1f1a")
    container.place(relx=0.5, rely=0.5, anchor="center")

    eyebrow = tk.Label(
        container,
        text="PROCTOR STATION",
        fg="#9fc7ad",
        bg="#0f1f1a",
        font=("Helvetica", 14, "bold"),
    )
    eyebrow.pack(pady=(0, 16))

    title = tk.Label(
        container,
        text=message,
        fg="white",
        bg="#0f1f1a",
        wraplength=900,
        justify="center",
        font=("Helvetica", 30, "bold"),
    )
    title.pack(pady=(0, 18))

    subtitle = tk.Label(
        container,
        text="A prova iniciara automaticamente quando sua identidade for confirmada.",
        fg="#d7e4d5",
        bg="#0f1f1a",
        wraplength=780,
        justify="center",
        font=("Helvetica", 18),
    )
    subtitle.pack()

    _add_camera_preview(container, preview_url=preview_url, bg="#0f1f1a")

    root.mainloop()
    return 0


def _confirmation_mode(
    student_id: str,
    student_name: str,
    timeout_sec: float,
    confirm_url: str,
    cancel_url: str,
    preview_url: str,
) -> int:
    import tkinter as tk

    root = tk.Tk()
    root.title("Confirmação antes da prova")
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.configure(bg="#0f1f1a")

    container = tk.Frame(root, bg="#0f1f1a", padx=60, pady=40)
    container.place(relx=0.5, rely=0.5, anchor="center")

    tk.Label(
        container,
        text="Confirme sua identidade antes de iniciar a prova",
        fg="white",
        bg="#0f1f1a",
        wraplength=1000,
        justify="center",
        font=("Helvetica", 28, "bold"),
    ).pack(pady=(0, 22))
    tk.Label(
        container,
        text=f"Aluno: {student_name}\nUsuário: {student_id}",
        fg="#b9dcc4",
        bg="#0f1f1a",
        justify="center",
        font=("Helvetica", 18, "bold"),
    ).pack(pady=(0, 24))

    _add_camera_preview(
        container,
        preview_url=preview_url,
        bg="#0f1f1a",
        max_size=(320, 240),
    )

    notice = (
        "Durante esta prova serão coletados e processados, exclusivamente para fins de "
        "proctoring: imagem da câmera, áudio ambiente, atividade do teclado e gravação da tela.\n\n"
        "Não tente burlar o sistema. Esta estação é destinada somente à realização da prova "
        "e seu uso é monitorado.\n\n"
        "Não é permitida consulta, interação com outra pessoa ou uso de celular durante a prova."
    )
    tk.Label(
        container,
        text=notice,
        fg="#d7e4d5",
        bg="#0f1f1a",
        wraplength=1050,
        justify="left",
        font=("Helvetica", 16),
    ).pack(pady=(0, 22))

    confirmed = tk.BooleanVar(value=False)
    confirm_button: tk.Button
    remaining_label = tk.Label(container, fg="#f2c9ad", bg="#0f1f1a", font=("Helvetica", 14))

    def set_confirm_enabled() -> None:
        confirm_button.configure(state="normal" if confirmed.get() else "disabled")

    tk.Checkbutton(
        container,
        text="Confirmo que sou o aluno identificado acima e desejo iniciar a prova.",
        variable=confirmed,
        command=set_confirm_enabled,
        fg="white",
        bg="#0f1f1a",
        activeforeground="white",
        activebackground="#0f1f1a",
        selectcolor="#1d4d36",
        font=("Helvetica", 14, "bold"),
    ).pack(pady=(0, 18))

    buttons = tk.Frame(container, bg="#0f1f1a")
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
        text="Cancelar",
        command=lambda: respond(cancel_url),
        bg="#3b4348",
        fg="white",
        activebackground="#4a5459",
        activeforeground="white",
        relief="flat",
        padx=24,
        pady=12,
        font=("Helvetica", 13, "bold"),
    ).pack(side="left", padx=(0, 12))
    confirm_button = tk.Button(
        buttons,
        text="Iniciar prova",
        command=lambda: respond(confirm_url),
        state="disabled",
        bg="#287a4d",
        fg="white",
        activebackground="#1f633e",
        activeforeground="white",
        relief="flat",
        padx=24,
        pady=12,
        font=("Helvetica", 13, "bold"),
    )
    confirm_button.pack(side="left")
    remaining_label.pack(pady=(18, 0))

    deadline = root.tk.call("clock", "seconds") + max(1, int(timeout_sec))

    def update_countdown() -> None:
        remaining = max(0, deadline - root.tk.call("clock", "seconds"))
        remaining_label.configure(text=f"Tempo para responder: {remaining}s")
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
    parser.add_argument("--message", default="Sente-se e olhe para a camera para iniciar a prova.")
    parser.add_argument("--preview-url", default="http://127.0.0.1:8000/camera-preview.jpg")
    parser.add_argument("--student-id", default="")
    parser.add_argument("--student-name", default="")
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--confirm-url", default="http://127.0.0.1:8000/pre-exam/confirmation/accept")
    parser.add_argument("--cancel-url", default="http://127.0.0.1:8000/pre-exam/confirmation/cancel")
    parser.add_argument("--guard-height", type=int, default=32)
    args = parser.parse_args(argv)

    os.environ.setdefault("DISPLAY", os.environ.get("DISPLAY", ":1"))

    if args.mode == "controls":
        return _controls_mode(args.stop_url)
    if args.mode == "waiting":
        return _waiting_mode(args.message, args.preview_url)
    if args.mode == "confirmation":
        return _confirmation_mode(
            args.student_id,
            args.student_name,
            args.timeout_sec,
            args.confirm_url,
            args.cancel_url,
            args.preview_url,
        )
    if args.mode == "guard":
        return _guard_mode(args.guard_height)
    return _blocked_mode(args.reason, args.preview_url)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
