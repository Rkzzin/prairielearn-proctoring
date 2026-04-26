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


def _blocked_mode(reason: str) -> int:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Overlay da estação de prova")
    parser.add_argument("--mode", choices=["controls", "blocked"], required=True)
    parser.add_argument("--stop-url", default="http://127.0.0.1:8000/session/stop")
    parser.add_argument("--reason", default="")
    args = parser.parse_args(argv)

    os.environ.setdefault("DISPLAY", os.environ.get("DISPLAY", ":1"))

    if args.mode == "controls":
        return _controls_mode(args.stop_url)
    return _blocked_mode(args.reason)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
