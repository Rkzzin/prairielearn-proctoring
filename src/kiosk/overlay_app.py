from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request


def _send_stop_request(stop_url: str) -> bool:
    request = urllib.request.Request(stop_url, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=5):
            return True
    except (urllib.error.URLError, TimeoutError):
        return False


def _controls_mode(stop_url: str) -> int:
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.title("Proctor Controls")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.configure(bg="#1d2a33")
    root.geometry("+40+40")

    frame = tk.Frame(root, bg="#1d2a33", padx=10, pady=10)
    frame.pack()

    def on_stop() -> None:
        confirmed = messagebox.askyesno(
            "Encerrar prova",
            "Deseja realmente encerrar a prova?",
            parent=root,
        )
        if not confirmed:
            return
        if _send_stop_request(stop_url):
            root.destroy()
        else:
            messagebox.showerror(
                "Falha ao encerrar",
                "Não foi possível encerrar a prova pela API local.",
                parent=root,
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
