from __future__ import annotations

import argparse
import os

from src.kiosk.lockdown import Lockdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Restaura lockdown persistido do Proctor Station")
    parser.add_argument("--display", default=os.environ.get("DISPLAY", ":1"))
    args = parser.parse_args(argv)

    Lockdown(display=args.display).disable()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
