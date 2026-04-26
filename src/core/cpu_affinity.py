from __future__ import annotations

import os


def parse_cpu_set(value: str | None) -> set[int] | None:
    if value is None:
        return None

    text = value.strip()
    if not text:
        return None

    cpus: set[int] = set()
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                start, end = end, start
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(token))
    return cpus or None


def get_process_cpu_set() -> set[int] | None:
    if not hasattr(os, "sched_getaffinity"):
        return None
    try:
        return set(os.sched_getaffinity(0))
    except OSError:
        return None


def set_process_cpu_set(cpus: set[int] | None) -> bool:
    if not cpus or not hasattr(os, "sched_setaffinity"):
        return False
    try:
        os.sched_setaffinity(0, cpus)
        return True
    except OSError:
        return False


def auto_split_cpu_sets(
    *,
    available: set[int] | None,
    ffmpeg_override: set[int] | None = None,
    proctor_override: set[int] | None = None,
) -> tuple[set[int] | None, set[int] | None]:
    if available is None:
        return ffmpeg_override, proctor_override

    if ffmpeg_override is not None or proctor_override is not None:
        ffmpeg_cpus = ffmpeg_override
        proctor_cpus = proctor_override
        if proctor_cpus is None and ffmpeg_cpus:
            remaining = available - ffmpeg_cpus
            proctor_cpus = remaining or available
        return ffmpeg_cpus, proctor_cpus

    if len(available) < 4:
        return None, None

    ordered = sorted(available)
    ffmpeg_cpus = set(ordered[-2:])
    proctor_cpus = set(ordered[:-2])
    if not proctor_cpus:
        return None, None
    return ffmpeg_cpus, proctor_cpus


def split_ffmpeg_stream_cpu_sets(cpus: set[int] | None) -> dict[str, set[int] | None]:
    if not cpus:
        return {"webcam": None, "screen": None}

    ordered = sorted(cpus)
    if len(ordered) == 1:
        shared = {ordered[0]}
        return {"webcam": shared, "screen": shared}

    if len(ordered) == 2:
        return {"webcam": {ordered[0]}, "screen": {ordered[1]}}

    return {
        "webcam": set(ordered[:-1]),
        "screen": {ordered[-1]},
    }
