from __future__ import annotations

import re


_TIMECODE = re.compile(
    r"(?P<start>(?:\d{1,2}:)?\d{2}:\d{2}[,.]\d{3})\s+-->\s+"
    r"(?P<end>(?:\d{1,2}:)?\d{2}:\d{2}[,.]\d{3})"
)
_TAG = re.compile(r"<[^>]+>")


def _seconds(value: str) -> float:
    normalized = value.replace(",", ".")
    parts = normalized.split(":")
    if len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    else:
        hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _stamp(seconds: float) -> str:
    whole_seconds = max(0, int(seconds))
    return f"{whole_seconds // 3600:02d}:{(whole_seconds % 3600) // 60:02d}:{whole_seconds % 60:02d}"


def transcript_to_document(title: str, language: str, text: str) -> str:
    """Normalize plain text, WebVTT, or SRT into timestamped searchable passages."""
    cleaned_title = title.strip().replace("\r", " ")[:180]
    if not cleaned_title:
        raise ValueError("Укажите название видео")
    lines = [line.strip() for line in text.replace("\ufeff", "").splitlines()]
    output: list[str] = [f"# Транскрипт видео: {cleaned_title}", f"Язык: {language}", ""]
    index = 0
    timed_segments = 0
    while index < len(lines):
        match = _TIMECODE.search(lines[index])
        if not match:
            if lines[index] and lines[index].upper() != "WEBVTT" and not lines[index].isdigit():
                output.append(lines[index])
            index += 1
            continue
        start, end = _seconds(match.group("start")), _seconds(match.group("end"))
        if end <= start:
            raise ValueError("У субтитров время окончания должно быть позже времени начала")
        index += 1
        body: list[str] = []
        while index < len(lines) and lines[index] and not _TIMECODE.search(lines[index]):
            if lines[index].upper() != "WEBVTT" and not lines[index].isdigit():
                body.append(_TAG.sub("", lines[index]).strip())
            index += 1
        segment_text = " ".join(piece for piece in body if piece).strip()
        if segment_text:
            output.append(f"[{_stamp(start)}–{_stamp(end)}] {segment_text}")
            timed_segments += 1
    document = "\n".join(output).strip()
    if len(document) < 30:
        raise ValueError("В файле не найдено достаточно текста для индексации")
    if timed_segments == 0:
        document += "\n\nВременные метки не указаны в исходной расшифровке."
    return document
