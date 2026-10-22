import re

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|rules)", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior)\s+(instructions|rules)", re.I),
    re.compile(r"(system|developer)\s+prompt", re.I),
    re.compile(r"reveal\s+(the\s+)?(secret|api\s+key|password)", re.I),
    re.compile(r"игнорируй(те)?\s+(все\s+)?(предыдущие|прошлые)\s+инструкц", re.I),
    re.compile(r"игнорлаңыз\s+(барлық\s+)?алдыңғы\s+нұсқау", re.I),
    re.compile(r"құпия\s+(кілт|сөзді)\s+(көрсет|аш)", re.I),
]


def sanitize_document(text: str) -> tuple[str, int]:
    """Drop obvious instruction-like lines; this is a defense-in-depth heuristic."""
    kept: list[str] = []
    removed = 0
    for line in text.splitlines():
        if any(pattern.search(line) for pattern in INJECTION_PATTERNS):
            removed += 1
        else:
            kept.append(line)
    return "\n".join(kept).strip(), removed
