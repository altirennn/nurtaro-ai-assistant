import re

KAZAKH_MARKERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")


def detect_language(text: str, preferred_language: str = "ru") -> str:
    """Lightweight RU/KK detector; Kazakh-specific letters resolve mixed Cyrillic text."""
    if any(char in KAZAKH_MARKERS for char in text):
        return "kk"
    letters = re.sub(r"[^A-Za-zА-Яа-яЁё]", "", text)
    if letters and not re.search(r"[А-Яа-яЁё]", letters):
        return preferred_language if preferred_language in {"ru", "kk"} else "ru"
    return preferred_language if preferred_language in {"ru", "kk"} and len(letters) < 5 else "ru"
