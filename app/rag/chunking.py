def split_into_chunks(text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
    if max_chars < 50 or overlap < 0 or overlap >= max_chars:
        raise ValueError("Нужны max_chars >= 50 и 0 <= overlap < max_chars")
    paragraphs = [" ".join(part.split()) for part in text.splitlines() if part.strip()]
    # Keep Markdown headings with the following paragraph so a citation names the topic
    # while avoiding chunks that mix several unrelated activities.
    sections: list[str] = []
    pending_heading = ""
    for paragraph in paragraphs:
        if paragraph.startswith("#"):
            pending_heading = paragraph
            continue
        sections.append(f"{pending_heading} {paragraph}".strip() if pending_heading else paragraph)
        pending_heading = ""
    if pending_heading:
        sections.append(pending_heading)

    chunks: list[str] = []
    for section in sections:
        if len(section) > max_chars:
            start = 0
            while start < len(section):
                end = min(len(section), start + max_chars)
                if end < len(section):
                    boundary = section.rfind(" ", start + max_chars // 2, end)
                    if boundary > start:
                        end = boundary
                piece = section[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end >= len(section):
                    break
                start = max(start + 1, end - overlap)
        else:
            chunks.append(section)
    return chunks
