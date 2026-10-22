from __future__ import annotations


SYSTEM_PROMPT = (
    "Ты — помощник для родителей детей 0–3 лет. Отвечай кратко на языке вопроса. "
    "Опирайся только на вопрос, профиль и предоставленный контекст. Если информации недостаточно, "
    "скажи об этом прямо и не придумывай ответ. Не давай медицинских советов, диагнозов или выводов "
    "о развитии ребёнка."
)


def build_messages(instruction: str, context: str = "", language: str = "язык вопроса") -> list[dict[str, str]]:
    user_content = f"Язык ответа: {language}.\nВопрос: {instruction}"
    if context.strip():
        user_content += f"\n\nПрофиль и доступный контекст:\n{context}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def render_prompt(tokenizer, instruction: str, context: str = "", language: str = "язык вопроса") -> str:
    return tokenizer.apply_chat_template(
        build_messages(instruction, context, language),
        tokenize=False,
        add_generation_prompt=True,
    )
