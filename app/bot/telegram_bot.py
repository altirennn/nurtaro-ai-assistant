from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from dotenv import load_dotenv

from app.logging_config import configure_logging

load_dotenv()
logger = logging.getLogger(__name__)
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
BOT_API_KEY = os.getenv("BOT_API_KEY", "")


def telegram(method: str, payload: dict[str, Any]) -> Any:
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{BOT_TOKEN}/{method}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=40) as response:
        data = json.loads(response.read())
    if not data.get("ok"):
        raise RuntimeError("Telegram API returned an error")
    return data["result"]


def api_request(method: str, path: str, telegram_user_id: int, payload: dict[str, Any] | None = None) -> Any:
    if not BOT_API_KEY:
        raise RuntimeError("BOT_API_KEY is not configured")
    headers = {
        "X-API-Key": BOT_API_KEY,
        "X-Owner-ID": f"telegram_{telegram_user_id}",
        "Content-Type": "application/json",
    }
    request = urllib.request.Request(
        f"{API_BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def send_message(chat_id: int, text: str) -> None:
    telegram("sendMessage", {"chat_id": chat_id, "text": text[:4000], "disable_web_page_preview": True})


def format_sources(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "Источники не найдены."
    return "Источники:\n" + "\n".join(
        f"• {source['title']} · {source['source_id']}" for source in sources[:4]
    )


def handle_message(message: dict[str, Any], state: dict[int, dict[str, Any]]) -> None:
    chat = message.get("chat", {})
    sender = message.get("from", {})
    if not chat or not sender or chat.get("type") != "private":
        return
    chat_id = int(chat["id"])
    telegram_user_id = int(sender["id"])
    text = str(message.get("text", "")).strip()
    if not text:
        return
    parts = text.split(maxsplit=1)
    command = parts[0].split("@", 1)[0].lower() if parts[0].startswith("/") else ""
    argument = parts[1].strip() if len(parts) > 1 else ""

    if command in {"/start", "/help"}:
        state.setdefault(telegram_user_id, {})
        send_message(chat_id, "Сәлем! Я демонстрационный ассистент. Используйте /profile, затем /ask Вопрос. Команды: /start, /profile, /ask, /sources, /help.")
        return
    if command == "/sources":
        send_message(chat_id, format_sources(state.get(telegram_user_id, {}).get("sources", [])))
        return
    if command == "/profile":
        try:
            if argument:
                fields = argument.split(maxsplit=1)
                if len(fields) != 2 or not fields[0].isdigit() or fields[1].lower() not in {"ru", "kk"}:
                    send_message(chat_id, "Формат: /profile 8 kk — возраст в месяцах и язык ru или kk.")
                    return
                profile = api_request("POST", "/profiles", telegram_user_id, {
                    "age_months": int(fields[0]), "preferred_language": fields[1].lower(),
                    "interests": [], "available_materials": [], "parent_preferences": "",
                })
                state.setdefault(telegram_user_id, {})["profile_id"] = profile["id"]
            else:
                profiles = api_request("GET", "/profiles", telegram_user_id)
                if not profiles:
                    send_message(chat_id, "Профиль ещё не создан. Напишите /profile 8 kk или /profile 8 ru.")
                    return
                profile = profiles[0]
                state.setdefault(telegram_user_id, {})["profile_id"] = profile["id"]
            send_message(chat_id, f"Профиль: {profile['age_months']} мес.; язык {profile['preferred_language']}. Хранится без имени ребёнка.")
        except (urllib.error.URLError, RuntimeError, KeyError, ValueError):
            logger.warning("profile request failed", extra={"event": "telegram_profile_error"})
            send_message(chat_id, "Не удалось получить профиль. Проверьте подключение к API и конфигурацию бота.")
        return
    if command == "/ask" and not argument:
        state.setdefault(telegram_user_id, {})["awaiting_question"] = True
        send_message(chat_id, "Напишите вопрос о совместной игре или материале из демонстрационной базы.")
        return
    question = argument if command == "/ask" else (text if state.get(telegram_user_id, {}).pop("awaiting_question", False) else "")
    if command == "/ask" or question:
        payload: dict[str, Any] = {"question": question}
        profile_id = state.get(telegram_user_id, {}).get("profile_id")
        if profile_id:
            payload["profile_id"] = profile_id
        try:
            result = api_request("POST", "/chat", telegram_user_id, payload)
            state.setdefault(telegram_user_id, {})["sources"] = result.get("sources", [])
            send_message(chat_id, result["answer"] + "\n\n" + format_sources(result.get("sources", [])))
        except (urllib.error.URLError, RuntimeError, KeyError, ValueError):
            logger.warning("chat request failed", extra={"event": "telegram_chat_error"})
            send_message(chat_id, "Не удалось связаться с API. Попробуйте позже.")
        return
    if not command:
        send_message(chat_id, "Напишите /ask, затем задайте вопрос. Список команд: /help.")


def main() -> None:
    configure_logging()
    if not BOT_TOKEN or not BOT_API_KEY:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN and BOT_API_KEY in the environment")
    offset = 0
    state: dict[int, dict[str, Any]] = {}
    while True:
        try:
            updates = telegram("getUpdates", {"offset": offset, "timeout": 30, "allowed_updates": ["message"]})
            for update in updates:
                offset = max(offset, int(update["update_id"]) + 1)
                if "message" in update:
                    handle_message(update["message"], state)
        except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError):
            logger.warning("telegram polling failed", extra={"event": "telegram_poll_error"})
            time.sleep(3)


if __name__ == "__main__":
    main()
