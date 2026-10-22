import secrets
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings


@dataclass(frozen=True)
class Principal:
    principal_id: str
    owner_id: str
    can_scope_users: bool = False


def authenticate(
    settings: Settings,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_owner_id: str | None = Header(default=None, alias="X-Owner-ID"),
) -> Principal:
    if not x_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Требуется X-API-Key")
    principal_id = next(
        (principal for secret, principal in settings.api_keys.items() if secrets.compare_digest(secret, x_api_key)),
        None,
    )
    if principal_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Недействительный API-ключ")
    can_scope = principal_id == settings.bot_principal
    if x_owner_id and not can_scope:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Нельзя задавать область другого пользователя")
    owner_id = x_owner_id.strip() if can_scope and x_owner_id else principal_id
    if not owner_id or len(owner_id) > 128:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Некорректная область пользователя")
    return Principal(principal_id=principal_id, owner_id=owner_id, can_scope_users=can_scope)


def principal_dependency(settings: Settings):
    # A closure keeps each FastAPI app bound to its own Settings instance.
    def resolve(
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        x_owner_id: str | None = Header(default=None, alias="X-Owner-ID"),
    ) -> Principal:
        return authenticate(settings, x_api_key, x_owner_id)

    return resolve
