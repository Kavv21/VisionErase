from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.auth import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from api.core.config import get_settings
from api.core.database import get_db
from api.core.metrics import AUTH_REGISTRATIONS_TOTAL, AUTH_LOGINS_TOTAL
from api.models.user import User

log = structlog.get_logger(__name__)
router = APIRouter(tags=["auth"])
settings = get_settings()

DbDep = Annotated[AsyncSession, Depends(get_db)]
CurrentUser = Annotated[User, Depends(get_current_user)]


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    display_name: str = Field(min_length=1, max_length=120)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest, db: DbDep) -> TokenResponse:
    result = await db.execute(select(User).where(User.email == req.email))
    if result.scalar_one_or_none() is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    user = User(
        email=req.email,
        hashed_password=hash_password(req.password),
        display_name=req.display_name,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    log.info("user_registered", user_id=str(user.id), email=user.email)
    AUTH_REGISTRATIONS_TOTAL.inc()
    return TokenResponse(access_token=create_access_token(user.id, user.email))


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, db: DbDep) -> TokenResponse:
    result = await db.execute(select(User).where(User.email == req.email))
    user = result.scalar_one_or_none()

    if user is None or user.hashed_password is None or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    log.info("user_login", user_id=str(user.id), email=user.email)
    AUTH_LOGINS_TOTAL.inc()
    return TokenResponse(access_token=create_access_token(user.id, user.email))


@router.get("/google")
async def google_oauth_redirect() -> RedirectResponse:
    if not settings.google_client_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth not configured",
        )

    from authlib.integrations.starlette_client import OAuth  # noqa: PLC0415

    oauth = OAuth()
    oauth.register(
        name="google",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

    redirect_uri = settings.google_redirect_uri
    auth_data = await oauth.google.create_authorization_url(redirect_uri)
    auth_url = auth_data["url"] if isinstance(auth_data, dict) else auth_data
    return RedirectResponse(auth_url)


@router.get("/google/callback")
async def google_oauth_callback(code: str, db: DbDep) -> RedirectResponse:
    if not settings.google_client_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth not configured",
        )

    import httpx  # noqa: PLC0415

    # Exchange code for tokens
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": settings.google_redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        token_resp.raise_for_status()
        tokens = token_resp.json()

    # Verify id_token and extract claims
    from google.oauth2 import id_token as google_id_token  # noqa: PLC0415
    from google.auth.transport import requests as google_requests  # noqa: PLC0415

    id_info = google_id_token.verify_oauth2_token(
        tokens["id_token"],
        google_requests.Request(),
        settings.google_client_id,
    )

    google_sub: str = id_info["sub"]
    email: str = id_info["email"]
    display_name: str = id_info.get("name", email.split("@")[0])

    # Upsert user by google_sub
    result = await db.execute(select(User).where(User.google_sub == google_sub))
    user = result.scalar_one_or_none()

    if user is None:
        # Check if email already exists (link accounts)
        result2 = await db.execute(select(User).where(User.email == email))
        user = result2.scalar_one_or_none()
        if user is None:
            user = User(email=email, display_name=display_name, google_sub=google_sub)
            db.add(user)
        else:
            user.google_sub = google_sub
        await db.commit()
        await db.refresh(user)

    token = create_access_token(user.id, user.email)
    log.info("google_oauth_success", user_id=str(user.id))
    frontend_url = "http://localhost:5173"
    return RedirectResponse(f"{frontend_url}/auth/callback?token={token}")


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: CurrentUser) -> UserResponse:
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        display_name=current_user.display_name,
    )
