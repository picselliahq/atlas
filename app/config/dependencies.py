from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from typing import Annotated, Any

import matplotlib
from celery import Celery
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import config.celery
from config.settings import Settings


def get_settings(request: Request) -> Settings:
    return request.state.settings


def check_api_key(
    credentials: Annotated[
        HTTPAuthorizationCredentials, Depends(HTTPBearer(auto_error=False))
    ],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    if settings.api_key is None:
        return
    elif not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    elif credentials.credentials != settings.api_key:
        raise HTTPException(status_code=403, detail="Access denied")


def get_celery_app(request: Request) -> Celery:
    return request.state.celery_app


@asynccontextmanager
async def lifespan(_: FastAPI, settings: Settings) -> AsyncGenerator[Mapping[str, Any]]:
    matplotlib.use("agg")
    yield {
        "settings": settings,
        "celery_app": config.celery.create_app(settings=settings),
    }
