from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI

from api import agents_api
from config import dependencies
from config.sentry import init_sentry
from config.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    init_sentry(settings)

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI) -> AsyncGenerator[Mapping[str, Any]]:
        async with dependencies.lifespan(fastapi_app, settings=settings) as state:
            yield state

    app = FastAPI(
        title="PicselliaAgentsAPI",
        description="Picsellia Agents Microservice",
        version="0.1.0",
        dependencies=[Depends(dependencies.check_api_key)],
        redirect_slashes=False,
        lifespan=lifespan,
    )
    app.include_router(router=agents_api.router, tags=["agent"])
    return app
