from collections.abc import Generator
from pathlib import Path

import pytest
from celery import Celery
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from smolagents import LiteLLMModel
from testcontainers.ollama import OllamaContainer

import config.celery
import config.fastapi
from config.settings import LLMConfig, Settings


@pytest.fixture(scope="session")
def settings() -> Generator[Settings, None, None]:
    model_name = "llama3.2:latest"
    with OllamaContainer(
        "ollama/ollama:latest", ollama_home=Path.home() / ".ollama"
    ) as ollama:
        if model_name not in [e["name"] for e in ollama.list_models()]:
            print(f"did not find '{model_name}', pulling")
            ollama.pull_model(model_name)

        yield Settings(
            api_key="TEST_TOKEN",
            report_model=LLMConfig(base_url=ollama.get_endpoint() + "/v1"),
            chat_model=LLMConfig(
                name="ollama_chat/llama3.2",
                base_url=ollama.get_endpoint(),
            ),
        )


@pytest.fixture(scope="session")
def fastapi_app(settings: Settings) -> Generator[FastAPI, None, None]:
    app = config.fastapi.create_app(settings=settings)
    yield app


@pytest.fixture(scope="session", autouse=True)
def celery_app(settings: Settings) -> Generator[Celery, None, None]:
    app = config.celery.create_app(settings=settings)
    app.conf.task_always_eager = True
    app.conf.task_eager_propagates = True
    yield app


@pytest.fixture(scope="function")
def client(
    fastapi_app: FastAPI, settings: Settings
) -> Generator[TestClient, None, None]:
    with TestClient(
        app=fastapi_app,
        headers={
            "Authorization": f"Bearer {settings.api_key}",
            "Content-type": "application/json",
        },
    ) as client:
        yield client


@pytest.fixture(scope="session")
def report_model(settings: Settings) -> OpenAIModel:
    return OpenAIModel(
        model_name=settings.report_model.name,
        provider=OpenAIProvider(
            base_url=settings.report_model.base_url,
            api_key=settings.report_model.api_key,
        ),
    )


@pytest.fixture(scope="session")
def chat_model(settings: Settings) -> LiteLLMModel:
    return LiteLLMModel(
        model_id=settings.chat_model.name,
        api_base=settings.chat_model.base_url,
        api_key=settings.chat_model.api_key,
        temperature=1,
    )
