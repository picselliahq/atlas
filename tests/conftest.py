from collections.abc import Generator
from pathlib import Path
from unittest import mock

import httpx
import picsellia
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
            celery_broker_url="memory://localhost/",
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


@pytest.fixture(scope="function", autouse=True)
def mock_redis_client() -> Generator[mock.Mock, None, None]:
    with mock.patch("services.context.cache") as mock_redis_client:
        mock_redis_client.get.return_value = None
        mock_redis_client.set.return_value = None
        yield mock_redis_client


@pytest.fixture(scope="function", autouse=True)
def mock_picsellia_client() -> Generator[mock.Mock, None, None]:
    with mock.patch("services.context.get_client") as mock_picsellia_client:
        mock_client = mock.Mock(spec=picsellia.Client)

        mock_connexion = mock.Mock()
        mock_connexion.generate_report_object_name.return_value = "mocked_report_name"
        mock_client.connexion = mock_connexion

        mock_dataset_version = mock.Mock(spec=picsellia.DatasetVersion)
        mock_dataset_version.list_embeddings.return_value = []
        mock_dataset_version.list_annotations.return_value = []
        mock_client.get_dataset_version_by_id.return_value = mock_dataset_version

        mock_picsellia_client.return_value = mock_client
        yield mock_client


@pytest.fixture(scope="function", autouse=True)
def mock_load_assets() -> Generator[mock.Mock, None, None]:
    with mock.patch(
        "agents.common.metadata.image_metadata_processor.load_assets"
    ) as mock_load_assets:
        mock_instance = mock.Mock()
        mock_load_assets.return_value = []
        yield mock_instance


@pytest.fixture(scope="function", autouse=True)
def mock_file_service() -> Generator[mock.Mock, None, None]:
    with mock.patch("services.context.FileService") as mock_file_service:
        mock_instance = mock.Mock()
        mock_file_service.return_value = mock_instance
        mock_instance.upload.return_value = None
        mock_instance.download.return_value = None

        yield mock_instance


@pytest.fixture(scope="function", autouse=True)
def mock_callback_platform() -> Generator[mock.Mock, None, None]:
    with mock.patch("services.context.callback_platform") as mock_client:
        mock_response = mock.Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None

        mock_client.return_value = mock_response
        yield mock_client
