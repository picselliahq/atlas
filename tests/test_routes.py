import uuid

from fastapi import status
from fastapi.testclient import TestClient


def test_analysis(client: TestClient) -> None:
    payload = {
        "api_token": "test_api_token",
        "callback_url": "http://localhost:8000/callback",
        "organization_id": str(uuid.uuid4()),
        "dataset_version_id": str(uuid.uuid4()),
        "report_id": str(uuid.uuid4()),
        "report_object_name": "test_report_object_name",
        "chat_messages_object_name": "test_chat_messages_object_name",
    }
    response = client.post(url="/compute-analysis", json=payload)
    assert response.status_code == status.HTTP_200_OK, response.text


def test_analysis_without_payload(client: TestClient) -> None:
    response = client.post(url="/compute-analysis")
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_chat(client: TestClient) -> None:
    payload = {
        "api_token": "test_api_token",
        "callback_url": "http://localhost:8000/callback",
        "organization_id": str(uuid.uuid4()),
        "chat_id": str(uuid.uuid4()),
        "context": {"dataset_version_id": str(uuid.uuid4())},
        "chat_messages_object_name": "test_chat_messages_object_name",
        "message": "how are you?",
    }
    response = client.post(url="/chat", json=payload)
    assert response.status_code == status.HTTP_200_OK, response.text


def test_chat_without_payload(client: TestClient) -> None:
    response = client.post(url="/chat")
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
