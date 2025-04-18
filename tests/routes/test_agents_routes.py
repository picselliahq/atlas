from fastapi import status
from fastapi.testclient import TestClient


def test_compute_analysis_error(client: TestClient) -> None:
    response = client.post(url="/compute-analysis")
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
