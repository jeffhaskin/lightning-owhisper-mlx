from fastapi.testclient import TestClient

from lightning_owhisper_mlx.config import AppConfig, GeneralConfig, ModelConfig
from lightning_owhisper_mlx.server import create_app


def make_client(api_key: str | None) -> TestClient:
    config = AppConfig(
        general=GeneralConfig(api_key=api_key),
        models=[ModelConfig(id="demo", model="distil-small.en")],
    )
    app = create_app(config)
    return TestClient(app)


def test_status_endpoint_accessible_without_auth() -> None:
    client = make_client(api_key="secret")

    response = client.get("/v1/status")

    assert response.status_code == 204
    # FastAPI's TestClient returns an empty string for 204 responses.
    assert response.text == ""


def test_models_endpoint_allows_missing_auth_but_rejects_invalid() -> None:
    client = make_client(api_key="secret")

    unauthorized = client.get("/models", headers={"Authorization": "Token nope"})
    assert unauthorized.status_code == 401

    allowed = client.get("/models")
    assert allowed.status_code == 200
    data = allowed.json()
    assert data["object"] == "list"
    assert data["data"] == [{"id": "demo", "object": "model"}]


def test_health_endpoint_returns_ok_without_auth() -> None:
    client = make_client(api_key="secret")

    response = client.get("/health")

    assert response.status_code == 200
    assert response.text == "OK"
