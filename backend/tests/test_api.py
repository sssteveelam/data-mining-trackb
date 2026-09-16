from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from backend.config import Settings
from backend.inference import InferenceService
from backend.main import create_app
from backend.model import ArtifactStatus, ModelLoader


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    settings = Settings(artifact_dir=tmp_path)
    service = InferenceService(settings, loader=ModelLoader(tmp_path))
    return TestClient(create_app(settings=settings, service=service))


def test_health_reports_explicit_degraded_state_without_artifact(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


def test_model_info_exposes_nine_production_labels(client: TestClient) -> None:
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["class_count"] == 9
    assert "Horizontal_Stripes" not in body["labels"]


def test_predict_without_artifact_returns_503(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict",
        json={
            "wafer_map": [[0, 1, 2], [0, 1, 1]],
            "metadata": {"lot_name": "lot1", "wafer_index": 1},
        },
    )
    assert response.status_code == 503
    body = response.json()
    assert body["detail"]["code"] == "MODEL_UNAVAILABLE"


def test_predict_with_injected_model_returns_result(tmp_path: Path) -> None:
    settings = Settings(artifact_dir=tmp_path)
    loader = ModelLoader(tmp_path)
    loader._model = type(
        "FakeModel",
        (),
        {"predict": lambda self, values, verbose=0: [[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.90, 0.02, 0.02]]},
    )()
    loader._status = ArtifactStatus(
        model_version="test-model",
        labels=list(loader.status.labels),
        loaded=True,
        degraded=False,
        load_error=None,
        metrics=None,
        manifest=None,
        preprocess=None,
        model_sha256=None,
    )
    client = TestClient(create_app(settings=settings, service=InferenceService(settings, loader=loader)))
    response = client.post("/api/v1/predict", json={"wafer_map": [[0, 1, 2], [0, 1, 1]]})
    assert response.status_code == 200
    body = response.json()
    assert body["input_shape"] == [2, 3]
    assert body["processed_shape"] == [64, 64]
    assert len(body["top_k"]) == 3
    assert len(body["processed_map"]) == 64
    assert len(body["processed_map"][0]) == 64
    assert body["predicted_label"] == "Normal"
    assert body["review_required"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {"wafer_map": []},
        {"wafer_map": [[0, 1], [1]]},
        {"wafer_map": [[0, 3]]},
        {"wafer_map": [[0.5, 1]]},
        {"wafer_map": [["0", 1]]},
    ],
)
def test_predict_invalid_map_returns_422(client: TestClient, payload) -> None:
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422
