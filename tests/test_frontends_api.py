from fastapi.testclient import TestClient

from qamg.frontends import parse_text
from qamg.server import create_app


def test_text_is_bounded_and_dimensions_are_not_guessed():
    spec, warnings = parse_text("重载齿轮装配，轴长110mm，最大齿轮半径15mm")
    assert spec.family == "drive" and spec.shaft_length_mm == 110
    assert spec.max_gear_radius_mm == 15 and spec.load == "heavy"
    assert warnings
    clamp, _ = parse_text("夹持机构，开口35mm")
    assert clamp.family == "clamp" and clamp.opening_mm == 35


def test_api_status_parse_and_artifact_boundary(tmp_path):
    with TestClient(create_app(tmp_path/"runs", web_root=tmp_path/"no-web")) as client:
        assert client.get("/api/status").json()["hardware_verified"] is False
        assert client.get("/api/status").headers["cache-control"] == "no-store"
        assert client.post("/api/parse", json={"text": "夹持机构，开口35"}).json()["spec"]["opening_mm"] == 35
        assert client.post("/api/parse", json={"text": "unknown"}).status_code == 422
        assert client.get("/api/runs/not-a-run/secrets.env").status_code == 404
        assert client.post("/api/generate", json={"spec": {"groups": 999}}).status_code == 422
