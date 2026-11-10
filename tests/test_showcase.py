import json

import pytest
import trimesh
from fastapi.testclient import TestClient

from qamg.gallery import CATALOGUE, public_catalogue
from qamg.image_pipeline import load_run_parts, merge_run
from qamg.showcase import generate_showcase


def test_catalogue_has_ten_new_distinct_cases_and_keeps_both_cad_cases():
    assert len(CATALOGUE) == 10
    assert len({e["slug"] for e in public_catalogue()}) == 12
    assert {"model-y", "iphone", "optimus", "falcon", "mechanical-watch", "refinery"} <= {e["slug"] for e in CATALOGUE}


@pytest.fixture
def phone(tmp_path):
    directory = tmp_path/"artifacts/examples/iphone"
    summary = generate_showcase("iphone", directory)
    return directory, summary


def test_authored_demo_has_real_qb_witness_geometry_and_reversible_union(phone, tmp_path):
    directory, summary = phone
    assert summary["spec"]["family"] == "showcase"
    assert summary["provenance"]["photo_reconstruction_performed"] is False
    assert summary["provenance"]["official_product_cad"] is False
    assert summary["part_count"] >= 8 and summary["validation"]["valid"]
    witness = summary["provenance"]["selected_measurement"]
    trace = json.loads((directory/"trace.json").read_text())[0]["plan"]["trace"]
    assert any(t["job_id"] == witness["job_id"] and t["counts"].get(witness["bits"], 0) > 0 for t in trace)
    scene = trimesh.load(directory/"assembly.glb", force="scene")
    structure = json.loads((directory/"structure.json").read_text())
    assert len(scene.geometry) == summary["part_count"]
    assert {p["label"] for p in structure["parts"]} >= {"显示屏", "金属中框", "电池", "后盖"}
    originals = load_run_parts(directory, structure)
    assert all(part.volume() > 0 for part in originals.values())
    result = merge_run(directory, ["part-000", "part-001"], tmp_path/"merged", "example-iphone")
    assert result["part_count"] == summary["part_count"]-1
    assert result["validation"]["valid"]
    assert result["original_run_id"] == "example-iphone"
    assert result["provenance"]["quantum_role"] == "source_illustrative_layout_parameter_selection"
    assert result["disassembly"]["moves"] == len(json.loads((tmp_path/"merged/disassembly.json").read_text())["moves"])
    assert json.loads((directory/"run.json").read_text())["part_count"] == summary["part_count"]


def test_catalogued_demo_is_served_by_local_backend(phone, monkeypatch, tmp_path):
    from qamg import server

    monkeypatch.setattr(server, "PROJECT_ROOT", tmp_path)
    with TestClient(server.create_app(tmp_path/"runs", web_root=tmp_path/"no-web")) as client:
        entries = client.get("/api/runs").json()
        assert any(e["id"] == "example-iphone" for e in entries)
        assert client.get("/api/runs/example-iphone/assembly.glb").content[:4] == b"glTF"
        assert client.get("/api/runs/example-iphone/demo.json").json()["kind"] == "procedural_demo"
        assert client.post("/api/runs/example-iphone/container-fit").status_code == 422
