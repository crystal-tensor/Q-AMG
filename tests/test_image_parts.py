import io
import time

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from PIL import Image

from qamg import image_pipeline
from qamg.image_model import read_image
from qamg.mesh_parts import (
    CutPlane,
    as_manifold,
    cut_parts,
    merge_solids,
    partition_problem,
    validate_partition,
)
from qamg.schema import Image3DSettings
from qamg.server import create_app


def cube():
    mesh = trimesh.creation.box(extents=[100, 70, 60])
    mesh.visual.vertex_colors = np.tile([60, 130, 90, 255], (len(mesh.vertices), 1))
    return mesh


def png():
    stream = io.BytesIO()
    Image.new("RGB", (64, 64), (100, 150, 200)).save(stream, format="PNG")
    return stream.getvalue()


def test_split_and_boolean_union_preserve_actual_solid_not_just_volume():
    source = as_manifold(cube())
    parts = cut_parts(source, [CutPlane(0, 0, 1, "center")], (1,))
    assert len(parts) == 2
    result = validate_partition(source, parts)
    assert result["valid"] and result["symmetric_difference_mm3"] < 1e-5
    assert merge_solids(parts).volume() == pytest.approx(source.volume())
    moved = [parts[0].translate([20, 0, 0]), parts[1]]
    assert validate_partition(source, moved)["valid"] is False


def test_image_partition_uses_same_original_energy_and_exact_binary_encoding():
    p = partition_problem([CutPlane(i, 0, i+1, str(i)) for i in range(3)], "image-sha", 4, 42)
    assert p.spec.family == "image_parts"
    compact = p.encode().compact_binary()
    for state in p.assignments():
        original = p.original_energy(p.bits(state))
        assert original == pytest.approx(compact.energy(state))
    with pytest.raises(ValueError, match="没有选择"):
        cut_parts(as_manifold(cube()), [CutPlane(0, 0, 1, "x")], (0,))


def test_invalid_image_refuses_before_neural_inference():
    with pytest.raises(ValueError, match="无法读取"):
        read_image(b"not-an-image")
    result = read_image(png())
    assert result.mode == "RGBA" and result.size == (64, 64)


def test_upload_qb_partition_merge_and_source_restoration_contract(tmp_path, monkeypatch):
    # Explicit model stub only for the HTTP/export contract; real TripoSR runs
    # are saved separately and never established by this unit test.
    def model_stub(image, settings, progress):
        return cube(), image.convert("RGB"), {"model": "contract-stub-not-TripoSR", "timings": {"total_s": 0}}

    monkeypatch.setattr(image_pipeline, "generate_mesh", model_stub)
    settings = Image3DSettings(planner={"backend": "qb", "encoding": "compact_binary", "iterations": 0,
                                       "depth": 1, "shots": 256})
    with TestClient(create_app(tmp_path/"runs", web_root=tmp_path/"no-web")) as client:
        invalid = client.post("/api/generate-image", files={"file": ("bad.png", b"invalid", "image/png")})
        assert invalid.status_code == 422
        response = client.post("/api/generate-image", files={"file": ("image.png", png(), "image/png")},
                               data={"options": settings.model_dump_json()})
        assert response.status_code == 202

        def wait(id):
            for _ in range(500):
                value = client.get(f"/api/jobs/{id}").json()
                if value["status"] in {"failed", "completed"}:
                    assert value["status"] == "completed", value
                    return value["summary"]
                time.sleep(.01)
            pytest.fail("Contract job did not finish")

        id = response.json()["id"]
        summary = wait(id)
        assert summary["provenance"]["quantum_sdk"] == "QuantumBridge"
        assert summary["validation"]["valid"] and summary["part_count"] >= 2
        trace = client.get(f"/api/runs/{id}/trace.json").json()[0]["plan"]["trace"]
        bits = summary["selected_bits"]
        assert any(bits in batch["counts"] for batch in trace)
        structure = client.get(f"/api/runs/{id}/structure.json").json()
        chosen = [p["id"] for p in structure["parts"][:2]]
        merge = client.post(f"/api/runs/{id}/merge", json={"part_ids": chosen})
        merged_id = merge.json()["id"]
        merged = wait(merged_id)
        assert merged["part_count"] == summary["part_count"]-1
        assert merged["original_run_id"] == id
        assert merged["validation"]["valid"]
        assert client.get(f"/api/runs/{id}/run.json").json() == summary
        assert client.get(f"/api/runs/{merged_id}/parts.zip").status_code == 200
        assert client.get(f"/api/runs/{merged_id}/merge.json").json()["quantum_resampling_performed"] is False
        assert client.post(f"/api/runs/{id}/merge", json={"part_ids": ["missing", "also-missing"]}).status_code == 422
        assert client.post("/api/generate-image", headers={"Origin": "https://unrelated.invalid"},
                           files={"file": ("image.png", png(), "image/png")}).status_code == 403
        preflight = client.options("/api/generate-image", headers={"Origin": "https://crystal-tensor.github.io",
                                   "Access-Control-Request-Method": "POST"})
        assert preflight.headers["access-control-allow-origin"] == "https://crystal-tensor.github.io"
        assert preflight.headers["access-control-allow-private-network"] == "true"
