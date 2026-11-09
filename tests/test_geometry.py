import json

import cadquery as cq
import pytest
import trimesh

from qamg.geometry import decode_geometry, export_model, validate_geometry
from qamg.pipeline import generate
from qamg.problem import build_problem
from qamg.schema import DesignSpec, PlannerSettings


@pytest.mark.parametrize("family,selection", [
    ("drive", (0, 0, 0, 0, 0)), ("drive", (1, 1, 1, 1, 1)),
    ("clamp", (0, 0, 0, 0, 0)), ("clamp", (1, 1, 1, 1, 1)),
])
def test_executable_solid_variants(family, selection):
    p = build_problem(DesignSpec(family=family))
    m = decode_geometry(p, p.bits(selection))
    result = validate_geometry(m)
    assert 5 <= len(m.parts) <= 15
    assert result.valid, result.model_dump()
    assert all(part.shape.Volume() > 0 for part in m.parts)
    assert all(len(part.shape.Solids()) == 1 for part in m.parts)


def test_bad_fit_is_rejected_by_geometry():
    p = build_problem(DesignSpec(groups=3))
    m = decode_geometry(p, p.bits((1, 0, 0)))
    result = validate_geometry(m)
    assert not result.valid
    assert "bearing_fit" in [i.code for i in result.issues]


def test_sampled_disassembly_is_explicitly_bounded():
    from qamg.disassembly import plan_disassembly
    p = build_problem(DesignSpec(family="clamp", groups=3))
    model = decode_geometry(p, p.bits((0, 0, 0)))
    result = plan_disassembly(model, samples=12)
    assert result["complete_for_envelope_model"]
    assert len(result["moves"]) == len(model.parts)-1
    assert result["continuous_collision_proof"] is False
    assert result["thread_tool_force_or_robot_validated"] is False


def test_step_glb_round_trip_and_editable_parameters(tmp_path):
    p = build_problem(DesignSpec(family="clamp"))
    m = decode_geometry(p, p.bits((0, 0, 0, 0, 0)))
    export_model(m, tmp_path)
    imported = cq.importers.importStep(str(tmp_path/"assembly.step"))
    assert len(imported.solids().vals()) == len(m.parts)
    scene = trimesh.load(tmp_path/"assembly.glb", force="scene")
    assert len(scene.geometry) == len(m.parts)
    structure = json.loads((tmp_path/"structure.json").read_text())
    assert structure["explosion"]["collision_checked"] is False
    assert structure["selected_bits"] == m.bits


def test_grooved_pads_keep_continuous_backing_after_step_export(tmp_path):
    p = build_problem(DesignSpec(family="clamp", scale=.83))
    model = decode_geometry(p, p.bits((1, 1, 1, 0, 1)))
    assert validate_geometry(model).valid
    for part in model.parts:
        if part.id.endswith("_pad"):
            assert len(part.shape.Solids()) == 1
    export_model(model, tmp_path)
    assert len(cq.importers.importStep(str(tmp_path/"assembly.step")).solids().vals()) == len(model.parts)


def test_quantum_geometry_feedback_reencodes_and_resamples(tmp_path):
    r = generate(DesignSpec(family="drive", load="heavy", groups=3,
                            max_gear_radius_mm=15, seed=42),
                 PlannerSettings(iterations=6, shots=256), tmp_path)
    assert r["validation"]["valid"]
    assert r["provenance"]["generation_rounds"] == 2
    trace = json.loads((tmp_path/"trace.json").read_text())
    assert trace[0]["feedback_rules"] == [{"group": 2, "option": 1, "reason": "gear_envelope"}]
    assert r["decisions"]["gear"]["index"] == 0
    assert r["selected_bits"] in {
        bits for batch in trace[-1]["plan"]["trace"] for bits in batch["counts"]}

