import json
from zipfile import ZipFile

import pytest
import trimesh
from fastapi.testclient import TestClient

from qamg.engineering_models import BUILDERS
from qamg.gallery import CATALOGUE, public_catalogue
from qamg.image_pipeline import load_run_parts, merge_run, rename_run
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
    assert structure["decomposition"]["counts"]["components"] == summary["part_count"]
    parents = {g["id"]: g["parent"] for g in structure["groups"]}
    for part in structure["parts"]:
        assert parents[parents[part["parent"]]] == "root"
        assert len(part["explode_levels_mm"]) == 3
        assert part["source_refs"] and part["geometry_accuracy"] == "parametric_approximation"
    inventory = json.loads((directory/"bom.json").read_text())
    assert [p["name"] for p in inventory["components"]] == [p["label"] for p in structure["parts"]]
    assert inventory["complete_factory_bom"] is False
    assert "界面图案" not in " ".join(p["name"] for p in inventory["components"])
    originals = load_run_parts(directory, structure)
    assert all(part.volume() > 0 for part in originals.values())
    result = merge_run(directory, ["part-000", "part-001"], tmp_path/"merged", "example-iphone")
    assert result["part_count"] == summary["part_count"]-1
    assert result["validation"]["valid"]
    assert result["original_run_id"] == "example-iphone"
    assert result["provenance"]["quantum_role"] == "source_illustrative_layout_parameter_selection"
    assert result["disassembly"]["moves"] == len(json.loads((tmp_path/"merged/disassembly.json").read_text())["moves"])
    merged = json.loads((tmp_path/"merged/structure.json").read_text())
    assert merged["references"] == structure["references"]
    assert merged["parts"][0]["assembly_path"] == structure["parts"][2]["assembly_path"]
    assert json.loads((tmp_path/"merged/bom.json").read_text())["component_count"] == result["part_count"]
    renamed = tmp_path/"renamed"
    rename_run(tmp_path/"merged", {merged["parts"][0]["id"]: "人工核对后的部件名称"}, renamed, "merged")
    with ZipFile(renamed/"parts.zip") as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        assert json.loads(archive.read("bom.json"))["components"][0]["name"] == "人工核对后的部件名称"
    assert json.loads((directory/"run.json").read_text())["part_count"] == summary["part_count"]


def test_catalogued_demo_is_served_by_local_backend(phone, monkeypatch, tmp_path):
    from qamg import server

    monkeypatch.setattr(server, "PROJECT_ROOT", tmp_path)
    with TestClient(server.create_app(tmp_path/"runs", web_root=tmp_path/"no-web")) as client:
        entries = client.get("/api/runs").json()
        assert any(e["id"] == "example-iphone" for e in entries)
        assert client.get("/api/runs/example-iphone/assembly.glb").content[:4] == b"glTF"
        assert client.get("/api/runs/example-iphone/demo.json").json()["kind"] == "procedural_demo"
        assert client.get("/api/runs/example-iphone/bom.csv").status_code == 200
        assert client.get("/api/runs/example-iphone/references.json").json()["oem_cad"] is False
        assert client.post("/api/runs/example-iphone/container-fit").status_code == 422


@pytest.mark.parametrize(("slug", "required"), [
    ("aero-engine", ["风扇叶片 01", "第 1 级静子导叶 01", "低压内轴", "高压空心外轴", "外火焰筒"]),
    ("airliner", ["机身环形隔框 01", "左翼前翼梁", "右翼肋 01", "方向舵"]),
    ("luxury-liner", ["水密舱壁 1", "同步发电机 1", "左舷推进轴", "右舷螺旋桨叶片 1"]),
    ("model-y", ["前右制动盘", "后左制动卡钳", "前左减振活塞杆", "前右悬架螺旋弹簧"]),
    ("iphone", ["TrueDepth 相机总成", "Taptic Engine", "USB-C 连接器模组", "主麦克风"]),
    ("optimus", ["右食指第 1 指骨", "左拇指第 2 指骨", "右膝定子绕组", "左踝编码器码盘"]),
    ("falcon", ["Merlin 09 喷管", "MVac 真空喷管", "一级液氧贯穿输送管", "二级共底"]),
    ("mechanical-watch", ["主发条", "擒纵叉 · 通用拓扑", "游丝 · 曲线近似", "自动上链摆陀", "日历环 · 刻字不独立计件"]),
    ("refinery", ["常压蒸馏塔取样筛板塔盘 01", "换热器 1 取样传热管 01", "泵 1 机械密封"]),
    ("industrial-arm", ["J1 基座回转 · 柔轮", "J6 工具回转 · 波发生器", "J3 肘关节 · 编码器码盘", "同步传动齿轮"]),
])
def test_reference_models_resolve_internal_components_and_hierarchy(slug, required):
    model = BUILDERS[slug]()
    labels = {part.label for part in model.parts}
    assert labels >= set(required)
    assert len(model.parts) >= 45
    for part in model.parts:
        assert part.solid.volume() > 0 and not part.solid.is_empty()
        # Closed internal cavities have negative signed volume. Count only
        # substantial positive solids, so a hollow casing is accepted while
        # disconnected teeth or welded-frame islands are rejected.
        positive = [piece for piece in part.solid.decompose()
                    if piece.volume() > max(1e-8, part.solid.volume()*1e-6)]
        assert len(positive) == 1, part.label
        assert part.group.count(" / ") == 1
        assert part.group.split(" / ")[0] in model.groups
    if slug == "falcon":
        roles = [part.role for part in model.parts]
        assert roles.count("first_stage_engine_nozzle") == 9
        assert roles.count("second_stage_engine_nozzle") == 1
        assert roles.count("landing_leg_primary") == 4
        assert roles.count("grid_fin") == 4
    if slug == "model-y":
        wheels = {part.label: part.solid.bounding_box() for part in model.parts if part.label.endswith("轮胎")}
        def center(label, axis):
            bounds = wheels[label]
            return (bounds[axis]+bounds[axis+3])/2
        # Match the referenced classic-model axle/track proportions, preventing
        # a generic SUV with the wrong wheel spacing from replacing this model.
        assert (center("前左轮胎", 0)-center("后左轮胎", 0))*25 == pytest.approx(2890, abs=.1)
        assert (center("前左轮胎", 1)-center("前右轮胎", 1))*25 == pytest.approx(1636, abs=.1)
