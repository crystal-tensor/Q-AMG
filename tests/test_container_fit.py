import numpy as np
import pytest
import trimesh

from qamg.assembly_plan import direction_problem, selected_vectors
from qamg.container_fit import assisted_lid, hollow_cup, sleeve_mesh
from qamg.mesh_parts import as_manifold, as_trimesh


def test_cup_is_hollow_continuous_and_nested_sleeve_has_clearance():
    cup = as_manifold(hollow_cup(25, 30, 80, 1))
    sleeve = as_manifold(sleeve_mesh(25, 26.25, 20, .2, 1.8))
    assert len(cup.decompose()) == len(sleeve.decompose()) == 1
    assert (cup ^ sleeve).volume() < 1e-7
    assert as_trimesh(cup).bounds[1, 2] == pytest.approx(80)
    for z in [10, 40, 75]:
        probe = as_manifold(trimesh.creation.cylinder(radius=10, height=1)).translate([0, 0, z])
        assert (cup ^ probe).volume() < 1e-7
    bottom = as_manifold(trimesh.creation.cylinder(radius=10, height=.5)).translate([0, 0, .5])
    assert (cup ^ bottom).volume() > 150
    # The wall exists both within the sleeve and above it, rather than ending
    # at a horizontal partition that substitutes the sleeve for the cup bottom.
    for z in [10, 40]:
        radius = 25+5*z/80
        probe = as_manifold(trimesh.creation.box([.2, .2, .2])).translate([radius-.5, 0, z])
        assert (cup ^ probe).volume() > .007


def test_explicit_anchor_keeps_cup_stationary_and_rejects_unknown_part():
    parts = [as_manifold(trimesh.creation.box([20, 20, 4])).translate([0, 0, 82]),
             as_manifold(hollow_cup(25, 30, 80, 1)),
             as_manifold(sleeve_mesh(25, 26.25, 20, .2, 1.8))]
    problem, choices, anchor = direction_problem(parts, "synthetic-contract", 42, anchor_hint=1)
    assert anchor == 1
    vectors = selected_vectors(problem, problem.bits((0, 0)), choices, 3)
    assert vectors[1] == [0., 0., 0.]
    assert all(np.linalg.norm(vectors[i]) > 0 for i in [0, 2])
    with pytest.raises(ValueError, match="固定部件"):
        direction_problem(parts, "synthetic-contract", 42, anchor_hint=3)


def test_cap_rim_and_tab_form_one_part_above_the_hollow_cup():
    cup = as_manifold(hollow_cup(25, 30, 80, 1))
    cap = assisted_lid(33, 80, 100)
    assert len(cap.decompose()) == 1
    assert cap.volume() > 0 and (cap ^ cup).volume() < 1e-7
    assert as_trimesh(cap).bounds[0, 2] == pytest.approx(80)
