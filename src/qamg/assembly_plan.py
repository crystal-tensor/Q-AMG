"""QB-selected assembly display directions for jointly generated neural parts.

The neural part boundaries are preserved. Collision sampling supplies objective
coefficients; the selected QB measurement determines the exported directions.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .mesh_parts import as_trimesh
from .problem import StructureProblem
from .schema import ChoiceGroup, ImagePlanningContext, Option, PairRule


def _overlap(a, b, a_bounds, b_bounds, translation):
    shifted = a_bounds+translation
    overlap = np.minimum(shifted[1], b_bounds[1])-np.maximum(shifted[0], b_bounds[0])
    if np.any(overlap <= 1e-7):
        return 0.
    return max(0., (a.translate(translation) ^ b).volume())


def direction_problem(parts, image_sha: str, seed: int, *, anchor_hint: int | None = None):
    meshes = [as_trimesh(p) for p in parts]
    bounds = np.array([m.bounds for m in meshes])
    centers = np.array([m.centroid for m in meshes])
    if anchor_hint is not None and not 0 <= anchor_hint < len(parts):
        raise ValueError("固定部件不在当前装配中")
    anchor = int(np.argmax([p.volume() for p in parts])) if anchor_hint is None else anchor_hint
    span = float(np.linalg.norm(bounds[:, 1].max(axis=0)-bounds[:, 0].min(axis=0)))
    distance = max(25., span*1.25)
    candidates, groups = {}, []
    for i, part in enumerate(parts):
        if i == anchor:
            continue
        outward = centers[i]-centers[anchor]
        norm = float(np.linalg.norm(outward))
        outward = outward/norm if norm > 1e-7 else np.array([0., 0., 1.])
        directions = []
        for axis in range(3):
            for sign in [1., -1.]:
                direction = np.eye(3)[axis]*sign
                translation = direction*distance
                fractions = np.linspace(1/12, 1, 12)
                intersections = [sum(_overlap(part, other, bounds[i], bounds[j], translation*t)
                                     for j, other in enumerate(parts) if j != i) for t in fractions]
                fraction = float(np.mean(intersections)/part.volume())
                cost = round(1000*fraction+12*(1-float(direction @ outward)))
                label = f"{'XYZ'[axis]}{'+' if sign > 0 else '−'}"
                directions.append({"translation_mm": translation.tolist(), "label": label,
                                   "collision_fraction_mean": fraction, "cost": cost,
                                   "samples": len(fractions)})
        chosen = sorted(directions, key=lambda d: (d["cost"], d["label"]))[:2]
        candidates[i] = chosen
        groups.append(ChoiceGroup(key=f"part_{i}_direction", label=f"部件 {i+1:02d} 展开方向",
                                  options=[Option(key=f"direction_{j}", label=d["label"], cost=d["cost"],
                                                  attributes=d) for j, d in enumerate(chosen)]))
    pairs = []
    indices = list(candidates)
    for left, right in combinations(range(len(indices)), 2):
        i, j = indices[left], indices[right]
        for a in range(2):
            for b in range(2):
                delta_i = np.array(candidates[i][a]["translation_mm"])
                delta_j = np.array(candidates[j][b]["translation_mm"])
                minimum_gap = float(np.linalg.norm(centers[i]+delta_i-centers[j]-delta_j))
                penalty = round(15*max(0., 1-minimum_gap/distance))
                if penalty:
                    pairs.append(PairRule(left=(left, a), right=(right, b), cost=penalty,
                                          reason="exploded_part_crowding"))
    context = ImagePlanningContext(seed=seed, image_sha256=image_sha, desired_parts=len(parts),
                                   source="Part-aware neural parts; QB selects assembly display directions, without cutting parts")
    return StructureProblem(context, groups, pairs), candidates, anchor


def selected_vectors(problem, bits: str, candidates: dict, count: int) -> list[list[float]]:
    vectors = [[0., 0., 0.] for _ in range(count)]
    for index, option in zip(candidates, problem.selection(bits), strict=True):
        vectors[index] = candidates[index][option]["translation_mm"]
    return vectors


def selected_disassembly(parts, ids: list[str], vectors: list[list[float]], anchor: int, samples: int = 24):
    """Find an order using only the measured directions; never substitute a new direction."""
    bounds = np.array([as_trimesh(p).bounds for p in parts])
    remaining = set(range(len(parts)))
    moves, checks = [], 0
    tolerance = max(1e-7, sum(p.volume() for p in parts)*1e-7)
    while len(remaining) > 1:
        found = False
        for i in sorted(remaining-{anchor}):
            delta = np.asarray(vectors[i])
            if np.linalg.norm(delta) < 1e-7:
                continue
            clear = True
            others = np.asarray(sorted(remaining-{i}))
            for t in np.linspace(1/samples, 1, samples):
                # Broad phase in one array operation: dense reference assemblies
                # otherwise spend most of their time testing distant tiny parts.
                shifted = bounds[i]+delta*t
                overlap = np.minimum(shifted[1], bounds[others, 1])-np.maximum(shifted[0], bounds[others, 0])
                checks += len(others)
                for j in others[np.all(overlap > 1e-7, axis=1)]:
                    if _overlap(parts[i], parts[j], bounds[i], bounds[j], delta*t) > tolerance:
                        clear = False
                        break
                if not clear:
                    break
            if clear:
                moves.append({"part": ids[i], "translation_mm": delta.tolist(), "samples": samples})
                remaining.remove(i)
                found = True
                break
        if not found:
            break
    complete = remaining == {anchor}
    return {"moves": moves, "complete_for_generated_mesh": complete,
            "complete_for_envelope_model": complete, "anchor_part": ids[anchor],
            "remaining_parts": [ids[i] for i in sorted(remaining-{anchor})],
            "sampled_intersection_checks": checks, "samples_per_translation": samples,
            "directions_source": "selected_structure_measurement", "continuous_collision_proof": False,
            "physical_assembly_verified": False,
            "scope": "Discrete translations in QB-selected directions; neural geometry and manufacturing joints remain inferred"}
