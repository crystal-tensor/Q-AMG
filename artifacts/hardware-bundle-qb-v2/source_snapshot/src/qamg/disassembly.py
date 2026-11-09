"""Greedy, sampled B-rep translation checks; never a continuous-motion proof.

The base is fixed. Fasteners, friction, threads, loads, tools and robot kinematics
are outside this bounded envelope model. Exploded illustrations are separate.
"""
from __future__ import annotations

import time

import numpy as np

from .geometry import AssemblyModel, bbox_overlap


def plan_disassembly(model: AssemblyModel, samples: int = 24) -> dict:
    start = time.perf_counter()
    remaining = {p.id: p for p in model.parts if p.id != "base"}
    obstacles = {p.id: p for p in model.parts}
    coordinates = np.array([p.metadata()["bounds_mm"] for p in model.parts])
    distance = float(np.linalg.norm(coordinates[:, 1].max(axis=0)-coordinates[:, 0].min(axis=0))*1.5+20)
    moves, checked = [], 0
    while remaining:
        found = False
        for part in reversed(list(remaining.values())):
            preferred = np.asarray(part.explode)
            axis = int(np.argmax(np.abs(preferred)))
            sign = 1 if preferred[axis] >= 0 else -1
            directions = [(axis, sign)] + [(i, s) for i in range(3) for s in [-1, 1]
                                           if (i, s) != (axis, sign)]
            for ax, direction in directions:
                delta = np.zeros(3)
                delta[ax] = direction*distance
                collision = False
                for fraction in np.linspace(0, 1, samples+1)[1:]:
                    moved = part.shape.translate(tuple(delta*fraction))
                    for other in obstacles.values():
                        if other.id == part.id or not bbox_overlap(moved, other.shape):
                            continue
                        checked += 1
                        if moved.intersect(other.shape).Volume() > 1e-5*model.problem.spec.scale**3:
                            collision = True
                            break
                    if collision:
                        break
                if not collision:
                    moves.append({"part": part.id, "translation_mm": list(map(float, delta)),
                                  "samples": samples, "tested_against": sorted(k for k in obstacles if k != part.id)})
                    del obstacles[part.id]
                    del remaining[part.id]
                    found = True
                    break
            if found:
                break
        if not found:
            break
    return {"method": "sampled_brep_translation", "samples_per_move": samples,
            "fixed_parts": ["base"], "moves": moves, "unresolved_parts": sorted(remaining),
            "complete_for_envelope_model": not remaining, "continuous_collision_proof": False,
            "thread_tool_force_or_robot_validated": False, "checked_intersections": checked,
            "elapsed_s": time.perf_counter()-start,
            "warning": "Discrete collision samples can miss collisions between samples; not industrial disassembly certification."}

