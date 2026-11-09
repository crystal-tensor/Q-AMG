"""Original categorical design problem and independently checkable QUBO encoding.

Bit order is always x0, x1, ... (qubit zero at the LEFT of saved bitstrings).
Only backend adapters translate provider-specific bit order.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from .schema import ChoiceGroup, DesignSpec, ImagePlanningContext, Option, PairRule, UnaryRule


@dataclass
class Qubo:
    linear: np.ndarray
    quadratic: dict[tuple[int, int], float]
    offset: float
    groups: list[list[int]]

    @property
    def n(self) -> int:
        return len(self.linear)

    def energy(self, bits: Iterable[int] | str) -> float:
        x = np.fromiter((int(b) for b in bits), dtype=float)
        if x.shape != (self.n,) or not np.isin(x, [0, 1]).all():
            raise ValueError("Bitstring has wrong width or values")
        return float(self.offset + self.linear @ x + sum(
            value * x[i] * x[j] for (i, j), value in self.quadratic.items()
        ))

    def ising(self) -> tuple[float, np.ndarray, dict[tuple[int, int], float]]:
        # xi=(1-Zi)/2; offset is retained to verify physical energies.
        h = -self.linear / 2
        jz: dict[tuple[int, int], float] = {}
        offset = self.offset + float(self.linear.sum()) / 2
        for (i, j), value in self.quadratic.items():
            h[i] -= value / 4
            h[j] -= value / 4
            jz[i, j] = value / 4
            offset += value / 4
        return offset, h, jz

    def as_dict(self) -> dict:
        return {"linear": self.linear.tolist(), "quadratic": [
            {"i": i, "j": j, "value": v} for (i, j), v in self.quadratic.items()
        ], "offset": self.offset, "groups": self.groups, "ancillas": 0,
            "bit_order": "qubit0_left"}

    def compact_binary(self) -> Qubo:
        """Exactly eliminate one-hot redundancy: xa=1-b, xb=b for each pair.

        This is a representation change of the SAME original design problem,
        not a reduction in the number of design choices or hidden classical solving.
        """
        if any(len(g) != 2 for g in self.groups):
            raise ValueError("Compact binary encoding requires two options per group")
        mapping = {i: (group_index, 1 if choice == 0 else 0, -1 if choice == 0 else 1)
                   for group_index, ids in enumerate(self.groups) for choice, i in enumerate(ids)}
        linear = np.zeros(len(self.groups))
        quadratic: dict[tuple[int, int], float] = {}
        offset = self.offset
        for i, coefficient in enumerate(self.linear):
            g, constant, sign = mapping[i]
            offset += coefficient*constant
            linear[g] += coefficient*sign
        for (i, j), coefficient in self.quadratic.items():
            g, a, s = mapping[i]
            h, b, t = mapping[j]
            offset += coefficient*a*b
            linear[g] += coefficient*s*b
            linear[h] += coefficient*a*t
            if g == h:
                linear[g] += coefficient*s*t
            else:
                key = tuple(sorted([g, h]))
                quadratic[key] = quadratic.get(key, 0)+coefficient*s*t
        return Qubo(linear, {k: v for k, v in quadratic.items() if abs(v) > 1e-12},
                    float(offset), [[i] for i in range(len(self.groups))])


@dataclass
class StructureProblem:
    spec: DesignSpec | ImagePlanningContext
    groups: list[ChoiceGroup]
    pairs: list[PairRule] = field(default_factory=list)
    unary: list[UnaryRule] = field(default_factory=list)

    @property
    def variables(self) -> list[list[int]]:
        result, offset = [], 0
        for group in self.groups:
            result.append(list(range(offset, offset + len(group.options))))
            offset += len(group.options)
        return result

    @property
    def penalty(self) -> int:
        # A violation costs more than the entire possible range of the soft objective.
        return 1 + sum(abs(o.cost) for g in self.groups for o in g.options) + sum(
            abs(p.cost) for p in self.pairs if not p.forbidden
        )

    def bits(self, selection: Iterable[int]) -> str:
        choices = list(selection)
        if len(choices) != len(self.groups):
            raise ValueError("Wrong number of choices")
        return "".join("1" if j == choice else "0"
                       for g, choice in zip(self.groups, choices, strict=True)
                       for j in range(len(g.options)))

    def selection(self, bits: str) -> tuple[int, ...] | None:
        if len(bits) != sum(len(g.options) for g in self.groups) or set(bits) - {"0", "1"}:
            raise ValueError("Malformed bitstring")
        result = []
        for indices in self.variables:
            selected = [j for j, i in enumerate(indices) if bits[i] == "1"]
            if len(selected) != 1:
                return None
            result.append(selected[0])
        return tuple(result)

    def violations(self, selection: tuple[int, ...]) -> list[str]:
        if len(selection) != len(self.groups) or any(
            c not in range(len(g.options)) for c, g in zip(selection, self.groups, strict=True)
        ):
            return ["invalid_selection"]
        result = [r.reason for r in self.unary if selection[r.group] == r.option]
        result += [p.reason for p in self.pairs if p.forbidden
                   and selection[p.left[0]] == p.left[1]
                   and selection[p.right[0]] == p.right[1]]
        return result

    def cost(self, selection: tuple[int, ...]) -> float:
        return float(sum(g.options[c].cost for g, c in zip(self.groups, selection, strict=True))
                     + sum(p.cost for p in self.pairs if not p.forbidden
                           and selection[p.left[0]] == p.left[1]
                           and selection[p.right[0]] == p.right[1]))

    def feasible(self, bits: str) -> bool:
        s = self.selection(bits)
        return s is not None and not self.violations(s)

    def original_energy(self, bits: str) -> float:
        # Direct constraint expressions, not the QUBO polynomial.
        x = list(map(int, bits))
        if len(x) != sum(len(g.options) for g in self.groups) or any(b not in [0, 1] for b in x):
            raise ValueError("Malformed bitstring")
        indices = self.variables
        value = sum(g.options[k].cost * x[i] for g, ids in zip(self.groups, indices, strict=True)
                    for k, i in enumerate(ids))
        value += self.penalty * sum((sum(x[i] for i in ids) - 1)**2 for ids in indices)
        for p in self.pairs:
            value += (self.penalty if p.forbidden else p.cost) * (
                x[indices[p.left[0]][p.left[1]]] * x[indices[p.right[0]][p.right[1]]])
        value += sum(self.penalty * x[indices[u.group][u.option]] for u in self.unary)
        return float(value)

    def encode(self) -> Qubo:
        indices = self.variables
        linear = np.zeros(sum(map(len, indices)))
        quadratic: dict[tuple[int, int], float] = {}

        def pair(i: int, j: int, value: float) -> None:
            if i == j:
                linear[i] += value
            else:
                key = tuple(sorted([i, j]))
                quadratic[key] = quadratic.get(key, 0) + value

        for g, ids in zip(self.groups, indices, strict=True):
            for k, i in enumerate(ids):
                linear[i] += g.options[k].cost - self.penalty
                for j in ids[k + 1:]:
                    pair(i, j, 2 * self.penalty)
        for p in self.pairs:
            pair(indices[p.left[0]][p.left[1]], indices[p.right[0]][p.right[1]],
                 self.penalty if p.forbidden else p.cost)
        for u in self.unary:
            linear[indices[u.group][u.option]] += self.penalty
        return Qubo(linear, quadratic, len(indices) * self.penalty, indices)

    def assignments(self):
        return product(*(range(len(g.options)) for g in self.groups))

    def describe(self, bits: str) -> dict:
        selection = self.selection(bits)
        if selection is None:
            raise ValueError("Cannot decode a non-one-hot state")
        return {g.key: {"index": c, **g.options[c].model_dump()}
                for g, c in zip(self.groups, selection, strict=True)}

    def as_dict(self) -> dict:
        return {"spec": self.spec.model_dump(), "groups": [g.model_dump() for g in self.groups],
                "pairs": [p.model_dump() for p in self.pairs],
                "unary": [u.model_dump() for u in self.unary], "penalty": self.penalty,
                "qubo": self.encode().as_dict()}


def build_problem(spec: DesignSpec) -> StructureProblem:
    heavy = spec.load == "heavy"
    small_cost, large_cost = (12, 3) if heavy else (3, 9)

    def group(key, label, choices):
        return ChoiceGroup(key=key, label=label, options=[Option(
            key=c[0], label=c[1], cost=c[2], attributes=c[3]) for c in choices])

    if spec.family == "drive":
        groups = [
            group("shaft", "轴径", [("slim", "Ø8 mm 轴", small_cost, {"radius": 4}),
                                    ("stiff", "Ø12 mm 轴", large_cost, {"radius": 6})]),
            group("bearing", "轴承规格", [("608", "小孔径轴承", 4, {"bore": 4.25}),
                                         ("6001", "大孔径轴承", 8, {"bore": 6.25})]),
            group("gear", "齿轮轮廓", [("compact", "紧凑齿轮", 5 if not heavy else 13,
                                      {"radius": 14, "teeth": 16}),
                                     ("torque", "大直径齿轮", 11 if not heavy else 4,
                                      {"radius": 18, "teeth": 20})]),
            group("housing", "轴承座层级", [("block", "整体轴承座", 5, {"split": False}),
                                          ("split", "分体轴承座", 8, {"split": True})]),
            group("guard", "防护组件", [("open", "开放结构", 1, {"present": False}),
                                       ("ring", "环形防护", 7, {"present": True})]),
        ][:spec.groups]
        pairs = [PairRule(left=(0, 0), right=(1, 1), forbidden=True, reason="bearing_fit_loose"),
                 PairRule(left=(0, 1), right=(1, 0), forbidden=True, reason="bearing_bore_too_small"),
                 PairRule(left=(0, 0), right=(2, 1), cost=7, reason="torque_on_slim_shaft")]
    else:
        groups = [
            group("jaw", "夹爪宽度", [("narrow", "28 mm 夹爪", small_cost, {"width": 28}),
                                    ("wide", "40 mm 夹爪", large_cost, {"width": 40})]),
            group("frame", "导向间距", [("compact", "34 mm 导向间距", 3, {"inner": 34}),
                                       ("wide", "46 mm 导向间距", 9, {"inner": 46})]),
            group("actuator", "驱动组件", [("screw", "丝杆驱动", 4 if not heavy else 10,
                                           {"type": "screw"}),
                                          ("piston", "活塞式示意驱动", 10 if not heavy else 4,
                                           {"type": "piston"})]),
            group("guide", "导轨组合", [("dual", "双导轨", 6, {"type": "dual"}),
                                       ("central", "中央导轨", 3, {"type": "central"})]),
            group("pad", "夹持面", [("flat", "平面夹垫", 2, {"type": "flat"}),
                                    ("grooved", "带槽夹垫", 6, {"type": "grooved"})]),
        ][:spec.groups]
        pairs = [PairRule(left=(0, 1), right=(1, 0), forbidden=True, reason="jaw_guide_clearance"),
                 PairRule(left=(0, 0), right=(2, 1), cost=4, reason="actuator_oversized")]
    if len(groups) > 3:
        pairs.append(PairRule(left=(2, 1), right=(3, 0), cost=3,
                              reason="assembly_serviceability"))
    for group in groups:
        if group.key in spec.cost_overrides:
            costs = spec.cost_overrides[group.key]
            if len(costs) != len(group.options) or any(c < 0 or c > 10000 for c in costs):
                raise ValueError(f"Invalid catalogue costs for {group.key}")
            for option, cost in zip(group.options, costs, strict=True):
                option.cost = cost
        if spec.compact:
            group.options[1].cost += 3
    unknown = set(spec.cost_overrides) - {g.key for g in groups}
    if unknown:
        raise ValueError(f"Cost overrides refer to inactive groups: {sorted(unknown)}")
    return StructureProblem(spec, groups, pairs)

