# Copyright 2026 QuantumBridge Contributors. Apache-2.0.
"""Small industrial models, independently enumerated and natively sampled."""

from quantumbridge.finance.kernels import labels
from .common import optimization_case


def scheduling_case():
    candidates = []
    for bits in labels(4):
        x = list(map(int, bits))
        starts = [x[0], 1 + x[2], x[1], 1 + x[3]]
        operations = [
            {"name": name, "machine": machine, "start": start, "duration": 1}
            for name, machine, start in zip(
                ("A1", "A2", "B1", "B2"), (0, 1, 0, 1), starts
            )
        ]
        precedence = int(starts[1] < starts[0] + 1) + int(starts[3] < starts[2] + 1)
        conflicts = int(starts[0] == starts[2]) + int(starts[1] == starts[3])
        objective = 2 * (starts[1] + 1) + (starts[3] + 1)
        candidates.append(
            {
                "bitstring": bits,
                "operations": operations,
                "precedence_violations": precedence,
                "machine_conflicts": conflicts,
                "feasible": precedence + conflicts == 0,
                "weighted_completion": objective,
                "makespan": max(starts) + 1,
                "objective": objective,
                "penalized_cost": objective + 10 * (precedence + conflicts),
            }
        )
    return optimization_case(
        "routing-scheduling",
        "industry",
        "Two-job two-machine scheduling",
        {
            "jobs": ["A", "B"],
            "machine_order": [0, 1],
            "operation_duration": 1,
            "start_windows": {"first_operation": [0, 1], "second_operation": [1, 2]},
            "completion_weights": {"A": 2, "B": 1},
            "penalty": 10,
            "scope": "fixed four-operation job shop; no vehicle routing or TSP claim",
        },
        candidates,
    )


def supply_case():
    candidates = []
    cost = [2, 5, 4, 1]
    for bits in labels(4):
        x = list(map(int, bits))
        supplied = [x[0] + x[1], x[2] + x[3]]
        received = [x[0] + x[2], x[1] + x[3]]
        violation = sum((value - 1) ** 2 for value in received) + sum(
            max(0, value - 1) ** 2 for value in supplied
        )
        objective = sum(c * b for c, b in zip(cost, x))
        candidates.append(
            {
                "bitstring": bits,
                "shipping_matrix": [x[:2], x[2:]],
                "supplier_load": supplied,
                "customer_received": received,
                "violations": violation,
                "feasible": violation == 0,
                "objective": objective,
                "penalized_cost": objective + 20 * violation,
            }
        )
    return optimization_case(
        "supply-chain",
        "industry",
        "Two-supplier two-customer allocation",
        {
            "shipping_cost": [cost[:2], cost[2:]],
            "supplier_capacity": [1, 1],
            "customer_demand": [1, 1],
            "bit_order": ["S0-C0", "S0-C1", "S1-C0", "S1-C1"],
            "penalty": 20,
            "units": "one indivisible shipment per selected edge; synthetic costs",
        },
        candidates,
    )


def energy_case():
    candidates = []
    for bits in labels(4):
        x = list(map(int, bits))
        status = [x[:2], x[2:]]
        available = [2 * row[0] + row[1] for row in status]
        deficit = [
            max(0, demand - capacity) for demand, capacity in zip((2, 3), available)
        ]
        starts = sum(status[0]) + sum(
            int(status[1][i] and not status[0][i]) for i in range(2)
        )
        cost = sum(3 * row[0] + 2 * row[1] for row in status) + starts
        violation = sum(value**2 for value in deficit)
        candidates.append(
            {
                "bitstring": bits,
                "commitment_by_time": status,
                "available_capacity": available,
                "capacity_deficit": deficit,
                "startups": starts,
                "feasible": violation == 0,
                "objective": cost,
                "penalized_cost": cost + 20 * violation,
            }
        )
    return optimization_case(
        "energy",
        "industry",
        "Two-period generating-capacity commitment",
        {
            "generator_capacity": [2, 1],
            "on_cost_per_period": [3, 2],
            "startup_cost": 1,
            "demand": [2, 3],
            "initial_status": [0, 0],
            "penalty": 20,
            "scope": "capacity adequacy only; no dispatch, ramping, reserve, storage or AC power flow",
        },
        candidates,
    )


def maxcut_case():
    edges = [(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 0, 2), (0, 2, 0.5)]
    candidates = []
    for bits in labels(4):
        cut = sum(weight for a, b, weight in edges if bits[a] != bits[b])
        candidates.append(
            {
                "bitstring": bits,
                "partition0": [i for i, bit in enumerate(bits) if bit == "0"],
                "cut_weight": cut,
                "feasible": True,
                "objective": -cut,
                "penalized_cost": -cut,
            }
        )
    return optimization_case(
        "combinatorial",
        "industry",
        "Weighted four-node MaxCut",
        {
            "nodes": 4,
            "edges": edges,
            "objective": "minimize minus cut weight",
            "penalty": 0,
        },
        candidates,
    )


def integer_case():
    candidates = []
    weight = 1.2
    for bits in labels(4):
        a, b = int(bits[:2], 2), int(bits[2:], 2)
        resource, profit, emissions = 2 * a + b, 5 * a + 2 * b, 4 * a + b
        cost = -profit + weight * emissions
        candidates.append(
            {
                "bitstring": bits,
                "allocation_A": a,
                "allocation_B": b,
                "resource_use": resource,
                "profit": profit,
                "emissions": emissions,
                "feasible": resource <= 5,
                "objective": cost,
                "penalized_cost": cost + 20 * max(0, resource - 5) ** 2,
            }
        )
    feasible = [row for row in candidates if row["feasible"]]
    pareto = [
        row
        for row in feasible
        if not any(
            other["profit"] >= row["profit"]
            and other["emissions"] <= row["emissions"]
            and (
                other["profit"] > row["profit"] or other["emissions"] < row["emissions"]
            )
            for other in feasible
        )
    ]
    return optimization_case(
        "integer-multiobjective",
        "industry",
        "Bounded integer profit-emissions trade-off",
        {
            "variables": {"A": [0, 1, 2, 3], "B": [0, 1, 2, 3]},
            "constraint": "2*A+B <= 5",
            "profit": "5*A+2*B",
            "emissions": "4*A+B",
            "scalarization_weight": weight,
            "objective": "-profit + weight*emissions",
            "penalty": 20,
        },
        candidates,
        intermediate={
            "pareto_front": pareto,
            "pareto_source": "complete classical feasible enumeration, not a quantum Pareto solver",
        },
    )


BUILDERS = {
    "routing-scheduling": scheduling_case,
    "supply-chain": supply_case,
    "energy": energy_case,
    "combinatorial": maxcut_case,
    "integer-multiobjective": integer_case,
}


def build_industry_case(direction):
    try:
        builder = BUILDERS[direction]
    except KeyError as error:
        raise ValueError("Unknown industrial direction.") from error
    return builder()
