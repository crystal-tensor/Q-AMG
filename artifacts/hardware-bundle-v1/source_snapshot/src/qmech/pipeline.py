from __future__ import annotations

import json
import time
from pathlib import Path

from .backends import CircuitBackend, make_backend
from .classical import solve_classical
from .geometry import decode_geometry, export_model, validate_geometry
from .problem import build_problem
from .quantum import QuantumStructurePrior
from .schema import DesignSpec, PairRule, PlannerSettings, UnaryRule


class GenerationFailure(RuntimeError):
    def __init__(self, message: str, trace: list[dict]):
        super().__init__(message)
        self.trace = trace


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def generate(spec: DesignSpec, settings: PlannerSettings, output: str | Path,
             backend: CircuitBackend | None = None) -> dict:
    start = time.perf_counter()
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    problem = build_problem(spec)
    _write_json(directory/"initial_problem.json", problem.as_dict())
    if settings.backend != "classical":
        backend = backend or make_backend(settings.backend, settings.max_qubits)
    initialized = time.perf_counter()
    trace = []
    geometry_s, validation_s, planning_s = 0., 0., 0.
    selected_model = None
    selected_candidate = None
    for round_index in range(settings.max_feedback_rounds):
        before = time.perf_counter()
        if settings.backend == "classical":
            result = solve_classical(problem, "cpsat", spec.seed+round_index,
                                     timeout_s=settings.timeout_s)
        else:
            result = QuantumStructurePrior(backend).plan(problem, settings, spec.seed+round_index)
        planning_s += time.perf_counter()-before
        round_trace = {"round": round_index, "problem": problem.as_dict(), "plan": result.as_dict(),
                       "geometry_attempts": [], "feedback_rules": []}
        if result.circuit_qasm3:
            (directory/f"round-{round_index}.qasm").write_text(result.circuit_qasm3)
        for block_index, block in enumerate(result.trace):
            if block.get("circuit_qasm3"):
                (directory/f"round-{round_index}-block-{block_index}.qasm").write_text(block["circuit_qasm3"])
        learned = False
        for candidate in [c for c in result.candidates if c.feasible][:settings.max_candidates]:
            before = time.perf_counter()
            model = decode_geometry(problem, candidate.bits)
            geometry_s += time.perf_counter()-before
            before = time.perf_counter()
            validation = validate_geometry(model)
            validation_s += time.perf_counter()-before
            round_trace["geometry_attempts"].append({"bits": candidate.bits,
                                                     "job_id": candidate.first_job_id,
                                                     "validation": validation.model_dump()})
            if validation.valid:
                selected_model, selected_candidate = model, candidate
                break
            for issue in validation.issues:
                choices = issue.culprit_choices
                if len(choices) == 1:
                    group, option = choices[0]
                    if not any(u.group == group and u.option == option for u in problem.unary):
                        rule = UnaryRule(group=group, option=option, reason=issue.code)
                        problem.unary.append(rule)
                        round_trace["feedback_rules"].append(rule.model_dump())
                        learned = True
                elif len(choices) == 2:
                    a, b = choices
                    if not any(p.forbidden and {p.left, p.right} == {a, b} for p in problem.pairs):
                        rule = PairRule(left=a, right=b, forbidden=True, reason=issue.code)
                        problem.pairs.append(rule)
                        round_trace["feedback_rules"].append(rule.model_dump())
                        learned = True
            if learned:
                # Re-encode and resample with the actual geometry feedback.
                break
        trace.append(round_trace)
        _write_json(directory/"trace.json", trace)
        if selected_model:
            break
        if not learned:
            break
    if selected_model is None:
        failure = {"status": "generation_failed", "spec": spec.model_dump(),
                   "settings": settings.model_dump(), "trace": trace,
                   "reason": "No sampled candidate passed geometry validation; no silent classical fallback"}
        _write_json(directory/"run.json", failure)
        raise GenerationFailure(failure["reason"], trace)
    validation = trace[-1]["geometry_attempts"][-1]["validation"]
    provenance = {
        "mode": settings.backend,
        "execution_source": trace[-1]["plan"]["execution_source"],
        "selected_measurement": {"bits": selected_candidate.bits,
                                 "kind": "conditioned_composite" if result.resources.get("decomposed") else "direct_or_exactly_decoded",
                                 "job_id": selected_candidate.first_job_id,
                                 "observed_count": selected_candidate.count,
                                 "count_scope": "selected_block_candidate" if result.resources.get("decomposed") else "all_batches_in_this_round",
                                 "job_id_scope": "first batch observing the candidate"},
        "fallback_used": False,
        "hardware_verified": False,
        "quantum_speedup_verified": False,
        "generation_rounds": len(trace),
    }
    before = time.perf_counter()
    from .disassembly import plan_disassembly
    disassembly = plan_disassembly(selected_model)
    disassembly_s = time.perf_counter()-before
    _write_json(directory/"disassembly.json", disassembly)
    before = time.perf_counter()
    files = export_model(selected_model, directory, {"validation": validation, "provenance": provenance,
                                                     "disassembly": disassembly})
    files["disassembly"] = "disassembly.json"
    export_s = time.perf_counter()-before
    summary = {"status": "generated", "spec": spec.model_dump(), "settings": settings.model_dump(),
               "validation": validation, "provenance": provenance,
               "selected_bits": selected_candidate.bits,
               "decisions": problem.describe(selected_candidate.bits),
               "part_count": len(selected_model.parts), "files": files,
               "disassembly": {"complete_for_envelope_model": disassembly["complete_for_envelope_model"],
                                "continuous_collision_proof": False,
                                "moves": len(disassembly["moves"])},
               "timings": {"initialization_s": initialized-start, "planning_s": planning_s,
                           "geometry_s": geometry_s, "validation_s": validation_s,
                           "disassembly_s": disassembly_s, "export_s": export_s,
                           "end_to_end_s": time.perf_counter()-start},
               "resources": trace[-1]["plan"]["resources"]}
    _write_json(directory/"run.json", summary)
    return summary
