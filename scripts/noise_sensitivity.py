"""Fixed-angle characterization only; illustrative errors are NOT device calibration."""
from pathlib import Path

from qamg.backends import QBBackend
from qamg.benchmark import dump
from qamg.problem import build_problem
from qamg.quantum import qaoa_circuit
from qamg.schema import DesignSpec


def main():
    output = Path("artifacts/noise-characterization.json")
    problem = build_problem(DesignSpec(groups=3))
    original = problem.encode()
    results = []
    for encoding in ["onehot", "compact_binary"]:
        qubo = original if encoding == "onehot" else original.compact_binary()
        circuit, gamma, beta, _ = qaoa_circuit(qubo, 2, "xy" if encoding == "onehot" else "x")
        bound = circuit.assign_parameters({**dict(zip(gamma, [.4, .8], strict=True)),
                                            **dict(zip(beta, [.2, .3], strict=True))})
        for error in [0., .01, .03]:
            profile = {"name": f"illustrative_p2_{error}_NOT_DEVICE_CALIBRATION",
                       "one_qubit_error": error/10, "two_qubit_error": error,
                       "readout_error": error}
            backend = QBBackend(noisy=error > 0, device_profile=profile)
            compiled = backend.compile(bound, 42)
            batch = backend.sample(compiled, 2048, 42)
            counts = ({problem.bits(tuple(map(int, raw))): count for raw, count in batch.counts.items()}
                      if encoding == "compact_binary" else batch.counts)
            results.append({"encoding": encoding, "profile": profile,
                            "logical_qubits": qubo.n, "shots": batch.shots,
                            "onehot_valid_rate": sum(count for bits, count in counts.items()
                                                     if problem.selection(bits) is not None)/batch.shots,
                            "original_feasible_rate": sum(count for bits, count in counts.items()
                                                          if problem.feasible(bits))/batch.shots,
                            "mean_original_energy": sum(original.energy(bits)*count
                                                        for bits, count in counts.items())/batch.shots,
                            "raw_counts": batch.counts, "metadata": batch.metadata,
                            "execution_source": batch.execution_source})
    dump(output, {"fixed_angles": {"gamma": [.4, .8], "beta": [.2, .3]}, "results": results,
                   "scope": "One bounded assembly case; fixed angles; no tuning; synthetic stochastic errors; not company device prediction"})
    for result in results:
        print(result["encoding"], result["profile"]["two_qubit_error"],
              result["onehot_valid_rate"], result["original_feasible_rate"])


if __name__ == "__main__":
    main()
