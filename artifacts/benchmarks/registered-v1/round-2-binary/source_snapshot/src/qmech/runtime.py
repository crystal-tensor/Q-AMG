from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess

from qiskit_aer import AerSimulator


def environment_report() -> dict:
    devices = list(AerSimulator().available_devices())
    memory_bytes = None
    gpu_description = None
    if platform.system() == "Darwin":
        p = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False)
        if p.returncode == 0:
            memory_bytes = int(p.stdout.strip())
        gpu_description = "Apple GPU; Aer requires a CUDA backend for GPU simulation" if platform.machine() == "arm64" else None
    return {"platform": platform.platform(), "architecture": platform.machine(),
            "memory_bytes": memory_bytes, "aer_devices": devices,
            "selected_simulator_device": "GPU" if "GPU" in devices else "CPU",
            "gpu_note": gpu_description,
            "hardware_adapter": "superconducting gate-model reference contract; not connected",
            "packages": {name: importlib.metadata.version(name) for name in
                         ["qmech", "qiskit", "qiskit-aer", "cadquery", "numpy", "scipy", "ortools"]}}


def main():
    print(json.dumps(environment_report(), indent=2))

