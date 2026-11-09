from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess

from .torch_backend import available_gpu


def environment_report() -> dict:
    gpu = available_gpu()
    devices = ["QB CPU", *([f"QB / Torch {gpu.upper()}"] if gpu else [])]
    memory_bytes = None
    gpu_description = None
    if platform.system() == "Darwin":
        p = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False)
        if p.returncode == 0:
            memory_bytes = int(p.stdout.strip())
        gpu_description = "QuantumBridge native circuits compiled and simulated through the Torch MPS adapter" if platform.machine() == "arm64" else None
    return {"platform": platform.platform(), "architecture": platform.machine(),
            "memory_bytes": memory_bytes, "qb_devices": devices, "quantum_sdk": "QuantumBridge",
            "selected_simulator_device": gpu.upper() if gpu else "CPU",
            "gpu_note": gpu_description,
            "hardware_adapter": "superconducting gate-model reference contract; not connected",
            "packages": {name: importlib.metadata.version(name) for name in
                         ["q-amg", "quantumbridge-sdk", "cadquery", "numpy", "scipy", "ortools", "torch"]}}


def main():
    print(json.dumps(environment_report(), indent=2))

