# This file is independently implemented for QuantumBridge SDK.
"""Studio backend API boundary warnings."""

STUDIO_BACKEND_WARNING = (
    "QuantumBridge Studio backend API is a local executable service layer. "
    "It is not a production API server, not a frontend UI, and not an official "
    "IBM, Qiskit, PennyLane, Benchpress, Quafu, QOS-UQCI, MQT, Mitiq, or "
    "TorchQuantum service."
)

NO_CLOUD_TOKEN_HARDWARE_WARNING = (
    "Studio workflows are local-only by default and do not access cloud services, "
    "tokens, credentials, or real hardware."
)

CLEAN_ROOM_WARNING = (
    "This Studio API slice uses QuantumBridge-owned clean-room metadata and "
    "execution wrappers; no third-party source, UI, tutorial prose, or branding "
    "is copied."
)

NO_PRODUCTION_PARITY_WARNING = (
    "This slice does not claim full replacement, production parity, official "
    "benchmark status, or official endorsement."
)


def studio_warnings(*extra: str) -> list[str]:
    return [
        STUDIO_BACKEND_WARNING,
        NO_CLOUD_TOKEN_HARDWARE_WARNING,
        CLEAN_ROOM_WARNING,
        NO_PRODUCTION_PARITY_WARNING,
        *[str(value) for value in extra if value],
    ]
