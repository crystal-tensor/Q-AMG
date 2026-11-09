# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.


class JobError(Exception):
    """Raised when a QuantumBridge backend job cannot complete."""


class JobTimeoutError(TimeoutError):
    """Raised when waiting for a QuantumBridge job exceeds a timeout."""


class QiskitBackendNotFoundError(LookupError):
    """Raised when a provider cannot find a backend matching a query."""
