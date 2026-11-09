# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is part of QuantumBridge SDK.
# This implementation is developed for the QuantumBridge native architecture.

from pathlib import Path
from typing import Union


def validate_attribution_files(root: Union[str, Path] = ".") -> dict[str, bool]:
    base = Path(root)
    checks = {
        "third_party_notices": base.joinpath("THIRD_PARTY_NOTICES.md").is_file(),
        "migration_ledger": base.joinpath("docs/migration/source_migration_ledger.md").is_file(),
        "apache_license": base.joinpath("LICENSES/Apache-2.0.txt").is_file(),
        "qiskit_license": base.joinpath("LICENSES/QISKIT_LICENSE.txt").is_file(),
        "pennylane_license": base.joinpath("LICENSES/PENNYLANE_LICENSE.txt").is_file(),
    }
    return checks
