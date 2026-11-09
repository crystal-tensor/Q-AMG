# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
"""Optional ecosystem integration helpers for QuantumBridge.

Design source: docs/compat/full_ecosystem_coverage_policy_v0.1.md.
"""

from quantumbridge.ecosystem.capability import CapabilityRecord, CoverageLevel
from quantumbridge.ecosystem.catalog import EcosystemCatalog, EcosystemProject, make_catalog
from quantumbridge.ecosystem.dependency import dependency_available, get_upstream_version
from quantumbridge.ecosystem.industry_taxonomy import (
    INDUSTRY_CATEGORIES,
    SHARED_FOUNDATION,
    TAXONOMY_VERSION,
    IndustryCapability,
    IndustryCategory,
    industry_capabilities,
    industry_categories,
    industry_taxonomy,
    validate_industry_taxonomy,
)
from quantumbridge.ecosystem.provenance import provenance_metadata
from quantumbridge.ecosystem.registry import EcosystemAdapter

__all__ = [
    "CapabilityRecord",
    "CoverageLevel",
    "EcosystemCatalog",
    "EcosystemAdapter",
    "EcosystemProject",
    "INDUSTRY_CATEGORIES",
    "IndustryCapability",
    "IndustryCategory",
    "SHARED_FOUNDATION",
    "TAXONOMY_VERSION",
    "dependency_available",
    "get_upstream_version",
    "industry_capabilities",
    "industry_categories",
    "industry_taxonomy",
    "make_catalog",
    "provenance_metadata",
    "validate_industry_taxonomy",
]
