# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or IBM ecosystem packages was copied.

import json


REQUIRED_CONTRACT = {
    "capability_level",
    "production_ready",
    "native_implementation",
    "upstream_required",
    "dependency_available",
    "get_upstream_version",
    "get_dependency_report",
    "list_public_api_inventory",
    "get_public_object",
    "passthrough_call",
    "passthrough_class",
    "wrap_result",
    "to_quantumbridge_schema",
    "get_warnings",
    "get_provenance",
    "unsupported",
    "validate_environment",
}


def assert_qiskit_contract(module, *, advisory=False, offline_only=False):
    missing = sorted(name for name in REQUIRED_CONTRACT if not hasattr(module, name))
    assert missing == []
    assert module.production_ready is False
    assert module.native_implementation is False
    assert module.upstream_required is True
    assert module.validate_environment() is True

    report = module.get_dependency_report()
    assert report["advisory"] is advisory
    assert report["offline_only"] is offline_only
    assert report["cloud_access"] is False
    assert report["token_read"] is False
    assert report["token_storage"] is False

    warnings = module.get_warnings()
    assert any("not a full native Qiskit replacement" in warning.message for warning in warnings)

    provenance = module.get_provenance()
    assert provenance.official_endorsement is False
    assert provenance.cloud_access is False
    assert provenance.token_storage is False
    assert provenance.source_code_copied is False

    unsupported = module.unsupported("contract test unsupported path").to_dict()
    assert unsupported["supported"] is False
    assert unsupported["reason"] == "contract test unsupported path"
    assert unsupported["warnings"]

    result = module.wrap_result({"ok": True})
    assert result.validate() is True
    payload = result.to_dict()
    assert payload["advisory"] is advisory
    assert payload["offline_only"] is offline_only
    assert json.loads(result.to_json())["ecosystem"] == payload["ecosystem"]

    inventory = module.list_public_api_inventory()
    assert isinstance(inventory, list)
    assert inventory
    for row in inventory[:5]:
        assert {
            "ecosystem",
            "module",
            "name",
            "object_type",
            "importable",
            "callable",
            "quantumbridge_level",
            "adapter",
            "upstream_required",
            "supported",
            "unsupported_reason",
            "advisory",
            "risk",
            "notes",
        } <= set(row)

    if module.dependency_available():
        supported = next((row for row in inventory if row["supported"] and row["name"] != "inventory-truncated"), None)
        if supported is not None:
            obj = module.get_public_object(f"{supported['module']}.{supported['name']}")
            assert obj is not None
