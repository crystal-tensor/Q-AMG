# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Structured native interchange errors; no untrusted code execution.

from quantumbridge.core.classical import UnsupportedFeature as CoreUnsupportedFeature


class SerializationError(ValueError):
    code = "serialization_error"
    def __init__(self, reason, *, format=None, version=None, line=None, column=None, offset=None, feature=None,
                 recovery="Use a supported format/version and inspect the reported input location."):
        self.reason, self.format, self.version = reason, format, version
        self.line, self.column, self.offset = line, column, offset
        self.feature, self.recovery = feature, recovery
        location = f" at line {line}, column {column}" if line is not None else f" at byte {offset}" if offset is not None else ""
        ValueError.__init__(self,f"QuantumBridge {format or 'serialization'}{location}: {reason}")

    def to_dict(self):
        return {"code":self.code, "format":self.format, "version":self.version, "line":self.line,
                "column":self.column, "offset":self.offset, "reason":self.reason,
                "feature":self.feature, "recovery":self.recovery}


class ParseError(SerializationError): code = "parse_error"
class UnsupportedVersion(SerializationError): code = "unsupported_version"
class FormatValidationError(SerializationError): code = "format_validation_error"
class ResourceLimitError(SerializationError): code = "resource_limit"
class UnsafeInputError(SerializationError): code = "unsafe_input"


class UnsupportedFeature(SerializationError, CoreUnsupportedFeature):
    code = "unsupported_feature"
