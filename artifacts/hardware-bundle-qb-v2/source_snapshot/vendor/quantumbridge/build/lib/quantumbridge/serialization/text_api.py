# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
from .limits import SerializationLimits
from .errors import SerializationError, FormatValidationError, UnsupportedVersion, UnsupportedFeature
from quantumbridge.core.classical import UnsupportedFeature as CoreUnsupportedFeature
from .text_parser import SourceParser
from .text_lowering import Lowerer
from .text_exporter import SourceExporter
from .common import read_input, write_output


def source_loads(text,major,*,strict=True,limits=None,dialect="standard"):
    if not isinstance(strict,bool): raise TypeError("strict must be a boolean; safety limits are always applied.")
    if dialect not in {"standard","qiskit"} or major == 2 and dialect != "standard":
        raise ValueError("Only standard and explicit QASM3 qiskit dialects are supported.")
    from quantumbridge.core.symbolic_limits import symbolic_serialization_limits
    limits = symbolic_serialization_limits(limits)
    parser = SourceParser(text,major,limits)
    try: return Lowerer(parser.parse(),limits,strict,dialect).lower()
    except SerializationError: raise
    except (ValueError,TypeError,ArithmeticError,RecursionError) as exc:
        token = parser.peek()
        raise FormatValidationError(str(exc),format=parser.format,version=major,line=token.line,column=token.column) from exc


def qasm2_loads(text,**options): return source_loads(text,2,**options)
def qasm3_loads(text,**options): return source_loads(text,3,**options)
def source_dumps(circuit,version,**options):
    from quantumbridge.core.symbolic_limits import symbolic_serialization_limits
    options["limits"] = symbolic_serialization_limits(options.get("limits"))
    try: return SourceExporter(circuit,version,**options).dumps()
    except SerializationError: raise
    except CoreUnsupportedFeature as exc:
        raise UnsupportedFeature(str(exc),format="qasm2" if version == "2.0" else "qasm3",
            version=version,feature=exc.feature) from exc
    except (ValueError,TypeError,KeyError,RecursionError,OverflowError) as exc:
        raise FormatValidationError(str(exc),format="qasm2" if version == "2.0" else "qasm3",version=version) from exc


def qasm2_dumps(circuit,**options): return source_dumps(circuit,"2.0",**options)
def qasm3_dumps(circuit,*,version="3.0",**options):
    if version not in {"3.0","3.1"}: raise UnsupportedVersion("Only declared OpenQASM 3.0/3.1 subsets are supported.",format="qasm3",version=version)
    return source_dumps(circuit,version,**options)
def qasm2_load(source,**options): return qasm2_loads(read_input(source,limits=options.get("limits"),format="qasm2"),**options)
def qasm3_load(source,**options): return qasm3_loads(read_input(source,limits=options.get("limits"),format="qasm3"),**options)
def qasm2_dump(circuit,destination,**options): write_output(destination,qasm2_dumps(circuit,**options))
def qasm3_dump(circuit,destination,**options): write_output(destination,qasm3_dumps(circuit,**options))
