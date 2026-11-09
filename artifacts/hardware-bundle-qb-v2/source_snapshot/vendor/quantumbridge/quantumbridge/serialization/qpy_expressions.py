# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Public QPY v13 arithmetic replay protocol, implemented independently.

import math
import operator
import struct
from numbers import Complex, Integral, Real
from uuid import UUID
from quantumbridge.core import Parameter, ParameterExpression
from .limits import SerializationLimits
from .errors import UnsupportedFeature, FormatValidationError, SerializationError

RECORD = struct.Struct("!Bc16sc16s")
BINARY = {0:operator.add,1:operator.sub,2:operator.mul,3:operator.truediv,4:operator.pow,
          18:lambda left,right:right-left,19:lambda left,right:right/left,20:lambda left,right:right**left}
OPCODES = {"+":0,"-":1,"*":2,"/":3,"**":4}
UNARY = {5:"sin",6:"cos",7:"tan",8:"arcsin",9:"arccos",10:"exp",11:"log",
         12:"sign",14:"conjugate",16:"abs",17:"arctan"}
UNARY_CODES = {name:code for code,name in UNARY.items()}


def _charge_nodes(reader, count):
    reader.budget["expression_nodes"] = reader.budget.get("expression_nodes",0)+count
    reader.bounded("max_parameter_nodes",reader.budget["expression_nodes"])


def _charge_work(reader, count):
    from quantumbridge.core.gate_limits import current_gate_limits
    reader.budget["expression_work"] = reader.budget.get("expression_work",0)+count
    current_gate_limits().check("max_decomposition_work",reader.budget["expression_work"])


def _replay_records(reader, replay, base, *, preflight=False):
    offset = 0
    while offset < len(replay):
        _charge_work(reader,1)
        if preflight: _charge_nodes(reader,1)
        if len(replay)-offset < RECORD.size:
            raise FormatValidationError("Truncated symbolic replay record.",format="qpy",version=13,offset=base+offset)
        fields = RECORD.unpack_from(replay,offset)
        offset += RECORD.size
        mapping,location = None,base+offset
        if fields[0] == 15:
            _,left_kind,left_data,right_kind,right_data = fields
            if left_kind != b"u" or right_kind != b"n" or left_data[8:] != bytes(8) or right_data != bytes(16):
                raise FormatValidationError("Invalid v13 substitution replay fields.",format="qpy",version=13,offset=location-RECORD.size)
            size, = struct.unpack("!Q",left_data[:8])
            reader.bounded("max_input_bytes",size)
            if size < 8 or size > len(replay)-offset:
                raise FormatValidationError("Invalid substitution mapping length.",format="qpy",version=13,offset=location)
            mapping = memoryview(replay)[offset:offset+size]
            offset += size
        yield (*fields,mapping,location)


def _replay_limits(reader, depth):
    from dataclasses import replace
    from quantumbridge.core.symbolic_limits import current_symbolic_limits
    from quantumbridge.core.gate_limits import current_gate_limits
    limits = current_symbolic_limits()
    reader.bounded("max_symbolic_expression_depth",depth+1)
    return replace(limits,
        max_nodes=min(limits.max_nodes,reader.limits.max_parameter_nodes),
        max_depth=min(limits.max_depth,reader.limits.max_symbolic_expression_depth-depth),
        max_expansion_work=min(limits.max_expansion_work,current_gate_limits().max_decomposition_work))


def _substitute(reader, left, data, base, depth):
    from .qpy import Reader
    from .errors import ResourceLimitError
    from quantumbridge.core.parameters import _coerce_expression
    from quantumbridge.core.symbolic_limits import symbolic_expression_scope, SymbolicResourceLimitError
    from quantumbridge.core.gate_limits import current_gate_limits
    sub = Reader(data,reader.limits,base,reader.depth+1,reader.budget)
    count, = sub.unpack("!Q")
    _charge_nodes(reader,count)
    _charge_work(reader,2*count)
    current_gate_limits().check("max_gate_parameters",count)
    from quantumbridge.core.symbolic_limits import current_symbolic_limits
    from .errors import ResourceLimitError
    try: current_symbolic_limits().check("max_substitutions",count)
    except SymbolicResourceLimitError as exc: sub.error(str(exc),ResourceLimitError)
    if count > (len(sub.data)-sub.pos)//5: sub.error("Truncated substitution mapping entries.")
    expression = _coerce_expression(left)
    _charge_work(reader,len(expression.parameters))
    context = {parameter.name:parameter for parameter in expression.parameters}
    entries = {}
    for _ in range(count):
        length,kind,size = sub.unpack("!HcH")
        name = sub.text(length)
        if name in entries: sub.error("Duplicate substitution mapping key.")
        if name not in context: sub.error("Substitution name does not identify a parameter in this expression.")
        if kind not in {b"p",b"v",b"e"}:
            sub.error("Substitution values require Parameter or ParameterExpression records.",UnsupportedFeature)
        position = sub.pos
        sub.take(size)
        entries[name] = (kind,size,position)
    if sub.pos != len(sub.data): sub.error("Trailing bytes in substitution mapping.")
    # Replaced identities may disappear, including a legal p(uuid1)->p(uuid2).
    # Unreplaced parameters stay canonical; keys always resolve against the
    # original expression, never an unrelated Circuit-level name dictionary.
    parameters = {name:parameter for name,parameter in context.items() if name not in entries}
    replacements = {}
    try:
        with symbolic_expression_scope(_replay_limits(reader,depth)):
            for name,(kind,size,position) in entries.items():
                value_reader = Reader(sub.data[position:position+size],reader.limits,
                                      base+position,sub.depth,reader.budget)
                value_reader.parameters = parameters
                value_reader.parameter_identities = reader.parameter_identities
                value_reader.vectors = reader.vectors
                value = value_reader.value(kind,size)
                replacements[context[name]] = _coerce_expression(value)
                reader.max_depth = max(reader.max_depth,value_reader.max_depth)
            value = expression.subs(replacements)
    except SerializationError: raise
    except SymbolicResourceLimitError as exc: sub.error(str(exc),ResourceLimitError)
    except (ArithmeticError,TypeError,ValueError) as exc:
        sub.error(f"Invalid symbolic substitution: {type(exc).__name__}.")
    reader.bounded("max_parameter_nodes",value._nodes)
    reader.bounded("max_symbolic_expression_depth",depth+value._depth)
    for parameter in value.parameters: reader.parameters[parameter.name] = parameter
    reader.max_depth = max(reader.max_depth,sub.max_depth)
    return value


def read_expression(reader):
    count,size = reader.unpack("!QQ")
    reader.bounded("max_parameter_nodes",count)
    reader.bounded("max_input_bytes",size)
    _charge_nodes(reader,count)
    _charge_work(reader,count)
    replay_base = reader.base+reader.pos
    replay = reader.take(size)
    for _ in _replay_records(reader,replay,replay_base,preflight=True): pass
    symbols = {}
    for _ in range(count):
        key_type,value_type,value_size = reader.unpack("!ccQ")
        if key_type==b"v":
            key=reader.parameter_vector()
            uid=UUID(key.uuid).bytes
        elif key_type==b"p":
            key = reader.parameter()
            uid = UUID(key.uuid).bytes
        else:reader.error("Unsupported parameter symbol type.",UnsupportedFeature)
        if uid in symbols: reader.error("Duplicate symbolic mapping key.")
        if value_type in {b"p",b"v"}:
            if value_type!=key_type:reader.error("Symbolic parameter alias type does not match its key.")
            if value_size: reader.error("Parameter alias must have zero payload bytes.")
            value = key
        else: value = reader.value(value_type,value_size)
        symbols[uid] = value
    stack,frames = [],[]
    def admit_result(*values):
        nodes = 1+sum(value._nodes if type(value) is ParameterExpression else 1 for value in values)
        reader.bounded("max_parameter_nodes",nodes)
    def operand(kind,data):
        if kind == b"n":
            if data != bytes(16) or not stack: reader.error("Invalid symbolic replay stack reference.")
            return stack.pop()
        if kind == b"p":
            if data not in symbols: reader.error("Symbolic replay references unknown UUID.")
            return symbols[data],1
        if kind in {b"i",b"f"}:
            if data[:8] != bytes(8): reader.error("Invalid numeric symbolic padding.")
            value, = struct.unpack("!q" if kind == b"i" else "!d",data[8:])
            if not math.isfinite(value): reader.error("Nonfinite symbolic literal.")
            return value,1
        if kind == b"c":
            real,imag = struct.unpack("!dd",data)
            if not math.isfinite(real) or not math.isfinite(imag): reader.error("Nonfinite symbolic literal.")
            return complex(real,imag),1
        reader.error(f"Unsupported symbolic operand {kind!r}.",UnsupportedFeature)
    for op,left_kind,left_data,right_kind,right_data,mapping,location in _replay_records(reader,replay,replay_base):
        if op == 15:
            left,_ = operand(b"n",bytes(16))
            value = _substitute(reader,left,mapping,location,len(frames))
            stack.append((value,value._depth))
            continue
        if op == 255:
            if (left_data != bytes(16) or right_data != bytes(16)
                    or (left_kind,right_kind) not in {(b"s",b"n"),(b"n",b"s"),(b"e",b"n"),(b"n",b"e")}):
                reader.error("Invalid recursive symbolic replay marker.")
            if b"s" in (left_kind,right_kind):
                reader.bounded("max_symbolic_expression_depth",len(frames)+1)
                frames.append(stack)
                stack = []
            else:
                if not frames or len(stack) != 1: reader.error("Unmatched or empty recursive symbolic replay.")
                value = stack[0]
                stack = frames.pop()
                stack.append(value)
            continue
        if op == 13:
            if right_kind != b"p" or right_data not in symbols or not isinstance(symbols[right_data],Parameter):
                reader.error("A symbolic gradient requires a parameter UUID operand.")
            left,ld = operand(left_kind,left_data)
            reader.bounded("max_symbolic_expression_depth",len(frames)+ld+1)
            from quantumbridge.core.parameters import _coerce_expression, _with_derivative_order
            from quantumbridge.core.symbolic_limits import symbolic_expression_scope, SymbolicResourceLimitError
            from .errors import ResourceLimitError
            try:
                with symbolic_expression_scope(_replay_limits(reader,len(frames))):
                    source = _coerce_expression(left)
                    value = source.gradient(symbols[right_data])
                    value = _with_derivative_order(_coerce_expression(value),source._derivative_order+1)
            except SymbolicResourceLimitError as exc: reader.error(str(exc),ResourceLimitError)
            except (ArithmeticError,TypeError,ValueError) as exc: reader.error(f"Invalid symbolic gradient: {type(exc).__name__}.")
            reader.bounded("max_parameter_nodes",value._nodes)
            reader.bounded("max_symbolic_expression_depth",len(frames)+value._depth)
            stack.append((value,value._depth))
            continue
        if op in UNARY:
            if right_kind != b"n" or right_data != bytes(16): reader.error("Unary symbolic replay has an unexpected right operand.")
            left,ld = operand(left_kind,left_data)
            reader.bounded("max_symbolic_expression_depth",len(frames)+ld+1)
            admit_result(left)
            from quantumbridge.core.parameters import _coerce_expression
            try: value = getattr(_coerce_expression(left),UNARY[op])()
            except (ArithmeticError,TypeError,ValueError) as exc: reader.error(f"Invalid symbolic function: {type(exc).__name__}.")
            stack.append((value,ld+1))
            continue
        if op not in BINARY: reader.error(f"Unsupported symbolic replay opcode {op}.",UnsupportedFeature)
        right,rd = operand(right_kind,right_data)
        left,ld = operand(left_kind,left_data)
        depth = 1+max(ld,rd)
        reader.bounded("max_symbolic_expression_depth",len(frames)+depth)
        admit_result(left,right)
        base,exponent = (right,left) if op == 20 else (left,right)
        if op in {4,20} and isinstance(base,Integral) and isinstance(exponent,Integral) and exponent > 0 and max(1,base.bit_length())*exponent > 4096:
            reader.error("Numeric symbolic power exceeds the integer budget.")
        try: value = BINARY[op](left,right)
        except (ArithmeticError,TypeError,ValueError) as exc: reader.error(f"Invalid symbolic arithmetic: {type(exc).__name__}.")
        if isinstance(value,Real) and not math.isfinite(value): reader.error("Nonfinite symbolic arithmetic.")
        stack.append((value,depth))
    if frames or len(stack) != 1: reader.error("Symbolic replay must yield exactly one expression without open recursive frames.")
    return stack[0][0]


def write_expression(expression,limits=None):
    from .qpy import _uid, _parameter_value
    limits = limits or SerializationLimits()
    records = []
    symbols = {}
    def emit_math(value,depth=0):
        limits.check("max_symbolic_expression_depth",depth,format="qpy",version=13)
        if isinstance(value,Parameter):
            symbols[value.name] = value
            return b"p",_uid(value).bytes
        if isinstance(value,ParameterExpression):
            if value.op in {"const","parameter"}: return emit(value.args[0],depth+1)
            if value.op == "neg": return emit(value.args[0] * -1,depth+1)
            if value.op in UNARY_CODES:
                left = emit(value.args[0],depth+1)
                limits.check("max_parameter_nodes",len(records)+1,format="qpy",version=13)
                records.append(RECORD.pack(UNARY_CODES[value.op],*left,b"n",bytes(16)))
                return b"n",bytes(16)
            if value.op not in OPCODES: raise UnsupportedFeature("Unsupported arithmetic expression opcode.",format="qpy",version=13)
            left = emit(value.args[0],depth+1)
            right = emit(value.args[1],depth+1)
            records.append(RECORD.pack(OPCODES[value.op],*left,*right))
            limits.check("max_parameter_nodes",len(records),format="qpy",version=13)
            return b"n",bytes(16)
        if isinstance(value,Integral) and not isinstance(value,bool): return b"i",bytes(8)+struct.pack("!q",int(value))
        if isinstance(value,Real) and math.isfinite(value): return b"f",bytes(8)+struct.pack("!d",float(value))
        if isinstance(value,Complex) and math.isfinite(value.real) and math.isfinite(value.imag):
            return b"c",struct.pack("!dd",value.real,value.imag)
        raise UnsupportedFeature("Unsupported symbolic arithmetic value.",format="qpy",version=13)
    def emit(value,depth=0):
        result = emit_math(value,depth)
        if type(value) is ParameterExpression and value._retained_parameters:
            from quantumbridge.core.parameters import parameter_sort_key
            retained = value._retained_parameters
            limits.check("max_parameter_nodes",len(records)+2*len(retained),format="qpy",version=13)
            limits.check("max_symbolic_expression_depth",depth+max(value._depth,2)+len(retained),format="qpy",version=13)
            limits.check("max_input_bytes",(len(records)+2*len(retained))*RECORD.size,format="qpy",version=13)
            for parameter in sorted(retained,key=parameter_sort_key):
                identity = emit_math(parameter,depth+1)
                records.append(RECORD.pack(2,*identity,b"i",bytes(16)))
                records.append(RECORD.pack(0,*result,b"n",bytes(16)))
                result = b"n",bytes(16)
        return result
    emit(expression)
    if not records: raise FormatValidationError("Symbolic replay requires an arithmetic operation.",format="qpy",version=13)
    mapping = []
    for name,parameter in sorted(symbols.items()):
        kind,payload = _parameter_value(parameter,limits)
        mapping.extend((struct.pack("!ccQ",kind,kind,0),payload))
    return struct.pack("!QQ",len(symbols),len(records)*RECORD.size)+b"".join(records+mapping)
