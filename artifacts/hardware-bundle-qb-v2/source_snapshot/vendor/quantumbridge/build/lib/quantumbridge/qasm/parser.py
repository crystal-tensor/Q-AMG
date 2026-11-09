# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_execution_plan_v0.1.md.

from __future__ import annotations

import math

from .ast import Barrier, GateCall, IndexedRef, Measurement, QASMProgram, RegisterDecl
from .errors import QASMParseError
from .lexer import lex
from .tokens import Token


ONE_QUBIT = {"id", "h", "x", "y", "z", "s", "sdg", "t", "tdg", "reset"}
ROTATION = {"rx", "ry", "rz", "phase", "p"}
TWO_QUBIT = {"cx", "cz", "swap"}
THREE_QUBIT = {"ccx"}
SUPPORTED_GATES = ONE_QUBIT | ROTATION | TWO_QUBIT | THREE_QUBIT


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        token = self.peek()
        self.pos += 1
        return token

    def match(self, kind: str, value: str | None = None) -> Token | None:
        token = self.peek()
        if token.kind == kind and (value is None or token.value == value):
            return self.advance()
        return None

    def expect(self, kind: str, value: str | None = None, message: str | None = None) -> Token:
        token = self.match(kind, value)
        if token is None:
            current = self.peek()
            wanted = value or kind
            raise QASMParseError(message or f"expected {wanted!r}, found {current.value!r}", current.line, current.column)
        return token

    def expect_ident(self, value: str | None = None) -> Token:
        token = self.expect("IDENT", message="expected identifier")
        if value is not None and token.value != value:
            raise QASMParseError(f"expected {value!r}, found {token.value!r}", token.line, token.column)
        return token

    def parse(self) -> QASMProgram:
        self.expect_ident("OPENQASM")
        number = self.expect("NUMBER", message="expected OpenQASM version")
        if number.value != "2.0":
            raise QASMParseError("only OPENQASM 2.0 is supported", number.line, number.column)
        self.expect(";")
        include = None
        registers: list[RegisterDecl] = []
        statements: list[GateCall | Measurement | Barrier] = []
        if self.peek().kind == "IDENT" and self.peek().value == "include":
            self.advance()
            string = self.expect("STRING", message="expected include string")
            include = string.value
            if include != "qelib1.inc":
                raise QASMParseError("only include \"qelib1.inc\" is supported", string.line, string.column)
            self.expect(";")
        while self.peek().kind != "EOF":
            if self.peek().kind != "IDENT":
                token = self.peek()
                raise QASMParseError("expected statement", token.line, token.column)
            word = self.peek().value
            if word in {"qreg", "creg"}:
                registers.append(self.parse_register())
            elif word == "measure":
                statements.append(self.parse_measurement())
            elif word == "barrier":
                statements.append(self.parse_barrier())
            elif word in SUPPORTED_GATES:
                statements.append(self.parse_gate())
            else:
                token = self.peek()
                raise QASMParseError(f"does not support line: unsupported statement or gate {word!r}", token.line, token.column)
        return QASMProgram(version="2.0", include=include, registers=tuple(registers), statements=tuple(statements))

    def parse_register(self) -> RegisterDecl:
        kind = self.expect("IDENT").value
        name = self.expect("IDENT", message="expected register name").value
        self.expect("[")
        size = int(self.expect("NUMBER", message="expected register size").value)
        self.expect("]")
        self.expect(";")
        if size <= 0:
            raise QASMParseError("register size must be positive")
        return RegisterDecl(kind, name, size)

    def parse_ref(self) -> IndexedRef:
        name = self.expect("IDENT", message="expected register name").value
        self.expect("[")
        index = int(self.expect("NUMBER", message="expected register index").value)
        self.expect("]")
        return IndexedRef(name, index)

    def parse_ref_or_register(self) -> IndexedRef | str:
        name = self.expect("IDENT", message="expected register reference").value
        if self.match("["):
            index = int(self.expect("NUMBER", message="expected register index").value)
            self.expect("]")
            return IndexedRef(name, index)
        return name

    def parse_gate(self) -> GateCall:
        name_token = self.expect("IDENT")
        name = "phase" if name_token.value == "p" else name_token.value
        params: tuple[float, ...] = ()
        if name_token.value in ROTATION:
            self.expect("(")
            params = (self.parse_expression(),)
            self.expect(")")
        qubits = [self.parse_ref()]
        while self.match(","):
            qubits.append(self.parse_ref())
        self.expect(";")
        expected = 1 if name in ONE_QUBIT or name in ROTATION else 2 if name in TWO_QUBIT else 3
        if len(qubits) != expected:
            raise QASMParseError(f"gate {name!r} expects {expected} qubit reference(s)", name_token.line, name_token.column)
        return GateCall(name, tuple(qubits), params)

    def parse_measurement(self) -> Measurement:
        self.expect_ident("measure")
        qubit = self.parse_ref_or_register()
        self.expect("ARROW", message="expected measurement arrow '->'")
        bit = self.parse_ref_or_register()
        self.expect(";")
        return Measurement(qubit, bit)

    def parse_barrier(self) -> Barrier:
        self.expect_ident("barrier")
        refs = [self.parse_ref_or_register()]
        while self.match(","):
            refs.append(self.parse_ref_or_register())
        self.expect(";")
        return Barrier(tuple(refs))

    def parse_expression(self) -> float:
        value = self.parse_term()
        while self.peek().kind in {"+", "-"}:
            op = self.advance().kind
            rhs = self.parse_term()
            value = value + rhs if op == "+" else value - rhs
        return value

    def parse_term(self) -> float:
        value = self.parse_factor()
        while self.peek().kind in {"*", "/"}:
            op = self.advance().kind
            rhs = self.parse_factor()
            value = value * rhs if op == "*" else value / rhs
        return value

    def parse_factor(self) -> float:
        sign = 1.0
        if self.match("-"):
            sign = -1.0
        elif self.match("+"):
            sign = 1.0
        token = self.peek()
        if token.kind == "NUMBER":
            return sign * float(self.advance().value)
        if token.kind == "IDENT" and token.value == "pi":
            self.advance()
            return sign * math.pi
        raise QASMParseError("expected numeric parameter expression", token.line, token.column)


def parse(text: str) -> QASMProgram:
    return Parser(lex(text)).parse()
