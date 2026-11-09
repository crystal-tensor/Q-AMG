# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# This file is independently implemented for QuantumBridge SDK.
# No source code from Qiskit or PennyLane was copied.
# Design source: docs/roadmap/p2_execution_plan_v0.1.md.

from __future__ import annotations

from .errors import QASMParseError
from .tokens import Token


SYMBOLS = {";", ",", "[", "]", "(", ")", "/", "-", "+", "*"}


def lex(text: str) -> list[Token]:
    tokens: list[Token] = []
    line = 1
    column = 1
    index = 0
    while index < len(text):
        char = text[index]
        if char in " \t\r":
            index += 1
            column += 1
            continue
        if char == "\n":
            index += 1
            line += 1
            column = 1
            continue
        if char == "/" and index + 1 < len(text) and text[index + 1] == "/":
            while index < len(text) and text[index] != "\n":
                index += 1
                column += 1
            continue
        if char == "-" and index + 1 < len(text) and text[index + 1] == ">":
            tokens.append(Token("ARROW", "->", line, column))
            index += 2
            column += 2
            continue
        if char == '"':
            start_column = column
            index += 1
            column += 1
            value = []
            while index < len(text) and text[index] != '"':
                if text[index] == "\n":
                    raise QASMParseError("unterminated string literal", line, start_column)
                value.append(text[index])
                index += 1
                column += 1
            if index >= len(text):
                raise QASMParseError("unterminated string literal", line, start_column)
            index += 1
            column += 1
            tokens.append(Token("STRING", "".join(value), line, start_column))
            continue
        if char in SYMBOLS:
            tokens.append(Token(char, char, line, column))
            index += 1
            column += 1
            continue
        if char.isdigit() or char == ".":
            start = index
            start_column = column
            saw_dot = char == "."
            index += 1
            column += 1
            while index < len(text) and (text[index].isdigit() or (text[index] == "." and not saw_dot)):
                saw_dot = saw_dot or text[index] == "."
                index += 1
                column += 1
            if index < len(text) and text[index] in {"e", "E"}:
                index += 1
                column += 1
                if index < len(text) and text[index] in {"+", "-"}:
                    index += 1
                    column += 1
                while index < len(text) and text[index].isdigit():
                    index += 1
                    column += 1
            tokens.append(Token("NUMBER", text[start:index], line, start_column))
            continue
        if char.isalpha() or char == "_":
            start = index
            start_column = column
            index += 1
            column += 1
            while index < len(text) and (text[index].isalnum() or text[index] == "_"):
                index += 1
                column += 1
            tokens.append(Token("IDENT", text[start:index], line, start_column))
            continue
        raise QASMParseError(f"unexpected character {char!r}", line, column)
    tokens.append(Token("EOF", "", line, column))
    return tokens
