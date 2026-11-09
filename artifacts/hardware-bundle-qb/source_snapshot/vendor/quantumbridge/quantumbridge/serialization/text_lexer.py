# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Bounded OpenQASM lexical analysis, independent of upstream parsers.
import re
from quantumbridge.qasm.tokens import Token
from .limits import SerializationLimits
from .errors import ParseError

_NUMBER = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_PAIRS = {"->","==","!=","<=",">=","&&","||","<<",">>","**","+=","-=","*=","/=","%=","++","--"}
_SYMBOLS = set(";,:[](){}+-*/%^&|~!=<>")


def tokenize(text, *, limits=None, format="qasm"):
    limits = limits or SerializationLimits()
    if not isinstance(text,str): raise TypeError("OpenQASM source must be text.")
    limits.check("max_input_bytes",len(text.encode("utf-8")),format=format)
    tokens, index, line, column = [],0,1,1
    def error(reason): raise ParseError(reason,format=format,line=line,column=column)
    while index < len(text):
        start, sl, sc = index,line,column
        char = text[index]
        if char.isspace():
            index += 1
            if char == "\n": line += 1; column = 1
            else: column += 1
            continue
        if text.startswith("//",index):
            end = text.find("\n",index)
            end = len(text) if end < 0 else end
            column += end-index; index = end; continue
        if text.startswith("/*",index):
            end = text.find("*/",index+2)
            if end < 0: error("Unterminated block comment.")
            fragment = text[index:end+2]
            line += fragment.count("\n")
            column = len(fragment.rsplit("\n",1)[-1])+1 if "\n" in fragment else column+len(fragment)
            index = end+2; continue
        if char == '"':
            index += 1
            while index < len(text) and text[index] != '"':
                if text[index] in "\n\r\\": error("Unterminated or escaped string literal is unsupported.")
                index += 1
            if index == len(text): error("Unterminated string literal.")
            value = text[start+1:index]; index += 1; column += index-start
            tokens.append(Token("STRING",value,sl,sc))
        elif char.isdigit() or char == ".":
            match = _NUMBER.match(text,index)
            if match is None: error("Malformed numeric literal.")
            value = match.group(); index = match.end(); column += index-start
            tokens.append(Token("NUMBER",value,sl,sc))
        elif char.isalpha() or char == "_":
            index += 1
            while index < len(text) and (text[index].isalnum() or text[index] == "_"): index += 1
            value = text[start:index]; column += index-start
            tokens.append(Token("IDENT",value,sl,sc))
        else:
            value = text[index:index+2] if text[index:index+2] in _PAIRS else char
            if value not in _PAIRS and value not in _SYMBOLS: error(f"Unexpected character {char!r}.")
            index += len(value); column += len(value)
            tokens.append(Token(value,value,sl,sc))
        limits.check("max_tokens",len(tokens),format=format,line=sl,column=sc)
    tokens.append(Token("EOF","",line,column))
    return tokens
