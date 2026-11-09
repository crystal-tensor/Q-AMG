# Copyright 2026 QuantumBridge Contributors.
# Licensed under the Apache License, Version 2.0.
# Independent recursive syntax parser and Pratt expression parser for OpenQASM.
from .text_ast import Expression, Statement, Program
from .text_lexer import tokenize
from .limits import SerializationLimits
from .errors import ParseError, UnsupportedVersion, UnsupportedFeature, UnsafeInputError

PRECEDENCE = {"||":1,"&&":2,"|":3,"^":4,"&":5,"==":6,"!=":6,"<":7,">":7,"<=":7,">=":7,
              "<<":8,">>":8,"+":9,"-":9,"*":10,"/":10,"%":10,"**":12}


class SourceParser:
    def __init__(self,text,version,limits=None):
        self.limits, self.version = limits or SerializationLimits(), version
        self.format = "qasm2" if version == 2 else "qasm3"
        self.tokens = tokenize(text,limits=self.limits,format=self.format)
        self.pos, self.nodes = 0,1

    def peek(self): return self.tokens[self.pos]
    def take(self):
        token = self.peek(); self.pos += 1
        return token
    def accept(self,value):
        if self.peek().value == value: return self.take()
        return None
    def error(self,reason,token=None,cls=ParseError):
        token = token or self.peek()
        raise cls(reason,format=self.format,version=self.version,line=token.line,column=token.column)
    def expect(self,value):
        token = self.accept(value)
        if token is None: self.error(f"Expected {value!r}, found {self.peek().value!r}.")
        return token
    def ident(self):
        if self.peek().kind != "IDENT": self.error("Expected identifier.")
        return self.take().value
    def node(self,kind,data,token):
        self.nodes += 1
        self.limits.check("max_ast_nodes",self.nodes,format=self.format,line=token.line,column=token.column)
        return Statement(kind,data,token.line,token.column)
    def exprnode(self,op,value,args,token):
        self.nodes += 1
        self.limits.check("max_ast_nodes",self.nodes,format=self.format,line=token.line,column=token.column)
        return Expression(op,value,tuple(args),token.line,token.column)

    def expression(self,minimum=0,depth=0):
        token = self.peek()
        self.limits.check("max_expression_depth",depth,format=self.format,line=token.line,column=token.column)
        if self.accept("("):
            left = self.expression(depth=depth+1); self.expect(")")
        elif self.peek().value in {"-","+","!","~"}:
            op = self.take().value
            left = self.exprnode("unary",op,(self.expression(11,depth+1),),token)
        elif token.kind == "NUMBER":
            self.take()
            value = float(token.value) if any(c in token.value for c in ".eE") else int(token.value)
            left = self.exprnode("literal",value,(),token)
        elif token.kind == "STRING":
            self.take(); left = self.exprnode("string",token.value,(),token)
        elif token.kind == "IDENT":
            name = self.ident()
            if name in {"true","false"}: left = self.exprnode("literal",name == "true",(),token)
            else:
                width = None
                if name in {"uint","int","float","angle","bit","bool"} and self.accept("["):
                    width = self.expression(depth=depth+1); self.expect("]")
                if self.accept("("):
                    args = self.expression_list(")",depth+1)
                    left = self.exprnode("call",(name,width),args,token)
                else:
                    if width is not None: self.error("Type expression requires a cast argument.",token)
                    left = self.exprnode("name",name,(),token)
        else: self.error("Expected expression.",token)
        while True:
            if self.accept("["):
                index = self.expression(depth=depth+1); self.expect("]")
                left = self.exprnode("index",None,(left,index),token); continue
            op = self.peek().value
            priority = PRECEDENCE.get(op,-1)
            if priority < minimum: break
            self.take()
            right = self.expression(priority if op == "**" else priority+1,depth+1)
            left = self.exprnode("binary",op,(left,right),token)
        return left

    def expression_list(self,end,depth=0):
        args = []
        if not self.accept(end):
            while True:
                args.append(self.expression(depth=depth))
                if self.accept(end): break
                self.expect(",")
        return tuple(args)

    def body(self,depth):
        self.limits.check("max_nesting_depth",depth,format=self.format,line=self.peek().line,column=self.peek().column)
        if not self.accept("{"): return (self.statement(depth+1),)
        statements = []
        while not self.accept("}"):
            if self.peek().kind == "EOF": self.error("Unterminated source block.")
            statements.append(self.statement(depth+1))
        return tuple(statements)

    def type_spec(self):
        name = self.ident(); width = None
        if self.accept("["):
            width = self.expression(); self.expect("]")
        return name,width

    def statement(self,depth=0):
        token = self.peek(); word = token.value
        self.limits.check("max_nesting_depth",depth,format=self.format,line=token.line,column=token.column)
        if self.accept(";"): return self.node("empty",{},token)
        if self.accept("include"):
            if self.peek().kind != "STRING": self.error("Expected include filename.")
            name = self.take().value; self.expect(";")
            if name not in {"qelib1.inc","stdgates.inc"}: self.error("Only built-in standard library includes are allowed; no paths or network includes.",token,UnsafeInputError)
            self.limits.check("max_include_depth",1,format=self.format,line=token.line,column=token.column)
            return self.node("include",{"name":name},token)
        if word in {"qreg","creg"}:
            self.take(); name = self.ident(); self.expect("["); size = self.expression(); self.expect("]"); self.expect(";")
            return self.node("register",{"kind":"q" if word == "qreg" else "c","name":name,"size":size},token)
        if word in {"gate","opaque"}:
            self.take(); name = self.ident(); params = []
            if self.accept("("):
                if not self.accept(")"):
                    while True:
                        params.append(self.ident())
                        if self.accept(")"): break
                        self.expect(",")
            wires = [self.ident()]
            while self.accept(","): wires.append(self.ident())
            if word == "opaque": self.expect(";"); body = None
            else: body = self.body(depth+1)
            return self.node("definition",{"name":name,"parameters":tuple(params),"wires":tuple(wires),"body":body},token)
        if word in {"qubit","bit","bool","uint","int","float","angle","const","input","output"}:
            qualifier = self.take().value if word in {"const","input","output"} else None
            dtype,width = self.type_spec(); name = self.ident()
            if self.accept("="):
                if self.accept("measure"):
                    value = self.expression(); measure = True
                else: value,measure = self.expression(),False
            else: value,measure = None,False
            self.expect(";")
            return self.node("declare",{"qualifier":qualifier,"type":dtype,"width":width,"name":name,"value":value,"measure":measure},token)
        if self.accept("if"):
            self.expect("("); condition = self.expression(); self.expect(")"); body = self.body(depth+1)
            alternate = self.body(depth+1) if self.accept("else") else None
            return self.node("if",{"condition":condition,"body":body,"alternate":alternate},token)
        if self.accept("while"):
            self.expect("("); condition = self.expression(); self.expect(")")
            return self.node("while",{"condition":condition,"body":self.body(depth+1)},token)
        if self.accept("for"):
            dtype,width = self.type_spec(); name = self.ident(); self.expect("in")
            if self.accept("["): values = self.expression_list(":"); rest = [self.expression()]
            else: values = None
            if values is not None:
                if len(values) != 1: self.error("Invalid range start.",token)
                if self.accept(":"): rest.append(self.expression())
                self.expect("]"); sequence = ("range",values+tuple(rest))
            else:
                self.expect("{"); sequence = ("set",self.expression_list("}"))
            return self.node("for",{"type":dtype,"width":width,"name":name,"sequence":sequence,"body":self.body(depth+1)},token)
        if self.accept("switch"):
            self.expect("("); selector = self.expression(); self.expect(")"); self.expect("{")
            cases,default = [],None
            while not self.accept("}"):
                if self.accept("case"):
                    values = [self.expression()]
                    while self.accept(","): values.append(self.expression())
                    self.accept(":")
                    cases.append((tuple(values),self.body(depth+1)))
                elif self.accept("default"):
                    if default is not None: self.error("Duplicate switch default.")
                    self.accept(":"); default = self.body(depth+1)
                else: self.error("Expected switch case/default block.")
            return self.node("switch",{"selector":selector,"cases":tuple(cases),"default":default},token)
        if word in {"break","continue"}:
            self.take(); self.expect(";"); return self.node(word,{},token)
        if self.accept("measure"):
            source = self.expression(); target = None
            if self.accept("->"): target = self.expression()
            self.expect(";"); return self.node("measure",{"source":source,"target":target},token)
        if word in {"reset","barrier"}:
            self.take(); args = self.expression_list(";")
            return self.node(word,{"args":args},token)
        if self.accept("delay"):
            self.expect("["); value = self.expression(); unit = self.ident(); self.expect("]")
            return self.node("delay",{"value":value,"unit":unit,"args":self.expression_list(";")},token)
        if word in {"defcal","cal","extern","def","box","stretch","array","let","pragma"}:
            self.error(f"Unsupported source feature {word!r}; it cannot be discarded.",token,UnsupportedFeature)
        if self.peek().kind != "IDENT": self.error("Expected source statement.")
        name = self.ident()
        if self.peek().value in {"[","=","+=","-=","*=","/=","%=","++","--"}:
            target = self.exprnode("name",name,(),token)
            if self.accept("["):
                index = self.expression(); self.expect("]"); target = self.exprnode("index",None,(target,index),token)
            operation = self.take().value
            if operation not in {"=","+=","-=","*=","/=","%=","++","--"}: self.error("Expected assignment operator.")
            measurement = operation == "=" and bool(self.accept("measure"))
            value = None if operation in {"++","--"} else self.expression()
            self.expect(";")
            return self.node("assignment",{"target":target,"operator":operation,"value":value,"measure":measurement},token)
        params = self.expression_list(")") if self.accept("(") else ()
        if self.accept(";"): args = ()
        else: args = self.expression_list(";")
        return self.node("gate",{"name":name,"params":params,"args":args},token)

    def parse(self):
        self.expect("OPENQASM")
        version = self.take()
        if version.kind != "NUMBER" or version.value not in ({"2.0"} if self.version == 2 else {"3.0","3.1"}):
            self.error("Unsupported OpenQASM version.",version,UnsupportedVersion)
        self.expect(";")
        result = []
        while self.peek().kind != "EOF": result.append(self.statement())
        return Program(version.value,tuple(result))
