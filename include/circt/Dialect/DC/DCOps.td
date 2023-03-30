//===- DCOps.td - DC dialect operations --------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DC_OPS_TD
#define CIRCT_DIALECT_DC_OPS_TD

include "circt/Dialect/DC/DCDialect.td"
include "circt/Dialect/DC/DCTypes.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"

class DCOp<string mnemonic, list<Trait> traits = []> :
  Op<DCDialect, mnemonic, traits>;

def JoinOp : DCOp<"join"> {
  let summary = "Synchronizes the incoming tokens with the outgoing token";
  let description = [{
    This operator synchronizes all incoming tokens. Synchronization implies applying
    join semantics in between all in- and output ports.

    Example:
    ```mlir
    %0 = dc.join %a, %b
    ```
  }];

  let arguments = (ins Variadic<TokenType>:$inputs);
  let results = (outs TokenType:$output);

  let assemblyFormat = "$inputs attr-dict";
  let hasCanonicalizer = 1;
}

def ForkOp : DCOp<"fork"> {
    let summary = "Splits the incoming token into multiple outgoing tokens";
    let description = [{
        This operator splits the incoming token into multiple outgoing tokens.
    
        Example:
        ```mlir
        %0, %1 = dc.fork %a : 
        ```
    }];
    
    let arguments = (ins TokenType:$input);
    let results = (outs Variadic<TokenType>:$outputs);

    let assemblyFormat = "$input attr-dict (`:` qualified(type($outputs))^)?";
    let hasCanonicalizer = 1;
}

def BranchOp : DCOp<"branch"> {
    let summary = "Branch operation";
    let description = [{
        The token is propagated to the output based on the value of the
        conditional signal.
    }];

    let arguments = (ins I1:$cond, TokenType:$data);
    let results = (outs TokenType:$true, TokenType:$false);

    let assemblyFormat = "$cond, $data attr-dict";
}


def SelectOp : DCOp<"select"> {
    let summary = "Select operation";
    let description = [{
        An input token is selected based on the value of the incoming select
        signal, and propagated to the single output.
    }];

    let arguments = (ins AnyIntegerType:$sel, TokenType:$data);
    let results = (outs Variadic<TokenType>:$outputs);
    let assemblyFormat = "$sel, $data attr-dict (`:` qualified(type($outputs))^)?";
}

def SinkOp : DCOp<"sink"> {
    let summary = "Sink operation";
    let description = [{
        The sink operation will always accept any incoming tokens, and 
        discard them.
    }];

    let arguments = (ins TokenType:$in);
    let results = (outs);
    let assemblyFormat = "$in attr-dict";
}

def SourceOp : DCOp<"source"> {
    let summary = "Source operation";
    let description = [{
        The source operation will always produce a token.
    }];

    let arguments = (ins);
    let results = (outs TokenType:$out);
    let assemblyFormat = "attr-dict `:` type($out)";
}

def PackOp : DCOp<"pack"> {
    let summary = "Pack operation";
    let description = [{
        An operation which packs together a !dc.token value with some other
        value - mostly used to facilitate conversion, given the lack of
        1:N value replacement in MLIRs pattern rewrite infrastructure.
    }];

    let arguments = (ins TokenType:$token, AnyType:$value);
    let results = (outs ValueType:$out);
    let assemblyFormat = "$token, $value attr-dict `:` type($out)";
}

def UnpackOp : DCOp<"unpack"> {
    let summary = "Unpack operation";
    let description = [{
        An operation which unpacks a !dc.value value into a !dc.token value
        and its constituent values - mostly used to facilitate conversion, given the
        lack of 1:N value replacement in MLIRs pattern rewrite infrastructure.
    }];

    let arguments = (ins ValueType:$value);
    let results = (outs TokenType:$token, Variadic<AnyType>:$out);
    let assemblyFormat = "$value attr-dict (`:` qualified(type($out))^)?";
    let hasCanonicalizer = 1;
}

/*
def SymbolOp : DCOp<"symbol", [
        Symbol
    ]> {
    let summary = "Indicates a symbol in the control and dataflow graph";
    let description = [{
        This operator indicates a symbol in the control and dataflow graph.
        A symbol can be referenced from within `dc.node` operations.
    
        Example:
        ```mlir
        dc.node @foo
        ```
    }];

    let arguments = (ins SymbolNameAttr:$sym_name);
    let results = (outs);

    let assemblyFormat = "$sym_name attr-dict";
}

class SameTypeConstraint<string lhs, string rhs>
 : TypesMatchWith<"lhs and rhs types should be equivalent",
                   lhs, rhs, [{ $_self }]>;

// Define `dc.node` which is an operation that contains a list of references to
// `dc.symbol` ops.
class NodeOp<string mnemonic, list<Trait> traits = []> : DCOp<mnemonic,
    !listconcat(traits,[
        SingleBlock,
        SameTypeConstraint<"out", "in">
    ])> {
    let arguments = (ins
        AnyType:$in,
        FlatSymbolRefArrayAttr:$symbols
    );
    let results = (outs AnyType:$out);
    let assemblyFormat = "$in `[` $symbols `]` attr-dict `:` type($out)";
}

def ControlNode : NodeOp<"control"> {}
def DataNode : NodeOp<"data"> {}

*/

#endif // CIRCT_DIALECT_DC_OPS_TD