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

#endif // CIRCT_DIALECT_DC_OPS_TD