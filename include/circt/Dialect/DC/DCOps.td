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

def SyncOp : DCOp<"sync"> {
  let summary = "Synchronizes the incoming tokens with the outgoing token";
  let description = [{
    This operator synchronizes all incoming tokens. Synchronization implies applying
    join semantics in between all in- and output ports.

    Example:
    ```mlir
    %0 = dc.sync %a, %b
    ```
  }];

  let arguments = (ins Variadic<TokenType>:$inputs);
  let results = (outs TokenType:$output);

  let assemblyFormat = "$inputs attr-dict";
}

#endif // CIRCT_DIALECT_DC_OPS_TD