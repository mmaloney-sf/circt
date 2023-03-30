//===- DCTypes.td - DC dialect types -----------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DC_TYPES_TD
#define CIRCT_DIALECT_DC_TYPES_TD

include "circt/Dialect/DC/DCDialect.td"
include "mlir/IR/AttrTypeBase.td"

class DCTypeDef<string name> : TypeDef<DCDialect, name> { }

def TokenType : DCTypeDef<"Token"> {
  let mnemonic = "token";
  let parameters = (ins );
  let assemblyFormat = "";
}

// A value type is a set of values which is wrapped with token semantics.
def ValueType : DCTypeDef<"Value"> {
  let mnemonic = "value";
  let parameters = (ins ArrayRefParameter<"Type">:$innerTypes);
  let assemblyFormat = "`<`$innerTypes`>`";
}

#endif // CIRCT_DIALECT_DC_TYPES_TD
