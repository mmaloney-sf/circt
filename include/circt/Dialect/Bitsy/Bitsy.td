#ifndef BITSY_OPS
#define BITSY_OPS

include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Bitsy_Dialect : Dialect {
  let name = "bitsy";
  let cppNamespace = "bitsy";
}

class Bitsy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Bitsy_Dialect, mnemonic, traits>;

def ConstantOp : Bitsy_Op<"constant", [Pure]> {
  let summary = "constant";
  let description = [{
    Constant
  }];

  let arguments = (ins I32Attr:$value);
  let results = (outs I32);
}

def AddOp : Bitsy_Op<"add", [Pure]> {
  let arguments = (ins
     I32:$op0,
     I32:$op1
  );
  let results = (outs
    I32:$result
  );
}

#endif
