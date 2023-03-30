//===- DCOps.h - DC dialect operations --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DC_OPS_H
#define CIRCT_DIALECT_DC_OPS_H

#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "circt/Dialect/DC/DCDialect.h"
#include "circt/Dialect/DC/DCTypes.h"

#define GET_OP_CLASSES
#include "circt/Dialect/DC/DC.h.inc"

#endif // CIRCT_DIALECT_DC_OPS_H
