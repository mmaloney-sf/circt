//===- DCOps.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/DC/DCOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace circt;
using namespace dc;
using namespace mlir;

namespace circt {
namespace dc {
#include "circt/Dialect/DC/DCCanonicalization.h.inc"

// =============================================================================
// SyncOp
// =============================================================================

void SyncOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<circt::dc::EliminateSimpleSyncPattern>(context);
}

} // namespace dc
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/DC/DC.cpp.inc"
