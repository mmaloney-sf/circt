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
// JoinOp
// =============================================================================

void JoinOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<circt::dc::EliminateSimpleJoinPattern>(context);
}

// =============================================================================
// ForkOp
// =============================================================================

void ForkOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<circt::dc::EliminateSimpleForkPattern>(context);
}

// =============================================================================
// UnpackOp
// =============================================================================

struct EliminatePackUnpackPattern : public OpRewritePattern<UnpackOp> {
  using OpRewritePattern<UnpackOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(UnpackOp unpack,
                                PatternRewriter &rewriter) const override {
    auto pack = unpack.data().getDefiningOp<PackOp>();
    if (!pack)
      return failure();

    // Check that the pack value type is the same as the unpack value type.
    if (pack.getType() != unpack.getType())
      return failure();

    rewriter.replaceOp(unpack, pack.data());
    return success();
  }
};

void UnpackOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<circt::dc::EliminatePackUnpackPattern>(context);
}

} // namespace dc
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/DC/DC.cpp.inc"
