//===- HandshakeToDC.cpp - Translate Handshake into DC --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main Handshake to DC Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/HandshakeToDC.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/DC/DCDialect.h"
#include "circt/Dialect/DC/DCOps.h"
#include "circt/Dialect/DC/DCTypes.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Dialect/Handshake/Visitor.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

using namespace mlir;
using namespace circt;
using namespace circt::handshake;
using namespace circt::dc;

namespace {

struct DCTuple {
  DCTuple() {}
  DCTuple(Value token, ValueRange data) : token(token), data(data) {}
  DCTuple(UnpackOp unpack) : token(unpack.token()), data(unpack.data()) {}
  Value token;
  ValueRange data;
};

static DCTuple unpack(OpBuilder &b, Value v) {
  DCTuple res;
  dc::ValueType vt = v.getType().cast<dc::ValueType>();
  auto res = b.create<dc::UnpackOp>(v.getLoc(), vt, v);
  return DCTuple(res);
}

class CondBranchConversionPattern
    : public OpConversionPattern<handshake::ConditionalBranchOp> {
public:
  using OpConversionPattern<
      handshake::ConditionalBranchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(handshake::ConditionalBranchOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto cond = operands[0];
    auto input = unpack(rewriter, operands[1]);
    rewriter.replaceOpWithNewOp<dc::CondBranchOp>(op, cond, trueOp, falseOp);
    return success();
  }
};

class HandshakeToDCPass : public HandshakeToDCBase<HandshakeToDCPass> {
public:
  void runOnOperation() override {}
};
} // namespace

std::unique_ptr<mlir::Pass> circt::createHandshakeToDCPass() {
  return std::make_unique<HandshakeToDCPass>();
}
