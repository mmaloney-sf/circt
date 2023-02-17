//===- FIRRTLFieldSource.h - Field Source (points-to) -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the FIRRTL Field Source Analysis, which is a points-to
// analysis which identifies the source field and storage of any value derived
// from a storage location.  Values derived from reads of storage locations do
// not appear in this analysis.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLFIELDSOURCE_H
#define CIRCT_DIALECT_FIRRTL_FIRRTLFIELDSOURCE_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/LLVM.h"

namespace circt {
namespace firrtl {

/// To use this class, retrieve a cached copy from the analysis manager:
///   auto &fieldsource = getAnalysis<FieldSource>(getOperation());
class FieldSource {

public:
  explicit FieldSource(Operation *operation);

  struct PathNode {
    PathNode(Value src, ArrayRef<int64_t> ar) : src(src), path(ar) {}
    Value src;
    SmallVector<int64_t, 4> path;

    /// Roots are operations which define the storage or aggregate value.
    bool isRoot() const { return path.empty(); }

    /// Writable sources can appear as a LHS of a connect.
    // FIXME: This is just flow.
    bool isSrcWritable() const {
      // Ports need to check direction
      if (!src.getDefiningOp()) {
        auto barg = cast<BlockArgument>(src);
        auto module = cast<FModuleOp>(barg.getOwner()->getParentOp());
        auto direction = module.getPortDirection(barg.getArgNumber());
        return direction != Direction::In;
      }
      // over approximate instances too
      return isa<WireOp, RegOp, RegResetOp, InstanceOp, MemOp>(
          src.getDefiningOp());
    }

    /// Transparent sources reflect a value written to them in the same cycle it
    /// is written.  These are sources which provide dataflow backwards in SSA
    /// during one logical execution of a module body.
    /// This property only makes sense to ask of writable sources.
    bool isSrcTransparent() const {
      // ports are wires
      if (!src.getDefiningOp())
        return true;
      // ports are wires, on this side of the instance too.
      return isa<WireOp, InstanceOp>(src.getDefiningOp());
    }
  };

  const PathNode *nodeForValue(Value v) const;

private:
  void visitOp(Operation *op);
  void visitSubfield(SubfieldOp sf);
  void visitSubindex(SubindexOp si);
  void visitSubaccess(SubaccessOp sa);
  void visitMem(MemOp mem);
  void visitInst(InstanceOp inst);

  void makeNodeForValue(Value dst, Value src, ArrayRef<int64_t> path);

  DenseMap<Value, PathNode> paths;
};

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLFIELDSOURCE_H