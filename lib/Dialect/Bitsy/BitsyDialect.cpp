#include "circt/Dialect/Bitsy/BitsyOps.h"

using namespace circt;
using namespace bitsy;

void BitsyDialect::initialize() {
//  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Bitsy/Bitsy.cpp.inc"
      >();
}

#include "circt/Dialect/Bitsy/BitsyDialect.cpp.inc"
