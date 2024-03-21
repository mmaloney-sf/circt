#include "circt/Dialect/Bitsy/Dialect.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include <iostream>

using namespace std;

int main(int argc, char **argv) {
  cout << "Hello, CIRCT world!" << endl;

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
 //cl::ParseCommandLineOptions(argc, argv, "Bitsy compiler\n");


  return 0;
}

int dumpMLIR() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<bitsy::BitsyDialect>();

//  auto moduleAST = parseInputFile("test.mlir");
//  if (!moduleAST)
//    return 6;
//  mlir::OwningOpRef<mlir::ModuleOp> module = mlirGen(context, *moduleAST);
//  if (!module)
//    return 1;
//
//  module->dump();
//  return 0;
}
