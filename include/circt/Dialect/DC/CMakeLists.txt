add_circt_dialect(DC DC)
add_circt_dialect_doc(DC DC)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTDCTransformsIncGen)
add_circt_doc(Passes DCPasses -gen-pass-doc)
