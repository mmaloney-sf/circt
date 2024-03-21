add_circt_dialect(Bitsy bitsy)
add_circt_dialect_doc(Bitsy bitsy)

set(LLVM_TARGET_DEFINITIONS Bitsy.td)
# mlir_tablegen(BitsyDialect.h.inc -gen-dialect-decls)
# mlir_tablegen(BitsyOp.h.inc -gen-op-decls)

#mlir_tablegen(BitsyOpEnums.h.inc -gen-enum-decls)
#mlir_tablegen(BitsyOpEnums.cpp.inc -gen-enum-defs)
#add_circt_doc(BitsyOp BitsyOp -gen-pass-doc)

# add_public_tablegen_target(BitsyIncGen)
