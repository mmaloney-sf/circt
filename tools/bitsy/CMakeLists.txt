set(LLVM_LINK_COMPONENTS Support)

add_circt_tool(bitsy-mlir bitsy.cpp)
target_link_libraries(bitsy-mlir
    PRIVATE
    CIRCTBitsy)
