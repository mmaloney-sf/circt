add_mlir_dialect_library(CIRCTBitsy
        BitsyDialect.cpp
        BitsyOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Dialect/Bitsy

        DEPENDS
        #        BitsyIncGen
        MLIRBitsyIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
