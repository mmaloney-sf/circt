add_circt_dialect_library(CIRCTArc
  Dialect.cpp
  Ops.cpp
  Types.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Arc

  DEPENDS
  CIRCTArcEnumsIncGen
  MLIRArcIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)

add_dependencies(circt-headers
  MLIRArcIncGen
)

add_subdirectory(Interfaces)
add_subdirectory(Transforms)
