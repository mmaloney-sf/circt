add_circt_conversion_library(CIRCTArcToLLVM
  LowerArcToLLVM.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTArc
  CIRCTComb
  CIRCTCombToLLVM
  CIRCTHWToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRLLVMCommonConversion
  MLIRSCFToControlFlow
  MLIRTransforms
)
