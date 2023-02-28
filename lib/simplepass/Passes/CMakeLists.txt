add_mlir_dialect_library(MLIRSimplePassTransform
  CustomAttrToSCF.cpp

  ${PROJECT_SOURCE_DIR}/include/simplepass/Passes

  DEPENDS
  MLIRSimplePassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSCFDialect
)