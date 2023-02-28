set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name SimplePass)
add_public_tablegen_target(MLIRSimplePassIncGen)

add_mlir_doc(Passes MLIRSimplePassPasses ./ -gen-pass-doc)