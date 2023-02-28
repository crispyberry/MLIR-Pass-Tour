#ifndef SIMPLE_MLIR_PASSES_H
#define SIMPLE_MLIR_PASSES_H

include "mlir/Pass/PassBase.td"

def CustomAttrToSCFPass : Pass<"custom-attr-to-scf-pass"> {
  let summary = "Add custom atrributes to SCF operations";
  let constructor = "mlir::simplepass::createCustomAttrToSCFPass()";
}
#endif // SIMPLE_MLIR_PASSES_H