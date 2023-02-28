#ifndef SIMPLEMLIR_TRANSFORMS_PASSDETAILS_H
#define SIMPLEMLIR_TRANSFORMS_PASSDETAILS_H

#include "mlir/Pass/Pass.h"

#include "simplepass/Passes/Passes.h"

namespace mlir {
namespace simplepass {

#define GEN_PASS_CLASSES
#include "simplepass/Passes/Passes.h.inc"

} // namespace simplepass
} // namespace mlir

#endif // SIMPLEMLIR_TRANSFORMS_PASSDETAILS_H
