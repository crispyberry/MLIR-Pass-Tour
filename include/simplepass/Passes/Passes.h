#ifndef SIMPLEMLIR_PASSES_H
#define SIMPLEMLIR_PASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;
namespace simplepass {
std::unique_ptr<Pass> createCustomAttrToSCFPass();
} // namespace simplepass
} // namespace mlir

#endif // SIMPLEMLIR_PASSES_H