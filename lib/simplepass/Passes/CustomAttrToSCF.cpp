#include "PassDetails.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "simplepass/Passes/Passes.h"

using namespace mlir;
using namespace simplepass;

namespace {
struct MyAttribute {
  static StringAttr get(MLIRContext *context) {
    return StringAttr::get(context, "my_string_value");
  }
};

struct CustomAttrToSCFPass
    : public PassWrapper<CustomAttrToSCFPass, OperationPass<>> {
  void runOnOperation() override;
};
} // namespace

void CustomAttrToSCFPass::runOnOperation() {
  auto ops = getOperation();
  auto &context = getContext();
  ops->walk([&](scf::ForOp forOp) {
    // Create the custom attribute.
    StringAttr attr = MyAttribute::get(&context);

    // Set the custom attribute on the `scf.for` operation.
    forOp->setAttr("my_attribute", attr);
  });
}

std::unique_ptr<Pass> mlir::simplepass::createCustomAttrToSCFPass() {
  return std::make_unique<CustomAttrToSCFPass>();
}