#include "simplepass/Passes/Passes.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace mlir;
using namespace simplepass;

static cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                         llvm::cl::desc("<input file>"),
                                         llvm::cl::init("-"),
                                         llvm::cl::value_desc("filename"));
static cl::opt<std::string> outputFilename("o",
                                          llvm::cl::desc("Output filename"),
                                          llvm::cl::value_desc("filename"),
                                          llvm::cl::init("-"));
static cl::opt<bool>
    CustomAttrToSCFPass("customattr-to-scf-pass", cl::init(true),
                       cl::desc("Turn on customattr-to-scf-pass"));

int main(int argc, char **argv) {
  // Register all MLIR dialects and passes.

  mlir::registerAllPasses();

  // Parse command line arguments.
  mlir::DialectRegistry registry;

  mlir::registerAllDialects(registry);

  MLIRContext context(registry);

  mlir::PassManager pm(&context);

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "\n");

  llvm::errs() << inputFilename << "!\n";
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  llvm::errs() << "LOADed\n";
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  llvm::SourceMgr sourceMgr;

  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());

  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error loading input file\n";
    return 1;
  }

  // Add the custom attribute pass to the pass manager.
  if (CustomAttrToSCFPass) {
    llvm::errs() << "In\n";
    pm.addPass(mlir::simplepass::createCustomAttrToSCFPass());

    // Run the pass on the module.
    if (failed(pm.run(*module))) {
      llvm::errs() << "Error running pass\n";
      return 1;
    }
  }
  // Write the output file.
  std::error_code error;
  llvm::raw_fd_ostream output(outputFilename, error, llvm::sys::fs::OF_Text);
  if (error) {
    llvm::errs() << "Error opening output file: " << error.message() << "\n";
    return 1;
  }
  module->print(output);

  return 0;
}

// static cl::opt<bool>
//     CustomAttrToSCFPass("customattr-to-scf-pass", cl::init(true),
//                        cl::desc("Turn on customattr-to-scf-pass"));

// static cl::opt<std::string> inputFilename(cl::Positional,
//                                           cl::desc("<input .mlir>"),
//                                           cl::init("-"),
//                                           cl::value_desc("filename"));

// int loadMLIR(llvm::SourceMgr &sourceMgr, mlir::MLIRContext &context,
//              mlir::OwningOpRef<mlir::ModuleOp> &module) {

//   // Otherwise, the input is '.mlir'.
//   llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
//       llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
//   if (std::error_code ec = fileOrErr.getError()) {
//     llvm::errs() << "Could not open input file: " << ec.message() << "\n";
//     return -1;
//   }

//   // Parse the input mlir.
//   sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
//   module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
//   if (!module) {
//     llvm::errs() << "Error can't load file " << inputFilename << "\n";
//     return 3;
//   }
//   return 0;
// }

// int main(int argc, char **argv) {
//   mlir::MLIRContext context;

//   mlir::OwningOpRef<mlir::ModuleOp> module;
//   llvm::SourceMgr sourceMgr;
//   mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
//   if (int error = loadMLIR(sourceMgr, context, module))
//     return error;

//   cl::ParseCommandLineOptions(argc, argv, "simplepass!\n");

//   mlir::PassManager pm(module.get()->getName());
//   if (CustomAttrToSCFPass) {
//     pm.addPass(mlir::simplepass::createCustomAttrToSCFPass());
//     module->dump();
//   }
//   return 0;
// }