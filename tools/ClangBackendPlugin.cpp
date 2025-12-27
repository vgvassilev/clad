//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clang/Basic/Version.h" // for CLANG_VERSION_MAJOR

#include "ClangBackendPlugin.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Config/llvm-config.h" // for CLANG_VERSION_MAJOR
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#if CLANG_VERSION_MAJOR > 15
#include "llvm/Passes/OptimizationLevel.h"
#endif
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

#include <string>

#ifdef CLAD_ENABLE_ENZYME_BACKEND
extern "C" void registerEnzyme(llvm::PassBuilder& PB);
#endif // CLAD_ENABLE_ENZYME_BACKEND

namespace clad {
using namespace llvm;
void ClangBackendPluginPass::registerCallbacks(PassBuilder& PB) {
// Enable backend plugins only with the new pass manager.
#if CLANG_VERSION_MAJOR > 15
#ifdef CLAD_ENABLE_ENZYME_BACKEND
  registerEnzyme(PB);
#endif // CLAD_ENABLE_ENZYME_BACKEND
  // Pre
  PB.registerPipelineStartEPCallback([&](llvm::ModulePassManager& MPM,
                                         llvm::OptimizationLevel) {
    MPM.addPass(ClangBackendPluginPass());
    // registerEnzyme adds enzyme via registerPipelineParsingCallback,
    // however, clang does not trigger this callback unless
    // parsePassPipeline is called.
#if CLAD_ENABLE_ENZYME_BACKEND
    std::string Pipeline = "enzyme";
    if (auto Err = PB.parsePassPipeline(MPM, Pipeline))
      report_fatal_error(Twine("unable to parse pass pipeline description '") +
                         Pipeline + "': " + toString(std::move(Err)));
#endif // CLAD_ENABLE_ENZYME_BACKEND
  });
  // Post
  // PB.registerOptimizerEarlyEPCallback(
  //     [](ModulePassManager &MPM, OptimizationLevel) {
  //     });
#endif // CLANG_VERSION_MAJOR > 15
}
} // namespace clad
