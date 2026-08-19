//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clang/Basic/Version.h" // for CLANG_VERSION_MAJOR

#include "ClangBackendPlugin.h"

#include "clad/Differentiator/Compatibility.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"

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
extern "C" void registerEnzymeAndPassPipeline(llvm::PassBuilder& PB,
                                              bool augment);
#endif // CLAD_ENABLE_ENZYME_BACKEND

namespace clad {
using namespace llvm;
void ClangBackendPluginPass::registerCallbacks(PassBuilder& PB) {
// Enable backend plugins only with the new pass manager.
#if CLANG_VERSION_MAJOR > 15
#ifdef CLAD_ENABLE_ENZYME_BACKEND
  // Register Enzyme's canonical pipeline: pre-simplification (loop rotation,
  // GVN, SROA), EnzymeNewPM at the optimizer-early extension point, and the
  // post cleanup. Splicing a bare "enzyme" pass at pipeline start hands
  // Enzyme unoptimized IR, which it is neither designed nor tuned for.
  registerEnzymeAndPassPipeline(PB, /*augment=*/true);

  // Enzyme's passes are registered for every compilation, but only a module
  // that actually contains a request needs them. Decide per module rather
  // than per process: a session-based host compiles many modules through one
  // plugin, so anything remembered across them would be wrong for all but the
  // one that set it.
  //
  // Skipping requires the passes to be optional, which they are not upstream;
  // patches/enzyme.patch makes them so. Without that patch the callback is
  // simply never consulted and every module keeps the passes, as before.
  if (PassInstrumentationCallbacks* PIC = PB.getPassInstrumentationCallbacks())
    PIC->registerShouldRunOptionalPassCallback([](llvm::StringRef PassID,
                                                  llvm::Any IR) {
      if (!PassID.contains("Enzyme") && !PassID.contains("PreserveNVVM"))
        return true;
      // Both are module passes, so anything else is not ours to skip.
      // The name is mangled -- clad declares the callee in C++ -- so
      // look for it inside the symbol rather than at its start.
      const auto* const* M = llvm::any_cast<const llvm::Module*>(&IR);
      return !M || llvm::any_of((*M)->functions(), [](const llvm::Function& F) {
        return F.getName().contains("__enzyme_autodiff");
      });
    });
#endif // CLAD_ENABLE_ENZYME_BACKEND
  PB.registerPipelineStartEPCallback(
      [&](llvm::ModulePassManager& MPM, llvm::OptimizationLevel) {
        MPM.addPass(ClangBackendPluginPass());
      });
#endif // CLANG_VERSION_MAJOR > 15
}
} // namespace clad
