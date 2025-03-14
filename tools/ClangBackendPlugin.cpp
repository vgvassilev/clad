//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clang/Basic/Version.h" // for CLANG_VERSION_MAJOR

#include "ClangBackendPlugin.h"

#include "llvm/Passes/PassBuilder.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Config/llvm-config.h" // for CLANG_VERSION_MAJOR
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"

#if CLANG_VERSION_MAJOR < 16
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#endif // CLANG_VERSION_MAJOR < 16

namespace clad {
using namespace llvm;
void ClangBackendPluginPass::registerCallbacks(PassBuilder& PB) {
  PB.registerPipelineParsingCallback(
      [](StringRef Name, ModulePassManager& PM,
         ArrayRef<PassBuilder::PipelineElement> InnerPipeline) {
        if (Name == "plugin-pass") {
          PM.addPass(ClangBackendPluginPass());
          return true;
        }
        return false;
      });
}
} // namespace clad

#if LLVM_VERSION_MAJOR < 16

static void loadEnzymePass(const llvm::PassManagerBuilder& Builder,
                           llvm::legacy::PassManagerBase& PM) {
  llvm::PassRegistry* PR = llvm::PassRegistry::getPassRegistry();
  const llvm::PassInfo* enzymePassInfo =
      PR->getPassInfo(llvm::StringRef("enzyme"));
  // if enzyme pass is not found, then return
  if (!enzymePassInfo)
    return;
  llvm::Pass* enzymePassInstance = enzymePassInfo->createPass();
  PM.add(enzymePassInstance);
}
static void loadNVVMPass(const llvm::PassManagerBuilder& Builder,
                         llvm::legacy::PassManagerBase& PM) {
  llvm::PassRegistry* PR = llvm::PassRegistry::getPassRegistry();
  const llvm::PassInfo* nvvmPassInfo =
      PR->getPassInfo(llvm::StringRef("preserve-nvvm"));
  // if nvvm pass is not found, then return
  if (!nvvmPassInfo)
    return;
  llvm::Pass* nvvmPassInstance = nvvmPassInfo->createPass();
  PM.add(nvvmPassInstance);
}

// These constructors add our pass to a list of global extensions.
static llvm::RegisterStandardPasses
    enzymePassLoader_Ox(llvm::PassManagerBuilder::EP_VectorizerStart,
                        loadEnzymePass);
static llvm::RegisterStandardPasses
    enzymePassLoader_O0(llvm::PassManagerBuilder::EP_EnabledOnOptLevel0,
                        loadEnzymePass);
static llvm::RegisterStandardPasses
    nvvmPassLoader_OEarly(llvm::PassManagerBuilder::EP_EarlyAsPossible,
                          loadNVVMPass);

#endif // LLVM_VERSION_MAJOR < 16
