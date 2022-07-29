//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clang/Basic/Version.h" // for CLANG_VERSION_MAJOR

#if CLANG_VERSION_MAJOR > 8

#include "ClangBackendPlugin.h"

#include "llvm/Passes/PassBuilder.h"

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

#endif // CLANG_VERSION_MAJOR > 8
