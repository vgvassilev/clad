//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLANG_BACKEND_PLUGIN_H
#define CLANG_BACKEND_PLUGIN_H

#include "llvm/IR/PassManager.h"
#if CLANG_VERSION_MAJOR < 22
#include "llvm/Passes/PassPlugin.h"
#else
#include "llvm/Plugins/PassPlugin.h"
#endif



namespace clad {

struct ClangBackendPluginPass
    : public llvm::PassInfoMixin<ClangBackendPluginPass> {
  llvm::PreservedAnalyses run(llvm::Module& M,
                              llvm::ModuleAnalysisManager& MAM) {
    return llvm::PreservedAnalyses::all();
  }

  static void registerCallbacks(llvm::PassBuilder& PB);
};
} // end namespace clad

#endif // CLANG_BACKEND_PLUGIN_H
