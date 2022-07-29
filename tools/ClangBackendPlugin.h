//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLANG_BACKEND_PLUGIN_H
#define CLANG_BACKEND_PLUGIN_H

#include "llvm/Config/llvm-config.h" // for CLANG_VERSION_MAJOR

#if CLANG_VERSION_MAJOR > 8

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"

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

#endif // CLANG_VERSION_MAJOR > 8
#endif // CLANG_BACKEND_PLUGIN_H
