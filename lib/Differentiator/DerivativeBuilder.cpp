//--------------------------------------------------------------------*- C++ -*-
// AutoDiff - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "autodiff/Differentiator/DerivativeBuilder.h"

using namespace clang;

namespace autodiff {
  const FunctionDecl* DerivativeBuilder::Derive(const FunctionDecl* FD) const {
    return FD;
  }

  bool DerivativeBuilder::VisitDeclRefExpr(DeclRefExpr* DRE) {
    return true;     // return false to abort visiting.
  }

} // end namespace autodiff
