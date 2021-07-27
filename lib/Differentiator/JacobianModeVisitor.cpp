//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clad/Differentiator/JacobianModeVisitor.h"

#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/ReverseModeVisitor.h"
#include "clad/Differentiator/StmtClone.h"

#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Template.h"

#include "llvm/Support/SaveAndRestore.h"

#include <algorithm>
#include <numeric>

#include "clad/Differentiator/Compatibility.h"

using namespace clang;

namespace clad {
  JacobianModeVisitor::JacobianModeVisitor(DerivativeBuilder& builder)
      : VisitorBase(builder), builder(builder) {}

  JacobianModeVisitor::~JacobianModeVisitor() {}

  OverloadedDeclWithContext
  JacobianModeVisitor::Derive(const clang::FunctionDecl* FD,
                              const DiffRequest& request) {
    FD = FD->getDefinition();
    OverloadedDeclWithContext result{};

    ReverseModeVisitor V(this->builder);
    result = V.Derive(FD, request);

    return result;
  }
} // end namespace clad