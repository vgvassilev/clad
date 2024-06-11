//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_DIFFERENTIATOR_PUSHFORWARDMODEVISITOR_H
#define CLAD_DIFFERENTIATOR_PUSHFORWARDMODEVISITOR_H

#include "BaseForwardModeVisitor.h"

namespace clad {

/// A visitor for processing the function code in forward mode.
/// Used to compute derivatives by clad::differentiate.
class PushForwardModeVisitor : public BaseForwardModeVisitor {

public:
  PushForwardModeVisitor(DerivativeBuilder& builder,
                         const DiffRequest& request);
  ~PushForwardModeVisitor() override;

  StmtDiff VisitReturnStmt(const clang::ReturnStmt* RS) override;
};
} // end namespace clad

#endif // CLAD_DIFFERENTIATOR_PUSHFORWARDMODEVISITOR_H
