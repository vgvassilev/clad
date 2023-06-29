//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_FORWARD_MODE_VISITOR_H
#define CLAD_FORWARD_MODE_VISITOR_H

#include "BaseForwardModeVisitor.h"

namespace clad {
  /// A visitor for processing the function code in forward mode.
  /// Used to compute derivatives by clad::differentiate.
class ForwardModeVisitor : public BaseForwardModeVisitor {

public:
  ForwardModeVisitor(DerivativeBuilder& builder);
  ~ForwardModeVisitor();

  DerivativeAndOverload DerivePushforward(const clang::FunctionDecl* FD,
                                          const DiffRequest& request);

  /// Returns the return type for the pushforward function of the function
  /// `m_Function`.
  /// \note `m_Function` field should be set before using this function.
  clang::QualType ComputePushforwardFnReturnType();

  StmtDiff VisitReturnStmt(const clang::ReturnStmt* RS) override;
};
} // end namespace clad

#endif // CLAD_FORWARD_MODE_VISITOR_H
