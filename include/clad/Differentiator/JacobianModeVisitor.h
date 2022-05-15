//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_JACOBIAN_MODE_VISITOR_H
#define CLAD_JACOBIAN_MODE_VISITOR_H

#include "Compatibility.h"
#include "VisitorBase.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Sema/Sema.h"

#include <array>
#include <stack>
#include <unordered_map>

namespace clad {
  /// A visitor for processing the function code to generate jacobian
  /// Used to compute Jacobian matrices by clad::jacobian.
  class JacobianModeVisitor
      : public clang::ConstStmtVisitor<JacobianModeVisitor, StmtDiff>,
        public VisitorBase {
  private:
    DerivativeBuilder& builder;

  public:
    JacobianModeVisitor(DerivativeBuilder& builder);
    ~JacobianModeVisitor();

    ///\brief Produces the jacobian matrix of a given function.
    ///
    ///\param[in] FD - the function that will be differentiated.
    ///
    ///\returns A function containing jacobian matrix.
    ///
    DerivativeAndOverload Derive(const clang::FunctionDecl* FD,
                                 const DiffRequest& request);
  };
} // end namespace clad

#endif // CLAD_JACOBIAN_MODE_VISITOR_H