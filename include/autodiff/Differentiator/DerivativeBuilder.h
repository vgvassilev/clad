//--------------------------------------------------------------------*- C++ -*-
// AutoDiff - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef AUTODIFF_DERIVATIVE_BUILDER_H
#define AUTODIFF_DERIVATIVE_BUILDER_H

#include "clang/AST/RecursiveASTVisitor.h"

namespace clang {
  class DeclRefExpr;
  class FunctionDecl;
}

namespace autodiff {
  class DerivativeBuilder 
    : public clang::RecursiveASTVisitor<DerivativeBuilder> {
    
  public:

    ///\brief Produces the first derivative of a given function.
    ///
    ///\param[in] FD - the function that will be differentiated.
    ///
    ///\returns The differentiated function.
    ///
    const clang::FunctionDecl* Derive(const clang::FunctionDecl* FD) const;

    bool VisitDeclRefExpr(clang::DeclRefExpr* DRE);        
  };

} // end namespace autodiff

#endif // AUTODIFF_DERIVATIVE_BUILDER_H
