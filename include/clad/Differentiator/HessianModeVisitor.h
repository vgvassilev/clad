//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_HESSIAN_MODE_VISITOR_H
#define CLAD_HESSIAN_MODE_VISITOR_H

#include "Compatibility.h"
#include "VisitorBase.h"
#include "clad/Differentiator/ParseDiffArgsTypes.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/SmallVector.h"

#include <array>
#include <cstddef>
#include <stack>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace clad {
  /// A visitor for processing the function code to generate hessians
  /// Used to compute Hessian matrices by clad::hessian.
  class HessianModeVisitor
      : public clang::ConstStmtVisitor<HessianModeVisitor, StmtDiff>,
        public VisitorBase {
  private:
    /// A helper method that combines all the generated second derivatives
    /// (contained within a vector) obtained from Derive
    /// into a single FunctionDecl f_hessian
    DerivativeAndOverload
    Merge(std::vector<clang::FunctionDecl*> secDerivFuncs,
          llvm::SmallVector<size_t, 16> IndependentArgsSize,
          size_t TotalIndependentArgsSize, const std::string& hessianFuncName,
          clang::DeclContext* FD, clang::QualType hessianFuncType);

    /// Derives the pair of functions a hessian-vector product is built from:
    /// a pushforward of the function, whose direction is a run-time argument,
    /// and a pullback of that pushforward that takes adjoints for the
    /// parameters in \p args. Only called when the planner committed to the
    /// scheme (DiffRequest::UseHessianVectorProducts) and scheduled the
    /// pushforward.
    ///
    /// \returns both, or a pair of nulls when either could not be built --
    /// an error, since no per-direction derivatives were scheduled to fall
    /// back on.
    std::pair<clang::FunctionDecl*, clang::FunctionDecl*>
    DeriveVectorProductFunctions(const DiffParams& args);

    /// Builds `f_hessian` as a sequence of hessian-vector products.
    ///
    /// One pushforward of the function carries the direction as a run-time
    /// argument, and one pullback of that pushforward, seeded with
    /// `{.value = 0, .pushforward = 1}`, turns a direction into the matching
    /// row of the hessian. So two derivatives serve every direction, and the
    /// wrapper only has to seed a tangent and call. \p IndependentArgsSize
    /// holds the number of requested directions per independent parameter, in
    /// parameter order, and \p TotalIndependentArgsSize their sum -- the row
    /// length of the hessian matrix.
    DerivativeAndOverload BuildHessianFromVectorProducts(
        clang::FunctionDecl* pushforwardFD, clang::FunctionDecl* pullbackFD,
        const DiffParams& args, const IndexIntervalTable& indexIntervalTable,
        llvm::SmallVector<size_t, 16> IndependentArgsSize,
        size_t TotalIndependentArgsSize, const std::string& hessianFuncName,
        clang::DeclContext* DC, clang::QualType hessianFunctionType);

  public:
    HessianModeVisitor(DerivativeBuilder& builder, const DiffRequest& request);
    ~HessianModeVisitor() override = default;

    ///\brief Produces the hessian second derivative columns of a given
    /// function.
    ///
    ///\returns A function containing second derivatives (columns) of a hessian
    /// matrix and potentially created enclosing context.
    ///
    /// We name the hessian of f as 'f_hessian'. Uses ForwardModeVisitor and
    /// ReverseModeVisitor to generate second derivatives that correspond to
    /// columns of the Hessian. uses Merge to return a FunctionDecl
    /// containing CallExprs to the generated second derivatives.
    DerivativeAndOverload Derive() override;
  };
} // end namespace clad

#endif // CLAD_HESSIAN_MODE_VISITOR_H
