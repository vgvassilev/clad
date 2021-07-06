//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_FORWARD_MODE_VISITOR_H
#define CLAD_FORWARD_MODE_VISITOR_H

#include "Compatibility.h"
#include "VisitorBase.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Sema/Sema.h"

#include <array>
#include <stack>
#include <unordered_map>

namespace clad {
  /// A visitor for processing the function code in forward mode.
  /// Used to compute derivatives by clad::differentiate.
  class ForwardModeVisitor
      : public clang::ConstStmtVisitor<ForwardModeVisitor, StmtDiff>,
        public VisitorBase {
  private:
    const clang::VarDecl* m_IndependentVar = nullptr;
    unsigned m_IndependentVarIndex = ~0;
    unsigned m_DerivativeOrder = ~0;
    unsigned m_ArgIndex = ~0;

  public:
    ForwardModeVisitor(DerivativeBuilder& builder);
    ~ForwardModeVisitor();

    ///\brief Produces the first derivative of a given function.
    ///
    ///\param[in] FD - the function that will be differentiated.
    ///
    ///\returns The differentiated and potentially created enclosing
    /// context.
    ///
    OverloadedDeclWithContext Derive(const clang::FunctionDecl* FD,
                                     const DiffRequest& request);
    StmtDiff VisitArraySubscriptExpr(const clang::ArraySubscriptExpr* ASE);
    StmtDiff VisitBinaryOperator(const clang::BinaryOperator* BinOp);
    StmtDiff VisitCallExpr(const clang::CallExpr* CE);
    StmtDiff VisitCompoundStmt(const clang::CompoundStmt* CS);
    StmtDiff VisitConditionalOperator(const clang::ConditionalOperator* CO);
    StmtDiff VisitCXXBoolLiteralExpr(const clang::CXXBoolLiteralExpr* BL);
    StmtDiff VisitCXXDefaultArgExpr(const clang::CXXDefaultArgExpr* DE);
    StmtDiff VisitCXXOperatorCallExpr(const clang::CXXOperatorCallExpr* OpCall);
    StmtDiff VisitDeclRefExpr(const clang::DeclRefExpr* DRE);
    StmtDiff VisitDeclStmt(const clang::DeclStmt* DS);
    StmtDiff VisitFloatingLiteral(const clang::FloatingLiteral* FL);
    StmtDiff VisitForStmt(const clang::ForStmt* FS);
    StmtDiff VisitIfStmt(const clang::IfStmt* If);
    StmtDiff VisitImplicitCastExpr(const clang::ImplicitCastExpr* ICE);
    StmtDiff VisitInitListExpr(const clang::InitListExpr* ILE);
    StmtDiff VisitIntegerLiteral(const clang::IntegerLiteral* IL);
    StmtDiff VisitMemberExpr(const clang::MemberExpr* ME);
    StmtDiff VisitParenExpr(const clang::ParenExpr* PE);
    StmtDiff VisitReturnStmt(const clang::ReturnStmt* RS);
    StmtDiff VisitStmt(const clang::Stmt* S);
    StmtDiff VisitUnaryOperator(const clang::UnaryOperator* UnOp);
    // Decl is not Stmt, so it cannot be visited directly.
    VarDeclDiff DifferentiateVarDecl(const clang::VarDecl* VD);
    /// Shorthand for warning on differentiation of unsupported operators
    void unsupportedOpWarn(clang::SourceLocation loc,
                           llvm::ArrayRef<llvm::StringRef> args = {}) {
      diag(clang::DiagnosticsEngine::Warning,
           loc,
           "attempt to differentiate unsupported operator,  derivative \
                         set to 0",
           args);
    }
  };
} // end namespace clad

#endif // CLAD_FORWARD_MODE_VISITOR_H
