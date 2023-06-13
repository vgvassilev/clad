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
    const clang::ValueDecl* m_IndependentVar = nullptr;
    unsigned m_IndependentVarIndex = ~0;
    unsigned m_DerivativeOrder = ~0;
    unsigned m_ArgIndex = ~0;
    /// Map used to keep track of variable declarations and match them
    /// with their vectorized derivatives - only used in vector mode.
    std::unordered_map<const clang::ValueDecl*, clang::Expr*> m_VectorVariables;

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
    DerivativeAndOverload Derive(const clang::FunctionDecl* FD,
                                 const DiffRequest& request);
    DerivativeAndOverload DerivePushforward(const clang::FunctionDecl* FD,
                                            const DiffRequest& request);
    DerivativeAndOverload DeriveVectorMode(const clang::FunctionDecl* FD,
                                           const DiffRequest& request);
    StmtDiff VisitArraySubscriptExpr(const clang::ArraySubscriptExpr* ASE);
    StmtDiff VisitBinaryOperator(const clang::BinaryOperator* BinOp);
    StmtDiff VisitCallExpr(const clang::CallExpr* CE);
    StmtDiff VisitCompoundStmt(const clang::CompoundStmt* CS);
    StmtDiff VisitConditionalOperator(const clang::ConditionalOperator* CO);
    StmtDiff VisitCXXBoolLiteralExpr(const clang::CXXBoolLiteralExpr* BL);
    StmtDiff VisitCXXDefaultArgExpr(const clang::CXXDefaultArgExpr* DE);
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
    StmtDiff VisitWhileStmt(const clang::WhileStmt* WS);
    StmtDiff VisitDoStmt(const clang::DoStmt* DS);
    StmtDiff VisitContinueStmt(const clang::ContinueStmt* ContStmt);

    StmtDiff VisitSwitchStmt(const clang::SwitchStmt* SS);
    StmtDiff VisitBreakStmt(const clang::BreakStmt* BS);
    StmtDiff VisitCXXConstructExpr(const clang::CXXConstructExpr* CE);
    StmtDiff VisitExprWithCleanups(const clang::ExprWithCleanups* EWC);
    StmtDiff
    VisitMaterializeTemporaryExpr(const clang::MaterializeTemporaryExpr* MTE);
    StmtDiff
    VisitCXXTemporaryObjectExpr(const clang::CXXTemporaryObjectExpr* TOE);
    StmtDiff VisitCXXThisExpr(const clang::CXXThisExpr* CTE);
    StmtDiff VisitCXXNewExpr(const clang::CXXNewExpr* CNE);
    StmtDiff VisitCXXDeleteExpr(const clang::CXXDeleteExpr* CDE);
    StmtDiff VisitCXXStaticCastExpr(const clang::CXXStaticCastExpr* CSE);
    StmtDiff VisitCXXFunctionalCastExpr(const clang::CXXFunctionalCastExpr* FCE);
    StmtDiff VisitCXXBindTemporaryExpr(const clang::CXXBindTemporaryExpr* BTE);
    StmtDiff VisitCXXNullPtrLiteralExpr(const clang::CXXNullPtrLiteralExpr* NPL);
    StmtDiff VisitUnaryExprOrTypeTraitExpr(const clang::UnaryExprOrTypeTraitExpr* UE);
    StmtDiff VisitPseudoObjectExpr(const clang::PseudoObjectExpr* POE);

    /// Builds and returns the sequence of derived function parameters for
    //  vectorized forward mode.
    ///
    /// Information about the original function, derived function, derived
    /// function parameter types and the differentiation mode are implicitly
    /// taken from the data member variables. In particular, `m_Function`,
    /// `m_Mode` and `m_Derivative` should be correctly set before using this
    /// function.
    llvm::SmallVector<clang::ParmVarDecl*, 8>
    BuildVectorModeParams(DiffParams& diffParams);

    /// Get an expression used to initialize the one-hot vector for the
    /// given index and size. A one-hot vector is a vector with all elements
    /// set to 0 except for one element which is set to 1.
    ///
    /// For example: for index = 2 and size = 4, the returned expression
    /// is: {0, 0, 1, 0}
    clang::Expr* getOneHotInitExpr(size_t index, size_t size);

  private:
    /// Helper function for differentiating the switch statement body.
    ///
    /// It manages scopes and blocks for the switch case labels, checks if
    /// compound statement to be differentiated is supported and returns the
    /// active switch case label after processing the given `stmt` argument.
    ///
    /// Scope and and block for the last switch case label have to be managed
    /// manually outside the function because this function have no way of
    /// knowing when all the statements belonging to last switch case label have
    /// been processed.
    ///
    /// \param[in] stmt Current statement to derive
    /// \param[in] activeSC Current active switch case label
    /// \return active switch case label after processing `stmt`
    clang::SwitchCase* DeriveSwitchStmtBodyHelper(const clang::Stmt* stmt,
                                                  clang::SwitchCase* activeSC);

    /// Returns the return type for the pushforward function of the function
    /// `m_Function`.
    /// \note `m_Function` field should be set before using this function.
    clang::QualType ComputePushforwardFnReturnType();
  };
} // end namespace clad

#endif // CLAD_FORWARD_MODE_VISITOR_H
