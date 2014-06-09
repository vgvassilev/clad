//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------
//
// File originates from the Scout project (http://scout.zih.tu-dresden.de/)

#ifndef CLAD_UTILS_STMTCLONE_H
#define CLAD_UTILS_STMTCLONE_H

#include "clang/AST/StmtVisitor.h"

#include "llvm/ADT/DenseMap.h"

namespace clang {
  class Stmt;
  class ValueDecl;
}

namespace clad {
namespace utils {

  struct StmtCloneMapping;

  class StmtClone : public clang::StmtVisitor<StmtClone, clang::Stmt*>  {
  public:
    // first: original stmt, second: appropriate cloned stmt
    typedef llvm::DenseMap<clang::Stmt*, clang::Stmt*> StmtMapping; 
    typedef llvm::DenseMap<clang::ValueDecl*, clang::ValueDecl*> DeclMapping; 
    typedef StmtCloneMapping Mapping; 

  private:  
    clang::ASTContext& Ctx;
    Mapping* m_OriginalToClonedStmts;

    clang::Decl* CloneDecl(clang::Decl* Node);
    clang::VarDecl* CloneDeclOrNull(clang::VarDecl* Node);

  public:
    StmtClone(clang::ASTContext& ctx, Mapping* originalToClonedStmts = 0)
      : Ctx(ctx), m_OriginalToClonedStmts(originalToClonedStmts) {}

    template<class StmtTy>
    StmtTy* Clone(StmtTy* S);

  // visitor part (not for public use)
  // Stmt.def could be used if ABSTR_STMT is introduced
#define DECLARE_CLONE_FN(CLASS) clang::Stmt* Visit ## CLASS(clang::CLASS *Node);
    DECLARE_CLONE_FN(BinaryOperator)
    DECLARE_CLONE_FN(UnaryOperator)
    DECLARE_CLONE_FN(ReturnStmt)
    DECLARE_CLONE_FN(GotoStmt)
    DECLARE_CLONE_FN(IfStmt)
    DECLARE_CLONE_FN(ForStmt)
    DECLARE_CLONE_FN(NullStmt)
    DECLARE_CLONE_FN(LabelStmt)
    DECLARE_CLONE_FN(CompoundStmt)
    DECLARE_CLONE_FN(DeclRefExpr)
    DECLARE_CLONE_FN(DeclStmt)
    DECLARE_CLONE_FN(IntegerLiteral)
    DECLARE_CLONE_FN(SwitchStmt)
    DECLARE_CLONE_FN(CaseStmt)
    DECLARE_CLONE_FN(DefaultStmt)
    DECLARE_CLONE_FN(WhileStmt)
    DECLARE_CLONE_FN(DoStmt)
    DECLARE_CLONE_FN(ContinueStmt)
    DECLARE_CLONE_FN(BreakStmt)
    DECLARE_CLONE_FN(CXXCatchStmt)
    DECLARE_CLONE_FN(CXXTryStmt)
    DECLARE_CLONE_FN(PredefinedExpr)
    DECLARE_CLONE_FN(CharacterLiteral)
    DECLARE_CLONE_FN(FloatingLiteral)
    DECLARE_CLONE_FN(ImaginaryLiteral)
    DECLARE_CLONE_FN(StringLiteral)
    DECLARE_CLONE_FN(ParenExpr)
    DECLARE_CLONE_FN(ArraySubscriptExpr)
    DECLARE_CLONE_FN(MemberExpr)
    DECLARE_CLONE_FN(CompoundLiteralExpr)
    DECLARE_CLONE_FN(ImplicitCastExpr)
    DECLARE_CLONE_FN(UnresolvedLookupExpr)
    DECLARE_CLONE_FN(CStyleCastExpr)
    DECLARE_CLONE_FN(CompoundAssignOperator)
    DECLARE_CLONE_FN(ConditionalOperator)
    DECLARE_CLONE_FN(InitListExpr)
    DECLARE_CLONE_FN(DesignatedInitExpr)
    DECLARE_CLONE_FN(AddrLabelExpr)
    DECLARE_CLONE_FN(StmtExpr)
    DECLARE_CLONE_FN(ChooseExpr)
    DECLARE_CLONE_FN(GNUNullExpr)
    DECLARE_CLONE_FN(VAArgExpr)
    DECLARE_CLONE_FN(ImplicitValueInitExpr)
    DECLARE_CLONE_FN(ExtVectorElementExpr)
    DECLARE_CLONE_FN(UnaryExprOrTypeTraitExpr)
    DECLARE_CLONE_FN(CallExpr)
    DECLARE_CLONE_FN(ShuffleVectorExpr)
    DECLARE_CLONE_FN(CXXOperatorCallExpr)
    DECLARE_CLONE_FN(CXXMemberCallExpr)
    DECLARE_CLONE_FN(CXXStaticCastExpr)
    DECLARE_CLONE_FN(CXXDynamicCastExpr)
    DECLARE_CLONE_FN(CXXReinterpretCastExpr)
    DECLARE_CLONE_FN(CXXConstCastExpr)
    DECLARE_CLONE_FN(CXXFunctionalCastExpr)
    DECLARE_CLONE_FN(CXXBoolLiteralExpr)
    DECLARE_CLONE_FN(CXXNullPtrLiteralExpr)
    DECLARE_CLONE_FN(CXXThisExpr)
    DECLARE_CLONE_FN(CXXThrowExpr)
    DECLARE_CLONE_FN(CXXConstructExpr)
    DECLARE_CLONE_FN(CXXTemporaryObjectExpr)
    DECLARE_CLONE_FN(MaterializeTemporaryExpr)

    clang::Stmt* VisitStmt(clang::Stmt*);
  };

  // Not a StmtClone member class to make it forwardable:
  struct StmtCloneMapping {
    StmtClone::StmtMapping m_StmtMapping; 
    StmtClone::DeclMapping m_DeclMapping;
  };

  template<class StmtTy>
  StmtTy* StmtClone::Clone(StmtTy* S) {
    if (!S)
      return 0;

    clang::Stmt* clonedStmt = Visit(S);

    if (m_OriginalToClonedStmts)
      m_OriginalToClonedStmts->m_StmtMapping[S] = clonedStmt;

    return static_cast<StmtTy*>(clonedStmt);
  }

} // namespace ASTProcessing
} // namespace clang

#endif  //CLAD_UTILS_STMTCLONE_H
