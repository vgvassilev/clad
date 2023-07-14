//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------
//
// File originates from the Scout project (http://scout.zih.tu-dresden.de/)

#ifndef CLAD_UTILS_STMTCLONE_H
#define CLAD_UTILS_STMTCLONE_H

#include "clang/AST/StmtVisitor.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Version.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/Scope.h"

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
    typedef llvm::DenseMap<const clang::Stmt*, clang::Stmt*> StmtMapping;
    typedef llvm::DenseMap<clang::ValueDecl*, clang::ValueDecl*> DeclMapping;
    typedef StmtCloneMapping Mapping;

  private:
    clang::Sema& m_Sema;
    clang::ASTContext& Ctx;
    Mapping* m_OriginalToClonedStmts;

    clang::Decl* CloneDecl(clang::Decl* Node);
    clang::VarDecl* CloneDeclOrNull(clang::VarDecl* Node);

  public:
    StmtClone(clang::Sema& sema, clang::ASTContext& ctx, Mapping* originalToClonedStmts = 0)
      : m_Sema(sema), Ctx(ctx), m_OriginalToClonedStmts(originalToClonedStmts) {}

    template<class StmtTy>
    StmtTy* Clone(const StmtTy* S);

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
    DECLARE_CLONE_FN(ExprWithCleanups)
    DECLARE_CLONE_FN(CXXOperatorCallExpr)
    DECLARE_CLONE_FN(CXXMemberCallExpr)
    DECLARE_CLONE_FN(CXXStaticCastExpr)
    DECLARE_CLONE_FN(CXXDynamicCastExpr)
    DECLARE_CLONE_FN(CXXReinterpretCastExpr)
    DECLARE_CLONE_FN(CXXConstCastExpr)
    DECLARE_CLONE_FN(CXXDefaultArgExpr)
    DECLARE_CLONE_FN(CXXFunctionalCastExpr)
    DECLARE_CLONE_FN(CXXBoolLiteralExpr)
    DECLARE_CLONE_FN(CXXNullPtrLiteralExpr)
    DECLARE_CLONE_FN(CXXThisExpr)
    DECLARE_CLONE_FN(CXXThrowExpr)
    DECLARE_CLONE_FN(CXXConstructExpr)
    DECLARE_CLONE_FN(CXXTemporaryObjectExpr)
    DECLARE_CLONE_FN(MaterializeTemporaryExpr)
    DECLARE_CLONE_FN(PseudoObjectExpr)
    DECLARE_CLONE_FN(SubstNonTypeTemplateParmExpr)
    // `ConstantExpr` node is only available after clang 7.
    #if CLANG_VERSION_MAJOR > 7
    DECLARE_CLONE_FN(ConstantExpr)
    #endif
    
    clang::Stmt* VisitStmt(clang::Stmt*);
  };

  // Not a StmtClone member class to make it forwardable:
  struct StmtCloneMapping {
    StmtClone::StmtMapping m_StmtMapping;
    StmtClone::DeclMapping m_DeclMapping;
  };

  template<class StmtTy>
  StmtTy* StmtClone::Clone(const StmtTy* S) {
    if (!S)
      return 0;

    clang::Stmt* clonedStmt = Visit(const_cast<StmtTy*>(S));

    if (m_OriginalToClonedStmts)
      m_OriginalToClonedStmts->m_StmtMapping[S] = clonedStmt;

    return static_cast<StmtTy*>(clonedStmt);
  }

  class ReferencesUpdater :
    public clang::RecursiveASTVisitor<ReferencesUpdater> {
  private:
    clang::Sema& m_Sema; // We don't own.
    clang::Scope* m_CurScope; // We don't own.
    const clang::FunctionDecl* m_Function; // We don't own.
  public:
    ReferencesUpdater(clang::Sema& SemaRef, clang::Scope* S,
                      const clang::FunctionDecl* FD);
    bool VisitDeclRefExpr(clang::DeclRefExpr* DRE);
  };
} // namespace utils
} // namespace clad

#endif  //CLAD_UTILS_STMTCLONE_H
