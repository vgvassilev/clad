//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------
//
// File originates from the Scout project (http://scout.zih.tu-dresden.de/)

#include "clad/Differentiator/StmtClone.h"

#include "clang/Sema/Lookup.h"

#include "llvm/ADT/SmallVector.h"

#include "clad/Differentiator/Compatibility.h"

using namespace clang;

namespace clad {
namespace utils {


#define DEFINE_CLONE_STMT(CLASS, CTORARGS)    \
Stmt* StmtClone::Visit ## CLASS(CLASS *Node)  \
{                                             \
  return new (Ctx) CLASS CTORARGS;            \
}

#define DEFINE_CLONE_STMT_CO(CLASS, CTORARGS)                                  \
  Stmt* StmtClone::Visit##CLASS(CLASS* Node) {                                 \
    return (CLASS::Create CTORARGS);                                           \
  }

#define DEFINE_CLONE_EXPR(CLASS, CTORARGS)              \
Stmt* StmtClone::Visit ## CLASS(CLASS *Node)            \
{                                                       \
  CLASS* result = new (Ctx) CLASS CTORARGS;             \
  clad_compat::ExprSetDeps(result, Node);               \
  return result;                                        \
}

#define DEFINE_CREATE_EXPR(CLASS, CTORARGS)             \
Stmt* StmtClone::Visit ## CLASS(CLASS *Node)            \
{                                                       \
  CLASS* result = CLASS::Create CTORARGS;               \
  clad_compat::ExprSetDeps(result, Node);               \
  return result;                                        \
}

#define DEFINE_CLONE_EXPR_CO(CLASS, CTORARGS)                                  \
  Stmt* StmtClone::Visit##CLASS(CLASS* Node) {                                 \
    CLASS* result = (CLASS::Create CTORARGS);                                  \
    clad_compat::ExprSetDeps(result, Node);                                    \
    return result;                                                             \
  }

#define DEFINE_CLONE_EXPR_CO11(CLASS, CTORARGS)         \
Stmt* StmtClone::Visit ## CLASS(CLASS *Node)            \
{                                                       \
  CLASS* result = CLAD_COMPAT_CREATE11(CLASS, CTORARGS);\
  clad_compat::ExprSetDeps(result, Node);               \
  return result;                                        \
}
// NOLINTBEGIN(modernize-use-auto)
DEFINE_CLONE_EXPR_CO11(
    BinaryOperator,
    (CLAD_COMPAT_CLANG11_Ctx_ExtraParams Clone(Node->getLHS()),
     Clone(Node->getRHS()), Node->getOpcode(), CloneType(Node->getType()),
     Node->getValueKind(), Node->getObjectKind(), Node->getOperatorLoc(),
     Node->getFPFeatures(CLAD_COMPAT_CLANG11_LangOptions_EtraParams)))
DEFINE_CLONE_EXPR_CO11(
    UnaryOperator,
    (CLAD_COMPAT_CLANG11_Ctx_ExtraParams Clone(Node->getSubExpr()),
     Node->getOpcode(), CloneType(Node->getType()), Node->getValueKind(),
     Node->getObjectKind(), Node->getOperatorLoc(),
     Node->canOverflow() CLAD_COMPAT_CLANG11_UnaryOperator_ExtraParams))
Stmt* StmtClone::VisitDeclRefExpr(DeclRefExpr *Node) {
  TemplateArgumentListInfo TAListInfo;
  Node->copyTemplateArgumentsInto(TAListInfo);
  return DeclRefExpr::Create(
      Ctx, Node->getQualifierLoc(), Node->getTemplateKeywordLoc(),
      Node->getDecl(), Node->refersToEnclosingVariableOrCapture(),
      Node->getNameInfo(), CloneType(Node->getType()), Node->getValueKind(),
      Node->getFoundDecl(),
      &TAListInfo CLAD_COMPAT_CLANG9_MemberExpr_ExtraParams(
          Node->isNonOdrUse()));
}
DEFINE_CREATE_EXPR(IntegerLiteral,
                   (Ctx, Node->getValue(), CloneType(Node->getType()),
                    Node->getLocation()))
DEFINE_CLONE_EXPR_CO(PredefinedExpr,
                     (Ctx, Node->getLocation(), CloneType(Node->getType()),
                      Node->getIdentKind()
                          CLAD_COMPAT_CLANG17_IsTransparent(Node),
                      Node->getFunctionName()))
DEFINE_CLONE_EXPR(CharacterLiteral,
                  (Node->getValue(), Node->getKind(),
                   CloneType(Node->getType()), Node->getLocation()))
DEFINE_CLONE_EXPR(ImaginaryLiteral,
                  (Clone(Node->getSubExpr()), CloneType(Node->getType())))
DEFINE_CLONE_EXPR(ParenExpr, (Node->getLParen(), Node->getRParen(), Clone(Node->getSubExpr())))
DEFINE_CLONE_EXPR(ArraySubscriptExpr,
                  (Clone(Node->getLHS()), Clone(Node->getRHS()),
                   CloneType(Node->getType()), Node->getValueKind(),
                   Node->getObjectKind(), Node->getRBracketLoc()))
DEFINE_CREATE_EXPR(CXXDefaultArgExpr, (Ctx, SourceLocation(), Node->getParam() CLAD_COMPAT_CLANG16_CXXDefaultArgExpr_getRewrittenExpr_Param(Node) CLAD_COMPAT_CLANG9_CXXDefaultArgExpr_getUsedContext_Param(Node)))

Stmt* StmtClone::VisitMemberExpr(MemberExpr* Node) {
  TemplateArgumentListInfo TemplateArgs;
  if (Node->hasExplicitTemplateArgs())
    Node->copyTemplateArgumentsInto(TemplateArgs);
  MemberExpr* result = MemberExpr::Create(
      Ctx, Clone(Node->getBase()), Node->isArrow(), Node->getOperatorLoc(),
      Node->getQualifierLoc(), Node->getTemplateKeywordLoc(),
      Node->getMemberDecl(), Node->getFoundDecl(), Node->getMemberNameInfo(),
      &TemplateArgs, CloneType(Node->getType()), Node->getValueKind(),
      Node->getObjectKind()
          CLAD_COMPAT_CLANG9_MemberExpr_ExtraParams(Node->isNonOdrUse()));
  // Copy Value and Type dependent
  clad_compat::ExprSetDeps(result, Node);
  return result;
}
DEFINE_CLONE_EXPR(CompoundLiteralExpr,
                  (Node->getLParenLoc(), Node->getTypeSourceInfo(),
                   CloneType(Node->getType()), Node->getValueKind(),
                   Clone(Node->getInitializer()), Node->isFileScope()))
DEFINE_CREATE_EXPR(
    ImplicitCastExpr,
    (Ctx, CloneType(Node->getType()), Node->getCastKind(),
     Clone(Node->getSubExpr()), nullptr,
     Node->getValueKind() /*EP*/ CLAD_COMPAT_CLANG12_CastExpr_GetFPO(Node)))
DEFINE_CREATE_EXPR(CStyleCastExpr,
                   (Ctx, CloneType(Node->getType()), Node->getValueKind(),
                    Node->getCastKind(), Clone(Node->getSubExpr()),
                    nullptr /*EP*/ CLAD_COMPAT_CLANG12_CastExpr_GetFPO(Node),
                    Node->getTypeInfoAsWritten(), Node->getLParenLoc(),
                    Node->getRParenLoc()))
DEFINE_CREATE_EXPR(
    CXXStaticCastExpr,
    (Ctx, CloneType(Node->getType()), Node->getValueKind(), Node->getCastKind(),
     Clone(Node->getSubExpr()), nullptr,
     Node->getTypeInfoAsWritten() /*EP*/ CLAD_COMPAT_CLANG12_CastExpr_GetFPO(
         Node),
     Node->getOperatorLoc(), Node->getRParenLoc(), Node->getAngleBrackets()))
DEFINE_CREATE_EXPR(CXXDynamicCastExpr,
                   (Ctx, CloneType(Node->getType()), Node->getValueKind(),
                    Node->getCastKind(), Clone(Node->getSubExpr()), nullptr,
                    Node->getTypeInfoAsWritten(), Node->getOperatorLoc(),
                    Node->getRParenLoc(), Node->getAngleBrackets()))
DEFINE_CREATE_EXPR(CXXReinterpretCastExpr,
                   (Ctx, CloneType(Node->getType()), Node->getValueKind(),
                    Node->getCastKind(), Clone(Node->getSubExpr()), nullptr,
                    Node->getTypeInfoAsWritten(), Node->getOperatorLoc(),
                    Node->getRParenLoc(), Node->getAngleBrackets()))
DEFINE_CREATE_EXPR(CXXConstCastExpr,
                   (Ctx, CloneType(Node->getType()), Node->getValueKind(),
                    Clone(Node->getSubExpr()), Node->getTypeInfoAsWritten(),
                    Node->getOperatorLoc(), Node->getRParenLoc(),
                    Node->getAngleBrackets()))
DEFINE_CREATE_EXPR(
    CXXConstructExpr,
    (Ctx, CloneType(Node->getType()), Node->getLocation(),
     Node->getConstructor(), Node->isElidable(),
     clad_compat::makeArrayRef(Node->getArgs(), Node->getNumArgs()),
     Node->hadMultipleCandidates(), Node->isListInitialization(),
     Node->isStdInitListInitialization(), Node->requiresZeroInitialization(),
     Node->getConstructionKind(), Node->getParenOrBraceRange()))
DEFINE_CREATE_EXPR(CXXFunctionalCastExpr,
                   (Ctx, CloneType(Node->getType()), Node->getValueKind(),
                    Node->getTypeInfoAsWritten(), Node->getCastKind(),
                    Clone(Node->getSubExpr()),
                    nullptr /*EP*/ CLAD_COMPAT_CLANG12_CastExpr_GetFPO(Node),
                    Node->getLParenLoc(), Node->getRParenLoc()))
DEFINE_CREATE_EXPR(ExprWithCleanups, (Ctx, Node->getSubExpr(),
                                      Node->cleanupsHaveSideEffects(), {}))

DEFINE_CREATE_EXPR(ConstantExpr,
                   (Ctx, Clone(Node->getSubExpr())
                             CLAD_COMPAT_ConstantExpr_Create_ExtraParams))

DEFINE_CLONE_EXPR_CO(
    CXXTemporaryObjectExpr,
    (Ctx, Node->getConstructor(), CloneType(Node->getType()),
     Node->getTypeSourceInfo(),
     clad_compat::makeArrayRef(Node->getArgs(), Node->getNumArgs()),
     Node->getSourceRange(), Node->hadMultipleCandidates(),
     Node->isListInitialization(), Node->isStdInitListInitialization(),
     Node->requiresZeroInitialization()))

DEFINE_CLONE_EXPR(MaterializeTemporaryExpr,
                  (CloneType(Node->getType()),
                   CLAD_COMPAT_CLANG10_GetTemporaryExpr(Node),
                   Node->isBoundToLvalueReference()))
DEFINE_CLONE_EXPR_CO11(
    CompoundAssignOperator,
    (CLAD_COMPAT_CLANG11_Ctx_ExtraParams Clone(Node->getLHS()),
     Clone(Node->getRHS()), Node->getOpcode(), CloneType(Node->getType()),
     Node->getValueKind(), Node->getObjectKind(),
     CLAD_COMPAT_CLANG11_CompoundAssignOperator_EtraParams_Removed Node
         ->getOperatorLoc(),
     Node->getFPFeatures(CLAD_COMPAT_CLANG11_LangOptions_EtraParams)
         CLAD_COMPAT_CLANG11_CompoundAssignOperator_EtraParams_Moved))
DEFINE_CLONE_EXPR(ConditionalOperator,
                  (Clone(Node->getCond()), Node->getQuestionLoc(),
                   Clone(Node->getLHS()), Node->getColonLoc(),
                   Clone(Node->getRHS()), CloneType(Node->getType()),
                   Node->getValueKind(), Node->getObjectKind()))
DEFINE_CLONE_EXPR(AddrLabelExpr, (Node->getAmpAmpLoc(), Node->getLabelLoc(),
                                  Node->getLabel(), CloneType(Node->getType())))
DEFINE_CLONE_EXPR(StmtExpr,
                  (Clone(Node->getSubStmt()), CloneType(Node->getType()),
                   Node->getLParenLoc(),
                   Node->getRParenLoc()
                       CLAD_COMPAT_CLANG10_StmtExpr_Create_ExtraParams))
DEFINE_CLONE_EXPR(ChooseExpr,
                  (Node->getBuiltinLoc(), Clone(Node->getCond()),
                   Clone(Node->getLHS()), Clone(Node->getRHS()),
                   CloneType(Node->getType()), Node->getValueKind(),
                   Node->getObjectKind(), Node->getRParenLoc(),
                   Node->isConditionTrue()
                       CLAD_COMPAT_CLANG11_ChooseExpr_EtraParams_Removed))
DEFINE_CLONE_EXPR(GNUNullExpr,
                  (CloneType(Node->getType()), Node->getTokenLocation()))
DEFINE_CLONE_EXPR(VAArgExpr,
                  (Node->getBuiltinLoc(), Clone(Node->getSubExpr()),
                   Node->getWrittenTypeInfo(), Node->getRParenLoc(),
                   CloneType(Node->getType()), Node->isMicrosoftABI()))
DEFINE_CLONE_EXPR(ImplicitValueInitExpr, (CloneType(Node->getType())))
DEFINE_CLONE_EXPR(CXXScalarValueInitExpr,
                  (CloneType(Node->getType()), Node->getTypeSourceInfo(),
                   Node->getRParenLoc()))
DEFINE_CLONE_EXPR(ExtVectorElementExpr, (Node->getType(), Node->getValueKind(), Clone(Node->getBase()), Node->getAccessor(), Node->getAccessorLoc()))
DEFINE_CLONE_EXPR(CXXBoolLiteralExpr, (Node->getValue(), Node->getType(), Node->getSourceRange().getBegin()))
DEFINE_CLONE_EXPR(CXXNullPtrLiteralExpr, (Node->getType(), Node->getSourceRange().getBegin()))

CLAD_COMPAT_CLANG17_CXXThisExpr_ExtraParam Node->getSourceRange().getBegin(), Node->getType(), Node->isImplicit())) 

DEFINE_CLONE_EXPR(CXXThrowExpr, (Clone(Node->getSubExpr()), Node->getType(), Node->getThrowLoc(), Node->isThrownVariableInScope()))
#if CLANG_VERSION_MAJOR < 16
DEFINE_CLONE_EXPR(
    SubstNonTypeTemplateParmExpr,
    (CloneType(Node->getType()), Node->getValueKind(), Node->getBeginLoc(),
     Node->getParameter(),
     CLAD_COMPAT_SubstNonTypeTemplateParmExpr_isReferenceParameter_ExtraParam(
         Node) Node->getReplacement()))
#else
DEFINE_CLONE_EXPR(SubstNonTypeTemplateParmExpr,
                  (CloneType(Node->getType()), Node->getValueKind(),
                   Node->getBeginLoc(), Node->getReplacement(),
                   Node->getAssociatedDecl(), Node->getIndex(),
                   Node->getPackIndex(), Node->isReferenceParameter()));
#endif
DEFINE_CREATE_EXPR(PseudoObjectExpr, (Ctx, Node->getSyntacticForm(), llvm::SmallVector<Expr*, 4>(Node->semantics_begin(), Node->semantics_end()), Node->getResultExprIndex()))
// NOLINTEND(modernize-use-auto)
// BlockExpr
// BlockDeclRefExpr

Stmt* StmtClone::VisitStringLiteral(StringLiteral* Node) {
  llvm::SmallVector<SourceLocation, 4> concatLocations(Node->tokloc_begin(),
                                                       Node->tokloc_end());
  return StringLiteral::Create(Ctx, Node->getString(), Node->getKind(),
                               Node->isPascal(), CloneType(Node->getType()),
                               &concatLocations[0], concatLocations.size());
}

Stmt* StmtClone::VisitFloatingLiteral(FloatingLiteral* Node) {
  FloatingLiteral* clone =
      FloatingLiteral::Create(Ctx, Node->getValue(), Node->isExact(),
                              CloneType(Node->getType()), Node->getLocation());
  clone->setSemantics(Node->getSemantics());
  return clone;
}

Stmt* StmtClone::VisitInitListExpr(InitListExpr* Node) {
  llvm::SmallVector<Expr*, 8> initExprs(Node->getNumInits());
  for (unsigned i = 0, e = Node->getNumInits(); i < e; ++i)
    initExprs[i] = Clone(Node->getInit(i));

  SourceLocation lBrace = Node->getLBraceLoc();
  SourceLocation rBrace = Node->getRBraceLoc();

  InitListExpr* result = llvm::cast<InitListExpr>(m_Sema.ActOnInitList(lBrace, initExprs, rBrace).get());

  result->setInitializedFieldInUnion(Node->getInitializedFieldInUnion());
  // FIXME: clone the syntactic form, can this become recursive?
  return result;
}

//---------------------------------------------------------
Stmt* StmtClone::VisitDesignatedInitExpr(DesignatedInitExpr* Node) {
  llvm::SmallVector<Expr*, 8> indexExprs(Node->getNumSubExprs());
  for (int i = 0, e = indexExprs.size(); i < e; ++i)
    indexExprs[i] = Clone(Node->getSubExpr(i));

  // no &indexExprs[1]
  llvm::ArrayRef<Expr*> indexExprsRef =
      clad_compat::makeArrayRef(&indexExprs[0] + 1, indexExprs.size() - 1);

  return DesignatedInitExpr::Create(Ctx, Node->designators(),
                                    indexExprsRef,
                                    Node->getEqualOrColonLoc(),
                                    Node->usesGNUSyntax(), Node->getInit());
}

Stmt* StmtClone::VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr* Node) {
  if (Node->isArgumentType())
    return new (Ctx)
        UnaryExprOrTypeTraitExpr(Node->getKind(), Node->getArgumentTypeInfo(),
                                 CloneType(Node->getType()),
                                 Node->getOperatorLoc(), Node->getRParenLoc());
  return new (Ctx) UnaryExprOrTypeTraitExpr(
      Node->getKind(), Clone(Node->getArgumentExpr()),
      CloneType(Node->getType()), Node->getOperatorLoc(), Node->getRParenLoc());
}

Stmt* StmtClone::VisitCallExpr(CallExpr* Node) {
  CallExpr* result = clad_compat::CallExpr_Create(
      Ctx, Clone(Node->getCallee()), llvm::ArrayRef<Expr*>(),
      CloneType(Node->getType()), Node->getValueKind(),
      Node->getRParenLoc() CLAD_COMPAT_CLANG8_CallExpr_ExtraParams);
  result->setNumArgsUnsafe(Node->getNumArgs());
  for (unsigned i = 0, e = Node->getNumArgs(); i < e; ++i)
    result->setArg(i, Clone(Node->getArg(i)));

  // Copy Value and Type dependent
  clad_compat::ExprSetDeps(result, Node);

  return result;
}

Stmt* StmtClone::VisitUnresolvedLookupExpr(UnresolvedLookupExpr* Node) {
  TemplateArgumentListInfo TemplateArgs;
  if (Node->hasExplicitTemplateArgs())
    Node->copyTemplateArgumentsInto(TemplateArgs);
  Stmt* result = clad_compat::UnresolvedLookupExpr_Create(
      Ctx, Node->getNamingClass(), Node->getQualifierLoc(),
      Node->getTemplateKeywordLoc(), Node->getNameInfo(), Node->requiresADL(),
      // They get copied again by
      // OverloadExpr, so we are safe.
      &TemplateArgs, Node->decls_begin(), Node->decls_end());
  return result;
}

Stmt* StmtClone::VisitCXXOperatorCallExpr(CXXOperatorCallExpr* Node) {
  llvm::SmallVector<Expr*, 4> clonedArgs;
  for (Expr* arg : Node->arguments()) {
    clonedArgs.push_back(Clone(arg));
  }
  CallExpr::ADLCallKind UsesADL = CallExpr::NotADL;
  CXXOperatorCallExpr* result = CXXOperatorCallExpr::Create(
      Ctx, Node->getOperator(), Clone(Node->getCallee()), clonedArgs,
      CloneType(Node->getType()), Node->getValueKind(), Node->getRParenLoc(),
      Node->getFPFeatures()
          CLAD_COMPAT_CLANG11_CXXOperatorCallExpr_Create_ExtraParamsUse);

  //###  result->setNumArgs(Ctx, Node->getNumArgs());
  result->setNumArgsUnsafe(Node->getNumArgs());
  for (unsigned i = 0, e = Node->getNumArgs(); i < e; ++i)
    result->setArg(i, Clone(Node->getArg(i)));

  // Copy Value and Type dependent
  clad_compat::ExprSetDeps(result, Node);

  return result;
}

Stmt* StmtClone::VisitCXXMemberCallExpr(CXXMemberCallExpr * Node) {
  CXXMemberCallExpr* result = clad_compat::CXXMemberCallExpr_Create(
      Ctx, Clone(Node->getCallee()), {}, CloneType(Node->getType()),
      Node->getValueKind(),
      Node->getRParenLoc()
      /*FP*/ CLAD_COMPAT_CLANG12_CastExpr_GetFPO(Node));
  // ###  result->setNumArgs(Ctx, Node->getNumArgs());
  result->setNumArgsUnsafe(Node->getNumArgs());

  for (unsigned i = 0, e = Node->getNumArgs(); i < e; ++i)
    result->setArg(i, Clone(Node->getArg(i)));

  // Copy Value and Type dependent
  clad_compat::ExprSetDeps(result, Node);

  return result;
}

Stmt* StmtClone::VisitShuffleVectorExpr(ShuffleVectorExpr* Node) {
  llvm::SmallVector<Expr*, 8> cloned(std::max(1u, Node->getNumSubExprs()));
  for (unsigned i = 0, e = Node->getNumSubExprs(); i < e; ++i)
    cloned[i] = Clone(Node->getExpr(i));
  llvm::ArrayRef<Expr*> clonedRef =
      clad_compat::makeArrayRef(cloned.data(), cloned.size());
  return new (Ctx)
      ShuffleVectorExpr(Ctx, clonedRef, CloneType(Node->getType()),
                        Node->getBuiltinLoc(), Node->getRParenLoc());
}

Stmt* StmtClone::VisitCaseStmt(CaseStmt* Node) {
  CaseStmt* result = CaseStmt::Create(
      Ctx, Clone(Node->getLHS()), Clone(Node->getRHS()), Node->getCaseLoc(),
      Node->getEllipsisLoc(), Node->getColonLoc());
  result->setSubStmt(Clone(Node->getSubStmt()));
  return result;
}

Stmt* StmtClone::VisitSwitchStmt(SwitchStmt* Node) {
  SourceLocation noLoc;
  SwitchStmt* result
    = clad_compat::SwitchStmt_Create(Ctx,
        Node->getInit(), Node->getConditionVariable(), Node->getCond(),
        noLoc, noLoc);
  result->setBody(Clone(Node->getBody()));
  result->setSwitchLoc(Node->getSwitchLoc());
  return result;
}

DEFINE_CLONE_STMT_CO(ReturnStmt,
                     (Ctx, Node->getReturnLoc(), Clone(Node->getRetValue()), 0))
DEFINE_CLONE_STMT(DefaultStmt, (Node->getDefaultLoc(), Node->getColonLoc(), Clone(Node->getSubStmt())))
DEFINE_CLONE_STMT(GotoStmt, (Node->getLabel(), Node->getGotoLoc(), Node->getLabelLoc()))
DEFINE_CLONE_STMT_CO(WhileStmt, (Ctx, CloneDeclOrNull(Node->getConditionVariable()), Clone(Node->getCond()), Clone(Node->getBody()), Node->getWhileLoc() CLAD_COMPAT_CLANG11_WhileStmt_ExtraParams))
DEFINE_CLONE_STMT(DoStmt, (Clone(Node->getBody()), Clone(Node->getCond()), Node->getDoLoc(), Node->getWhileLoc(), Node->getRParenLoc()))
DEFINE_CLONE_STMT_CO(IfStmt, (Ctx, Node->getIfLoc(),CLAD_COMPAT_IfStmt_Create_IfStmtKind_Param(Node), Node->getInit(), CloneDeclOrNull(Node->getConditionVariable()), Clone(Node->getCond()) /*EPs*/CLAD_COMPAT_CLANG12_LR_ExtraParams(Node), Clone(Node->getThen()), Node->getElseLoc(), Clone(Node->getElse())))
DEFINE_CLONE_STMT(LabelStmt, (Node->getIdentLoc(), Node->getDecl(), Clone(Node->getSubStmt())))
DEFINE_CLONE_STMT(NullStmt, (Node->getSemiLoc()))
DEFINE_CLONE_STMT(ForStmt, (Ctx, Clone(Node->getInit()), Clone(Node->getCond()), CloneDeclOrNull(Node->getConditionVariable()), Clone(Node->getInc()), Clone(Node->getBody()),
                            Node->getForLoc(), Node->getLParenLoc(), Node->getRParenLoc()))
DEFINE_CLONE_STMT(ContinueStmt, (Node->getContinueLoc()))
DEFINE_CLONE_STMT(BreakStmt, (Node->getBreakLoc()))
DEFINE_CLONE_STMT(CXXCatchStmt, (Node->getCatchLoc(),
                                 CloneDeclOrNull(Node->getExceptionDecl()),
                                 Clone(Node->getHandlerBlock())))

#if CLANG_VERSION_MAJOR > 8
DEFINE_CLONE_STMT(ValueStmt, (Node->getStmtClass()))
#endif

Stmt* StmtClone::VisitCXXTryStmt(CXXTryStmt* Node) {
  llvm::SmallVector<Stmt*, 4> CatchStmts(std::max(1u, Node->getNumHandlers()));
  for (unsigned i = 0, e = Node->getNumHandlers(); i < e; ++i)
  {
    CatchStmts[i] = Clone(Node->getHandler(i));
  }
  llvm::ArrayRef<Stmt*> handlers =
      clad_compat::makeArrayRef(CatchStmts.data(), CatchStmts.size());
  return CXXTryStmt::Create(Ctx, Node->getTryLoc(), Clone(Node->getTryBlock()),
                            handlers);
}

Stmt* StmtClone::VisitCompoundStmt(CompoundStmt *Node) {
  llvm::SmallVector<Stmt*, 8> clonedBody;
  for (CompoundStmt::const_body_iterator i = Node->body_begin(),
         e = Node->body_end(); i != e; ++i)
    clonedBody.push_back(Clone(*i));

  llvm::ArrayRef<Stmt*> stmtsRef =
      clad_compat::makeArrayRef(clonedBody.data(), clonedBody.size());
  return clad_compat::CompoundStmt_Create(Ctx, stmtsRef /**/ CLAD_COMPAT_CLANG15_CompoundStmt_Create_ExtraParam1(Node),
                              Node->getLBracLoc(), Node->getLBracLoc());
}

VarDecl* StmtClone::CloneDeclOrNull(VarDecl* Node)  {
  if (!Node)
    return 0;
  return cast_or_null<VarDecl>(CloneDecl(Node));
}

Decl* StmtClone::CloneDecl(Decl* Node)  {
  // we support only exactly this class, so no visitor is needed (yet?)
  if (Node->getKind() == Decl::Var) {
    VarDecl* VD = static_cast<VarDecl*>(Node);

    VarDecl* cloned_Decl = VarDecl::Create(
        Ctx, VD->getDeclContext(), VD->getLocation(), VD->getInnerLocStart(),
        VD->getIdentifier(), CloneType(VD->getType()), VD->getTypeSourceInfo(),
        VD->getStorageClass());
    if (VD->getInit())
      m_Sema.AddInitializerToDecl(cloned_Decl, Clone(VD->getInit()), VD->isDirectInit());
    cloned_Decl->setTSCSpec(VD->getTSCSpec());
    //cloned_Decl->setDeclaredInCondition(VD->isDeclaredInCondition());
    if (m_OriginalToClonedStmts != 0)
      m_OriginalToClonedStmts->m_DeclMapping[VD] = cloned_Decl;

    return cloned_Decl;
  }
  assert(0 && "other decl clones aren't supported");
  return 0;
}

Stmt* StmtClone::VisitDeclStmt(DeclStmt* Node) {
  DeclGroupRef clonedDecls;
  if (Node->isSingleDecl())
    clonedDecls = DeclGroupRef(CloneDecl(Node->getSingleDecl()));
  else if (Node->getDeclGroup().isDeclGroup()) {
    llvm::SmallVector<Decl*, 8> clonedDeclGroup;
    const DeclGroupRef& dg = Node->getDeclGroup();
    for (DeclGroupRef::const_iterator i = dg.begin(), e = dg.end(); i != e; ++i)
      clonedDeclGroup.push_back(CloneDecl(*i));

    clonedDecls = DeclGroupRef(DeclGroup::Create(Ctx, clonedDeclGroup.data(),
                                                 clonedDeclGroup.size()));
  }
  return new (Ctx) DeclStmt(clonedDecls, Node->getBeginLoc(), Node->getEndLoc());
}

Stmt* StmtClone::VisitStmt(Stmt*) {
  assert(0 && "clone not fully implemented");
  return 0;
}

ReferencesUpdater::ReferencesUpdater(
    Sema& SemaRef, Scope* S, const FunctionDecl* FD,
    const std::unordered_map<const clang::VarDecl*, clang::VarDecl*>&
        DeclReplacements)
    : m_Sema(SemaRef), m_CurScope(S), m_Function(FD),
      m_DeclReplacements(DeclReplacements) {}

bool ReferencesUpdater::VisitDeclRefExpr(DeclRefExpr* DRE) {
  // We should only update references of the declarations that were inside
  // the original function declaration context.
  // Original function = function that we are currently differentiating.
  if (!DRE->getDecl()->getDeclContext()->Encloses(m_Function))
    return true;

  // Replace the declaration if it is present in `m_DeclReplacements`.
  if (VarDecl* VD = dyn_cast<VarDecl>(DRE->getDecl())) {
    auto it = m_DeclReplacements.find(VD);
    if (it != std::end(m_DeclReplacements)) {
      DRE->setDecl(it->second);
      QualType NonRefQT = it->second->getType().getNonReferenceType();
      if (NonRefQT != DRE->getType())
        DRE->setType(NonRefQT);
    }
  }

  DeclarationNameInfo DNI = DRE->getNameInfo();

  LookupResult R(m_Sema, DNI, Sema::LookupOrdinaryName);
  m_Sema.LookupName(R, m_CurScope, /*allowBuiltinCreation*/ false);

  if (R.empty())
    return true;  // Nothing to update.

  // FIXME: Handle the case when there are overloads found. Update
  // it with the best match.
  //
  // FIXME: This is the right way to go in principe, however there is no
  // properly built decl context.
  // m_Sema.MarkDeclRefReferenced(clonedDRE);
  if (!R.isSingleResult())
    return true;

  if (ValueDecl* VD = dyn_cast<ValueDecl>(R.getFoundDecl())) {
    DRE->setDecl(VD);
    VD->setReferenced();
    VD->setIsUsed();
  }
  updateType(DRE->getType());
  return true;
}

bool ReferencesUpdater::VisitStmt(clang::Stmt* S) {
  if (auto* E = dyn_cast<Expr>(S))
    updateType(E->getType());
  return true;
}

void ReferencesUpdater::updateType(QualType QT) {
  if (const auto* varArrType = dyn_cast<VariableArrayType>(QT))
    TraverseStmt(varArrType->getSizeExpr());
}

QualType StmtClone::CloneType(const clang::QualType T) {
  if (const auto* varArrType =
          dyn_cast<clang::VariableArrayType>(T.getTypePtr())) {
    auto elemType = varArrType->getElementType();
    return Ctx.getVariableArrayType(elemType, Clone(varArrType->getSizeExpr()),
                                    varArrType->getSizeModifier(),
                                    T.getQualifiers().getAsOpaqueValue(),
                                    SourceRange());
  }

  return clang::QualType(T.getTypePtr(), T.getQualifiers().getAsOpaqueValue());
}

//---------------------------------------------------------
  } // end namespace utils
} // end namespace clad
