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

#define DEFINE_CLONE_STMT_CO(CLASS, CTORARGS) \
Stmt* StmtClone::Visit ## CLASS(CLASS *Node)  \
{                                             \
  return CLAD_COMPAT_CREATE(CLASS, CTORARGS); \
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

#define DEFINE_CLONE_EXPR_CO(CLASS, CTORARGS)           \
Stmt* StmtClone::Visit ## CLASS(CLASS *Node)            \
{                                                       \
  CLASS* result = CLAD_COMPAT_CREATE(CLASS, CTORARGS);  \
  clad_compat::ExprSetDeps(result, Node);               \
  return result;                                        \
}

#define DEFINE_CLONE_EXPR_CO11(CLASS, CTORARGS)         \
Stmt* StmtClone::Visit ## CLASS(CLASS *Node)            \
{                                                       \
  CLASS* result = CLAD_COMPAT_CREATE11(CLASS, CTORARGS);\
  clad_compat::ExprSetDeps(result, Node);               \
  return result;                                        \
}

DEFINE_CLONE_EXPR_CO11(BinaryOperator, (CLAD_COMPAT_CLANG11_Ctx_ExtraParams Clone(Node->getLHS()), Clone(Node->getRHS()), Node->getOpcode(), Node->getType(), Node->getValueKind(), Node->getObjectKind(), Node->getOperatorLoc(), Node->getFPFeatures(CLAD_COMPAT_CLANG11_LangOptions_EtraParams)))
DEFINE_CLONE_EXPR_CO11(UnaryOperator, (CLAD_COMPAT_CLANG11_Ctx_ExtraParams Clone(Node->getSubExpr()), Node->getOpcode(), Node->getType(), Node->getValueKind(), Node->getObjectKind(), Node->getOperatorLoc() CLAD_COMPAT_CLANG7_UnaryOperator_ExtraParams CLAD_COMPAT_CLANG11_UnaryOperator_ExtraParams))
Stmt* StmtClone::VisitDeclRefExpr(DeclRefExpr *Node) {
  TemplateArgumentListInfo TAListInfo;
  Node->copyTemplateArgumentsInto(TAListInfo);
  return DeclRefExpr::Create(Ctx, Node->getQualifierLoc(), Node->getTemplateKeywordLoc(), Node->getDecl(), Node->refersToEnclosingVariableOrCapture(), Node->getNameInfo(), Node->getType(), Node->getValueKind(), Node->getFoundDecl(), &TAListInfo);
}
DEFINE_CREATE_EXPR(IntegerLiteral, (Ctx, Node->getValue(), Node->getType(), Node->getLocation()))
DEFINE_CLONE_EXPR_CO(PredefinedExpr, (CLAD_COMPAT_CLANG8_Ctx_ExtraParams Node->getLocation(), Node->getType(), Node->getIdentKind(), Node->getFunctionName()))
DEFINE_CLONE_EXPR(CharacterLiteral, (Node->getValue(), Node->getKind(), Node->getType(), Node->getLocation()))
DEFINE_CLONE_EXPR(ImaginaryLiteral, (Clone(Node->getSubExpr()), Node->getType()))
DEFINE_CLONE_EXPR(ParenExpr, (Node->getLParen(), Node->getRParen(), Clone(Node->getSubExpr())))
DEFINE_CLONE_EXPR(ArraySubscriptExpr, (Clone(Node->getLHS()), Clone(Node->getRHS()), Node->getType(), Node->getValueKind(), Node->getObjectKind(), Node->getRBracketLoc()))
Stmt* StmtClone::VisitMemberExpr(MemberExpr* Node) {
  TemplateArgumentListInfo TemplateArgs;
  if (Node->hasExplicitTemplateArgs())
    Node->copyTemplateArgumentsInto(TemplateArgs);
  MemberExpr* result = MemberExpr::Create(Ctx,
                                  Clone(Node->getBase()),
                                  Node->isArrow(),
                                  Node->getOperatorLoc(),
                                  Node->getQualifierLoc(),
                                  Node->getTemplateKeywordLoc(),
                                  Node->getMemberDecl(),
                                  Node->getFoundDecl(),
                                  Node->getMemberNameInfo(),
                                  &TemplateArgs,
                                  Node->getType(),
                                  Node->getValueKind(),
                                  Node->getObjectKind()
                                  CLAD_COMPAT_CLANG9_MemberExpr_ExtraParams(
                                      Node->isNonOdrUse()));
  // Copy Value and Type dependent
  clad_compat::ExprSetDeps(result, Node);
  return result;
}
DEFINE_CLONE_EXPR(CompoundLiteralExpr, (Node->getLParenLoc(), Node->getTypeSourceInfo(), Node->getType(), Node->getValueKind(), Clone(Node->getInitializer()), Node->isFileScope()))
DEFINE_CREATE_EXPR(ImplicitCastExpr, (Ctx, Node->getType(), Node->getCastKind(), Clone(Node->getSubExpr()), 0, Node->getValueKind() /*EP*/CLAD_COMPAT_CLANG12_CastExpr_GetFPO(Node) ))
DEFINE_CREATE_EXPR(CStyleCastExpr, (Ctx, Node->getType(), Node->getValueKind(), Node->getCastKind(), Clone(Node->getSubExpr()), 0 /*EP*/CLAD_COMPAT_CLANG12_CastExpr_GetFPO(Node), Node->getTypeInfoAsWritten(), Node->getLParenLoc(), Node->getRParenLoc()))
DEFINE_CREATE_EXPR(CXXStaticCastExpr, (Ctx, Node->getType(), Node->getValueKind(), Node->getCastKind(), Clone(Node->getSubExpr()), 0, Node->getTypeInfoAsWritten() /*EP*/CLAD_COMPAT_CLANG12_CastExpr_GetFPO(Node), Node->getOperatorLoc(), Node->getRParenLoc(), Node->getAngleBrackets()))
DEFINE_CREATE_EXPR(CXXDynamicCastExpr, (Ctx, Node->getType(), Node->getValueKind(), Node->getCastKind(), Clone(Node->getSubExpr()), 0, Node->getTypeInfoAsWritten(), Node->getOperatorLoc(), Node->getRParenLoc(), Node->getAngleBrackets()))
DEFINE_CREATE_EXPR(CXXReinterpretCastExpr, (Ctx, Node->getType(), Node->getValueKind(), Node->getCastKind(), Clone(Node->getSubExpr()), 0, Node->getTypeInfoAsWritten(), Node->getOperatorLoc(), Node->getRParenLoc(), Node->getAngleBrackets()))
DEFINE_CREATE_EXPR(CXXConstCastExpr, (Ctx, Node->getType(), Node->getValueKind(), Clone(Node->getSubExpr()), Node->getTypeInfoAsWritten(), Node->getOperatorLoc(), Node->getRParenLoc(), Node->getAngleBrackets()))
DEFINE_CREATE_EXPR(CXXConstructExpr, (Ctx, Node->getType(), Node->getLocation(), Node->getConstructor(), Node->isElidable(), llvm::makeArrayRef(Node->getArgs(), Node->getNumArgs()), Node->hadMultipleCandidates(), Node->isListInitialization(), Node->isStdInitListInitialization(), Node->requiresZeroInitialization(), Node->getConstructionKind(), Node->getParenOrBraceRange()))
DEFINE_CREATE_EXPR(CXXFunctionalCastExpr, (Ctx, Node->getType(), Node->getValueKind(), Node->getTypeInfoAsWritten(), Node->getCastKind(), Clone(Node->getSubExpr()), 0 /*EP*/CLAD_COMPAT_CLANG12_CastExpr_GetFPO(Node), Node->getLParenLoc(), Node->getRParenLoc()))
DEFINE_CREATE_EXPR(ExprWithCleanups, (Ctx, Node->getSubExpr(),
                                      Node->cleanupsHaveSideEffects(), {}))
// clang <= 7 do not have `ConstantExpr` node.
#if CLANG_VERSION_MAJOR > 7
DEFINE_CREATE_EXPR(ConstantExpr, (Ctx, Clone(Node->getSubExpr()) CLAD_COMPAT_ConstantExpr_Create_ExtraParams));
#endif

DEFINE_CLONE_EXPR_CO(CXXTemporaryObjectExpr, (Ctx, Node->getConstructor(), Node->getType(), Node->getTypeSourceInfo(), llvm::makeArrayRef(Node->getArgs(), Node->getNumArgs()), Node->getSourceRange(), Node->hadMultipleCandidates(), Node->isListInitialization(), Node->isStdInitListInitialization(), Node->requiresZeroInitialization()))

DEFINE_CLONE_EXPR(MaterializeTemporaryExpr, (Node->getType(), CLAD_COMPAT_CLANG10_GetTemporaryExpr(Node), Node->isBoundToLvalueReference()))
DEFINE_CLONE_EXPR_CO11(CompoundAssignOperator, (CLAD_COMPAT_CLANG11_Ctx_ExtraParams Clone(Node->getLHS()), Clone(Node->getRHS()), Node->getOpcode(), Node->getType(),
                                           Node->getValueKind(), Node->getObjectKind(), CLAD_COMPAT_CLANG11_CompoundAssignOperator_EtraParams_Removed  Node->getOperatorLoc(), Node->getFPFeatures(CLAD_COMPAT_CLANG11_LangOptions_EtraParams) CLAD_COMPAT_CLANG11_CompoundAssignOperator_EtraParams_Moved))
DEFINE_CLONE_EXPR(ConditionalOperator, (Clone(Node->getCond()), Node->getQuestionLoc(), Clone(Node->getLHS()), Node->getColonLoc(), Clone(Node->getRHS()), Node->getType(), Node->getValueKind(), Node->getObjectKind()))
DEFINE_CLONE_EXPR(AddrLabelExpr, (Node->getAmpAmpLoc(), Node->getLabelLoc(), Node->getLabel(), Node->getType()))
DEFINE_CLONE_EXPR(StmtExpr, (Clone(Node->getSubStmt()), Node->getType(), Node->getLParenLoc(), Node->getRParenLoc() CLAD_COMPAT_CLANG10_StmtExpr_Create_ExtraParams ))
DEFINE_CLONE_EXPR(ChooseExpr, (Node->getBuiltinLoc(), Clone(Node->getCond()), Clone(Node->getLHS()), Clone(Node->getRHS()), Node->getType(), Node->getValueKind(), Node->getObjectKind(), Node->getRParenLoc(), Node->isConditionTrue() CLAD_COMPAT_CLANG11_ChooseExpr_EtraParams_Removed))
DEFINE_CLONE_EXPR(GNUNullExpr, (Node->getType(), Node->getTokenLocation()))
DEFINE_CLONE_EXPR(VAArgExpr, (Node->getBuiltinLoc(), Clone(Node->getSubExpr()), Node->getWrittenTypeInfo(), Node->getRParenLoc(), Node->getType(), Node->isMicrosoftABI()))
DEFINE_CLONE_EXPR(ImplicitValueInitExpr, (Node->getType()))
DEFINE_CLONE_EXPR(ExtVectorElementExpr, (Node->getType(), Node->getValueKind(), Clone(Node->getBase()), Node->getAccessor(), Node->getAccessorLoc()))
DEFINE_CLONE_EXPR(CXXBoolLiteralExpr, (Node->getValue(), Node->getType(), Node->getSourceRange().getBegin()))
DEFINE_CLONE_EXPR(CXXNullPtrLiteralExpr, (Node->getType(), Node->getSourceRange().getBegin()))
DEFINE_CLONE_EXPR(CXXThisExpr, (Node->getSourceRange().getBegin(), Node->getType(), Node->isImplicit()))
DEFINE_CLONE_EXPR(CXXThrowExpr, (Clone(Node->getSubExpr()), Node->getType(), Node->getThrowLoc(), Node->isThrownVariableInScope()))
//BlockExpr
//BlockDeclRefExpr

Stmt* StmtClone::VisitStringLiteral(StringLiteral* Node) {
  llvm::SmallVector<SourceLocation, 4> concatLocations(Node->tokloc_begin(),
                                                       Node->tokloc_end());
  return StringLiteral::Create(Ctx, Node->getString(), Node->getKind(),
                               Node->isPascal(), Node->getType(),
                               &concatLocations[0], concatLocations.size());
}

Stmt* StmtClone::VisitFloatingLiteral(FloatingLiteral* Node) {
  FloatingLiteral* clone = FloatingLiteral::Create(Ctx, Node->getValue(),
                                                   Node->isExact(),
                                                   Node->getType(),
                                                   Node->getLocation());
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
  llvm::ArrayRef<Expr*> indexExprsRef
    = llvm::makeArrayRef(&indexExprs[0] + 1, indexExprs.size() - 1);

  return DesignatedInitExpr::Create(Ctx, Node->designators(),
                                    indexExprsRef,
                                    Node->getEqualOrColonLoc(),
                                    Node->usesGNUSyntax(), Node->getInit());
}

Stmt* StmtClone::VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr* Node) {
  if (Node->isArgumentType())
    return new (Ctx) UnaryExprOrTypeTraitExpr(Node->getKind(),
                                              Node->getArgumentTypeInfo(),
                                              Node->getType(),
                                              Node->getOperatorLoc(),
                                              Node->getRParenLoc());
  return new (Ctx) UnaryExprOrTypeTraitExpr(Node->getKind(),
                                            Clone(Node->getArgumentExpr()),
                                            Node->getType(),
                                            Node->getOperatorLoc(),
                                            Node->getRParenLoc());
}

Stmt* StmtClone::VisitCallExpr(CallExpr* Node) {
  CallExpr* result = clad_compat::CallExpr_Create(Ctx, Clone(Node->getCallee()),
                                        llvm::ArrayRef<Expr*>(),
                                        Node->getType(),
                                        Node->getValueKind(),
                                        Node->getRParenLoc()
                                        CLAD_COMPAT_CLANG8_CallExpr_ExtraParams);
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
  Stmt* result = UnresolvedLookupExpr::Create(Ctx,
                                              Node->getNamingClass(),
                                              Node->getQualifierLoc(),
                                              Node->getTemplateKeywordLoc(),
                                              Node->getNameInfo(),
                                              Node->requiresADL(),
                                              // They get copied again by
                                              // OverloadExpr, so we are safe.
                                              &TemplateArgs,
                                              Node->decls_begin(),
                                              Node->decls_end()
                                              );
  return result;
}

Stmt* StmtClone::VisitCXXOperatorCallExpr(CXXOperatorCallExpr* Node) {
  CXXOperatorCallExpr* result
    = clad_compat::CXXOperatorCallExpr_Create(Ctx, Node->getOperator(),
                                    Clone(Node->getCallee()), 0,
                                    Node->getType(),
                                    Node->getValueKind(),
                                    Node->getRParenLoc(),
                                    Node->getFPFeatures()
                                    CLAD_COMPAT_CLANG11_CXXOperatorCallExpr_Create_ExtraParams
                                    );
//###  result->setNumArgs(Ctx, Node->getNumArgs());
  result->setNumArgsUnsafe(Node->getNumArgs());
  for (unsigned i = 0, e = Node->getNumArgs(); i < e; ++i)
    result->setArg(i, Clone(Node->getArg(i)));

  // Copy Value and Type dependent
  clad_compat::ExprSetDeps(result, Node);

  return result;
}

Stmt* StmtClone::VisitCXXMemberCallExpr(CXXMemberCallExpr * Node) {
  CXXMemberCallExpr* result
    = clad_compat::CXXMemberCallExpr_Create(Ctx, Clone(Node->getCallee()), 0,
                                  Node->getType(),
                                  Node->getValueKind(),
                                  Node->getRParenLoc()
                                  /*FP*/CLAD_COMPAT_CLANG12_CastExpr_GetFPO(Node)
                                  );
//###  result->setNumArgs(Ctx, Node->getNumArgs());
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
  llvm::ArrayRef<Expr*> clonedRef
    = llvm::makeArrayRef(cloned.data(), cloned.size());
  return new (Ctx) ShuffleVectorExpr(Ctx, clonedRef, Node->getType(),
                                     Node->getBuiltinLoc(),
                                     Node->getRParenLoc());
}

Stmt* StmtClone::VisitCaseStmt(CaseStmt* Node) {
  CaseStmt* result = clad_compat::CaseStmt_Create(Ctx, Clone(Node->getLHS()),
                                        Clone(Node->getRHS()),
                                        Node->getCaseLoc(),
                                        Node->getEllipsisLoc(),
                                        Node->getColonLoc());
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

DEFINE_CLONE_STMT_CO(ReturnStmt, (CLAD_COMPAT_CLANG8_Ctx_ExtraParams Node->getReturnLoc(), Clone(Node->getRetValue()), 0))
DEFINE_CLONE_STMT(DefaultStmt, (Node->getDefaultLoc(), Node->getColonLoc(), Clone(Node->getSubStmt())))
DEFINE_CLONE_STMT(GotoStmt, (Node->getLabel(), Node->getGotoLoc(), Node->getLabelLoc()))
DEFINE_CLONE_STMT_CO(WhileStmt, (Ctx, CloneDeclOrNull(Node->getConditionVariable()), Clone(Node->getCond()), Clone(Node->getBody()), Node->getWhileLoc() CLAD_COMPAT_CLANG11_WhileStmt_ExtraParams))
DEFINE_CLONE_STMT(DoStmt, (Clone(Node->getBody()), Clone(Node->getCond()), Node->getDoLoc(), Node->getWhileLoc(), Node->getRParenLoc()))
DEFINE_CLONE_STMT_CO(IfStmt, (Ctx, Node->getIfLoc(), Node->isConstexpr(), Node->getInit(), CloneDeclOrNull(Node->getConditionVariable()), Clone(Node->getCond()) /*EPs*/CLAD_COMPAT_CLANG12_LR_ExtraParams(Node), Clone(Node->getThen()), Node->getElseLoc(), Clone(Node->getElse())))
DEFINE_CLONE_STMT(LabelStmt, (Node->getIdentLoc(), Node->getDecl(), Clone(Node->getSubStmt())))
DEFINE_CLONE_STMT(NullStmt, (Node->getSemiLoc()))
DEFINE_CLONE_STMT(ForStmt, (Ctx, Clone(Node->getInit()), Clone(Node->getCond()), CloneDeclOrNull(Node->getConditionVariable()), Clone(Node->getInc()), Clone(Node->getBody()),
                            Node->getForLoc(), Node->getLParenLoc(), Node->getRParenLoc()))
DEFINE_CLONE_STMT(ContinueStmt, (Node->getContinueLoc()))
DEFINE_CLONE_STMT(BreakStmt, (Node->getBreakLoc()))
DEFINE_CLONE_STMT(CXXCatchStmt, (Node->getCatchLoc(), static_cast<VarDecl*>(CloneDecl(Node->getExceptionDecl())), Clone(Node->getHandlerBlock())))

Stmt* StmtClone::VisitCXXTryStmt(CXXTryStmt* Node) {
  llvm::SmallVector<Stmt*, 4> CatchStmts(std::max(1u, Node->getNumHandlers()));
  for (unsigned i = 0, e = Node->getNumHandlers(); i < e; ++i)
  {
    CatchStmts[i] = Clone(Node->getHandler(i));
  }
  llvm::ArrayRef<Stmt*> handlers = llvm::makeArrayRef(CatchStmts.data(),
                                                      CatchStmts.size());
  return CXXTryStmt::Create(Ctx, Node->getTryLoc(), Clone(Node->getTryBlock()),
                            handlers);
}

Stmt* StmtClone::VisitCompoundStmt(CompoundStmt *Node) {
  llvm::SmallVector<Stmt*, 8> clonedBody;
  for (CompoundStmt::const_body_iterator i = Node->body_begin(),
         e = Node->body_end(); i != e; ++i)
    clonedBody.push_back(Clone(*i));

  llvm::ArrayRef<Stmt*> stmtsRef = llvm::makeArrayRef(clonedBody.data(),
                                                      clonedBody.size());
  return clad_compat::CompoundStmt_Create(Ctx, stmtsRef, Node->getLBracLoc(),
                              Node->getLBracLoc());
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

    VarDecl* cloned_Decl = VarDecl::Create(Ctx, VD->getDeclContext(),
                                           VD->getLocation(),
                                           VD->getInnerLocStart(),
                                           VD->getIdentifier(), VD->getType(),
                                           VD->getTypeSourceInfo(),
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

ReferencesUpdater::ReferencesUpdater(Sema& SemaRef, utils::StmtClone* C,
                                     Scope* S, const FunctionDecl* FD)
    : m_Sema(SemaRef), m_NodeCloner(C), m_CurScope(S), m_Function(FD) {}

bool ReferencesUpdater::VisitDeclRefExpr(DeclRefExpr* DRE) {
  // If the declaration's decl context encloses the derivative's decl
  // context we must not update anything.
  //if (DRE->getDecl()->getDeclContext()->Encloses(m_Sema.CurContext)) {
  //  return true;
  //}

  // We should only update references of the declarations that were inside
  // the original function declaration context.
  // Original function = function that we are currently differentiating.
  if (!DRE->getDecl()->getDeclContext()->Encloses(m_Function))
    return true;
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
  return true;
}

//---------------------------------------------------------
  } // end namespace utils
} // end namespace clad
