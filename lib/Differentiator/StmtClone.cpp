//--------------------------------------------------------------------*- C++ -*-
// AutoDiff - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------
//
// File originates from the Scout project (http://scout.zih.tu-dresden.de/)

#include "autodiff/Differentiator/StmtClone.h"
#include "llvm/ADT/SmallVector.h"

using namespace clang;

namespace autodiff {
namespace utils {

#define DEFINE_CLONE_STMT(CLASS, CTORARGS)    \
Stmt* StmtClone::Visit ## CLASS(CLASS *Node)  \
{                                             \
  return new (Ctx) CLASS CTORARGS;            \
}                 

#define DEFINE_CLONE_EXPR(CLASS, CTORARGS)    \
Stmt* StmtClone::Visit ## CLASS(CLASS *Node)  \
{                                             \
  CLASS* result = new (Ctx) CLASS CTORARGS;   \
  result->setValueDependent(Node->isValueDependent());  \
  result->setTypeDependent(Node->isTypeDependent());    \
  return result;                              \
}                 

#define DEFINE_CREATE_EXPR(CLASS, CTORARGS)    \
Stmt* StmtClone::Visit ## CLASS(CLASS *Node)  \
{                                             \
  CLASS* result = CLASS::Create CTORARGS;   \
  result->setValueDependent(Node->isValueDependent());  \
  result->setTypeDependent(Node->isTypeDependent());    \
  return result;                              \
}                 

DEFINE_CLONE_EXPR(BinaryOperator, (Clone(Node->getLHS()), Clone(Node->getRHS()), Node->getOpcode(), Node->getType(), Node->getValueKind(), Node->getObjectKind(), Node->getOperatorLoc(), Node->isFPContractable()))
DEFINE_CLONE_EXPR(UnaryOperator, (Clone(Node->getSubExpr()), Node->getOpcode(), Node->getType(), Node->getValueKind(), Node->getObjectKind(), Node->getOperatorLoc()))
DEFINE_CLONE_EXPR(DeclRefExpr, (Node->getDecl(), Node->refersToEnclosingLocal(), Node->getType(), Node->getValueKind(), Node->getLocation()))
DEFINE_CREATE_EXPR(IntegerLiteral, (Ctx, Node->getValue(), Node->getType(), Node->getLocation()))
DEFINE_CLONE_EXPR(PredefinedExpr, (Node->getLocation(), Node->getType(), Node->getIdentType()))
DEFINE_CLONE_EXPR(CharacterLiteral, (Node->getValue(), Node->getKind(), Node->getType(), Node->getLocation()))
DEFINE_CREATE_EXPR(FloatingLiteral, (Ctx, Node->getValue(), Node->isExact(), Node->getType(), Node->getLocation()))
DEFINE_CLONE_EXPR(ImaginaryLiteral, (Clone(Node->getSubExpr()), Node->getType()))
DEFINE_CLONE_EXPR(ParenExpr, (Node->getLParen(), Node->getRParen(), Clone(Node->getSubExpr())))
DEFINE_CLONE_EXPR(ArraySubscriptExpr, (Clone(Node->getLHS()), Clone(Node->getRHS()), Node->getType(), Node->getValueKind(), Node->getObjectKind(), Node->getRBracketLoc()))
DEFINE_CLONE_EXPR(MemberExpr, (Clone(Node->getBase()), Node->isArrow(), Node->getMemberDecl(), Node->getMemberLoc(), Node->getType(), Node->getValueKind(), Node->getObjectKind()))
DEFINE_CLONE_EXPR(CompoundLiteralExpr, (Node->getLParenLoc(), Node->getTypeSourceInfo(), Node->getType(), Node->getValueKind(), Clone(Node->getInitializer()), Node->isFileScope()))
DEFINE_CREATE_EXPR(ImplicitCastExpr, (Ctx, Node->getType(), Node->getCastKind(), Clone(Node->getSubExpr()), 0, Node->getValueKind()))
DEFINE_CREATE_EXPR(CStyleCastExpr, (Ctx, Node->getType(), Node->getValueKind(), Node->getCastKind(), Clone(Node->getSubExpr()), 0, Node->getTypeInfoAsWritten(), Node->getLParenLoc(), Node->getRParenLoc()))
DEFINE_CREATE_EXPR(CXXStaticCastExpr, (Ctx, Node->getType(), Node->getValueKind(), Node->getCastKind(), Clone(Node->getSubExpr()), 0, Node->getTypeInfoAsWritten(), Node->getOperatorLoc(), Node->getRParenLoc(), Node->getAngleBrackets()))
DEFINE_CREATE_EXPR(CXXDynamicCastExpr, (Ctx, Node->getType(), Node->getValueKind(), Node->getCastKind(), Clone(Node->getSubExpr()), 0, Node->getTypeInfoAsWritten(), Node->getOperatorLoc(), Node->getRParenLoc(), Node->getAngleBrackets()))
DEFINE_CREATE_EXPR(CXXReinterpretCastExpr, (Ctx, Node->getType(), Node->getValueKind(), Node->getCastKind(), Clone(Node->getSubExpr()), 0, Node->getTypeInfoAsWritten(), Node->getOperatorLoc(), Node->getRParenLoc(), Node->getAngleBrackets()))
DEFINE_CREATE_EXPR(CXXConstCastExpr, (Ctx, Node->getType(), Node->getValueKind(), Clone(Node->getSubExpr()), Node->getTypeInfoAsWritten(), Node->getOperatorLoc(), Node->getRParenLoc(), Node->getAngleBrackets()))
DEFINE_CREATE_EXPR(CXXConstructExpr, (Ctx, Node->getType(), Node->getLocation(), Node->getConstructor(), Node->isElidable(), llvm::makeArrayRef(Node->getArgs(), Node->getNumArgs()), Node->hadMultipleCandidates(), Node->isListInitialization(), Node->requiresZeroInitialization(), Node->getConstructionKind(), Node->getParenOrBraceRange()))
DEFINE_CREATE_EXPR(CXXFunctionalCastExpr, (Ctx, Node->getType(), Node->getValueKind(), Node->getTypeInfoAsWritten(), Node->getCastKind(), Clone(Node->getSubExpr()), 0, Node->getLParenLoc(), Node->getRParenLoc()))
DEFINE_CLONE_EXPR(CXXTemporaryObjectExpr, (Ctx, Node->getConstructor(), Node->getTypeSourceInfo(), llvm::makeArrayRef(Node->getArgs(), Node->getNumArgs()), Node->getSourceRange(), Node->hadMultipleCandidates(), Node->isListInitialization(), Node->requiresZeroInitialization()))
DEFINE_CLONE_EXPR(MaterializeTemporaryExpr, (Node->getType(), Clone(Node->GetTemporaryExpr()), Node->isBoundToLvalueReference(), Node->getExtendingDecl())) 
DEFINE_CLONE_EXPR(CompoundAssignOperator, (Clone(Node->getLHS()), Clone(Node->getRHS()), Node->getOpcode(), Node->getType(),
                                           Node->getValueKind(), Node->getObjectKind(), Node->getComputationLHSType(), Node->getComputationResultType(), Node->getOperatorLoc(), Node->isFPContractable()))
DEFINE_CLONE_EXPR(ConditionalOperator, (Clone(Node->getCond()), Node->getQuestionLoc(), Clone(Node->getLHS()), Node->getColonLoc(), Clone(Node->getRHS()), Node->getType(), Node->getValueKind(), Node->getObjectKind()))
DEFINE_CLONE_EXPR(AddrLabelExpr, (Node->getAmpAmpLoc(), Node->getLabelLoc(), Node->getLabel(), Node->getType()))
DEFINE_CLONE_EXPR(StmtExpr, (Clone(Node->getSubStmt()), Node->getType(), Node->getLParenLoc(), Node->getRParenLoc()))
DEFINE_CLONE_EXPR(BinaryTypeTraitExpr, (Node->getSourceRange().getBegin(), Node->getTrait(), Node->getLhsTypeSourceInfo(), Node->getRhsTypeSourceInfo(), Node->getValue(), Node->getSourceRange().getEnd(), Node->getType()))
DEFINE_CLONE_EXPR(ChooseExpr, (Node->getBuiltinLoc(), Clone(Node->getCond()), Clone(Node->getLHS()), Clone(Node->getRHS()), Node->getType(), Node->getValueKind(), Node->getObjectKind(), Node->getRParenLoc(), Node->isConditionTrue(), Node->isTypeDependent(), Node->isValueDependent()))
DEFINE_CLONE_EXPR(GNUNullExpr, (Node->getType(), Node->getTokenLocation()))
DEFINE_CLONE_EXPR(VAArgExpr, (Node->getBuiltinLoc(), Clone(Node->getSubExpr()), Node->getWrittenTypeInfo(), Node->getRParenLoc(), Node->getType()))
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

Stmt* StmtClone::VisitInitListExpr(InitListExpr* Node) {
  llvm::SmallVector<Expr*, 8> initExprs(Node->getNumInits());
  for (unsigned i = 0, e = Node->getNumInits(); i < e; ++i)
    initExprs[i] = Clone(Node->getInit(i));

  SourceLocation lBrace = Node->getLBraceLoc();
  SourceLocation rBrace = Node->getRBraceLoc();

  InitListExpr* result 
    = initExprs.empty() ? new (Ctx) InitListExpr(Ctx, lBrace, 0, rBrace) 
    : new (Ctx) InitListExpr(Ctx, lBrace, initExprs, rBrace);

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

  return DesignatedInitExpr::Create(Ctx, Node->getDesignator(0), Node->size(),
                                    indexExprsRef,
                                    Node->getEqualOrColonLoc(),
                                    Node->usesGNUSyntax(), indexExprs[0]);
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
  CallExpr* result = new (Ctx) CallExpr(Ctx, Clone(Node->getCallee()), 
                                        llvm::ArrayRef<Expr*>(), 
                                        Node->getType(), 
                                        Node->getValueKind(), 
                                        Node->getRParenLoc());
  result->setNumArgs(Ctx, Node->getNumArgs());
  for (unsigned i = 0, e = Node->getNumArgs(); i < e; ++i)
    result->setArg(i, Clone(Node->getArg(i)));

  result->setValueDependent(Node->isValueDependent());  
  result->setTypeDependent(Node->isTypeDependent());    

  return result;
}

Stmt* StmtClone::VisitCXXOperatorCallExpr(CXXOperatorCallExpr* Node) {
  CXXOperatorCallExpr* result 
    = new (Ctx) CXXOperatorCallExpr(Ctx, Node->getOperator(), 
                                    Clone(Node->getCallee()), 0, 
                                    Node->getType(), 
                                    Node->getValueKind(), 
                                    Node->getRParenLoc(), 
                                    Node->isFPContractable());
  result->setNumArgs(Ctx, Node->getNumArgs());
  for (unsigned i = 0, e = Node->getNumArgs(); i < e; ++i)
    result->setArg(i, Clone(Node->getArg(i)));

  result->setValueDependent(Node->isValueDependent());  
  result->setTypeDependent(Node->isTypeDependent());    

  return result;
}

Stmt* StmtClone::VisitCXXMemberCallExpr(CXXMemberCallExpr * Node) {
  CXXMemberCallExpr* result 
    = new (Ctx) CXXMemberCallExpr(Ctx, Clone(Node->getCallee()), 0, 
                                  Node->getType(), 
                                  Node->getValueKind(), Node->getRParenLoc());
  result->setNumArgs(Ctx, Node->getNumArgs());

  for (unsigned i = 0, e = Node->getNumArgs(); i < e; ++i)
    result->setArg(i, Clone(Node->getArg(i)));

  result->setValueDependent(Node->isValueDependent());  
  result->setTypeDependent(Node->isTypeDependent());    

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
  CaseStmt* result = new (Ctx) CaseStmt(Clone(Node->getLHS()), 
                                        Clone(Node->getRHS()), 
                                        Node->getCaseLoc(), 
                                        Node->getEllipsisLoc(), 
                                        Node->getColonLoc());
  result->setSubStmt(Clone(Node->getSubStmt()));
  return result;
}

Stmt* StmtClone::VisitSwitchStmt(SwitchStmt* Node) {
  SwitchStmt* result 
    = new (Ctx) SwitchStmt(Ctx, CloneDeclOrNull(Node->getConditionVariable()), 
                           Clone(Node->getCond()));
  result->setBody(Clone(Node->getBody()));
  result->setSwitchLoc(Node->getSwitchLoc());
  return result;
}

DEFINE_CLONE_STMT(ReturnStmt, (Node->getReturnLoc(), Clone(Node->getRetValue()), 0))
DEFINE_CLONE_STMT(DefaultStmt, (Node->getDefaultLoc(), Node->getColonLoc(), Clone(Node->getSubStmt())))
DEFINE_CLONE_STMT(GotoStmt, (Node->getLabel(), Node->getGotoLoc(), Node->getLabelLoc()))
DEFINE_CLONE_STMT(WhileStmt, (Ctx, CloneDeclOrNull(Node->getConditionVariable()), Clone(Node->getCond()), Clone(Node->getBody()), Node->getWhileLoc()))
DEFINE_CLONE_STMT(DoStmt, (Clone(Node->getBody()), Clone(Node->getCond()), Node->getDoLoc(), Node->getWhileLoc(), Node->getRParenLoc()))
DEFINE_CLONE_STMT(IfStmt, (Ctx, Node->getIfLoc(), CloneDeclOrNull(Node->getConditionVariable()), Clone(Node->getCond()), Clone(Node->getThen()), Node->getElseLoc(), Clone(Node->getElse())))
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
  return new (Ctx) CompoundStmt(Ctx, stmtsRef, Node->getLBracLoc(), 
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
    cloned_Decl->setInit(Clone(VD->getInit()));
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
  return new (Ctx) DeclStmt(clonedDecls, Node->getStartLoc(), Node->getEndLoc());
}

Stmt* StmtClone::VisitStmt(Stmt*) { 
  assert(0 && "clone not fully implemented"); 
  return 0; 
}

//--------------------------------------------------------- 
  } // end namespace utils
} // end namespace autodiff
