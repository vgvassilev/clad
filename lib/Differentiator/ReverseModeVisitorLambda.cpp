//===- ReverseModeVisitorLambda.cpp - Lambdas in reverse mode -------------===//
//
// The pullback of a lambda, built as a lambda of its own, and of a call that
// passes a lambda to a function template.
//
//===----------------------------------------------------------------------===//

#include "clad/Differentiator/ReverseModeVisitor.h"

#include "LoopScope.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/DerivativeBuilder.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/StmtClone.h"
#include "clad/Differentiator/VisitorBase.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/Type.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/Lambda.h"
#include "clang/Basic/Version.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <utility>

using namespace clang;

namespace clad {
#if CLANG_VERSION_MAJOR > 16
clang::Expr* ReverseModeVisitor::buildDerivedLambda(const LambdaExpr* LE) {
  LambdaBuilder LBuilder(*this);
  LBuilder.start(LE, GetLambdaDerivativeType(LE),
                 [&](llvm::SmallVectorImpl<ParmVarDecl*>& params) {
                   BuildParams(params, LE);
                 });

  Stmts OuterGlobals;
  std::swap(m_Globals, OuterGlobals);

  beginBlock();

  StmtDiff BodyDiff = Visit(LE->getCallOperator()->getBody());
  for (auto* S : cast<CompoundStmt>(BodyDiff.getStmt())->body())
    addToCurrentBlock(S);
  for (auto* S : cast<CompoundStmt>(BodyDiff.getStmt_dx())->body())
    addToCurrentBlock(S);

  CompoundStmt* DerivedBody = endBlock();

  if (!m_Globals.empty()) {
    llvm::SmallVector<Stmt*, 32> NewBodyStmts;
    NewBodyStmts.append(m_Globals.begin(), m_Globals.end());
    NewBodyStmts.append(DerivedBody->body().begin(), DerivedBody->body().end());

    DerivedBody = CompoundStmt::Create(
        m_Sema.getASTContext(), NewBodyStmts, FPOptionsOverride(),
        DerivedBody->getLBracLoc(), DerivedBody->getRBracLoc());
  }
  m_Globals.clear();
  std::swap(m_Globals, OuterGlobals);

  return LBuilder.finish(DerivedBody);
}

StmtDiff ReverseModeVisitor::VisitLambdaExpr(const LambdaExpr* LE) {
  // Snapshot by value captures at creation; the reverse sweep
  // feeds the snapshot adjoint back into the capture
  for (const LambdaCapture& Capture : LE->captures()) {
    if (!Capture.capturesVariable() || Capture.getCaptureKind() != LCK_ByCopy)
      continue;
    auto* capVD = cast<VarDecl>(Capture.getCapturedVar());
    const FieldDecl* field = getCaptureField(LE, capVD);
    if (m_LambdaCaptureSnapshots.contains(field))
      continue;
    VarDecl* encVD = getEnclosingCaptureClone(capVD);
    QualType ty =
        utils::getNonConstType(capVD->getType().getNonReferenceType(), m_Sema);
    VarDecl* snap = BuildVarDecl(ty, "_t", BuildDeclRef(encVD));
    addToCurrentBlock(BuildDeclStmt(snap), direction::forward);
    VarDecl* dsnap = BuildVarDecl(ty, "_d_t", getZeroInit(ty));
    AddToGlobalBlock(BuildDeclStmt(dsnap));
    m_Variables[snap] = {dsnap};
    auto it = m_Variables.find(encVD);
    if (it != m_Variables.end())
      addToCurrentBlock(
          BuildOp(BO_AddAssign, buildAdjoint(it->second), BuildDeclRef(dsnap)),
          direction::reverse);
    m_LambdaCaptureSnapshots.try_emplace(field, snap);
  }
  DiffRequest LambdaReq = m_DiffReq;
  LambdaReq.Function = LE->getCallOperator();
  LambdaReq.Functor = LE->getLambdaClass();

  ReverseModeVisitor NestedVisitor(m_Builder, LambdaReq);
  Expr* lambdaE = NestedVisitor.buildDerivedLambda(LE);

  // Build the primal with a fresh closure so it does not share the operator()
  // body with the original lambda (or with a second primal clone emitted in
  // the derivative lambda).
  return {buildClonedLambda(LE), lambdaE};
}
#endif // CLANG_VERSION_MAJOR

StmtDiff ReverseModeVisitor::DifferentiateCallbackCall(const CallExpr* CE,
                                                       const LambdaExpr* LE) {
  if (diagnoseUnsupportedCallback(CE, LE))
    return {};
  const CXXMethodDecl* CallOp = LE->getCallOperator();

  beginBlock(direction::reverse);
  LoopCounter loopCounter(*this);
  ScopeRAII callScope(*this, Scope::DeclScope | Scope::ControlScope |
                                 Scope::BreakScope | Scope::ContinueScope);
  LoopScope Loop(*this);
  Loop.Tapes = true;
  Loop.Facts = &m_DiffReq.getLoopFacts(CallOp->getBody());
  llvm::SaveAndRestore<Expr*> SaveCurrentBreakFlagExpr(m_CurrentBreakFlagExpr);
  m_CurrentBreakFlagExpr = nullptr;
  llvm::SaveAndRestore<llvm::SmallVector<Expr*, 4>> SaveLoopAdjointIterators(
      m_LoopAdjointIterators);
  m_LoopAdjointIterators.push_back(nullptr);

  // Each parameter becomes a variable of the derivative, which every call
  // sets in the forward sweep and the reverse sweep restores.
  llvm::SmallVector<VarDecl*, 2> ParamVars;
  llvm::SmallVector<StmtDiff, 2> ParamStores;
  for (const ParmVarDecl* PVD : CallOp->parameters()) {
    QualType T = PVD->getType().getNonReferenceType().getUnqualifiedType();
    VarDecl* VD = BuildGlobalVarDecl(
        T, PVD->getDeclName() ? PVD->getName() : "_t", getZeroInit(T));
    AddToGlobalBlock(BuildDeclStmt(VD));
    m_DeclReplacements[PVD] = VD;
    ParamVars.push_back(VD);
    ParamStores.push_back(StoreAndRestore(BuildDeclRef(VD)));
  }
  StmtDiff BodyDiff = DifferentiateLoopBody(
      CallOp->getBody(), loopCounter, /*condVarDiff=*/nullptr,
      /*forLoopIncDiff=*/nullptr, /*isForLoop=*/false, LE->getBeginLoc());
  for (const ParmVarDecl* PVD : CallOp->parameters())
    m_DeclReplacements.erase(PVD);
  // The reverse sweep runs in the derivative, where what the lambda captured
  // is a local again.
  LambdaCaptures Locals(*this);
  Locals.collect(BodyDiff.getStmt_dx());
  Locals.resolve(BodyDiff.getStmt_dx());

  // [&](int i0) { clad::push(_t1, i); i = i0; <forward sweep of the body> }
  Expr* Closure =
      buildCallbackClosure(LE, [&](llvm::ArrayRef<ParmVarDecl*> Params) {
        for (unsigned I = 0, N = ParamVars.size(); I < N; ++I) {
          addToCurrentBlock(ParamStores[I].getStmt(), direction::forward);
          addToCurrentBlock(BuildOp(BO_Assign, BuildDeclRef(ParamVars[I]),
                                    BuildDeclRef(Params[I])),
                            direction::forward);
        }
        for (Stmt* S : cast<CompoundStmt>(BodyDiff.getStmt())->body())
          addToCurrentBlock(S, direction::forward);
      });

  Expr* Call = buildCallWithCallback(CE, LE, Closure);

  // while (_t0) { <reverse sweep of the body>; _t0--; i = clad::pop(_t1); }
  // A reverse block comes out last statement first.
  beginBlock(direction::reverse);
  for (StmtDiff& Store : ParamStores)
    addToCurrentBlock(Store.getStmt_dx(), direction::reverse);
  addToCurrentBlock(BodyDiff.getStmt_dx(), direction::reverse);
  CompoundStmt* RevBody = endBlock(direction::reverse);
  Sema::ConditionResult CounterCondition =
      loopCounter.getCounterConditionResult();
  addToCurrentBlock(m_Sema
                        .ActOnWhileStmt(/*WhileLoc=*/noLoc, /*LParenLoc=*/noLoc,
                                        CounterCondition,
                                        /*RParenLoc=*/noLoc, RevBody)
                        .get(),
                    direction::reverse);
  Stmt* Reverse = utils::unwrapIfSingleStmt(endBlock(direction::reverse));
  addToCurrentBlock(Reverse, direction::reverse);
  return StmtDiff(Call);
}
} // end namespace clad
