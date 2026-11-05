//===- BaseForwardModeVisitorLambda.cpp - Lambdas in forward mode ---------===//
//
// The pushforward of a lambda, built as a lambda of its own, and of a call
// that passes a lambda to a function template.
//
//===----------------------------------------------------------------------===//

#include "clad/Differentiator/BaseForwardModeVisitor.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/DiffMode.h"
#include "clad/Differentiator/VisitorBase.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/Type.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/Version.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/SmallVector.h"

#include <cassert>

using namespace clang;

namespace clad {
clang::QualType
BaseForwardModeVisitor::GetLambdaPushforwardType(const LambdaExpr* LE) {
  FunctionDecl* FD = LE->getCallOperator();
  llvm::SmallVector<const ValueDecl*, 4> diffParams;
  for (const auto* param : FD->parameters())
    diffParams.push_back(param);
  QualType baseTy =
      utils::GetDerivativeType(m_Sema, FD, DiffMode::pushforward, diffParams,
                               /*forCustomDerv=*/false,
                               /*shouldUseRestoreTracker=*/false);
  const auto* FPT = baseTy->castAs<FunctionProtoType>();
  llvm::SmallVector<QualType, 8> paramTypes(FPT->param_types().begin(),
                                            FPT->param_types().end());
  for (const LambdaCapture& Capture : LE->captures()) {
    const auto* capVD = cast<VarDecl>(Capture.getCapturedVar());
    QualType valTy =
        utils::getNonConstType(capVD->getType().getNonReferenceType(), m_Sema);
    paramTypes.push_back(valTy);
    paramTypes.push_back(utils::GetParameterDerivativeType(
        m_Sema, DiffMode::pushforward, valTy));
  }
  return m_Context.getFunctionType(FPT->getReturnType(), paramTypes,
                                   FPT->getExtProtoInfo());
}

#if CLANG_VERSION_MAJOR > 16
clang::Expr*
BaseForwardModeVisitor::buildPushforwardLambda(const LambdaExpr* LE) {
  QualType dFnType = GetLambdaPushforwardType(LE);
  // auto cannot deduce the return type from a braced init list, so we set it.
  QualType retTy = dFnType->castAs<FunctionProtoType>()->getReturnType();

  LambdaBuilder LBuilder(*this);
  LBuilder.start(
      LE, dFnType,
      [&](llvm::SmallVectorImpl<ParmVarDecl*>& params) {
        SetupDerivativeParameters(params);
        for (const LambdaCapture& Capture : LE->captures()) {
          auto* capVD = cast<VarDecl>(Capture.getCapturedVar());
          QualType valTy = utils::getNonConstType(
              capVD->getType().getNonReferenceType(), m_Sema);
          auto* valPVD = utils::BuildParmVarDecl(
              m_Sema, m_Derivative,
              CreateUniqueIdentifier(capVD->getNameAsString()), valTy);
          m_Sema.PushOnScopeChains(valPVD, getCurrentScope(),
                                   /*AddToContext=*/false);
          params.push_back(valPVD);
          m_DeclReplacements[capVD] = valPVD;
          auto* dPVD = utils::BuildParmVarDecl(
              m_Sema, m_Derivative,
              CreateUniqueIdentifier("_d_" + capVD->getNameAsString()),
              utils::GetParameterDerivativeType(m_Sema, DiffMode::pushforward,
                                                valTy));
          m_Sema.PushOnScopeChains(dPVD, getCurrentScope(),
                                   /*AddToContext=*/false);
          params.push_back(dPVD);
          m_Variables[valPVD] = {dPVD};
        }
      },
      retTy);

  beginBlock();
  StmtDiff bodyDiff = Visit(LE->getCallOperator()->getBody());
  for (auto* S : cast<CompoundStmt>(bodyDiff.getStmt())->body())
    addToCurrentBlock(S);
  CompoundStmt* body = endBlock();

  return LBuilder.finish(body);
}
#endif // CLANG_VERSION_MAJOR

StmtDiff
BaseForwardModeVisitor::DifferentiateCallbackCall(const CallExpr* CE,
                                                  const LambdaExpr* LE) {
  if (diagnoseUnsupportedCallback(CE, LE))
    return {};
  const CXXMethodDecl* CallOp = LE->getCallOperator();
  // [&](int i0) { <the body and its derivative> }
  Expr* Closure =
      buildCallbackClosure(LE, [&](llvm::ArrayRef<ParmVarDecl*> Params) {
        for (unsigned I = 0, N = Params.size(); I < N; ++I)
          m_DeclReplacements[CallOp->getParamDecl(I)] = Params[I];
        StmtDiff BodyDiff = Visit(CallOp->getBody());
        for (Stmt* S : cast<CompoundStmt>(BodyDiff.getStmt())->body())
          addToCurrentBlock(S);
      });
  for (const ParmVarDecl* PVD : CallOp->parameters())
    m_DeclReplacements.erase(PVD);
  return StmtDiff(buildCallWithCallback(CE, LE, Closure));
}
} // end namespace clad
