#include "clad/Differentiator/ReverseModeForwPassVisitor.h"
#include "clad/Differentiator/VisitorBase.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/ErrorEstimator.h"
#include "clad/Differentiator/ParseDiffArgsTypes.h"

#include "llvm/Support/SaveAndRestore.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/OperationKinds.h"
#include "clang/Basic/LLVM.h"

#include <algorithm>
#include <iterator>

using namespace clang;

namespace clad {

ReverseModeForwPassVisitor::ReverseModeForwPassVisitor(
    DerivativeBuilder& builder, const DiffRequest& request)
    : ReverseModeVisitor(builder, request) {}

DerivativeAndOverload ReverseModeForwPassVisitor::Derive() {
  const FunctionDecl* FD = m_DiffReq.Function;

  assert(m_DiffReq.Mode == DiffMode::reverse_mode_forward_pass);

  assert(m_DiffReq.Function && "Must not be null.");

  auto fnName =
      clad::utils::ComputeEffectiveFnName(m_DiffReq.Function) + "_reverse_forw";
  auto fnDNI = utils::BuildDeclarationNameInfo(m_Sema, fnName);

  auto fnType = GetDerivativeType();

  llvm::SaveAndRestore<DeclContext*> saveContext(m_Sema.CurContext);
  llvm::SaveAndRestore<Scope*> saveScope(getCurrentScope(),
                                         getEnclosingNamespaceOrTUScope());
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  auto* DC = const_cast<DeclContext*>(m_DiffReq->getDeclContext());

  m_Sema.CurContext = DC;
  SourceLocation validLoc{m_DiffReq->getLocation()};
  DeclWithContext fnBuildRes = m_Builder.cloneFunction(
      m_DiffReq.Function, *this, m_Sema.CurContext, validLoc, fnDNI, fnType);
  m_Derivative = fnBuildRes.first;

  beginScope(Scope::FunctionPrototypeScope | Scope::FunctionDeclarationScope |
             Scope::DeclScope);
  m_Sema.PushFunctionScope();
  m_Sema.PushDeclContext(getCurrentScope(), m_Derivative);

  DiffParams args{};
  std::copy(FD->param_begin(), FD->param_end(), std::back_inserter(args));
  auto params = BuildParams(args);
  m_Derivative->setParams(params);
  m_Derivative->setBody(nullptr);

  if (!m_DiffReq.DeclarationOnly) {
    beginScope(Scope::FnScope | Scope::DeclScope);
    m_DerivativeFnScope = getCurrentScope();

    beginBlock();
    beginBlock(direction::reverse);

    // If we the differentiated function is a constructor, generate `this`
    // object and differentiate its inits.
    Stmt* ctorReturnStmt = nullptr;
    if (const auto* CD = dyn_cast<CXXConstructorDecl>(m_DiffReq.Function)) {
      QualType thisTy = CD->getThisType();
      StmtDiff thisObj = BuildThisExpr(thisTy);
      StmtDiff dthisObj = BuildThisExpr(thisTy, /*isDerivedThis=*/true);
      m_ThisExprDerivative = dthisObj.getExpr();
      addToCurrentBlock(dthisObj.getStmt_dx());
      for (CXXCtorInitializer* CI : CD->inits()) {
        StmtDiff CI_diff = DifferentiateCtorInit(CI, thisObj.getExpr());
        addToCurrentBlock(CI_diff.getStmt(), direction::forward);
        addToCurrentBlock(CI_diff.getStmt_dx(), direction::forward);
      }
      // Build `return {*_this, *_d_this};`
      SourceLocation validLoc{CD->getBeginLoc()};
      llvm::SmallVector<Expr*, 2> returnArgs = {
          BuildOp(UO_Deref, thisObj.getExpr()),
          BuildOp(UO_Deref, dthisObj.getExpr())};
      Expr* returnInitList =
          m_Sema.ActOnInitList(validLoc, returnArgs, validLoc).get();
      ctorReturnStmt = m_Sema.BuildReturnStmt(validLoc, returnInitList).get();
    }

    StmtDiff bodyDiff = Visit(m_DiffReq->getBody());
    Stmt* forward = bodyDiff.getStmt();

    for (Stmt* S : ReverseModeVisitor::m_Globals)
      addToCurrentBlock(S);

    if (auto* CS = dyn_cast_or_null<CompoundStmt>(forward))
      for (Stmt* S : CS->body())
        addToCurrentBlock(S);
    else
      addToCurrentBlock(forward);

    addToCurrentBlock(ctorReturnStmt);

    Stmt* fnBody = endBlock();
    m_Derivative->setBody(fnBody);
    endScope();
  }
  m_Sema.PopFunctionScopeInfo();
  m_Sema.PopDeclContext();
  endScope();
  return DerivativeAndOverload{m_Derivative, nullptr};
}

llvm::SmallVector<clang::ParmVarDecl*, 8>
ReverseModeForwPassVisitor::BuildParams(DiffParams& diffParams) {
  llvm::SmallVector<clang::ParmVarDecl*, 8> params;
  llvm::SmallVector<clang::ParmVarDecl*, 8> paramDerivatives;
  params.reserve(m_DiffReq->getNumParams() + diffParams.size());
  const auto* derivativeFnType =
      cast<FunctionProtoType>(m_Derivative->getType());

  std::size_t dParamTypesIdx = m_DiffReq->getNumParams();

  QualType tagType;
  if (const auto* CD = dyn_cast<CXXConversionDecl>(m_DiffReq.Function))
    tagType = CD->getConversionType();
  else if (const auto* CD = dyn_cast<CXXConstructorDecl>(m_DiffReq.Function))
    tagType = CD->getThisType()->getPointeeType();
  if (!tagType.isNull()) {
    QualType typeTag = utils::GetCladTagOfType(m_Sema, tagType);
    IdentifierInfo* emptyII = &m_Context.Idents.get("");
    ParmVarDecl* typeTagPVD =
        utils::BuildParmVarDecl(m_Sema, m_Derivative, emptyII, typeTag);
    params.push_back(typeTagPVD);
    m_Sema.PushOnScopeChains(typeTagPVD, getCurrentScope(),
                             /*AddToContext=*/false);
    ++dParamTypesIdx;
  }

  if (const auto* MD = dyn_cast<CXXMethodDecl>(m_DiffReq.Function)) {
    const CXXRecordDecl* RD = MD->getParent();
    if (MD->isInstance() && !RD->isLambda() && !isa<CXXConstructorDecl>(MD)) {
      auto* thisDerivativePVD = utils::BuildParmVarDecl(
          m_Sema, m_Derivative, CreateUniqueIdentifier("_d_this"),
          derivativeFnType->getParamType(dParamTypesIdx));
      paramDerivatives.push_back(thisDerivativePVD);

      if (thisDerivativePVD->getIdentifier())
        m_Sema.PushOnScopeChains(thisDerivativePVD, getCurrentScope(),
                                 /*AddToContext=*/false);

      m_ThisExprDerivative = BuildDeclRef(thisDerivativePVD);
      ++dParamTypesIdx;
    }
  }
  for (auto* PVD : m_DiffReq->parameters()) {
    // FIXME: Call expression may contain default arguments that we are now
    // removing. This may cause issues.
    auto* newPVD = utils::BuildParmVarDecl(
        m_Sema, m_Derivative, PVD->getIdentifier(), PVD->getType(),
        PVD->getStorageClass(), /*DefArg=*/nullptr, PVD->getTypeSourceInfo());
    params.push_back(newPVD);

    if (newPVD->getIdentifier())
      m_Sema.PushOnScopeChains(newPVD, getCurrentScope(),
                               /*AddToContext=*/false);
    else {
      IdentifierInfo* newName = CreateUniqueIdentifier("arg");
      newPVD->setDeclName(newName);
      m_DeclReplacements[PVD] = newPVD;
    }

    auto* it = std::find(std::begin(diffParams), std::end(diffParams), PVD);
    if (it != std::end(diffParams)) {
      *it = newPVD;
      QualType dType = derivativeFnType->getParamType(dParamTypesIdx);
      IdentifierInfo* dII =
          CreateUniqueIdentifier("_d_" + newPVD->getNameAsString());
      auto* dPVD = utils::BuildParmVarDecl(m_Sema, m_Derivative, dII, dType,
                                           PVD->getStorageClass());
      paramDerivatives.push_back(dPVD);
      ++dParamTypesIdx;

      if (dPVD->getIdentifier())
        m_Sema.PushOnScopeChains(dPVD, getCurrentScope(),
                                 /*AddToContext=*/false);
      m_Variables[*it] = BuildDeclRef(dPVD), m_DiffReq->getLocation();
    }
  }
  if (m_DiffReq.UseRestoreTracker) {
    QualType trackerTy = utils::GetRestoreTrackerType(m_Sema);
    trackerTy = m_Sema.getASTContext().getLValueReferenceType(trackerTy);
    ParmVarDecl* trackerPVD = utils::BuildParmVarDecl(
        m_Sema, m_Derivative, CreateUniqueIdentifier("_tracker"), trackerTy);
    paramDerivatives.push_back(trackerPVD);
    m_Sema.PushOnScopeChains(trackerPVD, getCurrentScope(),
                             /*AddToContext=*/false);
    m_RestoreTracker = BuildDeclRef(trackerPVD);
  }
  params.insert(params.end(), paramDerivatives.begin(), paramDerivatives.end());
  return params;
}

StmtDiff ReverseModeForwPassVisitor::StoreAndRestore(clang::Expr* E,
                                                     llvm::StringRef prefix,
                                                     bool moveToTape) {
  if (!m_RestoreTracker)
    return {};
  if (const auto* DRE = dyn_cast<DeclRefExpr>(E->IgnoreCasts())) {
    const auto* VD = cast<VarDecl>(DRE->getDecl());
    if (!VD->getType()->isReferenceType())
      return {};
  }
  Expr* storeCall = BuildCallExprToMemFn(m_RestoreTracker,
                                         /*MemberFunctionName=*/"store", {E});
  return {storeCall};
}

StmtDiff ReverseModeForwPassVisitor::ProcessSingleStmt(const clang::Stmt* S) {
  StmtDiff SDiff = Visit(S);
  return {SDiff.getStmt()};
}

StmtDiff
ReverseModeForwPassVisitor::VisitCompoundStmt(const clang::CompoundStmt* CS) {
  beginScope(Scope::DeclScope);
  beginBlock();
  for (Stmt* S : CS->body()) {
    StmtDiff SDiff = ProcessSingleStmt(S);
    addToCurrentBlock(SDiff.getStmt());
  }
  CompoundStmt* forward = endBlock();
  endScope();
  return {forward};
}

StmtDiff ReverseModeForwPassVisitor::VisitUnaryOperator(
    const clang::UnaryOperator* UnOp) {
  StmtDiff UnOpDiff = ReverseModeVisitor::VisitUnaryOperator(UnOp);
  if (UnOp->isIncrementDecrementOp()) {
    auto* base = cast<UnaryOperator>(UnOpDiff.getExpr())->getSubExpr();
    StmtDiff pushPop = StoreAndRestore(base);
    addToCurrentBlock(pushPop.getExpr());
  }
  return UnOpDiff;
}

ReverseModeForwPassVisitor::DelayedStoreResult
ReverseModeForwPassVisitor::DelayedGlobalStoreAndRef(Expr* E,
                                                     llvm::StringRef prefix,
                                                     bool forceStore) {
  assert(E && "must be provided");
  StmtDiff Ediff = Visit(E);
  return DelayedStoreResult{*this, Ediff,
                            /*Declaration=*/nullptr,
                            /*isInsideLoop=*/false,
                            /*isFnScope=*/false};
}

StmtDiff ReverseModeForwPassVisitor::VisitDeclRefExpr(const DeclRefExpr* DRE) {
  DeclRefExpr* clonedDRE = nullptr;
  // Check if referenced Decl was "replaced" with another identifier inside
  // the derivative
  const auto* VD = dyn_cast<VarDecl>(DRE->getDecl());
  auto it = m_DeclReplacements.find(VD);
  if (it != std::end(m_DeclReplacements))
    clonedDRE = BuildDeclRef(it->second);
  else
    clonedDRE = cast<DeclRefExpr>(Clone(DRE));

  auto* decl = dyn_cast<VarDecl>(clonedDRE->getDecl());
  auto foundAdjoint = m_Variables.find(decl);
  Expr* adjoint = nullptr;
  if (foundAdjoint != m_Variables.end())
    adjoint = foundAdjoint->second;

  return StmtDiff(clonedDRE, adjoint);
}

StmtDiff
ReverseModeForwPassVisitor::VisitReturnStmt(const clang::ReturnStmt* RS) {
  const Expr* value = RS->getRetValue();
  StmtDiff returnDiff;
  if (value)
    returnDiff = Visit(value);
  SourceLocation validLoc{RS->getBeginLoc()};
  if (!utils::isMemoryType(m_DiffReq->getReturnType()))
    return m_Sema.BuildReturnStmt(validLoc, returnDiff.getExpr()).get();
  llvm::SmallVector<Expr*, 2> returnArgs = {returnDiff.getExpr(),
                                            returnDiff.getExpr_dx()};
  Expr* returnInitList =
      m_Sema.ActOnInitList(validLoc, returnArgs, validLoc).get();
  Stmt* newRS = m_Sema.BuildReturnStmt(validLoc, returnInitList).get();
  return {newRS};
}

DeclDiff<clang::VarDecl>
ReverseModeForwPassVisitor::DifferentiateVarDecl(const clang::VarDecl* VD,
                                                 bool /*keepLocal*/) {
  QualType DerivedType = CloneType(VD->getType());
  StmtDiff initDiff;
  if (const Expr* init = VD->getInit())
    initDiff = Visit(init);
  // Adjoints should always be initialized
  if (!initDiff.getExpr_dx()) {
    Expr* zero = getZeroInit(DerivedType);
    initDiff.updateStmtDx(zero);
  }
  auto* VDCloned = BuildGlobalVarDecl(DerivedType, VD->getNameAsString(),
                                      initDiff.getExpr(), VD->isDirectInit());
  auto* VDDerived =
      BuildGlobalVarDecl(DerivedType, "_d_" + VD->getNameAsString(),
                         initDiff.getExpr_dx(), VD->isDirectInit());
  m_Variables.emplace(VDCloned, BuildDeclRef(VDDerived));
  if ((VD->getDeclName() != VDCloned->getDeclName() ||
       DerivedType != VD->getType()))
    m_DeclReplacements[VD] = VDCloned;
  return {VDCloned, VDDerived};
}

} // namespace clad
