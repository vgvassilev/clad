#include "clad/Differentiator/ErrorEstimator.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/DerivativeBuilder.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/ReverseModeVisitor.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/Type.h"
#include "clang/Basic/LLVM.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/TemplateDeduction.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>
#include <string>

using namespace clang;

namespace clad {

QualType getUnderlyingArrayType(QualType baseType, ASTContext& C) {
  if (baseType->isArrayType()) {
    return C.getBaseElementType(baseType);
  } else if (auto PTType = baseType->getAs<PointerType>()) {
    return PTType->getPointeeType();
  }
  return baseType;
}

DeclRefExpr* ErrorEstimationHandler::BuildFinalErrorExpr() {
  return m_RMV->BuildDeclRef(m_Params->back());
}

void ErrorEstimationHandler::BuildReturnErrorStmt() {
  // If we encountered any arithmetic expression in the return statement,
  // we must add its error to the final estimate.
  if (m_RetErrorExpr) {
    auto flitr =
        FloatingLiteral::Create(m_RMV->m_Context, llvm::APFloat(1.0), true,
                                m_RMV->m_Context.DoubleTy, noLoc);
    Expr* finExpr = AssignError(StmtDiff(m_RetErrorExpr, flitr), "return_expr");
    m_RMV->addToCurrentBlock(
        m_RMV->BuildOp(BO_AddAssign, BuildFinalErrorExpr(), finExpr),
        direction::forward);
  }
}

void ErrorEstimationHandler::AddErrorStmtToBlock(Expr* errorExpr) {
  Expr* FinalError = BuildFinalErrorExpr();
  Stmt* errorStmt = m_RMV->BuildOp(BO_AddAssign, FinalError, errorExpr);
  auto& block = m_RMV->getCurrentBlock(direction::reverse);
  block.insert(block.begin(), errorStmt);
}

void ErrorEstimationHandler::EmitErrorEstimationStmts(
    ReverseModeVisitor::direction d /*=forward*/) {
  if (d == direction::forward) {
    while (!m_ForwardReplStmts.empty())
      m_RMV->addToCurrentBlock(m_ForwardReplStmts.pop_back_val(), d);
  } else {
    while (!m_ReverseErrorStmts.empty())
      m_RMV->addToCurrentBlock(m_ReverseErrorStmts.pop_back_val(), d);
  }
}

void ErrorEstimationHandler::SaveReturnExpr(Expr* retExpr) {
  // If the return expression is a declRefExpr or is a non-floating point
  // type, we should not do anything.
  if (GetUnderlyingDeclRefOrNull(retExpr) ||
      !retExpr->getType()->isFloatingType())
    return;

  // Build a variable to store the current return value.
  // This will be helpful in the case that we have multiple
  // returns.
  if (!m_RetErrorExpr) {
    auto retVarDecl =
        m_RMV->BuildVarDecl(m_RMV->m_Context.DoubleTy, "_ret_value",
                            m_RMV->getZeroInit(m_RMV->m_Context.DoubleTy));
    m_RMV->AddToGlobalBlock(m_RMV->BuildDeclStmt(retVarDecl));
    m_RetErrorExpr = m_RMV->BuildDeclRef(retVarDecl);
  }
  m_RMV->addToCurrentBlock(m_RMV->BuildOp(BO_Assign, m_RetErrorExpr, retExpr),
                           direction::forward);
}

void ErrorEstimationHandler::EmitNestedFunctionParamError(
    FunctionDecl* fnDecl, llvm::SmallVectorImpl<Expr*>& derivedCallArgs,
    llvm::SmallVectorImpl<Expr*>& ArgResult, size_t numArgs) {
  assert(fnDecl && "Must have a value");
  for (size_t i = 0; i < numArgs; i++) {
    if (!fnDecl->getParamDecl(0)->getType()->isLValueReferenceType())
      continue;
    // // FIXME: Argument passed by reference do not have any corresponding
    // // `ArgResultDecl`. Handle arguments passed by reference in error
    // // estimation.
    // if (utils::IsReferenceOrPointerType(fnDecl->getParamDecl(i)->getType()))
    //   continue;
    auto* derefExpr = m_RMV->BuildOp(UO_Deref, ArgResult[i]);
    Expr* errorExpr = AssignError({derivedCallArgs[i], derefExpr},
                                  fnDecl->getNameInfo().getAsString() +
                                      "_param_" + std::to_string(i));
    Expr* FinalError = BuildFinalErrorExpr();
    Expr* errorStmt = m_RMV->BuildOp(BO_AddAssign, FinalError, errorExpr);
    m_ReverseErrorStmts.push_back(errorStmt);
  }
}

bool ErrorEstimationHandler::ShouldEstimateErrorFor(VarDecl* VD) {

  // Get the types on the declartion and initalization expression.
  QualType varDeclBase = VD->getType();
  QualType varDeclType = getUnderlyingArrayType(varDeclBase, m_RMV->m_Context);
  const Expr* init = VD->getInit();
  // If declarationg type in not floating point type, we want to do two
  // things.
  if (!varDeclType->isFloatingType()) {
    // Firstly, we want to check if the declaration is a lossy conversion.
    // For example, if we have something like:
    // double y = 2.77788;
    // int x = y <-- This causes truncation in y,
    // making _delta_x = y - (double)x
    // For now, we will just warn the user of casts like these
    // because we assume the cast is intensional.
    if (init && init->IgnoreImpCasts()->getType()->isFloatingType())
      m_RMV->diag(DiagnosticsEngine::Warning, VD->getEndLoc(),
                  "lossy assignment from %0 to %1, this error will not be "
                  "taken into cosideration while estimation")
          << init->IgnoreImpCasts()->getType() << varDeclBase;
    // Secondly, we want to only register floating-point types
    // So return false here.
    return false;
  }
  // Next, we want to check if there is an assignment that leads to
  // truncation, for example, something like so
  // double y = ...some double value...
  // float x = y; <-- This leads to rounding of the lower bits
  // making _delta_x = y - (double)x
  // For now, we shall just warn against such assignments...
  // FIXME: figure how to do this out elegantly.

  // Now, we can register the variable.
  // So return true here.
  return true;
}

DeclRefExpr* ErrorEstimationHandler::GetUnderlyingDeclRefOrNull(Expr* expr) {
  // First check if it is an array subscript expression.
  ArraySubscriptExpr* temp =
      dyn_cast<ArraySubscriptExpr>(expr->IgnoreImplicit());
  // The see if it is convertiable to a DeclRefExpr.
  if (temp)
    return dyn_cast<DeclRefExpr>(temp->getBase()->IgnoreImplicit());
  else
    return dyn_cast<DeclRefExpr>(expr->IgnoreImplicit());
}

void ErrorEstimationHandler::EmitFinalErrorStmts(
    llvm::SmallVectorImpl<ParmVarDecl*>& params, unsigned numParams) {
  // Emit error variables of parameters at the end.
  for (size_t i = 0; i < numParams; i++) {
    auto* decl = cast<VarDecl>(params[i]);
    if (ShouldEstimateErrorFor(decl)) {
      if (!m_RMV->isArrayOrPointerType(params[i]->getType())) {
        auto* paramClone = m_RMV->BuildDeclRef(decl);
        // Finally emit the error.
        auto* errorExpr = GetError(paramClone, m_RMV->m_Variables[decl],
                                   params[i]->getNameAsString());
        m_RMV->addToCurrentBlock(
            m_RMV->BuildOp(BO_AddAssign, BuildFinalErrorExpr(), errorExpr));
      } else {
        auto LdiffExpr = m_RMV->m_Variables[decl];
        Expr* size = getSizeExpr(decl);
        VarDecl* idxExprDecl = nullptr;
        // Save our index expression so it can be used later.
        if (!m_IdxExpr) {
          idxExprDecl =
              m_RMV->BuildVarDecl(m_RMV->m_Context.IntTy, "i",
                                  m_RMV->getZeroInit(m_RMV->m_Context.IntTy));
          m_IdxExpr = m_RMV->BuildDeclRef(idxExprDecl);
        }
        Expr* Ldiff = nullptr;
        Ldiff = m_RMV->BuildArraySubscript(LdiffExpr, m_IdxExpr);
        auto* paramClone = m_RMV->BuildDeclRef(decl);
        auto* LRepl = m_RMV->BuildArraySubscript(paramClone, m_IdxExpr);
        // Build the loop to put in reverse mode.
        Expr* errorExpr = GetError(LRepl, Ldiff, params[i]->getNameAsString());
        Expr* finalAssignExpr =
            m_RMV->BuildOp(BO_AddAssign, BuildFinalErrorExpr(), errorExpr);
        Expr* conditionExpr = m_RMV->BuildOp(BO_LE, m_IdxExpr, size);
        Expr* incExpr = m_RMV->BuildOp(UO_PostInc, m_IdxExpr);
        Stmt* ArrayParamLoop = new (m_RMV->m_Context)
            ForStmt(m_RMV->m_Context, nullptr, conditionExpr, nullptr, incExpr,
                    finalAssignExpr, noLoc, noLoc, noLoc);
        // For multiple array parameters, we want to keep the same
        // iterative variable, so reset that here in the case that this
        // is not out first array.
        if (!idxExprDecl) {
          m_RMV->addToCurrentBlock(
              m_RMV->BuildOp(BO_Assign, m_IdxExpr,
                             m_RMV->getZeroInit(m_RMV->m_Context.IntTy)));
        } else {
          m_RMV->addToCurrentBlock(m_RMV->BuildDeclStmt(idxExprDecl));
        }
        m_RMV->addToCurrentBlock(ArrayParamLoop);
      }
    }
  }
  BuildReturnErrorStmt();
}

void ErrorEstimationHandler::EmitUnaryOpErrorStmts(StmtDiff var,
                                                   bool isInsideLoop) {
  // If the sub-expression is a declRefExpr, we should emit an error.
  if (DeclRefExpr* DRE = GetUnderlyingDeclRefOrNull(var.getExpr())) {
    // First check if it was registered.
    // If not, we don't care about it.
    if (ShouldEstimateErrorFor(cast<VarDecl>(DRE->getDecl()))) {
      Expr* erroExpr =
          GetError(DRE, var.getExpr_dx(), DRE->getDecl()->getNameAsString());
      AddErrorStmtToBlock(erroExpr);
    }
  }
}

void ErrorEstimationHandler::EmitBinaryOpErrorStmts(Expr* LExpr,
                                                    Expr* oldValue) {
  // Assign the error.
  auto decl = GetUnderlyingDeclRefOrNull(LExpr)->getDecl();
  if (!ShouldEstimateErrorFor(cast<VarDecl>(decl)))
    return;
  if (m_ErrorFromFunctionCall)
    return;
  Expr* errorExpr = GetError(LExpr, oldValue, decl->getNameAsString());
  AddErrorStmtToBlock(errorExpr);
  // If there are assign statements to emit in reverse, do that.
  EmitErrorEstimationStmts(direction::reverse);
}

void ErrorEstimationHandler::EmitDeclErrorStmts(DeclDiff<VarDecl> VDDiff,
                                                bool isInsideLoop) {
  auto VD = VDDiff.getDecl();
  if (!ShouldEstimateErrorFor(VD))
    return;
  if (m_ErrorFromFunctionCall)
    return;
  // Build the delta expresion for the variable to be registered.
  DeclRefExpr* VDRef = m_RMV->BuildDeclRef(VD);
  // FIXME: We should do this for arrays too.
  if (!VD->getType()->isArrayType()) {
    // If the VarDecl has an init, we should assign it with an error.
    if (VD->getInit() && !GetUnderlyingDeclRefOrNull(VD->getInit())) {
      Expr* errorExpr =
          GetError(VDRef, m_RMV->BuildDeclRef(VDDiff.getDecl_dx()),
                   VD->getNameAsString());
      AddErrorStmtToBlock(errorExpr);
    }
  }
}

void ErrorEstimationHandler::InitialiseRMV(ReverseModeVisitor& RMV) {
  m_RMV = &RMV;
  LookupCustomErrorFunction();
}

void ErrorEstimationHandler::ForgetRMV() { m_RMV = nullptr; }

void ErrorEstimationHandler::ActBeforeCreatingDerivedFnParamTypes(
    unsigned& numExtraParam) {
  numExtraParam += 1;
}

void ErrorEstimationHandler::ActAfterCreatingDerivedFnParams(
    llvm::SmallVectorImpl<ParmVarDecl*>& params) {
  m_Params = &params;
  // If in error estimation mode, create the error parameter
  ASTContext& C = m_RMV->m_Context;
  // Repeat the above but for the error ouput var "_final_error"
  QualType LastParamTy = C.getLValueReferenceType(C.DoubleTy);
  ParmVarDecl* errorVarDecl = ParmVarDecl::Create(
      C, m_RMV->m_Derivative, noLoc, noLoc, &C.Idents.get("_final_error"),
      LastParamTy, C.getTrivialTypeSourceInfo(LastParamTy, noLoc),
      params.front()->getStorageClass(), /*DefArg=*/nullptr);
  params.push_back(errorVarDecl);
  m_RMV->m_Sema.PushOnScopeChains(params.back(), m_RMV->getCurrentScope(),
                                  /*AddToContext=*/false);
}

void ErrorEstimationHandler::ActOnEndOfDerivedFnBody() {
  // Since 'return' is not an assignment, add its error to _final_error
  // given it is not a DeclRefExpr.
  EmitFinalErrorStmts(*m_Params, m_RMV->m_DiffReq->getNumParams());
}

void ErrorEstimationHandler::ActBeforeDifferentiatingStmtInVisitCompoundStmt() {
  m_ShouldEmit.push(true);
}

void ErrorEstimationHandler::ActAfterProcessingStmtInVisitCompoundStmt() {
  // In error estimation mode, if we have any residual statements
  // to be emitted into the forward or revese blocks, we should
  // emit them here. This is to maintain the correct order of
  // statements generated.
  EmitErrorEstimationStmts(direction::forward);
  EmitErrorEstimationStmts(direction::reverse);
}

void ErrorEstimationHandler::ActAfterProcessingArraySubscriptExpr(
    const Expr* revArrSub) {
  if (const auto* ASE = dyn_cast<ArraySubscriptExpr>(revArrSub)) {
    if (const auto* DRE =
            dyn_cast<DeclRefExpr>(ASE->getBase()->IgnoreImplicit())) {
      const auto* VD = cast<VarDecl>(DRE->getDecl());
      Expr* VDdiff = m_RMV->m_Variables[VD];
      // We only need to track sizes for arrays and pointers.
      if (!utils::isArrayOrPointerType(VDdiff->getType()))
        return;

      // We only need to know the size of independent arrays.
      auto params = m_RMV->m_Derivative->parameters();
      if (llvm::find(params, VD) == params.end())
        return;

      // Construct `var_size = max(var_size, idx);`
      // to update `var_size` to the biggest index used if necessary.
      Expr* size = getSizeExpr(VD);
      Expr* idx = m_RMV->Clone(ASE->getIdx());
      idx =
          m_RMV->m_Sema.ImpCastExprToType(idx, size->getType(), CK_IntegralCast)
              .get();
      llvm::SmallVector<clang::Expr*, 2> args{size, idx};
      Expr* extendedSize = m_RMV->GetFunctionCall("max", "std", args);
      size = m_RMV->Clone(size);
      Stmt* updateSize = m_RMV->BuildOp(BO_Assign, size, extendedSize);
      m_RMV->addToCurrentBlock(updateSize, direction::reverse);
    }
  }
}

Expr* ErrorEstimationHandler::getSizeExpr(const VarDecl* VD) {
  // For every array/pointer variable `arr`
  // we create `arr_size` to track the size.
  auto foundSize = m_ArrSizes.find(VD);
  // If the size variable is already generated, just clone the decl ref.
  if (foundSize != m_ArrSizes.end())
    return m_RMV->Clone(foundSize->second);
  // If the size variable is not generated yet,
  // generate it now.
  QualType intTy = m_RMV->m_Context.getSizeType();
  VarDecl* sizeVD = m_RMV->BuildGlobalVarDecl(
      intTy, VD->getNameAsString() + "_size", m_RMV->getZeroInit(intTy));
  m_RMV->AddToGlobalBlock(m_RMV->BuildDeclStmt(sizeVD));
  Expr* size = m_RMV->BuildDeclRef(sizeVD);
  m_ArrSizes[VD] = size;
  return size;
}

void ErrorEstimationHandler::
    ActBeforeDifferentiatingSingleStmtBranchInVisitIfStmt() {
  m_ShouldEmit.push(true);
}

void ErrorEstimationHandler::
    ActBeforeFinalizingVisitBranchSingleStmtInIfVisitStmt() {
  // In error estimation, manually emit the code here instead of
  // DifferentiateSingleStmt to maintain correct order.
  EmitErrorEstimationStmts(direction::forward);
}

void ErrorEstimationHandler::ActBeforeDifferentiatingLoopInitStmt() {
  m_ShouldEmit.push(true);
}

void ErrorEstimationHandler::ActBeforeDifferentiatingSingleStmtLoopBody() {
  m_ShouldEmit.push(false);
}

void ErrorEstimationHandler::ActAfterProcessingSingleStmtBodyInVisitForLoop() {
  // Emit some statemnts later to maintain correct statement order.
  EmitErrorEstimationStmts(direction::forward);
}

void ErrorEstimationHandler::ActBeforeFinalizingVisitReturnStmt(
    StmtDiff& retExprDiff) {
  // If the return expression is not a DeclRefExpression and is of type
  // float, we should add it to the error estimate because returns are
  // similiar to implicit assigns.
  SaveReturnExpr(retExprDiff.getExpr());
}

void ErrorEstimationHandler::ActBeforeFinalizingPostIncDecOp(StmtDiff& diff) {
  EmitUnaryOpErrorStmts(diff, m_RMV->isInsideLoop);
}

// FIXME: Issue a warning that error estimation may produce incorrect result if
// any of the arguments are being passed by reference to the call expression
// `CE`.
void ErrorEstimationHandler::ActBeforeFinalizingVisitCallExpr(
    const clang::CallExpr*& CE, clang::Expr*& OverloadedDerivedFn,
    llvm::SmallVectorImpl<Expr*>& derivedCallArgs,
    llvm::SmallVectorImpl<Expr*>& ArgResult, bool asGrad) {
  if (OverloadedDerivedFn && asGrad) {
    // Derivative was found.
    FunctionDecl* fnDecl =
        dyn_cast<CallExpr>(OverloadedDerivedFn)->getDirectCallee();

    // If in error estimation, build the statement for the error
    // in the input prameters (if of reference type) to call and save to
    // emit them later.

    EmitNestedFunctionParamError(fnDecl, derivedCallArgs, ArgResult,
                                 CE->getNumArgs());
  }
}

void ErrorEstimationHandler::ActBeforeFinalizingAssignOp(
    clang::Expr*& LCloned, clang::Expr*& oldValue, clang::Expr*& R,
    clang::BinaryOperator::Opcode& opCode) {
  DeclRefExpr* RRef = GetUnderlyingDeclRefOrNull(R);
  // In the case that an RHS expression is a declReference, we do not emit
  // any error because the assignment operation entials zero error.
  // However, for compound assignment operators, the RHS may be a
  // declRefExpr but here we will need to emit its error.
  // This checks for the above conditions.
  if (opCode != BO_Assign || !RRef)
    EmitBinaryOpErrorStmts(LCloned, oldValue);
}

void ErrorEstimationHandler::ActBeforeFinalizingDifferentiateSingleStmt(
    const direction& d) {
  // We might have some expressions to emit, so do that here.
  if (m_ShouldEmit.top())
    EmitErrorEstimationStmts(d);
  m_ShouldEmit.pop();
}

void ErrorEstimationHandler::ActBeforeFinalizingDifferentiateSingleExpr(
    const direction& d) {
  // We might have some expressions to emit, so do that here.
  EmitErrorEstimationStmts(d);
}

void ErrorEstimationHandler::ActBeforeDifferentiatingCallExpr(
    llvm::SmallVectorImpl<clang::Expr*>& pullbackArgs) {
  m_ErrorFromFunctionCall = true;
  pullbackArgs.push_back(BuildFinalErrorExpr());
}

void ErrorEstimationHandler::LookupCustomErrorFunction() {
  Sema& S = m_RMV->m_Sema;
  ASTContext& C = m_RMV->m_Context;
  NamespaceDecl* cladNS = utils::LookupNSD(S, "clad", /*shouldExist=*/true);
  IdentifierInfo* II = &C.Idents.get("getErrorVal");
  DeclarationNameInfo DNInfo(DeclarationName(II), utils::GetValidSLoc(S));
  LookupResult R(S, DNInfo, Sema::LookupOrdinaryName);
  S.LookupQualifiedName(R, cladNS);
  if (R.empty())
    return;

  FunctionProtoType::ExtProtoInfo EPI;
  QualType ConstCharPtr = C.getPointerType(C.getConstType(C.CharTy));
  QualType DoubleTy = C.DoubleTy;
  llvm::SmallVector<QualType, 3> FnTypes = {DoubleTy, DoubleTy, ConstCharPtr};
  QualType FnTy = C.getFunctionType(DoubleTy, FnTypes, EPI);
  TemplateSpecCandidateSet FailedCandidates(utils::GetValidSLoc(S),
                                            /*ForTakingAddress=*/false);
  if (utils::MatchOverloadType(S, FnTy, R, FailedCandidates)) {
    // FIXME: MatchOverloadType returns an overload expr without the `clad::`
    // namespace specifier. Here, we rebuild manually.
    CXXScopeSpec SS;
    SS.Extend(C, cladNS, noLoc, noLoc);
    m_CustomErrorFunction =
        S.BuildDeclarationNameExpr(SS, R, /*ADL=*/false).get();
    return;
  }

  // We did not match the found candidates. Warn and offer the user hints.
  auto errId = S.Diags.getCustomDiagID(
      DiagnosticsEngine::Error,
      "user-defined derivative error function was provided but not used; "
      "expected signature %0 does not match");
  S.Diag(m_RMV->m_DiffReq->getLocation(), errId) << FnTy;
  FailedCandidates.NoteCandidates(S, utils::GetValidSLoc(S));
  utils::DiagnoseSignatureMismatch(S, FnTy, R);
}

Expr* ErrorEstimationHandler::AssignError(StmtDiff refExpr,
                                          const std::string& varName) {
  Sema& S = m_RMV->m_Sema;
  ASTContext& C = m_RMV->m_Context;
  if (m_CustomErrorFunction) {
    llvm::SmallVector<clang::Expr*, 3> callParams{
        refExpr.getExpr_dx(), refExpr.getExpr(),
        clad::utils::CreateStringLiteral(C, varName)};
    return S
        .ActOnCallExpr(m_RMV->getCurrentScope(), m_CustomErrorFunction, noLoc,
                       callParams, noLoc)
        .get();
  }
  // Get the machine epsilon value.
  double val = std::numeric_limits<float>::epsilon();
  // Convert it into a floating point literal clang::Expr.
  Expr* epsExpr =
      FloatingLiteral::Create(C, llvm::APFloat(val), true, C.DoubleTy, noLoc);
  // Here, we first build a multiplication operation for the following:
  // refExpr * <--floating point literal (i.e. machine dependent constant)-->
  // Build another multiplication operation with above and the derivative
  Expr* errExpr =
      m_RMV->BuildOp(BO_Mul, refExpr.getExpr_dx(),
                     m_RMV->BuildOp(BO_Mul, refExpr.getExpr(), epsExpr));
  // Next, build a llvm vector-like container to store the parameters
  // of the function call.
  llvm::SmallVector<Expr*, 1> params{errExpr};
  // Finally, build a call to std::abs
  Expr* absExpr = m_RMV->GetFunctionCall("abs", "std", params);
  // Return the built error expression.
  return absExpr;
}

void ErrorEstimationHandler::ActBeforeFinalizingVisitDeclStmt(
    llvm::SmallVectorImpl<Decl*>& decls,
    llvm::SmallVectorImpl<Decl*>& declsDiff) {
  // For all dependent variables, we register them for estimation
  // here.
  for (size_t i = 0; i < decls.size(); i++) {
    DeclDiff<VarDecl> VDDiff(cast<VarDecl>(decls[0]),
                             cast<VarDecl>(declsDiff[0]));
    EmitDeclErrorStmts(VDDiff, m_RMV->isInsideLoop);
  }
}
} // namespace clad
