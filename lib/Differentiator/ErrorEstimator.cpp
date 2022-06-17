#include "clad/Differentiator/ErrorEstimator.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/DerivativeBuilder.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/ReverseModeVisitor.h"

#include "clang/AST/Decl.h"

using namespace clang;

namespace clad {

QualType getUnderlyingArrayType(QualType baseType, ASTContext& C) {
  if (baseType->isArrayType()) {
    return C.getBaseElementType(baseType);
  } else if (auto PTType = baseType->getAs<PointerType>()) {
    return PTType->getPointeeType();
  }
  assert(0 && "Unreachable");
  return {};
}

Expr* UpdateErrorForFuncCallAssigns(ErrorEstimationHandler* handler,
                                    Expr* savedExpr, Expr* origExpr,
                                    Expr*& callError) {
  Expr* errorExpr = nullptr;
  if (!callError)
    errorExpr = handler->GetError(savedExpr, origExpr);
  else {
    errorExpr = callError;
    callError = nullptr;
  }
  return errorExpr;
}

void ErrorEstimationHandler::SetErrorEstimationModel(
    FPErrorEstimationModel* estModel) {
  m_EstModel = estModel;
}

Expr* ErrorEstimationHandler::getArraySubscriptExpr(
    Expr* arrBase, Expr* idx, bool isCladSpType /*=true*/) {
  if (isCladSpType) {
    return m_RMV->m_Sema
        .ActOnArraySubscriptExpr(m_RMV->getCurrentScope(), arrBase,
                                 arrBase->getExprLoc(), idx, noLoc)
        .get();
  } else {
    return m_RMV->m_Sema
        .CreateBuiltinArraySubscriptExpr(arrBase, noLoc, idx, noLoc)
        .get();
  }
}

void ErrorEstimationHandler::BuildFinalErrorStmt() {
  Expr* finExpr = nullptr;
  // If we encountered any arithmetic expression in the return statement,
  // we must add its error to the final estimate.
  if (m_RetErrorExpr) {
    auto flitr =
        FloatingLiteral::Create(m_RMV->m_Context, llvm::APFloat(1.0), true,
                                m_RMV->m_Context.DoubleTy, noLoc);
    finExpr = m_EstModel->AssignError(StmtDiff(m_RetErrorExpr, flitr));
  }

  // Build the final error statement with the sum of all _delta_*.
  Expr* addErrorExpr = m_EstModel->CalculateAggregateError();
  if (addErrorExpr) {
    if (finExpr)
      addErrorExpr = m_RMV->BuildOp(BO_Add, addErrorExpr, finExpr);
  } else if (finExpr) {
    addErrorExpr = finExpr;
  }

  // Finally add the final error expression to the derivative body.
  // Here, since this is the final error, we do not print it when error
  // printing is requested, users can print this error themselves if they
  // so feel to.
  m_RMV->addToCurrentBlock(
      m_RMV->BuildOp(BO_AddAssign, m_FinalError, addErrorExpr),
      direction::forward);

  if (m_ErrorFile) {
    for (auto var : m_EstModel->m_EstimateVar) {
      auto delta = var.second;
      auto deltaDecl = var.first;
      Expr* printDelta = m_RMV->BuildOp(
          BO_Shl, m_ErrorFile,
          m_EstModel->getAsExpr("\nFinal error contribution by " +
                                deltaDecl->getNameAsString() + " = "));
      printDelta = m_RMV->BuildOp(BO_Shl, printDelta, delta);
      m_RMV->addToCurrentBlock(m_RMV->BuildOp(BO_Shl, printDelta, m_EstModel->getAsExpr("\n")),
                               direction::forward);
    }
  }
}

void ErrorEstimationHandler::AddErrorStmtToBlock(
    Expr* var, Expr* deltaVar, Expr* errorExpr, bool isInsideLoop /*=false*/,
    Expr* errorPrint /*=nullptr*/) {
  if (auto ASE = dyn_cast<ArraySubscriptExpr>(var)) {
    // If inside loop, the index has been pushed twice
    // (once by ArraySubscriptExpr and the second time by us)
    // pop and store it in a temporary variable to reuse later.
    // FIXME: build add assign into he same expression i.e.
    // _final_error += _delta_arr[pop(_t0)] += <-Error Expr->
    // to avoid storage of the pop value.
    Expr* popVal = ASE->getIdx();
    if (isInsideLoop) {
      LookupResult& Pop = m_RMV->GetCladTapePop();
      CXXScopeSpec CSS;
      CSS.Extend(m_RMV->m_Context, m_RMV->GetCladNamespace(), noLoc, noLoc);
      auto PopDRE = m_RMV->m_Sema
                        .BuildDeclarationNameExpr(CSS, Pop,
                                                  /*AcceptInvalidDecl=*/false)
                        .get();
      Expr* tapeRef = dyn_cast<CallExpr>(popVal)->getArg(0);
      popVal = m_RMV->m_Sema
                   .ActOnCallExpr(m_RMV->getCurrentScope(), PopDRE, noLoc,
                                  tapeRef, noLoc)
                   .get();
      popVal = m_RMV->StoreAndRef(popVal, direction::reverse);
    }
    // If the variable declration refers to an array element
    // create the suitable _delta_arr[i] (because we have not done
    // this before).
    deltaVar = getArraySubscriptExpr(deltaVar, popVal);
    m_RMV->addToCurrentBlock(m_RMV->BuildOp(BO_AddAssign, deltaVar, errorExpr),
                             direction::reverse);
    // immediately emit fin_err += delta_[].
    // This is done to avoid adding all errors at the end
    // and only add the errors that were calculated.
    m_RMV->addToCurrentBlock(
        m_RMV->BuildOp(BO_AddAssign, m_FinalError, deltaVar),
        direction::reverse);
  } else
    m_RMV->addToCurrentBlock(m_RMV->BuildOp(BO_AddAssign, deltaVar, errorExpr),
                             direction::reverse);
  // Add the print error statement if any printing was requested.
  m_RMV->addToCurrentBlock(errorPrint, direction::reverse);
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

void ErrorEstimationHandler::SaveReturnExpr(Expr* retExpr,
                                            DeclRefExpr* retDeclRefExpr) {
  // If the return expression is a declRefExpr or is a non-floating point
  // type, we should not do anything.
  if (GetUnderlyingDeclRefOrNull(retExpr) ||
      !retDeclRefExpr->getType()->isFloatingType())
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
  m_RMV->addToCurrentBlock(
      m_RMV->BuildOp(BO_Assign, m_RetErrorExpr, retDeclRefExpr),
      direction::forward);
}

void ErrorEstimationHandler::EmitNestedFunctionParamError(
    FunctionDecl* fnDecl, llvm::SmallVectorImpl<Expr*>& derivedCallArgs,
    llvm::SmallVectorImpl<VarDecl*>& ArgResultDecls, size_t numArgs) {
  assert(fnDecl && "Must have a value");
  for (size_t i = 0; i < numArgs; i++) {
    if (!fnDecl->getParamDecl(0)->getType()->isLValueReferenceType())
      continue;
    // // FIXME: Argument passed by reference do not have any corresponding
    // // `ArgResultDecl`. Handle arguments passed by reference in error
    // // estimation.
    // if (utils::IsReferenceOrPointerType(fnDecl->getParamDecl(i)->getType()))
    //   continue;
    Expr* errorExpr = m_EstModel->AssignError(
        {derivedCallArgs[i], m_RMV->BuildDeclRef(ArgResultDecls[i])});
    Expr* errorStmt = m_RMV->BuildOp(BO_AddAssign, m_FinalError, errorExpr);
    m_ReverseErrorStmts.push_back(errorStmt);
  }
}

StmtDiff ErrorEstimationHandler::SaveValue(Expr* val,
                                           bool isInsideLoop /*=false*/) {
  // Definite not null.
  DeclRefExpr* declRefVal = GetUnderlyingDeclRefOrNull(val);
  assert(declRefVal && "Val cannot be null!");
  std::string name = "_EERepl_" + declRefVal->getDecl()->getNameAsString();
  if (isInsideLoop) {
    auto tape = m_RMV->MakeCladTapeFor(val, name);
    m_ForwardReplStmts.push_back(tape.Push);
    // Nice to store pop values becuase user might refer to getExpr
    // multiple times in Assign Error.
    Expr* popVal = m_RMV->StoreAndRef(tape.Pop, direction::reverse);
    return StmtDiff(tape.Push, popVal);
  } else {
    QualType QTval = val->getType();
    if (auto AType = dyn_cast<ArrayType>(QTval))
      QTval = AType->getElementType();

    auto savedVD = m_RMV->GlobalStoreImpl(QTval, name);
    auto savedRef = m_RMV->BuildDeclRef(savedVD);
    m_ForwardReplStmts.push_back(m_RMV->BuildOp(BO_Assign, savedRef, val));
    return StmtDiff(savedRef, savedRef);
  }
}

void ErrorEstimationHandler::SaveParamValue(DeclRefExpr* paramRef) {
  assert(paramRef && "Must have a value");
  VarDecl* paramDecl = cast<VarDecl>(paramRef->getDecl());
  QualType paramType = paramRef->getType();
  std::string name = "_EERepl_" + paramDecl->getNameAsString();
  VarDecl* savedDecl;
  if (utils::isArrayOrPointerType(paramType)) {
    auto diffVar = m_RMV->m_Variables[paramDecl];
    auto QType = m_RMV->GetCladArrayOfType(
        getUnderlyingArrayType(paramType, m_RMV->m_Context));
    savedDecl = m_RMV->BuildVarDecl(
        QType, name, m_RMV->BuildArrayRefSizeExpr(diffVar),
        /*DirectInit=*/false,
        /*TSI=*/nullptr, VarDecl::InitializationStyle::CallInit);
    m_RMV->AddToGlobalBlock(m_RMV->BuildDeclStmt(savedDecl));
    ReverseModeVisitor::Stmts loopBody;
    // Get iter variable.
    auto loopIdx =
        m_RMV->BuildVarDecl(m_RMV->m_Context.IntTy, "i",
                            m_RMV->getZeroInit(m_RMV->m_Context.IntTy));
    auto currIdx = m_RMV->BuildDeclRef(loopIdx);
    // Build the assign expression.
    loopBody.push_back(m_RMV->BuildOp(
        BO_Assign,
        getArraySubscriptExpr(m_RMV->BuildDeclRef(savedDecl), currIdx),
        getArraySubscriptExpr(paramRef, currIdx,
                              /*isCladSpType=*/false)));
    Expr* conditionExpr =
        m_RMV->BuildOp(BO_LT, currIdx, m_RMV->BuildArrayRefSizeExpr(diffVar));
    Expr* incExpr = m_RMV->BuildOp(UO_PostInc, currIdx);
    // Make for loop.
    Stmt* ArrayParamLoop = new (m_RMV->m_Context) ForStmt(
        m_RMV->m_Context, m_RMV->BuildDeclStmt(loopIdx), conditionExpr, nullptr,
        incExpr, m_RMV->MakeCompoundStmt(loopBody), noLoc, noLoc, noLoc);
    m_RMV->AddToGlobalBlock(ArrayParamLoop);
  } else
    savedDecl = m_RMV->GlobalStoreImpl(paramType, name, paramRef);
  m_ParamRepls.emplace(paramDecl, m_RMV->BuildDeclRef(savedDecl));
}

Expr* ErrorEstimationHandler::RegisterVariable(VarDecl* VD,
                                               bool toCurrentScope /*=false*/) {
  if (!CanRegisterVariable(VD))
    return nullptr;
  // Get the init error from setError.
  Expr* init = m_EstModel->SetError(VD);
  auto VDType = VD->getType();
  // The type of the _delta_ value should be customisable.
  QualType QType;
  Expr* deltaVar = nullptr;
  auto diffVar = m_RMV->m_Variables[VD];
  if (m_RMV->isCladArrayType(diffVar->getType())) {
    VarDecl* EstVD;
    auto sizeExpr = m_RMV->BuildArrayRefSizeExpr(diffVar);
    QType = m_RMV->GetCladArrayOfType(
        getUnderlyingArrayType(VDType, m_RMV->m_Context));
    EstVD = m_RMV->BuildVarDecl(
        QType, "_delta_" + VD->getNameAsString(), sizeExpr,
        /*DirectInit=*/false,
        /*TSI=*/nullptr, VarDecl::InitializationStyle::CallInit);
    if (!toCurrentScope)
      m_RMV->AddToGlobalBlock(m_RMV->BuildDeclStmt(EstVD));
    else
      m_RMV->addToCurrentBlock(m_RMV->BuildDeclStmt(EstVD), direction::forward);
    deltaVar = m_RMV->BuildDeclRef(EstVD);
  } else {
    QType = utils::isArrayOrPointerType(VDType) ? VDType
                                                : m_RMV->m_Context.DoubleTy;
    init = init ? init : m_RMV->getZeroInit(QType);
    // Store the "_delta_*" value.
    if (!toCurrentScope) {
      auto EstVD = m_RMV->GlobalStoreImpl(
          QType, "_delta_" + VD->getNameAsString(), init);
      deltaVar = m_RMV->BuildDeclRef(EstVD);
    } else {
      deltaVar = m_RMV->StoreAndRef(init, QType, direction::forward,
                                    "_delta_" + VD->getNameAsString(),
                                    /*forceDeclCreation=*/true);
    }
  }
  // Register the variable for estimate calculation.
  m_EstModel->AddVarToEstimate(VD, deltaVar);
  return deltaVar;
}

bool ErrorEstimationHandler::CanRegisterVariable(VarDecl* VD) {

  // Get the types on the declartion and initalization expression.
  QualType varDeclBase = VD->getType();
  QualType varDeclType =
      utils::isArrayOrPointerType(varDeclBase)
          ? getUnderlyingArrayType(varDeclBase, m_RMV->m_Context)
          : varDeclBase;
  const Expr* init = VD->getInit();
  // If declaration type in not floating point type, we want to do two
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
                  "Lossy assignment from '%0' to '%1', this error will not be "
                  "taken into cosideration while estimation",
                  {init->IgnoreImpCasts()->getType().getAsString(),
                   varDeclBase.getAsString()});
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

Expr* ErrorEstimationHandler::GetParamReplacement(const ParmVarDecl* VD) {
  auto it = m_ParamRepls.find(VD);
  if (it != m_ParamRepls.end())
    return it->second;
  return nullptr;
}

Expr* ErrorEstimationHandler::GetPrintExpr(std::string var_name, Expr* var,
                                           Expr* var_dx, Expr* var_err) {
  if (!m_ErrorFile)
    return nullptr;
  llvm::SmallVector<Expr*, 8> toPrnt = {};
  m_EstModel->Print(var_name, {var, var_dx}, var_err, toPrnt);
  assert(toPrnt.size() &&
         "To print expression is empty even when error printing is enabled.");
  Expr* printExpr = m_ErrorFile;
  for (size_t i = 0; i < toPrnt.size(); i++) {
    printExpr = m_RMV->BuildOp(BO_Shl, printExpr, toPrnt[i]);
  }
  return printExpr;
}

void ErrorEstimationHandler::EmitFinalErrorStmts(
    llvm::SmallVectorImpl<ParmVarDecl*>& params, unsigned numParams) {
  // Emit error variables of parameters at the end.
  for (size_t i = 0; i < numParams; i++) {
    // Right now, we just ignore them since we have no way of knowing
    // the size of an array.
    // if (isArrayOrPointerType(params[i]->getType()))
    //   continue;
    // Check if the declaration was registered
    auto decl = dyn_cast<VarDecl>(params[i]);
    Expr* deltaVar = IsRegistered(decl);
    // If not registered, check if it is eligible for registration and do
    // the needful.
    if (!deltaVar) {
      deltaVar = RegisterVariable(decl, /*toCurrentScope=*/true);
    }
    // If till now, we have a delta declaration, emit it into the code.
    if (deltaVar) {
      if (!clad::utils::isArrayOrPointerType(params[i]->getType())) {
        // Since we need the input value of x, check for a replacement.
        // If no replacement found, use the actual declRefExpr.
        auto savedVal = GetParamReplacement(params[i]);
        savedVal = savedVal ? savedVal : m_RMV->BuildDeclRef(decl);
        // Finally emit the error.
        auto errorExpr = GetError(savedVal, m_RMV->m_Variables[decl]);
        auto errorPrntExpr =
            GetPrintExpr(params[i]->getNameAsString(), savedVal,
                         m_RMV->m_Variables[decl], errorExpr);
        m_RMV->addToCurrentBlock(
            m_RMV->BuildOp(BO_AddAssign, deltaVar, errorExpr));
        m_RMV->addToCurrentBlock(errorPrntExpr);
      } else {
        auto LdiffExpr = m_RMV->m_Variables[decl];
        VarDecl* idxExprDecl = nullptr;
        // Save our index expression so it can be used later.
        if (!m_IdxExpr) {
          idxExprDecl =
              m_RMV->BuildVarDecl(m_RMV->m_Context.IntTy, "i",
                                  m_RMV->getZeroInit(m_RMV->m_Context.IntTy));
          m_IdxExpr = m_RMV->BuildDeclRef(idxExprDecl);
        }
        Expr *Ldiff, *Ldelta;
        Ldiff = getArraySubscriptExpr(
            LdiffExpr, m_IdxExpr, m_RMV->isCladArrayType(LdiffExpr->getType()));
        Ldelta = getArraySubscriptExpr(deltaVar, m_IdxExpr);
        auto savedVal = GetParamReplacement(params[i]);
        savedVal = savedVal ? savedVal : m_RMV->BuildDeclRef(decl);
        auto LRepl = getArraySubscriptExpr(savedVal, m_IdxExpr);
        // Build the loop to put in reverse mode.
        Expr* errorExpr = GetError(LRepl, Ldiff);
        auto commonVarDecl =
            m_RMV->BuildVarDecl(errorExpr->getType(), "_t", errorExpr);
        Expr* commonVarExpr = m_RMV->BuildDeclRef(commonVarDecl);
        Expr* deltaAssignExpr =
            m_RMV->BuildOp(BO_AddAssign, Ldelta, commonVarExpr);
        Expr* finalAssignExpr =
            m_RMV->BuildOp(BO_AddAssign, m_FinalError, commonVarExpr);
        Stmts loopBody;
        loopBody.push_back(m_RMV->BuildDeclStmt(commonVarDecl));
        loopBody.push_back(deltaAssignExpr);
        // Build and add the print error expression.
        if (m_ErrorFile)
          loopBody.push_back(GetPrintExpr(params[i]->getNameAsString(), LRepl,
                                          Ldiff, commonVarExpr));
        loopBody.push_back(finalAssignExpr);
        Expr* conditionExpr = m_RMV->BuildOp(
            BO_LT, m_IdxExpr, m_RMV->BuildArrayRefSizeExpr(LdiffExpr));
        Expr* incExpr = m_RMV->BuildOp(UO_PostInc, m_IdxExpr);
        Stmt* ArrayParamLoop = new (m_RMV->m_Context)
            ForStmt(m_RMV->m_Context, nullptr, conditionExpr, nullptr, incExpr,
                    m_RMV->MakeCompoundStmt(loopBody), noLoc, noLoc, noLoc);
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
  BuildFinalErrorStmt();
}

void ErrorEstimationHandler::EmitUnaryOpErrorStmts(StmtDiff var,
                                                   bool isInsideLoop) {
  // If the sub-expression is a declRefExpr, we should emit an error.
  if (DeclRefExpr* DRE = GetUnderlyingDeclRefOrNull(var.getExpr())) {
    // First check if it was registered.
    // If not, we don't care about it.
    auto decl = cast<VarDecl>(DRE->getDecl());
    if (auto deltaVar = IsRegistered(decl)) {
      // Create a variable/tape call to store the current value of the
      // the sub-expression so that it can be used later.
      StmtDiff savedVar = m_RMV->GlobalStoreAndRef(
          DRE, "_EERepl_" + DRE->getDecl()->getNameAsString());
      if (isInsideLoop) {
        // It is nice to save the pop value.
        // We do not know how many times the user will use dx,
        // hence we should pop values beforehand to avoid unequal pushes
        // and and pops.
        Expr* popVal =
            m_RMV->StoreAndRef(savedVar.getExpr_dx(), direction::reverse);
        savedVar = {savedVar.getExpr(), popVal};
      }
      Expr* erroExpr = GetError(savedVar.getExpr_dx(), var.getExpr_dx());
      Expr* prntExpr =
          GetPrintExpr(decl->getNameAsString(), savedVar.getExpr_dx(),
                       var.getExpr_dx(), erroExpr);
      AddErrorStmtToBlock(var.getExpr(), deltaVar, erroExpr, isInsideLoop,
                          prntExpr);
    }
  }
}

Expr* ErrorEstimationHandler::RegisterBinaryOpLHS(Expr* LExpr, Expr* RExpr,
                                                  bool isAssign) {
  DeclRefExpr* LRef = GetUnderlyingDeclRefOrNull(LExpr);
  DeclRefExpr* RRef = GetUnderlyingDeclRefOrNull(RExpr);
  VarDecl* Ldecl = LRef ? dyn_cast<VarDecl>(LRef->getDecl()) : nullptr;
  // In the case that an RHS expression is a declReference, we do not emit
  // any error because the assignment operation entials zero error.
  // However, for compound assignment operators, the RHS may be a
  // declRefExpr but here we will need to emit its error.
  // This variable checks for the above conditions.
  bool declRefOk = !RRef || !isAssign;
  Expr* deltaVar = nullptr;
  // If the LHS can be decayed to a VarDecl and all other requirements
  // are met, we should register the variable if it has not been already.
  // We also do not support array input types yet.
  if (Ldecl && declRefOk) {
    deltaVar = IsRegistered(Ldecl);
    // Usually we would expect independent variable to qualify for these
    // checks.
    if (!deltaVar) {
      deltaVar = RegisterVariable(Ldecl);
      SaveParamValue(LRef);
    }
  }
  return deltaVar;
}

void ErrorEstimationHandler::EmitBinaryOpErrorStmts(Expr* LExpr, Expr* oldValue,
                                                    Expr* deltaVar,
                                                    bool isInsideLoop) {
  if (!deltaVar)
    return;
  // For now save all lhs.
  // FIXME: We can optimize stores here by using the ones created
  // previously.
  StmtDiff savedExpr = SaveValue(LExpr, isInsideLoop);
  // Assign the error.
  auto decl = GetUnderlyingDeclRefOrNull(LExpr)->getDecl();
  Expr* errorExpr = UpdateErrorForFuncCallAssigns(this, savedExpr.getExpr_dx(),
                                                  oldValue, m_NestedFuncError);
  Expr* prntExpr = GetPrintExpr(decl->getNameAsString(), savedExpr.getExpr_dx(),
                                oldValue, errorExpr);
  AddErrorStmtToBlock(LExpr, deltaVar, errorExpr, isInsideLoop, prntExpr);
  // If there are assign statements to emit in reverse, do that.
  EmitErrorEstimationStmts(direction::reverse);
}

void ErrorEstimationHandler::EmitDeclErrorStmts(VarDeclDiff VDDiff,
                                                bool isInsideLoop) {
  auto VD = VDDiff.getDecl();
  if (!CanRegisterVariable(VD))
    return;
  // Build the delta expresion for the variable to be registered.
  auto EstVD = RegisterVariable(VD);
  DeclRefExpr* VDRef = m_RMV->BuildDeclRef(VD);
  // FIXME: We should do this for arrays too.
  if (!VD->getType()->isArrayType()) {
    StmtDiff savedDecl = SaveValue(VDRef, isInsideLoop);
    // If the VarDecl has an init, we should assign it with an error.
    if (VD->getInit() && !GetUnderlyingDeclRefOrNull(VD->getInit())) {
      auto varDeclExpr = m_RMV->BuildDeclRef(VDDiff.getDecl_dx());
      Expr* errorExpr = UpdateErrorForFuncCallAssigns(
          this, savedDecl.getExpr_dx(),
          m_RMV->BuildDeclRef(VDDiff.getDecl_dx()), m_NestedFuncError);
      Expr* prntExpr =
          GetPrintExpr(VD->getNameAsString(), savedDecl.getExpr_dx(),
                       varDeclExpr, errorExpr);
      AddErrorStmtToBlock(VDRef, EstVD, errorExpr, isInsideLoop, prntExpr);
    }
  }
}

void ErrorEstimationHandler::InitialiseRMV(ReverseModeVisitor& RMV) {
  m_RMV = &RMV;
}

void ErrorEstimationHandler::ForgetRMV() { m_RMV = nullptr; }

void ErrorEstimationHandler::ActBeforeCreatingDerivedFnParamTypes(
    unsigned& numExtraParam) {
  numExtraParam += 1 + m_PrintErrors;
}

void ErrorEstimationHandler::ActAfterCreatingDerivedFnParamTypes(
    llvm::SmallVectorImpl<QualType>& paramTypes) {
  // If we are performing error estimation, our gradient function
  // will have an extra argument which will hold the final error value
  paramTypes.push_back(
      m_RMV->m_Context.getLValueReferenceType(m_RMV->m_Context.DoubleTy));
  // If error printing was enabled, add another param type for it.
  if (m_PrintErrors) {
    paramTypes.push_back(
        m_RMV->m_Context.getLValueReferenceType(m_ErrorFileType));
  }
};

void ErrorEstimationHandler::ActAfterCreatingDerivedFnParams(
    llvm::SmallVectorImpl<ParmVarDecl*>& params) {
  m_Params = &params;
  // If in error estimation mode, create the error parameter
  ASTContext& context = m_RMV->m_Context;
  QualType finErrorType = context.getLValueReferenceType(context.DoubleTy);
  // Repeat the above but for the error ouput var "_final_error"
  ParmVarDecl* errorVarDecl =
      ParmVarDecl::Create(context, m_RMV->m_Derivative, noLoc, noLoc,
                          &context.Idents.get("_final_error"), finErrorType,
                          context.getTrivialTypeSourceInfo(finErrorType, noLoc),
                          params.front()->getStorageClass(),
                          /*DefArg=*/nullptr);
  m_FinalError = m_RMV->BuildDeclRef(errorVarDecl);
  params.push_back(errorVarDecl);
  m_RMV->m_Sema.PushOnScopeChains(errorVarDecl, m_RMV->getCurrentScope(),
                                  /*AddToContext=*/false);
  // If printing of error estimates was requested, build another parameter to
  // store the ofstream object.
  if (m_PrintErrors) {
    QualType prntErrorType = context.getLValueReferenceType(m_ErrorFileType);
    ParmVarDecl* VD = ParmVarDecl::Create(
        context, m_RMV->m_Derivative, noLoc, noLoc,
        &context.Idents.get("_error_stream"), prntErrorType,
        context.getTrivialTypeSourceInfo(prntErrorType, noLoc),
        params.front()->getStorageClass(),
        /*DefArg=*/nullptr);
    m_ErrorFile = m_RMV->BuildDeclRef(VD);
    params.push_back(VD);
    m_RMV->m_Sema.PushOnScopeChains(VD, m_RMV->getCurrentScope(),
                                    /*AddToContext=*/false);
  }
}

void ErrorEstimationHandler::ActOnEndOfDerivedFnBody() {
  // Since 'return' is not an assignment, add its error to _final_error
  // given it is not a DeclRefExpr.
  EmitFinalErrorStmts(*m_Params, m_RMV->m_Function->getNumParams());
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

void ErrorEstimationHandler::
    ActBeforeDifferentiatingSingleStmtBranchInVisitIfStmt() {
  m_ShouldEmit.push(true);
}

void ErrorEstimationHandler::
    ActBeforeFinalisingVisitBranchSingleStmtInIfVisitStmt() {
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

void ErrorEstimationHandler::ActBeforeFinalisingVisitReturnStmt(
    StmtDiff& ExprDiff, clang::Expr*& retDeclRefExpr) {
  // If the return expression is not a DeclRefExpression and is of type
  // float, we should add it to the error estimate because returns are
  // similiar to implicit assigns.
  SaveReturnExpr(ExprDiff.getExpr(), cast<DeclRefExpr>(retDeclRefExpr));
}

void ErrorEstimationHandler::ActBeforeFinalisingPostIncDecOp(StmtDiff& diff) {
  EmitUnaryOpErrorStmts(diff, m_RMV->isInsideLoop);
}

// FIXME: Issue a warning that error estimation may produce incorrect result if
// any of the arguments are being passed by reference to the call expression
// `CE`.
void ErrorEstimationHandler::ActBeforeFinalizingVisitCallExpr(
    const clang::CallExpr*& CE, clang::Expr*& OverloadedDerivedFn,
    llvm::SmallVectorImpl<Expr*>& derivedCallArgs,
    llvm::SmallVectorImpl<VarDecl*>& ArgResultDecls, bool asGrad) {
  if (OverloadedDerivedFn && asGrad) {
    // Derivative was found.
    FunctionDecl* fnDecl =
        dyn_cast<CallExpr>(OverloadedDerivedFn)->getDirectCallee();

    // If in error estimation, build the statement for the error
    // in the input prameters (if of reference type) to call and save to
    // emit them later.

    EmitNestedFunctionParamError(fnDecl, derivedCallArgs, ArgResultDecls,
                                 CE->getNumArgs());
  }
}

void ErrorEstimationHandler::ActAfterCloningLHSOfAssignOp(
    clang::Expr*& LCloned, clang::Expr*& R,
    clang::BinaryOperator::Opcode& opCode) {
  m_DeltaVar = RegisterBinaryOpLHS(LCloned, R,
                                   /*isAssign=*/opCode == BO_Assign);
}

void ErrorEstimationHandler::ActBeforeFinalisingAssignOp(
    clang::Expr*& LCloned, clang::Expr*& oldValue) {
  // Now, we should emit the delta for LHS if it met all the
  // requirements previously.
  EmitBinaryOpErrorStmts(LCloned, oldValue, m_DeltaVar, m_RMV->isInsideLoop);
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
    llvm::SmallVectorImpl<clang::Expr*>& pullbackArgs,
    llvm::SmallVectorImpl<DeclStmt*>& ArgDecls, bool hasAssignee) {
  auto errorRef =
      m_RMV->BuildVarDecl(m_RMV->m_Context.DoubleTy, "_t",
                          m_RMV->getZeroInit(m_RMV->m_Context.DoubleTy));
  ArgDecls.push_back(m_RMV->BuildDeclStmt(errorRef));
  auto finErr = m_RMV->BuildDeclRef(errorRef);
  pullbackArgs.push_back(finErr);
  // If Error printing was enabled, pass the same file object onto the next function.
  if (m_PrintErrors)
    pullbackArgs.push_back(m_ErrorFile);
  if (hasAssignee) {
    if (m_NestedFuncError)
      m_NestedFuncError = m_RMV->BuildOp(BO_Add, m_NestedFuncError, finErr);
    else
      m_NestedFuncError = finErr;
  }
}

void ErrorEstimationHandler::ActBeforeFinalizingVisitDeclStmt(
    llvm::SmallVectorImpl<Decl*>& decls,
    llvm::SmallVectorImpl<Decl*>& declsDiff) {
  // For all dependent variables, we register them for estimation
  // here.
  for (size_t i = 0; i < decls.size(); i++) {
    VarDeclDiff VDDiff(static_cast<VarDecl*>(decls[0]),
                       static_cast<VarDecl*>(declsDiff[0]));
    EmitDeclErrorStmts(VDDiff, m_RMV->isInsideLoop);
  }
}
} // namespace clad
