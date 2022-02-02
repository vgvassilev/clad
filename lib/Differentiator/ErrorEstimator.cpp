#include "clad/Differentiator/ErrorEstimator.h"
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
}

  void ErrorEstimationHandler::SetErrorEstimationModel(
      FPErrorEstimationModel* estModel) {
    m_EstModel = estModel;
  }

  Expr*
  ErrorEstimationHandler::getArraySubscriptExpr(Expr* arrBase, Expr* idx,
                                                bool isCladSpType /*=true*/) {
    if (isCladSpType) {
      return m_Sema
          .ActOnArraySubscriptExpr(getCurrentScope(), arrBase,
                                   arrBase->getExprLoc(), idx, noLoc)
          .get();
    } else {
      return m_Sema.CreateBuiltinArraySubscriptExpr(arrBase, noLoc, idx, noLoc)
          .get();
    }
  }

  void ErrorEstimationHandler::BuildFinalErrorStmt() {
    Expr* finExpr = nullptr;
    // If we encountered any arithmetic expression in the return statement,
    // we must add its error to the final estimate.
    if (m_RetErrorExpr) {
      auto flitr = FloatingLiteral::Create(m_Context, llvm::APFloat(1.0), true,
                                           m_Context.DoubleTy, noLoc);
      finExpr = m_EstModel->AssignError(StmtDiff(m_RetErrorExpr, flitr));
    }

    // Build the final error statement with the sum of all _delta_*.
    Expr* addErrorExpr = m_EstModel->CalculateAggregateError();
    if (addErrorExpr) {
      if (finExpr)
        addErrorExpr = BuildOp(BO_Add, addErrorExpr, finExpr);
    } else if (finExpr) {
      addErrorExpr = finExpr;
    }

    // Finally add the final error expression to the derivative body.
    addToCurrentBlock(BuildOp(BO_AddAssign, m_FinalError, addErrorExpr),
                      forward);
  }

  void
  ErrorEstimationHandler::AddErrorStmtToBlock(Expr* var, Expr* deltaVar,
                                              Expr* errorExpr,
                                              bool isInsideLoop /*=false*/) {
    if (auto ASE = dyn_cast<ArraySubscriptExpr>(var)) {
      // If inside loop, the index has been pushed twice
      // (once by ArraySubscriptExpr and the second time by us)
      // pop and store it in a temporary variable to reuse later.
      // FIXME: build add assign into he same expression i.e.
      // _final_error += _delta_arr[pop(_t0)] += <-Error Expr->
      // to avoid storage of the pop value.
      Expr* popVal = ASE->getIdx();
      if (isInsideLoop) {
        LookupResult& Pop = GetCladTapePop();
        CXXScopeSpec CSS;
        CSS.Extend(m_Context, GetCladNamespace(), noLoc, noLoc);
        auto PopDRE = m_Sema
                          .BuildDeclarationNameExpr(CSS, Pop,
                                                    /*AcceptInvalidDecl=*/false)
                          .get();
        Expr* tapeRef = dyn_cast<CallExpr>(popVal)->getArg(0);
        popVal = m_Sema
                     .ActOnCallExpr(getCurrentScope(), PopDRE, noLoc, tapeRef,
                                    noLoc)
                     .get();
        popVal = StoreAndRef(popVal, reverse);
      }
      // If the variable declration refers to an array element
      // create the suitable _delta_arr[i] (because we have not done
      // this before).
      deltaVar = getArraySubscriptExpr(deltaVar, popVal);
      addToCurrentBlock(BuildOp(BO_AddAssign, deltaVar, errorExpr), reverse);
      // immediately emit fin_err += delta_[].
      // This is done to avoid adding all errors at the end
      // and only add the errors that were calculated.
      addToCurrentBlock(BuildOp(BO_AddAssign, m_FinalError, deltaVar), reverse);

    } else
      addToCurrentBlock(BuildOp(BO_AddAssign, deltaVar, errorExpr), reverse);
  }

  void
  ErrorEstimationHandler::EmitErrorEstimationStmts(direction d /*=forward*/) {
    if (d == forward) {
      while (!m_ForwardReplStmts.empty())
        addToCurrentBlock(m_ForwardReplStmts.pop_back_val(), d);
    } else {
      while (!m_ReverseErrorStmts.empty())
        addToCurrentBlock(m_ReverseErrorStmts.pop_back_val(), d);
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
      auto retVarDecl = BuildVarDecl(m_Context.DoubleTy, "_ret_value",
                                     getZeroInit(m_Context.DoubleTy));
      AddToGlobalBlock(BuildDeclStmt(retVarDecl));
      m_RetErrorExpr = BuildDeclRef(retVarDecl);
    }
    addToCurrentBlock(BuildOp(BO_Assign, m_RetErrorExpr, retDeclRefExpr),
                      forward);
  }

  void ErrorEstimationHandler::EmitNestedFunctionParamError(
      FunctionDecl* fnDecl, llvm::SmallVectorImpl<Expr*>& CallArgs,
      llvm::SmallVectorImpl<VarDecl*>& ArgResultDecls, size_t numArgs) {
    assert(fnDecl && "Must have a value");
    for (size_t i = 0; i < numArgs; i++) {
      if (!fnDecl->getParamDecl(0)->getType()->isLValueReferenceType())
        continue;
      Expr* errorExpr = m_EstModel->AssignError(
          {CallArgs[i], BuildDeclRef(ArgResultDecls[i])});
      Expr* errorStmt = BuildOp(BO_AddAssign, m_FinalError, errorExpr);
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
      auto tape = MakeCladTapeFor(val, name);
      m_ForwardReplStmts.push_back(tape.Push);
      // Nice to store pop values becuase user might refer to getExpr
      // multiple times in Assign Error.
      Expr* popVal = StoreAndRef(tape.Pop, reverse);
      return StmtDiff(tape.Push, popVal);
    } else {
      QualType QTval = val->getType();
      if (auto AType = dyn_cast<ArrayType>(QTval))
        QTval = AType->getElementType();

      auto savedVD = GlobalStoreImpl(QTval, name);
      auto savedRef = BuildDeclRef(savedVD);
      m_ForwardReplStmts.push_back(BuildOp(BO_Assign, savedRef, val));
      return StmtDiff(savedRef, savedRef);
    }
  }

  void ErrorEstimationHandler::SaveParamValue(DeclRefExpr* paramRef) {
    assert(paramRef && "Must have a value");
    VarDecl* paramDecl = cast<VarDecl>(paramRef->getDecl());
    QualType paramType = paramRef->getType();
    std::string name = "_EERepl_" + paramDecl->getNameAsString();
    VarDecl* savedDecl;
    if (isArrayOrPointerType(paramType)) {
      auto diffVar = m_Variables[paramDecl];
      auto QType =
          GetCladArrayOfType(getUnderlyingArrayType(paramType, m_Context));
      savedDecl =
          BuildVarDecl(QType, name, BuildArrayRefSizeExpr(diffVar),
                       /*DirectInit=*/false,
                       /*TSI=*/nullptr, VarDecl::InitializationStyle::CallInit);
      AddToGlobalBlock(BuildDeclStmt(savedDecl));
      Stmts loopBody;
      // Get iter variable.
      auto loopIdx =
          BuildVarDecl(m_Context.IntTy, "i", getZeroInit(m_Context.IntTy));
      auto currIdx = BuildDeclRef(loopIdx);
      // Build the assign expression.
      loopBody.push_back(BuildOp(
          BO_Assign, getArraySubscriptExpr(BuildDeclRef(savedDecl), currIdx),
          getArraySubscriptExpr(paramRef, currIdx,
                                /*isCladSpType=*/false)));
      Expr* conditionExpr =
          BuildOp(BO_LT, currIdx, BuildArrayRefSizeExpr(diffVar));
      Expr* incExpr = BuildOp(UO_PostInc, currIdx);
      // Make for loop.
      Stmt* ArrayParamLoop = new (m_Context)
          ForStmt(m_Context, BuildDeclStmt(loopIdx), conditionExpr, nullptr,
                  incExpr, MakeCompoundStmt(loopBody), noLoc, noLoc, noLoc);
      AddToGlobalBlock(ArrayParamLoop);
    } else
      savedDecl = GlobalStoreImpl(paramType, name, paramRef);
    m_ParamRepls.emplace(paramDecl, BuildDeclRef(savedDecl));
  }

  Expr*
  ErrorEstimationHandler::RegisterVariable(VarDecl* VD,
                                           bool toCurrentScope /*=false*/) {
    if (!CanRegisterVariable(VD))
      return nullptr;
    // Get the init error from setError.
    Expr* init = m_EstModel->SetError(VD);
    auto VDType = VD->getType();
    // The type of the _delta_ value should be customisable.
    QualType QType;
    Expr* deltaVar = nullptr;
    auto diffVar = m_Variables[VD];
    if (isArrayRefType(diffVar->getType())) {
      VarDecl* EstVD;
      auto sizeExpr = BuildArrayRefSizeExpr(diffVar);
      QType = GetCladArrayOfType(getUnderlyingArrayType(VDType, m_Context));
      EstVD =
          BuildVarDecl(QType, "_delta_" + VD->getNameAsString(), sizeExpr,
                       /*DirectInit=*/false,
                       /*TSI=*/nullptr, VarDecl::InitializationStyle::CallInit);
      if (!toCurrentScope)
        AddToGlobalBlock(BuildDeclStmt(EstVD));
      else
        addToCurrentBlock(BuildDeclStmt(EstVD), forward);
      deltaVar = BuildDeclRef(EstVD);
    } else {
      QType = isArrayOrPointerType(VDType) ? VDType : m_Context.DoubleTy;
      init = init ? init : getZeroInit(QType);
      // Store the "_delta_*" value.
      if (!toCurrentScope) {
        auto EstVD =
            GlobalStoreImpl(QType, "_delta_" + VD->getNameAsString(), init);
        deltaVar = BuildDeclRef(EstVD);
      } else {
        deltaVar =
            StoreAndRef(init, QType, forward, "_delta_" + VD->getNameAsString(),
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
    QualType varDeclType = isArrayOrPointerType(varDeclBase)
                               ? getUnderlyingArrayType(varDeclBase, m_Context)
                               : varDeclBase;
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
        diag(DiagnosticsEngine::Warning, VD->getEndLoc(),
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
    ArraySubscriptExpr* temp = dyn_cast<ArraySubscriptExpr>(
        expr->IgnoreImplicit());
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
        if (!isArrayOrPointerType(params[i]->getType())) {
          // Since we need the input value of x, check for a replacement.
          // If no replacement found, use the actual declRefExpr.
          auto savedVal = GetParamReplacement(params[i]);
          savedVal = savedVal ? savedVal : BuildDeclRef(decl);
          // Finally emit the error.
          auto errorExpr = GetError(savedVal, m_Variables[decl]);
          addToCurrentBlock(BuildOp(BO_AddAssign, deltaVar, errorExpr));
        } else {
          auto LdiffExpr = m_Variables[decl];
          VarDecl* idxExprDecl = nullptr;
          // Save our index expression so it can be used later.
          if (!m_IdxExpr) {
            idxExprDecl = BuildVarDecl(m_Context.IntTy, "i",
                                       getZeroInit(m_Context.IntTy));
            m_IdxExpr = BuildDeclRef(idxExprDecl);
          }
          Expr *Ldiff, *Ldelta;
          Ldiff = getArraySubscriptExpr(LdiffExpr, m_IdxExpr,
                                        isArrayRefType(LdiffExpr->getType()));
          Ldelta = getArraySubscriptExpr(deltaVar, m_IdxExpr);
          auto savedVal = GetParamReplacement(params[i]);
          savedVal = savedVal ? savedVal : BuildDeclRef(decl);
          auto LRepl = getArraySubscriptExpr(savedVal, m_IdxExpr);
          // Build the loop to put in reverse mode.
          Expr* errorExpr = GetError(LRepl, Ldiff);
          auto commonVarDecl =
              BuildVarDecl(errorExpr->getType(), "_t", errorExpr);
          Expr* commonVarExpr = BuildDeclRef(commonVarDecl);
          Expr* deltaAssignExpr = BuildOp(BO_AddAssign, Ldelta, commonVarExpr);
          Expr* finalAssignExpr =
              BuildOp(BO_AddAssign, m_FinalError, commonVarExpr);
          Stmts loopBody;
          loopBody.push_back(BuildDeclStmt(commonVarDecl));
          loopBody.push_back(deltaAssignExpr);
          loopBody.push_back(finalAssignExpr);
          Expr* conditionExpr = BuildOp(BO_LT, m_IdxExpr,
                                        BuildArrayRefSizeExpr(LdiffExpr));
          Expr* incExpr = BuildOp(UO_PostInc, m_IdxExpr);
          Stmt* ArrayParamLoop = new (m_Context)
              ForStmt(m_Context, nullptr, conditionExpr, nullptr, incExpr,
                      MakeCompoundStmt(loopBody), noLoc, noLoc, noLoc);
          // For multiple array parameters, we want to keep the same
          // iterative variable, so reset that here in the case that this
          // is not out first array.
          if (!idxExprDecl) {
            addToCurrentBlock(
                BuildOp(BO_Assign, m_IdxExpr, getZeroInit(m_Context.IntTy)));
          } else {
            addToCurrentBlock(BuildDeclStmt(idxExprDecl));
          }
          addToCurrentBlock(ArrayParamLoop);
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
      if (auto deltaVar = IsRegistered(cast<VarDecl>(DRE->getDecl()))) {
        // Create a variable/tape call to store the current value of the
        // the sub-expression so that it can be used later.
        StmtDiff savedVar = GlobalStoreAndRef(DRE, "_EERepl_" +
                                                       DRE->getDecl()
                                                           ->getNameAsString());
        if (isInsideLoop) {
          // It is nice to save the pop value.
          // We do not know how many times the user will use dx,
          // hence we should pop values beforehand to avoid unequal pushes
          // and and pops.
          Expr* popVal = StoreAndRef(savedVar.getExpr_dx(), reverse);
          savedVar = {savedVar.getExpr(), popVal};
        }
        Expr* erroExpr = GetError(savedVar.getExpr_dx(), var.getExpr_dx());
        AddErrorStmtToBlock(var.getExpr(), deltaVar, erroExpr, isInsideLoop);
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

  void ErrorEstimationHandler::EmitBinaryOpErrorStmts(Expr* LExpr,
                                                      Expr* oldValue,
                                                      Expr* deltaVar,
                                                      bool isInsideLoop) {
    if (!deltaVar)
      return;
    // For now save all lhs.
    // FIXME: We can optimize stores here by using the ones created
    // previously.
    StmtDiff savedExpr = SaveValue(LExpr, isInsideLoop);
    // Assign the error.
    Expr* errorExpr = GetError(savedExpr.getExpr_dx(), oldValue);
    AddErrorStmtToBlock(LExpr, deltaVar, errorExpr, isInsideLoop);
    // If there are assign statements to emit in reverse, do that.
    EmitErrorEstimationStmts(reverse);
  }

  void ErrorEstimationHandler::EmitDeclErrorStmts(VarDeclDiff VDDiff,
                                                  bool isInsideLoop) {
    auto VD = VDDiff.getDecl();
    if (!CanRegisterVariable(VD))
      return;
    // Build the delta expresion for the variable to be registered.
    auto EstVD = RegisterVariable(VD);
    DeclRefExpr* VDRef = BuildDeclRef(VD);
    // FIXME: We should do this for arrays too.
    if (!VD->getType()->isArrayType()) {
      StmtDiff savedDecl = SaveValue(VDRef, isInsideLoop);
      // If the VarDecl has an init, we should assign it with an error.
      if (VD->getInit() && !GetUnderlyingDeclRefOrNull(VD->getInit())) {
        Expr* errorExpr = GetError(savedDecl.getExpr_dx(),
                                   BuildDeclRef(VDDiff.getDecl_dx()));
        AddErrorStmtToBlock(VDRef, EstVD, errorExpr, isInsideLoop);
      }
    }
  }

} // namespace clad
