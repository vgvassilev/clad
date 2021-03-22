#include "clad/Differentiator/ErrorEstimator.h"
#include "clad/Differentiator/DerivativeBuilder.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/ReverseModeVisitor.h"

#include "clang/AST/Decl.h"

using namespace clang;

namespace clad {
  void ErrorEstimationHandler::SetErrorEstimationModel(
      FPErrorEstimationModel* estModel) {
    m_EstModel = estModel;
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
    if (addErrorExpr)
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
      if (isInsideLoop)
        popVal = StoreAndRef(popVal, reverse);
      // If the variable declration referes to an array element
      // create the suitable _delta_arr[i] (because we have not done
      // this before).
      deltaVar = m_Sema
                     .CreateBuiltinArraySubscriptExpr(deltaVar, noLoc, popVal,
                                                      noLoc)
                     .get();
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

  void
  ErrorEstimationHandler::EmitNestedFunctionParamError(FunctionDecl* fnDecl,
                                                       StmtDiff arg) {
    if (!fnDecl || !fnDecl->getParamDecl(0)->getType()->isLValueReferenceType())
      return;

    Expr* errorExpr = m_EstModel->AssignError(arg);
    Expr* errorStmt = BuildOp(BO_AddAssign, m_FinalError, errorExpr);
    m_ReverseErrorStmts.push_back(errorStmt);
  }

  StmtDiff ErrorEstimationHandler::SaveValue(Expr* val,
                                             bool isInsideLoop /*=false*/) {
    // Definite not null.
    std::string name = "_EERepl_" + GetUnderlyingDeclRefOrNull(val)
                                        ->getDecl()
                                        ->getNameAsString();
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
    if (!paramRef)
      return;
    VarDecl* paramDecl = cast<VarDecl>(paramRef->getDecl());
    m_ParamRepls.emplace(paramDecl,
                         StoreAndRef(paramRef, forward,
                                     "_EERepl_" + paramDecl->getNameAsString(),
                                     /*forceDeclCreation=*/true));
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
    auto QType = VDType->isArrayType() ? VDType : m_Context.DoubleTy;
    init = init ? init : getZeroInit(QType);
    Expr* deltaVar = nullptr;
    // Store the "_delta_*" value.
    if (!toCurrentScope) {
      auto EstVD = GlobalStoreImpl(QType, "_delta_" + VD->getNameAsString(),
                                   init);
      deltaVar = BuildDeclRef(EstVD);
    } else {
      deltaVar = StoreAndRef(init, QType, forward,
                             "_delta_" + VD->getNameAsString(),
                             /*forceDeclCreation=*/true);
    }
    // Register the variable for estimate calculation.
    m_EstModel->AddVarToEstimate(VD, deltaVar);
    return deltaVar;
  }

  bool ErrorEstimationHandler::CanRegisterVariable(VarDecl* VD) {

    // Get the types on the declartion and initalization expression.
    QualType varDeclBase = VD->getType();
    const Type* varDeclType = varDeclBase->isArrayType()
                                  ? varDeclBase->getArrayElementTypeNoTypeQual()
                                  : varDeclBase.getTypePtr();
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
      llvm::SmallVector<ParmVarDecl*, 4> params, int numParams) {
    // Emit error variables of parameters at the end.
    for (size_t i = 0; i < numParams; i++) {
      // Right now, we just ignore them since we have no way of knowing
      // the size of an array.
      if (params[i]->getType()->isArrayType() ||
          params[i]->getType()->isPointerType())
        continue;

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
        // Since we need the input value of x, check for a replacement.
        // If no replacement found, use the actual declRefExpr.
        auto savedVal = GetParamReplacement(dyn_cast<ParmVarDecl>(decl));
        savedVal = savedVal ? savedVal : BuildDeclRef(decl);
        // Finally emit the error.
        auto errorExpr = GetError(savedVal, m_Variables[decl]);
        addToCurrentBlock(BuildOp(BO_AddAssign, deltaVar, errorExpr));
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
    bool declRefOk = (!RRef || !isAssign) &&
                     (!LExpr->getType()->isArrayType() ||
                      !LExpr->getType()->isPointerType());
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
