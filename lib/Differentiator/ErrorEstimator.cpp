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
        auto flitr = FloatingLiteral::Create(
            m_Context, llvm::APFloat(1.0), true, m_Context.DoubleTy, noLoc);
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
  ErrorEstimationHandler::AddErrorStmtToBlock(Expr* var,
                                              Expr* deltaVar,
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
      deltaVar =
          m_Sema.CreateBuiltinArraySubscriptExpr(deltaVar, noLoc, popVal, noLoc)
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
      auto retVarDecl = BuildVarDecl(m_Context.DoubleTy,
                                     "_ret_value",
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
    std::string name =
        "_EERepl_" +
        GetUnderlyingDeclRefOrNull(val)->getDecl()->getNameAsString();
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
    if(!paramRef)
      return;
    VarDecl* paramDecl = cast<VarDecl>(paramRef->getDecl());
    m_ParamRepls.emplace(paramDecl,
                         StoreAndRef(paramRef,
                                     forward,
                                     "_EERepl_" + paramDecl->getNameAsString(),
                                     /*forceDeclCreation=*/true));
  }

  Expr* ErrorEstimationHandler::RegisterVariable(VarDecl* VD) {
    if (!CanRegisterVariable(VD))
      return nullptr;
    // Get the init error from setError.
    Expr* init = m_EstModel->SetError(VD);
    auto VDType = VD->getType();
    // The type of the _delta_ value should be customisable.
    auto QType = VDType->isArrayType() ? VDType : m_Context.DoubleTy;
    init = init ? init : getZeroInit(QType);
    // Store the "_delta_*" value.
    auto EstVD =
        GlobalStoreImpl(QType, "_delta_" + VD->getNameAsString(), init);
    auto deltaVar = BuildDeclRef(EstVD);
    // Register the variable for estimate calculation.
    m_EstModel->AddVarToEstimate(VD, deltaVar);
    return deltaVar;
  }

  bool ErrorEstimationHandler::CanRegisterVariable(VarDecl* VD) {
    // Get the types on the declartion and initalization expression.
    QualType varDeclBase = VD->getType();
    const Type *varDeclType =
        varDeclBase->isArrayType()
            ? varDeclBase->getArrayElementTypeNoTypeQual()
            : varDeclBase.getTypePtr();
    const Expr* init = VD->getInit();
    // If declarationg type in not floating point type, we want to do two things.
    if (!varDeclType->isFloatingType()) {
      // Firstly, we want to check if the declaration is a lossy conversion.
      // For example, if we have something like:
      // double y = 2.77788;
      // int x = y <-- This causes truncation in y,
      // making _delta_x = y - (double)x
      // For now, we will just warn the user of casts like these
      // because we assume the cast is intensional.
      if (init && init->IgnoreImpCasts()->getType()->isFloatingType())
        diag(DiagnosticsEngine::Warning,
             VD->getEndLoc(),
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

} // namespace clad
