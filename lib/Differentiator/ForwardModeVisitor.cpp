//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clad/Differentiator/ForwardModeVisitor.h"

#include "ConstantFolder.h"

#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/StmtClone.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/TemplateBase.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Template.h"

#include "llvm/Support/SaveAndRestore.h"

#include <algorithm>

#include "clad/Differentiator/Compatibility.h"

using namespace clang;

namespace clad {
  ForwardModeVisitor::ForwardModeVisitor(DerivativeBuilder& builder)
      : VisitorBase(builder) {}

  ForwardModeVisitor::~ForwardModeVisitor() {}

  OverloadedDeclWithContext
  ForwardModeVisitor::Derive(const FunctionDecl* FD,
                             const DiffRequest& request) {
    silenceDiags = !request.VerboseDiags;
    m_Function = FD;
    assert(!m_DerivativeInFlight &&
           "Doesn't support recursive diff. Use DiffPlan.");
    m_DerivativeInFlight = true;

    DiffParams args{};
    IndexIntervalTable indexIntervalTable{};
    if (request.Args)
      std::tie(args, indexIntervalTable) = parseDiffArgs(request.Args, FD);
    else {
      // FIXME: implement gradient-vector products to fix the issue.
      assert((FD->getNumParams() <= 1) &&
             "nested forward mode differentiation for several args is broken");
      std::copy(FD->param_begin(), FD->param_end(), std::back_inserter(args));
    }
    if (args.empty())
      return {};
    // Check that only one arg is requested and if the arg requested is of array
    // or pointer type, only one of the indices have been requested
    if (args.size() > 1 || (isArrayOrPointerType(args[0]->getType()) &&
                            (indexIntervalTable.size() != 1 || indexIntervalTable[0].size() != 1))) {
      diag(DiagnosticsEngine::Error,
           request.Args ? request.Args->getEndLoc() : noLoc,
           "Forward mode differentiation w.r.t. several parameters at once is "
           "not "
           "supported, call 'clad::differentiate' for each parameter "
           "separately");
      return {};
    }

    m_IndependentVar = args.back();
    std::string derivativeSuffix("");
    // If param is not real (i.e. floating point or integral), a pointer to a
    // real type, or an array of a real type we cannot differentiate it.
    // FIXME: we should support custom numeric types in the future.
    if (isArrayOrPointerType(m_IndependentVar->getType())) {
      if (!m_IndependentVar->getType()
               ->getPointeeOrArrayElementType()
               ->isRealType()) {
        diag(DiagnosticsEngine::Error,
             m_IndependentVar->getEndLoc(),
             "attempted differentiation w.r.t. a parameter ('%0') which is not"
             " an array or pointer of a real type",
             {m_IndependentVar->getNameAsString()});
        return {};
      }
      m_IndependentVarIndex = indexIntervalTable[0].Start;
      derivativeSuffix = "_" + std::to_string(m_IndependentVarIndex);
    } else if (!m_IndependentVar->getType()->isRealType()) {
      diag(DiagnosticsEngine::Error,
           m_IndependentVar->getEndLoc(),
           "attempted differentiation w.r.t. a parameter ('%0') which is not "
           "of a real type",
           {m_IndependentVar->getNameAsString()});
      return {};
    }
    m_DerivativeOrder = request.CurrentDerivativeOrder;
    std::string s = std::to_string(m_DerivativeOrder);
    std::string derivativeBaseName;
    if (m_DerivativeOrder == 1)
      s = "";
    switch (FD->getOverloadedOperator()) {
      default: derivativeBaseName = request.BaseFunctionName; break;
      case OO_Call: derivativeBaseName = "operator_call"; break;
    }

    m_ArgIndex = std::distance(
        FD->param_begin(),
        std::find(FD->param_begin(), FD->param_end(), m_IndependentVar));
    IdentifierInfo* II =
        &m_Context.Idents.get(derivativeBaseName + "_d" + s + "arg" +
                              std::to_string(m_ArgIndex) + derivativeSuffix);
    DeclarationNameInfo name(II, noLoc);
    llvm::SaveAndRestore<DeclContext*> SaveContext(m_Sema.CurContext);
    llvm::SaveAndRestore<Scope*> SaveScope(m_CurScope);
    DeclContext* DC = const_cast<DeclContext*>(m_Function->getDeclContext());
    m_Sema.CurContext = DC;
    DeclWithContext result = m_Builder.cloneFunction(
        FD, *this, DC, m_Sema, m_Context, noLoc, name, FD->getType());
    FunctionDecl* derivedFD = result.first;
    m_Derivative = derivedFD;

    llvm::SmallVector<ParmVarDecl*, 4> params;
    ParmVarDecl* newPVD = nullptr;
    const ParmVarDecl* PVD = nullptr;

    // Function declaration scope
    beginScope(Scope::FunctionPrototypeScope | Scope::FunctionDeclarationScope |
               Scope::DeclScope);
    m_Sema.PushFunctionScope();
    m_Sema.PushDeclContext(getCurrentScope(), m_Derivative);

    // FIXME: We should implement FunctionDecl and ParamVarDecl cloning.
    for (size_t i = 0, e = FD->getNumParams(); i < e; ++i) {
      PVD = FD->getParamDecl(i);
      Expr* clonedPVDDefaultArg = 0;
      if (PVD->hasDefaultArg())
        clonedPVDDefaultArg = Clone(PVD->getDefaultArg());

      newPVD = ParmVarDecl::Create(m_Context,
                                   m_Sema.CurContext,
                                   noLoc,
                                   noLoc,
                                   PVD->getIdentifier(),
                                   PVD->getType(),
                                   PVD->getTypeSourceInfo(),
                                   PVD->getStorageClass(),
                                   clonedPVDDefaultArg);

      // Make m_IndependentVar to point to the argument of the newly created
      // derivedFD.
      if (PVD == m_IndependentVar)
        m_IndependentVar = newPVD;

      params.push_back(newPVD);
      // Add the args in the scope and id chain so that they could be found.
      if (newPVD->getIdentifier())
        m_Sema.PushOnScopeChains(newPVD,
                                 getCurrentScope(),
                                 /*AddToContext*/ false);
    }

    llvm::ArrayRef<ParmVarDecl*> paramsRef =
        llvm::makeArrayRef(params.data(), params.size());
    derivedFD->setParams(paramsRef);
    derivedFD->setBody(nullptr);

    // Function body scope
    beginScope(Scope::FnScope | Scope::DeclScope);
    m_DerivativeFnScope = getCurrentScope();
    beginBlock();
    // For each function parameter variable, store its derivative value.
    for (auto param : params) {
      if (!param->getType()->isRealType())
        continue;
      // If param is independent variable, its derivative is 1, otherwise 0.
      int dValue = (param == m_IndependentVar);
      auto dParam =
          ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, dValue);
      // For each function arg, create a variable _d_arg to store derivatives
      // of potential reassignments, e.g.:
      // double f_darg0(double x, double y) {
      //   double _d_x = 1;
      //   double _d_y = 0;
      //   ...
      auto dParamDecl = BuildVarDecl(param->getType(),
                                     "_d_" + param->getNameAsString(),
                                     dParam);
      addToCurrentBlock(BuildDeclStmt(dParamDecl));
      dParam = BuildDeclRef(dParamDecl);
      // Memorize the derivative of param, i.e. whenever the param is visited
      // in the future, it's derivative dParam is found (unless reassigned with
      // something new).
      m_Variables[param] = dParam;
    }

    Stmt* BodyDiff = Visit(FD->getBody()).getStmt();
    if (auto CS = dyn_cast<CompoundStmt>(BodyDiff))
      for (Stmt* S : CS->body())
        addToCurrentBlock(S);
    else
      addToCurrentBlock(BodyDiff);
    Stmt* derivativeBody = endBlock();
    derivedFD->setBody(derivativeBody);

    endScope(); // Function body scope
    m_Sema.PopFunctionScopeInfo();
    m_Sema.PopDeclContext();
    endScope(); // Function decl scope

    m_DerivativeInFlight = false;

    return OverloadedDeclWithContext{result.first, result.second, nullptr};
  }

  StmtDiff ForwardModeVisitor::VisitStmt(const Stmt* S) {
    diag(
        DiagnosticsEngine::Warning,
        S->getBeginLoc(),
        "attempted to differentiate unsupported statement, no changes applied");
    // Unknown stmt, just clone it.
    return StmtDiff(Clone(S));
  }

  StmtDiff ForwardModeVisitor::VisitCompoundStmt(const CompoundStmt* CS) {
    beginScope(Scope::DeclScope);
    beginBlock();
    for (Stmt* S : CS->body()) {
      StmtDiff SDiff = Visit(S);
      addToCurrentBlock(SDiff.getStmt_dx());
      addToCurrentBlock(SDiff.getStmt());
    }
    CompoundStmt* Result = endBlock();
    endScope();
    // Differentation of CompundStmt produces another CompoundStmt with both
    // original and derived statements, i.e. Stmt() is Result and Stmt_dx() is
    // null.
    return StmtDiff(Result);
  }

  StmtDiff ForwardModeVisitor::VisitIfStmt(const IfStmt* If) {
    // Control scope of the IfStmt. E.g., in if (double x = ...) {...}, x goes
    // to this scope.
    beginScope(Scope::DeclScope | Scope::ControlScope);
    // Create a block "around" if statement, e.g:
    // {
    //   ...
    //  if (...) {...}
    // }
    beginBlock();
    const Stmt* init = If->getInit();
    StmtDiff initResult = init ? Visit(init) : StmtDiff{};
    // If there is Init, it's derivative will be output in the block before if:
    // E.g., for:
    // if (int x = 1; ...) {...}
    // result will be:
    // {
    //   int _d_x = 0;
    //   if (int x = 1; ...) {...}
    // }
    // This is done to avoid variable names clashes.
    addToCurrentBlock(initResult.getStmt_dx());

    VarDecl* condVarClone = nullptr;
    if (const VarDecl* condVarDecl = If->getConditionVariable()) {
      VarDeclDiff condVarDeclDiff = DifferentiateVarDecl(condVarDecl);
      condVarClone = condVarDeclDiff.getDecl();
      if (condVarDeclDiff.getDecl_dx())
        addToCurrentBlock(BuildDeclStmt(condVarDeclDiff.getDecl_dx()));
    }

    // Condition is just cloned as it is, not derived.
    // FIXME: if condition changes one of the variables, it may be reasonable
    // to derive it, e.g.
    // if (x += x) {...}
    // should result in:
    // {
    //   _d_y += _d_x
    //   if (y += x) {...}
    // }
    Expr* cond = Clone(If->getCond());

    auto VisitBranch = [this](const Stmt* Branch) -> Stmt* {
      if (!Branch)
        return nullptr;

      if (isa<CompoundStmt>(Branch)) {
        StmtDiff BranchDiff = Visit(Branch);
        return BranchDiff.getStmt();
      } else {
        beginBlock();
        beginScope(Scope::DeclScope);
        StmtDiff BranchDiff = Visit(Branch);
        for (Stmt* S : BranchDiff.getBothStmts())
          addToCurrentBlock(S);
        CompoundStmt* Block = endBlock();
        endScope();
        if (Block->size() == 1)
          return Block->body_front();
        else
          return Block;
      }
    };

    Stmt* thenDiff = VisitBranch(If->getThen());
    Stmt* elseDiff = VisitBranch(If->getElse());

    Stmt* ifDiff = clad_compat::IfStmt_Create(m_Context,
                                              noLoc,
                                              If->isConstexpr(),
                                              initResult.getStmt(),
                                              condVarClone,
                                              cond,
                                              noLoc,
                                              noLoc,
                                              thenDiff,
                                              noLoc,
                                              elseDiff);
    addToCurrentBlock(ifDiff);
    CompoundStmt* Block = endBlock();
    // If IfStmt is the only statement in the block, remove the block:
    endScope();
    // {
    //   if (...) {...}
    // }
    // ->
    // if (...) {...}
    StmtDiff Result = (Block->size() == 1) ? StmtDiff(ifDiff) : StmtDiff(Block);
    return Result;
  }

  StmtDiff
  ForwardModeVisitor::VisitConditionalOperator(const ConditionalOperator* CO) {
    Expr* cond = Clone(CO->getCond());
    // FIXME: fix potential side-effects from evaluating both sides of
    // conditional.
    StmtDiff ifTrueDiff = Visit(CO->getTrueExpr());
    StmtDiff ifFalseDiff = Visit(CO->getFalseExpr());

    cond = StoreAndRef(cond);
    cond = m_Sema
               .ActOnCondition(
                   m_CurScope, noLoc, cond, Sema::ConditionKind::Boolean)
               .get()
               .second;

    Expr* condExpr =
        m_Sema
            .ActOnConditionalOp(
                noLoc, noLoc, cond, ifTrueDiff.getExpr(), ifFalseDiff.getExpr())
            .get();

    Expr* condExprDiff = m_Sema
                             .ActOnConditionalOp(noLoc,
                                                 noLoc,
                                                 cond,
                                                 ifTrueDiff.getExpr_dx(),
                                                 ifFalseDiff.getExpr_dx())
                             .get();

    return StmtDiff(condExpr, condExprDiff);
  }

  StmtDiff ForwardModeVisitor::VisitForStmt(const ForStmt* FS) {
    beginScope(Scope::DeclScope | Scope::ControlScope | Scope::BreakScope |
               Scope::ContinueScope);
    beginBlock();
    const Stmt* init = FS->getInit();
    StmtDiff initDiff = init ? Visit(init) : StmtDiff{};
    addToCurrentBlock(initDiff.getStmt_dx());
    VarDecl* condVarDecl = FS->getConditionVariable();
    VarDecl* condVarClone = nullptr;
    if (condVarDecl) {
      VarDeclDiff condVarResult = DifferentiateVarDecl(condVarDecl);
      condVarClone = condVarResult.getDecl();
      if (condVarResult.getDecl_dx())
        addToCurrentBlock(BuildDeclStmt(condVarResult.getDecl_dx()));
    }
    Expr* cond = FS->getCond() ? Clone(FS->getCond()) : nullptr;
    const Expr* inc = FS->getInc();

    // Differentiate the increment expression of the for loop
    beginBlock();
    StmtDiff incDiff = inc ? Visit(inc) : StmtDiff{};
    CompoundStmt* decls = endBlock();
    Expr* incResult = nullptr;
    if (decls->size()) {
      // If differentiation of the increment produces a statement for
      // temporary variable declaration, enclose the increment in lambda
      // since only expressions are allowed in the increment part of the for
      // loop. E.g.:
      // for (...; ...; x = x * std::sin(x))
      // ->
      // for (int i = 0; i < 10; [&] {
      //  double _t1 = std::sin(x);
      //  _d_x = _d_x * _t1 + x * custom_derivatives::sin_darg0(x) * (_d_x);
      //  x = x * _t1;
      // }())

      incResult = wrapInLambda(*this, m_Sema, inc, [&] {
        StmtDiff incDiff = inc ? Visit(inc) : StmtDiff{};
        addToCurrentBlock(incDiff.getStmt_dx());
        addToCurrentBlock(incDiff.getStmt());
      });
    } else if (incDiff.getExpr_dx() && incDiff.getExpr()) {
      // If no declarations are required and only two Expressions are produced,
      // join them with comma expression.
      if (!isUnusedResult(incDiff.getExpr_dx()))
        incResult = BuildOp(BO_Comma,
                            BuildParens(incDiff.getExpr_dx()),
                            BuildParens(incDiff.getExpr()));
      else
        incResult = incDiff.getExpr();
    } else if (incDiff.getExpr()) {
      incResult = incDiff.getExpr();
    }

    const Stmt* body = FS->getBody();
    beginScope(Scope::DeclScope);
    Stmt* bodyResult = nullptr;
    if (isa<CompoundStmt>(body)) {
      bodyResult = Visit(body).getStmt();
    } else {
      beginBlock();
      StmtDiff Result = Visit(body);
      for (Stmt* S : Result.getBothStmts())
        addToCurrentBlock(S);
      CompoundStmt* Block = endBlock();
      if (Block->size() == 1)
        bodyResult = Block->body_front();
      else
        bodyResult = Block;
    }
    endScope();

    Stmt* forStmtDiff = new (m_Context) ForStmt(m_Context,
                                                initDiff.getStmt(),
                                                cond,
                                                condVarClone,
                                                incResult,
                                                bodyResult,
                                                noLoc,
                                                noLoc,
                                                noLoc);

    addToCurrentBlock(forStmtDiff);
    CompoundStmt* Block = endBlock();
    endScope();

    StmtDiff Result =
        (Block->size() == 1) ? StmtDiff(forStmtDiff) : StmtDiff(Block);
    return Result;
  }

  StmtDiff ForwardModeVisitor::VisitReturnStmt(const ReturnStmt* RS) {
    StmtDiff retValDiff = Visit(RS->getRetValue());
    Stmt* returnStmt =
        m_Sema
            .ActOnReturnStmt(noLoc,
                             retValDiff.getExpr_dx(), // return the derivative
                             m_CurScope)
            .get();
    return StmtDiff(returnStmt);
  }

  StmtDiff ForwardModeVisitor::VisitParenExpr(const ParenExpr* PE) {
    StmtDiff subStmtDiff = Visit(PE->getSubExpr());
    return StmtDiff(BuildParens(subStmtDiff.getExpr()),
                    BuildParens(subStmtDiff.getExpr_dx()));
  }

  StmtDiff ForwardModeVisitor::VisitMemberExpr(const MemberExpr* ME) {
    auto clonedME = dyn_cast<MemberExpr>(Clone(ME));
    // Copy paste from VisitDeclRefExpr.
    QualType Ty = ME->getType();
    if (clonedME->getMemberDecl() == m_IndependentVar)
      return StmtDiff(clonedME,
                      ConstantFolder::synthesizeLiteral(Ty, m_Context, 1));
    return StmtDiff(clonedME,
                    ConstantFolder::synthesizeLiteral(Ty, m_Context, 0));
  }

  StmtDiff ForwardModeVisitor::VisitInitListExpr(const InitListExpr* ILE) {
    llvm::SmallVector<Expr*, 16> clonedExprs(ILE->getNumInits());
    llvm::SmallVector<Expr*, 16> derivedExprs(ILE->getNumInits());
    for (unsigned i = 0, e = ILE->getNumInits(); i < e; i++) {
      StmtDiff ResultI = Visit(ILE->getInit(i));
      clonedExprs[i] = ResultI.getExpr();
      derivedExprs[i] = ResultI.getExpr_dx();
    }

    Expr* clonedILE = m_Sema.ActOnInitList(noLoc, clonedExprs, noLoc).get();
    Expr* derivedILE = m_Sema.ActOnInitList(noLoc, derivedExprs, noLoc).get();
    return StmtDiff(clonedILE, derivedILE);
  }

  StmtDiff
  ForwardModeVisitor::VisitArraySubscriptExpr(const ArraySubscriptExpr* ASE) {
    auto ASI = SplitArraySubscript(ASE);
    const Expr* Base = ASI.first;
    const auto& Indices = ASI.second;
    Expr* clonedBase = Clone(Base);
    llvm::SmallVector<Expr*, 4> clonedIndices(Indices.size());
    std::transform(std::begin(Indices),
                   std::end(Indices),
                   std::begin(clonedIndices),
                   [this](const Expr* E) { return Clone(E); });
    auto cloned = BuildArraySubscript(clonedBase, clonedIndices);

    auto zero =
        ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
    if (!isa<DeclRefExpr>(clonedBase->IgnoreParenImpCasts()))
      return StmtDiff(cloned, zero);
    auto DRE = cast<DeclRefExpr>(clonedBase->IgnoreParenImpCasts());
    if (!isa<VarDecl>(DRE->getDecl()))
      return StmtDiff(cloned, zero);
    auto VD = cast<VarDecl>(DRE->getDecl());
    if (VD == m_IndependentVar) {
      llvm::APSInt index;
      Expr* diffExpr = nullptr;

      if (!clad_compat::Expr_EvaluateAsInt(
              clonedIndices.back(), index, m_Context)) {
        diffExpr = BuildParens(
            BuildOp(BO_EQ,
                    clonedIndices.back(),
                    ConstantFolder::synthesizeLiteral(
                        m_Context.IntTy, m_Context, m_IndependentVarIndex)));
      } else if (index.getExtValue() == m_IndependentVarIndex) {
        diffExpr =
            ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 1);
      } else {
        diffExpr = zero;
      }
      return StmtDiff(cloned, diffExpr);
    }
    // Check DeclRefExpr is a reference to an independent variable.
    auto it = m_Variables.find(VD);
    if (it == std::end(m_Variables))
      // Is not an independent variable, ignored.
      return StmtDiff(cloned, zero);

    Expr* target = it->second;
    // FIXME: fix when adding array inputs
    if (!isArrayOrPointerType(target->getType()))
      return StmtDiff(cloned, zero);
    // llvm::APSInt IVal;
    // if (!I->EvaluateAsInt(IVal, m_Context))
    //  return;
    // Create the _result[idx] expression.
    auto result_at_is = BuildArraySubscript(target, clonedIndices);
    return StmtDiff(cloned, result_at_is);
  }

  StmtDiff ForwardModeVisitor::VisitDeclRefExpr(const DeclRefExpr* DRE) {
    DeclRefExpr* clonedDRE = nullptr;
    // Check if referenced Decl was "replaced" with another identifier inside
    // the derivative
    if (auto VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      auto it = m_DeclReplacements.find(VD);
      if (it != std::end(m_DeclReplacements))
        clonedDRE = BuildDeclRef(it->second);
      else
        clonedDRE = cast<DeclRefExpr>(Clone(DRE));
      // If current context is different than the context of the original
      // declaration (e.g. we are inside lambda), rebuild the DeclRefExpr
      // with Sema::BuildDeclRefExpr. This is required in some cases, e.g.
      // Sema::BuildDeclRefExpr is responsible for adding captured fields
      // to the underlying struct of a lambda.
      if (clonedDRE->getDecl()->getDeclContext() != m_Sema.CurContext) {
        auto referencedDecl = cast<VarDecl>(clonedDRE->getDecl());
        clonedDRE = cast<DeclRefExpr>(BuildDeclRef(referencedDecl));
      }
    } else
      clonedDRE = cast<DeclRefExpr>(Clone(DRE));

    if (auto VD = dyn_cast<VarDecl>(clonedDRE->getDecl())) {
      // If DRE references a variable, try to find if we know something about
      // how it is related to the independent variable.
      auto it = m_Variables.find(VD);
      if (it != std::end(m_Variables)) {
        // If a record was found, use the recorded derivative.
        auto dExpr = it->second;
        if (auto dVarDRE = dyn_cast<DeclRefExpr>(dExpr)) {
          auto dVar = cast<VarDecl>(dVarDRE->getDecl());
          if (dVar->getDeclContext() != m_Sema.CurContext)
            dExpr = BuildDeclRef(dVar);
        }
        return StmtDiff(clonedDRE, dExpr);
      }
    }
    // Is not a variable or is a reference to something unrelated to independent
    // variable. Derivative is 0.
    auto zero =
        ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
    return StmtDiff(clonedDRE, zero);
  }

  StmtDiff ForwardModeVisitor::VisitIntegerLiteral(const IntegerLiteral* IL) {
    llvm::APInt zero(m_Context.getIntWidth(m_Context.IntTy), /*value*/ 0);
    auto constant0 =
        IntegerLiteral::Create(m_Context, zero, m_Context.IntTy, noLoc);
    return StmtDiff(Clone(IL), constant0);
  }

  StmtDiff ForwardModeVisitor::VisitFloatingLiteral(const FloatingLiteral* FL) {
    llvm::APFloat zero = llvm::APFloat::getZero(FL->getSemantics());
    auto constant0 =
        FloatingLiteral::Create(m_Context, zero, true, FL->getType(), noLoc);
    return StmtDiff(Clone(FL), constant0);
  }

  // This method is derived from the source code of both
  // buildOverloadedCallSet() in SemaOverload.cpp
  // and ActOnCallExpr() in SemaExpr.cpp.
  bool
  DerivativeBuilder::noOverloadExists(Expr* UnresolvedLookup,
                                      llvm::MutableArrayRef<Expr*> ARargs) {
    if (UnresolvedLookup->getType() == m_Context.OverloadTy) {
      OverloadExpr::FindResult find = OverloadExpr::find(UnresolvedLookup);

      if (!find.HasFormOfMemberPointer) {
        OverloadExpr* ovl = find.Expression;

        if (isa<UnresolvedLookupExpr>(ovl)) {
          ExprResult result;
          SourceLocation Loc;
          OverloadCandidateSet CandidateSet(Loc,
                                            OverloadCandidateSet::CSK_Normal);
          Scope* S = m_Sema.getScopeForContext(m_Sema.CurContext);
          UnresolvedLookupExpr* ULE = cast<UnresolvedLookupExpr>(ovl);
          // Populate CandidateSet.
          m_Sema.buildOverloadedCallSet(
              S, UnresolvedLookup, ULE, ARargs, Loc, &CandidateSet, &result);

          OverloadCandidateSet::iterator Best;
          OverloadingResult OverloadResult = CandidateSet.BestViableFunction(
              m_Sema, UnresolvedLookup->getBeginLoc(), Best);
          if (OverloadResult) // No overloads were found.
            return true;
        }
      }
    }
    return false;
  }

  static NamespaceDecl* LookupBuiltinDerivativesNSD(ASTContext& C, Sema& S) {
    // Find the builtin derivatives namespace
    DeclarationName Name = &C.Idents.get("custom_derivatives");
    LookupResult R(S,
                   Name,
                   SourceLocation(),
                   Sema::LookupNamespaceName,
                   clad_compat::Sema_ForVisibleRedeclaration);
    S.LookupQualifiedName(R,
                          C.getTranslationUnitDecl(),
                          /*allowBuiltinCreation*/ false);
    assert(!R.empty() && "Cannot find builtin derivatives!");
    return cast<NamespaceDecl>(R.getFoundDecl());
  }

  Expr* DerivativeBuilder::findOverloadedDefinition(
      DeclarationNameInfo DNI, llvm::SmallVectorImpl<Expr*>& CallArgs) {
    if (!m_BuiltinDerivativesNSD)
      m_BuiltinDerivativesNSD = LookupBuiltinDerivativesNSD(m_Context, m_Sema);

    LookupResult R(m_Sema, DNI, Sema::LookupOrdinaryName);
    m_Sema.LookupQualifiedName(R,
                               m_BuiltinDerivativesNSD,
                               /*allowBuiltinCreation*/ false);
    Expr* OverloadedFn = 0;
    if (!R.empty()) {
      CXXScopeSpec CSS;
      CSS.Extend(m_Context, m_BuiltinDerivativesNSD, noLoc, noLoc);
      Expr* UnresolvedLookup =
          m_Sema.BuildDeclarationNameExpr(CSS, R, /*ADL*/ false).get();

      llvm::MutableArrayRef<Expr*> MARargs =
          llvm::MutableArrayRef<Expr*>(CallArgs);

      SourceLocation Loc;
      Scope* S = m_Sema.getScopeForContext(m_Sema.CurContext);

      if (noOverloadExists(UnresolvedLookup, MARargs)) {
        return 0;
      }

      OverloadedFn =
          m_Sema.ActOnCallExpr(S, UnresolvedLookup, Loc, MARargs, Loc).get();
    }
    return OverloadedFn;
  }

  StmtDiff ForwardModeVisitor::VisitCallExpr(const CallExpr* CE) {
    const FunctionDecl* FD = CE->getDirectCallee();
    if (!FD) {
      diag(DiagnosticsEngine::Warning,
           CE->getBeginLoc(),
           "Differentiation of only direct calls is supported. Ignored");
      return StmtDiff(Clone(CE));
    }
    // Find the built-in derivatives namespace.
    std::string s = std::to_string(m_DerivativeOrder);
    if (m_DerivativeOrder == 1)
      s = "";

    IdentifierInfo* II =
        &m_Context.Idents.get(FD->getNameAsString() + "_d" + s + "arg0");
    DeclarationName name(II);
    SourceLocation DeclLoc;
    DeclarationNameInfo DNInfo(name, DeclLoc);

    SourceLocation noLoc;
    llvm::SmallVector<Expr*, 4> CallArgs{};
    // For f(g(x)) = f'(x) * g'(x)
    Expr* Multiplier = nullptr;
    for (size_t i = 0, e = CE->getNumArgs(); i < e; ++i) {
      StmtDiff argDiff = Visit(CE->getArg(i));
      if (!Multiplier)
        Multiplier = argDiff.getExpr_dx();
      else {
        Multiplier = BuildOp(BO_Add, Multiplier, argDiff.getExpr_dx());
      }
      CallArgs.push_back(argDiff.getExpr());
    }

    Expr* call = m_Sema
                     .ActOnCallExpr(getCurrentScope(),
                                    Clone(CE->getCallee()),
                                    noLoc,
                                    llvm::MutableArrayRef<Expr*>(CallArgs),
                                    noLoc)
                     .get();

    // Try to find an overloaded derivative in 'custom_derivatives'
    Expr* callDiff = m_Builder.findOverloadedDefinition(DNInfo, CallArgs);

    // FIXME: add gradient-vector products to fix that.
    if (!callDiff)
      assert((CE->getNumArgs() <= 1) &&
             "forward differentiation of multi-arg calls is currently broken");

    // Check if it is a recursive call.
    if (!callDiff && (FD == m_Function)) {
      // The differentiated function is called recursively.
      Expr* derivativeRef =
          m_Sema
              .BuildDeclarationNameExpr(CXXScopeSpec(),
                                        m_Derivative->getNameInfo(),
                                        m_Derivative)
              .get();
      callDiff =
          m_Sema
              .ActOnCallExpr(m_Sema.getScopeForContext(m_Sema.CurContext),
                             derivativeRef,
                             noLoc,
                             llvm::MutableArrayRef<Expr*>(CallArgs),
                             noLoc)
              .get();
    }

    if (!callDiff) {
      // Overloaded derivative was not found, request the CladPlugin to
      // derive the called function.
      DiffRequest request{};
      request.Function = FD;
      request.BaseFunctionName = FD->getNameAsString();
      request.Mode = DiffMode::forward;
      // Silence diag outputs in nested derivation process.
      request.VerboseDiags = false;

      FunctionDecl* derivedFD =
          plugin::ProcessDiffRequest(m_CladPlugin, request);
      // Clad failed to derive it.
      if (!derivedFD) {
        // Function was not derived => issue a warning.
        diag(DiagnosticsEngine::Warning,
             CE->getBeginLoc(),
             "function '%0' was not differentiated because clad failed to "
             "differentiate it and no suitable overload was found in "
             "namespace 'custom_derivatives'",
             {FD->getNameAsString()});

        auto zero =
            ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
        return StmtDiff(call, zero);
      }

      callDiff = m_Sema
                     .ActOnCallExpr(getCurrentScope(),
                                    BuildDeclRef(derivedFD),
                                    noLoc,
                                    llvm::MutableArrayRef<Expr*>(CallArgs),
                                    noLoc)
                     .get();
    }

    if (Multiplier)
      callDiff = BuildOp(BO_Mul, callDiff, BuildParens(Multiplier));
    return StmtDiff(call, callDiff);
  }

  void VisitorBase::updateReferencesOf(Stmt* InSubtree) {
    utils::ReferencesUpdater up(m_Sema,
                                m_Builder.m_NodeCloner.get(),
                                getCurrentScope());
    up.TraverseStmt(InSubtree);
  }

  StmtDiff ForwardModeVisitor::VisitUnaryOperator(const UnaryOperator* UnOp) {
    StmtDiff diff = Visit(UnOp->getSubExpr());
    auto opKind = UnOp->getOpcode();
    Expr* op = BuildOp(opKind, diff.getExpr());
    // If opKind is unary plus or minus, apply that op to derivative.
    // Otherwise, the derivative is 0.
    // FIXME: add support for other unary operators
    if (opKind == UO_Plus || opKind == UO_Minus)
      return StmtDiff(op, BuildOp(opKind, diff.getExpr_dx()));
    else if (opKind == UO_PostInc || opKind == UO_PostDec ||
             opKind == UO_PreInc || opKind == UO_PreDec) {
      return StmtDiff(op, diff.getExpr_dx());
    } else {
      unsupportedOpWarn(UnOp->getEndLoc());
      auto zero =
          ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
      return StmtDiff(op, zero);
    }
  }

  StmtDiff
  ForwardModeVisitor::VisitBinaryOperator(const BinaryOperator* BinOp) {
    StmtDiff Ldiff = Visit(BinOp->getLHS());
    StmtDiff Rdiff = Visit(BinOp->getRHS());

    ConstantFolder folder(m_Context);
    auto opCode = BinOp->getOpcode();
    Expr* opDiff = nullptr;

    auto deriveMul = [this](StmtDiff& Ldiff, StmtDiff& Rdiff) {
      Expr* LHS = BuildOp(BO_Mul,
                          BuildParens(Ldiff.getExpr_dx()),
                          BuildParens(Rdiff.getExpr()));

      Expr* RHS = BuildOp(BO_Mul,
                          BuildParens(Ldiff.getExpr()),
                          BuildParens(Rdiff.getExpr_dx()));

      return BuildOp(BO_Add, LHS, RHS);
    };

    auto deriveDiv = [this](StmtDiff& Ldiff, StmtDiff& Rdiff) {
      Expr* LHS = BuildOp(BO_Mul,
                          BuildParens(Ldiff.getExpr_dx()),
                          BuildParens(Rdiff.getExpr()));

      Expr* RHS = BuildOp(BO_Mul,
                          BuildParens(Ldiff.getExpr()),
                          BuildParens(Rdiff.getExpr_dx()));

      Expr* nominator = BuildOp(BO_Sub, LHS, RHS);

      Expr* RParens = BuildParens(Rdiff.getExpr());
      Expr* denominator = BuildOp(BO_Mul, RParens, RParens);

      return BuildOp(BO_Div, BuildParens(nominator), BuildParens(denominator));
    };

    if (opCode == BO_Mul) {
      // If Ldiff.getExpr() and Rdiff.getExpr() require evaluation, store the
      // expressions in variables to avoid reevaluation.
      Ldiff = {StoreAndRef(Ldiff.getExpr()), Ldiff.getExpr_dx()};
      Rdiff = {StoreAndRef(Rdiff.getExpr()), Rdiff.getExpr_dx()};

      opDiff = deriveMul(Ldiff, Rdiff);
    } else if (opCode == BO_Div) {
      Ldiff = {StoreAndRef(Ldiff.getExpr()), Ldiff.getExpr_dx()};
      Rdiff = {StoreAndRef(Rdiff.getExpr()), Rdiff.getExpr_dx()};

      opDiff = deriveDiv(Ldiff, Rdiff);
    } else if (opCode == BO_Add)
      opDiff = BuildOp(BO_Add, Ldiff.getExpr_dx(), Rdiff.getExpr_dx());
    else if (opCode == BO_Sub)
      opDiff =
          BuildOp(BO_Sub, Ldiff.getExpr_dx(), BuildParens(Rdiff.getExpr_dx()));
    else if (BinOp->isAssignmentOp()) {
      if (Ldiff.getExpr_dx()->isModifiableLvalue(m_Context) !=
          Expr::MLV_Valid) {
        diag(DiagnosticsEngine::Warning,
             BinOp->getEndLoc(),
             "derivative of an assignment attempts to assign to unassignable "
             "expr, assignment ignored");
        opDiff =
            ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
      } else if (opCode == BO_Assign || opCode == BO_AddAssign ||
                 opCode == BO_SubAssign)
        opDiff = BuildOp(opCode, Ldiff.getExpr_dx(), Rdiff.getExpr_dx());
      else if (opCode == BO_MulAssign) {
        Ldiff = {StoreAndRef(Ldiff.getExpr()), Ldiff.getExpr_dx()};
        Rdiff = {StoreAndRef(Rdiff.getExpr()), Rdiff.getExpr_dx()};
        opDiff =
            BuildOp(BO_Assign, Ldiff.getExpr_dx(), deriveMul(Ldiff, Rdiff));
      } else if (opCode == BO_DivAssign) {
        Ldiff = {StoreAndRef(Ldiff.getExpr()), Ldiff.getExpr_dx()};
        Rdiff = {StoreAndRef(Rdiff.getExpr()), Rdiff.getExpr_dx()};
        opDiff =
            BuildOp(BO_Assign, Ldiff.getExpr_dx(), deriveDiv(Ldiff, Rdiff));
      }
    } else if (opCode == BO_Comma) {
      if (!isUnusedResult(Ldiff.getExpr_dx()))
        opDiff = BuildOp(BO_Comma,
                         BuildParens(Ldiff.getExpr_dx()),
                         BuildParens(Rdiff.getExpr_dx()));
      else
        opDiff = Rdiff.getExpr_dx();
    }
    if (!opDiff) {
      // FIXME: add support for other binary operators
      unsupportedOpWarn(BinOp->getEndLoc());
      opDiff = ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
    }
    opDiff = folder.fold(opDiff);
    // Recover the original operation from the Ldiff and Rdiff instead of
    // cloning the tree.
    Expr* op = BuildOp(opCode, Ldiff.getExpr(), Rdiff.getExpr());

    return StmtDiff(op, opDiff);
  }

  VarDeclDiff ForwardModeVisitor::DifferentiateVarDecl(const VarDecl* VD) {
    StmtDiff initDiff = VD->getInit() ? Visit(VD->getInit()) : StmtDiff{};
    VarDecl* VDClone = BuildVarDecl(VD->getType(),
                                    VD->getNameAsString(),
                                    initDiff.getExpr(),
                                    VD->isDirectInit());
    VarDecl* VDDerived = BuildVarDecl(VD->getType(),
                                      "_d_" + VD->getNameAsString(),
                                      initDiff.getExpr_dx());
    m_Variables.emplace(VDClone, BuildDeclRef(VDDerived));
    return VarDeclDiff(VDClone, VDDerived);
  }

  StmtDiff ForwardModeVisitor::VisitDeclStmt(const DeclStmt* DS) {
    llvm::SmallVector<Decl*, 4> decls;
    llvm::SmallVector<Decl*, 4> declsDiff;
    // For each variable declaration v, create another declaration _d_v to
    // store derivatives for potential reassignments. E.g.
    // double y = x;
    // ->
    // double _d_y = _d_x; double y = x;
    for (auto D : DS->decls()) {
      if (auto VD = dyn_cast<VarDecl>(D)) {
        VarDeclDiff VDDiff = DifferentiateVarDecl(VD);
        // Check if decl's name is the same as before. The name may be changed
        // if decl name collides with something in the derivative body.
        // This can happen in rare cases, e.g. when the original function
        // has both y and _d_y (here _d_y collides with the name produced by
        // the derivation process), e.g.
        // double f(double x) {
        //   double y = x;
        //   double _d_y = x;
        // }
        // ->
        // double f_darg0(double x) {
        //   double _d_x = 1;
        //   double _d_y = _d_x; // produced as a derivative for y
        //   double y = x;
        //   double _d__d_y = _d_x;
        //   double _d_y = x; // copied from original funcion, collides with
        //   _d_y
        // }
        if (VDDiff.getDecl()->getDeclName() != VD->getDeclName())
          m_DeclReplacements[VD] = VDDiff.getDecl();
        decls.push_back(VDDiff.getDecl());
        declsDiff.push_back(VDDiff.getDecl_dx());
      } else {
        diag(DiagnosticsEngine::Warning,
             D->getEndLoc(),
             "Unsupported declaration");
      }
    }

    Stmt* DSClone = BuildDeclStmt(decls);
    Stmt* DSDiff = BuildDeclStmt(declsDiff);
    return StmtDiff(DSClone, DSDiff);
  }

  StmtDiff
  ForwardModeVisitor::VisitImplicitCastExpr(const ImplicitCastExpr* ICE) {
    StmtDiff subExprDiff = Visit(ICE->getSubExpr());
    // Casts should be handled automatically when the result is used by
    // Sema::ActOn.../Build...
    return StmtDiff(subExprDiff.getExpr(), subExprDiff.getExpr_dx());
  }

  StmtDiff ForwardModeVisitor::VisitCXXOperatorCallExpr(
      const CXXOperatorCallExpr* OpCall) {
    // This operator gets emitted when there is a binary operation containing
    // overloaded operators. Eg. x+y, where operator+ is overloaded.
    diag(DiagnosticsEngine::Error,
         OpCall->getEndLoc(),
         "We don't support overloaded operators yet!");
    return {};
  }

  StmtDiff
  ForwardModeVisitor::VisitCXXDefaultArgExpr(const CXXDefaultArgExpr* DE) {
    return Visit(DE->getExpr());
  }

  StmtDiff
  ForwardModeVisitor::VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr* BL) {
    llvm::APInt zero(m_Context.getIntWidth(m_Context.IntTy), /*value*/ 0);
    auto constant0 =
        IntegerLiteral::Create(m_Context, zero, m_Context.IntTy, noLoc);
    return StmtDiff(Clone(BL), constant0);
  }
} // end namespace clad