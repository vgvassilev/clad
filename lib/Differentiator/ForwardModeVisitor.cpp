//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clad/Differentiator/ForwardModeVisitor.h"

#include "ConstantFolder.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/ErrorEstimator.h"
#include "clad/Differentiator/StmtClone.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/TemplateBase.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
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
  ForwardModeVisitor::DerivePushforward(const FunctionDecl* FD,
                                        const DiffRequest& request) {
    m_Function = FD;
    m_Functor = request.Functor;
    m_DerivativeOrder = request.CurrentDerivativeOrder;
    m_Mode = DiffMode::experimental_pushforward;
    assert(!m_DerivativeInFlight &&
           "Doesn't support recursive diff. Use DiffPlan.");
    m_DerivativeInFlight = true;

    IdentifierInfo* derivedFnII =
        &m_Context.Idents.get(m_Function->getNameAsString() + "_pushforward");
    DeclarationNameInfo derivedFnName(derivedFnII, noLoc);
    llvm::SmallVector<QualType, 16> paramTypes, derivedParamTypes;

    for (auto* PVD : m_Function->parameters()) {
      paramTypes.push_back(PVD->getType());

      // Pushforward functions currently only support real parameters
      QualType nonRefParamType = PVD->getType().getNonReferenceType();
      if (nonRefParamType->isRealType())
        derivedParamTypes.push_back(PVD->getType());
    }
    paramTypes.insert(paramTypes.end(), derivedParamTypes.begin(),
                      derivedParamTypes.end());

    auto originalFnType = dyn_cast<FunctionProtoType>(m_Function->getType());
    QualType derivedFnType =
        m_Context.getFunctionType(m_Function->getReturnType(), paramTypes,
                                  originalFnType->getExtProtoInfo());
    llvm::SaveAndRestore<DeclContext*> saveContext(m_Sema.CurContext);
    llvm::SaveAndRestore<Scope*> saveScope(m_CurScope);
    DeclContext* DC = const_cast<DeclContext*>(m_Function->getDeclContext());
    m_Sema.CurContext = DC;

    DeclWithContext cloneFunctionResult =
        m_Builder.cloneFunction(m_Function, *this, DC, m_Sema, m_Context, noLoc,
                                derivedFnName, derivedFnType);
    m_Derivative = cloneFunctionResult.first;

    llvm::SmallVector<ParmVarDecl*, 16> params;
    llvm::SmallVector<ParmVarDecl*, 16> derivedParams;
    beginScope(Scope::FunctionPrototypeScope | Scope::FunctionDeclarationScope |
               Scope::DeclScope);
    m_Sema.PushFunctionScope();
    m_Sema.PushDeclContext(getCurrentScope(), m_Derivative);

    std::size_t numParamsOriginalFn = m_Function->getNumParams();
    for (std::size_t i = 0; i < numParamsOriginalFn; ++i) {
      auto PVD = m_Function->getParamDecl(i);
      auto newPVD = CloneParmVarDecl(PVD, PVD->getIdentifier(),
                                     /*pushOnScopeChains=*/true);
      params.push_back(newPVD);

      QualType nonRefParamType = PVD->getType().getNonReferenceType();
      if (!nonRefParamType->isRealType())
        continue;
      auto derivedPVDName = "_d_" + PVD->getNameAsString();
      auto derivedPVDII = &m_Context.Idents.get(derivedPVDName);
      // TODO: Check for name conflicts.
      auto derivedPVD = CloneParmVarDecl(PVD, derivedPVDII,
                                         /*pushOnScopeChains=*/true);
      derivedParams.push_back(derivedPVD);
      m_Variables[newPVD] = BuildDeclRef(derivedPVD);
    }

    params.insert(params.end(), derivedParams.begin(), derivedParams.end());
    m_Derivative->setParams(params);
    m_Derivative->setBody(nullptr);

    beginScope(Scope::FnScope | Scope::DeclScope);
    m_DerivativeFnScope = getCurrentScope();
    beginBlock();

    Stmt* bodyDiff = Visit(FD->getBody()).getStmt();
    CompoundStmt* CS = cast<CompoundStmt>(bodyDiff);
    for (Stmt* S : CS->body())
      addToCurrentBlock(S);

    Stmt* derivativeBody = endBlock();
    m_Derivative->setBody(derivativeBody);

    endScope(); // Function body scope
    m_Sema.PopFunctionScopeInfo();
    m_Sema.PopDeclContext();
    endScope(); // Function decl scope

    m_DerivativeInFlight = false;
    return OverloadedDeclWithContext{cloneFunctionResult.first,
                                     cloneFunctionResult.second,
                                     /*OverloadFunctionDecl=*/nullptr};
  }

  OverloadedDeclWithContext
  ForwardModeVisitor::Derive(const FunctionDecl* FD,
                             const DiffRequest& request) {
    silenceDiags = !request.VerboseDiags;
    m_Function = FD;
    m_Functor = request.Functor;
    m_Mode = DiffMode::forward;
    assert(!m_DerivativeInFlight &&
           "Doesn't support recursive diff. Use DiffPlan.");
    m_DerivativeInFlight = true;

    DiffParams args{};
    IndexIntervalTable indexIntervalTable{};
    if (request.Args)
      std::tie(args, indexIntervalTable) = request.DiffParamsInfo;
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
    if (m_DerivativeOrder == 1)
      s = "";

    // If we are differentiating a call operator, that has no parameters,
    // then the specified independent argument is a member variable of the
    // class defining the call operator. 
    // Thus, we need to find index of the member variable instead.
    if (m_Function->param_empty() && m_Functor) {
      m_ArgIndex = std::distance(m_Functor->field_begin(),
                                 std::find(m_Functor->field_begin(),
                                           m_Functor->field_end(),
                                           m_IndependentVar));
    } else {
      m_ArgIndex = std::distance(FD->param_begin(),
                                 std::find(FD->param_begin(), FD->param_end(),
                                           m_IndependentVar));
    }

    IdentifierInfo* II =
        &m_Context.Idents.get(request.BaseFunctionName + "_d" + s + "arg" +
                              std::to_string(m_ArgIndex) + derivativeSuffix);
    SourceLocation loc{m_Function->getLocation()};
    DeclarationNameInfo name(II, loc);
    llvm::SaveAndRestore<DeclContext*> SaveContext(m_Sema.CurContext);
    llvm::SaveAndRestore<Scope*> SaveScope(m_CurScope);
    DeclContext* DC = const_cast<DeclContext*>(m_Function->getDeclContext());
    m_Sema.CurContext = DC;
    DeclWithContext result =
        m_Builder.cloneFunction(FD, *this, DC, m_Sema, m_Context, loc, name,
                                FD->getType());
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

    // Create derived variable for each member variable if we are
    // differentiating a call operator.
    if (m_Functor) {
      for (FieldDecl* fieldDecl : m_Functor->fields()) {
        Expr* dInitializer = nullptr;
        QualType fieldType = fieldDecl->getType();

        if (auto arrType = dyn_cast<ConstantArrayType>(
                fieldType.getTypePtr())) {
          if (!arrType->getElementType()->isRealType())
            continue;

          auto arrSize = arrType->getSize().getZExtValue();
          std::vector<Expr*> dArrVal;

          // Create an initializer list to initialize derived variable created
          // for array member variable.
          // For example, if we are differentiating wrt arr[3], then
          // ```
          // double arr[7];
          // ```
          // will get differentiated to,
          //
          // ```
          // double _d_arr[7] = {0, 0, 0, 1, 0, 0, 0};
          // ```
          for (size_t i = 0; i < arrSize; ++i) {
            int dValue = (fieldDecl == m_IndependentVar &&
                          i == m_IndependentVarIndex);
            auto dValueLiteral = ConstantFolder::synthesizeLiteral(m_Context
                                                                       .IntTy,
                                                                   m_Context,
                                                                   dValue);
            dArrVal.push_back(dValueLiteral);
          }
          dInitializer = m_Sema.ActOnInitList(noLoc, dArrVal, noLoc).get();
        } else if (auto ptrType = dyn_cast<PointerType>(
                       fieldType.getTypePtr())) {
          if (!ptrType->getPointeeType()->isRealType())
            continue;
          // Pointer member variables should be initialised by `nullptr`.
          dInitializer = m_Sema.ActOnCXXNullPtrLiteral(noLoc).get();
        } else {
          int dValue = (fieldDecl == m_IndependentVar);
          dInitializer = ConstantFolder::synthesizeLiteral(m_Context.IntTy,
                                                           m_Context, dValue);
        }
        VarDecl*
            derivedFieldDecl = BuildVarDecl(fieldType.getNonReferenceType(),
                                            "_d_" +
                                                fieldDecl->getNameAsString(),
                                            dInitializer);
        addToCurrentBlock(BuildDeclStmt(derivedFieldDecl));
        m_Variables.emplace(fieldDecl, BuildDeclRef(derivedFieldDecl));
      }
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

    return OverloadedDeclWithContext{result.first, result.second,
                                     /*OverloadFunctionDecl=*/nullptr};
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
    auto memberDecl = ME->getMemberDecl();
    auto clonedME = dyn_cast<MemberExpr>(Clone(ME));

    // Currently, we only differentiate member variables if we are
    // differentiating a call operator.
    if (m_Functor) {
      // Try to find the derivative of the member variable wrt independent
      // variable
      if (m_Variables.find(memberDecl) != std::end(m_Variables)) {
        return StmtDiff(clonedME, m_Variables[memberDecl]);
      }

      // Is not a real variable. Therefore, derivative is 0.
      auto zero = ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context,
                                                    0);
      return StmtDiff(clonedME, zero);
    } else {
      QualType Ty = ME->getType();
      return StmtDiff(clonedME,
                      ConstantFolder::synthesizeLiteral(Ty, m_Context, 0));
    }
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
    ValueDecl* VD;        
    // Derived variables for member variables are also created when we are 
    // differentiating a call operator.
    if (m_Functor) {
      if (auto ME = dyn_cast<MemberExpr>(clonedBase->IgnoreParenImpCasts())) {
        ValueDecl* decl = ME->getMemberDecl();
        auto it = m_Variables.find(decl);
        // If the original field is of constant array type, then,
        // the derived variable of `arr[i]` is `_d_arr[i]`.
        if (it != m_Variables.end() && decl->getType()->isConstantArrayType()) {
          auto result_at_i = BuildArraySubscript(it->second, clonedIndices);
          return StmtDiff{cloned, result_at_i};
        }

        VD = decl;
      }
    } else {
      if (!isa<DeclRefExpr>(clonedBase->IgnoreParenImpCasts()))
        return StmtDiff(cloned, zero);
      auto DRE = cast<DeclRefExpr>(clonedBase->IgnoreParenImpCasts());
      assert(isa<VarDecl>(DRE->getDecl()) &&
             "declaration represented by clonedBase Should always be VarDecl "
             "when clonedBase is DeclRefExpr");
      VD = DRE->getDecl();
    }        
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

  static NamespaceDecl* LookupNSD(ASTContext& C, Sema& S,
                                  llvm::StringRef namespc, bool shouldExist) {
    // Find the builtin derivatives/numerical diff namespace
    DeclarationName Name = &C.Idents.get(namespc);
    LookupResult R(S,
                   Name,
                   SourceLocation(),
                   Sema::LookupNamespaceName,
                   clad_compat::Sema_ForVisibleRedeclaration);
    S.LookupQualifiedName(R,
                          C.getTranslationUnitDecl(),
                          /*allowBuiltinCreation*/ false);
    if (!shouldExist && R.empty())
      return nullptr;
    assert(!R.empty() && "Cannot find the specified namespace!");
    return cast<NamespaceDecl>(R.getFoundDecl());
  }

  Expr* DerivativeBuilder::findOverloadedDefinition(
      DeclarationNameInfo DNI, llvm::SmallVectorImpl<Expr*>& CallArgs,
      bool forCustomDerv /*=true*/, bool namespaceShouldExist /*=true*/) {
    NamespaceDecl* NSD;
    std::string namespaceID;
    if (forCustomDerv) {
      NSD = m_BuiltinDerivativesNSD;
      namespaceID = "custom_derivatives";
    } else {
      NSD = m_NumericalDiffNSD;
      namespaceID = "numerical_diff";
    }
    if (!NSD){
      NSD = LookupNSD(m_Context, m_Sema, namespaceID, namespaceShouldExist);
      if (!forCustomDerv && !NSD) {
        diag(DiagnosticsEngine::Warning, noLoc,
             "Numerical differentiation is diabled using the "
             "-DCLAD_NO_NUM_DIFF "
             "flag, this means that every try to numerically differentiate a "
             "function will fail! Remove the flag to revert to default "
             "behaviour.");
        return nullptr;
      }
    }
    LookupResult R(m_Sema, DNI, Sema::LookupOrdinaryName);
    m_Sema.LookupQualifiedName(R,
                               NSD,
                               /*allowBuiltinCreation*/ false);
    Expr* OverloadedFn = 0;
    if (!R.empty()) {
      CXXScopeSpec CSS;
      CSS.Extend(m_Context, NSD, noLoc, noLoc);
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
    llvm::SmallVector<Expr*, 4> diffArgs;
    // For f(g(x)) = f'(x) * g'(x)
    Expr* Multiplier = nullptr;
    for (size_t i = 0, e = CE->getNumArgs(); i < e; ++i) {
      const Expr* arg = CE->getArg(i);
      StmtDiff argDiff = Visit(arg);
      if (!Multiplier)
        Multiplier = argDiff.getExpr_dx();
      else {
        Multiplier = BuildOp(BO_Add, Multiplier, argDiff.getExpr_dx());
      }
      CallArgs.push_back(argDiff.getExpr());
      // FIXME: Add support for pointer and array arguments in the
      // pushforward mode.
      if (arg->getType().getNonReferenceType()->isRealType())
        diffArgs.push_back(argDiff.getExpr_dx());
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
    if (callDiff && Multiplier)
      callDiff = BuildOp(BO_Mul, callDiff, BuildParens(Multiplier));
      
    // Check if it is a recursive call.
    if (!callDiff && (FD == m_Function) && m_Mode == DiffMode::experimental_pushforward) {
      // The differentiated function is called recursively.
      Expr* derivativeRef =
          m_Sema
              .BuildDeclarationNameExpr(
                  CXXScopeSpec(), m_Derivative->getNameInfo(), m_Derivative)
              .get();
      CallArgs.insert(CallArgs.end(), diffArgs.begin(), diffArgs.end());
      callDiff =
          m_Sema
              .ActOnCallExpr(m_Sema.getScopeForContext(m_Sema.CurContext),
                             derivativeRef, noLoc,
                             llvm::MutableArrayRef<Expr*>(CallArgs), noLoc)
              .get();
    }

    if (!callDiff) {
      // Overloaded derivative was not found, request the CladPlugin to
      // derive the called function.
      DiffRequest pushforwardFnRequest;
      pushforwardFnRequest.Function = FD;
      pushforwardFnRequest.Mode = DiffMode::experimental_pushforward;
      pushforwardFnRequest.BaseFunctionName = FD->getNameAsString();
      // pushforwardFnRequest.RequestedDerivativeOrder = m_DerivativeOrder;
      // Silence diag outputs in nested derivation process.
      pushforwardFnRequest.VerboseDiags = false;
      FunctionDecl* pushforwardFD =
          plugin::ProcessDiffRequest(m_CladPlugin, pushforwardFnRequest);
      // If clad failed to derive it, try finding its derivative using
      // numerical diff.
      if (!pushforwardFD) {
        // FIXME: Extend this for multiarg support
        // Check if the function is eligible for numerical differentiation.
        if (CE->getNumArgs() == 1) {
          Expr* fnCallee = cast<CallExpr>(call)->getCallee();
          callDiff = GetSingleArgCentralDiffCall(fnCallee, CallArgs[0],
                                                 /*targetPos=*/0, /*numArgs=*/1,
                                                 CallArgs);
        }
        CallExprDiffDiagnostics(FD->getNameAsString(), CE->getBeginLoc(),
                                  callDiff);
        if (!callDiff) {
          auto zero =
              ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
          return StmtDiff(call, zero);
        }
        if (Multiplier)
          callDiff = BuildOp(BO_Mul, callDiff, BuildParens(Multiplier));
      } else {
        CallArgs.insert(CallArgs.end(), diffArgs.begin(), diffArgs.end());
        callDiff = m_Sema
                       .ActOnCallExpr(getCurrentScope(),
                                      BuildDeclRef(pushforwardFD),
                                      noLoc,
                                      llvm::MutableArrayRef<Expr*>(CallArgs),
                                      noLoc)
                       .get();
      }
    }
    return StmtDiff(call, callDiff);
  }

  // TODO: Move this to VisitorBase
  void VisitorBase::updateReferencesOf(Stmt* InSubtree) {
    utils::ReferencesUpdater up(m_Sema, m_Builder.m_NodeCloner.get(),
                                getCurrentScope(), m_Function);
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
      else if (opCode == BO_MulAssign || opCode == BO_DivAssign) {
        // if both original expression and derived expression and evaluatable,
        // then derived expression reference needs to be stored before
        // the original expression reference to correctly evaluate
        // the derivative. For example, 
        //
        // ```
        // (t *= x) *= 1;
        // ```
        //
        // Should evaluate to, 
        //
        // ```
        // double &_t0 = (_d_t = _d_t*x + t*_d_x); // derived statement 
        //                                            reference
        // double &_t1 = (t*=x);  // original statement reference
        // _t0 = _t0*1 + _t1*0;
        // _t1 *= 1;
        // ```
        // 
        auto LdiffExprDx = StoreAndRef(Ldiff.getExpr_dx());
        Ldiff = {StoreAndRef(Ldiff.getExpr()), LdiffExprDx};
        auto RdiffExprDx = StoreAndRef(Rdiff.getExpr_dx());
        Rdiff = {StoreAndRef(Rdiff.getExpr()), RdiffExprDx};
        if (opCode == BO_MulAssign)
          opDiff = BuildOp(BO_Assign, Ldiff.getExpr_dx(),
                           deriveMul(Ldiff, Rdiff));
        else if (opCode == BO_DivAssign)
          opDiff = BuildOp(BO_Assign, Ldiff.getExpr_dx(),
                           deriveDiv(Ldiff, Rdiff));
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

  StmtDiff ForwardModeVisitor::VisitWhileStmt(const WhileStmt* WS) {
    // begin scope for while loop
    beginScope(Scope::ContinueScope | Scope::BreakScope | Scope::DeclScope |
               Scope::ControlScope);

    const VarDecl* condVar = WS->getConditionVariable();
    VarDecl* condVarClone = nullptr;
    VarDeclDiff condVarRes;
    if (condVar) {
      condVarRes = DifferentiateVarDecl(condVar);
      condVarClone = condVarRes.getDecl();
    }
    Expr* condClone = WS->getCond() ? Clone(WS->getCond()) : nullptr;

    Sema::ConditionResult condRes;
    if (condVarClone) {
      condRes = m_Sema.ActOnConditionVariable(condVarClone, noLoc,
                                              Sema::ConditionKind::Boolean);
    } else {
      condRes = m_Sema.ActOnCondition(getCurrentScope(), noLoc, condClone,
                                      Sema::ConditionKind::Boolean);
    }

    const Stmt* body = WS->getBody();
    Stmt* bodyResult = nullptr;
    if (isa<CompoundStmt>(body)) {
      bodyResult = Visit(body).getStmt();
    } else {
      beginScope(Scope::DeclScope);
      beginBlock();
      StmtDiff Result = Visit(body);
      for (Stmt* S : Result.getBothStmts())
        addToCurrentBlock(S);
      CompoundStmt* Block = endBlock();
      endScope();
      bodyResult = Block;
    }
    // Since condition variable is created and initialized at each iteration,
    // derivative of condition variable should also get created and initialized
    // at each iteratrion. Therefore, we need to insert declaration statement
    // of derivative of condition variable, if any, on top of the derived body
    // of the while loop.
    //
    // while (double b = a) {
    //   ...
    //   ...
    // }
    //
    // gets differentiated to,
    //
    // while (double b = a) {
    //   double _d_b = _d_a;
    //   ...
    //   ...
    // }
    if (condVarClone) {
      bodyResult =
          utils::PrependAndCreateCompoundStmt(m_Sema.getASTContext(),
                                       cast<CompoundStmt>(bodyResult),
                                       BuildDeclStmt(condVarRes.getDecl_dx()));
    }

    Stmt* WSDiff = clad_compat::Sema_ActOnWhileStmt(m_Sema, condRes, bodyResult)
                      .get();
    // end scope for while loop
    endScope();
    return StmtDiff(WSDiff);
  }

  StmtDiff ForwardModeVisitor::VisitContinueStmt(const ContinueStmt* ContStmt) {
    return StmtDiff(Clone(ContStmt));
  }

  StmtDiff ForwardModeVisitor::VisitDoStmt(const DoStmt* DS) {
    // begin scope for do-while statement
    beginScope(Scope::ContinueScope | Scope::BreakScope);
    Expr* clonedCond = DS->getCond() ? Clone(DS->getCond()) : nullptr;
    const Stmt* body = DS->getBody();

    Stmt* bodyResult = nullptr;
    if (isa<CompoundStmt>(body)) {
      bodyResult = Visit(body).getStmt();
    } else {
      beginScope(Scope::DeclScope);
      beginBlock();
      StmtDiff Result = Visit(body);
      for (Stmt* S : Result.getBothStmts())
        addToCurrentBlock(S);
      CompoundStmt* Block = endBlock();
      endScope();
      bodyResult = Block;
    }

    Stmt* S = m_Sema
                  .ActOnDoStmt(/*DoLoc=*/noLoc, bodyResult, /*WhileLoc=*/noLoc,
                               /*CondLParen=*/noLoc, clonedCond,
                               /*CondRParen=*/noLoc)
                  .get();

    // end scope for do-while statement
    endScope();
    return StmtDiff(S);
  }

  /// returns first switch case label contained in the compound statement `CS`.
  static SwitchCase* getContainedSwitchCaseStmt(const CompoundStmt* CS) {
    for (Stmt* stmt : CS->body()) {
      if (auto SC = dyn_cast<SwitchCase>(stmt))
        return SC;
      else if (auto nestedCS = dyn_cast<CompoundStmt>(stmt)) {
        if (SwitchCase* nestedRes = getContainedSwitchCaseStmt(nestedCS))
          return nestedRes;
      }
    }
    return nullptr;
  }

  static void setSwitchCaseSubStmt(SwitchCase* SC, Stmt* subStmt) {
    if (auto caseStmt = dyn_cast<CaseStmt>(SC)) {
      caseStmt->setSubStmt(subStmt);
    } else if (auto defaultStmt = dyn_cast<DefaultStmt>(SC)) {
      defaultStmt->setSubStmt(subStmt);
    } else {
      assert(0 && "Unsupported switch case statement");
    }
  }

  /// Returns top switch statement in the `SwitchStack` of the given
  /// Function Scope.
  static SwitchStmt*
  getTopSwitchStmtOfSwitchStack(sema::FunctionScopeInfo* FSI) {
  #if CLANG_VERSION_MAJOR < 7
    return FSI->SwitchStack.back();
  #elif CLANG_VERSION_MAJOR >= 7
    return FSI->SwitchStack.back().getPointer();
  #endif
  }

  StmtDiff ForwardModeVisitor::VisitSwitchStmt(const SwitchStmt* SS) {
    // Scope and block for initializing derived variables for condition
    // variable and switch-init declaration.
    beginScope(Scope::DeclScope);
    beginBlock();

    const VarDecl* condVarDecl = SS->getConditionVariable();
    VarDecl* condVarClone = nullptr;
    if (condVarDecl) {
      VarDeclDiff condVarDeclDiff = DifferentiateVarDecl(condVarDecl);
      condVarClone = condVarDeclDiff.getDecl();
      addToCurrentBlock(BuildDeclStmt(condVarDeclDiff.getDecl_dx()));
    }

    StmtDiff initVarRes = (SS->getInit() ? Visit(SS->getInit()) : StmtDiff());
    addToCurrentBlock(initVarRes.getStmt_dx());

    // TODO: we can check if expr is null in `VisitorBase::Clone`, if it is
    // null then it can be safely returned without any cloning.
    Expr* clonedCond = (SS->getCond() ? Clone(SS->getCond()) : nullptr);

    Sema::ConditionResult condResult;
    if (condVarClone)
      condResult = m_Sema.ActOnConditionVariable(condVarClone, noLoc,
                                                 Sema::ConditionKind::Switch);
    else
      condResult = m_Sema.ActOnCondition(getCurrentScope(), noLoc, clonedCond,
                                         Sema::ConditionKind::Switch);

    // Scope for the switch statement
    beginScope(Scope::SwitchScope | Scope::ControlScope | Scope::BreakScope |
               Scope::DeclScope);
    Stmt* switchStmtDiff = clad_compat::
                               Sema_ActOnStartOfSwitchStmt(m_Sema,
                                                           initVarRes.getStmt(),
                                                           condResult)
                                   .get();
    // Scope and block for the corresponding compound statement of the
    // switch statement
    beginScope(Scope::DeclScope);
    beginBlock();

    // stores currently active switch case label. It is used to determine
    // the corresponding case label of the statements that are currently
    // being processed.
    // It will always be equal to the last visited case/default label.
    SwitchCase* activeSC = nullptr;

    if (auto CS = dyn_cast<CompoundStmt>(SS->getBody())) {
      // Visit(CS) cannot be used because then we will not be easily able to
      // determine when active switch case label should be changed.
      for (Stmt* stmt : CS->body()) {
        activeSC = DeriveSwitchStmtBodyHelper(stmt, activeSC);
      }
    } else {
      activeSC = DeriveSwitchStmtBodyHelper(SS->getBody(), activeSC);
    }

    // scope and block of the last switch case label is not popped in
    // `DeriveSwitchStmtBodyHelper` because it have no way of knowing
    // when all the statements belonging to last switch case label have
    // been processed aka when all the statments in switch statement body
    // have been processed.
    if (activeSC) {
      setSwitchCaseSubStmt(activeSC, endBlock());
      endScope();
      activeSC = nullptr;
    }

    // for corresponding compound statement of the switch block
    endScope();
    switchStmtDiff = m_Sema
                         .ActOnFinishSwitchStmt(noLoc, switchStmtDiff,
                                                endBlock())
                         .get();

    // for switch statement
    endScope();

    addToCurrentBlock(switchStmtDiff);
    // for scope created for derived condition variable and switch init
    // statement.
    endScope();
    return StmtDiff(endBlock());
  }

  SwitchCase*
  ForwardModeVisitor::DeriveSwitchStmtBodyHelper(const Stmt* stmt,
                                                 SwitchCase* activeSC) {
    if (auto SC = dyn_cast<SwitchCase>(stmt)) {
      // New switch case label have been visited. Pop the scope and block
      // corresponding to the active switch case label, and update its
      // substatement.
      if (activeSC) {
        setSwitchCaseSubStmt(activeSC, endBlock());
        endScope();
      }
      // sub statement will be updated later, either when the corresponding
      // next label is visited or the corresponding switch statement ends.
      SwitchCase* newActiveSC = nullptr;

      // We are not cloning the switch case label here because cloning will
      // also unnecessary clone substatement of the switch case label.
      if (auto newCaseSC = dyn_cast<CaseStmt>(SC)) {
        Expr* lhsClone = (newCaseSC->getLHS() ? Clone(newCaseSC->getLHS())
                                              : nullptr);
        Expr* rhsClone = (newCaseSC->getRHS() ? Clone(newCaseSC->getRHS())
                                              : nullptr);
        newActiveSC = clad_compat::CaseStmt_Create(m_Sema.getASTContext(),
                                                   lhsClone, rhsClone, noLoc,
                                                   noLoc, noLoc);
      } else if (isa<DefaultStmt>(SC)) {
        newActiveSC = new (m_Sema.getASTContext())
            DefaultStmt(noLoc, noLoc, nullptr);
      }

      SwitchStmt* activeSwitch = getTopSwitchStmtOfSwitchStack(
          m_Sema.getCurFunction());
      activeSwitch->addSwitchCase(newActiveSC);
      // Add new switch case label to the switch statement block and
      // create new scope and block for it to store statements belonging to it.
      addToCurrentBlock(newActiveSC);
      beginScope(Scope::DeclScope);
      beginBlock();

      activeSC = newActiveSC;
      activeSC = DeriveSwitchStmtBodyHelper(SC->getSubStmt(), activeSC);
      return activeSC;
    } else {
      if (auto CS = dyn_cast<CompoundStmt>(stmt)) {
        if (auto containedSC = getContainedSwitchCaseStmt(CS)) {
          // FIXME: One way to support this is strategically modifying the
          // compound statement blocks such that the meaning of code remains
          // the same and no switch case label is contained in the compound
          // statement.
          //
          // For example,
          // switch(var) {
          //  {
          //    case 1:
          //    ...
          //    ...
          //  }
          //  ...
          //  case 2:
          //  ...
          // }
          //
          // this code snippet can safely be transformed to,
          // switch(var) {
          //
          //  case 1: {
          //    ...
          //    ...
          //  }
          //  ...
          //  case 2:
          //  ...
          // }
          //
          // We can also solve this issue by creating new scope and compound
          // statement block wherever they are required instead of enclosing all
          // the statements of a case label in a single compound statement.
          diag(DiagnosticsEngine::Error, containedSC->getBeginLoc(),
               "Differentiating switch case label contained in a compound "
               "statement, other than the switch statement compound "
               "statement, is not supported.");
          return activeSC;
        }
      }
      StmtDiff stmtRes = Visit(stmt);
      addToCurrentBlock(stmtRes.getStmt_dx());
      addToCurrentBlock(stmtRes.getStmt());
      return activeSC;
    }
  }

  StmtDiff ForwardModeVisitor::VisitBreakStmt(const BreakStmt* stmt) {
    return StmtDiff(Clone(stmt));
  }
} // end namespace clad
