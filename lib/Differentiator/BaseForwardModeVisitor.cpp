//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clad/Differentiator/BaseForwardModeVisitor.h"

#include "ConstantFolder.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/ErrorEstimator.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTLambda.h"
#include "clang/AST/Expr.h"
#include "clang/AST/TemplateBase.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Template.h"

#include "llvm/IR/Constants.h"
#include "llvm/Support/SaveAndRestore.h"

#include <algorithm>

#include "clad/Differentiator/Compatibility.h"

using namespace clang;

namespace clad {
BaseForwardModeVisitor::BaseForwardModeVisitor(DerivativeBuilder& builder,
                                               const DiffRequest& request)
    : VisitorBase(builder, request) {}

BaseForwardModeVisitor::~BaseForwardModeVisitor() {}

bool BaseForwardModeVisitor::IsDifferentiableType(QualType T) {
  QualType origType = T;
  // FIXME: arbitrary dimension array type as well.
  while (utils::isArrayOrPointerType(T))
    T = utils::GetValueType(T);
  T = T.getNonReferenceType();
  if (T->isEnumeralType())
    return false;
  if (T->isRealType() || T->isStructureOrClassType())
    return true;
  if (origType->isPointerType() && T->isVoidType())
    return true;
  return false;
}

bool IsRealNonReferenceType(QualType T) {
  return T.getNonReferenceType()->isRealType();
}

DerivativeAndOverload
BaseForwardModeVisitor::Derive(const FunctionDecl* FD,
                               const DiffRequest& request) {
  assert(m_DiffReq == request && "Can't pass two different requests!");
  m_Functor = request.Functor;
  assert(m_DiffReq.Mode == DiffMode::forward);
  assert(!m_DerivativeInFlight &&
         "Doesn't support recursive diff. Use DiffPlan.");
  m_DerivativeInFlight = true;

  DiffInputVarsInfo DVI = request.DVI;

  DVI = request.DVI;

  // FIXME: Shouldn't we give error here that no arg is specified?
  if (DVI.empty())
    return {};

  DiffInputVarInfo diffVarInfo = DVI.back();

  // Check that only one arg is requested and if the arg requested is of array
  // or pointer type, only one of the indices have been requested
  if (DVI.size() > 1 || (isArrayOrPointerType(diffVarInfo.param->getType()) &&
                         (diffVarInfo.paramIndexInterval.size() != 1))) {
    diag(DiagnosticsEngine::Error,
         request.Args ? request.Args->getEndLoc() : noLoc,
         "Forward mode differentiation w.r.t. several parameters at once is "
         "not "
         "supported, call 'clad::differentiate' for each parameter "
         "separately");
    return {};
  }

  // FIXME: implement gradient-vector products to fix the issue.
  assert((DVI.size() == 1) &&
         "nested forward mode differentiation for several args is broken");

  // FIXME: Differentiation variable cannot always be represented just by
  // `ValueDecl*` variable. For example -- `u.mem1.mem2,`, `arr[7]` etc.
  // FIXME: independent variable is misleading terminology, what we actually
  // mean here is 'variable' with respect to which differentiation is being
  // performed. Mathematically, independent variables are all the function
  // parameters, thus, does not convey the intendend meaning.
  m_IndependentVar = DVI.back().param;
  std::string derivativeSuffix("");
  // If param is not real (i.e. floating point or integral), a pointer to a
  // real type, or an array of a real type we cannot differentiate it.
  // FIXME: we should support custom numeric types in the future.
  if (isArrayOrPointerType(m_IndependentVar->getType())) {
    if (!m_IndependentVar->getType()
             ->getPointeeOrArrayElementType()
             ->isRealType()) {
      diag(DiagnosticsEngine::Error, m_IndependentVar->getEndLoc(),
           "attempted differentiation w.r.t. a parameter ('%0') which is not"
           " an array or pointer of a real type",
           {m_IndependentVar->getNameAsString()});
      return {};
    }
    m_IndependentVarIndex = diffVarInfo.paramIndexInterval.Start;
    derivativeSuffix = "_" + std::to_string(m_IndependentVarIndex);
  } else {
    QualType T = m_IndependentVar->getType();
    bool isField = false;
    if (auto RD = diffVarInfo.param->getType()->getAsCXXRecordDecl()) {
      llvm::SmallVector<llvm::StringRef, 4> ref(diffVarInfo.fields.begin(),
                                                diffVarInfo.fields.end());
      T = utils::ComputeMemExprPathType(m_Sema, RD, ref);
      isField = true;
    }
    if (!IsRealNonReferenceType(T)) {
      diag(DiagnosticsEngine::Error, request.Args->getEndLoc(),
           "Attempted differentiation w.r.t. %0 '%1' which is not "
           "of real type.",
           {(isField ? "member" : "parameter"), diffVarInfo.source});
      return {};
    }
  }

  // If we are differentiating a call operator, that has no parameters,
  // then the specified independent argument is a member variable of the
  // class defining the call operator.
  // Thus, we need to find index of the member variable instead.
  unsigned argIndex = ~0;
  if (m_DiffReq->param_empty() && m_Functor)
    argIndex =
        std::distance(m_Functor->field_begin(),
                      std::find(m_Functor->field_begin(),
                                m_Functor->field_end(), m_IndependentVar));
  else
    argIndex = std::distance(
        FD->param_begin(),
        std::find(FD->param_begin(), FD->param_end(), m_IndependentVar));

  std::string argInfo = std::to_string(argIndex);
  for (auto field : diffVarInfo.fields)
    argInfo += "_" + field;

  std::string s;
  if (request.CurrentDerivativeOrder > 1)
    s = std::to_string(request.CurrentDerivativeOrder);

  // Check if the function is already declared as a custom derivative.
  std::string gradientName =
      request.BaseFunctionName + "_d" + s + "arg" + argInfo + derivativeSuffix;
  // FIXME: We should not use const_cast to get the decl context here.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  auto* DC = const_cast<DeclContext*>(m_DiffReq->getDeclContext());
  if (FunctionDecl* customDerivative =
          m_Builder.LookupCustomDerivativeDecl(gradientName, DC, FD->getType()))
    return DerivativeAndOverload{customDerivative, nullptr};

  IdentifierInfo* II = &m_Context.Idents.get(gradientName);
  SourceLocation validLoc{m_DiffReq->getLocation()};
  DeclarationNameInfo name(II, validLoc);
  llvm::SaveAndRestore<DeclContext*> SaveContext(m_Sema.CurContext);
  llvm::SaveAndRestore<Scope*> SaveScope(getCurrentScope());

  m_Sema.CurContext = DC;
  DeclWithContext result =
      m_Builder.cloneFunction(FD, *this, DC, validLoc, name, FD->getType());
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

    newPVD = ParmVarDecl::Create(m_Context, m_Sema.CurContext, validLoc,
                                 validLoc, PVD->getIdentifier(), PVD->getType(),
                                 PVD->getTypeSourceInfo(),
                                 PVD->getStorageClass(), clonedPVDDefaultArg);

    // Make m_IndependentVar to point to the argument of the newly created
    // derivedFD.
    if (PVD == m_IndependentVar)
      m_IndependentVar = newPVD;

    params.push_back(newPVD);
    // Add the args in the scope and id chain so that they could be found.
    if (newPVD->getIdentifier())
      m_Sema.PushOnScopeChains(newPVD, getCurrentScope(),
                               /*AddToContext*/ false);
  }

  llvm::ArrayRef<ParmVarDecl*> paramsRef =
      clad_compat::makeArrayRef(params.data(), params.size());
  derivedFD->setParams(paramsRef);
  derivedFD->setBody(nullptr);

  if (!request.DeclarationOnly) {
    // Function body scope
    beginScope(Scope::FnScope | Scope::DeclScope);
    m_DerivativeFnScope = getCurrentScope();
    beginBlock();
    // For each function parameter variable, store its derivative value.
    for (auto* param : params) {
      // We cannot create derivatives of reference type since seed value is
      // always a constant (r-value). We assume that all the arguments have no
      // relation among them, thus it is safe (correct) to use the corresponding
      // non-reference type for creating the derivatives.
      QualType dParamType = param->getType().getNonReferenceType();
      // We do not create derived variable for array/pointer parameters.
      if (!BaseForwardModeVisitor::IsDifferentiableType(dParamType) ||
          utils::isArrayOrPointerType(dParamType))
        continue;
      Expr* dParam = nullptr;
      if (dParamType->isRealType()) {
        // If param is independent variable, its derivative is 1, otherwise 0.
        int dValue = (param == m_IndependentVar);
        dParam = ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context,
                                                   dValue);
      }
      // For each function arg, create a variable _d_arg to store derivatives
      // of potential reassignments, e.g.:
      // double f_darg0(double x, double y) {
      //   double _d_x = 1;
      //   double _d_y = 0;
      //   ...
      auto* dParamDecl =
          BuildVarDecl(dParamType, "_d_" + param->getNameAsString(), dParam);
      addToCurrentBlock(BuildDeclStmt(dParamDecl));
      dParam = BuildDeclRef(dParamDecl);
      if (dParamType->isRecordType() && param == m_IndependentVar) {
        llvm::SmallVector<llvm::StringRef, 4> ref(diffVarInfo.fields.begin(),
                                                  diffVarInfo.fields.end());
        Expr* memRef =
            utils::BuildMemberExpr(m_Sema, getCurrentScope(), dParam, ref);
        assert(memRef->getType()->isRealType() &&
               "Forward mode can only differentiate w.r.t builtin scalar "
               "numerical types.");
        addToCurrentBlock(BuildOp(
            BinaryOperatorKind::BO_Assign, memRef,
            ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 1)));
      }
      // Memorize the derivative of param, i.e. whenever the param is visited
      // in the future, it's derivative dParam is found (unless reassigned with
      // something new).
      m_Variables[param] = dParam;
    }

    if (const auto* MD = dyn_cast<CXXMethodDecl>(FD)) {
      // We cannot create derivative of lambda yet because lambdas default
      // constructor is deleted.
      if (MD->isInstance() && !MD->getParent()->isLambda()) {
        QualType thisObjectType =
            clad_compat::CXXMethodDecl_GetThisObjectType(m_Sema, MD);
        QualType thisType = MD->getThisType();
        // Here we are effectively doing:
        // ```
        // Class _d_this_obj;
        // Class* _d_this = &_d_this_obj;
        // ```
        // We are not creating `this` expression derivative using `new` because
        // then we would be responsible for freeing the memory as well and its
        // more convenient to let compiler handle the object lifecycle.
        VarDecl* derivativeVD = BuildVarDecl(thisObjectType, "_d_this_obj");
        DeclRefExpr* derivativeE = BuildDeclRef(derivativeVD);
        VarDecl* thisExprDerivativeVD =
            BuildVarDecl(thisType, "_d_this",
                         BuildOp(UnaryOperatorKind::UO_AddrOf, derivativeE));
        addToCurrentBlock(BuildDeclStmt(derivativeVD));
        addToCurrentBlock(BuildDeclStmt(thisExprDerivativeVD));
        m_ThisExprDerivative = BuildDeclRef(thisExprDerivativeVD);
      }
    }

    // Create derived variable for each member variable if we are
    // differentiating a call operator.
    if (m_Functor) {
      for (FieldDecl* fieldDecl : m_Functor->fields()) {
        Expr* dInitializer = nullptr;
        QualType fieldType = fieldDecl->getType();

        if (const auto* arrType =
                dyn_cast<ConstantArrayType>(fieldType.getTypePtr())) {
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
            int dValue =
                (fieldDecl == m_IndependentVar && i == m_IndependentVarIndex);
            auto* dValueLiteral = ConstantFolder::synthesizeLiteral(
                m_Context.IntTy, m_Context, dValue);
            dArrVal.push_back(dValueLiteral);
          }
          dInitializer =
              m_Sema.ActOnInitList(validLoc, dArrVal, validLoc).get();
        } else if (const auto* ptrType =
                       dyn_cast<PointerType>(fieldType.getTypePtr())) {
          if (!ptrType->getPointeeType()->isRealType())
            continue;
          // Pointer member variables should be initialised by `nullptr`.
          dInitializer = m_Sema.ActOnCXXNullPtrLiteral(validLoc).get();
        } else {
          int dValue = (fieldDecl == m_IndependentVar);
          dInitializer = ConstantFolder::synthesizeLiteral(m_Context.IntTy,
                                                           m_Context, dValue);
        }
        VarDecl* derivedFieldDecl =
            BuildVarDecl(fieldType.getNonReferenceType(),
                         "_d_" + fieldDecl->getNameAsString(), dInitializer);
        addToCurrentBlock(BuildDeclStmt(derivedFieldDecl));
        m_Variables.emplace(fieldDecl, BuildDeclRef(derivedFieldDecl));
      }
    }

    Stmt* BodyDiff = Visit(FD->getBody()).getStmt();
    if (auto* CS = dyn_cast<CompoundStmt>(BodyDiff))
      for (Stmt* S : CS->body())
        addToCurrentBlock(S);
    else
      addToCurrentBlock(BodyDiff);
    Stmt* derivativeBody = endBlock();
    derivedFD->setBody(derivativeBody);

    endScope(); // Function body scope

    // Size >= current derivative order means that there exists a declaration
    // or prototype for the currently derived function.
    if (request.DerivedFDPrototypes.size() >= request.CurrentDerivativeOrder)
      m_Derivative->setPreviousDeclaration(
          request.DerivedFDPrototypes[request.CurrentDerivativeOrder - 1]);
  }
  m_Sema.PopFunctionScopeInfo();
  m_Sema.PopDeclContext();
  endScope(); // Function decl scope

  m_DerivativeInFlight = false;

  return DerivativeAndOverload{result.first,
                               /*OverloadFunctionDecl=*/nullptr};
}

clang::QualType BaseForwardModeVisitor::ComputePushforwardFnReturnType() {
  assert(m_DiffReq.Mode == GetPushForwardMode());
  QualType originalFnRT = m_DiffReq->getReturnType();
  if (originalFnRT->isVoidType())
    return m_Context.VoidTy;
  TemplateDecl* valueAndPushforward =
      LookupTemplateDeclInCladNamespace("ValueAndPushforward");
  assert(valueAndPushforward &&
         "clad::ValueAndPushforward template not found!!");
  QualType RT = InstantiateTemplate(
      valueAndPushforward,
      {originalFnRT, GetPushForwardDerivativeType(originalFnRT)});
  return RT;
}

void BaseForwardModeVisitor::ExecuteInsidePushforwardFunctionBlock() {
  Stmt* bodyDiff = Visit(m_DiffReq->getBody()).getStmt();
  auto* CS = cast<CompoundStmt>(bodyDiff);
  for (Stmt* S : CS->body())
    addToCurrentBlock(S);
}

DerivativeAndOverload
BaseForwardModeVisitor::DerivePushforward(const FunctionDecl* FD,
                                          const DiffRequest& request) {
  // FIXME: We must not reset the diff request here.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  const_cast<DiffRequest&>(m_DiffReq) = request;
  m_Functor = request.Functor;
  assert(m_DiffReq.Mode == GetPushForwardMode());
  assert(!m_DerivativeInFlight &&
         "Doesn't support recursive diff. Use DiffPlan.");
  m_DerivativeInFlight = true;

  auto originalFnEffectiveName =
      utils::ComputeEffectiveFnName(m_DiffReq.Function);

  IdentifierInfo* derivedFnII = &m_Context.Idents.get(
      originalFnEffectiveName + GetPushForwardFunctionSuffix());
  DeclarationNameInfo derivedFnName(derivedFnII, m_DiffReq->getLocation());
  llvm::SmallVector<QualType, 16> paramTypes;
  llvm::SmallVector<QualType, 16> derivedParamTypes;

  // If we are differentiating an instance member function then
  // create a parameter type for the parameter that will represent the
  // derivative of `this` pointer with respect to the independent parameter.
  if (const auto* MD = dyn_cast<CXXMethodDecl>(FD)) {
    if (MD->isInstance()) {
      QualType thisType = MD->getThisType();
      derivedParamTypes.push_back(thisType);
    }
  }

  for (auto* PVD : m_DiffReq->parameters()) {
    paramTypes.push_back(PVD->getType());

    if (BaseForwardModeVisitor::IsDifferentiableType(PVD->getType()))
      derivedParamTypes.push_back(GetPushForwardDerivativeType(PVD->getType()));
  }

  paramTypes.insert(paramTypes.end(), derivedParamTypes.begin(),
                    derivedParamTypes.end());

  const auto* originalFnType =
      dyn_cast<FunctionProtoType>(m_DiffReq->getType());
  QualType returnType = ComputePushforwardFnReturnType();
  QualType derivedFnType = m_Context.getFunctionType(
      returnType, paramTypes, originalFnType->getExtProtoInfo());
  llvm::SaveAndRestore<DeclContext*> saveContext(m_Sema.CurContext);
  llvm::SaveAndRestore<Scope*> saveScope(getCurrentScope(),
                                         getEnclosingNamespaceOrTUScope());
  // FIXME: We should not use const_cast to get the decl context here.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  auto* DC = const_cast<DeclContext*>(m_DiffReq->getDeclContext());
  m_Sema.CurContext = DC;

  SourceLocation loc{m_DiffReq->getLocation()};
  DeclWithContext cloneFunctionResult = m_Builder.cloneFunction(
      m_DiffReq.Function, *this, DC, loc, derivedFnName, derivedFnType);
  m_Derivative = cloneFunctionResult.first;

  llvm::SmallVector<ParmVarDecl*, 16> params;
  llvm::SmallVector<ParmVarDecl*, 16> derivedParams;
  beginScope(Scope::FunctionPrototypeScope | Scope::FunctionDeclarationScope |
             Scope::DeclScope);
  m_Sema.PushFunctionScope();
  m_Sema.PushDeclContext(getCurrentScope(), m_Derivative);

  // If we are differentiating an instance member function then
  // create a parameter for representing derivative of
  // `this` pointer with respect to the independent parameter.
  if (const auto* MFD = dyn_cast<CXXMethodDecl>(FD)) {
    if (MFD->isInstance()) {
      auto thisType = MFD->getThisType();
      IdentifierInfo* derivedPVDII = CreateUniqueIdentifier("_d_this");
      auto* derivedPVD = utils::BuildParmVarDecl(m_Sema, m_Sema.CurContext,
                                                 derivedPVDII, thisType);
      m_Sema.PushOnScopeChains(derivedPVD, getCurrentScope(),
                               /*AddToContext=*/false);
      derivedParams.push_back(derivedPVD);
      m_ThisExprDerivative = BuildDeclRef(derivedPVD);
    }
  }

  std::size_t numParamsOriginalFn = m_DiffReq->getNumParams();
  for (std::size_t i = 0; i < numParamsOriginalFn; ++i) {
    const auto* PVD = m_DiffReq->getParamDecl(i);
    // Some of the special member functions created implicitly by compilers
    // have missing parameter identifier.
    bool identifierMissing = false;
    IdentifierInfo* PVDII = PVD->getIdentifier();
    if (!PVDII || PVDII->getLength() == 0) {
      PVDII = CreateUniqueIdentifier("param");
      identifierMissing = true;
    }
    auto* newPVD = CloneParmVarDecl(PVD, PVDII,
                                    /*pushOnScopeChains=*/true,
                                    /*cloneDefaultArg=*/false);
    params.push_back(newPVD);

    if (identifierMissing)
      m_DeclReplacements[PVD] = newPVD;

    if (!BaseForwardModeVisitor::IsDifferentiableType(PVD->getType()))
      continue;
    auto derivedPVDName = "_d_" + std::string(PVDII->getName());
    IdentifierInfo* derivedPVDII = CreateUniqueIdentifier(derivedPVDName);
    auto* derivedPVD = utils::BuildParmVarDecl(
        m_Sema, m_Derivative, derivedPVDII,
        GetPushForwardDerivativeType(PVD->getType()), PVD->getStorageClass());
    derivedParams.push_back(derivedPVD);
    m_Variables[newPVD] = BuildDeclRef(derivedPVD);
  }

  params.insert(params.end(), derivedParams.begin(), derivedParams.end());
  m_Derivative->setParams(params);
  m_Derivative->setBody(nullptr);

  if (!request.DeclarationOnly) {
    beginScope(Scope::FnScope | Scope::DeclScope);
    m_DerivativeFnScope = getCurrentScope();
    beginBlock();

    // execute the functor inside the function body.
    ExecuteInsidePushforwardFunctionBlock();

    Stmt* derivativeBody = endBlock();
    m_Derivative->setBody(derivativeBody);

    endScope(); // Function body scope

    // Size >= current derivative order means that there exists a declaration
    // or prototype for the currently derived function.
    if (request.DerivedFDPrototypes.size() >= request.CurrentDerivativeOrder)
      m_Derivative->setPreviousDeclaration(
          request.DerivedFDPrototypes[request.CurrentDerivativeOrder - 1]);
  }

  m_Sema.PopFunctionScopeInfo();
  m_Sema.PopDeclContext();
  endScope(); // Function decl scope

  m_DerivativeInFlight = false;
  return DerivativeAndOverload{cloneFunctionResult.first};
}

StmtDiff BaseForwardModeVisitor::VisitStmt(const Stmt* S) {
  diag(DiagnosticsEngine::Warning, S->getBeginLoc(),
       "attempted to differentiate unsupported statement, no changes applied");
  // Unknown stmt, just clone it.
  return StmtDiff(Clone(S));
}

StmtDiff BaseForwardModeVisitor::VisitCompoundStmt(const CompoundStmt* CS) {
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

StmtDiff BaseForwardModeVisitor::VisitIfStmt(const IfStmt* If) {
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
    DeclDiff<VarDecl> condVarDeclDiff = DifferentiateVarDecl(condVarDecl);
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

  Stmt* ifDiff = clad_compat::IfStmt_Create(
      m_Context, noLoc, If->isConstexpr(), initResult.getStmt(), condVarClone,
      cond, noLoc, noLoc, thenDiff, noLoc, elseDiff);
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

StmtDiff BaseForwardModeVisitor::VisitConditionalOperator(
    const ConditionalOperator* CO) {
  Expr* cond = Clone(CO->getCond());
  // FIXME: fix potential side-effects from evaluating both sides of
  // conditional.
  StmtDiff ifTrueDiff = Visit(CO->getTrueExpr());
  StmtDiff ifFalseDiff = Visit(CO->getFalseExpr());

  cond = StoreAndRef(cond);
  cond = m_Sema
             .ActOnCondition(getCurrentScope(), noLoc, cond,
                             Sema::ConditionKind::Boolean)
             .get()
             .second;

  Expr* condExpr =
      m_Sema
          .ActOnConditionalOp(noLoc, noLoc, cond, ifTrueDiff.getExpr(),
                              ifFalseDiff.getExpr())
          .get();

  Expr* condExprDiff =
      m_Sema
          .ActOnConditionalOp(noLoc, noLoc, cond, ifTrueDiff.getExpr_dx(),
                              ifFalseDiff.getExpr_dx())
          .get();

  return StmtDiff(condExpr, condExprDiff);
}

StmtDiff
BaseForwardModeVisitor::VisitCXXForRangeStmt(const CXXForRangeStmt* FRS) {
  beginScope(Scope::DeclScope | Scope::ControlScope | Scope::BreakScope |
             Scope::ContinueScope);
  // Visiting for range-based ststement produces __range1, __begin1 and __end1
  // variables, so for(auto i: a){
  //      ...
  //}
  //
  // is equivalent to
  //
  // auto&& __range1 = a
  // auto __begin1 = __range1;
  // auto __end1 = __range1 + OUL
  // for(;__begin != __end1; ++__begin){
  //  auto i = *__begin1;
  //        ...
  //}
  const Stmt* RangeDecl = FRS->getRangeStmt();
  const Stmt* BeginDecl = FRS->getBeginStmt();
  const Stmt* EndDecl = FRS->getEndStmt();

  StmtDiff VisitRange = Visit(RangeDecl);
  StmtDiff VisitBegin = Visit(BeginDecl);
  StmtDiff VisitEnd = Visit(EndDecl);
  addToCurrentBlock(VisitRange.getStmt_dx());
  addToCurrentBlock(VisitRange.getStmt());
  addToCurrentBlock(VisitBegin.getStmt_dx());
  addToCurrentBlock(VisitBegin.getStmt());
  addToCurrentBlock(VisitEnd.getStmt());
  // Build d_begin preincrementation.

  auto* BeginAdjExpr = BuildDeclRef(
      cast<VarDecl>(cast<DeclStmt>(VisitBegin.getStmt_dx())->getSingleDecl()));
  // Build begin preincrementation.

  Expr* IncAdjBegin = BuildOp(UO_PreInc, BeginAdjExpr);
  auto* BeginVarDecl =
      cast<VarDecl>(cast<DeclStmt>(VisitBegin.getStmt())->getSingleDecl());
  DeclRefExpr* BeginExpr = BuildDeclRef(BeginVarDecl);
  Expr* IncBegin = BuildOp(UO_PreInc, BeginExpr);
  Expr* Inc = BuildOp(BO_Comma, IncAdjBegin, IncBegin);

  auto* EndExpr = BuildDeclRef(
      cast<VarDecl>(cast<DeclStmt>(VisitEnd.getStmt())->getSingleDecl()));
  // Build begin != end condition.
  Expr* cond = BuildOp(BO_NE, BeginExpr, EndExpr);

  const VarDecl* VD = FRS->getLoopVariable();
  DeclDiff<VarDecl> VDDiff = DifferentiateVarDecl(VD);
  // Differentiate body and add both Item and it's derivative.
  Stmt* body = Clone(FRS->getBody());
  Stmt* bodyResult = Visit(body).getStmt();
  Visit(body).getStmt();
  Stmt* bodyWithItem = utils::PrependAndCreateCompoundStmt(
      m_Sema.getASTContext(), bodyResult, BuildDeclStmt(VDDiff.getDecl()));
  bodyResult = utils::PrependAndCreateCompoundStmt(
      m_Sema.getASTContext(), bodyWithItem, BuildDeclStmt(VDDiff.getDecl_dx()));

  Stmt* forStmtDiff = new (m_Context)
      ForStmt(m_Context, nullptr, cond, /*condVar=*/nullptr, Inc, bodyResult,
              FRS->getForLoc(), FRS->getBeginLoc(), FRS->getEndLoc());
  return StmtDiff(forStmtDiff);
}

StmtDiff BaseForwardModeVisitor::VisitForStmt(const ForStmt* FS) {
  beginScope(Scope::DeclScope | Scope::ControlScope | Scope::BreakScope |
             Scope::ContinueScope);
  beginBlock();
  const Stmt* init = FS->getInit();
  StmtDiff initDiff = init ? Visit(init) : StmtDiff{};
  addToCurrentBlock(initDiff.getStmt_dx());

  StmtDiff condDiff = Clone(FS->getCond());
  Expr* cond = condDiff.getExpr();

  // The declaration in the condition needs to be differentiated.
  if (VarDecl* condVarDecl = FS->getConditionVariable()) {
    // Here we create a fictional cond that is equal to the assignment used in
    // the declaration. The declaration itself is thrown before the for-loop
    // without any init value. The fictional condition is then differentiated as
    // a normal condition would be (see below). For example, the declaration
    // inside `for (;double t = x;) {}` will be first processed into the
    // following code:
    // ```
    // {
    // double t;
    // for (;t = x;) {}
    // }
    // ```
    // which will then get differentiated normally as a for-loop with a
    // differentiable condition in the next section.
    DeclDiff<VarDecl> condVarResult =
        DifferentiateVarDecl(condVarDecl, /*ignoreInit=*/true);
    VarDecl* condVarClone = condVarResult.getDecl();
    if (condVarResult.getDecl_dx())
      addToCurrentBlock(BuildDeclStmt(condVarResult.getDecl_dx()));
    auto condInit = condVarClone->getInit();
    condVarClone->setInit(nullptr);
    cond = BuildOp(BO_Assign, BuildDeclRef(condVarClone), condInit);
    addToCurrentBlock(BuildDeclStmt(condVarClone));
  }

  // Condition differentiation.
  // This adds support for assignments in conditions.
  if (cond) {
    cond = cond->IgnoreParenImpCasts();
    // If it's a supported differentiable operator we wrap it back into
    // parentheses and then visit. To ensure the correctness, a comma operator
    // expression (cond_dx, cond) is generated and put instead of the condition.
    // FIXME: Add support for other expressions in cond (comparisons, function
    // calls, etc.). Ideally, we should be able to simply always call
    // Visit(cond)
    auto* condBO = dyn_cast<BinaryOperator>(cond);
    auto* condUO = dyn_cast<UnaryOperator>(cond);
    // FIXME: Currently we only support logical and assignment operators.
    if ((condBO && (condBO->isLogicalOp() || condBO->isAssignmentOp())) ||
        condUO) {
      condDiff = Visit(cond);
      if (condDiff.getExpr_dx() && (!isUnusedResult(condDiff.getExpr_dx())))
        cond = BuildOp(BO_Comma, BuildParens(condDiff.getExpr_dx()),
                       BuildParens(condDiff.getExpr()));
      else
        cond = condDiff.getExpr();
    }
  }

  // Differentiate the increment expression of the for loop
  const Expr* inc = FS->getInc();
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
      incResult = BuildOp(BO_Comma, BuildParens(incDiff.getExpr_dx()),
                          BuildParens(incDiff.getExpr()));
    else
      incResult = incDiff.getExpr();
  } else if (incDiff.getExpr()) {
    incResult = incDiff.getExpr();
  }

  // Build the derived for loop body.
  const Stmt* body = FS->getBody();
  beginScope(Scope::DeclScope);
  Stmt* bodyResult = nullptr;
  beginBlock();
  StmtDiff bodyVisited = Visit(body);
  for (Stmt* S : bodyVisited.getBothStmts())
    addToCurrentBlock(S);
  bodyResult = utils::unwrapIfSingleStmt(endBlock());
  endScope();

  Stmt* forStmtDiff = new (m_Context)
      ForStmt(m_Context, initDiff.getStmt(), cond, /*condVar=*/nullptr,
              incResult, bodyResult, noLoc, noLoc, noLoc);

  addToCurrentBlock(forStmtDiff);
  CompoundStmt* Block = endBlock();
  endScope();

  StmtDiff Result =
      (Block->size() == 1) ? StmtDiff(forStmtDiff) : StmtDiff(Block);
  return Result;
}

StmtDiff BaseForwardModeVisitor::VisitReturnStmt(const ReturnStmt* RS) {
  // If there is no return value, we must not attempt to differentiate
  if (!RS->getRetValue())
    return nullptr;

  StmtDiff retValDiff = Visit(RS->getRetValue());
  Stmt* returnStmt =
      m_Sema.ActOnReturnStmt(noLoc, retValDiff.getExpr_dx(), getCurrentScope())
          .get();
  return StmtDiff(returnStmt);
}

StmtDiff BaseForwardModeVisitor::VisitParenExpr(const ParenExpr* PE) {
  StmtDiff subStmtDiff = Visit(PE->getSubExpr());
  return StmtDiff(BuildParens(subStmtDiff.getExpr()),
                  BuildParens(subStmtDiff.getExpr_dx()));
}

StmtDiff BaseForwardModeVisitor::VisitMemberExpr(const MemberExpr* ME) {
  auto clonedME = dyn_cast<MemberExpr>(Clone(ME));
  // Currently, we only differentiate member variables if we are
  // differentiating a call operator.
  if (m_Functor) {
    if (isa<CXXThisExpr>(ME->getBase()->IgnoreParenImpCasts())) {
      // Try to find the derivative of the member variable wrt independent
      // variable
      auto memberDecl = ME->getMemberDecl();
      if (m_Variables.find(memberDecl) != std::end(m_Variables)) {
        return StmtDiff(clonedME, m_Variables[memberDecl]);
      }
    }
    // Is not a real variable. Therefore, derivative is 0.
    auto zero =
        ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
    return StmtDiff(clonedME, zero);
  } else {
    auto zero =
        ConstantFolder::synthesizeLiteral(m_Context.DoubleTy, m_Context, 0);
    if (clad::utils::hasNonDifferentiableAttribute(ME))
      return {clonedME, zero};
    auto baseDiff = Visit(ME->getBase());
    // No derivative found for base. Therefore, derivative is 0.
    if (isa<IntegerLiteral>(baseDiff.getExpr_dx()) ||
        isa<FloatingLiteral>(baseDiff.getExpr_dx()))
      return {clonedME, zero};

    auto field = ME->getMemberDecl();
    assert(!isa<FunctionDecl>(field) &&
           "Member functions are not supported yet!");
    auto clonedME = utils::BuildMemberExpr(
        m_Sema, getCurrentScope(), baseDiff.getExpr(), field->getName());
    // Here we are implicitly assuming that the derived type and the original
    // types are same. This may not be necessarily true in the future.
    auto derivedME = utils::BuildMemberExpr(
        m_Sema, getCurrentScope(), baseDiff.getExpr_dx(), field->getName());
    return {clonedME, derivedME};
  }
}

StmtDiff BaseForwardModeVisitor::VisitInitListExpr(const InitListExpr* ILE) {
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
BaseForwardModeVisitor::VisitArraySubscriptExpr(const ArraySubscriptExpr* ASE) {
  auto ASI = SplitArraySubscript(ASE);
  QualType ExprTy = ASE->getType();
  if (ExprTy->isPointerType())
    ExprTy = ExprTy->getPointeeType();
  ExprTy = ExprTy->getCanonicalTypeInternal();
  const Expr* base = ASI.first;
  const auto& Indices = ASI.second;
  StmtDiff cloneDiff = Visit(base);
  Expr* clonedBase = cloneDiff.getExpr();
  llvm::SmallVector<Expr*, 4> clonedIndices(Indices.size());
  std::transform(std::begin(Indices), std::end(Indices),
                 std::begin(clonedIndices),
                 [this](const Expr* E) { return Clone(E); });
  Expr* cloned = BuildArraySubscript(clonedBase, clonedIndices);

  Expr* zero = getZeroInit(ExprTy);
  ValueDecl* VD = nullptr;
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
  } else if (isa<MemberExpr>(clonedBase->IgnoreParenImpCasts())) {
    auto derivedME = cloneDiff.getExpr_dx();
    if (!isa<MemberExpr>(derivedME->IgnoreParenImpCasts())) {
      return {cloned, zero};
    }
    auto derivedAS = BuildArraySubscript(derivedME, clonedIndices);
    return {cloned, derivedAS};
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
    Expr::EvalResult res;
    Expr::SideEffectsKind AllowSideEffects =
        Expr::SideEffectsKind::SE_NoSideEffects;
    if (!clonedIndices.back()->EvaluateAsInt(res, m_Context,
                                             AllowSideEffects)) {
      diffExpr =
          BuildParens(BuildOp(BO_EQ, clonedIndices.back(),
                              ConstantFolder::synthesizeLiteral(
                                  ExprTy, m_Context, m_IndependentVarIndex)));
    } else if (res.Val.getInt().getExtValue() == m_IndependentVarIndex) {
      diffExpr = ConstantFolder::synthesizeLiteral(ExprTy, m_Context, 1);
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

StmtDiff BaseForwardModeVisitor::VisitDeclRefExpr(const DeclRefExpr* DRE) {
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
      clang::Expr* dExpr = it->second;
      // If a record was found, use the recorded derivative.
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
  // If DRE is of type pointer, then the derivative is a null pointer.
  if (clonedDRE->getType()->isPointerType())
    return StmtDiff(clonedDRE, nullptr);
  return StmtDiff(clonedDRE, ConstantFolder::synthesizeLiteral(
                                 m_Context.IntTy, m_Context, /*val=*/0));
}

StmtDiff BaseForwardModeVisitor::VisitIntegerLiteral(const IntegerLiteral* IL) {
  QualType T = IL->getType();
  llvm::APInt zero(m_Context.getIntWidth(T), /*value*/ 0);
  auto* constant0 = IntegerLiteral::Create(m_Context, zero, T, noLoc);
  return StmtDiff(Clone(IL), constant0);
}

StmtDiff
BaseForwardModeVisitor::VisitFloatingLiteral(const FloatingLiteral* FL) {
  llvm::APFloat zero = llvm::APFloat::getZero(FL->getSemantics());
  auto* constant0 =
      FloatingLiteral::Create(m_Context, zero, true, FL->getType(), noLoc);
  return StmtDiff(Clone(FL), constant0);
}

QualType
BaseForwardModeVisitor::GetPushForwardDerivativeType(QualType ParamType) {
  return ParamType;
}

std::string BaseForwardModeVisitor::GetPushForwardFunctionSuffix() {
  return "_pushforward";
}

DiffMode BaseForwardModeVisitor::GetPushForwardMode() {
  return DiffMode::experimental_pushforward;
}

StmtDiff BaseForwardModeVisitor::VisitCallExpr(const CallExpr* CE) {
  const FunctionDecl* FD = CE->getDirectCallee();
  if (!FD) {
    diag(DiagnosticsEngine::Warning, CE->getBeginLoc(),
         "Differentiation of only direct calls is supported. Ignored");
    return StmtDiff(Clone(CE));
  }

  SourceLocation validLoc{CE->getBeginLoc()};

  // Calls to lambda functions are processed differently
  bool isLambda = isLambdaCallOperator(FD);

  // If the function is non_differentiable, return zero derivative.
  if (clad::utils::hasNonDifferentiableAttribute(CE)) {
    // Calling the function without computing derivatives
    llvm::SmallVector<Expr*, 4> ClonedArgs;
    for (unsigned i = 0, e = CE->getNumArgs(); i < e; ++i)
      ClonedArgs.push_back(Clone(CE->getArg(i)));

    Expr* Call = m_Sema
                     .ActOnCallExpr(getCurrentScope(), Clone(CE->getCallee()),
                                    validLoc, ClonedArgs, validLoc)
                     .get();
    // Creating a zero derivative
    auto* zero =
        ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);

    // Returning the function call and zero derivative
    return StmtDiff(Call, zero);
  }

  // Find the built-in derivatives namespace.
  llvm::SmallVector<Expr*, 4> CallArgs{};
  llvm::SmallVector<Expr*, 4> diffArgs;

  // Represents `StmtDiff` result of the 'base' object if are differentiating
  // a direct or indirect (operator overload) call to member function.
  StmtDiff baseDiff;
  // Add derivative of the implicit `this` pointer to the `diffArgs`.
  if (isLambda) {
    if (const auto* OCE = dyn_cast<CXXOperatorCallExpr>(CE)) {
      QualType ptrType = m_Context.getPointerType(m_Context.getRecordType(
          FD->getDeclContext()->getOuterLexicalRecordContext()));
      // For now, only lambdas with no captures are supported, so we just pass
      // a nullptr instead of the diff object.
      baseDiff =
          StmtDiff(Clone(OCE->getArg(0)),
                   new (m_Context) CXXNullPtrLiteralExpr(ptrType, validLoc));
      diffArgs.push_back(baseDiff.getExpr_dx());
    }
  } else if (const auto* MD =
                 dyn_cast<CXXMethodDecl>(FD)) { // isLambda == false
    if (MD->isInstance()) {
      const Expr* baseOriginalE = nullptr;
      if (const auto* MCE = dyn_cast<CXXMemberCallExpr>(CE))
        baseOriginalE = MCE->getImplicitObjectArgument();
      else if (const auto* OCE = dyn_cast<CXXOperatorCallExpr>(CE))
        baseOriginalE = OCE->getArg(0);
      baseDiff = Visit(baseOriginalE);
      Expr* baseDerivative = baseDiff.getExpr_dx();
      if (!baseDerivative->getType()->isPointerType())
        baseDerivative = BuildOp(UnaryOperatorKind::UO_AddrOf, baseDerivative);
      diffArgs.push_back(baseDerivative);
    }
  }

  // `CXXOperatorCallExpr` have the `base` expression as the first argument.
  // This representation conflict with calls to member functions.  Thus, to
  // maintain consistency, we are following this:
  //
  // `baseDiff` contains differentiation result of the corresponding `base`
  // object of the call, if any.
  // `CallArgs` contains clones of all the original call arguments.
  // `DiffArgs` contains derivatives of all the call arguments.
  bool skipFirstArg = false;

  // Here we do not need to check if FD is an instance method or a static
  // method because C++ forbids creating operator overloads as static methods.
  if (isa<CXXOperatorCallExpr>(CE) && isa<CXXMethodDecl>(FD))
    skipFirstArg = true;

  // For f(g(x)) = f'(x) * g'(x)
  Expr* Multiplier = nullptr;
  for (size_t i = skipFirstArg, e = CE->getNumArgs(); i < e; ++i) {
    const Expr* arg = CE->getArg(i);
    StmtDiff argDiff = Visit(arg);

    // If original argument is an RValue and function expects an RValue
    // parameter, then convert the cloned argument and the corresponding
    // derivative to RValue if they are not RValue.
    QualType paramType = FD->getParamDecl(i - skipFirstArg)->getType();
    if (utils::IsRValue(arg) && paramType->isRValueReferenceType()) {
      if (!utils::IsRValue(argDiff.getExpr())) {
        Expr* castE = utils::BuildStaticCastToRValue(m_Sema, argDiff.getExpr());
        argDiff.updateStmt(castE);
      }
      if (!utils::IsRValue(argDiff.getExpr_dx())) {
        Expr* castE =
            utils::BuildStaticCastToRValue(m_Sema, argDiff.getExpr_dx());
        argDiff.updateStmtDx(castE);
      }
    }
    CallArgs.push_back(argDiff.getExpr());
    if (BaseForwardModeVisitor::IsDifferentiableType(arg->getType())) {
      Expr* dArg = argDiff.getExpr_dx();
      // FIXME: What happens when dArg is nullptr?
      diffArgs.push_back(dArg);
    }
  }

  llvm::SmallVector<Expr*, 16> pushforwardFnArgs;
  pushforwardFnArgs.insert(pushforwardFnArgs.end(), CallArgs.begin(),
                           CallArgs.end());
  pushforwardFnArgs.insert(pushforwardFnArgs.end(), diffArgs.begin(),
                           diffArgs.end());

  auto customDerivativeArgs = pushforwardFnArgs;

  if (Expr* baseE = baseDiff.getExpr()) {
    if (!baseE->getType()->isPointerType())
      baseE = BuildOp(UnaryOperatorKind::UO_AddrOf, baseE);
    customDerivativeArgs.insert(customDerivativeArgs.begin(), baseE);
  }

  // Try to find a user-defined overloaded derivative.
  Expr* callDiff = nullptr;
  std::string customPushforward =
      clad::utils::ComputeEffectiveFnName(FD) + GetPushForwardFunctionSuffix();
  callDiff = m_Builder.BuildCallToCustomDerivativeOrNumericalDiff(
      customPushforward, customDerivativeArgs, getCurrentScope(),
      const_cast<DeclContext*>(FD->getDeclContext()));

  if (!isLambda) {
    // Check if it is a recursive call.
    if (!callDiff && (FD == m_DiffReq.Function) &&
        m_DiffReq.Mode == GetPushForwardMode()) {
      // The differentiated function is called recursively.
      Expr* derivativeRef =
          m_Sema
              .BuildDeclarationNameExpr(
                  CXXScopeSpec(), m_Derivative->getNameInfo(), m_Derivative)
              .get();
      callDiff = m_Sema
                     .ActOnCallExpr(
                         m_Sema.getScopeForContext(m_Sema.CurContext),
                         derivativeRef, validLoc, pushforwardFnArgs, validLoc)
                     .get();
    }
  }

  // If all arguments are constant literals, then this does not contribute to
  // the gradient.
  // FIXME: revert this when this is integrated in the activity analysis pass.
  if (!callDiff) {
    if (!isa<CXXOperatorCallExpr>(CE) && !isa<CXXMemberCallExpr>(CE)) {
      bool allArgsHaveZeroDerivatives = true;
      for (unsigned i = 0, e = CE->getNumArgs(); i < e; ++i) {
        Expr* dArg = diffArgs[i];
        // If argDiff.expr_dx is nullptr or is a constant 0, then the derivative
        // of the function call is 0.
        if (!clad::utils::IsZeroOrNullValue(dArg)) {
          allArgsHaveZeroDerivatives = false;
          break;
        }
      }
      if (allArgsHaveZeroDerivatives) {
        Expr* call =
            m_Sema
                .ActOnCallExpr(getCurrentScope(), Clone(CE->getCallee()),
                               validLoc, llvm::MutableArrayRef<Expr*>(CallArgs),
                               validLoc)
                .get();
        auto* zero = getZeroInit(CE->getType());
        return StmtDiff(call, zero);
      }
    }
  }

  if (!callDiff) {
    // Overloaded derivative was not found, request the CladPlugin to
    // derive the called function.
    DiffRequest pushforwardFnRequest;
    pushforwardFnRequest.Function = FD;
    pushforwardFnRequest.Mode = GetPushForwardMode();
    pushforwardFnRequest.BaseFunctionName = utils::ComputeEffectiveFnName(FD);
    // Silence diag outputs in nested derivation process.
    pushforwardFnRequest.VerboseDiags = false;

    // Check if request already derived in DerivedFunctions.
    FunctionDecl* pushforwardFD =
        m_Builder.HandleNestedDiffRequest(pushforwardFnRequest);

    if (pushforwardFD) {
      if (baseDiff.getExpr()) {
        callDiff =
            BuildCallExprToMemFn(baseDiff.getExpr(), pushforwardFD->getName(),
                                 pushforwardFnArgs, CE->getBeginLoc());
      } else {
        Expr* execConfig = nullptr;
        if (auto KCE = dyn_cast<CUDAKernelCallExpr>(CE))
          execConfig = Clone(KCE->getConfig());
        callDiff = m_Sema
                       .ActOnCallExpr(getCurrentScope(),
                                      BuildDeclRef(pushforwardFD), validLoc,
                                      pushforwardFnArgs, validLoc, execConfig)
                       .get();
      }
    }
  }

  // If clad failed to derive it, try finding its derivative using
  // numerical diff.
  if (!callDiff) {
    Multiplier = diffArgs[0];
    Expr* call =
        m_Sema
            .ActOnCallExpr(getCurrentScope(), Clone(CE->getCallee()), validLoc,
                           llvm::MutableArrayRef<Expr*>(CallArgs), validLoc)
            .get();
    // FIXME: Extend this for multiarg support
    // Check if the function is eligible for numerical differentiation.
    if (CE->getNumArgs() == 1) {
      Expr* fnCallee = cast<CallExpr>(call)->getCallee();
      callDiff =
          GetSingleArgCentralDiffCall(fnCallee, CallArgs[0],
                                      /*targetPos=*/0, /*numArgs=*/1, CallArgs);
    }
    CallExprDiffDiagnostics(FD, CE->getBeginLoc());
    if (!callDiff) {
      auto zero =
          ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
      return StmtDiff(call, zero);
    }
    if (Multiplier)
      callDiff = BuildOp(BO_Mul, callDiff, BuildParens(Multiplier));
    return {call, callDiff};
  }

  if (FD->getReturnType()->isVoidType())
    return StmtDiff(callDiff, nullptr);
  auto valueAndPushforward =
      StoreAndRef(callDiff, "_t", /*forceDeclCreation=*/true);
  Expr* returnValue = utils::BuildMemberExpr(m_Sema, getCurrentScope(),
                                             valueAndPushforward, "value");
  Expr* pushforward = utils::BuildMemberExpr(
      m_Sema, getCurrentScope(), valueAndPushforward, "pushforward");
  return StmtDiff(returnValue, pushforward);
}

StmtDiff BaseForwardModeVisitor::VisitUnaryOperator(const UnaryOperator* UnOp) {
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
    Expr* derivedOp = diff.getExpr_dx();
    if (diff.getExpr_dx()->getType()->isPointerType())
      derivedOp = BuildOp(opKind, diff.getExpr_dx());
    return StmtDiff(op, derivedOp);
  } /* For supporting complex types */
  else if (opKind == UnaryOperatorKind::UO_Real ||
           opKind == UnaryOperatorKind::UO_Imag) {
    return StmtDiff(op, BuildOp(opKind, diff.getExpr_dx()));
  } else if (opKind == UnaryOperatorKind::UO_Deref) {
    if (Expr* dx = diff.getExpr_dx())
      return StmtDiff(op, BuildOp(opKind, dx));
    return StmtDiff(op, ConstantFolder::synthesizeLiteral(
                            m_Context.IntTy, m_Context, /*val=*/0));
  } else if (opKind == UnaryOperatorKind::UO_AddrOf) {
    return StmtDiff(op, BuildOp(opKind, diff.getExpr_dx()));
  } else if (opKind == UnaryOperatorKind::UO_LNot) {
    Expr* zero = getZeroInit(UnOp->getType());
    if (diff.getExpr_dx() && !isUnusedResult(diff.getExpr_dx()))
      return {BuildOp(BO_Comma, BuildParens(diff.getExpr_dx()), op), zero};
    return {op, zero};
  } else if (opKind == UnaryOperatorKind::UO_Not) {
    // ~x is 2^n - 1 - x for unsigned types and -x - 1 for the signed ones.
    // Either way, taking a derivative gives us -_d_x.
    Expr* derivedOp = BuildOp(UO_Minus, diff.getExpr_dx());
    return {op, derivedOp};
  } else {
    unsupportedOpWarn(UnOp->getEndLoc());
    auto zero =
        ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
    return StmtDiff(op, zero);
  }
}

StmtDiff
BaseForwardModeVisitor::VisitBinaryOperator(const BinaryOperator* BinOp) {
  StmtDiff Ldiff = Visit(BinOp->getLHS());
  StmtDiff Rdiff = Visit(BinOp->getRHS());

  ConstantFolder folder(m_Context);
  auto opCode = BinOp->getOpcode();
  Expr* opDiff = nullptr;

  auto deriveMul = [this](StmtDiff& Ldiff, StmtDiff& Rdiff) {
    Expr* LHS = BuildOp(BO_Mul, BuildParens(Ldiff.getExpr_dx()),
                        BuildParens(Rdiff.getExpr()));

    Expr* RHS = BuildOp(BO_Mul, BuildParens(Ldiff.getExpr()),
                        BuildParens(Rdiff.getExpr_dx()));

    return BuildOp(BO_Add, LHS, RHS);
  };

  auto deriveDiv = [this](StmtDiff& Ldiff, StmtDiff& Rdiff) {
    Expr* LHS = BuildOp(BO_Mul, BuildParens(Ldiff.getExpr_dx()),
                        BuildParens(Rdiff.getExpr()));

    Expr* RHS = BuildOp(BO_Mul, BuildParens(Ldiff.getExpr()),
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
  } else if (opCode == BO_Add || opCode == BO_Sub) {
    Expr* derivedL = nullptr;
    Expr* derivedR = nullptr;
    ComputeEffectiveDOperands(Ldiff, Rdiff, derivedL, derivedR);
    if (opCode == BO_Sub)
      derivedR = BuildParens(derivedR);
    opDiff = BuildOp(opCode, derivedL, derivedR);
  } else if (BinOp->isAssignmentOp()) {
    if (Ldiff.getExpr_dx()->isModifiableLvalue(m_Context) != Expr::MLV_Valid) {
      diag(DiagnosticsEngine::Warning, BinOp->getEndLoc(),
           "derivative of an assignment attempts to assign to unassignable "
           "expr, assignment ignored");
      opDiff = ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
    } else if (opCode == BO_Assign || opCode == BO_AddAssign ||
               opCode == BO_SubAssign) {
      Expr* derivedL = nullptr;
      Expr* derivedR = nullptr;
      ComputeEffectiveDOperands(Ldiff, Rdiff, derivedL, derivedR);
      opDiff = BuildOp(opCode, derivedL, derivedR);
    } else if (opCode == BO_MulAssign || opCode == BO_DivAssign) {
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
        opDiff =
            BuildOp(BO_Assign, Ldiff.getExpr_dx(), deriveMul(Ldiff, Rdiff));
      else if (opCode == BO_DivAssign)
        opDiff =
            BuildOp(BO_Assign, Ldiff.getExpr_dx(), deriveDiv(Ldiff, Rdiff));
    }
  } else if (opCode == BO_Comma) {
    // if expression is (E1, E2) then derivative is (E1', E1, E2')
    // because E1 may change some variables that E2 depends on.
    if (!isUnusedResult(Ldiff.getExpr_dx())) {
      opDiff = BuildOp(BO_Comma, BuildParens(Ldiff.getExpr_dx()),
                       BuildParens(Ldiff.getExpr()));
      opDiff = BuildOp(BO_Comma, BuildParens(opDiff),
                       BuildParens(Rdiff.getExpr_dx()));
    } else
      opDiff = BuildOp(BO_Comma, BuildParens(Ldiff.getExpr()),
                       BuildParens(Rdiff.getExpr_dx()));
  } else if (BinOp->isLogicalOp() || BinOp->isBitwiseOp() ||
             BinOp->isComparisonOp() || opCode == BO_Rem) {
    // For (A && B) return ((dA, A) && (dB, B)) to ensure correct evaluation and
    // correct derivative execution.
    auto buildOneSide = [this](StmtDiff& Xdiff) {
      if (Xdiff.getExpr_dx() && !isUnusedResult(Xdiff.getExpr_dx()))
        return BuildParens(BuildOp(BO_Comma, BuildParens(Xdiff.getExpr_dx()),
                                   BuildParens(Xdiff.getExpr())));
      return BuildParens(Xdiff.getExpr());
    };
    // dLL = (dL, L)
    Expr* dLL = buildOneSide(Ldiff);
    // dRR = (dR, R)
    Expr* dRR = buildOneSide(Rdiff);
    opDiff = BuildOp(opCode, dLL, dRR);

    // Since the both parts are included in the opDiff, there's no point in
    // including it as a Stmt_dx. Moreover, the fact that Stmt_dx is left
    // zero is used for treating expressions like ((A && B) && C) correctly.
    return StmtDiff(opDiff, getZeroInit(BinOp->getType()));
  } else if (BinOp->isShiftOp()) {
    // Shifting is essentially multiplicating the LHS by 2^RHS (or 2^-RHS).
    // We should do the same to the derivarive.
    opDiff = BuildOp(opCode, Ldiff.getExpr_dx(), Rdiff.getExpr());
  } else {
    // FIXME: add support for other binary operators
    unsupportedOpWarn(BinOp->getEndLoc());
    opDiff = ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
  }
  if (opDiff)
    opDiff = folder.fold(opDiff);
  // Recover the original operation from the Ldiff and Rdiff instead of
  // cloning the tree.
  Expr* op;
  if (opCode == BO_Comma)
    // Ldiff.getExpr() is already included in opDiff.
    op = Rdiff.getExpr();
  else
    op = BuildOp(opCode, Ldiff.getExpr(), Rdiff.getExpr());
  return StmtDiff(op, opDiff);
}

DeclDiff<VarDecl>
BaseForwardModeVisitor::DifferentiateVarDecl(const VarDecl* VD) {
  return DifferentiateVarDecl(VD, false);
}

DeclDiff<VarDecl>
BaseForwardModeVisitor::DifferentiateVarDecl(const VarDecl* VD,
                                             bool ignoreInit) {
  StmtDiff initDiff{};
  const Expr* init = VD->getInit();
  if (init) {
    if (!ignoreInit)
      initDiff = Visit(init);
    else
      initDiff = StmtDiff(Clone(init));
  }

  // Here we are assuming that derived type and the original type are same.
  // This may not necessarily be true in the future.
  VarDecl* VDClone =
      BuildVarDecl(VD->getType(), VD->getNameAsString(), initDiff.getExpr(),
                   VD->isDirectInit(), /*TSI=*/nullptr, VD->getInitStyle());
  // FIXME: Create unique identifier for derivative.
  Expr* initDx = initDiff.getExpr_dx();
  if (VD->getType()->isPointerType() && !initDx) {
    // initialize with nullptr.
    // NOLINTBEGIN(cppcoreguidelines-owned-memory)
    initDx =
        new (m_Context) CXXNullPtrLiteralExpr(VD->getType(), VD->getBeginLoc());
    // NOLINTEND(cppcoreguidelines-owned-memory)
  }
  VarDecl* VDDerived =
      BuildVarDecl(VD->getType(), "_d_" + VD->getNameAsString(), initDx,
                   VD->isDirectInit(), /*TSI=*/nullptr, VD->getInitStyle());
  m_Variables.emplace(VDClone, BuildDeclRef(VDDerived));
  return DeclDiff<VarDecl>(VDClone, VDDerived);
}

StmtDiff BaseForwardModeVisitor::VisitDeclStmt(const DeclStmt* DS) {
  llvm::SmallVector<Decl*, 4> decls;
  llvm::SmallVector<Decl*, 4> declsDiff;
  // If the type is marked as non_differentiable, skip generating its derivative
  // Get the iterator
  const auto* declsBegin = DS->decls().begin();
  const auto* declsEnd = DS->decls().end();

  // If the DeclStmt is not empty, check the first declaration.
  if (declsBegin != declsEnd && isa<VarDecl>(*declsBegin)) {
    auto* VD = dyn_cast<VarDecl>(*declsBegin);
    // Check for non-differentiable types.
    QualType QT = VD->getType();
    if (QT->isPointerType())
      QT = QT->getPointeeType();
    auto* typeDecl = QT->getAsCXXRecordDecl();
    // For lambda functions, we should also simply copy the original lambda. The
    // differentiation of lambdas is happening in the `VisitCallExpr`. For now,
    // only the declarations with lambda expressions without captures are
    // supported.
    if (typeDecl && (clad::utils::hasNonDifferentiableAttribute(typeDecl) ||
                     typeDecl->isLambda())) {
      for (auto* D : DS->decls()) {
        if (auto* VD = dyn_cast<VarDecl>(D))
          decls.push_back(VD);
        else
          diag(DiagnosticsEngine::Warning, D->getEndLoc(),
               "Unsupported declaration");
      }
      Stmt* DSClone = BuildDeclStmt(decls);
      return StmtDiff(DSClone, nullptr);
    }
  }

  // For each variable declaration v, create another declaration _d_v to
  // store derivatives for potential reassignments. E.g.
  // double y = x;
  // ->
  // double _d_y = _d_x; double y = x;
  for (auto D : DS->decls()) {
    if (auto VD = dyn_cast<VarDecl>(D)) {
      DeclDiff<VarDecl> VDDiff = DifferentiateVarDecl(VD);
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
    } else if (auto* SAD = dyn_cast<StaticAssertDecl>(D)) {
      DeclDiff<StaticAssertDecl> SADDiff = DifferentiateStaticAssertDecl(SAD);
      if (SADDiff.getDecl())
        decls.push_back(SADDiff.getDecl());
      if (SADDiff.getDecl_dx())
        declsDiff.push_back(SADDiff.getDecl_dx());
    } else {
      diag(DiagnosticsEngine::Warning, D->getEndLoc(),
           "Unsupported declaration");
    }
  }

  Stmt* DSClone = nullptr;
  Stmt* DSDiff = nullptr;
  if (!decls.empty())
    DSClone = BuildDeclStmt(decls);
  if (!declsDiff.empty())
    DSDiff = BuildDeclStmt(declsDiff);
  return StmtDiff(DSClone, DSDiff);
}

StmtDiff
BaseForwardModeVisitor::VisitImplicitCastExpr(const ImplicitCastExpr* ICE) {
  StmtDiff subExprDiff = Visit(ICE->getSubExpr());
  // Casts should be handled automatically when the result is used by
  // Sema::ActOn.../Build...
  return StmtDiff(subExprDiff.getExpr(), subExprDiff.getExpr_dx());
}

StmtDiff BaseForwardModeVisitor::VisitImplicitValueInitExpr(
    const ImplicitValueInitExpr* E) {
  return StmtDiff(Clone(E), Clone(E));
}

StmtDiff
BaseForwardModeVisitor::VisitCXXConstCastExpr(const CXXConstCastExpr* CCE) {
  StmtDiff subExprDiff = Visit(CCE->getSubExpr());
  Expr* castExpr =
      m_Sema
          .BuildCXXNamedCast(CCE->getBeginLoc(), tok::kw_const_cast,
                             CCE->getTypeInfoAsWritten(), subExprDiff.getExpr(),
                             CCE->getAngleBrackets(), CCE->getSourceRange())
          .get();
  Expr* castExprDiff =
      m_Sema
          .BuildCXXNamedCast(CCE->getBeginLoc(), tok::kw_const_cast,
                             CCE->getTypeInfoAsWritten(),
                             subExprDiff.getExpr_dx(), CCE->getAngleBrackets(),
                             CCE->getSourceRange())
          .get();
  return StmtDiff(castExpr, castExprDiff);
}

StmtDiff
BaseForwardModeVisitor::VisitCStyleCastExpr(const CStyleCastExpr* CSCE) {
  StmtDiff subExprDiff = Visit(CSCE->getSubExpr());
  // Create a new CStyleCastExpr with the same type and the same subexpression
  // as the original one.
  Expr* castExpr = m_Sema
                       .BuildCStyleCastExpr(
                           CSCE->getLParenLoc(), CSCE->getTypeInfoAsWritten(),
                           CSCE->getRParenLoc(), subExprDiff.getExpr())
                       .get();
  Expr* castExprDiff =
      m_Sema
          .BuildCStyleCastExpr(CSCE->getLParenLoc(),
                               CSCE->getTypeInfoAsWritten(),
                               CSCE->getRParenLoc(), subExprDiff.getExpr_dx())
          .get();
  return StmtDiff(castExpr, castExprDiff);
}

StmtDiff
BaseForwardModeVisitor::VisitCXXDefaultArgExpr(const CXXDefaultArgExpr* DE) {
  // FIXME: Shouldn't we simply clone the CXXDefaultArgExpr?
  // return {Clone(DE), Clone(DE)};
  return Visit(DE->getExpr());
}

StmtDiff
BaseForwardModeVisitor::VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr* BL) {
  llvm::APInt zero(m_Context.getIntWidth(m_Context.IntTy), /*value*/ 0);
  auto* constant0 =
      IntegerLiteral::Create(m_Context, zero, m_Context.IntTy, noLoc);
  return StmtDiff(Clone(BL), constant0);
}

StmtDiff
BaseForwardModeVisitor::VisitCharacterLiteral(const CharacterLiteral* CL) {
  llvm::APInt zero(m_Context.getIntWidth(m_Context.IntTy), /*value*/ 0);
  auto* constant0 =
      IntegerLiteral::Create(m_Context, zero, m_Context.IntTy, noLoc);
  return StmtDiff(Clone(CL), constant0);
}

StmtDiff BaseForwardModeVisitor::VisitStringLiteral(const StringLiteral* SL) {
  return StmtDiff(Clone(SL), StringLiteral::Create(
                                 m_Context, "", SL->getKind(), SL->isPascal(),
                                 SL->getType(), utils::GetValidSLoc(m_Sema)));
}

StmtDiff BaseForwardModeVisitor::VisitWhileStmt(const WhileStmt* WS) {
  // begin scope for while loop
  beginScope(Scope::ContinueScope | Scope::BreakScope | Scope::DeclScope |
             Scope::ControlScope);

  const VarDecl* condVar = WS->getConditionVariable();
  VarDecl* condVarClone = nullptr;
  DeclDiff<VarDecl> condVarRes;

  StmtDiff condDiff = Clone(WS->getCond());
  Expr* cond = condDiff.getExpr();

  // Check if the condition contais a variable declaration and create a
  // declaration of both the variable and it's adjoint before the while-loop.
  if (condVar) {
    condVarRes = DifferentiateVarDecl(condVar, /*ignoreInit=*/true);
    condVarClone = condVarRes.getDecl();
    if (condVarRes.getDecl_dx())
      addToCurrentBlock(BuildDeclStmt(condVarRes.getDecl_dx()));
    auto* condInit = condVarClone->getInit();
    condVarClone->setInit(nullptr);
    cond = BuildOp(BO_Assign, BuildDeclRef(condVarClone), condInit);
    addToCurrentBlock(BuildDeclStmt(condVarClone));
  }
  // Assignments in the condition are allowed, differentiate.
  if (cond) {
    cond = cond->IgnoreParenImpCasts();
    auto* condBO = dyn_cast<BinaryOperator>(cond);
    auto* condUO = dyn_cast<UnaryOperator>(cond);
    // FIXME: Currently we only support logical and assignment operators.
    if ((condBO && (condBO->isLogicalOp() || condBO->isAssignmentOp())) ||
        condUO) {
      StmtDiff condDiff = Visit(cond);
      // After Visit(cond) is called the derivative could either be recorded in
      // condDiff.getExpr() or condDiff.getExpr_dx(), hence we should build cond
      // differently which is implemented below visiting statements like "(x=0)"
      // records the differentiated statement in condDiff.getExpr_dx(), meaning
      // we have to build in the form ((cond_dx), (cond)), wrapping cond_dx and
      // cond into parentheses.
      //
      // Visiting statements like "(x=0) || false" records the result in
      // condDiff.getExpr(), meaning the differentiated condition is already.
      if (condDiff.getExpr_dx() &&
          (!isUnusedResult(condDiff.getExpr_dx()) || condUO))
        cond = BuildOp(BO_Comma, BuildParens(condDiff.getExpr_dx()),
                       BuildParens(condDiff.getExpr()));
      else
        cond = condDiff.getExpr();
    }
  }

  Sema::ConditionResult condRes;
  condRes = m_Sema.ActOnCondition(getCurrentScope(), noLoc, cond,
                                  Sema::ConditionKind::Boolean);

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

  Stmt* WSDiff =
      clad_compat::Sema_ActOnWhileStmt(m_Sema, condRes, bodyResult).get();
  // end scope for while loop
  endScope();
  return StmtDiff(WSDiff);
}

StmtDiff
BaseForwardModeVisitor::VisitContinueStmt(const ContinueStmt* ContStmt) {
  return StmtDiff(Clone(ContStmt));
}

StmtDiff BaseForwardModeVisitor::VisitDoStmt(const DoStmt* DS) {
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

/// Returns top switch statement in the `SwitchStack` of the given
/// Function Scope.
static SwitchStmt* getTopSwitchStmtOfSwitchStack(sema::FunctionScopeInfo* FSI) {
  return FSI->SwitchStack.back().getPointer();
}

StmtDiff BaseForwardModeVisitor::VisitSwitchStmt(const SwitchStmt* SS) {
  // Scope and block for initializing derived variables for condition
  // variable and switch-init declaration.
  beginScope(Scope::DeclScope);
  beginBlock();

  const VarDecl* condVarDecl = SS->getConditionVariable();
  VarDecl* condVarClone = nullptr;
  if (condVarDecl) {
    DeclDiff<VarDecl> condVarDeclDiff = DifferentiateVarDecl(condVarDecl);
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
  Stmt* switchStmtDiff = clad_compat::Sema_ActOnStartOfSwitchStmt(
                             m_Sema, initVarRes.getStmt(), condResult)
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
    utils::SetSwitchCaseSubStmt(activeSC, endBlock());
    endScope();
    activeSC = nullptr;
  }

  // for corresponding compound statement of the switch block
  endScope();
  switchStmtDiff =
      m_Sema.ActOnFinishSwitchStmt(noLoc, switchStmtDiff, endBlock()).get();

  // for switch statement
  endScope();

  addToCurrentBlock(switchStmtDiff);
  // for scope created for derived condition variable and switch init
  // statement.
  endScope();
  return StmtDiff(endBlock());
}

SwitchCase*
BaseForwardModeVisitor::DeriveSwitchStmtBodyHelper(const Stmt* stmt,
                                                   SwitchCase* activeSC) {
  if (auto SC = dyn_cast<SwitchCase>(stmt)) {
    // New switch case label have been visited. Pop the scope and block
    // corresponding to the active switch case label, and update its
    // substatement.
    if (activeSC) {
      utils::SetSwitchCaseSubStmt(activeSC, endBlock());
      endScope();
    }
    // sub statement will be updated later, either when the corresponding
    // next label is visited or the corresponding switch statement ends.
    SwitchCase* newActiveSC = nullptr;

    // We are not cloning the switch case label here because cloning will
    // also unnecessary clone substatement of the switch case label.
    if (auto newCaseSC = dyn_cast<CaseStmt>(SC)) {
      Expr* lhsClone =
          (newCaseSC->getLHS() ? Clone(newCaseSC->getLHS()) : nullptr);
      Expr* rhsClone =
          (newCaseSC->getRHS() ? Clone(newCaseSC->getRHS()) : nullptr);
      newActiveSC = CaseStmt::Create(m_Sema.getASTContext(), lhsClone, rhsClone,
                                     noLoc, noLoc, noLoc);

    } else if (isa<DefaultStmt>(SC)) {
      newActiveSC =
          new (m_Sema.getASTContext()) DefaultStmt(noLoc, noLoc, nullptr);
    }

    SwitchStmt* activeSwitch =
        getTopSwitchStmtOfSwitchStack(m_Sema.getCurFunction());
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

StmtDiff BaseForwardModeVisitor::VisitBreakStmt(const BreakStmt* stmt) {
  return StmtDiff(Clone(stmt));
}

StmtDiff
BaseForwardModeVisitor::VisitCXXConstructExpr(const CXXConstructExpr* CE) {
  llvm::SmallVector<Expr*, 4> clonedArgs, derivedArgs;
  for (auto arg : CE->arguments()) {
    auto argDiff = Visit(arg);
    clonedArgs.push_back(argDiff.getExpr());
    derivedArgs.push_back(argDiff.getExpr_dx());
  }

  Expr* pushforwardCall =
      BuildCustomDerivativeConstructorPFCall(CE, clonedArgs, derivedArgs);
  if (pushforwardCall) {
    auto valueAndPushforwardE = StoreAndRef(pushforwardCall);
    Expr* valueE = utils::BuildMemberExpr(m_Sema, getCurrentScope(),
                                          valueAndPushforwardE, "value");
    Expr* pushforwardE = utils::BuildMemberExpr(
        m_Sema, getCurrentScope(), valueAndPushforwardE, "pushforward");
    return StmtDiff(valueE, pushforwardE);
  }

  // Custom derivative not found. Create simple constructor calls based on the
  // given arguments. For example, if the primal constructor call is
  // 'C(a, b, c)' then we use the constructor call 'C(d_a, d_b, d_c)' for the
  // derivative.
  // FIXME: This is incorrect. It only works for very simple types such as
  // std::complex. We should ideally treat a constructor like a function and
  // thus differentiate its body, create a pushforward and use the pushforward
  // in the derivative code instead of the original constructor.
  Expr* clonedArgsE = nullptr;
  Expr* derivedArgsE = nullptr;
  // FIXME: Currently if the original initialisation expression is `{a, 1,
  // b}`, then we are creating derived initialisation expression as `{_d_a,
  // 0., _d_b}`. This is essentially incorrect and we should actually create a
  // forward mode derived constructor that would require same arguments as of
  // a pushforward function, that is, `{a, 1, b, _d_a, 0., _d_b}`.
  if (CE->getNumArgs() != 1) {
    if (CE->getNumArgs() == 0 && !CE->isListInitialization()) {
      // ParenList is empty -- default initialisation.
      // Passing empty parenList here will silently cause 'most vexing
      // parse' issue.
      return StmtDiff();
    } else {
      // Rely on the initializer list expressions as they seem to be more
      // flexible in terms of conversions and other similar scenarios where a
      // constructor is called implicitly.
      clonedArgsE = m_Sema.ActOnInitList(noLoc, clonedArgs, noLoc).get();
      derivedArgsE = m_Sema.ActOnInitList(noLoc, derivedArgs, noLoc).get();
    }
  } else {
    clonedArgsE = clonedArgs[0];
    derivedArgsE = derivedArgs[0];
  }
  // `CXXConstructExpr` node will be created automatically by passing these
  // initialiser to higher level `ActOn`/`Build` Sema functions.
  return {clonedArgsE, derivedArgsE};
}

StmtDiff BaseForwardModeVisitor::VisitExprWithCleanups(
    const clang::ExprWithCleanups* EWC) {
  // `ExprWithCleanups` node will be created automatically if it is required
  // by `ActOn`/`Build` Sema functions.
  return Visit(EWC->getSubExpr());
}

StmtDiff BaseForwardModeVisitor::VisitMaterializeTemporaryExpr(
    const clang::MaterializeTemporaryExpr* MTE) {
  // `MaterializeTemporaryExpr` node will be created automatically if it is
  // required by `ActOn`/`Build` Sema functions.
  StmtDiff MTEDiff = Visit(clad_compat::GetSubExpr(MTE));
  return MTEDiff;
}

StmtDiff BaseForwardModeVisitor::VisitCXXTemporaryObjectExpr(
    const clang::CXXTemporaryObjectExpr* TOE) {
  llvm::SmallVector<Expr*, 4> clonedArgs, derivedArgs;
  // FIXME: Currently if the original initialisation expression is `{a, 1,
  // b}`, then we are creating derived initialisation expression as `{_d_a,
  // 0., _d_b}`. This is essentially incorrect and we should actually create a
  // forward mode derived constructor that would require same arguments as of
  // a pushforward function, that is, `{a, 1, b, _d_a, 0., _d_b}`.
  for (auto arg : TOE->arguments()) {
    auto argDiff = Visit(arg);
    clonedArgs.push_back(argDiff.getExpr());
    derivedArgs.push_back(argDiff.getExpr_dx());
  }

  Expr* clonedTOE =
      m_Sema
          .ActOnCXXTypeConstructExpr(OpaquePtr<QualType>::make(TOE->getType()),
                                     utils::GetValidSLoc(m_Sema), clonedArgs,
                                     utils::GetValidSLoc(m_Sema),
                                     TOE->isListInitialization())
          .get();
  Expr* derivedTOE =
      m_Sema
          .ActOnCXXTypeConstructExpr(OpaquePtr<QualType>::make(TOE->getType()),
                                     utils::GetValidSLoc(m_Sema), derivedArgs,
                                     utils::GetValidSLoc(m_Sema),
                                     TOE->isListInitialization())
          .get();
  return {clonedTOE, derivedTOE};
}

StmtDiff
BaseForwardModeVisitor::VisitCXXThisExpr(const clang::CXXThisExpr* CTE) {
  return StmtDiff(const_cast<CXXThisExpr*>(CTE), m_ThisExprDerivative);
}

StmtDiff BaseForwardModeVisitor::VisitCXXNewExpr(const clang::CXXNewExpr* CNE) {
  StmtDiff initializerDiff;
  if (CNE->hasInitializer())
    initializerDiff = Visit(CNE->getInitializer());

  Expr* clonedArraySizeE = nullptr;
  Expr* derivedArraySizeE = nullptr;
  if (CNE->getArraySize()) {
    // FIXME: Only compute clone of original expression.
    // We can use `clone(..)`, but `clone` does not perform declaration
    // replacements and thus can cause issues.
    clonedArraySizeE =
        Visit(clad_compat::ArraySize_GetValue(CNE->getArraySize())).getExpr();
    // Array size is a non-differentiable expression, thus the original value
    // should be used in both the cloned and the derived statements.
    derivedArraySizeE = Clone(clonedArraySizeE);
  }
  Expr* clonedNewE = utils::BuildCXXNewExpr(
      m_Sema, CNE->getAllocatedType(), clonedArraySizeE,
      initializerDiff.getExpr(), CNE->getAllocatedTypeSourceInfo());
  Expr* derivedNewE = utils::BuildCXXNewExpr(
      m_Sema, CNE->getAllocatedType(), derivedArraySizeE,
      initializerDiff.getExpr_dx(), CNE->getAllocatedTypeSourceInfo());
  return {clonedNewE, derivedNewE};
}

StmtDiff
BaseForwardModeVisitor::VisitCXXDeleteExpr(const clang::CXXDeleteExpr* CDE) {
  StmtDiff argDiff = Visit(CDE->getArgument());
  Expr* clonedDeleteE =
      m_Sema
          .ActOnCXXDelete(noLoc, CDE->isGlobalDelete(), CDE->isArrayForm(),
                          argDiff.getExpr())
          .get();
  Expr* derivedDeleteE =
      m_Sema
          .ActOnCXXDelete(noLoc, CDE->isGlobalDelete(), CDE->isArrayForm(),
                          argDiff.getExpr_dx())
          .get();
  return {clonedDeleteE, derivedDeleteE};
}

StmtDiff BaseForwardModeVisitor::VisitCXXStaticCastExpr(
    const clang::CXXStaticCastExpr* CSE) {
  auto diff = Visit(CSE->getSubExpr());
  Expr* clonedE =
      m_Sema
          .BuildCXXNamedCast(noLoc, tok::TokenKind::kw_static_cast,
                             CSE->getTypeInfoAsWritten(), diff.getExpr(),
                             SourceRange(), SourceRange())
          .get();
  Expr* derivedE =
      m_Sema
          .BuildCXXNamedCast(noLoc, tok::TokenKind::kw_static_cast,
                             CSE->getTypeInfoAsWritten(), diff.getExpr_dx(),
                             SourceRange(), SourceRange())
          .get();
  return {clonedE, derivedE};
}

StmtDiff BaseForwardModeVisitor::VisitCXXFunctionalCastExpr(
    const clang::CXXFunctionalCastExpr* FCE) {
  StmtDiff castExprDiff = Visit(FCE->getSubExpr());
  SourceLocation fakeLoc = utils::GetValidSLoc(m_Sema);
  Expr* clonedFCE = m_Sema
                        .BuildCXXFunctionalCastExpr(
                            FCE->getTypeInfoAsWritten(), FCE->getType(),
                            fakeLoc, castExprDiff.getExpr(), fakeLoc)
                        .get();
  Expr* derivedFCE = m_Sema
                         .BuildCXXFunctionalCastExpr(
                             FCE->getTypeInfoAsWritten(), FCE->getType(),
                             fakeLoc, castExprDiff.getExpr_dx(), fakeLoc)
                         .get();
  return {clonedFCE, derivedFCE};
}

StmtDiff BaseForwardModeVisitor::VisitCXXBindTemporaryExpr(
    const clang::CXXBindTemporaryExpr* BTE) {
  // `CXXBindTemporaryExpr` node will be created automatically, if it is
  // required, by `ActOn`/`Build` Sema functions.
  StmtDiff BTEDiff = Visit(BTE->getSubExpr());
  return BTEDiff;
}

StmtDiff BaseForwardModeVisitor::VisitCXXNullPtrLiteralExpr(
    const clang::CXXNullPtrLiteralExpr* NPL) {
  return {Clone(NPL), Clone(NPL)};
}

StmtDiff BaseForwardModeVisitor::VisitUnaryExprOrTypeTraitExpr(
    const clang::UnaryExprOrTypeTraitExpr* UE) {
  return {Clone(UE), Clone(UE)};
}

StmtDiff BaseForwardModeVisitor::VisitPseudoObjectExpr(
    const clang::PseudoObjectExpr* POE) {
  return {Clone(POE),
          ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0)};
}

StmtDiff BaseForwardModeVisitor::VisitSubstNonTypeTemplateParmExpr(
    const clang::SubstNonTypeTemplateParmExpr* NTTP) {
  return Visit(NTTP->getReplacement());
}

DeclDiff<StaticAssertDecl>
BaseForwardModeVisitor::DifferentiateStaticAssertDecl(
    const clang::StaticAssertDecl* SAD) {
  return DeclDiff<StaticAssertDecl>();
}

StmtDiff BaseForwardModeVisitor::VisitCXXStdInitializerListExpr(
    const clang::CXXStdInitializerListExpr* ILE) {
  return Visit(ILE->getSubExpr());
}

StmtDiff BaseForwardModeVisitor::VisitCXXScalarValueInitExpr(
    const CXXScalarValueInitExpr* SVIE) {
  return {Clone(SVIE), Clone(SVIE)};
}

clang::Expr* BaseForwardModeVisitor::BuildCustomDerivativeConstructorPFCall(
    const clang::CXXConstructExpr* CE,
    llvm::SmallVectorImpl<clang::Expr*>& clonedArgs,
    llvm::SmallVectorImpl<clang::Expr*>& derivedArgs) {
  llvm::SmallVector<Expr*, 4> customPushforwardArgs;
  QualType constructorPushforwardTagT = GetCladConstructorPushforwardTagOfType(
      CE->getType().withoutLocalFastQualifiers());
  // Builds clad::ConstructorPushforwardTag<T> declaration
  Expr* constructorPushforwardTagArg =
      m_Sema
          .BuildCXXTypeConstructExpr(
              m_Context.getTrivialTypeSourceInfo(constructorPushforwardTagT,
                                                 utils::GetValidSLoc(m_Sema)),
              noLoc, MultiExprArg{}, noLoc, /*ListInitialization=*/false)
          .get();
  customPushforwardArgs.push_back(constructorPushforwardTagArg);
  customPushforwardArgs.append(clonedArgs.begin(), clonedArgs.end());
  customPushforwardArgs.append(derivedArgs.begin(), derivedArgs.end());
  std::string customPushforwardName =
      clad::utils::ComputeEffectiveFnName(CE->getConstructor()) +
      GetPushForwardFunctionSuffix();
  // FIXME: We should not use const_cast to get the decl context here.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  Expr* pushforwardCall = m_Builder.BuildCallToCustomDerivativeOrNumericalDiff(
      customPushforwardName, customPushforwardArgs, getCurrentScope(),
      const_cast<DeclContext*>(CE->getConstructor()->getDeclContext()));
  return pushforwardCall;
}
} // end namespace clad
