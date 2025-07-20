//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clad/Differentiator/ReverseModeVisitor.h"

#include "ConstantFolder.h"

#include "TBRAnalyzer.h"
#include "clad/Differentiator/DerivativeBuilder.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/ErrorEstimator.h"
#include "clad/Differentiator/ExternalRMVSource.h"
#include "clad/Differentiator/MultiplexExternalRMVSource.h"
#include "clad/Differentiator/StmtClone.h"
#include "clad/Differentiator/VisitorBase.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTLambda.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/Type.h"
#include "clang/Basic/LLVM.h" // for clang::isa
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Basic/TypeTraits.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Template.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"

using namespace clang;

namespace clad {

Expr* getArraySizeExpr(const ArrayType* AT, ASTContext& context,
                       ReverseModeVisitor& rvm) {
  if (const auto* const CAT = dyn_cast<ConstantArrayType>(AT))
    return ConstantFolder::synthesizeLiteral(context.getSizeType(), context,
                                             CAT->getSize().getZExtValue());
  if (const auto* VSAT = dyn_cast<VariableArrayType>(AT))
    return rvm.Clone(VSAT->getSizeExpr());

  return nullptr;
}

Expr* ReverseModeVisitor::getStdInitListSizeExpr(const Expr* E) {
  if (E)
    if (const auto* CXXILE =
            dyn_cast<CXXStdInitializerListExpr>(E->IgnoreImplicit()))
      if (const auto* ILE =
              dyn_cast<InitListExpr>(CXXILE->getSubExpr()->IgnoreImplicit())) {
        unsigned numInits = ILE->getNumInits();
        return ConstantFolder::synthesizeLiteral(m_Context.getSizeType(),
                                                 m_Context, numInits);
      }
  return nullptr;
}

  Expr* ReverseModeVisitor::CladTapeResult::Last() {
    LookupResult& Back = V.GetCladTapeBack();
    CXXScopeSpec CSS;
    CSS.Extend(V.m_Context, utils::GetCladNamespace(V.m_Sema), noLoc, noLoc);
    Expr* BackDRE = V.m_Sema
                        .BuildDeclarationNameExpr(CSS, Back,
                                                  /*AcceptInvalidDecl=*/false)
                        .get();
    Expr* Call =
        V.m_Sema.ActOnCallExpr(V.getCurrentScope(), BackDRE, noLoc, Ref, noLoc)
            .get();
    return Call;
  }

  ReverseModeVisitor::CladTapeResult
  ReverseModeVisitor::MakeCladTapeFor(Expr* E, llvm::StringRef prefix,
                                      clang::QualType type) {
    assert(E && "must be provided");
    E = E->IgnoreImplicit();
    if (type.isNull())
      type = E->getType();
    QualType TapeType = GetCladTapeOfType(utils::getNonConstType(type, m_Sema));
    LookupResult& Push = GetCladTapePush();
    LookupResult& Pop = GetCladTapePop();
    Expr* TapeRef =
        BuildDeclRef(GlobalStoreImpl(TapeType, prefix, getZeroInit(TapeType)));
    auto* VD = cast<VarDecl>(cast<DeclRefExpr>(TapeRef)->getDecl());
    // Add fake location, since Clang AST does assert(Loc.isValid()) somewhere.
    VD->setLocation(m_DiffReq->getLocation());
    CXXScopeSpec CSS;
    CSS.Extend(m_Context, utils::GetCladNamespace(m_Sema), noLoc, noLoc);
    auto* PopDRE = m_Sema
                       .BuildDeclarationNameExpr(CSS, Pop,
                                                 /*AcceptInvalidDecl=*/false)
                       .get();
    auto* PushDRE = m_Sema
                        .BuildDeclarationNameExpr(CSS, Push,
                                                  /*AcceptInvalidDecl=*/false)
                        .get();
    Expr* PopExpr =
        m_Sema.ActOnCallExpr(getCurrentScope(), PopDRE, noLoc, TapeRef, noLoc)
            .get();
    Expr* CallArgs[] = {TapeRef, E};
    Expr* PushExpr =
        m_Sema.ActOnCallExpr(getCurrentScope(), PushDRE, noLoc, CallArgs, noLoc)
            .get();
    return CladTapeResult{*this, PushExpr, PopExpr, TapeRef};
  }

  bool ReverseModeVisitor::shouldUseCudaAtomicOps(const Expr* E) {
    if (!m_Context.getLangOpts().CUDA)
      return false;

    if (!isa<DeclRefExpr>(E))
      return false;

    const auto* DRE = cast<DeclRefExpr>(E);

    if (const auto* PVD = dyn_cast<ParmVarDecl>(DRE->getDecl())) {
      if (m_DiffReq->hasAttr<clang::CUDAGlobalAttr>())
        // Check whether this param is in the global memory of the GPU
        return m_DiffReq.HasIndependentParameter(PVD);
      if (m_DiffReq->hasAttr<clang::CUDADeviceAttr>()) {
        for (auto index : m_DiffReq.CUDAGlobalArgsIndexes) {
          const auto* PVDOrig = m_DiffReq->getParamDecl(index);
          if ("_d_" + PVDOrig->getNameAsString() == PVD->getNameAsString() &&
              (utils::isArrayOrPointerType(PVDOrig->getType()) ||
               PVDOrig->getType()->isReferenceType()))
            return true;
        }
      }
    }

    return false;
  }

  clang::Expr* ReverseModeVisitor::BuildCallToCudaAtomicAdd(clang::Expr* LHS,
                                                            clang::Expr* RHS) {
    DeclarationName atomicAddId = &m_Context.Idents.get("atomicAdd");
    LookupResult lookupResult(m_Sema, atomicAddId, SourceLocation(),
                              Sema::LookupOrdinaryName);
    m_Sema.LookupQualifiedName(lookupResult,
                               m_Context.getTranslationUnitDecl());

    CXXScopeSpec SS;
    Expr* UnresolvedLookup =
        m_Sema.BuildDeclarationNameExpr(SS, lookupResult, /*ADL=*/true).get();

    Expr* finalLHS = LHS;
    if (auto* UO = dyn_cast<UnaryOperator>(LHS)) {
      if (UO->getOpcode() == UnaryOperatorKind::UO_Deref)
        finalLHS = UO->getSubExpr()->IgnoreImplicit();
    } else if (!LHS->getType()->isPointerType() &&
               !LHS->getType()->isReferenceType())
      finalLHS = BuildOp(UnaryOperatorKind::UO_AddrOf, LHS);

    llvm::SmallVector<Expr*, 2> atomicArgs = {finalLHS, RHS};

    assert(!m_Builder.noOverloadExists(UnresolvedLookup, atomicArgs) &&
           "atomicAdd function not found");

    Expr* atomicAddCall =
        m_Sema
            .ActOnCallExpr(
                getCurrentScope(),
                /*Fn=*/UnresolvedLookup,
                /*LParenLoc=*/noLoc,
                /*ArgExprs=*/llvm::MutableArrayRef<Expr*>(atomicArgs),
                /*RParenLoc=*/m_DiffReq->getLocation())
            .get();

    return atomicAddCall;
  }

  Expr* ReverseModeVisitor::CheckAndBuildCallToMemset(Expr* LHS, Expr* RHS) {
    Expr* size = nullptr;
    if (auto* callExpr = dyn_cast_or_null<CallExpr>(RHS))
      if (auto* declRef =
              dyn_cast<DeclRefExpr>(callExpr->getCallee()->IgnoreImpCasts()))
        if (auto* FD = dyn_cast<FunctionDecl>(declRef->getDecl())) {
          if (FD->getNameAsString() == "malloc")
            size = callExpr->getArg(0);
          else if (FD->getNameAsString() == "realloc")
            size = callExpr->getArg(1);
        }

    if (size) {
      llvm::SmallVector<Expr*, 3> args = {LHS, getZeroInit(m_Context.IntTy),
                                          size};
      return GetFunctionCall("memset", "", args);
    }

    return nullptr;
  }

  ReverseModeVisitor::ReverseModeVisitor(DerivativeBuilder& builder,
                                         const DiffRequest& request)
      : VisitorBase(builder, request) {}

  ReverseModeVisitor::~ReverseModeVisitor() {
    if (m_ExternalSource) {
      // Inform external sources that `ReverseModeVisitor` object no longer
      // exists.
      // FIXME: Make this so the lifetime scope of the source matches.
      // m_ExternalSource->ForgetRMV();
      // Free the external sources multiplexer since we own this resource.
      delete m_ExternalSource;
    }
  }

  DerivativeAndOverload ReverseModeVisitor::Derive() {
    assert(m_DiffReq.Function && "Must not be null.");

    PrettyStackTraceDerivative CrashInfo(m_DiffReq, m_Blocks, m_Sema,
                                         &m_CurVisitedStmt);

    if (m_ExternalSource)
      m_ExternalSource->ActOnStartOfDerive();
    if (m_DiffReq.Mode == DiffMode::error_estimation)
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      const_cast<DiffRequest&>(m_DiffReq).Mode = DiffMode::reverse;

    QualType returnTy = m_DiffReq->getReturnType();
    // If reverse mode differentiates only part of the arguments it needs to
    // generate an overload that can take in all the diff variables
    bool shouldCreateOverload = false;
    // FIXME: Gradient overload doesn't know how to handle additional parameters
    // added by the plugins yet.
    if (m_DiffReq.Mode == DiffMode::reverse) {
      if (returnTy->isRealType())
        m_Pullback =
            ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context,
                                              /*val=*/1);
      else if (!returnTy->isVoidType()) {
        diag(DiagnosticsEngine::Warning, m_DiffReq.Function->getBeginLoc(),
             "clad::gradient only supports differentiation functions of real "
             "return types. Return stmt ignored.");
        diag(DiagnosticsEngine::Note, m_DiffReq.CallContext->getBeginLoc(),
             "Use clad::jacobian to compute derivatives of multiple real "
             "outputs w.r.t. multiple real inputs.");
      }
      shouldCreateOverload = !m_ExternalSource;
      if (!m_DiffReq.DeclarationOnly && !m_DiffReq.DerivedFDPrototypes.empty())
        // If the overload is already created, we don't need to create it again.
        shouldCreateOverload = false;
    }
    // For a function f of type R(A1, A2, ..., An),
    // the type of the gradient function is void(A1, A2, ..., An, R*, R*, ...,
    // R*) . the type of the jacobian function is void(A1, A2, ..., An, R*, R*)
    // and for error estimation, the function type is
    // void(A1, A2, ..., An, R*, R*, ..., R*, double&)
    llvm::SmallVector<QualType, 1> customParams{};
    if (m_ExternalSource)
      m_ExternalSource->ActAfterCreatingDerivedFnParamTypes(customParams);
    QualType dFnType = GetDerivativeType(customParams);

    // Check if the function is already declared as a custom derivative.
    std::string name = m_DiffReq.ComputeDerivativeName();

    // FIXME: We should not use const_cast to get the decl context here.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto* DC = const_cast<DeclContext*>(m_DiffReq->getDeclContext());
    if (FunctionDecl* customDerivative =
            m_Builder.LookupCustomDerivativeDecl(name, DC, dFnType)) {
      // Set m_Derivative for creating the overload.
      m_Derivative = customDerivative;
      if (shouldCreateOverload)
        return DerivativeAndOverload{m_Derivative, CreateDerivativeOverload()};
      return DerivativeAndOverload{m_Derivative, /*overload=*/nullptr};
    }

    // Create the gradient function declaration.
    llvm::SaveAndRestore<DeclContext*> SaveContext(m_Sema.CurContext);
    llvm::SaveAndRestore<Scope*> SaveScope(getCurrentScope(),
                                           getEnclosingNamespaceOrTUScope());
    m_Sema.CurContext = DC;
    SourceLocation loc = m_DiffReq->getLocation();
    DeclarationNameInfo DNI = utils::BuildDeclarationNameInfo(m_Sema, name);
    DeclWithContext result = m_Builder.cloneFunction(m_DiffReq.Function, *this,
                                                     DC, loc, DNI, dFnType);
    m_Derivative = result.first;

    if (m_ExternalSource)
      m_ExternalSource->ActBeforeCreatingDerivedFnScope();

    // Function declaration scope
    beginScope(Scope::FunctionPrototypeScope | Scope::FunctionDeclarationScope |
               Scope::DeclScope);
    m_Sema.PushFunctionScope();
    m_Sema.PushDeclContext(getCurrentScope(), m_Derivative);

    if (m_ExternalSource)
      m_ExternalSource->ActAfterCreatingDerivedFnScope();

    llvm::SmallVector<ParmVarDecl*, 8> params;
    BuildParams(params);

    if (m_ExternalSource)
      m_ExternalSource->ActAfterCreatingDerivedFnParams(params);

    m_Derivative->setParams(params);
    m_Derivative->setBody(nullptr);

    if (!m_DiffReq.DeclarationOnly) {
      if (m_ExternalSource)
        m_ExternalSource->ActBeforeCreatingDerivedFnBodyScope();

      // Function body scope.
      beginScope(Scope::FnScope | Scope::DeclScope);
      m_DerivativeFnScope = getCurrentScope();
      beginBlock();
      if (m_ExternalSource)
        m_ExternalSource->ActOnStartOfDerivedFnBody(m_DiffReq);

      if (m_DiffReq.use_enzyme) {
        assert(m_DiffReq.Mode == DiffMode::reverse && "Not in reverse?");
        DifferentiateWithEnzyme();
      } else {
        DifferentiateWithClad();
      }

      Stmt* fnBody = endBlock();
      m_Derivative->setBody(fnBody);

      endScope(); // Function body scope
    }

    m_Sema.PopFunctionScopeInfo();
    m_Sema.PopDeclContext();
    endScope(); // Function decl scope

    if (auto* RD = dyn_cast<RecordDecl>(m_Derivative->getDeclContext())) {
      DeclContext::lookup_result R =
          RD->getPrimaryContext()->lookup(m_Derivative->getDeclName());
      FunctionDecl* FoundFD =
          R.empty() ? nullptr : dyn_cast<FunctionDecl>(R.front());
      if (!RD->isLambda() && !R.empty() &&
          !m_Builder.m_DFC.IsCladDerivative(FoundFD)) {
        Sema::NestedNameSpecInfo IdInfo(RD->getIdentifier(), noLoc, noLoc,
                                        /*ObjectType=*/nullptr);
        // FIXME: Address nested classes where SS should be set.
        CXXScopeSpec SS;
        m_Sema.BuildCXXNestedNameSpecifier(getCurrentScope(), IdInfo,
                                           /*EnteringContext=*/true, SS,
                                           /*ScopeLookupResult=*/nullptr,
                                           /*ErrorRecoveryLookup=*/false);
        m_Derivative->setQualifierInfo(SS.getWithLocInContext(m_Context));
        m_Derivative->setLexicalDeclContext(RD->getParent());
      }
    }

    if (!shouldCreateOverload)
      return DerivativeAndOverload{result.first, /*overload=*/nullptr};

    return DerivativeAndOverload{result.first, CreateDerivativeOverload()};
  }

  void ReverseModeVisitor::DifferentiateWithClad() {
    if (m_DiffReq.Mode == DiffMode::reverse && !m_ExternalSource) {
      // create derived variables for parameters which are not part of
      // independent variables (args).
      for (const ParmVarDecl* param : m_NonIndepParams) {
        QualType paramTy = param->getType();
        if (utils::isArrayOrPointerType(paramTy)) {
          // We cannot initialize derived variable for pointer types because
          // we do not know the correct size.
          if (!utils::GetValueType(paramTy).isConstQualified()) {
            diag(DiagnosticsEngine::Error, param->getLocation(),
                 "Non-differentiable non-const pointer and array parameters "
                 "are not supported. Please differentiate w.r.t. '%0' or mark "
                 "it const.",
                 {param->getNameAsString()});
            return;
          }
          continue;
        }
        auto VDDerivedType = utils::getNonConstType(paramTy, m_Sema);
        auto* VDDerived =
            BuildGlobalVarDecl(VDDerivedType, "_d_" + param->getNameAsString(),
                               getZeroInit(VDDerivedType));
        m_Variables[param] = BuildDeclRef(VDDerived);
        addToBlock(BuildDeclStmt(VDDerived), m_Globals);
      }
    }

    // If we the differentiated function is a constructor, generate `this`
    // object and differentiate its inits.
    Stmts initsDiff;
    if (const auto* CD = dyn_cast<CXXConstructorDecl>(m_DiffReq.Function)) {
      StmtDiff thisObj;
      // Constructors with only linear operations do not require
      // `_this` in the reverse sweep.
      // FIXME: remove this check when our analysis is powerful enough.
      if (!utils::isLinearConstructor(CD, m_Context)) {
        QualType thisTy = CD->getThisType();
        thisObj = BuildThisExpr(thisTy);
        initsDiff.push_back(thisObj.getStmt_dx());
      }

      for (CXXCtorInitializer* CI : CD->inits()) {
        StmtDiff CI_diff = DifferentiateCtorInit(CI, thisObj.getExpr());
        addToCurrentBlock(CI_diff.getStmt(), direction::forward);
        initsDiff.push_back(CI_diff.getStmt_dx());
      }
    }

    // Start the visitation process which outputs the statements in the
    // current block.
    StmtDiff BodyDiff = Visit(m_DiffReq->getBody());
    Stmt* Forward = BodyDiff.getStmt();
    Stmt* Reverse = BodyDiff.getStmt_dx();
    // Create the body of the function.
    // Firstly, all "global" Stmts are put into fn's body.
    for (Stmt* S : m_Globals)
      addToCurrentBlock(S, direction::forward);
    // Forward pass.
    if (auto* CS = dyn_cast_or_null<CompoundStmt>(Forward))
      for (Stmt* S : CS->body())
        addToCurrentBlock(S, direction::forward);
    else
      addToCurrentBlock(Forward, direction::forward);
    // Reverse pass.
    if (auto* RCS = dyn_cast_or_null<CompoundStmt>(Reverse))
      for (Stmt* S : RCS->body())
        addToCurrentBlock(S, direction::forward);
    else
      addToCurrentBlock(Reverse, direction::forward);
    for (auto S = initsDiff.rbegin(), S_end = initsDiff.rend(); S != S_end; ++S)
      addToCurrentBlock(*S, direction::forward);
    // Add delete statements present in m_DeallocExprs to the current block.
    for (auto* S : m_DeallocExprs)
      if (auto* CS = dyn_cast<CompoundStmt>(S))
        for (Stmt* S : CS->body())
          addToCurrentBlock(S, direction::forward);
      else
        addToCurrentBlock(S, direction::forward);

    if (m_ExternalSource)
      m_ExternalSource->ActOnEndOfDerivedFnBody();
  }

  StmtDiff ReverseModeVisitor::BuildThisExpr(QualType thisTy) {
    // Build `sizeof(T)`
    QualType recordTy = thisTy->getPointeeType();
    TypeSourceInfo* TSI = m_Context.getTrivialTypeSourceInfo(recordTy, noLoc);
    Expr* size = new (m_Context) UnaryExprOrTypeTraitExpr(
        UETT_SizeOf, TSI, m_Context.getSizeType(), noLoc, noLoc);

    // Build `malloc(sizeof(T))`
    llvm::SmallVector<clang::Expr*, 1> param{size};
    Expr* init = GetFunctionCall("malloc", "", param);

    // Build `(T*)malloc(sizeof(T))`
    TypeSourceInfo* ptr_TSI = m_Context.getTrivialTypeSourceInfo(thisTy, noLoc);
    init = m_Sema.BuildCStyleCastExpr(noLoc, ptr_TSI, noLoc, init).get();

    // Build T* _this = (T*)malloc(sizeof(T));
    VarDecl* thisDecl = BuildGlobalVarDecl(thisTy, "_this", init);
    addToCurrentBlock(BuildDeclStmt(thisDecl), direction::forward);

    param[0] = BuildDeclRef(thisDecl);
    Expr* freeCall = GetFunctionCall("free", "", param);
    return {BuildDeclRef(thisDecl), freeCall};
  }

  StmtDiff ReverseModeVisitor::VisitCXXTryStmt(const CXXTryStmt* TS) {
    // FIXME: Add support for try statements.
    diag(DiagnosticsEngine::Warning, TS->getBeginLoc(),
         "Try statements are not supported, ignored.");
    return StmtDiff();
  }

  StmtDiff ReverseModeVisitor::DifferentiateCtorInit(CXXCtorInitializer* CI,
                                                     Expr* thisExpr) {
    llvm::StringRef fieldName = CI->getMember()->getName();
    Expr* memberDiff = utils::BuildMemberExpr(m_Sema, getCurrentScope(),
                                              m_ThisExprDerivative, fieldName);

    beginBlock(direction::reverse);
    QualType memberTy = CI->getMember()->getType();
    if (memberTy->isRealType()) {
      Stmt* assign_zero =
          BuildOp(BO_Assign, memberDiff, getZeroInit(memberDiff->getType()));
      addToCurrentBlock(assign_zero, direction::reverse);
    }
    StmtDiff initDiff = Visit(CI->getInit(), memberDiff);
    addToCurrentBlock(initDiff.getStmt_dx(), direction::reverse);
    Stmt* init = nullptr;
    if (thisExpr) {
      Expr* member = utils::BuildMemberExpr(m_Sema, getCurrentScope(), thisExpr,
                                            fieldName);
      init = BuildOp(BO_Assign, member, initDiff.getExpr());
    }
    return {init, endBlock(direction::reverse)};
  }

  void ReverseModeVisitor::DifferentiateWithEnzyme() {
    unsigned numParams = m_DiffReq->getNumParams();
    auto origParams = m_DiffReq->parameters();
    llvm::ArrayRef<ParmVarDecl*> paramsRef = m_Derivative->parameters();
    const auto* originalFnType =
        dyn_cast<FunctionProtoType>(m_DiffReq->getType());

    // Prepare Arguments and Parameters to enzyme_autodiff
    llvm::SmallVector<Expr*, 16> enzymeArgs;
    llvm::SmallVector<ParmVarDecl*, 16> enzymeParams;
    llvm::SmallVector<ParmVarDecl*, 16> enzymeRealParams;
    llvm::SmallVector<ParmVarDecl*, 16> enzymeRealParamsDerived;

    // First add the function itself as a parameter/argument
    // FIXME: We should not use const_cast to get the decl context here.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    enzymeArgs.push_back(
        BuildDeclRef(const_cast<FunctionDecl*>(m_DiffReq.Function)));
    // FIXME: We should not use const_cast to get the decl context here.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto* fdDeclContext = const_cast<DeclContext*>(m_DiffReq->getDeclContext());
    enzymeParams.push_back(m_Sema.BuildParmVarDeclForTypedef(
        fdDeclContext, noLoc, m_DiffReq->getType()));

    // Add rest of the parameters/arguments
    for (unsigned i = 0; i < numParams; i++) {
      // First Add the original parameter
      enzymeArgs.push_back(BuildDeclRef(paramsRef[i]));
      enzymeParams.push_back(m_Sema.BuildParmVarDeclForTypedef(
          fdDeclContext, noLoc, paramsRef[i]->getType()));

      QualType paramType = origParams[i]->getOriginalType();
      // If original parameter is of a differentiable real type(but not
      // array/pointer), then add it to the list of params whose gradient must
      // be extracted later from the EnzymeGradient structure
      if (paramType->isRealFloatingType()) {
        enzymeRealParams.push_back(paramsRef[i]);
        enzymeRealParamsDerived.push_back(paramsRef[numParams + i]);
      } else if (utils::isArrayOrPointerType(paramType)) {
        // Add the corresponding array/pointer variable
        enzymeArgs.push_back(BuildDeclRef(paramsRef[numParams + i]));
        enzymeParams.push_back(m_Sema.BuildParmVarDeclForTypedef(
            fdDeclContext, noLoc, paramsRef[numParams + i]->getType()));
      }
    }

    llvm::SmallVector<QualType, 16> enzymeParamsType;
    for (auto* i : enzymeParams)
      enzymeParamsType.push_back(i->getType());

    QualType QT;
    if (!enzymeRealParams.empty()) {
      // Find the EnzymeGradient datastructure
      auto* gradDecl =
          utils::LookupTemplateDeclInCladNamespace(m_Sema, "EnzymeGradient");

      TemplateArgumentListInfo TLI{};
      llvm::APSInt argValue(std::to_string(enzymeRealParams.size()));
      TemplateArgument TA(m_Context, argValue, m_Context.UnsignedIntTy);
      TLI.addArgument(TemplateArgumentLoc(TA, TemplateArgumentLocInfo()));

      QT = utils::InstantiateTemplate(m_Sema, gradDecl, TLI);
    } else {
      QT = m_Context.VoidTy;
    }

    // Prepare Function call
    std::string enzymeCallName =
        "__enzyme_autodiff_" + m_DiffReq->getNameAsString();
    IdentifierInfo* IIEnzyme = &m_Context.Idents.get(enzymeCallName);
    DeclarationName nameEnzyme(IIEnzyme);
    QualType enzymeFunctionType =
        m_Sema.BuildFunctionType(QT, enzymeParamsType, noLoc, nameEnzyme,
                                 originalFnType->getExtProtoInfo());
    FunctionDecl* enzymeCallFD = FunctionDecl::Create(
        m_Context, fdDeclContext, noLoc, noLoc, nameEnzyme, enzymeFunctionType,
        m_DiffReq->getTypeSourceInfo(), SC_Extern);
    enzymeCallFD->setParams(enzymeParams);
    Expr* enzymeCall = BuildCallExprToFunction(enzymeCallFD, enzymeArgs);

    // Prepare the statements that assign the gradients to
    // non array/pointer type parameters of the original function
    if (!enzymeRealParams.empty()) {
      VarDecl* gradVD = BuildVarDecl(QT, "grad", enzymeCall);
      addToCurrentBlock(BuildDeclStmt(gradVD), direction::forward);

      for (unsigned i = 0; i < enzymeRealParams.size(); i++) {
        auto* LHSExpr =
            BuildOp(UO_Deref, BuildDeclRef(enzymeRealParamsDerived[i]));

        auto* ME = utils::BuildMemberExpr(m_Sema, getCurrentScope(),
                                          BuildDeclRef(gradVD), "d_arr");

        Expr* gradIndex = dyn_cast<Expr>(
            IntegerLiteral::Create(m_Context, llvm::APSInt(std::to_string(i)),
                                   m_Context.UnsignedIntTy, noLoc));
        Expr* RHSExpr =
            m_Sema.CreateBuiltinArraySubscriptExpr(ME, noLoc, gradIndex, noLoc)
                .get();

        auto* assignExpr = BuildOp(BO_Assign, LHSExpr, RHSExpr);
        addToCurrentBlock(assignExpr, direction::forward);
      }
    } else {
      // Add Function call to block
      Expr* enzymeCall = BuildCallExprToFunction(enzymeCallFD, enzymeArgs);
      addToCurrentBlock(enzymeCall);
    }
  }

  StmtDiff ReverseModeVisitor::VisitCXXStdInitializerListExpr(
      const clang::CXXStdInitializerListExpr* ILE) {
    return Visit(ILE->getSubExpr(), dfdx());
  }

  StmtDiff ReverseModeVisitor::VisitStmt(const Stmt* S) {
    diag(
        DiagnosticsEngine::Warning, S->getBeginLoc(),
        "attempted to differentiate unsupported statement, no changes applied");
    // Unknown stmt, just clone it.
    return StmtDiff(Clone(S));
  }

  StmtDiff ReverseModeVisitor::VisitCXXFunctionalCastExpr(
      const clang::CXXFunctionalCastExpr* FCE) {
    StmtDiff castExprDiff = Visit(FCE->getSubExpr(), dfdx());
    castExprDiff.updateStmt(m_Sema
                                .BuildCXXFunctionalCastExpr(
                                    FCE->getTypeInfoAsWritten(), FCE->getType(),
                                    FCE->getBeginLoc(), castExprDiff.getExpr(),
                                    FCE->getEndLoc())
                                .get());
    return castExprDiff;
  }

  StmtDiff ReverseModeVisitor::VisitCompoundStmt(const CompoundStmt* CS) {
    int scopeFlags = Scope::DeclScope;
    // If this is the outermost compound statement of the function,
    // propagate the function scope.
    if (getCurrentScope() == m_DerivativeFnScope)
      scopeFlags |= Scope::FnScope;
    beginScope(scopeFlags);
    beginBlock(direction::forward);
    beginBlock(direction::reverse);
    for (Stmt* S : CS->body()) {
      if (m_ExternalSource)
        m_ExternalSource->ActBeforeDifferentiatingStmtInVisitCompoundStmt();
      StmtDiff SDiff = DifferentiateSingleStmt(S);
      addToCurrentBlock(SDiff.getStmt(), direction::forward);
      addToCurrentBlock(SDiff.getStmt_dx(), direction::reverse);

      if (m_ExternalSource)
        m_ExternalSource->ActAfterProcessingStmtInVisitCompoundStmt();
    }
    CompoundStmt* Forward = endBlock(direction::forward);
    CompoundStmt* Reverse = endBlock(direction::reverse);
    endScope();
    return StmtDiff(Forward, Reverse);
  }

  StmtDiff ReverseModeVisitor::VisitIfStmt(const clang::IfStmt* If) {
    // Control scope of the IfStmt. E.g., in if (double x = ...) {...}, x goes
    // to this scope.
    beginScope(Scope::DeclScope | Scope::ControlScope);

    // Create a block "around" if statement, e.g:
    // {
    //   ...
    //  if (...) {...}
    // }
    beginBlock(direction::forward);
    beginBlock(direction::reverse);
    StmtDiff condDiff;
    // if the statement has an init, we process it
    if (If->hasInitStorage()) {
      StmtDiff initDiff = Visit(If->getInit());
      addToCurrentBlock(initDiff.getStmt(), direction::forward);
      addToCurrentBlock(initDiff.getStmt_dx(), direction::reverse);
    }
    // this ensures we can differentiate conditions that affect the derivatives
    // as well as declarations inside the condition:
    beginBlock(direction::reverse);
    if (const auto* condDeclStmt = If->getConditionVariableDeclStmt())
      condDiff = Visit(condDeclStmt);
    else
      condDiff = Visit(If->getCond());
    CompoundStmt* RCS = endBlock(direction::reverse);
    if (!RCS->body_empty()) {
      std::reverse(
          RCS->body_begin(),
          RCS->body_end()); // it is reversed in the endBlock() but we don't
                            // actually need this, so we reverse it once again
      addToCurrentBlock(RCS, direction::reverse);
    }

    // Condition has to be stored as a "global" variable, to take the correct
    // branch in the reverse pass.
    Expr* condDiffStored =
        GlobalStoreAndRef(condDiff.getExpr(), m_Context.BoolTy, "_cond");
    // Convert cond to boolean condition.
    if (condDiffStored)
      condDiffStored =
          m_Sema
              .ActOnCondition(getCurrentScope(), noLoc, condDiffStored,
                              Sema::ConditionKind::Boolean)
              .get()
              .second;

    auto VisitBranch = [&](const Stmt* Branch) -> StmtDiff {
      if (!Branch)
        return {};
      if (isa<CompoundStmt>(Branch)) {
        StmtDiff BranchDiff = Visit(Branch);
        return BranchDiff;
      }
      beginBlock(direction::forward);
      if (m_ExternalSource)
        m_ExternalSource
            ->ActBeforeDifferentiatingSingleStmtBranchInVisitIfStmt();
      StmtDiff BranchDiff = DifferentiateSingleStmt(Branch, /*dfdS=*/nullptr);
      addToCurrentBlock(BranchDiff.getStmt(), direction::forward);

      if (m_ExternalSource)
        m_ExternalSource
            ->ActBeforeFinalizingVisitBranchSingleStmtInIfVisitStmt();

      Stmt* Forward = utils::unwrapIfSingleStmt(endBlock(direction::forward));
      Stmt* Reverse = utils::unwrapIfSingleStmt(BranchDiff.getStmt_dx());
      return StmtDiff(Forward, Reverse);
    };

    StmtDiff thenDiff = VisitBranch(If->getThen());
    StmtDiff elseDiff = VisitBranch(If->getElse());
    Stmt* Forward = clad_compat::IfStmt_Create(
        m_Context, noLoc, If->isConstexpr(), /*Init=*/nullptr, /*Var=*/nullptr,
        condDiffStored, noLoc, noLoc, thenDiff.getStmt(), noLoc,
        elseDiff.getStmt());
    addToCurrentBlock(Forward, direction::forward);

    Stmt* Reverse = nullptr;
    // thenDiff.getStmt_dx() might be empty if TBR is on leadinf to a crash in
    // case of the braceless if.
    if (thenDiff.getStmt_dx())
      Reverse = clad_compat::IfStmt_Create(
          m_Context, noLoc, If->isConstexpr(), /*Init=*/nullptr,
          /*Var=*/nullptr, condDiffStored, noLoc, noLoc, thenDiff.getStmt_dx(),
          noLoc, elseDiff.getStmt_dx());
    else if (elseDiff.getStmt_dx())
      Reverse = clad_compat::IfStmt_Create(
          m_Context, noLoc, If->isConstexpr(), /*Init=*/nullptr,
          /*Var=*/nullptr,
          BuildOp(clang::UnaryOperatorKind::UO_LNot,
                  BuildParens(condDiffStored)),
          noLoc, noLoc, elseDiff.getStmt_dx(), noLoc, {});
    addToCurrentBlock(Reverse, direction::reverse);
    CompoundStmt* ForwardBlock = endBlock(direction::forward);
    CompoundStmt* ReverseBlock = endBlock(direction::reverse);
    endScope();
    return StmtDiff(utils::unwrapIfSingleStmt(ForwardBlock),
                    utils::unwrapIfSingleStmt(ReverseBlock),
                    /*valueForRevSweep=*/condDiffStored);
  }

  StmtDiff ReverseModeVisitor::VisitConditionalOperator(
      const clang::ConditionalOperator* CO) {
    StmtDiff condDiff = Visit(CO->getCond());
    beginBlock(direction::reverse);
    addToCurrentBlock(condDiff.getStmt_dx(), direction::reverse);
    // Condition has to be stored as a "global" variable, to take the correct
    // branch in the reverse pass.
    Expr* condStored = GlobalStoreAndRef(condDiff.getExpr(), "_cond");
    // Convert cond to boolean condition.
    condStored = m_Sema
                     .ActOnCondition(getCurrentScope(), noLoc, condStored,
                                     Sema::ConditionKind::Boolean)
                     .get()
                     .second;

    auto* ifTrue = CO->getTrueExpr();
    auto* ifFalse = CO->getFalseExpr();

    auto VisitBranch = [&](const Expr* Branch,
                           Expr* dfdx) -> std::pair<StmtDiff, StmtDiff> {
      beginScope(Scope::DeclScope);
      auto Result = DifferentiateSingleExpr(Branch, dfdx);
      endScope();
      StmtDiff BranchDiff = Result.first;
      StmtDiff ExprDiff = Result.second;
      Stmt* Forward = utils::unwrapIfSingleStmt(BranchDiff.getStmt());
      Stmt* Reverse = utils::unwrapIfSingleStmt(BranchDiff.getStmt_dx());
      return {StmtDiff(Forward, Reverse), ExprDiff};
    };

    StmtDiff ifTrueDiff;
    StmtDiff ifTrueExprDiff;
    StmtDiff ifFalseDiff;
    StmtDiff ifFalseExprDiff;

    std::tie(ifTrueDiff, ifTrueExprDiff) = VisitBranch(ifTrue, dfdx());
    std::tie(ifFalseDiff, ifFalseExprDiff) = VisitBranch(ifFalse, dfdx());

    auto BuildIf = [&](Expr* Cond, Stmt* Then, Stmt* Else) -> Stmt* {
      if (!Then && !Else)
        return nullptr;
      if (!Then)
        Then = m_Sema.ActOnNullStmt(noLoc).get();
      return clad_compat::IfStmt_Create(m_Context,
                                        noLoc,
                                        false,
                                        nullptr,
                                        nullptr,
                                        Cond,
                                        noLoc,
                                        noLoc,
                                        Then,
                                        noLoc,
                                        Else);
    };

    Stmt* Forward =
        BuildIf(condStored, ifTrueDiff.getStmt(), ifFalseDiff.getStmt());
    Stmt* Reverse =
        BuildIf(condStored, ifTrueDiff.getStmt_dx(), ifFalseDiff.getStmt_dx());
    if (Forward)
      addToCurrentBlock(Forward, direction::forward);
    if (Reverse)
      addToCurrentBlock(Reverse, direction::reverse);

    Expr* condExpr = m_Sema
                         .ActOnConditionalOp(noLoc, noLoc, condStored,
                                             ifTrueExprDiff.getExpr(),
                                             ifFalseExprDiff.getExpr())
                         .get();
    // If result is a glvalue, we should keep it as it can potentially be
    // assigned as in (c ? a : b) = x;
    Expr* ResultRef = nullptr;
    if ((CO->isModifiableLvalue(m_Context) == Expr::MLV_Valid) &&
        ifTrueExprDiff.getExpr_dx() && ifFalseExprDiff.getExpr_dx()) {
      ResultRef = m_Sema
                      .ActOnConditionalOp(noLoc, noLoc, condStored,
                                          ifTrueExprDiff.getExpr_dx(),
                                          ifFalseExprDiff.getExpr_dx())
                      .get();
      if (ResultRef->isModifiableLvalue(m_Context) != Expr::MLV_Valid)
        ResultRef = nullptr;
    }
    Stmt* revBlock = utils::unwrapIfSingleStmt(endBlock(direction::reverse));
    addToCurrentBlock(revBlock, direction::reverse);
    return StmtDiff(condExpr, ResultRef);
  }

  StmtDiff
  ReverseModeVisitor::VisitCXXForRangeStmt(const CXXForRangeStmt* FRS) {
    const auto* RangeDecl = cast<VarDecl>(FRS->getRangeStmt()->getSingleDecl());
    const auto* BeginDecl = cast<VarDecl>(FRS->getBeginStmt()->getSingleDecl());
    DeclDiff<VarDecl> VisitRange =
        DifferentiateVarDecl(RangeDecl, /*keepLocal=*/true);
    DeclDiff<VarDecl> VisitBegin =
        DifferentiateVarDecl(BeginDecl, /*keepLocal=*/true);

    beginBlock(direction::reverse);
    LoopCounter loopCounter(*this);
    beginScope(Scope::DeclScope | Scope::ControlScope | Scope::BreakScope |
               Scope::ContinueScope);

    llvm::SaveAndRestore<Expr*> SaveCurrentBreakFlagExpr(
        m_CurrentBreakFlagExpr);
    m_CurrentBreakFlagExpr = nullptr;
    auto* activeBreakContHandler = PushBreakContStmtHandler();
    activeBreakContHandler->BeginCFSwitchStmtScope();
    const VarDecl* LoopVD = FRS->getLoopVariable();

    llvm::SaveAndRestore<bool> SaveIsInside(isInsideLoop,
                                            /*NewValue=*/false);

    beginBlock(direction::reverse);
    // Create all declarations needed.
    DeclRefExpr* beginDeclRef = BuildDeclRef(VisitBegin.getDecl());
    Expr* d_beginDeclRef = m_Variables[beginDeclRef->getDecl()];
    addToCurrentBlock(BuildDeclStmt(VisitRange.getDecl()));
    addToCurrentBlock(BuildDeclStmt(VisitRange.getDecl_dx()));
    addToCurrentBlock(BuildDeclStmt(VisitBegin.getDecl()));
    addToCurrentBlock(BuildDeclStmt(VisitBegin.getDecl_dx()));

    const auto* EndDecl = cast<VarDecl>(FRS->getEndStmt()->getSingleDecl());
    QualType endType = CloneType(EndDecl->getType());
    std::string endName = EndDecl->getNameAsString();
    Expr* endInit = Visit(EndDecl->getInit()).getExpr();
    VarDecl* endVarDecl =
        BuildGlobalVarDecl(endType, endName, endInit, /*DirectInit=*/false);
    addToCurrentBlock(BuildDeclStmt(endVarDecl));
    DeclRefExpr* endExpr = BuildDeclRef(endVarDecl);
    Expr* incBegin = BuildOp(UO_PreInc, beginDeclRef);

    beginBlock(direction::forward);
    DeclDiff<VarDecl> LoopVDDiff = DifferentiateVarDecl(LoopVD);
    Stmt* adjLoopVDAddAssign =
        utils::unwrapIfSingleStmt(endBlock(direction::forward));

    llvm::SaveAndRestore<bool> SaveIsInsideLoop(isInsideLoop,
                                                /*NewValue=*/true);

    Expr* d_incBegin = BuildOp(UO_PreInc, d_beginDeclRef);
    Expr* d_decBegin = BuildOp(UO_PostDec, d_beginDeclRef);
    Expr* forwardCond = BuildOp(BO_NE, beginDeclRef, endExpr);
    const Stmt* body = FRS->getBody();
    StmtDiff bodyDiff =
        DifferentiateLoopBody(body, loopCounter, nullptr, nullptr,
                              /*isForLoop=*/true);

    activeBreakContHandler->EndCFSwitchStmtScope();
    activeBreakContHandler->UpdateForwAndRevBlocks(bodyDiff);
    PopBreakContStmtHandler();

    StmtDiff storeLoop = StoreAndRestore(BuildDeclRef(LoopVDDiff.getDecl()));
    StmtDiff storeAdjLoop =
        StoreAndRestore(BuildDeclRef(LoopVDDiff.getDecl_dx()));
    addToCurrentBlock(BuildDeclStmt(LoopVDDiff.getDecl_dx()));

    Expr* loopInit = LoopVDDiff.getDecl()->getInit();
    SetDeclInit(LoopVDDiff.getDecl(),
                getZeroInit(LoopVDDiff.getDecl()->getType()));
    addToCurrentBlock(BuildDeclStmt(LoopVDDiff.getDecl()));
    Expr* assignLoop =
        BuildOp(BO_Assign, BuildDeclRef(LoopVDDiff.getDecl()), loopInit);

    if (!LoopVD->getType()->isReferenceType()) {
      Expr* d_LoopVD = BuildDeclRef(LoopVDDiff.getDecl_dx());
      adjLoopVDAddAssign =
          BuildOp(BO_Assign, d_LoopVD, BuildOp(UO_Deref, d_beginDeclRef));
    }

    beginBlock(direction::forward);
    addToCurrentBlock(adjLoopVDAddAssign);
    addToCurrentBlock(assignLoop);
    addToCurrentBlock(storeLoop.getStmt());
    addToCurrentBlock(storeAdjLoop.getStmt());
    CompoundStmt* LoopVDForwardDiff = endBlock(direction::forward);
    CompoundStmt* bodyForward = utils::PrependAndCreateCompoundStmt(
        m_Sema.getASTContext(), bodyDiff.getStmt(), LoopVDForwardDiff);

    beginBlock(direction::forward);
    addToCurrentBlock(d_decBegin);
    addToCurrentBlock(storeLoop.getStmt_dx());
    addToCurrentBlock(storeAdjLoop.getStmt_dx());
    CompoundStmt* LoopVDReverseDiff = endBlock(direction::forward);
    CompoundStmt* bodyReverse = utils::PrependAndCreateCompoundStmt(
        m_Sema.getASTContext(), bodyDiff.getStmt_dx(), LoopVDReverseDiff);

    Expr* inc = BuildOp(BO_Comma, incBegin, d_incBegin);
    Stmt* Forward = new (m_Context) ForStmt(
        m_Context, /*Init=*/nullptr, forwardCond, /*CondVar=*/nullptr, inc,
        bodyForward, FRS->getForLoc(), FRS->getBeginLoc(), FRS->getEndLoc());
    Expr* counterCondition =
        loopCounter.getCounterConditionResult().get().second;
    Expr* counterDecrement = loopCounter.getCounterDecrement();

    Stmt* Reverse = bodyReverse;
    addToCurrentBlock(Reverse, direction::reverse);
    Reverse = endBlock(direction::reverse);

    Reverse = new (m_Context)
        ForStmt(m_Context, /*Init=*/nullptr, counterCondition,
                /*CondVar=*/nullptr, counterDecrement, Reverse,
                FRS->getForLoc(), FRS->getBeginLoc(), FRS->getEndLoc());
    addToCurrentBlock(Reverse, direction::reverse);
    Reverse = endBlock(direction::reverse);
    endScope();

    return {utils::unwrapIfSingleStmt(Forward),
            utils::unwrapIfSingleStmt(Reverse)};
  }

  StmtDiff ReverseModeVisitor::VisitForStmt(const ForStmt* FS) {
    beginBlock(direction::reverse);
    LoopCounter loopCounter(*this);
    beginScope(Scope::DeclScope | Scope::ControlScope | Scope::BreakScope |
               Scope::ContinueScope);
    llvm::SaveAndRestore<Expr*> SaveCurrentBreakFlagExpr(
        m_CurrentBreakFlagExpr);
    m_CurrentBreakFlagExpr = nullptr;
    const Stmt* init = FS->getInit();
    if (m_ExternalSource)
      m_ExternalSource->ActBeforeDifferentiatingLoopInitStmt();
    StmtDiff initResult = init ? DifferentiateSingleStmt(init) : StmtDiff{};

    // Save the isInsideLoop value (we may be inside another loop).
    llvm::SaveAndRestore<bool> SaveIsInsideLoop(isInsideLoop);
    isInsideLoop = true;
    StmtDiff condVarRes;
    VarDecl* condVarClone = nullptr;
    if (FS->getConditionVariable()) {
      condVarRes = DifferentiateSingleStmt(FS->getConditionVariableDeclStmt());
      if (isa<DeclStmt>(condVarRes.getStmt())) {
        Decl* decl = cast<DeclStmt>(condVarRes.getStmt())->getSingleDecl();
        condVarClone = cast<VarDecl>(decl);
      }
    }

    // but it is not generally true, e.g. for (...; (x = y); ...)...
    StmtDiff condDiff;
    StmtDiff condExprDiff;
    if (FS->getCond())
      std::tie(condDiff, condExprDiff) = DifferentiateSingleExpr(FS->getCond());

    const auto* IDRE = dyn_cast<DeclRefExpr>(FS->getInc());
    const Expr* inc = IDRE ? Visit(FS->getInc()).getExpr() : FS->getInc();

    // Differentiate the increment expression of the for loop
    // incExprDiff.getExpr() is the reconstructed expression, incDiff.getStmt()
    // a block with all the intermediate statements used to reconstruct it on
    // the forward pass, incDiff.getStmt_dx() is the reverse pass block.
    StmtDiff incDiff;
    StmtDiff incExprDiff;
    if (inc)
      std::tie(incDiff, incExprDiff) = DifferentiateSingleExpr(inc);
    Expr* incResult = nullptr;
    // If any additional statements were created, enclose them into lambda.
    auto* Additional = cast<CompoundStmt>(incDiff.getStmt());
    bool anyNonExpr = std::any_of(Additional->body_begin(),
                                  Additional->body_end(),
                                  [](Stmt* S) { return !isa<Expr>(S); });
    if (anyNonExpr) {
      incResult = wrapInLambda(*this, m_Sema, inc, [&] {
        std::tie(incDiff, incExprDiff) = DifferentiateSingleExpr(inc);
        for (Stmt* S : cast<CompoundStmt>(incDiff.getStmt())->body())
          addToCurrentBlock(S);
        addToCurrentBlock(incDiff.getExpr());
      });
    }
    // Otherwise, join all exprs by comma operator.
    else if (incExprDiff.getExpr()) {
      auto CommaJoin = [this](Expr* Acc, Stmt* S) {
        Expr* E = cast<Expr>(S);
        return BuildOp(BO_Comma, E, BuildParens(Acc));
      };
      incResult = std::accumulate(Additional->body_rbegin(),
                                  Additional->body_rend(),
                                  incExprDiff.getExpr(),
                                  CommaJoin);
    }

    const Stmt* body = FS->getBody();
    StmtDiff BodyDiff = DifferentiateLoopBody(body, loopCounter,
                                              condVarRes.getStmt_dx(),
                                              incDiff.getStmt_dx(),
                                              /*isForLoop=*/true);

    /// FIXME: This part in necessary to replace local variables inside loops
    /// with function globals and replace initializations with assignments.
    /// This is a temporary measure to avoid the bug that arises from
    /// overwriting local variables on different loop passes.
    Expr* forwardCond = condExprDiff.getExpr();
    /// If there is a declaration in the condition, `cond` will be
    /// a DeclRefExpr of the declared variable. There is no point in
    /// inserting it since condVarRes.getExpr() represents an assignment with
    /// that variable on the LHS.
    /// e.g. for condition `int x = y`,
    /// condVarRes.getExpr() will represent `x = y`
    if (condVarRes.getExpr() != nullptr && isa<Expr>(condVarRes.getExpr()))
      forwardCond = cast<Expr>(condVarRes.getExpr());

    Stmt* breakStmt = m_Sema.ActOnBreakStmt(noLoc, getCurrentScope()).get();

    /// This part adds the forward pass of loop condition stmt in the body
    /// In this first loop condition diff stmts execute then loop condition
    /// is checked if and loop is terminated.
    beginBlock();
    if (utils::unwrapIfSingleStmt(condDiff.getStmt()))
      addToCurrentBlock(condDiff.getStmt());

    Stmt* IfStmt = clad_compat::IfStmt_Create(
        /*Ctx=*/m_Context, /*IL=*/noLoc, /*IsConstexpr=*/false,
        /*Init=*/nullptr, /*Var=*/nullptr,
        /*Cond=*/
        BuildOp(clang::UnaryOperatorKind::UO_LNot, BuildParens(forwardCond)),
        /*LPL=*/noLoc, /*RPL=*/noLoc,
        /*Then=*/breakStmt,
        /*EL=*/noLoc,
        /*Else=*/nullptr);
    addToCurrentBlock(IfStmt);

    Stmt* forwardCondStmts = endBlock();
    if (BodyDiff.getStmt()) {
      BodyDiff.updateStmt(utils::PrependAndCreateCompoundStmt(
          m_Context, BodyDiff.getStmt(), forwardCondStmts));
    } else {
      BodyDiff.updateStmt(utils::unwrapIfSingleStmt(forwardCondStmts));
    }

    Stmt* Forward = new (m_Context)
        ForStmt(m_Context, initResult.getStmt(), nullptr, condVarClone,
                incResult, BodyDiff.getStmt(), noLoc, noLoc, noLoc);

    // Create a condition testing counter for being zero, and its decrement.
    // To match the number of iterations in the forward pass, the reverse loop
    // will look like: for(; Counter; Counter--) ...
    Expr*
        CounterCondition = loopCounter.getCounterConditionResult().get().second;
    Expr* CounterDecrement = loopCounter.getCounterDecrement();

    /// This part adds the reverse pass of loop condition stmt in the body
    beginBlock(direction::reverse);
    Stmt* RevIfStmt = clad_compat::IfStmt_Create(
        /*Ctx=*/m_Context, /*IL=*/noLoc, /*IsConstexpr=*/false,
        /*Init=*/nullptr, /*Var=*/nullptr,
        /*Cond=*/BuildOp(clang::UnaryOperatorKind::UO_LNot, CounterCondition),
        /*LPL=*/noLoc, /*RPL=*/noLoc,
        /*Then=*/Clone(breakStmt),
        /*EL=*/noLoc,
        /*Else=*/nullptr);
    addToCurrentBlock(RevIfStmt, direction::reverse);

    if (condDiff.getStmt_dx()) {
      if (m_CurrentBreakFlagExpr) {
        Expr* loopBreakFlagCond =
            BuildOp(BinaryOperatorKind::BO_LOr,
                    BuildOp(UnaryOperatorKind::UO_LNot, CounterCondition),
                    BuildParens(m_CurrentBreakFlagExpr));
        auto* RevIfStmt = clad_compat::IfStmt_Create(
            m_Context, noLoc, false, nullptr, nullptr, loopBreakFlagCond, noLoc,
            noLoc, condDiff.getStmt_dx(), noLoc, nullptr);
        addToCurrentBlock(RevIfStmt, direction::reverse);
      } else {
        addToCurrentBlock(condDiff.getStmt_dx(), direction::reverse);
      }
    }

    Stmt* revPassCondStmts = endBlock(direction::reverse);
    if (BodyDiff.getStmt_dx()) {
      BodyDiff.updateStmtDx(utils::PrependAndCreateCompoundStmt(
          m_Context, BodyDiff.getStmt_dx(), revPassCondStmts));
    } else {
      BodyDiff.updateStmtDx(utils::unwrapIfSingleStmt(revPassCondStmts));
    }

    Stmt* revInit = loopCounter.getNumRevIterations()
                        ? BuildDeclStmt(loopCounter.getNumRevIterations())
                        : nullptr;
    Stmt* Reverse = new (m_Context)
        ForStmt(m_Context, revInit, nullptr, nullptr, CounterDecrement,
                BodyDiff.getStmt_dx(), noLoc, noLoc, noLoc);

    addToCurrentBlock(initResult.getStmt_dx(), direction::reverse);
    addToCurrentBlock(Reverse, direction::reverse);
    Reverse = endBlock(direction::reverse);
    endScope();

    return {utils::unwrapIfSingleStmt(Forward),
            utils::unwrapIfSingleStmt(Reverse)};
  }

  StmtDiff
  ReverseModeVisitor::VisitCXXDefaultArgExpr(const CXXDefaultArgExpr* DE) {
    return Visit(DE->getExpr(), dfdx());
  }

  StmtDiff
  ReverseModeVisitor::VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr* BL) {
    return Clone(BL);
  }

  StmtDiff
  ReverseModeVisitor::VisitCharacterLiteral(const CharacterLiteral* CL) {
    return Clone(CL);
  }

  StmtDiff ReverseModeVisitor::VisitStringLiteral(const StringLiteral* SL) {
    return StmtDiff(
        Clone(SL),
        StringLiteral::Create(m_Context, "", SL->getKind(), SL->isPascal(),
                              utils::getNonConstType(SL->getType(), m_Sema),
                              utils::GetValidSLoc(m_Sema)));
  }

  StmtDiff ReverseModeVisitor::VisitCXXNullPtrLiteralExpr(
      const CXXNullPtrLiteralExpr* NPE) {
    return StmtDiff(Clone(NPE), Clone(NPE));
  }

  StmtDiff ReverseModeVisitor::VisitReturnStmt(const ReturnStmt* RS) {
    // Initially, df/df = 1.
    if (m_DiffReq->getReturnType()->isVoidType())
      return {nullptr, nullptr};
    const Expr* value = RS->getRetValue();
    QualType type = value->getType();
    auto* dfdf = m_Pullback;
    if (dfdf && (isa<FloatingLiteral>(dfdf) || isa<IntegerLiteral>(dfdf)) &&
        type->isScalarType()) {
      ExprResult tmp = dfdf;
      dfdf = m_Sema
                 .ImpCastExprToType(tmp.get(), type,
                                    m_Sema.PrepareScalarCast(tmp, type))
                 .get();
    }
    auto ReturnResult = DifferentiateSingleExpr(value, dfdf);
    StmtDiff ReturnDiff = ReturnResult.first;
    StmtDiff ExprDiff = ReturnResult.second;
    Stmt* Reverse = ReturnDiff.getStmt_dx();
    for (Stmt* S : cast<CompoundStmt>(ReturnDiff.getStmt())->body())
      addToCurrentBlock(S, direction::forward);

    // FIXME: When the return type of a function is a class, ExprDiff.getExpr()
    // returns nullptr, which is a bug. For the time being, the only use case of
    // a return type being class is in pushforwards. Hence a special case has
    // been made to to not do the StoreAndRef operation when return type is
    // ValueAndPushforward.
    if (!utils::IsCladValueAndPushforwardType(type)) {
      if (m_ExternalSource)
        m_ExternalSource->ActBeforeFinalizingVisitReturnStmt(ExprDiff);
    }

    // If this return stmt is the last stmt in the function's body,
    // adding goto will only introduce
    // ```
    // goto _label0; // the forward sweep ends
    // _label0:  // the reverse sweep starts immediately
    // ```
    // Therefore, in this case, we can omit the goto.
    const Stmt* lastFuncStmt = m_DiffReq.Function->getBody();
    if (const auto* CS = dyn_cast<CompoundStmt>(lastFuncStmt))
      lastFuncStmt = *CS->body_rbegin();
    if (RS == lastFuncStmt)
      return {nullptr, Reverse};

    // If the original function returns at this point, some part of the reverse
    // pass (corresponding to other branches that do not return here) must be
    // skipped. We create a label in the reverse pass and jump to it via goto.
    LabelDecl* LD = LabelDecl::Create(m_Context, m_Sema.CurContext, noLoc,
                                      CreateUniqueIdentifier("_label"));
    m_Sema.PushOnScopeChains(LD, m_DerivativeFnScope, true);
    // Attach label to the last Stmt in the corresponding Reverse Stmt.
    if (!Reverse)
      Reverse = m_Sema.ActOnNullStmt(noLoc).get();
    Stmt* LS = m_Sema.ActOnLabelStmt(noLoc, LD, noLoc, Reverse).get();
    addToCurrentBlock(LS, direction::reverse);

    // Create goto to the label.
    return m_Sema.ActOnGotoStmt(noLoc, noLoc, LD).get();
  }

  StmtDiff ReverseModeVisitor::VisitParenExpr(const ParenExpr* PE) {
    StmtDiff subStmtDiff = Visit(PE->getSubExpr(), dfdx());
    return StmtDiff(BuildParens(subStmtDiff.getExpr()),
                    BuildParens(subStmtDiff.getExpr_dx()),
                    BuildParens(subStmtDiff.getRevSweepAsExpr()));
  }

  StmtDiff ReverseModeVisitor::VisitInitListExpr(const InitListExpr* ILE) {
    QualType ILEType = ILE->getType();
    llvm::SmallVector<Expr*, 16> clonedExprs(ILE->getNumInits());
    // Handle basic types like pointers or numericals.
    if (ILE->getNumInits() &&
        !(ILEType->isArrayType() || ILEType->isRecordType())) {
      StmtDiff initDiff = Visit(ILE->getInit(0), dfdx());
      clonedExprs[0] = initDiff.getExpr();
      initDiff.updateStmt(
          m_Sema.ActOnInitList(noLoc, clonedExprs, noLoc).get());
      return initDiff;
    }
    if (!dfdx())
      return StmtDiff(Clone(ILE));
    if (ILEType->isArrayType()) {
      for (unsigned i = 0, e = ILE->getNumInits(); i < e; i++) {
        Expr* I =
            ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, i);
        Expr* array_at_i = m_Sema
                               .ActOnArraySubscriptExpr(getCurrentScope(),
                                                        dfdx(), noLoc, I, noLoc)
                               .get();
        Expr* clonedEI = Visit(ILE->getInit(i), array_at_i).getExpr();
        clonedExprs[i] = clonedEI;
      }

      Expr* clonedILE = m_Sema.ActOnInitList(noLoc, clonedExprs, noLoc).get();
      return StmtDiff(clonedILE);
    }
    // Check if type is a CXXRecordDecl
    if (ILEType->isRecordType()) {
      for (unsigned i = 0, e = ILE->getNumInits(); i < e; i++) {
        // fetch ith field of the struct.
        auto field_iterator = ILEType->getAsCXXRecordDecl()->field_begin();
        std::advance(field_iterator, i);
        Expr* member_acess = utils::BuildMemberExpr(
            m_Sema, getCurrentScope(), dfdx(), (*field_iterator)->getName());
        clonedExprs[i] = Visit(ILE->getInit(i), member_acess).getExpr();
      }
      Expr* clonedILE = m_Sema.ActOnInitList(noLoc, clonedExprs, noLoc).get();
      return StmtDiff(clonedILE);
    }

    Expr* clonedILE = m_Sema.ActOnInitList(noLoc, {}, noLoc).get();
    return StmtDiff(clonedILE);
  }

  StmtDiff
  ReverseModeVisitor::VisitArraySubscriptExpr(const ArraySubscriptExpr* ASE) {
    auto ASI = SplitArraySubscript(ASE);
    const Expr* Base = ASI.first;
    const auto& Indices = ASI.second;
    StmtDiff BaseDiff = Visit(Base);
    llvm::SmallVector<Expr*, 4> clonedIndices(Indices.size());
    llvm::SmallVector<Expr*, 4> reverseIndices(Indices.size());
    for (std::size_t i = 0; i < Indices.size(); i++) {
      // FIXME: Remove redundant indices vectors.
      StmtDiff IdxDiff = Visit(Indices[i]);
      clonedIndices[i] = Clone(IdxDiff.getExpr());
      reverseIndices[i] = IdxDiff.getExpr();
    }
    auto* cloned = BuildArraySubscript(BaseDiff.getExpr(), clonedIndices);
    auto* valueForRevSweep =
        BuildArraySubscript(BaseDiff.getExpr(), reverseIndices);
    Expr* target = BaseDiff.getExpr_dx();
    if (!target)
      return cloned;
    Expr* result = nullptr;
    // Create the target[idx] expression.
    result = BuildArraySubscript(target, reverseIndices);
    // Create the (target += dfdx) statement.
    if (dfdx()) {
      Expr* add_assign = nullptr;
      if (shouldUseCudaAtomicOps(target))
        add_assign = BuildCallToCudaAtomicAdd(result, dfdx());
      else
        add_assign = BuildOp(BO_AddAssign, result, dfdx());

      addToCurrentBlock(add_assign, direction::reverse);
    }
    if (m_ExternalSource)
      m_ExternalSource->ActAfterProcessingArraySubscriptExpr(valueForRevSweep);
    return StmtDiff(cloned, result, valueForRevSweep);
  }

  StmtDiff ReverseModeVisitor::VisitDeclRefExpr(const DeclRefExpr* DRE) {
    Expr* clonedDRE = Clone(DRE);
    // Check if referenced Decl was "replaced" with another identifier inside
    // the derivative
    if (auto* VD = dyn_cast<VarDecl>(cast<DeclRefExpr>(clonedDRE)->getDecl())) {
      // If current context is different than the context of the original
      // declaration (e.g. we are inside lambda), rebuild the DeclRefExpr
      // with Sema::BuildDeclRefExpr. This is required in some cases, e.g.
      // Sema::BuildDeclRefExpr is responsible for adding captured fields
      // to the underlying struct of a lambda.
      if (VD->getDeclContext() != m_Sema.CurContext) {
        auto* ccDRE = dyn_cast<DeclRefExpr>(clonedDRE);
        NestedNameSpecifier* NNS = DRE->getQualifier();
        auto* referencedDecl = cast<VarDecl>(ccDRE->getDecl());
        clonedDRE = BuildDeclRef(referencedDecl, NNS, DRE->getValueKind());
      }
      // This case happens when ref-type variables have to become function
      // global. Ref-type declarations cannot be moved to the function global
      // scope because they can't be separated from their inits.
      if (DRE->getDecl()->getType()->isReferenceType() &&
          VD->getType()->isPointerType())
        clonedDRE = BuildOp(UO_Deref, clonedDRE);
      // Check DeclRefExpr is a reference to an independent variable.
      auto it = m_Variables.find(VD);
      if (it == std::end(m_Variables)) {
        if (VD->isFileVarDecl() && !VD->getType().isConstQualified()) {
          // VD is a global variable, attempt to find its adjoint.
          std::string nameDiff_str = "_d_" + VD->getNameAsString();
          DeclarationName nameDiff = &m_Context.Idents.get(nameDiff_str);
          DeclContext* DC = VD->getDeclContext();
          LookupResult result(m_Sema, nameDiff, noLoc,
                              Sema::LookupOrdinaryName);
          m_Sema.LookupQualifiedName(result, DC);
          // If not found, consider non-differentiable.
          if (result.empty())
            return StmtDiff(clonedDRE);
          // Found, return a reference
          Expr* foundExpr =
              m_Sema
                  .BuildDeclarationNameExpr(CXXScopeSpec{}, result,
                                            /*ADL=*/false)
                  .get();
          it = m_Variables.emplace(VD, foundExpr).first;
          // On the start of computing every derivative, we have to reset the
          // global adjoint to zero in case it was used by another gradient.
          if (m_DiffReq.Mode == DiffMode::reverse) {
            Expr* assignToZero = BuildOp(BO_Assign, Clone(foundExpr),
                                         getZeroInit(foundExpr->getType()));
            addToBlock(assignToZero, m_Globals);
          }
        } else
          // Is not an independent variable, ignored.
          return StmtDiff(clonedDRE);
      }
      // Create the (_d_param[idx] += dfdx) statement.
      QualType diffTy = it->second->getType();
      diffTy = diffTy.getNonReferenceType();
      if (dfdx() && diffTy->isRealType()) {
        Expr* base = it->second;
        if (auto* UO = dyn_cast<UnaryOperator>(it->second))
          base = UO->getSubExpr()->IgnoreImpCasts();
        Expr* add_assign = nullptr;
        if (shouldUseCudaAtomicOps(base))
          add_assign = BuildCallToCudaAtomicAdd(it->second, dfdx());
        else
          add_assign = BuildOp(BO_AddAssign, it->second, dfdx());

        addToCurrentBlock(add_assign, direction::reverse);
      }
      return StmtDiff(clonedDRE, it->second);
    }

    return StmtDiff(clonedDRE);
  }

  StmtDiff ReverseModeVisitor::VisitIntegerLiteral(const IntegerLiteral* IL) {
    auto* Constant0 =
        ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
    return StmtDiff(Clone(IL), Constant0);
  }

  StmtDiff ReverseModeVisitor::VisitFloatingLiteral(const FloatingLiteral* FL) {
    return StmtDiff(Clone(FL));
  }

  StmtDiff ReverseModeVisitor::VisitCallExpr(const CallExpr* CE) {
    const FunctionDecl* FD = CE->getDirectCallee();
    if (!FD) {
      diag(DiagnosticsEngine::Warning,
           CE->getEndLoc(),
           "Differentiation of only direct calls is supported. Ignored");
      return StmtDiff(Clone(CE));
    }

    // FIXME: Revisit this when variadic functions are supported.
    if (FD->getNameAsString() == "printf" || FD->getNameAsString() == "fprintf")
      return StmtDiff(Clone(CE));

    Expr* CUDAExecConfig = nullptr;
    if (const auto* KCE = dyn_cast<CUDAKernelCallExpr>(CE))
      CUDAExecConfig = Clone(KCE->getConfig());

    // begin and end are common enough to have a more efficient and nice-looking
    // special case. Instead of _forw and a useless _pullback functions, we can
    // express the result in terms of the same std::begin / std::end. Note:
    // since std::initializer_list is replaced with clad::array, this is the
    // simplest way to support begin/end functions of the former and not deal
    // with the type mismatch.
    std::string FDName = FD->getNameAsString();
    if (FDName == "begin" || FDName == "end") {
      const Expr* arg = nullptr;
      if (const auto* MCE = dyn_cast<CXXMemberCallExpr>(CE))
        arg = MCE->getImplicitObjectArgument();
      else
        arg = CE->getArg(0);
      if (const auto* CXXCE = dyn_cast<CXXConstructExpr>(arg))
        arg = CXXCE->getArg(0);
      StmtDiff argDiff = Visit(arg);
      llvm::SmallVector<Expr*, 1> params{argDiff.getExpr()};
      llvm::SmallVector<Expr*, 1> paramsDiff{argDiff.getExpr_dx()};
      Expr* call = GetFunctionCall(FDName, "std", params);
      Expr* callDiff = GetFunctionCall(FDName, "std", paramsDiff);
      return {call, callDiff};
    }

    auto NArgs = FD->getNumParams();
    SourceLocation Loc = CE->getExprLoc();

    // Stores the call arguments for the function to be derived
    llvm::SmallVector<Expr*, 16> CallArgs{};
    // Stores the dx of the call arguments for the function to be derived
    llvm::SmallVector<Expr*, 16> CallArgDx{};
    // Stores the call arguments for the derived function
    llvm::SmallVector<Expr*, 16> DerivedCallArgs{};
    // Stores tape decl and pushes for multiarg numerically differentiated
    // calls.
    llvm::SmallVector<Stmt*, 16> PostCallStmts{};

    // For calls to C-style memory allocation functions, we do not need to
    // differentiate the call. We just need to visit the arguments to the
    // function.
    if (utils::IsMemoryFunction(FD)) {
      for (const Expr* Arg : CE->arguments()) {
        StmtDiff ArgDiff = Visit(Arg, dfdx());
        CallArgs.push_back(ArgDiff.getExpr());
        if (Arg->getType()->isPointerType())
          DerivedCallArgs.push_back(ArgDiff.getExpr_dx());
        else
          DerivedCallArgs.push_back(ArgDiff.getExpr());
      }
      Expr* call =
          m_Sema
              .ActOnCallExpr(getCurrentScope(), Clone(CE->getCallee()), Loc,
                             llvm::MutableArrayRef<Expr*>(CallArgs), Loc)
              .get();
      Expr* call_dx =
          m_Sema
              .ActOnCallExpr(getCurrentScope(), Clone(CE->getCallee()), Loc,
                             llvm::MutableArrayRef<Expr*>(DerivedCallArgs), Loc)
              .get();
      if (FD->getNameAsString() == "cudaMalloc") {
        if (auto* addrOp = dyn_cast<UnaryOperator>(DerivedCallArgs[0]))
          if (addrOp->getOpcode() == UO_AddrOf)
            DerivedCallArgs[0] = addrOp->getSubExpr(); // get the pointer

        llvm::SmallVector<Expr*, 3> args = {DerivedCallArgs[0],
                                            getZeroInit(m_Context.IntTy),
                                            DerivedCallArgs[1]};
        addToCurrentBlock(call_dx, direction::forward);
        addToCurrentBlock(GetFunctionCall("cudaMemset", "", args));
        call_dx = nullptr;
      }
      return StmtDiff(call, call_dx);
    }
    // For calls to C-style memory deallocation functions, we do not need to
    // differentiate the call. We just need to visit the arguments to the
    // function. Also, don't add any statements either in forward or reverse
    // pass. Instead, add it in m_DeallocExprs.
    if (utils::IsMemoryDeallocationFunction(FD)) {
      for (const Expr* Arg : CE->arguments()) {
        StmtDiff ArgDiff = Visit(Arg, dfdx());
        CallArgs.push_back(ArgDiff.getExpr());
        if (const auto* DRE = dyn_cast<DeclRefExpr>(Arg->IgnoreImpCasts())) {
          // If the arg is used as independent variable, then we cannot free it
          // as it holds the result to be returned to the user.
          if (llvm::find(m_DiffReq.DVI, DRE->getDecl()) == m_DiffReq.DVI.end())
            DerivedCallArgs.push_back(ArgDiff.getExpr_dx());
        }
      }
      Expr* call =
          m_Sema
              .ActOnCallExpr(getCurrentScope(), Clone(CE->getCallee()), Loc,
                             llvm::MutableArrayRef<Expr*>(CallArgs), Loc)
              .get();
      m_DeallocExprs.push_back(call);

      if (!DerivedCallArgs.empty()) {
        Expr* call_dx =
            m_Sema
                .ActOnCallExpr(getCurrentScope(), Clone(CE->getCallee()), Loc,
                               llvm::MutableArrayRef<Expr*>(DerivedCallArgs),
                               Loc)
                .get();
        m_DeallocExprs.push_back(call_dx);
      }
      return StmtDiff();
    }

    // Determine the base of the call if any.
    const auto* MD = dyn_cast<CXXMethodDecl>(FD);
    const Expr* baseOriginalE = nullptr;
    if (MD && MD->isInstance()) {
      if (const auto* MCE = dyn_cast<CXXMemberCallExpr>(CE))
        baseOriginalE = MCE->getImplicitObjectArgument();
      else if (const auto* OCE = dyn_cast<CXXOperatorCallExpr>(CE))
        baseOriginalE = OCE->getArg(0);
    }

    // FIXME: Add support for lambdas used directly, e.g.
    // [](){return 12.;}()
    if (MD && isLambdaCallOperator(MD) &&
        !isa<DeclRefExpr>(baseOriginalE->IgnoreImplicit())) {
      diag(DiagnosticsEngine::Warning, baseOriginalE->getBeginLoc(),
           "Direct lambda calls are not supported, ignored.");
      return getZeroInit(CE->getType());
    }

    // FIXME: consider moving non-diff analysis to DiffPlanner.
    bool nonDiff = clad::utils::hasNonDifferentiableAttribute(CE);

    // If the result does not depend on the result of the call, just clone
    // the call and visit arguments (since they may contain side-effects like
    // f(x = y))
    // If the callee function takes arguments by reference then it can affect
    // derivatives even if there is no `dfdx()` and thus we should call the
    // derived function. In the case of member functions, `implicit`
    // this object is always passed by reference.
    if (!nonDiff && !dfdx() && !utils::HasAnyReferenceOrPointerArgument(FD) &&
        (!baseOriginalE || MD->isConst())) {
      // The result of the subscript operator may affect the derivative, such as
      // in a case like `list[i].modify(x)`. This makes clad handle those
      // normally.
      if (const auto* OCE = dyn_cast<CXXOperatorCallExpr>(CE)) {
        if (OCE->getOperator() != clang::OverloadedOperatorKind::OO_Subscript)
          nonDiff = true;
      } else
        nonDiff = true;
    }

    // If all arguments are constant literals, then this does not contribute to
    // the gradient.
    if (!nonDiff && !isa<CXXMemberCallExpr>(CE) &&
        !isa<CXXOperatorCallExpr>(CE)) {
      nonDiff = true;
      for (const Expr* arg : CE->arguments()) {
        if (m_DiffReq.isVaried(arg)) {
          nonDiff = false;
          break;
        }
      }
    }

    QualType returnType = FD->getReturnType();
    // FIXME: Decide this in the diff planner
    bool needsForwPass = utils::isNonConstReferenceType(returnType) ||
                         returnType->isPointerType();

    // FIXME: if the call is non-differentiable but needs a reverse forward
    // call, we still don't need to generate the pullback. The only challenge is
    // to refactor the code to be able to jump over the pullback part (maybe
    // move some functionality to subroutines).
    if (nonDiff && !needsForwPass) {
      for (const Expr* Arg : CE->arguments()) {
        StmtDiff ArgDiff = Visit(Arg);
        CallArgs.push_back(ArgDiff.getExpr());
      }
      Expr* call =
          m_Sema
              .ActOnCallExpr(getCurrentScope(), Clone(CE->getCallee()), Loc,
                             llvm::MutableArrayRef<Expr*>(CallArgs), Loc,
                             CUDAExecConfig)
              .get();
      return call;
    }

    llvm::SmallVector<Stmt*, 16> PreCallStmts{};
    // Save current index in the current block, to potentially put some
    // statements there later.
    std::size_t insertionPoint = getCurrentBlock(direction::reverse).size();

    // Method operators have a base like methods do but it's included in the
    // call arguments so we have to shift the indexing of call arguments.
    bool isMethodOperatorCall = MD && isa<CXXOperatorCallExpr>(CE);

    for (std::size_t i = static_cast<std::size_t>(isMethodOperatorCall),
                     e = CE->getNumArgs();
         i != e; ++i) {
      const Expr* arg = CE->getArg(i);
      const auto* PVD = FD->getParamDecl(
          i - static_cast<unsigned long>(isMethodOperatorCall));
      StmtDiff argDiff{};
      // We do not need to create result arg for arguments passed by reference
      // because the derivatives of arguments passed by reference are directly
      // modified by the derived callee function.
      if (utils::IsReferenceOrPointerArg(arg)) {
        argDiff = Visit(arg);
        CallArgDx.push_back(argDiff.getExpr_dx());
      } else if (!clad::utils::hasNonDifferentiableAttribute(arg)) {
        // Create temporary variables corresponding to derivative of each
        // argument, so that they can be referred to when arguments is visited.
        // Variables will be initialized later after arguments is visited. This
        // is done to reduce cloning complexity and only clone once. The type is
        // same as the call expression as it is the type used to declare the
        // _gradX array
        QualType dArgTy = utils::getNonConstType(arg->getType(), m_Sema);
        bool shouldCopyInitialize = false;
        if (const CXXRecordDecl* CRD = dArgTy->getAsCXXRecordDecl())
          shouldCopyInitialize = utils::isCopyable(CRD);
        Expr* rInit = getZeroInit(dArgTy);
        // Temporarily initialize the object with `*nullptr` to avoid
        // a potential error because of non-existing default constructor.
        if (shouldCopyInitialize) {
          QualType ptrType =
              m_Context.getPointerType(dArgTy.getUnqualifiedType());
          Expr* dummy = getZeroInit(ptrType);
          rInit = BuildOp(UO_Deref, dummy);
        }
        VarDecl* dArgDecl = BuildVarDecl(dArgTy, "_r", rInit);
        PreCallStmts.push_back(BuildDeclStmt(dArgDecl));
        DeclRefExpr* dArgRef = BuildDeclRef(dArgDecl);
        if (isa<CUDAKernelCallExpr>(CE)) {
          // Create variables to be allocated and initialized on the device, and
          // then be passed to the kernel pullback.
          //
          // These need to be pointers because cudaMalloc expects a
          // pointer-to-pointer as an arg.
          // The memory addresses they point to are initialized to zero through
          // cudaMemset.
          // After the pullback call, their values will be copied back to the
          // corresponding _r variables on the host and the device variables
          // will be freed.
          //
          // Example of the generated code:
          //
          // double _r0 = 0;
          // double* _r1 = nullptr;
          // cudaMalloc(&_r1, sizeof(double));
          // cudaMemset(_r1, 0, 8);
          // kernel_pullback<<<...>>>(..., _r1);
          // cudaMemcpy(&_r0, _r1, 8, cudaMemcpyDeviceToHost);
          // cudaFree(_r1);

          // Create a literal for the size of the type
          Expr* sizeLiteral = ConstantFolder::synthesizeLiteral(
              m_Context.IntTy, m_Context, m_Context.getTypeSize(dArgTy) / 8);
          dArgTy = m_Context.getPointerType(dArgTy);
          VarDecl* dArgDeclCUDA =
              BuildVarDecl(dArgTy, "_r", getZeroInit(dArgTy));

          // Create the cudaMemcpyDeviceToHost argument
          LookupResult deviceToHostResult =
              utils::LookupQualifiedName("cudaMemcpyDeviceToHost", m_Sema);
          if (deviceToHostResult.empty()) {
            diag(DiagnosticsEngine::Error, CE->getEndLoc(),
                 "Failed to create cudaMemcpy call; cudaMemcpyDeviceToHost not "
                 "found. Creating kernel pullback aborted.");
            return StmtDiff(Clone(CE));
          }
          CXXScopeSpec SS;
          Expr* deviceToHostExpr =
              m_Sema
                  .BuildDeclarationNameExpr(SS, deviceToHostResult,
                                            /*ADL=*/false)
                  .get();

          // Add calls to cudaMalloc, cudaMemset, cudaMemcpy, and cudaFree
          PreCallStmts.push_back(BuildDeclStmt(dArgDeclCUDA));
          Expr* refOp = BuildOp(UO_AddrOf, BuildDeclRef(dArgDeclCUDA));
          llvm::SmallVector<Expr*, 3> mallocArgs = {refOp, sizeLiteral};
          PreCallStmts.push_back(GetFunctionCall("cudaMalloc", "", mallocArgs));
          llvm::SmallVector<Expr*, 3> memsetArgs = {
              BuildDeclRef(dArgDeclCUDA), getZeroInit(m_Context.IntTy),
              sizeLiteral};
          PreCallStmts.push_back(GetFunctionCall("cudaMemset", "", memsetArgs));
          llvm::SmallVector<Expr*, 4> cudaMemcpyArgs = {
              BuildOp(UO_AddrOf, dArgRef), BuildDeclRef(dArgDeclCUDA),
              sizeLiteral, deviceToHostExpr};
          PostCallStmts.push_back(
              GetFunctionCall("cudaMemcpy", "", cudaMemcpyArgs));
          llvm::SmallVector<Expr*, 3> freeArgs = {BuildDeclRef(dArgDeclCUDA)};
          PostCallStmts.push_back(GetFunctionCall("cudaFree", "", freeArgs));

          // Update arg to be passed to pullback call
          dArgRef = BuildDeclRef(dArgDeclCUDA);
        }
        CallArgDx.push_back(dArgRef);
        // Visit using uninitialized reference.
        argDiff = Visit(arg, BuildDeclRef(dArgDecl));
        if (shouldCopyInitialize) {
          if (Expr* dInit = argDiff.getExpr_dx())
            SetDeclInit(dArgDecl, dInit);
          else
            SetDeclInit(dArgDecl, getZeroInit(dArgTy));
        }
      } else {
        CallArgDx.push_back(nullptr);
        argDiff = Visit(arg);
      }

      // Save cloned arg in a "global" variable, so that it is accessible from
      // the reverse pass.
      // For example:
      // ```
      // // forward pass
      // _t0 = a;
      // modify(a); // a is modified so we store it
      //
      // // reverse pass
      // a = _t0;
      // modify_pullback(a, ...); // the pullback should always keep `a` intact
      // ```
      // FIXME: Handle storing data passed through pointers and structures.
      // FIXME: Improve TBR to handle these stores.
      QualType paramTy = PVD->getType();
      bool passByRef = paramTy->isLValueReferenceType() &&
                       !paramTy.getNonReferenceType().isConstQualified();
      if (passByRef) {
        StmtDiff pushPop = StoreAndRestore(argDiff.getExpr());
        addToCurrentBlock(pushPop.getStmt());
        PreCallStmts.push_back(pushPop.getStmt_dx());
      }
      CallArgs.push_back(argDiff.getExpr());
      DerivedCallArgs.push_back(argDiff.getExpr());
    }
    // Store all the derived call output args (if any)
    llvm::SmallVector<Expr*, 16> DerivedCallOutputArgs{};
    // It is required because call to numerical diff and reverse mode diff
    // requires (slightly) different arguments.
    llvm::SmallVector<Expr*, 16> pullbackCallArgs{};

    // Stores differentiation result of implicit `this` object, if any.
    StmtDiff baseDiff;
    Expr* baseExpr = nullptr;
    size_t idx = 0;

    /// Add base derivative expression in the derived call output args list if
    /// `CE` is a call to an instance member function.
    if (MD) {
      if (isLambdaCallOperator(MD)) {
        QualType ptrType = m_Context.getPointerType(m_Context.getRecordType(
            FD->getDeclContext()->getOuterLexicalRecordContext()));
        baseDiff =
            StmtDiff(Clone(dyn_cast<CXXOperatorCallExpr>(CE)->getArg(0)),
                     new (m_Context) CXXNullPtrLiteralExpr(ptrType, Loc));
      } else if (MD->isInstance()) {
        bool isPassedByRef = utils::IsReferenceOrPointerArg(baseOriginalE);
        if (!isPassedByRef) {
          QualType dBaseTy =
              utils::getNonConstType(baseOriginalE->getType(), m_Sema);
          VarDecl* dBaseDecl =
              BuildVarDecl(dBaseTy, "_r", getZeroInit(dBaseTy));
          PreCallStmts.push_back(BuildDeclStmt(dBaseDecl));
          DeclRefExpr* dBaseRef = BuildDeclRef(dBaseDecl);
          baseDiff = Visit(baseOriginalE, dBaseRef);
          baseDiff.updateStmtDx(Clone(dBaseRef));
        } else
          baseDiff = Visit(baseOriginalE);
        baseExpr = baseDiff.getExpr();
        QualType baseTy = baseExpr->getType();
        if (baseTy->isPointerType())
          baseTy = baseTy->getPointeeType();
        CXXRecordDecl* baseRD = baseTy->getAsCXXRecordDecl();
        if (isPassedByRef && !MD->isConst() && utils::isCopyable(baseRD)) {
          Expr* baseDiffStore =
              GlobalStoreAndRef(baseDiff.getExpr(), "_t", /*force=*/true);
          Expr* assign = BuildOp(BO_Assign, baseDiff.getExpr(), baseDiffStore);
          PreCallStmts.push_back(assign);
        }
        Expr* baseDerivative = baseDiff.getExpr_dx();
        if (!baseDerivative->getType()->isPointerType())
          baseDerivative =
              BuildOp(UnaryOperatorKind::UO_AddrOf, baseDerivative);
        DerivedCallOutputArgs.push_back(baseDerivative);
      }
    }

    for (auto* argDerivative : CallArgDx) {
      Expr* gradArgExpr = nullptr;
      QualType paramTy = FD->getParamDecl(idx)->getType();
      if (!argDerivative || utils::isArrayOrPointerType(paramTy) ||
          isCladArrayType(argDerivative->getType()) ||
          isa<CUDAKernelCallExpr>(CE))
        gradArgExpr = argDerivative;
      else
        gradArgExpr =
            BuildOp(UO_AddrOf, argDerivative, m_DiffReq->getLocation());
      DerivedCallOutputArgs.push_back(gradArgExpr);
      idx++;
    }
    Expr* pullback = dfdx();

    if (returnType->isVoidType()) {
      assert(pullback == nullptr && returnType->isVoidType() &&
             "Call to function returning void type should not have any "
             "corresponding dfdx().");
    }

    if ((pullback == nullptr) &&
        !(returnType->isPointerType() || returnType->isVoidType()))
      pullback = getZeroInit(returnType.getNonReferenceType());

    for (Expr* arg : DerivedCallOutputArgs)
      if (arg)
        DerivedCallArgs.push_back(arg);
    pullbackCallArgs = DerivedCallArgs;

    if (pullback)
      pullbackCallArgs.insert(pullbackCallArgs.begin() + CE->getNumArgs() -
                                  static_cast<int>(isMethodOperatorCall),
                              pullback);

    // Build the DiffRequest
    DiffRequest pullbackRequest{};
    pullbackRequest.Function = FD;

    // If the function has a single arg and does not return a reference or take
    // arg by reference, we can request a derivative w.r.t. to this arg using
    // the forward mode.
    bool asGrad = !utils::canUsePushforwardInRevMode(FD);
    if (!asGrad) {
      pullbackCallArgs.resize(1);
      pullbackCallArgs.push_back(ConstantFolder::synthesizeLiteral(
          DerivedCallArgs.front()->getType(), m_Context, /*val=*/1));
    }

    pullbackRequest.BaseFunctionName = clad::utils::ComputeEffectiveFnName(FD);
    pullbackRequest.Mode = asGrad ? DiffMode::pullback : DiffMode::pushforward;
    bool hasDynamicNonDiffParams = false;

    // Silence diag outputs in nested derivation process.
    pullbackRequest.VerboseDiags = false;
    pullbackRequest.EnableTBRAnalysis = m_DiffReq.EnableTBRAnalysis;
    pullbackRequest.EnableVariedAnalysis = m_DiffReq.EnableVariedAnalysis;
    if (asGrad)
      for (size_t i = 0, e = FD->getNumParams(); i < e; ++i) {
        const auto* PVD = FD->getParamDecl(i);
        // static member function doesn't have `this` pointer
        size_t offset = (bool)MD && MD->isInstance();
        if (MD && isLambdaCallOperator(MD)) {
          pullbackRequest.DVI.push_back(PVD);
        } else if (DerivedCallOutputArgs[i + offset]) {
          if (!m_DiffReq.CUDAGlobalArgsIndexes.empty() &&
              m_DiffReq.HasIndependentParameter(PVD))
            pullbackRequest.CUDAGlobalArgsIndexes.push_back(i);
          pullbackRequest.DVI.push_back(PVD);
        } else
          hasDynamicNonDiffParams = true;
      }

    FunctionDecl* pullbackFD = nullptr;
    Expr* OverloadedDerivedFn = nullptr;

    // FIXME: Error estimation currently uses singleton objects -
    // m_ErrorEstHandler and m_EstModel, which is cleared after each
    // error_estimate request. This requires the pullback to be derived
    // at the same time to access the singleton objects.
    // No call context corresponds to second derivatives used in hessians,
    // which aren't scheduled statically yet.
    if (m_ExternalSource || !m_DiffReq.CallContext || hasDynamicNonDiffParams ||
        FD->getNameAsString() == "cudaMemcpy") {
      // Try to find it in builtin derivatives.
      std::string customPullback = pullbackRequest.ComputeDerivativeName();
      if (MD && MD->isInstance())
        pullbackCallArgs.insert(pullbackCallArgs.begin(), baseExpr);
      OverloadedDerivedFn =
          m_Builder.BuildCallToCustomDerivativeOrNumericalDiff(
              customPullback, pullbackCallArgs, getCurrentScope(), CE,
              /*forCustomDerv=*/true, /*namespaceShouldExist=*/true,
              CUDAExecConfig);
      if (MD && MD->isInstance())
        pullbackCallArgs.erase(pullbackCallArgs.begin());
      if (auto* foundCE = cast_or_null<CallExpr>(OverloadedDerivedFn))
        pullbackFD = foundCE->getDirectCallee();

      // Derivative was not found, request differentiation
      if (!pullbackFD) {
        if (m_ExternalSource) {
          m_ExternalSource->ActBeforeDifferentiatingCallExpr(
              pullbackCallArgs, PreCallStmts, dfdx());
          pullbackFD =
              plugin::ProcessDiffRequest(m_CladPlugin, pullbackRequest);
        } else
          pullbackFD = m_Builder.HandleNestedDiffRequest(pullbackRequest);
      }
    } else
      pullbackFD = FindDerivedFunction(pullbackRequest);

    if (pullbackFD) {
      auto* pullbackMD = dyn_cast<CXXMethodDecl>(pullbackFD);
      Expr* baseE = baseDiff.getExpr();
      if (pullbackMD && pullbackMD->isInstance()) {
        OverloadedDerivedFn = BuildCallExprToMemFn(baseE, pullbackFD->getName(),
                                                   pullbackCallArgs, Loc);
      } else {
        if (baseE) {
          baseE = BuildOp(UO_AddrOf, baseE);
          pullbackCallArgs.insert(pullbackCallArgs.begin(), baseE);
        }
        OverloadedDerivedFn =
            m_Sema
                .ActOnCallExpr(getCurrentScope(), BuildDeclRef(pullbackFD), Loc,
                               pullbackCallArgs, Loc, CUDAExecConfig)
                .get();
      }
    } else if (!utils::HasAnyReferenceOrPointerArgument(FD) && !MD) {
      // FIXME: Add support for reference arguments to the numerical diff. If
      // it already correctly support reference arguments then confirm the
      // support and add tests for the same.
      //
      // Clad failed to derive it. Try numerically deriving it.
      if (NArgs == 1) {
        OverloadedDerivedFn = GetSingleArgCentralDiffCall(
            Clone(CE->getCallee()), DerivedCallArgs[0],
            /*targetPos=*/0,
            /*numArgs=*/1, DerivedCallArgs, CUDAExecConfig);
        asGrad = !OverloadedDerivedFn;
      } else {
        auto CEType = utils::getNonConstType(CE->getType(), m_Sema);
        OverloadedDerivedFn = GetMultiArgCentralDiffCall(
            Clone(CE->getCallee()), CEType.getCanonicalType(), CE->getNumArgs(),
            dfdx(), PreCallStmts, PostCallStmts, DerivedCallArgs, CallArgDx,
            CUDAExecConfig);
      }
      CallExprDiffDiagnostics(FD, CE->getBeginLoc());
    }

    if (!OverloadedDerivedFn) {
      Stmts& block = getCurrentBlock(direction::reverse);
      block.insert(block.begin(), PreCallStmts.begin(), PreCallStmts.end());
      return StmtDiff(Clone(CE));
    }

    // Derivative was found.
    FunctionDecl* fnDecl =
        dyn_cast<CallExpr>(OverloadedDerivedFn)->getDirectCallee();
    // Put Result array declaration in the function body.
    // Call the gradient, passing Result as the last Arg.
    Stmts& block = getCurrentBlock(direction::reverse);
    Stmts::iterator it = std::begin(block) + insertionPoint;
    // Insert PreCallStmts
    it = block.insert(it, PreCallStmts.begin(), PreCallStmts.end());
    it += PreCallStmts.size();
    if (!asGrad) {
      if (utils::IsCladValueAndPushforwardType(fnDecl->getReturnType()))
        OverloadedDerivedFn = utils::BuildMemberExpr(
            m_Sema, getCurrentScope(), OverloadedDerivedFn, "pushforward");
      // If the derivative is called through _darg0 instead of _grad.
      Expr* d = BuildOp(BO_Mul, dfdx(), OverloadedDerivedFn);
      Expr* addGrad = BuildOp(BO_AddAssign, Clone(CallArgDx[0]), d);
      it = block.insert(it, addGrad);
      it++;
    } else {
      // Insert the CallExpr to the derived function
      it = block.insert(it, OverloadedDerivedFn);
      it++;
    }
    // Insert PostCallStmts
    block.insert(it, PostCallStmts.begin(), PostCallStmts.end());

    if (m_ExternalSource)
      m_ExternalSource->ActBeforeFinalizingVisitCallExpr(
          CE, OverloadedDerivedFn, DerivedCallArgs, CallArgDx, asGrad);

    if (isa<CUDAKernelCallExpr>(CE))
      return StmtDiff(Clone(CE));

    Expr* call = nullptr;
    // Lookup a reverse_forw function and build if necessary.
    DiffRequest calleeFnForwPassReq;
    calleeFnForwPassReq.Function = FD;
    calleeFnForwPassReq.Mode = DiffMode::reverse_mode_forward_pass;
    calleeFnForwPassReq.BaseFunctionName =
        clad::utils::ComputeEffectiveFnName(FD);
    calleeFnForwPassReq.VerboseDiags = true;

    FunctionDecl* calleeFnForwPassFD = FindDerivedFunction(calleeFnForwPassReq);
    if (calleeFnForwPassFD) {
      for (std::size_t i = 0, e = CE->getNumArgs() - isMethodOperatorCall;
           i != e; ++i) {
        const Expr* arg = CE->getArg(i + isMethodOperatorCall);
        if (!utils::IsReferenceOrPointerArg(arg) || arg->isXValue())
          CallArgDx[i] = getZeroInit(arg->getType());
      }
      if (baseDiff.getExpr_dx() &&
          !baseDiff.getExpr_dx()->getType()->isPointerType())
        CallArgDx.insert(
            CallArgDx.begin(),
            BuildOp(UnaryOperatorKind::UO_AddrOf, baseDiff.getExpr_dx(), Loc));
      CallArgs.insert(CallArgs.end(), CallArgDx.begin(), CallArgDx.end());
      const auto* forwPassMD = dyn_cast<CXXMethodDecl>(calleeFnForwPassFD);
      Expr* baseE = baseDiff.getExpr();
      if (forwPassMD && forwPassMD->isInstance()) {
        call = BuildCallExprToMemFn(
            baseDiff.getExpr(), calleeFnForwPassFD->getName(), CallArgs, Loc);
      } else {
        if (baseE) {
          baseE = BuildOp(UO_AddrOf, baseE);
          CallArgs.insert(CallArgs.begin(), baseE);
        }
        call = m_Sema
                   .ActOnCallExpr(getCurrentScope(),
                                  BuildDeclRef(calleeFnForwPassFD), Loc,
                                  CallArgs, Loc, CUDAExecConfig)
                   .get();
      }
      if (!needsForwPass && !m_TrackVarDeclConstructor)
        return StmtDiff(call);
      Expr* callRes = nullptr;
      if (isInsideLoop)
        callRes = GlobalStoreAndRef(call, /*prefix=*/"_t",
                                    /*force=*/true);
      else
        callRes = StoreAndRef(call);
      auto* resValue =
          utils::BuildMemberExpr(m_Sema, getCurrentScope(), callRes, "value");
      auto* resAdjoint =
          utils::BuildMemberExpr(m_Sema, getCurrentScope(), callRes, "adjoint");
      return StmtDiff(resValue, resAdjoint);
    } // Recreate the original call expression.

    if (const auto* OCE = dyn_cast<CXXOperatorCallExpr>(CE)) {
      if (OCE->getOperator() == clang::OverloadedOperatorKind::OO_Subscript) {
        // If the operator is subscript, we should return the adjoint expression
        auto AdjointCallArgs = CallArgs;
        CallArgs.insert(CallArgs.begin(), baseDiff.getExpr());
        AdjointCallArgs.insert(AdjointCallArgs.begin(), baseDiff.getExpr_dx());
        call = BuildOperatorCall(OCE->getOperator(), CallArgs);
        Expr* call_dx = BuildOperatorCall(OCE->getOperator(), AdjointCallArgs);
        return StmtDiff(call, call_dx);
      }
      if (isMethodOperatorCall)
        CallArgs.insert(CallArgs.begin(), baseDiff.getExpr());
      call = BuildOperatorCall(OCE->getOperator(), CallArgs);
      return StmtDiff(call);
    }

    call = m_Sema
               .ActOnCallExpr(getCurrentScope(), Clone(CE->getCallee()), Loc,
                              CallArgs, Loc, CUDAExecConfig)
               .get();
    return StmtDiff(call);
  }

  Expr* ReverseModeVisitor::GetMultiArgCentralDiffCall(
      Expr* targetFuncCall, QualType retType, unsigned numArgs, Expr* dfdx,
      llvm::SmallVectorImpl<Stmt*>& PreCallStmts,
      llvm::SmallVectorImpl<Stmt*>& PostCallStmts,
      llvm::SmallVectorImpl<Expr*>& args,
      llvm::SmallVectorImpl<Expr*>& outputArgs,
      Expr* CUDAExecConfig /*=nullptr*/) {
    int printErrorInf = m_Builder.shouldPrintNumDiffErrs();
    llvm::SmallVector<Expr*, 16U> NumDiffArgs = {};
    NumDiffArgs.push_back(targetFuncCall);
    // build the output array declaration.
    Expr* size =
        ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, numArgs);
    QualType GradType = clad_compat::getConstantArrayType(
        m_Context, retType,
        llvm::APInt(m_Context.getTargetInfo().getIntWidth(), numArgs),
        /*SizeExpr=*/size,
        /*ASM=*/clad_compat::ArraySizeModifier_Normal,
        /*IndexTypeQuals*/ 0);
    Expr* zero =
        ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
    Expr* init = m_Sema.ActOnInitList(noLoc, {zero}, noLoc).get();
    auto* VD = BuildVarDecl(GradType, "_grad", init);

    PreCallStmts.push_back(BuildDeclStmt(VD));
    NumDiffArgs.push_back(BuildDeclRef(VD));
    NumDiffArgs.push_back(ConstantFolder::synthesizeLiteral(
        m_Context.IntTy, m_Context, printErrorInf));

    // Build the tape push expressions.
    VD->setLocation(m_DiffReq->getLocation());
    for (unsigned i = 0, e = numArgs; i < e; i++) {
      Expr* gradRef = BuildDeclRef(VD);
      Expr* idx =
          ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, i);
      Expr* gradElem = BuildArraySubscript(gradRef, {idx});
      Expr* gradExpr = BuildOp(BO_Mul, dfdx, gradElem);
      // Inputs were not pointers, so the output args are not in global GPU
      // memory. Hence, no need to use atomic ops.
      PostCallStmts.push_back(BuildOp(BO_AddAssign, outputArgs[i], gradExpr));
      NumDiffArgs.push_back(args[i]);
    }
    std::string Name = "central_difference";
    return m_Builder.BuildCallToCustomDerivativeOrNumericalDiff(
        Name, NumDiffArgs, getCurrentScope(),
        /*callSite=*/nullptr,
        /*forCustomDerv=*/false,
        /*namespaceShouldExist=*/false, CUDAExecConfig);
  }

  StmtDiff ReverseModeVisitor::VisitUnaryOperator(const UnaryOperator* UnOp) {
    auto opCode = UnOp->getOpcode();
    Expr* valueForRevPass = nullptr;
    StmtDiff diff{};
    Expr* E = UnOp->getSubExpr();
    // If it is a post-increment/decrement operator, its result is a reference
    // and we should return it.
    Expr* ResultRef = nullptr;

    // For increment/decrement of pointer, perform the same on the
    // derivative pointer also.
    bool isPointerOp = E->getType()->isPointerType();

    if (opCode == UO_Plus)
      // xi = +xj
      // dxi/dxj = +1.0
      // df/dxj += df/dxi * dxi/dxj = df/dxi
      diff = Visit(E, dfdx());
    else if (opCode == UO_Minus) {
      // xi = -xj
      // dxi/dxj = -1.0
      // df/dxj += df/dxi * dxi/dxj = -df/dxi
      auto* d = BuildOp(UO_Minus, dfdx());
      diff = Visit(E, d);
    } else if (opCode == UO_PostInc || opCode == UO_PostDec) {
      diff = Visit(E, dfdx());
      Expr* diff_dx = diff.getExpr_dx();
      if (isPointerOp)
        addToCurrentBlock(BuildOp(opCode, diff_dx), direction::forward);
      if (m_DiffReq.shouldBeRecorded(E)) {
        auto op = opCode == UO_PostInc ? UO_PostDec : UO_PostInc;
        addToCurrentBlock(BuildOp(op, Clone(diff.getRevSweepAsExpr())),
                          direction::reverse);
        if (isPointerOp)
          addToCurrentBlock(BuildOp(op, diff_dx), direction::reverse);
      }

      ResultRef = diff_dx;
      valueForRevPass = diff.getRevSweepAsExpr();
      if (m_ExternalSource)
        m_ExternalSource->ActBeforeFinalizingPostIncDecOp(diff);
    } else if (opCode == UO_PreInc || opCode == UO_PreDec) {
      diff = Visit(E, dfdx());
      Expr* diff_dx = diff.getExpr_dx();
      if (isPointerOp)
        addToCurrentBlock(BuildOp(opCode, diff_dx), direction::forward);
      if (m_DiffReq.shouldBeRecorded(E)) {
        auto op = opCode == UO_PreInc ? UO_PreDec : UO_PreInc;
        addToCurrentBlock(BuildOp(op, Clone(diff.getRevSweepAsExpr())),
                          direction::reverse);
        if (isPointerOp)
          addToCurrentBlock(BuildOp(op, diff_dx), direction::reverse);
      }
      auto op = opCode == UO_PreInc ? BinaryOperatorKind::BO_Add
                                    : BinaryOperatorKind::BO_Sub;
      auto* sum = BuildOp(
          op, diff.getRevSweepAsExpr(),
          ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 1));
      valueForRevPass = utils::BuildParenExpr(m_Sema, sum);
    } else if (opCode == UnaryOperatorKind::UO_Real ||
               opCode == UnaryOperatorKind::UO_Imag) {
      diff = Visit(E);
      ResultRef = BuildOp(opCode, diff.getExpr_dx());
      /// Create and add `__real r += dfdx()` expression.
      if (dfdx()) {
        Expr* add_assign = BuildOp(BO_AddAssign, ResultRef, dfdx());
        // Add it to the body statements.
        addToCurrentBlock(add_assign, direction::reverse);
      }
    } else if (opCode == UnaryOperatorKind::UO_AddrOf) {
      diff = Visit(E);
      Expr* cloneE = BuildOp(UnaryOperatorKind::UO_AddrOf, diff.getExpr());
      Expr* derivedE = BuildOp(UnaryOperatorKind::UO_AddrOf, diff.getExpr_dx());
      return {cloneE, derivedE};
    } else if (opCode == UnaryOperatorKind::UO_Deref) {
      diff = Visit(E);
      Expr* cloneE = BuildOp(UnaryOperatorKind::UO_Deref, diff.getExpr());

      // If we have a pointer to a member expression, which is
      // non-differentiable, we just return a clone of the original expression.
      if (auto* ME = dyn_cast<MemberExpr>(diff.getExpr()))
        if (clad::utils::hasNonDifferentiableAttribute(ME->getMemberDecl()))
          return {cloneE};

      Expr* diff_dx = diff.getExpr_dx();
      Expr* derivedE = BuildOp(UnaryOperatorKind::UO_Deref, diff_dx);
      // Create the (target += dfdx) statement.
      if (dfdx() && derivedE && !derivedE->getType()->isRecordType()) {
        Expr* add_assign = nullptr;
        if (shouldUseCudaAtomicOps(diff_dx))
          add_assign = BuildCallToCudaAtomicAdd(diff_dx, dfdx());
        else
          add_assign = BuildOp(BO_AddAssign, derivedE, dfdx());

        addToCurrentBlock(add_assign, direction::reverse);
      }
      return {cloneE, derivedE};
    } else {
      if (opCode != UO_LNot)
        // We should only output warnings on visiting boolean conditions
        // when it is related to some indepdendent variable and causes
        // discontinuity in the function space.
        // FIXME: We should support boolean differentiation or ignore it
        // completely
        unsupportedOpWarn(UnOp->getOperatorLoc());
      diff = Visit(E);
      ResultRef = diff.getExpr_dx();
    }
    Expr* op = BuildOp(opCode, diff.getExpr());
    return StmtDiff(op, ResultRef, valueForRevPass);
  }

  StmtDiff
  ReverseModeVisitor::VisitBinaryOperator(const BinaryOperator* BinOp) {
    auto opCode = BinOp->getOpcode();
    StmtDiff Ldiff{};
    StmtDiff Rdiff{};
    StmtDiff Lstored{};
    Expr* valueForRevPass = nullptr;
    auto* L = BinOp->getLHS();
    auto* R = BinOp->getRHS();
    // If it is an assignment operator, its result is a reference to LHS and
    // we should return it.
    Expr* ResultRef = nullptr;

    bool isPointerOp =
        L->getType()->isPointerType() || R->getType()->isPointerType();

    if (opCode == BO_Add) {
      // xi = xl + xr
      // dxi/xl = 1.0
      // df/dxl += df/dxi * dxi/xl = df/dxi
      Ldiff = Visit(L, dfdx());
      // dxi/xr = 1.0
      // df/dxr += df/dxi * dxi/xr = df/dxi
      Rdiff = Visit(R, dfdx());
    } else if (opCode == BO_Sub) {
      // xi = xl - xr
      // dxi/xl = 1.0
      // df/dxl += df/dxi * dxi/xl = df/dxi
      Ldiff = Visit(L, dfdx());
      // dxi/xr = -1.0
      // df/dxl += df/dxi * dxi/xr = -df/dxi
      auto* dr = BuildOp(UO_Minus, dfdx());
      Rdiff = Visit(R, dr);
    } else if (opCode == BO_Mul) {
      // xi = xl * xr
      // dxi/xl = xr
      // df/dxl += df/dxi * dxi/xl = df/dxi * xr
      // Create uninitialized "global" variable for the right multiplier.
      // It will be assigned later after R is visited and cloned. This allows
      // to reduce cloning complexity and only clones once. Storing it in a
      // global variable allows to save current result and make it accessible
      // in the reverse pass.
      DelayedStoreResult RDelayed = DelayedGlobalStoreAndRef(R);
      StmtDiff& RResult = RDelayed.Result;

      Expr* dl = nullptr;
      if (dfdx())
        dl = BuildOp(BO_Mul, dfdx(), RResult.getRevSweepAsExpr());
      Ldiff = Visit(L, dl);
      // dxi/xr = xl
      // df/dxr += df/dxi * dxi/xr = df/dxi * xl
      // Store left multiplier and assign it with L.
      StmtDiff LStored = Ldiff;
      // Catch the pop statement and emit it after
      // the LStored value is used.
      // This workaround is necessary because GlobalStoreAndRef
      // is designed to work with the reversed order of statements
      // in the reverse sweep and in RMV::VisitBinaryOperator
      // the order is not reversed.
      beginBlock(direction::reverse);
      if (!ShouldRecompute(LStored.getExpr()))
        LStored = GlobalStoreAndRef(LStored.getExpr(), /*prefix=*/"_t",
                                    /*force=*/true);
      Stmt* LPop = endBlock(direction::reverse);
      Expr* dr = nullptr;
      if (dfdx())
        dr = BuildOp(BO_Mul, LStored.getRevSweepAsExpr(), dfdx());
      Rdiff = Visit(R, dr);
      // Assign right multiplier's variable with R.
      RDelayed.Finalize(Rdiff.getExpr());
      addToCurrentBlock(utils::unwrapIfSingleStmt(LPop), direction::reverse);
      std::tie(Ldiff, Rdiff) = std::make_pair(LStored, RResult);
    } else if (opCode == BO_Div) {
      // xi = xl / xr
      // dxi/xl = 1 / xr
      // df/dxl += df/dxi * dxi/xl = df/dxi * (1/xr)
      auto RDelayed = DelayedGlobalStoreAndRef(R, /*prefix=*/"_t",
                                               /*forceStore=*/true);
      StmtDiff& RResult = RDelayed.Result;
      Expr* dl = nullptr;
      if (dfdx())
        dl = BuildOp(BO_Div, dfdx(), RResult.getExpr());
      Ldiff = Visit(L, dl);
      StmtDiff LStored = Ldiff;
      // Catch the pop statement and emit it after
      // the LStored value is used.
      // This workaround is necessary because GlobalStoreAndRef
      // is designed to work with the reversed order of statements
      // in the reverse sweep and in RMV::VisitBinaryOperator
      // the order is not reversed.
      beginBlock(direction::reverse);
      if (!ShouldRecompute(LStored.getExpr()))
        LStored = GlobalStoreAndRef(LStored.getExpr(), /*prefix=*/"_t",
                                    /*force=*/true);
      Stmt* LPop = endBlock(direction::reverse);
      Expr::EvalResult dummy;
      if (!clad_compat::Expr_EvaluateAsConstantExpr(R, dummy, m_Context) ||
          RDelayed.needsUpdate) {
        // dxi/xr = -xl / (xr * xr)
        // df/dxl += df/dxi * dxi/xr = df/dxi * (-xl /(xr * xr))
        // Wrap R * R in parentheses: (R * R). otherwise code like 1 / R * R is
        // produced instead of 1 / (R * R).
        Expr* dr = nullptr;
        if (dfdx()) {
          Expr* RxR = BuildParens(
              BuildOp(BO_Mul, RResult.getExpr(), RResult.getExpr()));
          dr = BuildOp(BO_Mul, dfdx(),
                       BuildOp(UO_Minus,
                               BuildParens(BuildOp(
                                   BO_Div, LStored.getRevSweepAsExpr(), RxR))));
          dr = StoreAndRef(dr, direction::reverse);
        }
        Rdiff = Visit(R, dr);
        RDelayed.Finalize(Rdiff.getExpr());
      }
      addToCurrentBlock(utils::unwrapIfSingleStmt(LPop), direction::reverse);
      std::tie(Ldiff, Rdiff) = std::make_pair(LStored, RResult);
    } else if (BinOp->isAssignmentOp()) {
      if (L->isModifiableLvalue(m_Context) != Expr::MLV_Valid) {
        diag(DiagnosticsEngine::Warning,
             BinOp->getEndLoc(),
             "derivative of an assignment attempts to assign to unassignable "
             "expr, assignment ignored");
        return Clone(BinOp);
      }

      // Visit LHS, but delay emission of its derivative statements, save them
      // in Lblock
      beginBlock(direction::reverse);
      Ldiff = Visit(L, dfdx());

      if (L->HasSideEffects(m_Context)) {
        Expr* E = Ldiff.getExpr();
        llvm::SmallVector<Expr*, 4> returnExprs;
        utils::GetInnermostReturnExpr(E, returnExprs);
        if (returnExprs.size() == 1) {
          addToCurrentBlock(E, direction::forward);
          Ldiff.updateStmt(returnExprs[0]);
        } else {
          auto* storeE = GlobalStoreAndRef(BuildOp(UO_AddrOf, E));
          Ldiff.updateStmt(BuildOp(UO_Deref, storeE));
        }
      }

      Stmts Lblock = EndBlockWithoutCreatingCS(direction::reverse);

      Expr* LCloned = Ldiff.getExpr();
      // For x, ResultRef is _d_x, for x[i] its _d_x[i], for reference exprs
      // like (x = y) it propagates recursively, so _d_x is also returned.
      ResultRef = Ldiff.getExpr_dx();
      // If assigned expr is dependent, first update its derivative;
      if (dfdx() && !Lblock.empty()) {
        addToCurrentBlock(*Lblock.begin(), direction::reverse);
        Lblock.erase(Lblock.begin());
      }

      // Store the value of the LHS of the assignment in the forward pass
      // and restore it in the reverse pass
      if (m_DiffReq.shouldBeRecorded(L)) {
        StmtDiff pushPop = StoreAndRestore(LCloned);
        addToCurrentBlock(pushPop.getStmt(), direction::forward);
        addToCurrentBlock(pushPop.getStmt_dx(), direction::reverse);
      }

      if (!ResultRef)
        return Clone(BinOp);
      // We need to store values of derivative pointer variables in forward pass
      // and restore them in reverse pass.
      if (isPointerOp) {
        StmtDiff pushPop = StoreAndRestore(Ldiff.getExpr_dx());
        addToCurrentBlock(pushPop.getStmt(), direction::forward);
        addToCurrentBlock(pushPop.getStmt_dx(), direction::reverse);
      }

      if (m_ExternalSource)
        m_ExternalSource->ActAfterCloningLHSOfAssignOp(LCloned, R, opCode);

      // Save old value for the derivative of LHS, to avoid problems with cases
      // like x = x.
      clang::Expr* oldValue = nullptr;

      // For pointer types, no need to store old derivatives.
      if (!isPointerOp)
        oldValue = StoreAndRef(ResultRef, direction::reverse, "_r_d",
                               /*forceDeclCreation=*/true);
      if (opCode == BO_Assign) {
        if (!isPointerOp) {
          // Add the statement `dl = 0;`
          Expr* zero = getZeroInit(ResultRef->getType());
          addToCurrentBlock(BuildOp(BO_Assign, ResultRef, zero),
                            direction::reverse);
        }
        Rdiff = Visit(R, oldValue);
        valueForRevPass = Rdiff.getRevSweepAsExpr();
      } else if (opCode == BO_AddAssign) {
        Rdiff = Visit(R, oldValue);
        if (!isPointerOp)
          valueForRevPass = BuildOp(BO_Add, Rdiff.getRevSweepAsExpr(),
                                    Ldiff.getRevSweepAsExpr());
      } else if (opCode == BO_SubAssign) {
        Rdiff = Visit(R, BuildOp(UO_Minus, oldValue));
        if (!isPointerOp)
          valueForRevPass = BuildOp(BO_Sub, Rdiff.getRevSweepAsExpr(),
                                    Ldiff.getRevSweepAsExpr());
      } else if (opCode == BO_MulAssign) {
        // Create a reference variable to keep the result of LHS, since it
        // must be used on 2 places: when storing to a global variable
        // accessible from the reverse pass, and when rebuilding the original
        // expression for the forward pass. This allows to avoid executing
        // same expression with side effects twice. E.g., on
        //   double r = (x *= y) *= z;
        // instead of:
        //   _t0 = (x *= y);
        //   double r = (x *= y) *= z;
        // which modifies x twice, we get:
        //   double & _ref0 = (x *= y);
        //   _t0 = _ref0;
        //   double r = _ref0 *= z;
        if (isInsideLoop)
          addToCurrentBlock(LCloned, direction::forward);
        // Add the statement `dl = 0;`
        Expr* zero = getZeroInit(ResultRef->getType());
        addToCurrentBlock(BuildOp(BO_Assign, ResultRef, zero),
                          direction::reverse);
        /// Capture all the emitted statements while visiting R
        /// and insert them after `dl += dl * R`
        beginBlock(direction::reverse);
        Expr* dr = BuildOp(BO_Mul, LCloned, oldValue);
        Rdiff = Visit(R, dr);
        Stmts RBlock = EndBlockWithoutCreatingCS(direction::reverse);
        addToCurrentBlock(
            BuildOp(BO_AddAssign, ResultRef,
                    BuildOp(BO_Mul, oldValue, Rdiff.getRevSweepAsExpr())),
            direction::reverse);
        for (auto& S : RBlock)
          addToCurrentBlock(S, direction::reverse);
        valueForRevPass = BuildOp(BO_Mul, Rdiff.getRevSweepAsExpr(),
                                  Ldiff.getRevSweepAsExpr());
        std::tie(Ldiff, Rdiff) = std::make_pair(LCloned, Rdiff.getExpr());
      } else if (opCode == BO_DivAssign) {
        // Add the statement `dl = 0;`
        Expr* zero = getZeroInit(ResultRef->getType());
        addToCurrentBlock(BuildOp(BO_Assign, ResultRef, zero),
                          direction::reverse);
        auto RDelayed = DelayedGlobalStoreAndRef(R, /*prefix=*/"_t",
                                                 /*forceStore=*/true);
        StmtDiff& RResult = RDelayed.Result;
        Expr* RStored =
            StoreAndRef(RResult.getRevSweepAsExpr(), direction::reverse);
        addToCurrentBlock(BuildOp(BO_AddAssign, ResultRef,
                                  BuildOp(BO_Div, oldValue, RStored)),
                          direction::reverse);
        if (isInsideLoop)
          addToCurrentBlock(LCloned, direction::forward);
        Expr* RxR = BuildParens(BuildOp(BO_Mul, RStored, RStored));
        Expr* dr = BuildOp(BO_Mul, oldValue,
                           BuildOp(UO_Minus, BuildOp(BO_Div, LCloned, RxR)));
        dr = StoreAndRef(dr, direction::reverse);
        Rdiff = Visit(R, dr);
        RDelayed.Finalize(Rdiff.getExpr());
        valueForRevPass = BuildOp(BO_Div, Rdiff.getRevSweepAsExpr(),
                                  Ldiff.getRevSweepAsExpr());
        std::tie(Ldiff, Rdiff) = std::make_pair(LCloned, RResult);
      } else
        llvm_unreachable("unknown assignment opCode");
      if (m_ExternalSource)
        m_ExternalSource->ActBeforeFinalizingAssignOp(LCloned, ResultRef, R,
                                                      opCode);

      // Output statements from Visit(L).
      for (Stmt* S : Lblock)
        addToCurrentBlock(S, direction::reverse);
    } else if (opCode == BO_Comma) {
      auto* zero =
          ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
      Rdiff = Visit(R, dfdx());
      Ldiff = Visit(L, zero);
      valueForRevPass = Ldiff.getRevSweepAsExpr();
      ResultRef = Ldiff.getExpr();
    } else if (opCode == BO_LAnd) {
      VarDecl* condVar = GlobalStoreImpl(m_Context.BoolTy, "_cond");
      VarDecl* derivedCondVar = GlobalStoreImpl(
          m_Context.DoubleTy, "_d" + condVar->getNameAsString());
      addToBlock(BuildOp(BO_Assign, BuildDeclRef(derivedCondVar),
                         ConstantFolder::synthesizeLiteral(
                             m_Context.DoubleTy, m_Context, /*val=*/0)),
                 m_Globals);
      Expr* condVarRef = BuildDeclRef(condVar);
      Expr* assignExpr = BuildOp(BO_Assign, condVarRef, Clone(R));
      m_Variables.emplace(condVar, BuildDeclRef(derivedCondVar));
      auto* IfStmt = clad_compat::IfStmt_Create(
          /*Ctx=*/m_Context, /*IL=*/noLoc, /*IsConstexpr=*/false,
          /*Init=*/nullptr, /*Var=*/nullptr,
          /*Cond=*/L, /*LPL=*/noLoc, /*RPL=*/noLoc, /*Then=*/assignExpr,
          /*EL=*/noLoc,
          /*Else=*/nullptr);

      StmtDiff IfStmtDiff = VisitIfStmt(IfStmt);
      addToCurrentBlock(utils::unwrapIfSingleStmt(IfStmtDiff.getStmt()));
      addToCurrentBlock(utils::unwrapIfSingleStmt(IfStmtDiff.getStmt_dx()),
                        direction::reverse);
      auto* condDiffStored = IfStmtDiff.getRevSweepAsExpr();
      return BuildOp(BO_LAnd, condDiffStored, condVarRef);
    } else {
      // We should not output any warning on visiting boolean conditions
      // FIXME: We should support boolean differentiation or ignore it
      // completely
      if (!BinOp->isComparisonOp() && !BinOp->isLogicalOp())
        unsupportedOpWarn(BinOp->getOperatorLoc());

      return BuildOp(opCode, Visit(L).getExpr(), Visit(R).getExpr());
    }
    Expr* op = BuildOp(opCode, Ldiff.getExpr(), Rdiff.getExpr());

    // For pointer types.
    if (isPointerOp) {
      if (opCode == BO_Add || opCode == BO_Sub) {
        Expr* derivedL = nullptr;
        Expr* derivedR = nullptr;
        ComputeEffectiveDOperands(Ldiff, Rdiff, derivedL, derivedR);
        if (opCode == BO_Sub)
          derivedR = BuildParens(derivedR);
        return StmtDiff(op, BuildOp(opCode, derivedL, derivedR),
                        valueForRevPass);
      }
      if (opCode == BO_Assign || opCode == BO_AddAssign ||
          opCode == BO_SubAssign) {
        Expr* derivedL = nullptr;
        Expr* derivedR = nullptr;
        ComputeEffectiveDOperands(Ldiff, Rdiff, derivedL, derivedR);
        addToCurrentBlock(BuildOp(opCode, derivedL, derivedR),
                          direction::forward);
        if (opCode == BO_Assign && derivedL && derivedR)
          if (Expr* memsetCall = CheckAndBuildCallToMemset(
                  derivedL, derivedR->IgnoreParenCasts()))
            addToCurrentBlock(memsetCall, direction::forward);
      }
    }
    return StmtDiff(op, ResultRef, valueForRevPass);
  }

  QualType ReverseModeVisitor::CloneType(QualType T) {
    QualType dT = VisitorBase::CloneType(T);
    return utils::replaceStdInitListWithCladArray(m_Sema, dT);
  }

  DeclDiff<VarDecl> ReverseModeVisitor::DifferentiateVarDecl(const VarDecl* VD,
                                                             bool keepLocal) {
    StmtDiff initDiff;
    Expr* VDDerivedInit = nullptr;

    // Local declarations are promoted to the function global scope. This
    // procedure is done to make declarations visible in the reverse sweep.
    // The reverse_mode_forward_pass mode does not have a reverse pass so
    // declarations don't have to be moved to the function global scope.
    bool promoteToFnScope =
        !getCurrentScope()->isFunctionScope() &&
        m_DiffReq.Mode != DiffMode::reverse_mode_forward_pass && !keepLocal;
    QualType VDCloneType;
    QualType VDDerivedType;
    QualType VDType = VD->getType();
    // If the cloned declaration is moved to the function global scope,
    // change its type for the corresponding adjoint type.
    if (promoteToFnScope) {
      VDDerivedType = ComputeAdjointType(CloneType(VDType));
      VDCloneType = VDDerivedType;
      if (isa<ArrayType>(VDCloneType) && !isa<IncompleteArrayType>(VDCloneType))
        VDCloneType = utils::GetCladArrayOfType(
            m_Sema, m_Context.getBaseElementType(VDCloneType));
    } else {
      VDCloneType = CloneType(VDType);
      VDDerivedType = utils::getNonConstType(VDCloneType, m_Sema);
    }

    bool isRefType = VDType->isLValueReferenceType();
    VarDecl* VDDerived = nullptr;
    bool isPointerType = VDType->isPointerType();
    bool isInitializedByNewExpr = false;
    bool initializeDerivedVar = m_DiffReq.shouldHaveAdjoint(VD) &&
                                !clad::utils::hasNonDifferentiableAttribute(VD);

    if (Expr* size = getStdInitListSizeExpr(VD->getInit()))
      VDDerivedInit = size;

    // Check if the variable is pointer type and initialized by new expression
    if (isPointerType && VD->getInit() && isa<CXXNewExpr>(VD->getInit()))
      isInitializedByNewExpr = true;

    bool isConstructInit =
        VD->getInit() && isa<CXXConstructExpr>(VD->getInit()->IgnoreImplicit());
    const CXXRecordDecl* RD = VD->getType()->getAsCXXRecordDecl();
    bool isNonAggrClass = RD && !RD->isAggregate();

    // We initialize adjoints with original variables as part of
    // the strategy to maintain the structure of the original variable.
    // After that, we'll zero-initialize the adjoint. e.g.
    // ```
    // std::vector<...> v{x, y, z};
    // std::vector<...> _d_v{v}; // The length of the vector is preserved
    // clad::zero_init(_d_v);
    // ```
    // Also, if the original is initialized with a zero-constructor, it can be
    // used for the adjoint as well.
    bool shouldCopyInitialize =
        isConstructInit && isNonAggrClass &&
        cast<CXXConstructExpr>(VD->getInit()->IgnoreImplicit())->getNumArgs() &&
        utils::isCopyable(VDType->getAsCXXRecordDecl());

    // Temporarily initialize the object with `*nullptr` to avoid
    // a potential error because of non-existing default constructor.
    if (!VDDerivedInit && shouldCopyInitialize) {
      QualType ptrType =
          m_Context.getPointerType(VDDerivedType.getUnqualifiedType());
      Expr* dummy = getZeroInit(ptrType);
      VDDerivedInit = BuildOp(UO_Deref, dummy);
    }

    bool isDirectInit = VD->isDirectInit();
    // VDDerivedInit now serves two purposes -- as the initial derivative value
    // or the size of the derivative array -- depending on the primal type.
    if (promoteToFnScope)
      if (const auto* AT = dyn_cast<ArrayType>(VDType)) {
        // If an array-type declaration is promoted to function global,
        // its type is changed for clad::array. In that case we should
        // initialize it with its size.
        initDiff = getArraySizeExpr(AT, m_Context, *this);
        isDirectInit = true;
      }
    // If VD is a reference to a local variable, then the initial value is set
    // to the derived variable of the corresponding local variable.
    // If VD is a reference to a non-local variable (global variable, struct
    // member etc), then no derived variable is available, thus `VDDerived`
    // does not need to reference any variable, consequentially the
    // `VDDerivedType` is the corresponding non-reference type and the initial
    // value is set to 0.
    // Otherwise, for non-reference types, the initial value is set to 0.
    if (!VDDerivedInit)
      VDDerivedInit = getZeroInit(VDType);

    if (isRefType) {
      initDiff = Visit(VD->getInit());
      if (!initDiff.getStmt_dx()) {
        VDDerivedType = ComputeAdjointType(VDType.getNonReferenceType());
        isRefType = false;
      }
      if (promoteToFnScope || !isRefType)
        VDDerivedInit = getZeroInit(VDDerivedType);
      else
        VDDerivedInit = initDiff.getExpr_dx();
    }

    // if VD is a pointer type, then the initial value is set to the derived
    // expression of the corresponding pointer type.
    else if (isPointerType) {
      if (!isInitializedByNewExpr)
        initDiff = Visit(VD->getInit());

      // If the pointer is const and derived expression is not available, then
      // we should not create a derived variable for it. This will be useful
      // for reducing number of differentiation variables in pullbacks.
      bool constPointer = VDType->getPointeeType().isConstQualified();
      if (constPointer && !isInitializedByNewExpr && !initDiff.getExpr_dx())
        initializeDerivedVar = false;
      else {
        VDDerivedType = utils::getNonConstType(VDDerivedType, m_Sema);
        // If it's a pointer to a constant type, then remove the constness.
        if (constPointer) {
          // first extract the pointee type
          auto pointeeType = VDType->getPointeeType();
          // then remove the constness
          pointeeType.removeLocalConst();
          // then create a new pointer type with the new pointee type
          VDDerivedType = m_Context.getPointerType(pointeeType);
        }
        VDDerivedInit = getZeroInit(VDDerivedType);
      }
    }
    if (initializeDerivedVar)
      VDDerived = BuildGlobalVarDecl(
          VDDerivedType, "_d_" + VD->getNameAsString(), VDDerivedInit);

    // If `VD` is a reference to a local variable, then it is already
    // differentiated and should not be differentiated again.
    // If `VD` is a reference to a non-local variable then also there's no
    // need to call `Visit` since non-local variables are not differentiated.
    if (!isRefType && (!isPointerType || isInitializedByNewExpr)) {
      Expr* derivedE = nullptr;

      if (VDDerived) {
        derivedE = BuildDeclRef(VDDerived);
        if (isInitializedByNewExpr)
          derivedE = BuildOp(UnaryOperatorKind::UO_Deref, derivedE);
      }

      if (VD->getInit()) {
        llvm::SaveAndRestore<bool> saveTrackVarDecl(m_TrackVarDeclConstructor,
                                                    true);
        initDiff = Visit(VD->getInit(), derivedE);
      }

      // If we are differentiating `VarDecl` corresponding to a local variable
      // inside a loop, then we need to reset it to 0 at each iteration.
      //
      // for example, if defined inside a loop,
      // ```
      // double localVar = i;
      // ```
      // this statement should get differentiated to,
      // ```
      // {
      //   *_d_i += _d_localVar;
      //   _d_localVar = 0;
      // }
      if (VDDerived && isInsideLoop) {
        Stmt* assignToZero = nullptr;
        Expr* declRef = BuildDeclRef(VDDerived);
        if (isa<ArrayType>(VDDerivedType))
          assignToZero = GetCladZeroInit(declRef);
        else if (!isNonAggrClass)
          assignToZero = BuildOp(BinaryOperatorKind::BO_Assign, declRef,
                                 getZeroInit(VDDerivedType));
        if (!keepLocal)
          addToCurrentBlock(assignToZero, direction::reverse);
      }
    }

    VarDecl* VDClone = nullptr;
    Expr* derivedVDE = nullptr;
    if (VDDerived)
      derivedVDE = BuildDeclRef(VDDerived);
    // FIXME: Add extra parantheses if derived variable pointer is pointing to a
    // class type object.
    if (isRefType && promoteToFnScope) {
      Expr* assignDerivativeE =
          BuildOp(BinaryOperatorKind::BO_Assign, derivedVDE,
                  BuildOp(UnaryOperatorKind::UO_AddrOf, initDiff.getExpr_dx()));
      addToCurrentBlock(assignDerivativeE);
      if (isInsideLoop) {
        StmtDiff pushPop = StoreAndRestore(derivedVDE);
        if (!keepLocal)
          addToCurrentBlock(pushPop.getStmt(), direction::forward);
        m_LoopBlock.back().push_back(pushPop.getStmt_dx());
      }
      derivedVDE = BuildOp(UnaryOperatorKind::UO_Deref, derivedVDE);
    }

    // If a ref-type declaration is promoted to function global scope,
    // it's replaced with a pointer and should be initialized with the
    // address of the cloned init. e.g.
    // double& ref = x;
    // ->
    // double* ref;
    // ref = &x;
    if (isRefType && promoteToFnScope)
      VDClone = BuildGlobalVarDecl(
          VDCloneType, VD->getNameAsString(),
          BuildOp(UnaryOperatorKind::UO_AddrOf, initDiff.getExpr()),
          isDirectInit);
    else
      VDClone = BuildGlobalVarDecl(VDCloneType, VD->getNameAsString(),
                                   initDiff.getExpr(), isDirectInit);

    if (isConstructInit && VDDerived) {
      if (initDiff.getStmt_dx()) {
        SetDeclInit(VDDerived, initDiff.getExpr_dx());
      } else if (shouldCopyInitialize) {
        Expr* copyExpr = BuildDeclRef(VDClone);
        QualType origTy = VDClone->getType();
        if (isInsideLoop) {
          StmtDiff pushPop = StoreAndRestore(
              BuildDeclRef(VDDerived), /*prefix=*/"_t", /*moveToTape=*/true);
          addToCurrentBlock(pushPop.getStmt(), direction::forward);
          addToCurrentBlock(pushPop.getStmt_dx(), direction::reverse);
        }
        // if VDClone is volatile, we have to use const_cast to be able to use
        // most copy constructors.
        if (origTy.isVolatileQualified()) {
          Qualifiers quals(origTy.getQualifiers());
          quals.removeVolatile();
          QualType castTy = m_Sema.BuildQualifiedType(
              origTy.getUnqualifiedType(), noLoc, quals);
          castTy = m_Context.getLValueReferenceType(castTy);
          SourceRange range = utils::GetValidSRange(m_Sema);
          copyExpr =
              m_Sema
                  .BuildCXXNamedCast(noLoc, tok::kw_const_cast,
                                     m_Context.getTrivialTypeSourceInfo(
                                         castTy, utils::GetValidSLoc(m_Sema)),
                                     copyExpr, range, range)
                  .get();
        }
        SetDeclInit(VDDerived, copyExpr, /*DirectInit=*/true);
      }
    }

    if (isPointerType && derivedVDE) {
      if (promoteToFnScope) {
        Expr* assignDerivativeE = BuildOp(BinaryOperatorKind::BO_Assign,
                                          derivedVDE, initDiff.getExpr_dx());
        addToCurrentBlock(assignDerivativeE, direction::forward);
        if (isInsideLoop) {
          auto tape = MakeCladTapeFor(derivedVDE);
          if (!keepLocal)
            addToCurrentBlock(tape.Push);
          auto* reverseSweepDerivativePointerE =
              BuildVarDecl(derivedVDE->getType(), "_t", tape.Pop);
          m_LoopBlock.back().push_back(
              BuildDeclStmt(reverseSweepDerivativePointerE));
          derivedVDE = BuildDeclRef(reverseSweepDerivativePointerE);
        }
      } else {
        SetDeclInit(VDDerived, initDiff.getExpr_dx());
      }
    }

    if (derivedVDE)
      m_Variables.emplace(VDClone, derivedVDE);

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
    //   double _d_y = x; // copied from original function, collides with
    //   _d_y
    // }
    if ((VD->getDeclName() != VDClone->getDeclName() ||
         VDType != VDClone->getType()))
      m_DeclReplacements[VD] = VDClone;

    return DeclDiff<VarDecl>(VDClone, VDDerived);
  }

  // TODO: 'shouldEmit' parameter should be removed after converting
  // Error estimation framework to callback style. Some more research
  // need to be done to
  StmtDiff ReverseModeVisitor::DifferentiateSingleStmt(const Stmt* S,
                                                       Expr* dfdS) {
    if (m_ExternalSource)
      m_ExternalSource->ActOnStartOfDifferentiateSingleStmt();
    beginBlock(direction::reverse);
    StmtDiff SDiff = Visit(S, dfdS);

    if (m_ExternalSource)
      m_ExternalSource->ActBeforeFinalizingDifferentiateSingleStmt(direction::reverse);

    // If the statement is a standalone call to a memory function, we want to
    // add its derived statement in the same block as the original statement.
    // For ex: memset(x, 0, 10) -> memset(_d_x, 0, 10)
    Stmt* stmtDx = SDiff.getStmt_dx();
    bool dxInForward = false;
    if (auto* callExpr = dyn_cast_or_null<CallExpr>(stmtDx))
      if (auto* FD = dyn_cast<FunctionDecl>(callExpr->getCalleeDecl()))
        if (utils::IsMemoryFunction(FD))
          dxInForward = true;
    if (stmtDx) {
      if (dxInForward)
        addToCurrentBlock(stmtDx, direction::forward);
      else
        addToCurrentBlock(stmtDx, direction::reverse);
    }
    CompoundStmt* RCS = endBlock(direction::reverse);
    std::reverse(RCS->body_begin(), RCS->body_end());
    Stmt* ReverseResult = utils::unwrapIfSingleStmt(RCS);

    return StmtDiff(SDiff.getStmt(), ReverseResult);
  }

  std::pair<StmtDiff, StmtDiff>
  ReverseModeVisitor::DifferentiateSingleExpr(const Expr* E, Expr* dfdE) {
    beginBlock(direction::forward);
    beginBlock(direction::reverse);
    StmtDiff EDiff = Visit(E, dfdE);
    if (m_ExternalSource)
      m_ExternalSource->ActBeforeFinalizingDifferentiateSingleExpr(direction::reverse);
    CompoundStmt* RCS = endBlock(direction::reverse);
    Stmt* ForwardResult = endBlock(direction::forward);
    std::reverse(RCS->body_begin(), RCS->body_end());
    Stmt* ReverseResult = utils::unwrapIfSingleStmt(RCS);
    return {StmtDiff(ForwardResult, ReverseResult), EDiff};
  }

  StmtDiff ReverseModeVisitor::VisitDeclStmt(const DeclStmt* DS) {
    llvm::SmallVector<Stmt*, 16> inits;
    llvm::SmallVector<Decl*, 4> decls;
    llvm::SmallVector<Decl*, 4> declsDiff;
    llvm::SmallVector<Decl*, 4> classDeclsDiff;
    llvm::SmallVector<Stmt*, 4> memsetCalls;
    // Need to put array decls inlined.
    llvm::SmallVector<Decl*, 4> localDeclsDiff;
    // reverse_mode_forward_pass does not have a reverse pass so declarations
    // don't have to be moved to the function global scope.
    bool promoteToFnScope =
        !getCurrentScope()->isFunctionScope() &&
        m_DiffReq.Mode != DiffMode::reverse_mode_forward_pass;

    // If the DeclStmt is not empty, check the first declaration in case it is a
    // lambda function. This case it is treated separately for now and we don't
    // create a variable for its derivative.
    const auto* declsBegin = DS->decls().begin();
    if (declsBegin != DS->decls().end() && isa<VarDecl>(*declsBegin)) {
      auto* VD = dyn_cast<VarDecl>(*declsBegin);
      QualType QT = VD->getType();
      if (QT->isPointerType())
        QT = QT->getPointeeType();

      auto* typeDecl = QT->getAsCXXRecordDecl();
      // We should also simply copy the original lambda. The differentiation
      // of lambdas is happening in the `VisitCallExpr`. For now, only the
      // declarations with lambda expressions without captures are supported.
      if (typeDecl && typeDecl->isLambda()) {
        for (auto* D : DS->decls())
          if (auto* VD = dyn_cast<VarDecl>(D))
            decls.push_back(VD);
        Stmt* DSClone = BuildDeclStmt(decls);
        return StmtDiff(DSClone, nullptr);
      }
    }

    // For each variable declaration v, create another declaration _d_v to
    // store derivatives for potential reassignments. E.g.
    // double y = x;
    // ->
    // double _d_y = _d_x; double y = x;
    for (auto* D : DS->decls()) {
      if (auto* VD = dyn_cast<VarDecl>(D)) {
        DeclDiff<VarDecl> VDDiff;

        VDDiff = DifferentiateVarDecl(VD);

        // Here, we move the declaration to the function global scope.
        // Initialization is replaced with an assignment operation at the same
        // place as the original declaration. This procedure is done to make the
        // declaration visible in the reverse sweep. The variable is stored
        // before the assignment in case its value is overwritten in a loop.
        // e.g.
        // while (cond) {
        //   double x = k * n;
        // ...
        // ->
        // double x;
        // clad::tape<double> _t0 = {};
        // while (cond) {
        //   clad::push(_t0, x), x = k * n;
        // ...
        if (promoteToFnScope) {
          auto* decl = VDDiff.getDecl();
          if (VD->getInit()) {
            auto* declRef = BuildDeclRef(decl);
            auto* assignment = BuildOp(BO_Assign, declRef, decl->getInit());
            if (isInsideLoop) {
              auto pushPop = StoreAndRestore(declRef, /*prefix=*/"_t",
                                             /*moveToTape=*/true);
              if (pushPop.getExpr() != declRef)
                addToCurrentBlock(pushPop.getExpr_dx(), direction::reverse);
              assignment = BuildOp(BO_Comma, pushPop.getExpr(), assignment);
            }
            inits.push_back(assignment);
            if (const auto* AT = dyn_cast<ArrayType>(VD->getType()))
              SetDeclInit(decl, Clone(getArraySizeExpr(AT, m_Context, *this)),
                          /*DirectInit=*/true);
            else
              SetDeclInit(decl, getZeroInit(VD->getType()));
          }
        }

        decls.push_back(VDDiff.getDecl());
        if (VDDiff.getDecl_dx()) {
          const CXXRecordDecl* RD = VD->getType()->getAsCXXRecordDecl();
          bool isNonAggrClass = RD && !RD->isAggregate();
          if (isa<VariableArrayType>(VD->getType()))
            localDeclsDiff.push_back(VDDiff.getDecl_dx());
          else if (isNonAggrClass) {
            classDeclsDiff.push_back(VDDiff.getDecl_dx());
          } else {
            VarDecl* VDDerived = VDDiff.getDecl_dx();
            declsDiff.push_back(VDDerived);
            if (Stmt* memsetCall = CheckAndBuildCallToMemset(
                    BuildDeclRef(VDDerived),
                    VDDerived->getInit()->IgnoreCasts()))
              memsetCalls.push_back(memsetCall);
          }
        }
      } else if (auto* SAD = dyn_cast<StaticAssertDecl>(D)) {
        DeclDiff<StaticAssertDecl> SADDiff = DifferentiateStaticAssertDecl(SAD);
        if (SADDiff.getDecl())
          decls.push_back(SADDiff.getDecl());
        if (SADDiff.getDecl_dx())
          declsDiff.push_back(SADDiff.getDecl_dx());
      } else {
        diag(DiagnosticsEngine::Warning,
             D->getEndLoc(),
             "Unsupported declaration");
      }
    }

    Stmt* DSClone = nullptr;
    if (!decls.empty())
      DSClone = BuildDeclStmt(decls);

    if (!localDeclsDiff.empty()) {
      Stmt* localDSDIff = BuildDeclStmt(localDeclsDiff);
      addToCurrentBlock(
          localDSDIff,
          clad::rmv::forward); // Doesnt work for arrays decl'd in loops.
      for (Decl* decl : localDeclsDiff)
        if (const auto* VAT =
                dyn_cast<VariableArrayType>(cast<VarDecl>(decl)->getType())) {
          std::array<Expr*, 2> args{};
          args[0] = BuildDeclRef(cast<VarDecl>(decl));
          args[1] = Clone(VAT->getSizeExpr());
          Stmt* initCall = GetCladZeroInit(args);
          addToCurrentBlock(initCall, direction::forward);
        }
    }
    if (!declsDiff.empty()) {
      Stmt* DSDiff = BuildDeclStmt(declsDiff);
      Stmts& block =
          promoteToFnScope ? m_Globals : getCurrentBlock(direction::forward);
      addToBlock(DSDiff, block);
      for (Stmt* memset : memsetCalls)
        addToBlock(memset, block);
    }

    if (m_ExternalSource) {
      declsDiff.append(localDeclsDiff.begin(), localDeclsDiff.end());
      m_ExternalSource->ActBeforeFinalizingVisitDeclStmt(decls, declsDiff);
    }

    // This part in necessary to replace local variables inside loops
    // with function globals and replace initializations with assignments.
    if (promoteToFnScope) {
      // FIXME: We only need to produce separate decl stmts
      // because arrays promoted to the function scope are
      // turned into clad::array. This is done because of
      // mixed declarations.
      // e.g.
      // double a, b[5];
      // ->
      // double a, b(5UL);
      // when it should be
      // double a;
      // clad::array<double> b(5UL);
      // If we remove the need for clad::array here,
      // just add DSClone to the block.
      for (Decl* decl : decls)
        addToBlock(BuildDeclStmt(decl), m_Globals);
      Stmt* initAssignments = MakeCompoundStmt(inits);
      initAssignments = utils::unwrapIfSingleStmt(initAssignments);
      DSClone = initAssignments;
    }

    if (!classDeclsDiff.empty()) {
      addToCurrentBlock(DSClone, direction::forward);
      Stmts& block =
          promoteToFnScope ? m_Globals : getCurrentBlock(direction::forward);
      DSClone = nullptr;
      addToBlock(BuildDeclStmt(classDeclsDiff), block);
      for (Decl* decl : classDeclsDiff) {
        auto* vDecl = cast<VarDecl>(decl);
        Expr* init = vDecl->getInit();
        if (promoteToFnScope && init) {
          auto* declRef = BuildDeclRef(vDecl);
          auto* assignment = BuildOp(BO_Assign, declRef, init);
          addToCurrentBlock(assignment, direction::forward);
          SetDeclInit(vDecl, getZeroInit(vDecl->getType()),
                      /*DirectInit=*/true);
        }
        // Adjoints are initialized with copy-constructors only as a part of
        // the strategy to maintain the structure of the original variable.
        // In such cases, we need to zero-initialize the adjoint. e.g.
        // ```
        // std::vector<...> v{x, y, z};
        // std::vector<...> _d_v{v};
        // clad::zero_init(_d_v); // this line is generated below
        // ```
        const auto* CE = dyn_cast<CXXConstructExpr>(init->IgnoreImplicit());
        bool copyInit =
            CE && (CE->getNumArgs() == 0 ||
                   isa<DeclRefExpr>(CE->getArg(0)->IgnoreImplicit()));
        if (copyInit) {
          std::array<Expr*, 1> arg{BuildDeclRef(vDecl)};
          Stmt* initCall = GetCladZeroInit(arg);
          addToCurrentBlock(initCall, direction::forward);
        }
      }
    }

    return StmtDiff(DSClone);
  }

  StmtDiff
  ReverseModeVisitor::VisitImplicitCastExpr(const ImplicitCastExpr* ICE) {
    // Casts should be handled automatically when the result is used by
    // Sema::ActOn.../Build...
    return Visit(ICE->getSubExpr(), dfdx());
  }

  StmtDiff ReverseModeVisitor::VisitImplicitValueInitExpr(
      const ImplicitValueInitExpr* IVIE) {
    return {Clone(IVIE), Clone(IVIE)};
  }

  StmtDiff ReverseModeVisitor::VisitCStyleCastExpr(const CStyleCastExpr* CSCE) {
    StmtDiff subExprDiff = Visit(CSCE->getSubExpr(), dfdx());
    Expr* castExpr = m_Sema
                         .BuildCStyleCastExpr(
                             CSCE->getLParenLoc(), CSCE->getTypeInfoAsWritten(),
                             CSCE->getRParenLoc(), subExprDiff.getExpr())
                         .get();
    Expr* castExprDiff = subExprDiff.getExpr_dx();
    if (castExprDiff != nullptr)
      castExprDiff = m_Sema
                         .BuildCStyleCastExpr(
                             CSCE->getLParenLoc(), CSCE->getTypeInfoAsWritten(),
                             CSCE->getRParenLoc(), subExprDiff.getExpr_dx())
                         .get();
    return {castExpr, castExprDiff};
  }

  StmtDiff
  ReverseModeVisitor::VisitPseudoObjectExpr(const PseudoObjectExpr* POE) {
    // Used for CUDA Builtins
    return {Clone(POE), Clone(POE)};
  }

  StmtDiff
  ReverseModeVisitor::VisitCXXDefaultInitExpr(const CXXDefaultInitExpr* DIE) {
    return Visit(DIE->getExpr(), dfdx());
  }

  StmtDiff ReverseModeVisitor::VisitMemberExpr(const MemberExpr* ME) {
    auto baseDiff = Visit(ME->getBase());
    auto* field = ME->getMemberDecl();
    assert(!isa<CXXMethodDecl>(field) &&
           "CXXMethodDecl nodes not supported yet!");
    Expr* clonedME = baseDiff.getExpr();
    llvm::StringRef fieldName = field->getName();
    if (baseDiff.getExpr() && !fieldName.empty())
      clonedME = utils::BuildMemberExpr(m_Sema, getCurrentScope(),
                                        baseDiff.getExpr(), fieldName);
    if (clad::utils::hasNonDifferentiableAttribute(ME)) {
      auto* zero = ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context,
                                                     /*val=*/0);
      return {clonedME, zero};
    }
    if (!baseDiff.getExpr_dx())
      return {clonedME, nullptr};
    Expr* derivedME = baseDiff.getExpr_dx();
    if (!fieldName.empty())
      derivedME = utils::BuildMemberExpr(m_Sema, getCurrentScope(),
                                         baseDiff.getExpr_dx(), fieldName);
    if (dfdx() && clonedME->getType()->isRealType()) {
      Expr* addAssign =
          BuildOp(BinaryOperatorKind::BO_AddAssign, derivedME, dfdx());
      addToCurrentBlock(addAssign, direction::reverse);
    }
    return {clonedME, derivedME};
  }

  StmtDiff
  ReverseModeVisitor::VisitExprWithCleanups(const ExprWithCleanups* EWC) {
    // FIXME: We are unable to create cleanup objects currently, this can be
    // potentially problematic
    return Visit(EWC->getSubExpr(), dfdx());
  }

  /// Called in ShouldRecompute. In CUDA, to access a current thread/block id
  /// we use functions that do not change the state of any variable, since no
  /// point to store the value.
  static bool isCUDABuiltInIndex(const Expr* E) {
    const clang::Expr* B = E->IgnoreImplicit();
    if (const auto* pseudoE = llvm::dyn_cast<PseudoObjectExpr>(B)) {
      if (const auto* opaqueE =
              llvm::dyn_cast<OpaqueValueExpr>(pseudoE->getSemanticExpr(0))) {
        const Expr* innerE = opaqueE->getSourceExpr()->IgnoreImplicit();
        QualType innerT = innerE->getType();
        if (innerT.isConstQualified())
          return true;
      }
    }
    return false;
  }

  bool ReverseModeVisitor::ShouldRecompute(const Expr* E) {
    return !(utils::ContainsFunctionCalls(E) || E->HasSideEffects(m_Context)) ||
           isCUDABuiltInIndex(E);
  }

  bool ReverseModeVisitor::UsefulToStoreGlobal(Expr* E) {
    if (!E)
      return false;
    // Use stricter policy when inside loops: IsEvaluatable is also true for
    // arithmetical expressions consisting of constants, e.g. (1 + 2)*3. This
    // chech is more expensive, but it doesn't make sense to push such constants
    // into stack.
    if (isInsideLoop && E->isEvaluatable(m_Context, Expr::SE_NoSideEffects))
      return false;
    Expr* B = E->IgnoreParenImpCasts();
    // FIXME: find a more general way to determine that or add more options.
    if (isa<FloatingLiteral>(B) || isa<IntegerLiteral>(B))
      return false;
    if (isa<UnaryOperator>(B)) {
      auto* UO = cast<UnaryOperator>(B);
      auto OpKind = UO->getOpcode();
      if (OpKind == UO_Plus || OpKind == UO_Minus)
        return UsefulToStoreGlobal(UO->getSubExpr());
      return true;
    }

    // FIXME: Attach checkpointing.
    if (isa<CallExpr>(B))
      return false;

    // Assume E is useful to store.
    return true;
  }

  VarDecl* ReverseModeVisitor::GlobalStoreImpl(QualType Type,
                                               llvm::StringRef prefix,
                                               Expr* init) {
    // Create identifier before going to topmost scope
    // to let Sema::LookupName see the whole scope.
    auto* identifier = CreateUniqueIdentifier(prefix);
    // Save current scope and temporarily go to topmost function scope.
    llvm::SaveAndRestore<Scope*> SaveScope(getCurrentScope());
    assert(m_DerivativeFnScope && "must be set");
    setCurrentScope(m_DerivativeFnScope);

    VarDecl* Var = nullptr;
    if (isa<ArrayType>(Type))
      Type =
          utils::GetCladArrayOfType(m_Sema, m_Context.getBaseElementType(Type));
    Var = BuildVarDecl(Type, identifier, init);

    // Add the declaration to the body of the gradient function.
    addToBlock(BuildDeclStmt(Var), m_Globals);
    return Var;
  }

  Expr* ReverseModeVisitor::GlobalStoreAndRef(Expr* E, QualType Type,
                                              llvm::StringRef prefix,
                                              bool force) {
    assert(E && "must be provided, otherwise use DelayedGlobalStoreAndRef");
    assert(!isa<ArrayType>(Type) && "Array types cannot be stored.");
    if (!force && !UsefulToStoreGlobal(E))
      return E;

    if (isInsideLoop) {
      CladTapeResult CladTape = MakeCladTapeFor(E, prefix);
      addToCurrentBlock(CladTape.Push, direction::forward);
      addToCurrentBlock(CladTape.Pop, direction::reverse);

      return CladTape.Last();
    }

    VarDecl* VD = BuildGlobalVarDecl(Type, prefix);
    DeclStmt* decl = BuildDeclStmt(VD);
    Expr* Ref = BuildDeclRef(VD);
    bool isFnScope = getCurrentScope()->isFunctionScope() ||
                     m_DiffReq.Mode == DiffMode::reverse_mode_forward_pass;
    if (isFnScope) {
      addToCurrentBlock(decl, direction::forward);
      SetDeclInit(VD, E);
    } else {
      addToBlock(decl, m_Globals);
      Expr* Set = BuildOp(BO_Assign, Ref, E);
      addToCurrentBlock(Set, direction::forward);
    }

    return Ref;
  }

  Expr* ReverseModeVisitor::GlobalStoreAndRef(Expr* E, llvm::StringRef prefix,
                                              bool force) {
    assert(E && "cannot infer type");
    return GlobalStoreAndRef(E, utils::getNonConstType(E->getType(), m_Sema),
                             prefix, force);
  }

  StmtDiff ReverseModeVisitor::StoreAndRestore(clang::Expr* E,
                                               llvm::StringRef prefix,
                                               bool moveToTape) {
    assert(E && "must be provided");
    auto Type = utils::getNonConstType(E->getType(), m_Sema);

    if (isInsideLoop) {
      Expr* clone = Clone(E);
      if (moveToTape && E->getType()->isRecordType()) {
        llvm::SmallVector<Expr*, 1> args = {clone};
        clone = GetFunctionCall("move", "std", args);
      }
      auto CladTape = MakeCladTapeFor(clone, prefix, Type);
      Expr* Push = CladTape.Push;
      Expr* Pop = CladTape.Pop;
      auto* popAssign = BuildOp(BinaryOperatorKind::BO_Assign, Clone(E), Pop);
      return {Push, popAssign};
    }

    Expr* init = nullptr;
    if (const auto* AT = dyn_cast<ArrayType>(Type))
      init = getArraySizeExpr(AT, m_Context, *this);

    VarDecl* VD = BuildGlobalVarDecl(Type, prefix, init);
    DeclStmt* decl = BuildDeclStmt(VD);
    Expr* Ref = BuildDeclRef(VD);
    Stmt* Store = nullptr;
    bool isFnScope = getCurrentScope()->isFunctionScope() ||
                     m_DiffReq.Mode == DiffMode::reverse_mode_forward_pass;
    if (isFnScope) {
      Store = decl;
      SetDeclInit(VD, E);
    } else {
      addToBlock(decl, m_Globals);
      Store = BuildOp(BO_Assign, Ref, Clone(E));
    }

    Stmt* Restore = nullptr;
    if (E->isModifiableLvalue(m_Context) == Expr::MLV_Valid)
      Restore = BuildOp(BO_Assign, Clone(E), Ref);

    return {Store, Restore};
  }

  void ReverseModeVisitor::DelayedStoreResult::Finalize(Expr* New) {
    // Placeholders are used when we have to use an expr before we have that.
    // For instance, this is necessary for multiplication and division when the
    // RHS and LHS need the derivatives of each other to be differentiated. We
    // need placeholders to break this loop.
    class PlaceholderReplacer
        : public RecursiveASTVisitor<PlaceholderReplacer> {
    public:
      const Expr* placeholder;
      Sema& m_Sema;
      ASTContext& m_Context;
      Expr* newExpr{nullptr};
      PlaceholderReplacer(const Expr* Placeholder, Sema& S)
          : placeholder(Placeholder), m_Sema(S), m_Context(S.getASTContext()) {}

      void Replace(ReverseModeVisitor& RMV, Expr* New, StmtDiff& Result) {
        newExpr = New;
        for (Stmt* S : RMV.getCurrentBlock(direction::forward))
          TraverseStmt(S);
        for (Stmt* S : RMV.getCurrentBlock(direction::reverse))
          TraverseStmt(S);
        Result = New;
      }

      // We chose iteration rather than visiting because we only do this for
      // simple Expression subtrees and it is not worth it to implement an
      // entire visitor infrastructure for simple replacements.
      bool VisitExpr(Expr* E) const {
        for (Stmt*& S : E->children())
          if (S == placeholder) {
            // Since we are manually replacing the statement, implicit casts are
            // not generated automatically.
            ExprResult newExprRes{newExpr};
            QualType targetTy = cast<Expr>(S)->getType();
            CastKind kind = m_Sema.PrepareScalarCast(newExprRes, targetTy);
            // CK_NoOp casts trigger an assertion on debug Clang
            if (kind == CK_NoOp)
              S = newExpr;
            else
              S = m_Sema.ImpCastExprToType(newExpr, targetTy, kind).get();
          }
        return true;
      }
      PlaceholderReplacer(const PlaceholderReplacer&) = delete;
      PlaceholderReplacer(PlaceholderReplacer&&) = delete;
    };

    if (!needsUpdate)
      return;

    if (Placeholder) {
      PlaceholderReplacer repl(Placeholder, V.m_Sema);
      repl.Replace(V, New, Result);
      return;
    }

    if (isInsideLoop) {
      auto* Push = cast<CallExpr>(Result.getExpr());
      unsigned lastArg = Push->getNumArgs() - 1;
      Push->setArg(lastArg, V.m_Sema.DefaultLvalueConversion(New).get());
    } else if (isFnScope) {
      V.SetDeclInit(Declaration, New);
      V.addToCurrentBlock(V.BuildDeclStmt(Declaration), direction::forward);
    } else {
      V.addToCurrentBlock(V.BuildOp(BO_Assign, Result.getExpr(), New),
                          direction::forward);
    }
  }

  ReverseModeVisitor::DelayedStoreResult
  ReverseModeVisitor::DelayedGlobalStoreAndRef(Expr* E, llvm::StringRef prefix,
                                               bool forceStore) {
    assert(E && "must be provided");
    if (!UsefulToStore(E)) {
      StmtDiff Ediff = Visit(E);
      Expr::EvalResult evalRes;
      return DelayedStoreResult{*this, Ediff,
                                /*Declaration=*/nullptr,
                                /*isInsideLoop=*/false,
                                /*isFnScope=*/false};
    }
    if (!forceStore && ShouldRecompute(E)) {
      // The value of the literal has no. It's given a very particular value for
      // easier debugging.
      Expr* PH = ConstantFolder::synthesizeLiteral(E->getType(), m_Context,
                                                   /*val=*/~0U);
      return DelayedStoreResult{*this,
                                StmtDiff{PH, /*diff=*/nullptr, PH},
                                /*Declaration=*/nullptr,
                                /*isInsideLoop=*/false,
                                /*isFnScope=*/false,
                                /*pNeedsUpdate=*/true,
                                /*pPlaceholder=*/PH};
    }
    if (isInsideLoop) {
      Expr* dummy = E;
      auto CladTape = MakeCladTapeFor(dummy);
      Expr* Push = CladTape.Push;
      Expr* Pop = CladTape.Pop;
      return DelayedStoreResult{*this,
                                StmtDiff{Push, nullptr, Pop},
                                /*Declaration=*/nullptr,
                                /*isInsideLoop=*/true,
                                /*isFnScope=*/false,
                                /*pNeedsUpdate=*/true};
    }
    bool isFnScope = getCurrentScope()->isFunctionScope() ||
                     m_DiffReq.Mode == DiffMode::reverse_mode_forward_pass;
    VarDecl* VD = BuildGlobalVarDecl(
        utils::getNonConstType(E->getType(), m_Sema), prefix);
    Expr* Ref = BuildDeclRef(VD);
    if (!isFnScope)
      addToBlock(BuildDeclStmt(VD), m_Globals);
    // Return reference to the declaration instead of original expression.
    return DelayedStoreResult{*this,
                              StmtDiff{Ref, nullptr, Ref},
                              /*Declaration=*/VD,
                              /*isInsideLoop=*/false,
                              /*isFnScope=*/isFnScope,
                              /*pNeedsUpdate=*/true};
  }

  ReverseModeVisitor::LoopCounter::LoopCounter(ReverseModeVisitor& RMV)
      : m_RMV(RMV) {
    ASTContext& C = m_RMV.m_Context;
    Expr* zero = ConstantFolder::synthesizeLiteral(C.getSizeType(), C,
                                                   /*val=*/0);
    m_Ref = m_RMV.GlobalStoreAndRef(zero, C.getSizeType(), "_t",
                                    /*force=*/true);
  }

  StmtDiff ReverseModeVisitor::VisitWhileStmt(const WhileStmt* WS) {
    beginBlock(direction::reverse);
    LoopCounter loopCounter(*this);

    // begin scope for while statement
    beginScope(Scope::DeclScope | Scope::ControlScope | Scope::BreakScope |
               Scope::ContinueScope);

    llvm::SaveAndRestore<bool> SaveIsInsideLoop(isInsideLoop);
    isInsideLoop = true;
    llvm::SaveAndRestore<Expr*> SaveCurrentBreakFlagExpr(
        m_CurrentBreakFlagExpr);
    m_CurrentBreakFlagExpr = nullptr;

    Expr* condClone = (WS->getCond() ? Clone(WS->getCond()) : nullptr);
    const VarDecl* condVarDecl = WS->getConditionVariable();
    StmtDiff condVarRes;
    if (condVarDecl)
      condVarRes = DifferentiateSingleStmt(WS->getConditionVariableDeclStmt());

    // compute condition result object for the forward pass `while`
    // statement.
    Sema::ConditionResult condResult;
    if (condVarDecl) {
      if (condVarRes.getStmt()) {
        if (isa<DeclStmt>(condVarRes.getStmt())) {
          Decl* condVarClone =
              cast<DeclStmt>(condVarRes.getStmt())->getSingleDecl();
          condResult = m_Sema.ActOnConditionVariable(
              condVarClone, noLoc, Sema::ConditionKind::Boolean);
        } else {
          condResult = m_Sema.ActOnCondition(getCurrentScope(), noLoc,
                                             cast<Expr>(condVarRes.getStmt()),
                                             Sema::ConditionKind::Boolean);
        }
      }
    } else {
      condResult = m_Sema.ActOnCondition(getCurrentScope(), noLoc, condClone,
                                         Sema::ConditionKind::Boolean);
    }

    const Stmt* body = WS->getBody();
    StmtDiff bodyDiff = DifferentiateLoopBody(body, loopCounter,
                                              condVarRes.getStmt_dx());
    // Create forward-pass `while` loop.
    Stmt* forwardWS = clad_compat::Sema_ActOnWhileStmt(m_Sema, condResult,
                                                       bodyDiff.getStmt())
                          .get();

    // Create reverse-pass `while` loop.
    Sema::ConditionResult CounterCondition = loopCounter
                                                 .getCounterConditionResult();
    Stmt* reverseWS = clad_compat::Sema_ActOnWhileStmt(m_Sema, CounterCondition,
                                                       bodyDiff.getStmt_dx())
                          .get();
    // for while statement
    endScope();
    addToCurrentBlock(reverseWS, direction::reverse);
    reverseWS = utils::unwrapIfSingleStmt(endBlock(direction::reverse));
    return {forwardWS, reverseWS};
  }

  StmtDiff ReverseModeVisitor::VisitDoStmt(const DoStmt* DS) {
    beginBlock(direction::reverse);
    LoopCounter loopCounter(*this);

    // begin scope for do statement
    beginScope(Scope::ContinueScope | Scope::BreakScope);

    llvm::SaveAndRestore<bool> SaveIsInsideLoop(isInsideLoop);
    isInsideLoop = true;
    llvm::SaveAndRestore<Expr*> SaveCurrentBreakFlagExpr(
        m_CurrentBreakFlagExpr);
    m_CurrentBreakFlagExpr = nullptr;

    Expr* clonedCond = (DS->getCond() ? Clone(DS->getCond()) : nullptr);

    const Stmt* body = DS->getBody();
    StmtDiff bodyDiff = DifferentiateLoopBody(body, loopCounter);

    // Create forward-pass `do-while` statement.
    Stmt* forwardDS = m_Sema
                          .ActOnDoStmt(/*DoLoc=*/noLoc, bodyDiff.getStmt(),
                                       /*WhileLoc=*/noLoc,
                                       /*CondLParen=*/noLoc, clonedCond,
                                       /*CondRParen=*/noLoc)
                          .get();

    // create reverse-pass `do-while` statement.
    Expr*
        counterCondition = loopCounter.getCounterConditionResult().get().second;
    Stmt* reverseDS = m_Sema
                          .ActOnDoStmt(/*DoLoc=*/noLoc, bodyDiff.getStmt_dx(),
                                       /*WhileLoc=*/noLoc,
                                       /*CondLParen=*/noLoc, counterCondition,
                                       /*CondRParen=*/noLoc)
                          .get();
    // for do-while statement
    endScope();
    addToCurrentBlock(reverseDS, direction::reverse);
    reverseDS = utils::unwrapIfSingleStmt(endBlock(direction::reverse));
    return {forwardDS, reverseDS};
  }

  // Basic idea used for differentiating switch statement is that in the reverse
  // pass, processing of the differentiated statments of the switch statement
  // body should start either from a `break` statement or from the last
  // statement of the switch statement body and always end at a switch
  // case/default statement.
  //
  // Therefore, here we keep track of which `break` was hit in the forward pass,
  // or if we no `break` statement was hit at all in a variable or clad tape.
  // This information is further used by an auxilliary switch statement in the
  // reverse pass to jump the execution to the correct point (that is,
  // differentiated statement of the statement just before the `break` statement
  // that was hit in the forward pass)
  StmtDiff ReverseModeVisitor::VisitSwitchStmt(const SwitchStmt* SS) {
    // Scope and blocks for the compound statement that encloses the switch
    // statement in both the forward and the reverse pass. Block is required
    // for handling condition variable and switch-init statement.
    beginScope(Scope::DeclScope);
    beginBlock(direction::forward);
    beginBlock(direction::reverse);

    // Handles switch init statement
    if (SS->getInit()) {
      StmtDiff switchInitDiff = DifferentiateSingleStmt(SS->getInit());
      addToCurrentBlock(switchInitDiff.getStmt(), direction::forward);
      addToCurrentBlock(switchInitDiff.getStmt_dx(), direction::reverse);
    }

    // Handles condition variable
    if (SS->getConditionVariable()) {
      StmtDiff condVarDiff =
          DifferentiateSingleStmt(SS->getConditionVariableDeclStmt());
      addToCurrentBlock(condVarDiff.getStmt(), direction::forward);
      addToCurrentBlock(condVarDiff.getStmt_dx(), direction::reverse);
    }

    StmtDiff condDiff = DifferentiateSingleStmt(SS->getCond());
    addToCurrentBlock(condDiff.getStmt(), direction::forward);
    addToCurrentBlock(condDiff.getStmt_dx(), direction::reverse);
    Expr* condExpr = GlobalStoreAndRef(condDiff.getExpr(), "_cond");

    auto* activeBreakContHandler = PushBreakContStmtHandler(
        /*forSwitchStmt=*/true);
    activeBreakContHandler->BeginCFSwitchStmtScope();
    auto* SSData = PushSwitchStmtInfo();

    SSData->switchStmtCond = condExpr;

    // scope for the switch statement body.
    beginScope(Scope::DeclScope);

    const Stmt* body = SS->getBody();
    StmtDiff bodyDiff = nullptr;
    if (isa<CompoundStmt>(body))
      bodyDiff = Visit(body);
    else
      bodyDiff = DifferentiateSingleStmt(body);

    // Each switch case statement of the original function gets transformed to
    // an if condition in the reverse pass. The if condition decides at runtime
    // whether the processing of the differentiated statements of the switch
    // statement body should stop or continue. This is based on the fact that
    // processing of statements of switch statement body always starts at a case
    // statement. For example,
    // ```
    // case 3:
    // ```
    // gets transformed to,
    //
    // ```
    // if (3 == _cond)
    //   break;
    // ```
    //
    // This kind of if expression cannot by easily formed for the default
    // statement, thus, we instead compare value of the switch condition with
    // the values of all the case statements to determine if the default
    // statement was selected in the forward pass.
    // Therefore,
    //
    // ```
    // default:
    // ```
    //
    // will get transformed to something like,
    //
    // ```
    // if (_cond != 1 && _cond != 2 && _cond != 3)
    //   break;
    // ```
    if (SSData->defaultIfBreakExpr) {
      Expr* breakCond = nullptr;
      for (auto* SC : SSData->cases) {
        if (auto* CS = dyn_cast<CaseStmt>(SC)) {
          if (breakCond) {
            breakCond = BuildOp(BinaryOperatorKind::BO_LAnd, breakCond,
                                BuildOp(BinaryOperatorKind::BO_NE,
                                        SSData->switchStmtCond, CS->getLHS()));
          } else {
            breakCond = BuildOp(BinaryOperatorKind::BO_NE,
                                SSData->switchStmtCond, CS->getLHS());
          }
        }
      }
      if (!breakCond)
        breakCond = m_Sema.ActOnCXXBoolLiteral(noLoc, tok::kw_true).get();
      SSData->defaultIfBreakExpr->setCond(breakCond);
    }

    activeBreakContHandler->EndCFSwitchStmtScope();

    // If switch statement contains no cases, then, no statement of the switch
    // statement body will be processed in both the forward and the reverse
    // pass. Thus, we do not need to add them in the differentiated function.
    if (!(SSData->cases.empty())) {
      Sema::ConditionResult condRes = m_Sema.ActOnCondition(
          getCurrentScope(), noLoc, condExpr, Sema::ConditionKind::Switch);
      SwitchStmt* forwardSS =
          clad_compat::Sema_ActOnStartOfSwitchStmt(m_Sema, nullptr, condRes)
              .getAs<SwitchStmt>();
      activeBreakContHandler->UpdateForwAndRevBlocks(bodyDiff);

      // Registers all the cases to the switch statement.
      for (auto* SC : SSData->cases)
        forwardSS->addSwitchCase(SC);

      forwardSS =
          m_Sema.ActOnFinishSwitchStmt(noLoc, forwardSS, bodyDiff.getStmt())
              .getAs<SwitchStmt>();

      addToCurrentBlock(forwardSS, direction::forward);
      addToCurrentBlock(bodyDiff.getStmt_dx(), direction::reverse);
    }

    PopBreakContStmtHandler();
    PopSwitchStmtInfo();
    return {endBlock(direction::forward), endBlock(direction::reverse)};
  }

  StmtDiff ReverseModeVisitor::VisitCaseStmt(const CaseStmt* CS) {
    beginBlock(direction::forward);
    beginBlock(direction::reverse);
    SwitchStmtInfo* SSData = GetActiveSwitchStmtInfo();

    Expr* lhsClone = (CS->getLHS() ? Clone(CS->getLHS()) : nullptr);
    Expr* rhsClone = (CS->getRHS() ? Clone(CS->getRHS()) : nullptr);

    auto* newSC = CaseStmt::Create(m_Sema.getASTContext(), lhsClone, rhsClone,
                                   noLoc, noLoc, noLoc);

    Expr* ifCond = BuildOp(BinaryOperatorKind::BO_EQ, newSC->getLHS(),
                           SSData->switchStmtCond);
    Stmt* ifThen = m_Sema.ActOnBreakStmt(noLoc, getCurrentScope()).get();
    Stmt* ifBreakExpr = clad_compat::IfStmt_Create(
        m_Context, noLoc, false, nullptr, nullptr, ifCond, noLoc, noLoc, ifThen,
        noLoc, nullptr);
    SSData->cases.push_back(newSC);
    addToCurrentBlock(ifBreakExpr, direction::reverse);
    addToCurrentBlock(newSC, direction::forward);
    auto diff = DifferentiateSingleStmt(CS->getSubStmt());
    utils::SetSwitchCaseSubStmt(newSC, diff.getStmt());
    addToCurrentBlock(diff.getStmt_dx(), direction::reverse);
    return {endBlock(direction::forward), endBlock(direction::reverse)};
  }

  StmtDiff ReverseModeVisitor::VisitDefaultStmt(const DefaultStmt* DS) {
    beginBlock(direction::reverse);
    beginBlock(direction::forward);
    auto* SSData = GetActiveSwitchStmtInfo();
    auto* newDefaultStmt =
        new (m_Sema.getASTContext()) DefaultStmt(noLoc, noLoc, nullptr);
    Stmt* ifThen = m_Sema.ActOnBreakStmt(noLoc, getCurrentScope()).get();
    Stmt* ifBreakExpr = clad_compat::IfStmt_Create(
        m_Context, noLoc, false, nullptr, nullptr, nullptr, noLoc, noLoc,
        ifThen, noLoc, nullptr);
    SSData->cases.push_back(newDefaultStmt);
    SSData->defaultIfBreakExpr = cast<IfStmt>(ifBreakExpr);
    addToCurrentBlock(ifBreakExpr, direction::reverse);
    addToCurrentBlock(newDefaultStmt, direction::forward);
    auto diff = DifferentiateSingleStmt(DS->getSubStmt());
    utils::SetSwitchCaseSubStmt(newDefaultStmt, diff.getStmt());
    addToCurrentBlock(diff.getStmt_dx(), direction::reverse);
    return {endBlock(direction::forward), endBlock(direction::reverse)};
  }

  StmtDiff ReverseModeVisitor::DifferentiateLoopBody(const Stmt* body,
                                                     LoopCounter& loopCounter,
                                                     Stmt* condVarDiff,
                                                     Stmt* forLoopIncDiff,
                                                     bool isForLoop) {
    Expr* counterIncrement = loopCounter.getCounterIncrement();
    auto* activeBreakContHandler = PushBreakContStmtHandler();
    activeBreakContHandler->BeginCFSwitchStmtScope();
    m_LoopBlock.emplace_back();
    // differentiate loop body and add loop increment expression
    // in the forward block.
    StmtDiff bodyDiff = nullptr;
    if (isa<CompoundStmt>(body)) {
      bodyDiff = Visit(body);
      beginBlock(direction::forward);
      addToCurrentBlock(counterIncrement);
      for (Stmt* S : cast<CompoundStmt>(bodyDiff.getStmt())->body())
        addToCurrentBlock(S);
      bodyDiff = {endBlock(direction::forward), bodyDiff.getStmt_dx()};
    } else {
      // for forward-pass loop statement body
      beginScope(Scope::DeclScope);
      beginBlock(direction::forward);
      addToCurrentBlock(counterIncrement);
      if (m_ExternalSource)
        m_ExternalSource->ActBeforeDifferentiatingSingleStmtLoopBody();
      bodyDiff = DifferentiateSingleStmt(body, /*dfdS=*/nullptr);
      addToCurrentBlock(bodyDiff.getStmt());
      if (m_ExternalSource)
        m_ExternalSource->ActAfterProcessingSingleStmtBodyInVisitForLoop();

      Stmt* reverseBlock = utils::unwrapIfSingleStmt(bodyDiff.getStmt_dx());
      bodyDiff = {endBlock(direction::forward), reverseBlock};
      // for forward-pass loop statement body
      endScope();
    }
    Stmts revLoopBlock = m_LoopBlock.back();
    utils::AppendIndividualStmts(revLoopBlock, bodyDiff.getStmt_dx());
    if (!revLoopBlock.empty())
      bodyDiff.updateStmtDx(MakeCompoundStmt(revLoopBlock));
    m_LoopBlock.pop_back();

    activeBreakContHandler->EndCFSwitchStmtScope();
    activeBreakContHandler->UpdateForwAndRevBlocks(bodyDiff);
    PopBreakContStmtHandler();

    Expr* revCounter = loopCounter.getCounterConditionResult().get().second;
    if (m_CurrentBreakFlagExpr) {
      VarDecl* numRevIterations = BuildVarDecl(m_Context.getSizeType(),
                                               "_numRevIterations", revCounter);
      loopCounter.setNumRevIterations(numRevIterations);
    }

    // Increment statement in the for-loop is executed for every case
    if (forLoopIncDiff) {
      Stmt* forLoopIncDiffExpr = forLoopIncDiff;
      if (m_CurrentBreakFlagExpr) {
        m_CurrentBreakFlagExpr =
            BuildOp(BinaryOperatorKind::BO_LOr,
                    BuildOp(BinaryOperatorKind::BO_NE, revCounter,
                            BuildDeclRef(loopCounter.getNumRevIterations())),
                    BuildParens(m_CurrentBreakFlagExpr));
        forLoopIncDiffExpr = clad_compat::IfStmt_Create(
            m_Context, noLoc, false, nullptr, nullptr, m_CurrentBreakFlagExpr,
            noLoc, noLoc, forLoopIncDiff, noLoc, nullptr);
      }
      if (bodyDiff.getStmt_dx()) {
        bodyDiff.updateStmtDx(utils::PrependAndCreateCompoundStmt(
            m_Context, bodyDiff.getStmt_dx(), forLoopIncDiffExpr));
      } else {
        bodyDiff.updateStmtDx(forLoopIncDiffExpr);
      }
    }

    Expr* counterDecrement = loopCounter.getCounterDecrement();

    // Create reverse pass loop body statements by arranging various
    // differentiated statements in the correct order.
    // Order used:
    //
    // 1) `for` loop increment differentiation statements
    // 2) loop body differentiation statements
    // 3) condition variable differentiation statements
    // 4) counter decrement expression
    beginBlock(direction::reverse);
    // `for` loops have counter decrement expression in the
    // loop iteration-expression.
    if (!isForLoop)
      addToCurrentBlock(counterDecrement, direction::reverse);
    addToCurrentBlock(condVarDiff, direction::reverse);
    addToCurrentBlock(bodyDiff.getStmt_dx(), direction::reverse);
    bodyDiff = {bodyDiff.getStmt(),
                utils::unwrapIfSingleStmt(endBlock(direction::reverse))};
    return bodyDiff;
  }

  StmtDiff ReverseModeVisitor::VisitContinueStmt(const ContinueStmt* CS) {
    beginBlock(direction::forward);
    Stmt* newCS = m_Sema.ActOnContinueStmt(noLoc, getCurrentScope()).get();
    auto* activeBreakContHandler = GetActiveBreakContStmtHandler();
    Stmt* CFCaseStmt = activeBreakContHandler->GetNextCFCaseStmt();
    Stmt* pushExprToCurrentCase = activeBreakContHandler
                                      ->CreateCFTapePushExprToCurrentCase();
    addToCurrentBlock(pushExprToCurrentCase);
    addToCurrentBlock(newCS);
    return {endBlock(direction::forward), CFCaseStmt};
  }

  StmtDiff ReverseModeVisitor::VisitBreakStmt(const BreakStmt* BS) {
    beginBlock(direction::forward);
    Stmt* newBS = m_Sema.ActOnBreakStmt(noLoc, getCurrentScope()).get();
    auto* activeBreakContHandler = GetActiveBreakContStmtHandler();
    Stmt* CFCaseStmt = activeBreakContHandler->GetNextCFCaseStmt();
    Stmt* pushExprToCurrentCase = activeBreakContHandler
                                      ->CreateCFTapePushExprToCurrentCase();
    if (isInsideLoop && !activeBreakContHandler->m_IsInvokedBySwitchStmt) {
      Expr* tapeBackExprForCurrentCase =
          activeBreakContHandler->CreateCFTapeBackExprForCurrentCase();
      if (m_CurrentBreakFlagExpr) {
        m_CurrentBreakFlagExpr =
            BuildOp(BinaryOperatorKind::BO_LAnd, m_CurrentBreakFlagExpr,
                    tapeBackExprForCurrentCase);
      } else {
        m_CurrentBreakFlagExpr = tapeBackExprForCurrentCase;
      }
    }
    addToCurrentBlock(pushExprToCurrentCase);
    addToCurrentBlock(newBS);
    return {endBlock(direction::forward), CFCaseStmt};
  }

  Expr* ReverseModeVisitor::BreakContStmtHandler::CreateSizeTLiteralExpr(
      std::size_t value) {
    ASTContext& C = m_RMV.m_Context;
    auto* literalExpr =
        ConstantFolder::synthesizeLiteral(C.getSizeType(), C, value);
    return literalExpr;
  }

  void ReverseModeVisitor::BreakContStmtHandler::InitializeCFTape() {
    assert(!m_ControlFlowTape && "InitializeCFTape() should not be called if "
                                 "m_ControlFlowTape is already initialized");

    auto* zeroLiteral = CreateSizeTLiteralExpr(0);
    m_ControlFlowTape.reset(
        new CladTapeResult(m_RMV.MakeCladTapeFor(zeroLiteral)));
  }

  Expr* ReverseModeVisitor::BreakContStmtHandler::CreateCFTapePushExpr(
      std::size_t value) {
    Expr* pushDRE = m_RMV.GetCladTapePushDRE();
    Expr* callArgs[] = {m_ControlFlowTape->Ref, CreateSizeTLiteralExpr(value)};
    Expr* pushExpr = m_RMV.m_Sema
                         .ActOnCallExpr(m_RMV.getCurrentScope(), pushDRE, noLoc,
                                        callArgs, noLoc)
                         .get();
    return pushExpr;
  }

  void
  ReverseModeVisitor::BreakContStmtHandler::BeginCFSwitchStmtScope() const {
    m_RMV.beginScope(Scope::SwitchScope | Scope::ControlScope |
                     Scope::BreakScope | Scope::DeclScope);
  }

  void ReverseModeVisitor::BreakContStmtHandler::EndCFSwitchStmtScope() const {
    m_RMV.endScope();
  }

  CaseStmt* ReverseModeVisitor::BreakContStmtHandler::GetNextCFCaseStmt() {
    ++m_CaseCounter;
    auto* counterLiteral = CreateSizeTLiteralExpr(m_CaseCounter);
    CaseStmt* CS = CaseStmt::Create(m_RMV.m_Context, counterLiteral, nullptr,
                                    noLoc, noLoc, noLoc);

    // Initialise switch case statements with null statement because it is
    // necessary for switch case statements to have a substatement but it
    // is possible that there are no statements after the corresponding
    // break/continue statement. It's also easier to just set null statement
    // as substatement instead of keeping track of switch cases and
    // corresponding next statements.
    CS->setSubStmt(m_RMV.m_Sema.ActOnNullStmt(noLoc).get());

    m_SwitchCases.push_back(CS);
    return CS;
  }

  Expr* ReverseModeVisitor::BreakContStmtHandler::
      CreateCFTapeBackExprForCurrentCase() {
    return m_RMV.BuildOp(
        BinaryOperatorKind::BO_NE, m_ControlFlowTape->Last(),
        ConstantFolder::synthesizeLiteral(m_RMV.m_Context.IntTy,
                                          m_RMV.m_Context, m_CaseCounter));
  }

  Stmt* ReverseModeVisitor::BreakContStmtHandler::
      CreateCFTapePushExprToCurrentCase() {
    if (!m_ControlFlowTape)
      InitializeCFTape();
    return CreateCFTapePushExpr(m_CaseCounter);
  }

  void ReverseModeVisitor::BreakContStmtHandler::UpdateForwAndRevBlocks(
      StmtDiff& bodyDiff) {
    if (m_SwitchCases.empty() && !m_IsInvokedBySwitchStmt)
      return;

    // Add case statement in the beginning of the reverse block
    // and corresponding push expression for this case statement
    // at the end of the forward block to cover the case when no
    // `break`/`continue` statements are hit.
    auto* lastSC = GetNextCFCaseStmt();
    auto* pushExprToCurrentCase = CreateCFTapePushExprToCurrentCase();

    Stmt* forwBlock = nullptr;
    Stmt* revBlock = nullptr;

    forwBlock = utils::AppendAndCreateCompoundStmt(m_RMV.m_Context,
                                                   bodyDiff.getStmt(),
                                                   pushExprToCurrentCase);
    revBlock = utils::PrependAndCreateCompoundStmt(m_RMV.m_Context,
                                                   bodyDiff.getStmt_dx(),
                                                   lastSC);

    bodyDiff = {forwBlock, revBlock};

    auto condResult = m_RMV.m_Sema.ActOnCondition(m_RMV.getCurrentScope(),
                                                  noLoc, m_ControlFlowTape->Pop,
                                                  Sema::ConditionKind::Switch);
    auto* CFSS = clad_compat::Sema_ActOnStartOfSwitchStmt(m_RMV.m_Sema, nullptr,
                                                          condResult)
                     .getAs<SwitchStmt>();
    // Registers all the switch cases
    for (auto* SC : m_SwitchCases)
      CFSS->addSwitchCase(SC);
    m_RMV.m_Sema.ActOnFinishSwitchStmt(noLoc, CFSS, bodyDiff.getStmt_dx());

    bodyDiff = {bodyDiff.getStmt(), CFSS};
  }

  void ReverseModeVisitor::AddExternalSource(ExternalRMVSource& source) {
    if (!m_ExternalSource)
      m_ExternalSource = new MultiplexExternalRMVSource();
    source.InitialiseRMV(*this);
    m_ExternalSource->AddSource(source);
  }

  StmtDiff ReverseModeVisitor::VisitCXXThisExpr(const CXXThisExpr* CTE) {
    Expr* clonedCTE = nullptr;
    if (!isa<CXXConstructorDecl>(m_DiffReq.Function)) {
      clonedCTE = Clone(CTE);
    } else {
      // In constructor pullbacks, `this` is not taken as a parameter
      // and is built in the pullback body. Perform a lookup.
      IdentifierInfo* name = &m_Context.Idents.get("_this");
      LookupResult R(m_Sema, DeclarationName(name), noLoc,
                     Sema::LookupOrdinaryName);
      m_Sema.LookupName(R, getCurrentScope(), /*AllowBuiltinCreation*/ false);
      assert(!R.empty() && "_this was not found.");
      auto* thisDecl = cast<VarDecl>(R.getFoundDecl());
      clonedCTE = BuildDeclRef(thisDecl);
    }
    return {clonedCTE, m_ThisExprDerivative};
  }

  StmtDiff ReverseModeVisitor::VisitCXXTemporaryObjectExpr(
      const clang::CXXTemporaryObjectExpr* TOE) {
    return Clone(TOE);
  }

  StmtDiff ReverseModeVisitor::VisitCXXBindTemporaryExpr(
      const clang::CXXBindTemporaryExpr* BTE) {
    return Visit(BTE->getSubExpr(), dfdx());
  }

  StmtDiff ReverseModeVisitor::VisitCXXNewExpr(const clang::CXXNewExpr* CNE) {
    StmtDiff initializerDiff;
    if (CNE->hasInitializer())
      initializerDiff = Visit(CNE->getInitializer(), dfdx());

    Expr* clonedArraySizeE = nullptr;
    Expr* derivedArraySizeE = nullptr;
    if (CNE->getArraySize()) {
      clonedArraySizeE =
          Visit(clad_compat::ArraySize_GetValue(CNE->getArraySize())).getExpr();
      // Array size is a non-differentiable expression, thus the original value
      // should be used in both the cloned and the derived statements.
      derivedArraySizeE = Clone(clonedArraySizeE);
    }
    Expr* clonedNewE = utils::BuildCXXNewExpr(
        m_Sema, CNE->getAllocatedType(), clonedArraySizeE,
        initializerDiff.getExpr(), CNE->getAllocatedTypeSourceInfo());
    Expr* diffInit = initializerDiff.getExpr_dx();
    if (!diffInit) {
      // we should initialize it implicitly using ParenListExpr.
      diffInit = m_Sema.ActOnParenListExpr(noLoc, noLoc, {}).get();
    }
    Expr* derivedNewE = utils::BuildCXXNewExpr(
        m_Sema, CNE->getAllocatedType(), derivedArraySizeE, diffInit,
        CNE->getAllocatedTypeSourceInfo());
    return {clonedNewE, derivedNewE};
  }

  StmtDiff
  ReverseModeVisitor::VisitCXXDeleteExpr(const clang::CXXDeleteExpr* CDE) {
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
    // create a compound statement containing both the cloned and the derived
    // delete expressions.
    CompoundStmt* CS = MakeCompoundStmt({clonedDeleteE, derivedDeleteE});
    m_DeallocExprs.push_back(CS);
    return {nullptr, nullptr};
  }

  static bool isNAT(QualType T) {
    T = utils::GetValueType(T);
    if (const auto* RT = T->getAs<RecordType>()) {
      const RecordDecl* RD = RT->getDecl();
      if (RD->getNameAsString() == "__nat")
        return true;
    }
    return false;
  }

  StmtDiff
  ReverseModeVisitor::VisitCXXConstructExpr(const CXXConstructExpr* CE) {
    CXXConstructorDecl* CD = CE->getConstructor();
    llvm::SmallVector<Expr*, 4> primalArgs;
    llvm::SmallVector<Expr*, 4> adjointArgs;
    llvm::SmallVector<Expr*, 4> reverseForwAdjointArgs;
    // It is used to store '_r0' temporary gradient variables that are used for
    // differentiating non-reference args.
    llvm::SmallVector<Stmt*, 4> prePullbackCallStmts;

    // Insertion point is required because we need to insert pullback call
    // before the statements inserted by 'Visit(arg, ...)' calls for arguments.
    std::size_t insertionPoint = getCurrentBlock(direction::reverse).size();

    // FIXME: consider moving non-diff analysis to DiffPlanner.
    bool nonDiff = clad::utils::hasNonDifferentiableAttribute(CE);

    // If the result does not depend on the result of the call, just clone
    // the call and visit arguments (since they may contain side-effects like
    // f(x = y))
    // If the callee function takes arguments by reference then it can affect
    // derivatives even if there is no `dfdx()` and thus we should call the
    // derived function.
    if (!nonDiff && !dfdx())
      nonDiff = true;

    // If all arguments are constant literals, then this does not contribute to
    // the gradient.
    if (!nonDiff) {
      nonDiff = true;
      for (const Expr* arg : CE->arguments()) {
        if (m_DiffReq.isVaried(arg)) {
          nonDiff = false;
          break;
        }
      }
    }

    // FIXME: This logic is the same as in VisitCallExpr.
    // We should probably move this to a file static
    // FIXME: Restore arguments passed as non-const reference.
    for (std::size_t i = 0, e = CE->getNumArgs(); i != e; ++i) {
      const Expr* arg = CE->getArg(i);
      QualType ArgTy = arg->getType();
      // FIXME: We handle parameters with default values by setting them
      // explicitly. However, some of them have private types and cannot be set.
      // For this reason, we ignore std::__nat. We need to come up with a
      // general solution.
      if (isNAT(ArgTy))
        break;
      StmtDiff argDiff{};
      Expr* adjointArg = nullptr;
      if (utils::IsReferenceOrPointerArg(arg) || nonDiff) {
        argDiff = Visit(arg);
        adjointArg = argDiff.getExpr_dx();
      } else {
        // non-reference arguments are differentiated as follows:
        //
        // primal code:
        // ```
        // SomeClass c(u, ...);
        // ```
        //
        // Derivative code:
        // ```
        // // forward pass
        // ...
        // // reverse pass
        // double _r0 = 0;
        // SomeClass_pullback(c, u, ..., &_d_c, &_r0, ...);
        // _d_u += _r0;
        QualType dArgTy = utils::getNonConstType(CloneType(ArgTy), m_Sema);
        Expr* init = getStdInitListSizeExpr(arg);
        bool shouldCopyInitialize = false;
        if (!init) {
          if (const CXXRecordDecl* CRD = dArgTy->getAsCXXRecordDecl())
            shouldCopyInitialize = utils::isCopyable(CRD);
          // Temporarily initialize the object with `*nullptr` to avoid
          // a potential error because of non-existing default constructor.
          if (shouldCopyInitialize) {
            QualType ptrType =
                m_Context.getPointerType(dArgTy.getUnqualifiedType());
            Expr* dummy = getZeroInit(ptrType);
            init = BuildOp(UO_Deref, dummy);
          }
        }
        if (!init)
          init = getZeroInit(dArgTy);
        VarDecl* dArgDecl = BuildVarDecl(dArgTy, "_r", init);
        prePullbackCallStmts.push_back(BuildDeclStmt(dArgDecl));
        adjointArg = BuildDeclRef(dArgDecl);
        argDiff = Visit(arg, BuildDeclRef(dArgDecl));
        if (shouldCopyInitialize) {
          if (Expr* dInit = argDiff.getExpr_dx())
            SetDeclInit(dArgDecl, dInit);
          else
            SetDeclInit(dArgDecl, getZeroInit(dArgTy));
        }
      }

      if (utils::isArrayOrPointerType(CD->getParamDecl(i)->getType()) ||
          nonDiff) {
        reverseForwAdjointArgs.push_back(adjointArg);
        adjointArgs.push_back(adjointArg);
      } else {
        if (argDiff.getExpr_dx())
          reverseForwAdjointArgs.push_back(argDiff.getExpr_dx());
        else
          reverseForwAdjointArgs.push_back(getZeroInit(ArgTy));
        adjointArgs.push_back(BuildOp(UnaryOperatorKind::UO_AddrOf, adjointArg,
                                      m_DiffReq->getLocation()));
      }
      // If a function returns an object by value, there
      // are an implicit move constructor and an implicit
      // cast to XValue. However, when providing arguments,
      // we have to cast explicitly with std::move.
      if (arg->isXValue() && argDiff.getExpr()->isLValue()) {
        llvm::SmallVector<Expr*, 1> moveArg = {argDiff.getExpr()};
        Expr* moveCall = GetFunctionCall("move", "std", moveArg);
        primalArgs.push_back(moveCall);
      } else {
        primalArgs.push_back(argDiff.getExpr());
      }
    }

    const CXXRecordDecl* RD = CD->getParent();

    if (!nonDiff) {
      // Try to create a pullback constructor call
      llvm::SmallVector<Expr*, 4> pullbackArgs;
      Expr* dThisE = BuildOp(UnaryOperatorKind::UO_AddrOf, dfdx(),
                             m_DiffReq->getLocation());
      pullbackArgs.append(primalArgs.begin(), primalArgs.end());
      pullbackArgs.push_back(dThisE);
      pullbackArgs.append(adjointArgs.begin(), adjointArgs.end());

      Expr* pullbackCall = nullptr;
      Stmts& curRevBlock = getCurrentBlock(direction::reverse);
      Stmts::iterator it = std::begin(curRevBlock) + insertionPoint;
      curRevBlock.insert(it, prePullbackCallStmts.begin(),
                         prePullbackCallStmts.end());
      it += prePullbackCallStmts.size();

      // FIXME: No call context corresponds to second derivatives used in
      // hessians, which aren't scheduled statically yet.
      if (!m_DiffReq.CallContext) {
        std::string customPullbackName = "constructor_pullback";
        pullbackCall = m_Builder.BuildCallToCustomDerivativeOrNumericalDiff(
            customPullbackName, pullbackArgs, getCurrentScope(), CE);
      }

      if (!pullbackCall) {
        DiffRequest pullbackRequest{};
        pullbackRequest.Function = CD;

        // Mark the indexes of the global args. Necessary if the argument of the
        // call has a different name than the function's signature parameter.
        // pullbackRequest.CUDAGlobalArgsIndexes = globalCallArgs;

        pullbackRequest.BaseFunctionName = "constructor";
        pullbackRequest.Mode = DiffMode::pullback;
        // Silence diag outputs in nested derivation process.
        pullbackRequest.VerboseDiags = false;
        pullbackRequest.EnableTBRAnalysis = m_DiffReq.EnableTBRAnalysis;
        pullbackRequest.EnableVariedAnalysis = m_DiffReq.EnableVariedAnalysis;
        for (size_t i = 0, e = CD->getNumParams(); i < e; ++i)
          if (adjointArgs[i])
            pullbackRequest.DVI.push_back(CD->getParamDecl(i));

        FunctionDecl* pullbackFD = FindDerivedFunction(pullbackRequest);

        if (pullbackFD) {
          pullbackCall =
              m_Sema
                  .ActOnCallExpr(getCurrentScope(), BuildDeclRef(pullbackFD),
                                 m_DiffReq->getLocation(), pullbackArgs,
                                 m_DiffReq->getLocation())
                  .get();
        }
      }
      if (pullbackCall)
        curRevBlock.insert(it, pullbackCall);
    }

    // Create the constructor call in the forward-pass, or creates
    // 'constructor_reverse_forw' call if possible.

    // This works as follows:
    //
    // primal code:
    // ```
    // SomeClass c(u, v);
    // ```
    //
    // adjoint code:
    // ```
    // // forward-pass
    // clad::ValueAndAdjoint<SomeClass, SomeClass> _t0 =
    //   constructor_reverse_forw(clad::ConstructorReverseForwTag<SomeClass>{},
    //   u, v,
    //     _d_u, _d_v);
    // SomeClass _d_c = _t0.adjoint;
    // SomeClass c = _t0.value;
    // ```
    if (Expr* customReverseForwFnCall =
            BuildCallToCustomForwPassFn(CE, primalArgs, reverseForwAdjointArgs,
                                        /*baseExpr=*/nullptr)) {
      if (RD->isAggregate()) {
        SmallString<128> Name_class;
        llvm::raw_svector_ostream OS_class(Name_class);
        RD->getNameForDiagnostic(OS_class, m_Context.getPrintingPolicy(),
                                 /*qualified=*/true);
        diag(DiagnosticsEngine::Warning, CE->getBeginLoc(),
             "'%0' is an aggregate type and its constructor does not require a "
             "user-defined forward sweep function",
             {OS_class.str()});
        const FunctionDecl* constr_forw =
            cast<CallExpr>(customReverseForwFnCall)->getDirectCallee();
        SmallString<128> Name_forw;
        llvm::raw_svector_ostream OS_forw(Name_forw);
        constr_forw->getNameForDiagnostic(
            OS_forw, m_Context.getPrintingPolicy(), /*qualified=*/true);
        diag(DiagnosticsEngine::Note, constr_forw->getBeginLoc(),
             "'%0' is defined here", {OS_forw.str()});
      }
      Expr* callRes = StoreAndRef(customReverseForwFnCall);
      Expr* val =
          utils::BuildMemberExpr(m_Sema, getCurrentScope(), callRes, "value");
      Expr* adjoint =
          utils::BuildMemberExpr(m_Sema, getCurrentScope(), callRes, "adjoint");
      if (!utils::isCopyable(RD)) {
        val = utils::BuildStaticCastToRValue(m_Sema, val);
        adjoint = utils::BuildStaticCastToRValue(m_Sema, adjoint);
      }
      return {val, adjoint};
    }

    Expr* clonedArgsE = nullptr;

    if (CE->getNumArgs() != 1) {
      // FIXME: We generate a InitListExpr when the constructor is called
      // outside of a VarDecl init. This works out when it is later used in a
      // ReturnStmt. However, to support member exprs/calls of constructors, we
      // need to explicitly generate a constructor and not rely on higher level
      // Sema functions.
      if (CE->isListInitialization() || !m_TrackVarDeclConstructor) {
        clonedArgsE = m_Sema.ActOnInitList(noLoc, primalArgs, noLoc).get();
      } else {
        if (CE->getNumArgs() == 0) {
          // ParenList is empty -- default initialisation.
          // Passing empty parenList here will silently cause 'most vexing
          // parse' issue.
          return StmtDiff();
        }
        clonedArgsE = m_Sema.ActOnParenListExpr(noLoc, noLoc, primalArgs).get();
      }
    } else {
      clonedArgsE = primalArgs[0];
    }
    // `CXXConstructExpr` node will be created automatically by passing these
    // initialiser to higher level `ActOn`/`Build` Sema functions.
    return {clonedArgsE};
  }

  StmtDiff ReverseModeVisitor::VisitMaterializeTemporaryExpr(
      const clang::MaterializeTemporaryExpr* MTE) {
    // `MaterializeTemporaryExpr` node will be created automatically if it is
    // required by `ActOn`/`Build` Sema functions.
    StmtDiff MTEDiff = Visit(MTE->getSubExpr(), dfdx());
    return MTEDiff;
  }

  StmtDiff ReverseModeVisitor::VisitSubstNonTypeTemplateParmExpr(
      const clang::SubstNonTypeTemplateParmExpr* NTTP) {
    return Visit(NTTP->getReplacement());
  }

  StmtDiff ReverseModeVisitor::VisitUnaryExprOrTypeTraitExpr(
      const clang::UnaryExprOrTypeTraitExpr* UE) {
    return {Clone(UE), Clone(UE)};
  }

  DeclDiff<StaticAssertDecl> ReverseModeVisitor::DifferentiateStaticAssertDecl(
      const clang::StaticAssertDecl* SAD) {
    return DeclDiff<StaticAssertDecl>(nullptr, nullptr);
  }

  StmtDiff ReverseModeVisitor::VisitCXXStaticCastExpr(
      const clang::CXXStaticCastExpr* SCE) {
    StmtDiff subExprDiff = Visit(SCE->getSubExpr(), dfdx());

    // Reconstruct the cast
    TypeSourceInfo* TSI = SCE->getTypeInfoAsWritten();
    SourceLocation KWLoc = SCE->getOperatorLoc();
    SourceLocation RParenLoc = SCE->getRParenLoc();
    Expr* castExpr =
        m_Sema
            .BuildCXXNamedCast(KWLoc, tok::kw_static_cast, TSI,
                               subExprDiff.getExpr(), SCE->getAngleBrackets(),
                               SourceRange(KWLoc, RParenLoc))
            .get();
    subExprDiff.updateStmt(castExpr);

    return subExprDiff;
  }

  StmtDiff ReverseModeVisitor::VisitCXXConstCastExpr(
      const clang::CXXConstCastExpr* CCE) {
    StmtDiff subExprDiff = Visit(CCE->getSubExpr(), dfdx());
    return {Clone(CCE), subExprDiff.getExpr_dx()};
  }

  clang::QualType ReverseModeVisitor::ComputeAdjointType(clang::QualType T) {
    if (T->isReferenceType()) {
      QualType TValueType = utils::GetNonConstValueType(T);
      return m_Context.getPointerType(TValueType);
    }
    T.removeLocalConst();
    return T;
  }

  static bool needsDThis(const FunctionDecl* FD) {
    if (const auto* MD = dyn_cast<CXXMethodDecl>(FD)) {
      const CXXRecordDecl* RD = MD->getParent();
      if (MD->isInstance() && !RD->isLambda())
        return true;
    }
    return false;
  }

  void
  ReverseModeVisitor::BuildParams(llvm::SmallVectorImpl<ParmVarDecl*>& params) {
    const FunctionDecl* FD = m_DiffReq.Function;
    for (ParmVarDecl* PVD : FD->parameters()) {
      IdentifierInfo* PVDII = PVD->getIdentifier();
      // Implicitly created special member functions have no parameter names.
      if (!PVD->getDeclName())
        PVDII = CreateUniqueIdentifier("arg");

      auto* newPVD = CloneParmVarDecl(PVD, PVDII,
                                      /*pushOnScopeChains=*/true,
                                      /*cloneDefaultArg=*/false);

      if (!PVD->getDeclName()) // We can't use lookup-based replacements
        m_DeclReplacements[PVD] = newPVD;

      params.push_back(newPVD);
    }

    bool HasRet = false;
    // FIXME: We ignore the pointer return type for pullbacks.
    QualType dRetTy = FD->getReturnType().getNonReferenceType();
    dRetTy = utils::getNonConstType(dRetTy, m_Sema);
    if (m_DiffReq.Mode == DiffMode::pullback && !dRetTy->isVoidType() &&
        !dRetTy->isPointerType()) {
      auto paramNameExists = [&params](llvm::StringRef name) {
        for (ParmVarDecl* PVD : params)
          if (PVD->getName() == name)
            return true;
        return false;
      };

      // Make sure that we have no other parameter with the same name.
      // FIXME: This is to avoid changing a lot of tests which for some reason
      // add d_y when passing the return type value. We should probably not pick
      // a more appropriate name.
      std::string identifier = "y";
      for (unsigned idx = 0;; ++idx) {
        if (idx)
          identifier += std::to_string(idx - 1);
        if (!paramNameExists(identifier))
          break;
      }
      IdentifierInfo* II = &m_Context.Idents.get("_d_" + identifier);
      ParmVarDecl* retPVD =
          utils::BuildParmVarDecl(m_Sema, m_Derivative, II, dRetTy);
      m_Sema.PushOnScopeChains(retPVD, getCurrentScope(),
                               /*AddToContext=*/false);

      params.push_back(retPVD);
      m_Pullback = BuildDeclRef(retPVD);
      HasRet = true;
    }

    bool HasThis = needsDThis(FD);
    // If we are differentiating an instance member function then create a
    // parameter for representing derivative of `this` pointer with respect to
    // the independent parameter.
    if (HasThis) {
      IdentifierInfo* dThisII = &m_Context.Idents.get("_d_this");
      const auto* MD = cast<CXXMethodDecl>(FD);
      QualType thisTy = utils::GetParameterDerivativeType(
          m_Sema, m_DiffReq.Mode, MD->getThisType());

      auto* dPVD =
          utils::BuildParmVarDecl(m_Sema, m_Sema.CurContext, dThisII, thisTy);
      m_Sema.PushOnScopeChains(dPVD, getCurrentScope(), /*AddToContext=*/false);
      params.push_back(dPVD);
      // FIXME: Replace m_ThisExprDerivative in favor of lookups of _d_this.
      m_ThisExprDerivative = BuildDeclRef(dPVD);
    }

    const auto* FnType = cast<FunctionProtoType>(m_Derivative->getType());
    for (size_t i = 0, s = params.size(), p = s; i < s - HasThis - HasRet;
         ++i) {
      const ParmVarDecl* oPVD = FD->getParamDecl(i);

      if (clad::utils::hasNonDifferentiableAttribute(oPVD))
        continue;
      // FIXME: We can't use std::find(DVI.begin(), DVI.end()) because the
      // operator== considers params and intervals as different entities and
      // breaks the hessian tests. We should implement more robust checks in
      // DiffInputVarInfo to check if this is a variable we differentiate wrt.
      bool IsSelected = false;
      for (const DiffInputVarInfo& VarInfo : m_DiffReq.DVI) {
        if (VarInfo.param == oPVD) {
          IsSelected = true;
          break;
        }
      }

      const ParmVarDecl* PVD = params[i];
      if (!IsSelected) {
        m_NonIndepParams.push_back(PVD);
        continue;
      }
      IdentifierInfo* II =
          CreateUniqueIdentifier("_d_" + PVD->getNameAsString());
      QualType dPVDTy = FnType->getParamType(p++);
      auto* dPVD = utils::BuildParmVarDecl(m_Sema, m_Derivative, II, dPVDTy,
                                           PVD->getStorageClass());
      m_Sema.PushOnScopeChains(dPVD, getCurrentScope(), /*AddToContext=*/false);
      // Ensure that parameters passed by value are always dereferenced on use.
      // For example d_x in f(float x, float *d_x) should be used as (*d_x) to
      // matching the type of the input x from the original function.
      if (utils::isArrayOrPointerType(oPVD->getType())) {
        m_Variables[PVD] = BuildDeclRef(dPVD);

      } else {
        Expr* Deref =
            BuildOp(UO_Deref, BuildDeclRef(dPVD), oPVD->getLocation());
        if (dPVDTy->getPointeeType()->isRecordType())
          Deref = utils::BuildParenExpr(m_Sema, Deref);
        m_Variables[PVD] = Deref;
      }

      params.push_back(dPVD);
    }
  }

  Expr* ReverseModeVisitor::BuildCallToCustomForwPassFn(
      const Expr* callSite, llvm::ArrayRef<Expr*> primalArgs,
      llvm::ArrayRef<clang::Expr*> derivedArgs, Expr* baseExpr) {
    llvm::SmallVector<Expr*, 4> args;
    if (baseExpr) {
      baseExpr = BuildOp(UnaryOperatorKind::UO_AddrOf, baseExpr,
                         m_DiffReq->getLocation());
      args.push_back(baseExpr);
    }
    const FunctionDecl* FD = nullptr;
    if (const auto* CE = dyn_cast<CallExpr>(callSite))
      FD = CE->getDirectCallee();
    else
      FD = cast<CXXConstructExpr>(callSite)->getConstructor();

    if (auto CD = llvm::dyn_cast<CXXConstructorDecl>(FD)) {
      const RecordDecl* RD = CD->getParent();
      QualType constructorReverseForwTagT =
          GetCladConstructorReverseForwTagOfType(m_Context.getRecordType(RD));
      Expr* constructorReverseForwTagArg =
          m_Sema
              .BuildCXXTypeConstructExpr(
                  m_Context.getTrivialTypeSourceInfo(
                      constructorReverseForwTagT, utils::GetValidSLoc(m_Sema)),
                  utils::GetValidSLoc(m_Sema), MultiExprArg{},
                  utils::GetValidSLoc(m_Sema),
                  /*ListInitialization=*/false)
              .get();
      args.push_back(constructorReverseForwTagArg);
    }
    args.append(primalArgs.begin(), primalArgs.end());
    args.append(derivedArgs.begin(), derivedArgs.end());
    std::string forwPassFnName =
        clad::utils::ComputeEffectiveFnName(FD) + "_reverse_forw";
    Expr* customForwPassCE =
        m_Builder.BuildCallToCustomDerivativeOrNumericalDiff(
            forwPassFnName, args, getCurrentScope(), callSite);
    return customForwPassCE;
  }
} // end namespace clad
