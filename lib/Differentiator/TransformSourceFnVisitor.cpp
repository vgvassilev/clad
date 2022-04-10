#include "clad/Differentiator/TransformSourceFnVisitor.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/ErrorEstimator.h"

#include "llvm/Support/SaveAndRestore.h"

#include <algorithm>

using namespace clang;

namespace clad {

TransformSourceFnVisitor::TransformSourceFnVisitor(DerivativeBuilder& builder)
    : VisitorBase(builder) {}

OverloadedDeclWithContext
TransformSourceFnVisitor::Derive(const FunctionDecl* FD,
                                 const DiffRequest& request) {
  silenceDiags = !request.VerboseDiags;
  m_Function = FD;

  m_Mode = DiffMode::reverse_source_fn;

  assert(m_Function && "Must not be null.");

  DiffParams args{};
  std::copy(FD->param_begin(), FD->param_end(), std::back_inserter(args));

  auto fnName = m_Function->getNameAsString() + "_with_adjoint";
  auto fnDNI = utils::BuildDeclarationNameInfo(m_Sema, fnName);

  auto paramTypes = ComputeParamTypes(args);
  auto returnType = ComputeReturnType();
  auto sourceFnType = dyn_cast<FunctionProtoType>(m_Function->getType());
  auto fnType = m_Context.getFunctionType(returnType, paramTypes,
                                          sourceFnType->getExtProtoInfo());

  llvm::SaveAndRestore<DeclContext*> saveContext(m_Sema.CurContext);
  llvm::SaveAndRestore<Scope*> saveScope(m_CurScope);
  m_Sema.CurContext = const_cast<DeclContext*>(m_Function->getDeclContext());

  DeclWithContext fnBuildRes =
      m_Builder.cloneFunction(m_Function, *this, m_Sema.CurContext, m_Sema,
                              m_Context, noLoc, fnDNI, fnType);
  m_Derivative = fnBuildRes.first;

  beginScope(Scope::FunctionPrototypeScope | Scope::FunctionDeclarationScope |
             Scope::DeclScope);
  m_Sema.PushFunctionScope();
  m_Sema.PushDeclContext(getCurrentScope(), m_Derivative);

  auto params = BuildParams(args);
  llvm::errs() << "Derive: params.size(): " << params.size() << "\n";
  m_Derivative->setParams(params);
  m_Derivative->setBody(nullptr);

  beginScope(Scope::FnScope | Scope::DeclScope);
  m_DerivativeFnScope = getCurrentScope();

  beginBlock();

  StmtDiff bodyDiff = Visit(m_Function->getBody());
  Stmt* forward = bodyDiff.getStmt();

  for (Stmt* S : m_Globals)
    addToCurrentBlock(S);

  if (auto CS = dyn_cast<CompoundStmt>(forward))
    for (Stmt* S : CS->body())
      addToCurrentBlock(S);

  Stmt* fnBody = endBlock();
  llvm::errs() << "Derive: dumping fnBody:\n";
  fnBody->dumpColor();
  m_Derivative->setBody(fnBody);
  endScope();
  m_Sema.PopFunctionScopeInfo();
  m_Sema.PopDeclContext();
  endScope();
  llvm::errs() << "Derive: Dumping m_Derivative:\n";
  m_Derivative->dumpColor();
  return OverloadedDeclWithContext(m_Derivative, fnBuildRes.second, nullptr);
}

// FIXME: This function is copied from ReverseModeVisitor. Find a suitable place
// for it.
QualType TransformSourceFnVisitor::GetParameterDerivativeType(QualType yType,
                                                              QualType xType) {
  assert(yType.getNonReferenceType()->isRealType() &&
         "yType should be a builtin-numerical scalar type!!");
  QualType xValueType = utils::GetValueType(xType);
  // derivative variables should always be of non-const type.
  xValueType.removeLocalConst();
  QualType nonRefXValueType = xValueType.getNonReferenceType();
  if (nonRefXValueType->isRealType())
    return GetCladArrayRefOfType(yType);
  else
    return GetCladArrayRefOfType(nonRefXValueType);
}

llvm::SmallVector<clang::QualType, 8>
TransformSourceFnVisitor::ComputeParamTypes(const DiffParams& diffParams) {
  llvm::SmallVector<clang::QualType, 8> paramTypes;
  paramTypes.reserve(m_Function->getNumParams() * 2);
  for (auto PVD : m_Function->parameters())
    paramTypes.push_back(PVD->getType());

  QualType effectiveReturnType =
      m_Function->getReturnType().getNonReferenceType();

  if (auto MD = dyn_cast<CXXMethodDecl>(m_Function)) {
    const CXXRecordDecl* RD = MD->getParent();
    if (MD->isInstance() && !RD->isLambda()) {
      QualType thisType = clad_compat::CXXMethodDecl_getThisType(m_Sema, MD);
      paramTypes.push_back(
          GetParameterDerivativeType(effectiveReturnType, thisType));
    }
  }

  for (auto PVD : m_Function->parameters()) {
    auto it = std::find(std::begin(diffParams), std::end(diffParams), PVD);
    if (it != std::end(diffParams)) {
      paramTypes.push_back(
          GetParameterDerivativeType(effectiveReturnType, PVD->getType()));
    }
  }
  return paramTypes;
}

clang::QualType TransformSourceFnVisitor::ComputeReturnType() {
  auto valAndAdjointTempDecl = GetCladClassDecl("ValueAndAdjoint");
  auto RT = m_Function->getReturnType();
  auto T = GetCladClassOfType(valAndAdjointTempDecl, {RT, RT});
  return T;
}

llvm::SmallVector<clang::ParmVarDecl*, 8>
TransformSourceFnVisitor::BuildParams(DiffParams& diffParams) {
  llvm::SmallVector<clang::ParmVarDecl*, 8> params, paramDerivatives;
  params.reserve(m_Function->getNumParams() + diffParams.size());
  auto derivativeFnType = cast<FunctionProtoType>(m_Derivative->getType());

  std::size_t dParamTypesIdx = m_Function->getNumParams();

  if (auto MD = dyn_cast<CXXMethodDecl>(m_Function)) {
    const CXXRecordDecl* RD = MD->getParent();
    if (MD->isInstance() && !RD->isLambda()) {
      auto thisDerivativePVD = utils::BuildParmVarDecl(
          m_Sema, m_Derivative, CreateUniqueIdentifier("_d_this"),
          derivativeFnType->getParamType(dParamTypesIdx));
      paramDerivatives.push_back(thisDerivativePVD);

      if (thisDerivativePVD->getIdentifier())
        m_Sema.PushOnScopeChains(thisDerivativePVD, getCurrentScope(),
                                 /*AddToContext=*/false);

      Expr* deref =
          BuildOp(UnaryOperatorKind::UO_Deref, BuildDeclRef(thisDerivativePVD));
      m_ThisExprDerivative = utils::BuildParenExpr(m_Sema, deref);
      ++dParamTypesIdx;
    }
  }
  for (auto PVD : m_Function->parameters()) {
    // FIXME: Call expression may contain default arguments that we are now
    // removing. This may cause issues.
    auto newPVD = utils::BuildParmVarDecl(
        m_Sema, m_Derivative, PVD->getIdentifier(), PVD->getType(),
        PVD->getStorageClass(), /*DefArg=*/nullptr, PVD->getTypeSourceInfo());
    params.push_back(newPVD);

    if (newPVD->getIdentifier())
      m_Sema.PushOnScopeChains(newPVD, getCurrentScope(),
                               /*AddToContext=*/false);

    auto it = std::find(std::begin(diffParams), std::end(diffParams), PVD);
    if (it != std::end(diffParams)) {
      *it = newPVD;
      QualType dType = derivativeFnType->getParamType(dParamTypesIdx);
      IdentifierInfo* dII =
          CreateUniqueIdentifier("_d_" + PVD->getNameAsString());
      auto dPVD = utils::BuildParmVarDecl(m_Sema, m_Derivative, dII, dType,
                                          PVD->getStorageClass());
      paramDerivatives.push_back(dPVD);
      ++dParamTypesIdx;

      if (dPVD->getIdentifier())
        m_Sema.PushOnScopeChains(dPVD, getCurrentScope(),
                                 /*AddToContext=*/false);

      if (utils::isArrayOrPointerType(PVD->getType())) {
        m_Variables[*it] = (Expr*)BuildDeclRef(dPVD);
      } else {
        QualType valueType = DetermineCladArrayValueType(dPVD->getType());
        m_Variables[*it] =
            BuildOp(UO_Deref, BuildDeclRef(dPVD), m_Function->getLocation());
        // Add additional paranthesis if derivative is of record type
        // because `*derivative.someField` will be incorrectly evaluated if
        // the derived function is compiled standalone.
        if (valueType->isRecordType())
          m_Variables[*it] = utils::BuildParenExpr(m_Sema, m_Variables[*it]);
      }
    }
  }
  params.insert(params.end(), paramDerivatives.begin(), paramDerivatives.end());
  return params;
}

StmtDiff TransformSourceFnVisitor::ProcessSingleStmt(const clang::Stmt* S) {
  StmtDiff SDiff = Visit(S);
  return {SDiff.getStmt()};
}

// StmtDiff TransformSourceFnVisitor::ProcessSingleExpr(const clang::Expr* E) {
//   beginBlock();
//   StmtDiff EDiff = Visit(E);
//   Stmt* forward =
// }

StmtDiff TransformSourceFnVisitor::VisitStmt(const clang::Stmt* S) {
  // llvm::errs() << "Reaching TransformSourceFnVisitor::VisitStmt!!\n";
  return {Clone(S)};
}

StmtDiff
TransformSourceFnVisitor::VisitCompoundStmt(const clang::CompoundStmt* CS) {
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

StmtDiff TransformSourceFnVisitor::VisitDeclRefExpr(const DeclRefExpr* DRE) {
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

  if (auto decl = dyn_cast<VarDecl>(clonedDRE->getDecl())) {
    // Check DeclRefExpr is a reference to an independent variable.
    auto it = m_Variables.find(decl);
    if (it == std::end(m_Variables)) {
      // Is not an independent variable, ignored.
      return StmtDiff(clonedDRE);
    }
    return StmtDiff(clonedDRE, it->second);
  }

  return StmtDiff(clonedDRE);
}

StmtDiff
TransformSourceFnVisitor::VisitReturnStmt(const clang::ReturnStmt* RS) {
  const Expr* value = RS->getRetValue();
  auto returnDiff = Visit(value);
  llvm::SmallVector<Expr*, 2> returnArgs = {returnDiff.getExpr(),
                                            returnDiff.getExpr_dx()};
  Expr* returnInitList = m_Sema.BuildInitList(noLoc, returnArgs, noLoc).get();
  Stmt* newRS = m_Sema.BuildReturnStmt(noLoc, returnInitList).get();
  return {newRS};
}
} // namespace clad