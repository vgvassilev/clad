#include "clad/Differentiator/VectorForwardModeVisitor.h"

#include "ConstantFolder.h"
#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/DerivativeBuilder.h"
#include "clad/Differentiator/ParseDiffArgsTypes.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/TemplateName.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Lookup.h"

#include "llvm/Support/SaveAndRestore.h"

#include <algorithm>
#include <array>
#include <cstdint>

using namespace clang;

namespace clad {
VectorForwardModeVisitor::VectorForwardModeVisitor(DerivativeBuilder& builder,
                                                   const DiffRequest& request)
    : BaseForwardModeVisitor(builder, request) {}

VectorForwardModeVisitor::~VectorForwardModeVisitor() {}

Expr* VectorForwardModeVisitor::IndVarOffsetTracker::buildOffset() const {
  // The running sum keeps growing after this call, so hand out a clone.
  Expr* offset = m_V->CloneNode(m_ArrayCount);
  if (offset && m_ScalarCount == 0)
    return offset;
  Expr* scalars = ConstantFolder::synthesizeLiteral(
      m_V->m_Context.UnsignedLongTy, m_V->m_Context, m_ScalarCount);
  if (!offset)
    return scalars;
  return m_V->BuildOp(BinaryOperatorKind::BO_Add, offset, scalars);
}

void VectorForwardModeVisitor::IndVarOffsetTracker::advanceByArray(
    const Expr* size) {
  // The caller splices `size` itself into the derivative, so keep a clone.
  Expr* clonedSize = m_V->CloneNode(size);
  if (!m_ArrayCount)
    m_ArrayCount = clonedSize;
  else
    m_ArrayCount =
        m_V->BuildOp(BinaryOperatorKind::BO_Add, m_ArrayCount, clonedSize);
}

std::string VectorForwardModeVisitor::GetPushForwardFunctionSuffix() {
  return "_vector_pushforward";
}

DiffMode VectorForwardModeVisitor::GetPushForwardMode() {
  return DiffMode::vector_pushforward;
}

DerivativeAndOverload VectorForwardModeVisitor::Derive() {
  const FunctionDecl* FD = m_DiffReq.Function;
  assert(m_DiffReq.Mode == DiffMode::vector_forward_mode);

  // Generate the function type for the derivative.
  QualType vectorDiffFunctionType = GetDerivativeType();

  // Create the function declaration for the derivative.
  std::string derivedFnName = m_DiffReq.ComputeDerivativeName();

  IdentifierInfo* II = &m_Context.Idents.get(derivedFnName);
  SourceLocation loc{m_DiffReq->getLocation()};
  DeclarationNameInfo name(II, loc);

  // Save Sema state before cloneFunction mutates it.
  llvm::SaveAndRestore<DeclContext*> SaveContext(m_Sema.CurContext);
  llvm::SaveAndRestore<Scope*> SaveScope(getCurrentScope());
  // FIXME: We should not use const_cast to get the decl context here.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  auto* DC = const_cast<DeclContext*>(m_DiffReq->getDeclContext());
  m_Sema.CurContext = DC;
  // `result` owns the namespace Scopes cloneFunction opens; its
  // destructor pops them before SaveScope restores.
  ClonedFunction result = m_Builder.cloneFunction(
      m_DiffReq.Function, *this, DC, loc, name, vectorDiffFunctionType);
  FunctionDecl* vectorDiffFD = result.fd;
  m_Derivative = vectorDiffFD;

  // Function declaration scope
  beginScope(Scope::FunctionPrototypeScope | Scope::FunctionDeclarationScope |
             Scope::DeclScope);
  m_Sema.PushFunctionScope();
  m_Sema.PushDeclContext(getCurrentScope(), m_Derivative);

  // Set the parameters for the derivative.
  DiffParams args{};
  for (const auto& dParam : m_DiffReq.DVI)
    args.push_back(dParam.param);
  // Running sum of independent-variable counts, filled in by
  // BuildVectorModeParams and materialized into m_IndVarCountDecl below.
  Expr* indVarCountExpr = nullptr;
  auto params = BuildVectorModeParams(args, indVarCountExpr);
  utils::SetParams(vectorDiffFD,
                   clad_compat::makeArrayRef(params.data(), params.size()));
  vectorDiffFD->setBody(nullptr);

  // Create the body of the derivative.
  beginScope(Scope::FnScope | Scope::DeclScope);
  m_DerivativeFnScope = getCurrentScope();
  beginBlock();

  // Instantiate a variable indepVarCount to store the total number of
  // independent variables requested.
  // size_t indepVarCount = indVarCountExpr;
  auto* totalIndVars =
      BuildVarDecl(m_Context.UnsignedLongTy, "indepVarCount", indVarCountExpr);
  addToCurrentBlock(BuildDeclStmt(totalIndVars));
  m_IndVarCountDecl = totalIndVars;

  // Offset of the next independent variable into the vector of all of them.
  IndVarOffsetTracker offsetTracker(*this);

  // Current Index of independent variable in the param list of the function.
  size_t independentVarIndex = 0;

  for (size_t i = 0; i < m_DiffReq->getNumParams(); ++i) {
    bool is_array =
        utils::isArrayOrPointerType(m_DiffReq->getParamDecl(i)->getType());
    auto param = params[i];
    QualType dParamType = clad::utils::GetNonConstValueType(param->getType());

    Expr* dVectorParam = nullptr;
    if (m_IndependentVars.size() > independentVarIndex &&
        m_IndependentVars[independentVarIndex] == m_DiffReq->getParamDecl(i)) {

      // The tangents of an object are objects, which one-hot vectors and
      // identity matrices of numbers do not make.
      if (dParamType->isRecordType()) {
        diag(DiagnosticsEngine::Error,
             m_DiffReq->getParamDecl(i)->getLocation(),
             "vector mode cannot yet differentiate with respect to an object "
             "of type %0")
            << dParamType;
        ++independentVarIndex;
        continue;
      }

      Expr* offsetExpr = offsetTracker.buildOffset();

      if (is_array) {
        // Get size of the array.
        Expr* getSize =
            BuildArrayRefSizeExpr(m_ParamVariables[m_DiffReq->getParamDecl(i)]);

        // Create an identity matrix for the parameter,
        // with number of rows equal to the size of the array,
        // and number of columns equal to the number of independent variables
        llvm::SmallVector<Expr*, 3> args = {getSize, buildIndVarCountRef(),
                                            offsetExpr};
        dVectorParam = BuildIdentityMatrixExpr(dParamType, args);

        offsetTracker.advanceByArray(getSize);
      } else {
        // Create a one hot vector for the parameter.
        llvm::SmallVector<Expr*, 2> args = {buildIndVarCountRef(), offsetExpr};
        dVectorParam =
            BuildCallExprToCladFunction("one_hot_vector", args, {dParamType});
        offsetTracker.advanceByScalar();
      }
      ++independentVarIndex;
    } else {
      // We cannot initialize derived variable for pointer types because
      // we do not know the correct size.
      if (is_array)
        continue;
      // This parameter is not an independent variable.
      // Initialize by all zeros.
      Expr* dCount = buildIndVarCountRef();
      dVectorParam =
          BuildCallExprToCladFunction("zero_vector", {dCount}, {dParamType});
    }

    // For each function arg to be differentiated, create a variable
    // _d_vector_arg to store the vector of derivatives for that arg.
    // for ex: double f(double x, double y, double z);
    // and we want to differentiate w.r.t. x and z, then we will have
    // -> clad::array<double> _d_vector_x = {1, 0};
    // -> clad::array<double> _d_vector_y = {0, 0};
    // -> clad::array<double> _d_vector_z = {0, 1};
    QualType dVectorParamType;
    if (is_array)
      dVectorParamType = utils::GetCladMatrixOfType(m_Sema, dParamType);
    else
      dVectorParamType = utils::GetCladArrayOfType(m_Sema, dParamType);
    auto dVectorParamDecl =
        BuildVarDecl(dVectorParamType, "_d_vector_" + param->getNameAsString(),
                     dVectorParam);
    addToCurrentBlock(BuildDeclStmt(dVectorParamDecl));
    // Memorize the derivative vector for the parameter.
    m_Variables[param] = {dVectorParamDecl};
  }

  // Traverse the function body and generate the derivative.
  Stmt* BodyDiff = Visit(FD->getBody()).getStmt();
  if (auto CS = dyn_cast<CompoundStmt>(BodyDiff))
    for (Stmt* S : CS->body())
      addToCurrentBlock(S);
  else
    addToCurrentBlock(BodyDiff);

  Stmt* vectorDiffBody = endBlock();
  m_Derivative->setBody(vectorDiffBody);
  endScope(); // Function body scope
  m_Sema.PopFunctionScopeInfo();
  m_Sema.PopDeclContext();
  endScope(); // Function decl scope

  // Create the overload declaration for the derivative.
  FunctionDecl* overloadFD = CreateDerivativeOverload(/*derivative=*/nullptr,
                                                      OverloadKind::VectorMode);
  return DerivativeAndOverload{vectorDiffFD, overloadFD};
}

llvm::SmallVector<clang::ParmVarDecl*, 8>
VectorForwardModeVisitor::BuildVectorModeParams(DiffParams& diffParams,
                                                Expr*& indVarCountExpr) {
  llvm::SmallVector<clang::ParmVarDecl*, 8> params, paramDerivatives;
  params.reserve(m_DiffReq->getNumParams() + diffParams.size());
  auto derivativeFnType = cast<FunctionProtoType>(m_Derivative->getType());
  std::size_t dParamTypesIdx = m_DiffReq->getNumParams();

  // Count the number of non-array independent variables requested for
  // differentiation.
  size_t nonArrayIndVarCount = 0;

  for (auto* PVD : m_DiffReq->parameters()) {
    auto newPVD = utils::BuildParmVarDecl(
        m_Sema, m_Derivative, PVD->getIdentifier(), PVD->getType(),
        PVD->getStorageClass(), /*DefArg=*/nullptr, PVD->getTypeSourceInfo());
    params.push_back(newPVD);

    if (newPVD->getIdentifier())
      m_Sema.PushOnScopeChains(newPVD, getCurrentScope(),
                               /*AddToContext=*/false);

    auto it = std::find(std::begin(diffParams), std::end(diffParams), PVD);
    if (it == std::end(diffParams))
      continue; // This parameter is not in the diff list.

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
      m_ParamVariables[*it] = (Expr*)BuildDeclRef(dPVD);
      // dPVD will be a clad::array or clad::array_ref, both have size() method.
      // If indVarCountExpr is null, initialize it with dPVD.size().
      // Otherwise, increment it by dPVD.size().
      Expr* getSize = BuildArrayRefSizeExpr(m_ParamVariables[*it]);
      if (!indVarCountExpr) {
        indVarCountExpr = getSize;
      } else {
        indVarCountExpr =
            BuildOp(BinaryOperatorKind::BO_Add, indVarCountExpr, getSize);
      }
    } else {
      m_ParamVariables[*it] = BuildOp(UO_Deref, BuildDeclRef(dPVD), noLoc);
      nonArrayIndVarCount += 1;
    }
  }

  // Process the expression for the number independent variables.
  // This will be the sum of the sizes of all array parameters and the number
  // of non-array parameters.
  Expr* nonArrayIndVarCountExpr = ConstantFolder::synthesizeLiteral(
      m_Context.UnsignedLongTy, m_Context, nonArrayIndVarCount);
  if (!indVarCountExpr) {
    indVarCountExpr = nonArrayIndVarCountExpr;
  } else if (nonArrayIndVarCount != 0) {
    indVarCountExpr = BuildOp(BinaryOperatorKind::BO_Add, indVarCountExpr,
                              nonArrayIndVarCountExpr);
  }

  // insert the derivative parameters at the end of the parameter list.
  params.insert(params.end(), paramDerivatives.begin(), paramDerivatives.end());
  // store the independent variables for later use.
  m_IndependentVars.insert(m_IndependentVars.end(), diffParams.begin(),
                           diffParams.end());
  return params;
}

StmtDiff VectorForwardModeVisitor::VisitArraySubscriptExpr(
    const ArraySubscriptExpr* ASE) {
  auto ASI = SplitArraySubscript(ASE);
  const Expr* Base = ASI.first;
  StmtDiff BaseDiff = Visit(Base);
  const auto& Indices = ASI.second;
  Expr* clonedBase = BaseDiff.getExpr();
  llvm::SmallVector<Expr*, 4> clonedIndices(Indices.size());
  std::transform(std::begin(Indices), std::end(Indices),
                 std::begin(clonedIndices),
                 [this](const Expr* E) { return Clone(E); });
  auto* cloned = BuildArraySubscript(clonedBase, clonedIndices);

  auto* zero = ConstantFolder::synthesizeLiteral(ASE->getType(), m_Context, 0);
  Expr* diffExpr = zero;

  Expr* target = BaseDiff.getExpr_dx();
  if (target) {
    // Clone the index: it is already consumed by the primal subscript above,
    // so the derivative subscript needs its own node.
    diffExpr = BuildArraySubscript(target, CloneNode(clonedIndices.front()));
  }
  return StmtDiff(cloned, diffExpr);
}

StmtDiff VectorForwardModeVisitor::VisitReturnStmt(const ReturnStmt* RS) {
  const Expr* retVal = RS->getRetValue();
  QualType retType = retVal->getType();
  StmtDiff retValDiff = Visit(retVal);
  Expr* derivedRetValE = retValDiff.getExpr_dx();
  derivedRetValE = asTangentVector(derivedRetValE, retType);
  // If we are in vector mode, we need to wrap the return value in a
  // vector.
  QualType cladArrayType =
      utils::GetCladArrayOfType(m_Sema, utils::GetNonConstValueType(retType));
  VarDecl* dVectorParamDecl = BuildVarDecl(cladArrayType, "_d_vector_return",
                                           derivedRetValE, /*DirectInit=*/true);
  // Create an array of statements to hold the return statement and the
  // assignments to the derivatives of the parameters.
  Stmts returnStmts;
  returnStmts.push_back(BuildDeclStmt(dVectorParamDecl));
  // Assign values from return vector to the derivatives of the
  // parameters.
  auto dVectorRef = BuildDeclRef(dVectorParamDecl);

  // Offset of the next independent variable into the vector of all of them.
  IndVarOffsetTracker offsetTracker(*this);

  for (size_t i = 0; i < m_IndependentVars.size(); ++i) {
    // An object is refused as an independent variable.
    if (utils::GetNonConstValueType(m_IndependentVars[i]->getType())
            ->isRecordType())
      continue;
    // Get the derivative of the ith parameter.
    auto dParam = m_ParamVariables[m_IndependentVars[i]];
    Expr* dParamValue = nullptr;

    Expr* offsetExpr = offsetTracker.buildOffset();

    if (isCladArrayType(dParam->getType())) {
      // Get the size of the array.
      Expr* getSize = BuildArrayRefSizeExpr(dParam);

      // Create an expression to fetch slice of the return vector. dVectorRef
      // is reused for every parameter; clone so the slices/subscripts do not
      // share the return-vector reference.
      llvm::SmallVector<Expr*, 2> args = {offsetExpr, getSize};
      dParamValue = BuildArrayRefSliceExpr(CloneNode(dVectorRef), args);

      offsetTracker.advanceByArray(getSize);
    } else {
      dParamValue = BuildArraySubscript(CloneNode(dVectorRef), offsetExpr);
      offsetTracker.advanceByScalar();
    }
    // Create an assignment expression to assign the ith element of the
    // return vector to the derivative of the ith parameter.
    auto dParamAssign = BuildOp(BO_Assign, dParam, dParamValue);
    // Add the assignment statement to the array of statements.
    returnStmts.push_back(dParamAssign);
  }
  // Add an empty return statement to the array of statements.
  returnStmts.push_back(
      m_Sema.ActOnReturnStmt(noLoc, nullptr, getCurrentScope()).get());

  // Create a return statement from the compound statement.
  Stmt* returnStmt = MakeCompoundStmt(returnStmts);
  return StmtDiff(returnStmt);
}

/// Whether \p Dx reads the tangents of a member out of those of an object,
/// which are a copy: what is written to them would be lost.
static bool isMemberTangents(const Expr* Dx) {
  const auto* CE = Dx ? dyn_cast<CallExpr>(Dx->IgnoreImplicit()) : nullptr;
  const FunctionDecl* FD = CE ? CE->getDirectCallee() : nullptr;
  return FD && FD->getQualifiedNameAsString() == "clad::member_vector";
}

DeclDiff<VarDecl>
VectorForwardModeVisitor::DifferentiateVarDecl(const VarDecl* VD) {
  StmtDiff initDiff = VD->getInit() ? Visit(VD->getInit()) : StmtDiff{};
  // Here we are assuming that derived type and the original type are same.
  // This may not necessarily be true in the future.
  VarDecl* VDClone = BuildVarDecl(VD->getType(), VD->getNameAsString(),
                                  initDiff.getExpr(), VD->isDirectInit());
  Expr* initDx = initDiff.getExpr_dx();
  QualType T = VD->getType();
  QualType ObjTy = utils::GetNonConstValueType(T);
  if (utils::isNonConstReferenceType(T) && isMemberTangents(initDx))
    diag(DiagnosticsEngine::Error, VD->getLocation(),
         "vector mode cannot yet differentiate a reference to a member of an "
         "object");
  const auto* RD = ObjTy->getAsCXXRecordDecl();
  // An array of objects is not an object: its elements have no tangents of
  // their own yet.
  if (RD && T->isArrayType()) {
    diag(DiagnosticsEngine::Error, VD->getLocation(),
         "vector mode cannot yet differentiate an array of objects");
    initDx = nullptr;
  } else if (RD && T->isPointerType() && initDx) {
    // A pointer to an object with tangents; one to what has none, such as a
    // global, has none either.
    diag(DiagnosticsEngine::Error, VD->getLocation(),
         "vector mode cannot yet differentiate a pointer to an object");
    initDx = nullptr;
  } else if (!T->isReferenceType() && RD && utils::hasObjectTangents(ObjTy)) {
    initDx = objectTangents(ObjTy, VD->getInit(), initDx);
    // Initializing a clad::array considers its constructor from a
    // std::initializer_list, which, of a class no code has made one of yet,
    // Sema would instantiate where the initializer is: a generated expression
    // may have no location.
    m_Sema.isCompleteType(GenLoc(),
                          m_Sema.BuildStdInitializerList(ObjTy, GenLoc()));
  } else if (!T->isReferenceType() && RD && RD->hasDefinition() &&
             !RD->hasDefaultConstructor()) {
    // Its tangents are a clad::array of objects, which must be able to start
    // out empty.
    diag(DiagnosticsEngine::Error, VD->getLocation(),
         "vector mode cannot yet differentiate an object of type %0, which "
         "has no default constructor")
        << ObjTy;
    initDx = nullptr;
  } else if (VD->getInit())
    initDx = asTangentVector(initDx, T);
  VarDecl* VDDerived = BuildVarDecl(
      utils::GetCladArrayOfType(m_Sema,
                                utils::GetNonConstValueType(VD->getType())),
      "_d_vector_" + VD->getNameAsString(), initDx, /*DirectInit=*/true);

  m_Variables.emplace(VDClone, AdjointInfo{VDDerived});
  return DeclDiff<VarDecl>(VDClone, VDDerived);
}

StmtDiff VectorForwardModeVisitor::VisitFloatingLiteral(
    const clang::FloatingLiteral* FL) {
  Expr* dCount = buildIndVarCountRef();
  auto* zero_vec =
      BuildCallExprToCladFunction("zero_vector", {dCount}, {FL->getType()});
  return StmtDiff(Clone(FL), zero_vec);
}

/// Whether \p E is a vector of tangents, one per independent variable: a
/// clad::array or an expression of them. By the declaration: the type of a
/// call such as clad::zero_vector prints without its namespace.
static bool isTangentVector(const Expr* E) {
  const auto* RD = E ? E->getType()->getAsCXXRecordDecl() : nullptr;
  if (!RD)
    return false;
  std::string Name = RD->getQualifiedNameAsString();
  return Name == "clad::array" || Name == "clad::array_expression";
}

/// Whether evaluating \p E changes something: an assignment or increment,
/// through an operator of clad::array too, which HasSideEffects counts only
/// with every other call.
static bool changesState(const Expr* E, const ASTContext& C) {
  const auto* OCE = dyn_cast<CXXOperatorCallExpr>(E);
  if (E->HasSideEffects(C, /*IncludePossibleEffects=*/false) ||
      (OCE && (OCE->isAssignmentOp() || OCE->getOperator() == OO_PlusPlus ||
               OCE->getOperator() == OO_MinusMinus)))
    return true;
  for (const Stmt* Child : E->children())
    if (const auto* CE = dyn_cast_or_null<Expr>(Child))
      if (changesState(CE, C))
        return true;
  return false;
}

/// The type of the tangents in the vector \p E where it is a clad::array<T>;
/// \p Fallback where it is an expression of them.
static QualType tangentElementType(const Expr* E, QualType Fallback) {
  const auto* Spec = dyn_cast_or_null<ClassTemplateSpecializationDecl>(
      E->getType()->getAsCXXRecordDecl());
  if (Spec && Spec->getQualifiedNameAsString() == "clad::array")
    return Spec->getTemplateArgs().get(0).getAsType();
  return Fallback;
}

/// Whether \p Dx is a vector of the tangents of objects: a clad::array of a
/// class.
static bool isObjectTangents(const Expr* Dx) {
  QualType Elem = Dx ? tangentElementType(Dx, QualType()) : QualType();
  return !Elem.isNull() && Elem->isRecordType();
}

/// How \p Dx is a number that is zero whenever the code runs: what a
/// derivative clad leaves at zero becomes, through a negation, a cast or a
/// ?:. Every tangent that may not be zero is a vector, or holds one.
namespace {
enum class ScalarZero : std::uint8_t { No, Pure, WithEffects };
} // namespace
static ScalarZero isScalarZero(const Expr* Dx, ASTContext& C) {
  if (!Dx || !Dx->getType()->isArithmeticType() || Dx->isValueDependent())
    return ScalarZero::No;
  // A ?: of two zeros is one whichever way its condition, a variable, goes.
  if (const auto* CO = dyn_cast<ConditionalOperator>(Dx->IgnoreParens())) {
    // No, Pure and WithEffects are ordered: the lesser says whether both
    // are zeros, the greater whether either keeps an effect.
    ScalarZero T = isScalarZero(CO->getTrueExpr(), C);
    ScalarZero F = isScalarZero(CO->getFalseExpr(), C);
    return std::min(T, F) == ScalarZero::No ? ScalarZero::No : std::max(T, F);
  }
  Expr::EvalResult R;
  bool Zero = Dx->EvaluateAsRValue(R, C) &&
              ((R.Val.isInt() && R.Val.getInt().isZero()) ||
               (R.Val.isFloat() && R.Val.getFloat().isZero()));
  // `(x = x * 2, 0)` is zero, but replaced would drop the assignment.
  if (!Zero)
    return ScalarZero::No;
  return R.HasSideEffects ? ScalarZero::WithEffects : ScalarZero::Pure;
}

/// Whether \p Dx is a braced list of zeros, and of lists of them: the zero
/// tangent of an object in scalar mode.
static bool isZeroList(const Expr* Dx, ASTContext& C) {
  const auto* ILE = dyn_cast<InitListExpr>(Dx->IgnoreImplicit());
  if (!ILE)
    return isScalarZero(Dx, C) == ScalarZero::Pure;
  return std::all_of(ILE->begin(), ILE->end(), [&C](const Stmt* Init) {
    return isZeroList(cast<Expr>(Init), C);
  });
}

/// Whether \p Dx is a call to clad::zero_vector: the tangent vector mode
/// gives a literal.
static bool isZeroVectorCall(const Expr* Dx) {
  const auto* CE = Dx ? dyn_cast<CallExpr>(Dx->IgnoreImplicit()) : nullptr;
  const FunctionDecl* FD = CE ? CE->getDirectCallee() : nullptr;
  return FD && FD->getQualifiedNameAsString() == "clad::zero_vector";
}

Expr* VectorForwardModeVisitor::asTangentVector(Expr* Dx, QualType T) {
  // No tangent at all -- an indirect call, an unsupported statement -- is a
  // zero too, where it is one of numbers: one of an object, or of a pointer
  // or reference to one, has nothing to fill.
  if (!Dx && utils::GetNonConstValueType(T)->isRecordType())
    return Dx;
  ScalarZero Z = Dx ? isScalarZero(Dx, m_Context) : ScalarZero::Pure;
  if (Z == ScalarZero::No)
    return Dx;
  Expr* dCount = buildIndVarCountRef();
  Expr* Zeros = BuildCallExprToCladFunction("zero_vector", {dCount},
                                            {utils::GetNonConstValueType(T)});
  // Keep what the zero's evaluation changes.
  if (Z == ScalarZero::WithEffects)
    return keepEffects(Dx, Zeros);
  return Zeros;
}

Expr* VectorForwardModeVisitor::keepEffects(Expr* E, Expr* Value) {
  // Discarded explicitly: what is left of an effect, such as the negation in
  // `-(x = 2 * x, 0)`, would warn as a value with no use.
  auto* VoidTSI = m_Context.getTrivialTypeSourceInfo(m_Context.VoidTy);
  Expr* Discarded =
      m_Sema.BuildCStyleCastExpr(noLoc, VoidTSI, noLoc, BuildParens(E)).get();
  return BuildParens(BuildOp(BO_Comma, Discarded, Value));
}

void VectorForwardModeVisitor::matchBranchTangents(StmtDiff& TrueDiff,
                                                   StmtDiff& FalseDiff,
                                                   QualType T) {
  // Of objects, a branch without tangents, or with the zero of scalar mode,
  // has zero ones.
  QualType Obj = utils::GetNonConstValueType(T);
  if (Obj->isRecordType() && utils::hasObjectTangents(Obj)) {
    for (StmtDiff* Branch : {&TrueDiff, &FalseDiff}) {
      Expr* Dx = Branch->getExpr_dx();
      if (!Dx || isZeroList(Dx, m_Context))
        Branch->updateStmtDx(zeroObjects(Obj));
    }
    return;
  }
  // A branch that does not depend on the independent variables has a zero:
  // a scalar, which beside the other's vector of tangents would become a
  // vector of its own size, or a vector of zeros of the branch's type, which
  // the other's vector may not convert to. Give it the other's type.
  auto isZero = [this](const Expr* Dx) {
    return isScalarZero(Dx, m_Context) != ScalarZero::No ||
           isZeroVectorCall(Dx);
  };
  auto match = [&](StmtDiff& Vector, StmtDiff& Zero) {
    if (!isTangentVector(Vector.getExpr_dx()) || !isZero(Zero.getExpr_dx()))
      return false;
    QualType Elem =
        tangentElementType(Vector.getExpr_dx(), utils::GetNonConstValueType(T));
    Expr* ZeroDx = Zero.getExpr_dx();
    // A vector of zeros of another type is rebuilt; a scalar one, with
    // whatever its evaluation changes kept.
    if (isZeroVectorCall(ZeroDx))
      ZeroDx = nullptr;
    Zero.updateStmtDx(asTangentVector(ZeroDx, Elem));
    return true;
  };
  if (!match(TrueDiff, FalseDiff))
    match(FalseDiff, TrueDiff);
}

Expr* VectorForwardModeVisitor::castTangent(Expr* Dx, const CastExpr* Cast) {
  // A vector of tangents is one tangent per independent variable, which a
  // cast to a number would not convert but collapse. A cast to a floating
  // type has the derivative of what it casts; one to an integer, a bool or
  // an unscoped enumeration, zero wherever it has one. Scalar mode casts its
  // tangent instead, truncating it. A cast to a reference names what it
  // casts, tangent and all, and is left alone.
  QualType To = Cast->getType();
  // `M{x}` or `static_cast<M>(m)` makes an object, whose tangents those of
  // its members, or of the object it copies, make.
  QualType Obj = utils::GetNonConstValueType(To);
  if (Obj->isRecordType() && utils::hasObjectTangents(Obj)) {
    // `static_cast<M&&>(m)` names m, tangents and all.
    if (Cast->isGLValue())
      return isObjectTangents(Dx) ? Dx : nullptr;
    return objectTangents(Obj, Cast->getSubExpr(), Dx);
  }
  // A complex number is not one: its tangent is not the real one's.
  if (Cast->isGLValue() || !To->isRealType() || !isTangentVector(Dx))
    return nullptr;
  if (To->isRealFloatingType())
    return Dx;
  // The tangents of an expression of type From are a vector of From.
  QualType From = utils::GetNonConstValueType(Cast->getSubExpr()->getType());
  Expr* dCount = buildIndVarCountRef();
  Expr* Zeros = BuildCallExprToCladFunction("zero_vector", {dCount},
                                            {tangentElementType(Dx, From)});
  // What the operand's tangent changes, as in `(int)(x *= 2)`, is kept.
  if (changesState(Dx, m_Context))
    return keepEffects(Dx, Zeros);
  return Zeros;
}

Expr* VectorForwardModeVisitor::zeroObjects(QualType T) {
  Expr* dCount = buildIndVarCountRef();
  // Canonical, so that every call names the same specialization.
  return BuildCallExprToCladFunction("zero_vector", {dCount},
                                     {T.getCanonicalType()});
}

Expr* VectorForwardModeVisitor::objectTangents(QualType T, const Expr* Init,
                                               Expr* InitDx) {
  const Expr* E = Init ? Init->IgnoreImplicit() : nullptr;
  // `M m{x, {y}}`: the tangents of each member, a vector each, make those of
  // the object.
  if (const auto* ILE = dyn_cast_or_null<InitListExpr>(E)) {
    auto* DxILE =
        InitDx ? dyn_cast<InitListExpr>(InitDx->IgnoreImplicit()) : nullptr;
    llvm::SmallVector<Expr*, 4> Args{buildIndVarCountRef()};
    llvm::SmallVector<TemplateArgument, 4> MemberTys;
    auto Field = T->getAsRecordDecl()->field_begin();
    for (unsigned I = 0, N = ILE->getNumInits(); I < N; ++I, ++Field) {
      Expr* Dx =
          DxILE && I < DxILE->getNumInits() ? DxILE->getInit(I) : nullptr;
      QualType MemberTy = utils::GetNonConstValueType(Field->getType());
      if (MemberTy->isRecordType()) {
        Dx = objectTangents(MemberTy, ILE->getInit(I), Dx);
      } else {
        Dx = asTangentVector(Dx, MemberTy);
        // aggregate_vector takes a clad::array of each member's own type.
        QualType ArrTy = utils::GetCladArrayOfType(m_Sema, MemberTy);
        if (!m_Context.hasSameType(Dx->getType(), ArrTy))
          Dx = m_Sema
                   .BuildCStyleCastExpr(
                       noLoc, m_Context.getTrivialTypeSourceInfo(ArrTy), noLoc,
                       BuildParens(Dx))
                   .get();
      }
      Args.push_back(Dx);
      MemberTys.push_back(MemberTy.getCanonicalType());
    }
    std::array<TemplateArgument, 2> TArgs = {
        T.getCanonicalType(),
        TemplateArgument::CreatePackCopy(m_Context, MemberTys)};
    return BuildCallExprToCladFunction("aggregate_vector", Args, TArgs);
  }
  // What has no tangents, such as a global, has zero ones, as has what does
  // not depend on the independent variables: its tangent is the zero of
  // scalar mode, an object, `M()`, or a braced list of zeros.
  if (!InitDx || isZeroList(InitDx, m_Context) ||
      m_Context.hasSameUnqualifiedType(InitDx->getType(), T))
    return zeroObjects(T);
  // Otherwise it copies another object, or the result of a call, and their
  // tangents.
  assert(isObjectTangents(InitDx) && "the tangents of an object are objects");
  return InitDx;
}

Expr* VectorForwardModeVisitor::memberTangent(Expr* BaseDx,
                                              const MemberExpr* ME) {
  if (!isObjectTangents(BaseDx))
    return BaseForwardModeVisitor::memberTangent(BaseDx, ME);
  // A static member is not part of the object: it has no tangents, as a
  // global has none.
  const auto* FD = dyn_cast<FieldDecl>(ME->getMemberDecl());
  if (!FD)
    return nullptr;
  // clad::member_vector(_d_m, &M::k): k out of each of the tangents of m.
  CXXScopeSpec SS;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  utils::BuildNNS(m_Sema, const_cast<RecordDecl*>(FD->getParent()), SS);
  // A qualified reference to the field, which `&` makes a pointer to member
  // and not, inside a method, to the member of `this`.
  Expr* Member = m_Sema.BuildDeclRefExpr(
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      const_cast<FieldDecl*>(FD), FD->getType().getNonReferenceType(),
      VK_LValue, DeclarationNameInfo(FD->getDeclName(), noLoc), &SS);
  Expr* MemberPtr = BuildOp(UnaryOperatorKind::UO_AddrOf, Member);
  llvm::SmallVector<Expr*, 2> Args{BaseDx, MemberPtr};
  QualType ObjTy = tangentElementType(BaseDx, QualType());
  return BuildCallExprToCladFunction(
      "member_vector", Args,
      {ObjTy.getCanonicalType(), FD->getType().getCanonicalType()});
}

Expr* VectorForwardModeVisitor::thisTangentArgument(const CXXMethodDecl* MD,
                                                    const Expr* Base,
                                                    Expr* BaseDx) {
  // The pushforward of a method takes the tangents of its object. Those of a
  // member of an object are a copy, which a method that may change them
  // cannot take.
  if (isMemberTangents(BaseDx) && !MD->isConst()) {
    diag(DiagnosticsEngine::Error, Base->getBeginLoc(),
         "vector mode cannot yet differentiate a call to a non-const method "
         "of a member of an object");
    BaseDx = nullptr;
  }
  if (isObjectTangents(BaseDx))
    return BaseDx;
  // The tangents of a pointer or an array are a vector of what they reach, and
  // not one object per independent variable.
  if (BaseDx)
    diag(DiagnosticsEngine::Error, Base->getBeginLoc(),
         "vector mode cannot yet differentiate a call to a method of an "
         "object reached through a pointer or an array");
  QualType ObjTy = Base->getType();
  if (ObjTy->isPointerType())
    ObjTy = ObjTy->getPointeeType();
  ObjTy = utils::GetNonConstValueType(ObjTy);
  // An object without tangents has zero ones, which a method may change.
  VarDecl* dBaseVD =
      BuildVarDecl(utils::GetCladArrayOfType(m_Sema, ObjTy), "_d_base",
                   zeroObjects(ObjTy), /*DirectInit=*/true);
  addToCurrentBlock(BuildDeclStmt(dBaseVD));
  return BuildDeclRef(dBaseVD);
}

StmtDiff VectorForwardModeVisitor::VisitCallExpr(const CallExpr* CE) {
  const auto* OCE = dyn_cast<CXXOperatorCallExpr>(CE);
  if (!OCE || !utils::isObjectTangentsAssignment(OCE->getDirectCallee()))
    return BaseForwardModeVisitor::VisitCallExpr(CE);
  // `a = b` of objects copies b into a, and so its tangents into a's.
  StmtDiff L = Visit(OCE->getArg(0));
  StmtDiff R = Visit(OCE->getArg(1));
  QualType T = utils::GetNonConstValueType(OCE->getArg(0)->getType());
  Expr* Value = BuildOp(BO_Assign, L.getExpr(), R.getExpr());
  Expr* LDx = L.getExpr_dx();
  if (isMemberTangents(LDx) || !isObjectTangents(LDx)) {
    diag(DiagnosticsEngine::Error, OCE->getBeginLoc(),
         "vector mode cannot yet differentiate an assignment to %select{a "
         "member of an object|an object without a derivative}0")
        << !isMemberTangents(LDx);
    return {Value, LDx};
  }
  Expr* RDx = objectTangents(T, OCE->getArg(1), R.getExpr_dx());
  return {Value, BuildOp(BO_Assign, LDx, RDx)};
}

StmtDiff
VectorForwardModeVisitor::VisitIntegerLiteral(const clang::IntegerLiteral* IL) {
  Expr* dCount = buildIndVarCountRef();
  auto* zero_vec =
      BuildCallExprToCladFunction("zero_vector", {dCount}, {IL->getType()});
  return StmtDiff(Clone(IL), zero_vec);
}

} // namespace clad
