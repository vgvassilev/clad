//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// The Enzyme backend half of reverse mode: instead of a derivative body, emit
// a call to __enzyme_autodiff and let Enzyme differentiate the emitted IR.
// Kept apart from ReverseModeVisitor.cpp because it shares none of that file's
// machinery -- no visiting, no tape, no adjoint statements.
//----------------------------------------------------------------------------//

#include "clad/Differentiator/ReverseModeVisitor.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/Type.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/DerivativeBuilder.h"

#include <algorithm>
#include <string>

using namespace clang;

namespace clad {
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

  // First add the function itself as a parameter/argument.
  // FIXME: BuildDeclRef takes a DeclaratorDecl*, and Sema needs a mutable
  // ValueDecl underneath, so the cast cannot simply be pushed down.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  auto* origFn = const_cast<FunctionDecl*>(m_DiffReq.Function);
  enzymeArgs.push_back(BuildDeclRef(origFn));
  // FIXME: We should not use const_cast to get the decl context here.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  auto* fdDeclContext = const_cast<DeclContext*>(m_DiffReq->getDeclContext());
  enzymeParams.push_back(m_Sema.BuildParmVarDeclForTypedef(
      fdDeclContext, noLoc, m_DiffReq->getType()));

  // The derivative takes the original parameters followed by one adjoint per
  // *requested* variable, in request order. Locate each parameter's adjoint
  // through that request rather than assuming every parameter has one at a
  // fixed offset, which holds only when the whole signature is
  // differentiated.
  llvm::SmallVector<ParmVarDecl*, 16> adjointOf(numParams, nullptr);
  for (unsigned j = 0, e = m_DiffReq.DVI.size(); j < e; ++j)
    for (unsigned i = 0; i < numParams; ++i)
      if (m_DiffReq.DVI[j].param == origParams[i])
        adjointOf[i] = paramsRef[numParams + j];
  bool allParamsActive =
      std::none_of(adjointOf.begin(), adjointOf.end(),
                   [](const ParmVarDecl* PVD) { return !PVD; });

  // Enzyme's positional convention cannot express an inactive argument:
  // omitting a shadow shifts every later argument. Annotate activity
  // explicitly when the request leaves anything out, and keep the positional
  // form otherwise so a whole-signature request emits exactly what it did.
  // Set when a parameter cannot be described to Enzyme at all, so the call
  // is dropped rather than emitted with arguments that no longer line up.
  bool markersResolve = true;
  auto marker = [&](llvm::StringRef name) {
    if (allParamsActive)
      return;
    Expr* ref = utils::BuildEnzymeActivityMarkerRef(m_Sema, name);
    enzymeArgs.push_back(ref);
    enzymeParams.push_back(m_Sema.BuildParmVarDeclForTypedef(
        fdDeclContext, noLoc, ref->getType()));
  };

  // Add rest of the parameters/arguments
  for (unsigned i = 0; i < numParams; i++) {
    QualType paramType = origParams[i]->getOriginalType();
    bool isActive = adjointOf[i] != nullptr;

    if (!isActive)
      marker("enzyme_const");
    else if (utils::isArrayOrPointerType(paramType))
      marker("enzyme_dup");
    else if (paramType->isRealFloatingType())
      marker("enzyme_out");
    else if (!allParamsActive) {
      // An active parameter that is neither a pointer nor a real -- a
      // record, or a request naming one of its fields -- has no shadow to
      // pass and no slot in the returned struct. Emitting the call anyway
      // leaves its adjoint untouched and the gradient silently short. A
      // whole-signature request keeps its long-standing behaviour of
      // passing such a parameter through and leaving its adjoint at zero.
      diag(DiagnosticsEngine::Error, m_DiffReq.CallContext->getBeginLoc(),
           "cannot differentiate '%0' with the Enzyme backend: only real "
           "and pointer parameters are supported")
          << origParams[i]->getName();
      markersResolve = false;
    }

    // First Add the original parameter
    enzymeArgs.push_back(BuildDeclRef(paramsRef[i]));
    enzymeParams.push_back(m_Sema.BuildParmVarDeclForTypedef(
        fdDeclContext, noLoc, paramsRef[i]->getType()));

    if (!isActive)
      continue;

    // If original parameter is of a differentiable real type(but not
    // array/pointer), then add it to the list of params whose gradient must
    // be extracted later from the EnzymeGradient structure
    if (paramType->isRealFloatingType()) {
      enzymeRealParams.push_back(paramsRef[i]);
      enzymeRealParamsDerived.push_back(adjointOf[i]);
    } else if (utils::isArrayOrPointerType(paramType)) {
      // Add the corresponding array/pointer variable
      enzymeArgs.push_back(BuildDeclRef(adjointOf[i]));
      enzymeParams.push_back(m_Sema.BuildParmVarDeclForTypedef(
          fdDeclContext, noLoc, adjointOf[i]->getType()));
    }
  }

  if (!markersResolve)
    return;

  llvm::SmallVector<QualType, 16> enzymeParamsType;
  for (auto* i : enzymeParams)
    enzymeParamsType.push_back(i->getType());

  QualType QT;
  if (!enzymeRealParams.empty()) {
    // Find the EnzymeGradient datastructure
    auto* gradDecl =
        utils::LookupTemplateDeclInCladNamespace(m_Sema, "EnzymeGradient");

    TemplateArgumentListInfo TLI{};
    llvm::APSInt argValue = m_Context.MakeIntValue(enzymeRealParams.size(),
                                                   m_Context.UnsignedIntTy);
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
  SourceLocation loc = m_DiffReq->getLocation();
  FunctionDecl* enzymeCallFD = FunctionDecl::Create(
      m_Context, fdDeclContext, loc, loc, nameEnzyme, enzymeFunctionType,
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
      llvm::APSInt V = m_Context.MakeIntValue(i, m_Context.UnsignedIntTy);
      Expr* gradIndex = dyn_cast<Expr>(
          IntegerLiteral::Create(m_Context, V, m_Context.UnsignedIntTy, noLoc));
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
} // end namespace clad
