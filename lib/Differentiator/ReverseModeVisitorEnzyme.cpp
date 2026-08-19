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
#include "clang/AST/Expr.h"
#include "clang/Sema/Sema.h"

#include "clad/Differentiator/CladUtils.h"

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
