#include "clad/Differentiator/JacobianVectorProduct.h"
#include "clad/Differentiator/DerivativeBuilder.h"
#include "clad/Differentiator/DiffPlanner.h"
#include <clad/Differentiator/CladUtils.h>
#include <clad/Differentiator/DiffMode.h>
#include <clang/AST/Decl.h>
#include <string>
#include "./JacobianModeVisitor.h"

using namespace clad;
using namespace clang;

JacobianVectorProductModeVisitor::JacobianVectorProductModeVisitor(
    DerivativeBuilder& builder, const DiffRequest& request)
    : m_ModifiedDiffRequest(request),
      JacobianModeVisitor(builder, m_ModifiedDiffRequest) {
  m_ModifiedDiffRequest.Mode = DiffMode::jacobian;
}

DerivativeAndOverload JacobianVectorProductModeVisitor::Derive() {
  auto args = m_DiffReq.DVI;
  SourceLocation loc{m_DiffReq->getLocation()};
  auto* FD = m_ModifiedDiffRequest.Function;

  auto out = args.back();

  std::string jvp_name = m_DiffReq.BaseFunctionName + std::string{"_jvp"};
  std::cout << jvp_name << "\n";

  FunctionDecl* jacobian_fn = JacobianModeVisitor::Derive().overload;
  auto* DC = const_cast<clad_compat::DeclContext*>(m_DiffReq->getDeclContext());
  IdentifierInfo* II = &m_Context.Idents.get(jvp_name);
  DeclarationNameInfo name(II, loc);
  SmallVector<ParmVarDecl*> params{};
  SmallVector<QualType> types{};

  for (size_t p = 0; p < m_ModifiedDiffRequest.Function->parameters().size() - 1; p++) {
    params.push_back(m_ModifiedDiffRequest.Function->parameters()[p]);
    types.push_back(m_ModifiedDiffRequest.Function->parameters()[p]->getType());
  }

  auto* out_param = utils::BuildParmVarDecl(m_Sema, m_Sema.CurContext, II, m_ModifiedDiffRequest.DVI.back().jvp_out->getType());
  params.push_back(out_param);
  types.push_back(out_param->getType());

  auto& C = m_Sema.getASTContext();
  auto fn_type = C.getFunctionType(C.VoidTy, types, FunctionProtoType::ExtProtoInfo {});
  FunctionDecl* jvp_fn = FunctionDecl::Create(C, m_Sema.CurContext, noLoc, name, fn_type, nullptr, FD->getStorageClass() CLAD_COMPAT_FunctionDecl_UsesFPIntrin_Param(FD),
          FD->isInlineSpecified(), FD->hasWrittenPrototype(),
          FD->getConstexprKind(), CLAD_COMPAT_CLANG21_getTrailingRequiresClause(FD));

  m_Derivative = jvp_fn;

  // TODO

  DerivativeAndOverload d{m_Derivative, nullptr};
  return d;
}
