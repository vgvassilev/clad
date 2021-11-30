#include "clad/Differentiator/DerivedTypeInitialiser.hpp"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SaveAndRestore.h"
#include "clad/Differentiator/Compatibility.h"

#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/LangOptions.h"

using namespace clang;

namespace clad {
  static SourceLocation noLoc;

  DerivedTypeInitialiser::DerivedTypeInitialiser(Sema& semaRef, QualType yType,
                                                 QualType xType,
                                                 CXXRecordDecl* derivedRecord)
      : m_Sema(semaRef), m_Context(semaRef.getASTContext()),
        m_ASTHelper(semaRef), m_yQType(yType), m_xQType(xType),
        m_DerivedRecord(derivedRecord),
        m_DerivedType(
            derivedRecord->getTypeForDecl()->getCanonicalTypeInternal()) {
    m_DerivedAddFnDecl = CreateDerivedAddFn();
  }

  NamespaceDecl* DerivedTypeInitialiser::BuildCladNamespace() {
    auto prevCladND = m_ASTHelper.FindCladNamespace();
    auto newCladND = NamespaceDecl::Create(m_Context, m_Sema.CurContext, false,
                                           noLoc, noLoc,
                                           prevCladND->getIdentifier(),
                                           prevCladND);
    m_Sema.CurContext->addDecl(newCladND);
    newCladND->setAccess(AccessSpecifier::AS_public);
    return newCladND;
  }

  clang::QualType DerivedTypeInitialiser::GetDerivedParamType() const {
    auto paramType = m_Context.getLValueReferenceType(
        m_DerivedType.withConst());
    return paramType;
  }

  clang::QualType DerivedTypeInitialiser::GetNonDerivedParamType() const {
    return m_yQType;
  }

  QualType DerivedTypeInitialiser::ComputeDerivedAddSubFnType() const {
    auto paramType = GetDerivedParamType();
    llvm::SmallVector<QualType, 2> paramTypes = {paramType, paramType};
    auto returnType = m_DerivedType;
    auto fnType = m_Context.getFunctionType(returnType, paramTypes,
                                            FunctionProtoType::ExtProtoInfo());
    return fnType;
  }

  clang::FunctionDecl* DerivedTypeInitialiser::CreateDerivedAddFn() {
    m_Variables.clear();
    auto cladND = BuildCladNamespace();
    auto fnType = ComputeDerivedAddSubFnType();
    auto dAddDNameInfo = m_ASTHelper.CreateDeclNameInfo("dAdd");
    auto TSI = m_Context.getTrivialTypeSourceInfo(fnType);
    llvm::SaveAndRestore<DeclContext*> saveContext(m_Sema.CurContext);
    auto FD = FunctionDecl::Create(m_Context, cladND, noLoc, dAddDNameInfo,
                                   fnType, TSI, StorageClass::SC_None, false,
                                   true, ConstexprSpecKind::CSK_unspecified,
                                   nullptr);
    cladND->addDecl(FD);
    FD->setAccess(AccessSpecifier::AS_public);
    m_Sema.CurContext = FD;

    llvm::SmallVector<Stmt*, 16> block;
    auto params = CreateDerivedAddSubFnParams();
    FD->setParams(params);
    auto resVD = m_ASTHelper.BuildVarDecl(m_Sema.CurContext,
                                          &m_Context.Idents.get("d_res"),
                                          m_DerivedType);
    block.push_back(m_ASTHelper.BuildDeclStmt(resVD));
    for (auto param : params)
      m_Variables[param->getNameAsString()] = m_ASTHelper.BuildDeclRefExpr(
          param);
    m_Variables[resVD->getNameAsString()] = m_ASTHelper.BuildDeclRefExpr(resVD);
    for (auto field : m_DerivedRecord->fields()) {
      if (!(field->getType()->isRealType()))
        continue;
      auto dResMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_res"], field);
      auto dAMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_a"], field);
      auto dBMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_b"], field);
      auto addDerivExpr = m_Sema
                              .BuildBinOp(nullptr, noLoc,
                                          BinaryOperatorKind::BO_Add, dAMem,
                                          dBMem)
                              .get();
      auto assignExpr = m_Sema
                            .BuildBinOp(nullptr, noLoc,
                                        BinaryOperatorKind::BO_Assign, dResMem,
                                        addDerivExpr)
                            .get();
      block.push_back(assignExpr);
    }
    auto CS = clad_compat::CompoundStmt_Create(m_Context, block, noLoc, noLoc);
    FD->setBody(CS);
    llvm::errs() << "Dumping derived add fn\n";
    LangOptions langOpts;
    langOpts.CPlusPlus = true;
    clang::PrintingPolicy policy(langOpts);
    policy.Bool = true;
    FD->print(llvm::errs(), policy);
    llvm::errs()<<"FD address: "<<FD<<"\n";
    FD->dump();

    return FD;
  }

  llvm::SmallVector<clang::ParmVarDecl*, 2>
  DerivedTypeInitialiser::CreateDerivedAddSubFnParams() {
    auto d_a = m_ASTHelper.BuildParmVarDecl(m_Sema.CurContext,
                                            &m_Context.Idents.get("d_a"),
                                            GetDerivedParamType());
    auto d_b = m_ASTHelper.BuildParmVarDecl(m_Sema.CurContext,
                                            &m_Context.Idents.get("d_b"),
                                            GetDerivedParamType());
    llvm::SmallVector<clang::ParmVarDecl*, 2> params{d_a, d_b};
    return params;
  }

  DerivedTypeEssentials DerivedTypeInitialiser::CreateDerivedTypeEssentials() {
    return DerivedTypeEssentials(m_DerivedAddFnDecl);
  }
} // namespace clad
