#include "clad/Differentiator/DerivedTypeInitialiser.hpp"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Sema/Lookup.h"
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
      : m_Sema(semaRef), m_Context(semaRef.getASTContext()), m_CurScope(semaRef.TUScope),
        m_ASTHelper(semaRef), m_yQType(yType), m_xQType(xType),
        m_DerivedRecord(derivedRecord),
        m_DerivedType(
            derivedRecord->getTypeForDecl()->getCanonicalTypeInternal()) {
    FillDerivedRecord();              
    m_DerivedAddFnDecl = CreateDerivedAddFn();
  }
  
  void DerivedTypeInitialiser::beginScope(unsigned ScopeFlags) {
      // FIXME: since Sema::CurScope is private, we cannot access it and have
      // to use separate member variable m_CurScope. The only options to set
      // CurScope of Sema seemt to be through Parser or ContextAndScopeRAII.
      m_CurScope =
          new clang::Scope(m_CurScope, ScopeFlags, m_Sema.Diags);
    }

  void DerivedTypeInitialiser::endScope() {
      // This will remove all the decls in the scope from the IdResolver.
      m_Sema.ActOnPopScope(clang::SourceLocation(), m_CurScope);
      auto oldScope = m_CurScope;
      m_CurScope = oldScope->getParent();
      delete oldScope;
    }

  void DerivedTypeInitialiser::FillDerivedRecord() {
    auto xRD = m_xQType->getAsCXXRecordDecl();
    for (auto field : xRD->fields()) {
      auto FD = m_ASTHelper.BuildFieldDecl(m_DerivedRecord,
                                           field->getIdentifier(), m_yQType);
      FD->setAccess(AccessSpecifier::AS_public);
      m_DerivedRecord->addDecl(FD);
    }
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
    // auto cladND = BuildCladNamespace();
    auto fnType = ComputeDerivedAddSubFnType();
    auto dAddDNameInfo = m_ASTHelper.CreateDeclNameInfo("dAdd");
    auto TSI = m_Context.getTrivialTypeSourceInfo(fnType);
    llvm::SaveAndRestore<DeclContext*> saveContext(m_Sema.CurContext);
    auto FD = FunctionDecl::Create(m_Context, m_Sema.CurContext, noLoc, dAddDNameInfo,
                                   fnType, TSI, StorageClass::SC_None, false,
                                   true, ConstexprSpecKind::CSK_unspecified,
                                   nullptr);
    // cladND->addDecl(FD);
    // m_Sema.CurContext->addDecl(FD);
    // FD->setAccess(AccessSpecifier::AS_public);
    m_Sema.CurContext = FD->getDeclContext();

    beginScope(Scope::FunctionPrototypeScope | Scope::FunctionDeclarationScope |
               Scope::DeclScope);
    m_Sema.PushFunctionScope();
    m_Sema.PushDeclContext(m_CurScope, FD);

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

    auto returnExpr = m_Sema.ActOnReturnStmt(noLoc, m_Variables["d_res"], m_CurScope).get();
    block.push_back(returnExpr);
    auto CS = clad_compat::CompoundStmt_Create(m_Context, block, noLoc, noLoc);
    FD->setBody(CS);
    m_Sema.PopFunctionScopeInfo();
    m_Sema.PopDeclContext();
    endScope(); // Function decl scope

    LookupResult R(m_Sema, FD->getNameInfo(), Sema::LookupOrdinaryName);
    m_Sema.LookupQualifiedName(R, FD->getDeclContext(),
                                /*allowBuiltinCreation*/ false);
    FD->getDeclContext()->addDecl(FD);
    for (NamedDecl* D : R) {
      if (auto anotherFD = dyn_cast<FunctionDecl>(D)) {
        if (m_Sema.getASTContext()
                .hasSameFunctionTypeIgnoringExceptionSpec(FD->getType(),
                                                          anotherFD->getType())) {
          llvm::errs()<<"Same function found\n";
          // Register the function on the redecl chain.
          FD->setPreviousDecl(anotherFD);
        }
      }
    }
    llvm::errs() << "Dumping derived add fn\n";
    LangOptions langOpts;
    langOpts.CPlusPlus = true;
    clang::PrintingPolicy policy(langOpts);
    policy.Bool = true;
    FD->print(llvm::errs(), policy);
    llvm::errs()<<"FD address: "<<FD<<"\n";
    FD->dump();
    llvm::errs() << "isDefined: " << FD->isDefined() << " "
                 << "hasBody: " << FD->hasBody() << "\n";
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
