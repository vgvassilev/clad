#include "clad/Differentiator/DerivedTypeInitialiser.h"

#include "ConstantFolder.h"
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
      : m_Sema(semaRef), m_Context(semaRef.getASTContext()),
        m_CurScope(semaRef.TUScope), m_ASTHelper(semaRef), m_yQType(yType),
        m_xQType(xType), m_DerivedRecord(derivedRecord),
        m_DerivedType(
            derivedRecord->getTypeForDecl()->getCanonicalTypeInternal()) {
    FillDerivedRecord();
    if (m_yQType == m_xQType) {
      CreateInitialiseSeedsFn();
    }
    if (m_yQType->isRealType()) {
      m_DerivedAddFnDecl = CreateDerivedAddFn();
    }
  }

  void DerivedTypeInitialiser::beginScope(unsigned ScopeFlags) {
    m_CurScope = new clang::Scope(m_CurScope, ScopeFlags, m_Sema.Diags);
  }

  void DerivedTypeInitialiser::endScope() {
    m_Sema.ActOnPopScope(noLoc, m_CurScope);
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
    auto fnQType = ComputeDerivedAddSubFnType();
    auto dAddDNameInfo = m_ASTHelper.CreateDeclName("dAdd");
    llvm::SaveAndRestore<DeclContext*> saveContext(m_Sema.CurContext);
    auto FD = m_ASTHelper.BuildFnDecl(m_Sema.CurContext, dAddDNameInfo,
                                      fnQType);

    m_Sema.CurContext = FD->getDeclContext();

    beginScope(ASTHelper::Scope::FunctionBeginScope);
    m_Sema.PushFunctionScope();
    m_Sema.PushDeclContext(m_CurScope, FD);

    llvm::SmallVector<Stmt*, 16> block;

    auto params = CreateDerivedAddSubFnParams();
    FD->setParams(params);

    // Function body scope
    beginScope(ASTHelper::Scope::FunctionBodyScope);

    auto resVD = m_ASTHelper.BuildVarDecl(m_Sema.CurContext,
                                          &m_Context.Idents.get("d_res"),
                                          m_DerivedType);
    block.push_back(m_ASTHelper.BuildDeclStmt(resVD));

    ComputeAndStoreDRE({resVD, params[0], params[1]});

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

    auto returnExpr = m_Sema
                          .ActOnReturnStmt(noLoc, m_Variables["d_res"],
                                           m_CurScope)
                          .get();
    block.push_back(returnExpr);
    auto CS = m_ASTHelper.BuildCompoundStmt(block);
    FD->setBody(CS);
    endScope();
    m_Sema.PopFunctionScopeInfo();
    m_Sema.PopDeclContext();
    endScope(); // Function decl scope

    m_ASTHelper.RegisterFn(FD->getDeclContext(), FD);
    llvm::errs() << "Dumping derived add fn\n";
    // LangOptions langOpts;
    // langOpts.CPlusPlus = true;
    // clang::PrintingPolicy policy(langOpts);
    // policy.Bool = true;
    // FD->print(llvm::errs(), policy);
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

  void
  DerivedTypeInitialiser::ComputeAndStoreDRE(llvm::ArrayRef<ValueDecl*> decls) {
    for (auto decl : decls) {
      m_Variables[decl->getNameAsString()] = m_ASTHelper.BuildDeclRefExpr(decl);
    }
  }

  DerivedTypeEssentials DerivedTypeInitialiser::CreateDerivedTypeEssentials() {
    return DerivedTypeEssentials(m_DerivedAddFnDecl);
  }

  FunctionDecl* DerivedTypeInitialiser::CreateInitialiseSeedsFn() {
    m_Variables.clear();
    auto memFnType = ComputeInitialiseSeedsFnType();
    llvm::SaveAndRestore<DeclContext*> saveContext(m_Sema.CurContext);
    auto memFn = m_ASTHelper.BuildMemFnDecl(m_DerivedRecord,
                                            m_ASTHelper.CreateDeclNameInfo(
                                                "InitialiseSeeds"),
                                            memFnType);
    m_Sema.CurContext = memFn->getDeclContext();

    beginScope(ASTHelper::Scope::FunctionBeginScope);
    m_Sema.PushFunctionScope();
    m_Sema.PushDeclContext(m_CurScope, memFn);

    llvm::SmallVector<Stmt*, 16> block;

    memFn->setParams({});

    // Function body scope
    beginScope(ASTHelper::Scope::FunctionBodyScope);
    auto thisExpr = clad_compat::Sema_BuildCXXThisExpr(m_Sema, memFn);

    for (auto field : m_DerivedRecord->fields()) {
      auto RD = field->getType()->getAsCXXRecordDecl();
      FieldDecl* independentField = nullptr;
      LookupResult R(m_Sema, field->getDeclName(), noLoc,
                     Sema::LookupNameKind::LookupMemberName);
      CXXScopeSpec CSS();
      m_Sema.LookupQualifiedName(R, RD, CSS);
      if (R.isSingleResult()) {
        if (auto decl = dyn_cast<FieldDecl>(R.getFoundDecl())) {
          independentField = decl;
        }
      }
      if (!independentField || !(independentField->getType()->isRealType()))
        continue;
      auto baseExpr = m_ASTHelper.BuildMemberExpr(thisExpr, field);
      auto independentFieldExpr = m_ASTHelper.BuildMemberExpr(baseExpr,
                                                              independentField);
      independentFieldExpr->dumpColor();
      auto assignExpr = m_Sema
                            .BuildBinOp(m_CurScope, noLoc,
                                        BinaryOperatorKind::BO_Assign,
                                        independentFieldExpr,
                                        ConstantFolder::
                                            synthesizeLiteral(independentField
                                                                  ->getType(),
                                                              m_Context, 1))
                            .get();
      assignExpr->dumpColor();                            
      block.push_back(assignExpr);
    }
    auto CS = m_ASTHelper.BuildCompoundStmt(block);
    memFn->setBody(CS);
    endScope();
    m_Sema.PopFunctionScopeInfo();
    m_Sema.PopDeclContext();
    endScope(); // Function decl scope

    m_ASTHelper.RegisterFn(memFn->getDeclContext(), memFn);
    llvm::errs() << "Dumping derived add fn\n";
    LangOptions langOpts;
    langOpts.CPlusPlus = true;
    clang::PrintingPolicy policy(langOpts);
    policy.Bool = true;
    memFn->print(llvm::errs(), policy);
    return memFn;
  }

  QualType DerivedTypeInitialiser::ComputeInitialiseSeedsFnType() const {
    auto returnType = m_Context.VoidTy;
    auto fnType = m_Context.getFunctionType(returnType, {},
                                            FunctionProtoType::ExtProtoInfo());
    return fnType;
  }
} // namespace clad
