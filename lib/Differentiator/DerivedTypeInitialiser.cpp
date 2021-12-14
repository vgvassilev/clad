#include "clad/Differentiator/DerivedTypeInitialiser.h"

#include "ConstantFolder.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SaveAndRestore.h"
#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"
#include "clad/Differentiator/DerivedTypesHandler.h"

#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/LangOptions.h"

using namespace clang;

namespace clad {
  static SourceLocation noLoc;

  DerivedTypeInitialiser::DerivedTypeInitialiser(ASTConsumer& consumer,
                                                 Sema& semaRef,
                                                 DerivedTypesHandler& DTH,
                                                 QualType yType, QualType xType,
                                                 CXXRecordDecl* derivedRecord)
      : m_Sema(semaRef), m_Context(semaRef.getASTContext()), m_DTH(DTH),
        m_CurScope(semaRef.TUScope), m_ASTHelper(semaRef), m_yQType(yType),
        m_xQType(xType), m_DerivedRecord(derivedRecord) {
    assert((m_xQType->isRealType() || m_xQType->isClassType()) &&
           "x should either be of a real type or a class type");
    assert((m_yQType->isRealType() || m_yQType->isClassType()) &&
           "y should either be of a real type or a class type");
    BuildDerivedRecordDefinition();
    FillDerivedRecord();
    if (m_yQType == m_xQType) {
      m_InitialiseSeedsFn = BuildInitialiseSeedsFn();
    }
    if (m_yQType->isRealType()) {
      m_DerivedAddFn = BuildDerivedAddFn();
      m_DerivedSubFn = BuildDerivedSubFn();
      m_DerivedMultiplyFn = BuildDerivedMultiplyFn();
      m_DerivedDivideFn = BuildDerivedDivideFn();
    }
    ProcessTopLevelDeclarations(consumer);
  }

  void DerivedTypeInitialiser::BuildDerivedRecordDefinition() {
    // TODO: Properly handle CXXScopeSpec;
    CXXScopeSpec CSS;
    ParsedAttributesView PAV;
    bool ownedDecl, isDependent;
    TypeResult underlyingType;
    beginScope(Scope::DeclScope | Scope::ClassScope);
    auto temp = CXXRecordDecl::Create(m_Context, TagTypeKind::TTK_Class,
                                      m_Sema.CurContext, noLoc, noLoc,
                                      m_DerivedRecord->getIdentifier(),
                                      m_DerivedRecord);
    temp->startDefinition();
    temp->completeDefinition();
    m_DerivedRecord = cast<CXXRecordDecl>(temp);
    m_DerivedType = m_DerivedRecord->getTypeForDecl()
                        ->getCanonicalTypeInternal();
    endScope();
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
    if (m_yQType->isRealType()) {
      auto xRD = m_xQType->getAsCXXRecordDecl();
      for (auto field : xRD->fields()) {
        QualType derivedFieldType;
        if (field->getType()->isRealType()) {
          derivedFieldType = m_Context.DoubleTy;
        } else if (field->getType()->isClassType()) {
          derivedFieldType = m_DTH.GetDerivedType(m_Context.DoubleTy,
                                                  field->getType());
          assert(!derivedFieldType.isNull() &&
                 "Required derived field type not found!!");
        } else {
          continue;
        }
        auto FD = m_ASTHelper.BuildFieldDecl(m_DerivedRecord,
                                             field->getIdentifier(),
                                             derivedFieldType,
                                             AccessSpecifier::AS_public, true);
      }
    } else if (m_yQType->isClassType()) {
      auto yRD = m_yQType->getAsCXXRecordDecl();
      for (auto field : yRD->fields()) {
        QualType derivedFieldType;
        if (field->getType()->isRealType()) {
          if (m_xQType->isRealType()) {
            derivedFieldType = m_Context.DoubleTy;
          } else if (m_xQType->isClassType()) {
            derivedFieldType = m_DTH.GetDerivedType(m_Context.DoubleTy,
                                                    m_xQType);
            assert(!derivedFieldType.isNull() &&
                   "Required derived field type not found!!");
          }
        } else if (field->getType()->isClassType()) {
          derivedFieldType = m_DTH.GetDerivedType(field->getType(), m_xQType);
          assert(!derivedFieldType.isNull() &&
                 "Required derived field type not found!!");
        } else {
          continue;
        }
        auto FD = m_ASTHelper.BuildFieldDecl(m_DerivedRecord,
                                             field->getIdentifier(),
                                             derivedFieldType,
                                             AccessSpecifier::AS_public, true);
      }
    } else {
      assert("Unexpected case!! y Type should either be a real type or a class "
             "type.");
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

  clang::QualType DerivedTypeInitialiser::ComputeDerivedMultiplyDivideFnType() {
    auto paramTypes = {GetNonDerivedParamType(), GetDerivedParamType(),
                       GetNonDerivedParamType(), GetDerivedParamType()};
    auto returnType = m_DerivedType;
    auto fnType = m_Context.getFunctionType(returnType, paramTypes,
                                            FunctionProtoType::ExtProtoInfo());
  }

  template <class ComputeDerivedFnTypeT, class BuildDerivedFnParamsT,
            class BuildDiffBodyT>
  clang::FunctionDecl* DerivedTypeInitialiser::GenerateDerivedArithmeticFn(
      clang::DeclarationName fnDName,
      ComputeDerivedFnTypeT ComputeDerivedFnType,
      BuildDerivedFnParamsT BuildDerivedFnParams,
      BuildDiffBodyT buildDiffBody) {
    m_Variables.clear();
    auto fnQType = (this->*ComputeDerivedFnType)();
    llvm::SaveAndRestore<DeclContext*> saveContext(m_Sema.CurContext);
    auto FD = m_ASTHelper.BuildFnDecl(m_Sema.CurContext, fnDName, fnQType);

    m_Sema.CurContext = FD->getDeclContext();

    beginScope(ASTHelper::CustomScope::FunctionBeginScope);
    m_Sema.PushFunctionScope();
    m_Sema.PushDeclContext(m_CurScope, FD);

    auto params = (this->*BuildDerivedFnParams)();
    FD->setParams(params);

    // Function body scope
    beginScope(ASTHelper::CustomScope::FunctionBodyScope);

    BeginBlock();

    auto resVD = m_ASTHelper.BuildVarDecl(m_Sema.CurContext,
                                          &m_Context.Idents.get("d_res"),
                                          m_DerivedType);
    AddToCurrentBlock(m_ASTHelper.BuildDeclStmt(resVD));

    ComputeAndStoreDRE(resVD);
    for (auto param : params)
      ComputeAndStoreDRE(param);

    buildDiffBody();

    FD->setBody(EndBlock());
    endScope();
    m_Sema.PopFunctionScopeInfo();
    m_Sema.PopDeclContext();
    endScope(); // Function decl scope
    m_ASTHelper.RegisterFn(FD->getDeclContext(), FD);

    // llvm::errs() << "Dumping derived add fn\n";
    // LangOptions langOpts;
    // langOpts.CPlusPlus = true;
    // clang::PrintingPolicy policy(langOpts);
    // policy.Bool = true;
    // FD->print(llvm::errs(), policy);

    return FD;
  }

  clang::FunctionDecl* DerivedTypeInitialiser::BuildDerivedAddFn() {

    auto buildDiffBody = [this]() {
      for (auto field : m_DerivedRecord->fields()) {
        auto dResMem = m_ASTHelper.BuildMemberExpr(this->m_Variables["d_res"],
                                                   field);
        auto dAMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_a"], field);
        auto dBMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_b"], field);
        if (field->getType()->isRealType()) {
          auto addDerivExpr = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Add,
                                                  dAMem, dBMem);
          auto assignExpr = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Assign,
                                                dResMem, addDerivExpr);
          AddToCurrentBlock(assignExpr);
        } else if (field->getType()->isClassType()) {
          auto DTE = m_DTH.GetDTE(field->getType().getAsString());
          assert(DTE.isValid() && "Required Derived Type Essesentials not found!!");
          auto memberDAddFn = DTE.GetDerivedAddFn();
          // auto dAddFnCall = m_ASTHelper.BuildCall
        }
      }

      auto returnExpr = m_ASTHelper.BuildReturnStmt(m_Variables["d_res"],
                                                    m_CurScope);
      AddToCurrentBlock(returnExpr);
    };
    auto FD = GenerateDerivedArithmeticFn(
        m_ASTHelper.CreateDeclName("dAdd"),
        &DerivedTypeInitialiser::ComputeDerivedAddSubFnType,
        &DerivedTypeInitialiser::BuildDerivedAddSubFnParams, buildDiffBody);
    return FD;
  }

  clang::FunctionDecl* DerivedTypeInitialiser::BuildDerivedSubFn() {
    auto buildDiffBody = [this]() {
      for (auto field : m_DerivedRecord->fields()) {
        if (!(field->getType()->isRealType()))
          continue;
        auto dResMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_res"], field);
        auto dAMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_a"], field);
        auto dBMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_b"], field);
        auto addDerivExpr = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Sub,
                                                dAMem, dBMem);
        auto assignExpr = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Assign,
                                              dResMem, addDerivExpr);
        AddToCurrentBlock(assignExpr);
      }

      auto returnExpr = m_Sema
                            .ActOnReturnStmt(noLoc, m_Variables["d_res"],
                                             m_CurScope)
                            .get();
      AddToCurrentBlock(returnExpr);
    };
    auto FD = GenerateDerivedArithmeticFn(
        m_ASTHelper.CreateDeclName("dSub"),
        &DerivedTypeInitialiser::ComputeDerivedAddSubFnType,
        &DerivedTypeInitialiser::BuildDerivedAddSubFnParams, buildDiffBody);
    return FD;
  }

  clang::FunctionDecl* DerivedTypeInitialiser::BuildDerivedMultiplyFn() {
    auto buildDiffBody = [this]() {
      auto a = m_Variables["a"];
      auto b = m_Variables["b"];
      for (auto field : m_DerivedRecord->fields()) {
        if (!(field->getType()->isRealType()))
          continue;
        auto dResMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_res"], field);
        auto dAMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_a"], field);
        auto dBMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_b"], field);

        auto diff1 = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Mul, dAMem, b);

        auto diff2 = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Mul, a, dBMem);

        auto addDiffs = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Add, diff1,
                                            diff2);
        auto assignExpr = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Assign,
                                              dResMem, addDiffs);
        AddToCurrentBlock(assignExpr);
      }
      auto returnExpr = m_Sema
                            .ActOnReturnStmt(noLoc, m_Variables["d_res"],
                                             m_CurScope)
                            .get();
      AddToCurrentBlock(returnExpr);
    };
    auto FD = GenerateDerivedArithmeticFn(
        m_ASTHelper.CreateDeclName("dMultiply"),
        &DerivedTypeInitialiser::ComputeDerivedMultiplyDivideFnType,
        &DerivedTypeInitialiser::BuildDerivedMultiplyDivideFnParams,
        buildDiffBody);
    // llvm::errs() << "Dumping derived multiply fn\n";
    // LangOptions langOpts;
    // langOpts.CPlusPlus = true;
    // clang::PrintingPolicy policy(langOpts);
    // policy.Bool = true;
    // FD->print(llvm::errs(), policy);
    return FD;
  }

  clang::FunctionDecl* DerivedTypeInitialiser::BuildDerivedDivideFn() {
    auto buildDiffBody = [this]() {
      auto a = m_Variables["a"];
      auto b = m_Variables["b"];
      for (auto field : m_DerivedRecord->fields()) {
        if (!(field->getType()->isRealType()))
          continue;
        auto dResMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_res"], field);
        auto dAMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_a"], field);
        auto dBMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_b"], field);

        auto diff1 = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Mul, dAMem, b);

        auto diff2 = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Mul, a, dBMem);

        auto subDiffs = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Sub, diff1,
                                            diff2);
        auto bSquare = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Mul, b, b);
        auto divideExpr = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Div,
                                              subDiffs, bSquare);
        auto assignExpr = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Assign,
                                              dResMem, divideExpr);
        AddToCurrentBlock(assignExpr);
      }
      auto returnExpr = m_Sema
                            .ActOnReturnStmt(noLoc, m_Variables["d_res"],
                                             m_CurScope)
                            .get();
      AddToCurrentBlock(returnExpr);
    };
    auto FD = GenerateDerivedArithmeticFn(
        m_ASTHelper.CreateDeclName("dDivide"),
        &DerivedTypeInitialiser::ComputeDerivedMultiplyDivideFnType,
        &DerivedTypeInitialiser::BuildDerivedMultiplyDivideFnParams,
        buildDiffBody);
  }

  llvm::SmallVector<clang::ParmVarDecl*, 2>
  DerivedTypeInitialiser::BuildDerivedAddSubFnParams() {
    auto d_a = m_ASTHelper.BuildParmVarDecl(m_Sema.CurContext,
                                            &m_Context.Idents.get("d_a"),
                                            GetDerivedParamType());
    auto d_b = m_ASTHelper.BuildParmVarDecl(m_Sema.CurContext,
                                            &m_Context.Idents.get("d_b"),
                                            GetDerivedParamType());
    llvm::SmallVector<clang::ParmVarDecl*, 2> params{d_a, d_b};
    return params;
  }

  llvm::SmallVector<clang::ParmVarDecl*, 4>
  DerivedTypeInitialiser::BuildDerivedMultiplyDivideFnParams() {
    auto a = m_ASTHelper.BuildParmVarDecl(m_Sema.CurContext,
                                          &m_Context.Idents.get("a"),
                                          GetNonDerivedParamType());
    auto d_a = m_ASTHelper.BuildParmVarDecl(m_Sema.CurContext,
                                            &m_Context.Idents.get("d_a"),
                                            GetDerivedParamType());
    auto b = m_ASTHelper.BuildParmVarDecl(m_Sema.CurContext,
                                          &m_Context.Idents.get("b"),
                                          GetNonDerivedParamType());
    auto d_b = m_ASTHelper.BuildParmVarDecl(m_Sema.CurContext,
                                            &m_Context.Idents.get("d_b"),
                                            GetDerivedParamType());
    return {a, d_a, b, d_b};
  }

  void
  DerivedTypeInitialiser::ComputeAndStoreDRE(llvm::ArrayRef<ValueDecl*> decls) {
    for (auto decl : decls) {
      m_Variables[decl->getNameAsString()] = m_ASTHelper.BuildDeclRefExpr(decl);
    }
  }

  DerivedTypeEssentials DerivedTypeInitialiser::CreateDerivedTypeEssentials() {
    return DerivedTypeEssentials(m_DerivedRecord, m_DerivedAddFn,
                                 m_DerivedSubFn, m_DerivedMultiplyFn,
                                 m_DerivedDivideFn, m_InitialiseSeedsFn);
  }

  CXXMethodDecl* DerivedTypeInitialiser::BuildInitialiseSeedsFn() {
    m_Variables.clear();
    auto memFnType = ComputeInitialiseSeedsFnType();
    llvm::SaveAndRestore<DeclContext*> saveContext(m_Sema.CurContext);
    auto memFn = m_ASTHelper.BuildMemFnDecl(m_DerivedRecord,
                                            m_ASTHelper.CreateDeclNameInfo(
                                                "InitialiseSeeds"),
                                            memFnType);
    m_Sema.CurContext = memFn->getDeclContext();

    beginScope(ASTHelper::CustomScope::FunctionBeginScope);
    m_Sema.PushFunctionScope();
    m_Sema.PushDeclContext(m_CurScope, memFn);

    llvm::SmallVector<Stmt*, 16> block;

    memFn->setParams({});

    // Function body scope
    beginScope(ASTHelper::CustomScope::FunctionBodyScope);
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

  bool DerivedTypeInitialiser::AddToCurrentBlock(clang::Stmt* S) {
    assert(!m_Blocks.empty() && "No block available to add statement in");
    m_Blocks.back().push_back(S);
  }

  typename DerivedTypeInitialiser::Stmts& DerivedTypeInitialiser::BeginBlock() {
    m_Blocks.push_back({});
    return m_Blocks.back();
  }

  clang::CompoundStmt* DerivedTypeInitialiser::EndBlock() {
    auto CS = m_ASTHelper.BuildCompoundStmt(m_Blocks.back());
    m_Blocks.pop_back();
    return CS;
  }

  void DerivedTypeInitialiser::ProcessTopLevelDeclarations(
      clang::ASTConsumer& consumer) {
    auto processTopLevelDecl = [&consumer](Decl* D) {
      if (!D)
        return;
      bool isTU = D->getDeclContext()->isTranslationUnit();
      if (isTU)
        consumer.HandleTopLevelDecl(DeclGroupRef(D));
    };
    processTopLevelDecl(m_DerivedAddFn);
    processTopLevelDecl(m_DerivedSubFn);
    processTopLevelDecl(m_DerivedMultiplyFn);
    processTopLevelDecl(m_DerivedDivideFn);
  }
} // namespace clad
