#include "clad/Differentiator/DerivedTypeInitialiser.h"

#include "ConstantFolder.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
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

  static void PrintDecl(Decl* FD) {
    return;
    LangOptions langOpts;
    langOpts.CPlusPlus = true;
    clang::PrintingPolicy policy(langOpts);
    policy.Bool = true;
    FD->print(llvm::errs(), policy);
    llvm::errs() << "\n";
  }

  DerivedTypeInitialiser::DerivedTypeInitialiser(ASTConsumer& consumer,
                                                 Sema& semaRef,
                                                 DerivedTypesHandler& DTH,
                                                 QualType yType, QualType xType)
      : m_Sema(semaRef), m_Context(semaRef.getASTContext()), m_DTH(DTH),
        m_CurScope(semaRef.TUScope), m_ASTHelper(semaRef), m_YQType(yType),
        m_XQType(xType) {
    assert((m_XQType->isRealType() || m_XQType->isClassType()) &&
           "x should either be of a real type or a class type");
    assert((m_YQType->isRealType() || m_YQType->isClassType()) &&
           "y should either be of a real type or a class type");
    BuildTangentDefinition();
    BuildTangentTypeInfoSpecialisation();
    FillDerivedRecord();
    PrintDecl(m_TangentRD);
    if (m_YQType == m_XQType) {
      m_InitialiseSeedsFn = BuildInitialiseSeedsFn();
      m_DerivedAddFn = BuildDerivedAddFn();
    } else {
      m_DerivedAddFn = BuildDerivedAddFn();
      m_DerivedSubFn = BuildDerivedSubFn();
      m_DerivedMultiplyFn = BuildDerivedMultiplyFn();
      m_DerivedDivideFn = BuildDerivedDivideFn();
    }
    ProcessTopLevelDeclarations(consumer);
  }

  ClassTemplateDecl* DerivedTypeInitialiser::FindDerivedTypeInfoBaseTemplate() {
    auto cladNS = m_ASTHelper.FindCladNamespace();
    auto baseTemplate = m_ASTHelper.FindBaseTemplateClass(cladNS,
                                                          "TangentTypeInfo");
    return baseTemplate;
  }

  ClassTemplateDecl* DerivedTypeInitialiser::FindDerivativeOfBaseTemplate() {
    auto cladNS = m_ASTHelper.FindCladNamespace();
    auto baseTemplate = m_ASTHelper.FindBaseTemplateClass(cladNS,
                                                          "DerivativeOf");
    return baseTemplate;
  }

  void DerivedTypeInitialiser::BuildTangentDefinition() {
    auto cladNS = m_ASTHelper.FindCladNamespace();
    auto baseDerivativeOf = FindDerivativeOfBaseTemplate();

    beginScope(Scope::DeclScope | Scope::ClassScope);

    llvm::SmallVector<TemplateArgument, 2> templateArgs;
    templateArgs.push_back(m_YQType);
    templateArgs.push_back(m_XQType);

    auto tangentRecord = ClassTemplateSpecializationDecl::
        Create(m_Context, TagTypeKind::TTK_Class, m_Sema.CurContext, noLoc,
               noLoc, baseDerivativeOf, templateArgs, nullptr);
    tangentRecord->startDefinition();
    tangentRecord->completeDefinition();
    endScope();

    cladNS->addDecl(tangentRecord);

    m_TangentRD = tangentRecord;
    m_TangentQType = m_TangentRD->getTypeForDecl()->getCanonicalTypeInternal();

    m_ASTHelper.AddSpecialisation(baseDerivativeOf, m_TangentRD);
  }

  void DerivedTypeInitialiser::BuildTangentTypeInfoSpecialisation() {
    auto cladNS = m_ASTHelper.FindCladNamespace();
    auto derivedTypeInfoBase = FindDerivedTypeInfoBaseTemplate();
    beginScope(Scope::DeclScope | Scope::ClassScope);

    llvm::SmallVector<TemplateArgument, 2> templateArgs;
    templateArgs.push_back(m_YQType);
    templateArgs.push_back(m_XQType);

    auto derivedTypeInfoSpec = ClassTemplateSpecializationDecl::
        Create(m_Context, TagTypeKind::TTK_Struct, m_Sema.CurContext, noLoc,
               noLoc, derivedTypeInfoBase, templateArgs, nullptr);
    derivedTypeInfoSpec->startDefinition();
    derivedTypeInfoSpec->completeDefinition();
    endScope();

    cladNS->addDecl(derivedTypeInfoSpec);

    auto typeId = &m_Context.Idents.get("type");
    auto TSI = m_Context.getTrivialTypeSourceInfo(m_TangentQType);
    auto usingTypeDecl = TypeAliasDecl::Create(m_Context, derivedTypeInfoSpec,
                                               noLoc, noLoc, typeId, TSI);
    usingTypeDecl->setAccess(AccessSpecifier::AS_public);
    derivedTypeInfoSpec->addDecl(usingTypeDecl);

    m_DerivedTypeInfoSpec = derivedTypeInfoSpec;

    m_ASTHelper.AddSpecialisation(derivedTypeInfoBase, m_DerivedTypeInfoSpec);
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
    if (m_YQType->isRealType()) {
      auto xRD = m_XQType->getAsCXXRecordDecl();
      for (auto field : xRD->fields()) {
        QualType derivedFieldType;
        if (field->getType()->isRealType()) {
          derivedFieldType = m_Context.DoubleTy;
        } else if (field->getType()->isClassType()) {
          derivedFieldType = m_DTH.GetDerivedType(m_Context.DoubleTy,
                                                  field->getType());
          assert(!derivedFieldType.isNull() &&
                 "Required derived field type not found!!");
        } else if (field->getType()->isArrayType()) {
          derivedFieldType = field->getType();
        }
        else {
          continue;
        }
        Expr* init = nullptr;
        if (derivedFieldType == m_Context.DoubleTy)
          init = ConstantFolder::synthesizeLiteral(derivedFieldType, m_Context,
                                                   0);
        auto FD = m_ASTHelper.BuildFieldDecl(m_TangentRD,
                                             field->getIdentifier(),
                                             derivedFieldType, init,
                                             AccessSpecifier::AS_public, true);
      }
    } else if (m_YQType->isClassType()) {
      auto yRD = m_YQType->getAsCXXRecordDecl();
      for (auto field : yRD->fields()) {
        QualType derivedFieldType;
        if (field->getType()->isRealType()) {
          if (m_XQType->isRealType()) {
            derivedFieldType = m_Context.DoubleTy;
          } else if (m_XQType->isClassType()) {
            derivedFieldType = m_DTH.GetDerivedType(m_Context.DoubleTy,
                                                    m_XQType);
            assert(!derivedFieldType.isNull() &&
                   "Required derived field type not found!!");
          }
        } else if (field->getType()->isClassType()) {
          derivedFieldType = m_DTH.GetDerivedType(field->getType(), m_XQType);
          assert(!derivedFieldType.isNull() &&
                 "Required derived field type not found!!");
        } else {
          continue;
        }
        Expr* init = nullptr;
        if (derivedFieldType == m_Context.DoubleTy)
          init = ConstantFolder::synthesizeLiteral(derivedFieldType, m_Context,
                                                   0);
        auto FD = m_ASTHelper.BuildFieldDecl(m_TangentRD,
                                             field->getIdentifier(),
                                             derivedFieldType, init,
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
        m_TangentQType.withConst());
    return paramType;
  }

  clang::QualType DerivedTypeInitialiser::GetNonDerivedParamType() const {
    return m_YQType;
  }

  QualType DerivedTypeInitialiser::ComputeDerivedAddSubFnType() const {
    auto paramType = GetDerivedParamType();
    llvm::SmallVector<QualType, 2> paramTypes = {paramType, paramType};
    auto returnType = m_TangentQType;
    auto fnType = m_Context.getFunctionType(returnType, paramTypes,
                                            FunctionProtoType::ExtProtoInfo());
    return fnType;
  }

  clang::QualType DerivedTypeInitialiser::ComputeDerivedMultiplyDivideFnType() {
    auto paramTypes = {GetNonDerivedParamType(), GetDerivedParamType(),
                       GetNonDerivedParamType(), GetDerivedParamType()};
    auto returnType = m_TangentQType;
    auto fnType = m_Context.getFunctionType(returnType, paramTypes,
                                            FunctionProtoType::ExtProtoInfo());
    return fnType;                                            
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
                                          m_TangentQType);
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

    return FD;
  }

  clang::FunctionDecl* DerivedTypeInitialiser::BuildDerivedAddFn() {

    auto buildDiffBody = [this]() {
      for (auto field : m_TangentRD->fields()) {
        auto dResMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_res"], field);
        auto dAMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_a"], field);
        auto dBMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_b"], field);
        if (field->getType()->isRealType()) {
          auto addDerivExpr = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Add,
                                                  dAMem, dBMem);
          auto assignExpr = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Assign,
                                                dResMem, addDerivExpr);
          AddToCurrentBlock(assignExpr);
        } else if (field->getType()->isClassType()) {
          auto DTE = m_DTH.GetDTE(field->getType());
          assert(DTE.isValid() &&
                 "Required Derived Type Essesentials not found!!");
          auto memberDAddFn = DTE.GetDerivedAddFn();
          llvm::SmallVector<Expr*, 2> argsRef;
          argsRef.push_back(dAMem);
          argsRef.push_back(dBMem);
          auto dAddFnCall = m_ASTHelper.BuildCallToFn(GetCurrentScope(),
                                                      memberDAddFn, argsRef);
          auto assignExpr = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Assign,
                                                dResMem, dAddFnCall);
          AddToCurrentBlock(assignExpr);
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
      for (auto field : m_TangentRD->fields()) {
        auto dResMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_res"], field);
        auto dAMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_a"], field);
        auto dBMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_b"], field);
        
        if (field->getType()->isRealType()) {
          auto addDerivExpr = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Sub,
                                                  dAMem, dBMem);
          auto assignExpr = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Assign,
                                                dResMem, addDerivExpr);
          AddToCurrentBlock(assignExpr);
        } else if (field->getType()->isClassType()) {
          auto DTE = m_DTH.GetDTE(field->getType());
          assert(DTE.isValid() && "Required derived type essentials not found!!");
          auto memberDSubFn = DTE.GetDerivedAddFn();
          llvm::SmallVector<Expr*, 2> argsRef;
          argsRef.push_back(dAMem);
          argsRef.push_back(dBMem);
          auto dAddFnCall = m_ASTHelper.BuildCallToFn(GetCurrentScope(),
                                                      memberDSubFn, argsRef);
          auto assignExpr = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Assign,
                                                dResMem, dAddFnCall);
          AddToCurrentBlock(assignExpr);
        }
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
      for (auto field : m_TangentRD->fields()) {
        auto dResMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_res"],
                                                    field);
        auto dAMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_a"], field);
        auto dBMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_b"], field);
        if (field->getType()->isRealType()) {
          auto diff1 = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Mul, dAMem,
                                           b);

          auto diff2 = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Mul, a,
                                           dBMem);

          auto addDiffs = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Add, diff1,
                                              diff2);
          auto assignExpr = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Assign,
                                                dResMem, addDiffs);
          AddToCurrentBlock(assignExpr);
        } else if (field->getType()->isClassType()) {
          auto DTE = m_DTH.GetDTE(field->getType());
          assert(DTE.isValid() && "Required derived type essentials not found!!");
          auto dMultiply = DTE.GetDerivedMultiplyFn();
          llvm::SmallVector<Expr*, 4> argsRef = {a, dAMem, b, dBMem};
          auto dMultiplyFnCall = m_ASTHelper.BuildCallToFn(GetCurrentScope(),
                                                      dMultiply, argsRef);
          auto assignExpr = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Assign,
                                                dResMem, dMultiplyFnCall);
          AddToCurrentBlock(assignExpr);
        }
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
    return FD;
  }

  clang::FunctionDecl* DerivedTypeInitialiser::BuildDerivedDivideFn() {
    auto buildDiffBody = [this]() {
      auto a = m_Variables["a"];
      auto b = m_Variables["b"];
      for (auto field : m_TangentRD->fields()) {
        auto dResMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_res"], field);
        auto dAMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_a"], field);
        auto dBMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_b"], field);

        if (field->getType()->isRealType()) {
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
        } else if (field->getType()->isClassType()) {
          auto DTE = m_DTH.GetDTE(field->getType());
          assert(DTE.isValid() && "Required derived type essentials not found!!");
          auto dDivide = DTE.GetDerivedDivideFn();
          llvm::SmallVector<Expr*, 4> argsRef = {a, dAMem, b, dBMem};
          auto dMultiplyFnCall = m_ASTHelper.BuildCallToFn(GetCurrentScope(),
                                                      dDivide, argsRef);
          auto assignExpr = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Assign,
                                                dResMem, dMultiplyFnCall);
          AddToCurrentBlock(assignExpr);
        }
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
    return FD;
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
    return DerivedTypeEssentials(m_YQType, m_XQType, m_TangentQType,
                                 m_TangentRD, m_DerivedAddFn, m_DerivedSubFn,
                                 m_DerivedMultiplyFn, m_DerivedDivideFn,
                                 m_InitialiseSeedsFn);
  }

  void DerivedTypeInitialiser::InitialiseIndependentFields(
      Expr* base, llvm::SmallVector<std::string, 4> path) {
    QualType baseQType = base->getType();
    if (base->getType()->isPointerType()) {
      baseQType = base->getType()->getPointeeType();
    }

    if (m_DTH.GetYType(baseQType) == m_Context.DoubleTy) {
      Expr* E = base;
      for (auto fieldName : path) {
        auto RD = E->getType()->getAsCXXRecordDecl();
        auto field = m_ASTHelper.FindRecordDeclMember(RD, fieldName);
        E = m_ASTHelper.BuildMemberExpr(E, field);
      }
      auto assignExpr = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Assign, E,
                                            ConstantFolder::
                                                synthesizeLiteral(E->getType(),
                                                                  m_Context,
                                                                  1));
      AddToCurrentBlock(assignExpr);
      return;
    }

    auto RD = baseQType->getAsCXXRecordDecl();
    for (auto field : RD->fields()) {
      auto ME = m_ASTHelper.BuildMemberExpr(base, field);
      auto updated_path = path;
      updated_path.push_back(field->getNameAsString());
      InitialiseIndependentFields(ME, updated_path);
    }
  }

  CXXMethodDecl* DerivedTypeInitialiser::BuildInitialiseSeedsFn() {
    m_Variables.clear();
    auto memFnType = ComputeInitialiseSeedsFnType();
    llvm::SaveAndRestore<DeclContext*> saveContext(m_Sema.CurContext);
    auto memFn = m_ASTHelper.BuildMemFnDecl(m_TangentRD,
                                            m_ASTHelper.CreateDeclNameInfo(
                                                "InitialiseSeeds"),
                                            memFnType);
    m_Sema.CurContext = memFn->getDeclContext();

    beginScope(ASTHelper::CustomScope::FunctionBeginScope);
    m_Sema.PushFunctionScope();
    m_Sema.PushDeclContext(m_CurScope, memFn);

    BeginBlock();
    memFn->setParams({});

    // Function body scope
    beginScope(ASTHelper::CustomScope::FunctionBodyScope);
    // auto thisExpr = clad_compat::Sema_BuildCXXThisExpr(m_Sema, memFn);

    // for (auto field : m_TangentRD->fields()) {
    //   auto RD = field->getType()->getAsCXXRecordDecl();
    //   FieldDecl* independentField = nullptr;
    //   LookupResult R(m_Sema, field->getDeclName(), noLoc,
    //                  Sema::LookupNameKind::LookupMemberName);
    //   CXXScopeSpec CSS();
    //   m_Sema.LookupQualifiedName(R, RD, CSS);
    //   if (R.isSingleResult()) {
    //     if (auto decl = dyn_cast<FieldDecl>(R.getFoundDecl())) {
    //       independentField = decl;
    //     }
    //   }
    //   if (!independentField || !(independentField->getType()->isRealType()))
    //     continue;
    //   auto baseExpr = m_ASTHelper.BuildMemberExpr(thisExpr, field);
    //   auto independentFieldExpr = m_ASTHelper.BuildMemberExpr(baseExpr,
    //                                                           independentField);
    //   independentFieldExpr->dumpColor();
    //   auto assignExpr = m_Sema
    //                         .BuildBinOp(m_CurScope, noLoc,
    //                                     BinaryOperatorKind::BO_Assign,
    //                                     independentFieldExpr,
    //                                     ConstantFolder::
    //                                         synthesizeLiteral(independentField
    //                                                               ->getType(),
    //                                                           m_Context, 1))
    //                         .get();
    //   assignExpr->dumpColor();
    //   block.push_back(assignExpr);
    // }
    auto thisExpr = clad_compat::Sema_BuildCXXThisExpr(m_Sema, memFn);
    InitialiseIndependentFields(thisExpr, {});
    auto CS = m_ASTHelper.BuildCompoundStmt(EndBlock());
    memFn->setBody(CS);
    endScope();
    m_Sema.PopFunctionScopeInfo();
    m_Sema.PopDeclContext();
    endScope(); // Function decl scope

    m_ASTHelper.RegisterFn(memFn->getDeclContext(), memFn);
    PrintDecl(memFn);
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
    return true;
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
    // processTopLevelDecl(m_DerivedTypeInfoSpec);
  }

  Scope* DerivedTypeInitialiser::GetCurrentScope() { return m_CurScope; }
} // namespace clad
