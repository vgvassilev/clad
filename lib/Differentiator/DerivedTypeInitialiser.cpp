#include "clad/Differentiator/DerivedTypeInitialiser.h"

#include "ConstantFolder.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/ParsedTemplate.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SaveAndRestore.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"
#include "clad/Differentiator/DerivedTypesHandler.h"

#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/LangOptions.h"

#include <functional>

using namespace clang;

namespace clad {
  static SourceLocation noLoc;

  DerivedTypeInitialiser::DerivedTypeInitialiser(ASTConsumer& consumer,
                                                 Sema& semaRef,
                                                 DerivedTypesHandler& DTH,
                                                 QualType yType, QualType xType)
      : m_Sema(semaRef), m_Context(semaRef.getASTContext()), m_DTH(DTH),
        m_CurScope(semaRef.TUScope), m_ASTHelper(semaRef), m_YQType(yType),
        m_XQType(xType) {
    // TODO: Remove any qualifiers from m_XQType and m_YQType.
    assert(
        (m_YQType->isRealType() || m_YQType == m_XQType) &&
        "We do not yet support differentiating aggregating types with respect "
        "to aggregate types");
    assert(m_XQType->isStructureOrClassType() &&
           "DerivedTypeInitialiser shoud not be called for scalar x type!!");
    PerformPendingInstantiations();

    BuildTangentDefinition();
    BuildTangentTypeInfoSpecialisation();
    m_ASTHelper.PrintDecl(m_Tangent, llvm::errs());
    if (m_YQType == m_XQType) {
      m_InitialiseSeedsFn = BuildInitialiseSeedsFn();
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
    auto baseTemplate = m_ASTHelper
                            .FindBaseTemplateClass(cladNS,
                                                   m_ASTHelper
                                                       .BuildDeclNameInfo(
                                                           "TangentTypeInfo"));
    return baseTemplate;
  }

  ClassTemplateDecl* DerivedTypeInitialiser::FindDerivativeOfBaseTemplate() {
    auto cladNS = m_ASTHelper.FindCladNamespace();
    auto baseTemplate = m_ASTHelper
                            .FindBaseTemplateClass(cladNS,
                                                   m_ASTHelper
                                                       .BuildDeclNameInfo(
                                                           "DerivativeOf"));
    return baseTemplate;
  }

  void DerivedTypeInitialiser::BuildTangentDefinition() {
    auto cladNS = m_ASTHelper.FindCladNamespace();
    auto baseDerivativeOf = FindDerivativeOfBaseTemplate();

    llvm::SaveAndRestore<DeclContext*> saveContext(m_Sema.CurContext);
    // TODO: Confirm that correct scopes are being created.
    beginScope(Scope::DeclScope | Scope::ClassScope);

    llvm::SmallVector<TemplateArgument, 2> templateArgs;
    templateArgs.push_back(m_YQType);
    templateArgs.push_back(m_XQType);

    CXXScopeSpec CSS;
    CSS.Extend(m_Context, cladNS, noLoc, noLoc);

    ParsedAttributesView PAV;

    // llvm::SmallVector<ParsedTemplateArgument, 2> parsedTemplateArgs;
    // parsedTemplateArgs.push_back(m_Sema.ActOnTemplateTypeArgument(
    //     TypeResult(ParsedType::make(m_XQType))));
    // parsedTemplateArgs.push_back(m_Sema.ActOnTemplateTypeArgument(
    //     TypeResult(ParsedType::make(m_YQType))));
    CXXScopeSpec emptySS;
    // parsedTemplateArgs.push_back(
    //     ParsedTemplateArgument(ParsedTemplateArgument::Type,
    //                            m_XQType.getAsOpaquePtr(), noLoc));
    // parsedTemplateArgs.push_back(
    //     ParsedTemplateArgument(ParsedTemplateArgument::Type,
    //                            m_YQType.getAsOpaquePtr(), noLoc));

    // llvm::SmallVector<TemplateIdAnnotation*, 2> cleanupList;
    // auto templateId = TemplateIdAnnotation::
    //     Create(noLoc, noLoc, baseDerivativeOf->getIdentifier(),
    //            OverloadedOperatorKind::OO_None,
    //            ParsedTemplateTy::make(TemplateName(baseDerivativeOf)),
    //            TemplateNameKind::TNK_Type_template,
    //            baseDerivativeOf->getBeginLoc(),
    //            baseDerivativeOf->getEndLoc(), parsedTemplateArgs,
    //            cleanupList);

    // auto TPL = TemplateParameterList::Create(m_Context, noLoc, noLoc, {},
    // noLoc,
    //                                          /*RequiresClause=*/nullptr);
    m_Sema.CurContext = cladNS;

    auto tangentRD = ClassTemplateSpecializationDecl::
        Create(m_Context, TagTypeKind::TTK_Class, m_Sema.CurContext,
               baseDerivativeOf->getBeginLoc(), baseDerivativeOf->getBeginLoc(),
               baseDerivativeOf, templateArgs, nullptr);
    // auto newCladNS = NamespaceDecl::Create(m_Context, m_Sema.CurContext,
    // false,
    //                                        noLoc, noLoc,
    //                                        cladNS->getIdentifier(), cladNS);
    // m_Sema.CurContext->addDecl(newCladNS);
    // m_Sema.CurContext = newCladNS;

    // auto tangentRD = m_Sema.ActOnClassTemplateSpecialization(
    //     m_CurScope, TypeSpecifierType::TST_class,
    //     Sema::TagUseKind::TUK_Definition, baseDerivativeOf->getBeginLoc(),
    //     noLoc, CSS, *templateId, PAV,
    //     {TPL}).getAs<ClassTemplateSpecializationDecl>();

    // tangentRD->setTemplateSpecializationKind(
    //     TemplateSpecializationKind::TSK_ExplicitSpecialization);
    m_Tangent = tangentRD;
    tangentRD->startDefinition();

    m_Sema.ActOnTagStartDefinition(GetCurrentScope(), tangentRD);
    // m_Sema.PushDeclContext(GetCurrentScope(), tangentRD);

    m_Sema.ActOnStartCXXMemberDeclarations(GetCurrentScope(), tangentRD, noLoc,
                                           false, noLoc);
    FillDerivedRecord();

    m_Sema.ActOnFinishCXXMemberSpecification(GetCurrentScope(), noLoc,
                                             tangentRD, noLoc, noLoc, PAV);

    m_Sema.ActOnFinishCXXMemberDecls();

    m_Sema.ActOnFinishCXXNonNestedClass();
    // tangentRD->completeDefinition();
    m_Sema.ActOnTagFinishDefinition(GetCurrentScope(), tangentRD,
                                    SourceRange());
    // m_Sema.PopDeclContext();
    endScope();

    cladNS->addDecl(tangentRD);

    m_TangentQType = m_Tangent->getTypeForDecl()->getCanonicalTypeInternal();
    m_TangentQType = m_Context
                         .getElaboratedType(ElaboratedTypeKeyword::ETK_None,
                                            CSS.getScopeRep(), m_TangentQType);

    // auto canonType = m_Context.getTypeDeclType(tangentRD);
    // TemplateArgumentListInfo TLI;
    // TLI.addArgument(
    //     TemplateArgumentLoc(templateArgs[0], m_Context.getTrivialTypeSourceInfo(
    //                                              templateArgs[0].getAsType())));
    // TLI.addArgument(
    //     TemplateArgumentLoc(templateArgs[1], m_Context.getTrivialTypeSourceInfo(
    //                                              templateArgs[1].getAsType())));
    // TypeSourceInfo* WrittenTy = m_Context.getTemplateSpecializationTypeInfo(
    //     TemplateName(baseDerivativeOf), noLoc, TLI, canonType);
    // llvm::errs() << "WrittenTy: " << WrittenTy << "\n";
    // // tangentRD->setTypeAsWritten(0);

    // llvm::errs() << "Dumping WrittenType: " << tangentRD->getTypeAsWritten()
    //              << "\n";

    m_ASTHelper.AddSpecialisation(baseDerivativeOf, m_Tangent);
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
    m_ASTHelper.PrintDecl(m_DerivedTypeInfoSpec, llvm::errs());
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

  static bool IsDifferentiableType(QualType ty) {
    if (ty->isRealType() || ty->isClassType())
      return true;
    if (auto AT = dyn_cast<ArrayType>(ty)) {
      QualType elemType = AT->getElementType();
      // TODO: Update this after adding support for multi-dimensional arrays.
      if (elemType->isRealType() || ty->isClassType())
        return true;
    }
    return false;
  }

  void DerivedTypeInitialiser::AddDerivedField(QualType Y, QualType X,
                                               IdentifierInfo* II) {
    if (!IsDifferentiableType(Y) || !IsDifferentiableType(X)) {
      return;
    }
    llvm::errs()<<"Adding derived field for the following X and Y types:\n";
    X.dump();
    llvm::errs()<<"\n";
    Y.dump();
    llvm::errs()<<"\n\n";
    QualType derivedType;
    if (!(Y->isArrayType()) && !(X->isArrayType())) {
      derivedType = m_DTH.GetDerivedType(Y, X);
    } else if (Y->isArrayType() && !(X->isArrayType())) {
      if (auto CA = dyn_cast<ConstantArrayType>(Y.getTypePtr())) {
        QualType elementType = CA->getElementType();
        // TODO: Handle multiple dimensions array.
        auto size = CA->getSize();
        auto sizeExpr = ConstantFolder::synthesizeLiteral(m_Context
                                                              .UnsignedIntTy,
                                                          m_Context,
                                                          size.getZExtValue());
        auto derivedElementType = m_DTH.GetDerivedType(elementType, X);
        derivedType = clad_compat::
            getConstantArrayType(m_Context, derivedElementType, size, sizeExpr,
                                 ArrayType::ArraySizeModifier::Normal,
                                 /*IndexTypeQuals=*/0);
      } else {
        // TODO: Remove assert and change it to a warning/notice.
        assert("Only constant array types are supported as of now!!");
      }
    } else if (!(Y->isArrayType()) && X->isArrayType()) {
      if (auto CA = dyn_cast<ConstantArrayType>(X.getTypePtr())) {
        QualType elementType = CA->getElementType();
        // TODO: Handle multiple dimensions array.
        auto size = CA->getSize();
        auto sizeExpr = ConstantFolder::synthesizeLiteral(m_Context
                                                              .UnsignedIntTy,
                                                          m_Context,
                                                          size.getZExtValue());
        auto derivedElementType = m_DTH.GetDerivedType(Y, elementType);
        derivedType = clad_compat::
            getConstantArrayType(m_Context, derivedElementType, size, sizeExpr,
                                 ArrayType::ArraySizeModifier::Normal,
                                 /*IndexTypeQuals=*/0);
      } else {
        // TODO: Remove assert and change it to a warning/notice.
        assert("Only constant array types are supported as of now!!");
      }
    } else {
      assert("Direct differentiation of array wrt array is not supported");
    }
    Expr* init = nullptr;
    if (derivedType->isRealType())
      init = ConstantFolder::synthesizeLiteral(derivedType, m_Context, 0);
    if (auto AT = dyn_cast<ArrayType>(derivedType.getTypePtr())) {
      InitListExpr* ILE = new (m_Context)
          InitListExpr(m_Context, noLoc, {}, noLoc);
      ImplicitValueInitExpr* valueInitExpr = new (m_Context)
          ImplicitValueInitExpr(AT->getElementType());
      ILE->setArrayFiller(valueInitExpr);
      ILE->setType(derivedType);
      init = ILE;
      llvm::errs() << "Dumping ILE:\n";
      ILE->dumpColor();
      llvm::errs() << "\n";
    }
    m_ASTHelper.BuildFieldDecl(m_Tangent, II, derivedType, init,
                               AccessSpecifier::AS_public, /*addToDecl=*/true);
  }

  void DerivedTypeInitialiser::FillDerivedRecord() {
    ParsedAttributesView PAV;
    m_Sema.ActOnAccessSpecifier(AccessSpecifier::AS_public, noLoc, noLoc, PAV);
    if (m_YQType->isRealType()) {
      auto xRD = m_XQType->getAsCXXRecordDecl();
      for (auto field : xRD->fields()) {
        AddDerivedField(m_YQType, field->getType(), field->getIdentifier());
      }
    } else if (m_YQType->isClassType()) {
      auto yRD = m_YQType->getAsCXXRecordDecl();
      for (auto field : yRD->fields()) {
        AddDerivedField(field->getType(), m_XQType, field->getIdentifier());
      }
    } else {
      assert("Unexpected case!! Y Type should either be a real type or a class "
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

  clang::FunctionDecl* DerivedTypeInitialiser::BuildDerivedArithmeticFn(
      clang::DeclarationNameInfo nameInfo, clang::QualType T,
      std::function<llvm::ArrayRef<ParmVarDecl*>(FunctionDecl*)> buildFnParams,
      std::function<void(void)> buildDiffBody) {
    m_Variables.clear();

    auto buildBody = [&, this](FunctionDecl* FD) {
      BeginBlock();
      auto params = FD->parameters();
      auto resVD = m_ASTHelper.BuildVarDecl(m_Sema.CurContext,
                                            &m_Context.Idents.get("d_res"),
                                            m_TangentQType);
      AddToCurrentBlock(m_ASTHelper.BuildDeclStmt(resVD));

      BuildAndStoreDRE(resVD);
      for (auto param : params)
        BuildAndStoreDRE(param);

      buildDiffBody();
      return EndBlock();
    };
    auto FD = m_ASTHelper.BuildFunction(m_Sema.CurContext, nameInfo, T,
                                       ast_helper::ScopeHandler(m_Sema,
                                                                m_CurScope),
                                       buildFnParams, buildBody);
    return FD;
  }

  clang::FunctionDecl* DerivedTypeInitialiser::BuildDerivedAddFn() {

    auto buildDiffBody = [this]() {
      for (auto field : m_Tangent->fields()) {
        auto dResMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_res"], field);
        auto dAMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_a"], field);
        auto dBMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_b"], field);
        if (field->getType()->isRealType()) {
          auto addDerivExpr = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Add,
                                                  dAMem, dBMem);
          auto assignExpr = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Assign,
                                                dResMem, addDerivExpr);
          AddToCurrentBlock(assignExpr);
        } else if (field->getType()->isArrayType()) {

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
                                                    GetCurrentScope());
      AddToCurrentBlock(returnExpr);
    };
    auto FD = BuildDerivedArithmeticFn(
        m_ASTHelper.BuildDeclNameInfo("dAdd"), ComputeDerivedAddSubFnType(),
        [this](FunctionDecl* FD) -> llvm::ArrayRef<ParmVarDecl*> {
          return BuildDerivedAddSubFnParams(FD);
        },
        buildDiffBody);
    return FD;
  }

  clang::FunctionDecl* DerivedTypeInitialiser::BuildDerivedSubFn() {
    auto buildDiffBody = [this]() {
      for (auto field : m_Tangent->fields()) {
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
          assert(DTE.isValid() &&
                 "Required derived type essentials not found!!");
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
    auto FD = BuildDerivedArithmeticFn(
        m_ASTHelper.BuildDeclNameInfo("dSub"), ComputeDerivedAddSubFnType(),
        [this](FunctionDecl* FD) { return BuildDerivedAddSubFnParams(FD); },
        buildDiffBody);
    return FD;
  }

  clang::FunctionDecl* DerivedTypeInitialiser::BuildDerivedMultiplyFn() {
    auto buildDiffBody = [this]() {
      auto a = m_Variables["a"];
      auto b = m_Variables["b"];
      for (auto field : m_Tangent->fields()) {
        auto dResMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_res"], field);
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
          assert(DTE.isValid() &&
                 "Required derived type essentials not found!!");
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
                                             GetCurrentScope())
                            .get();
      AddToCurrentBlock(returnExpr);
    };
    auto FD = BuildDerivedArithmeticFn(
        m_ASTHelper.BuildDeclNameInfo("dMultiply"),
        ComputeDerivedMultiplyDivideFnType(),
        [this](FunctionDecl* FD) {
          return BuildDerivedMultiplyDivideFnParams(FD);
        },
        buildDiffBody);
    return FD;
  }

  clang::FunctionDecl* DerivedTypeInitialiser::BuildDerivedDivideFn() {
    auto buildDiffBody = [this]() {
      auto a = m_Variables["a"];
      auto b = m_Variables["b"];
      for (auto field : m_Tangent->fields()) {
        auto dResMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_res"], field);
        auto dAMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_a"], field);
        auto dBMem = m_ASTHelper.BuildMemberExpr(m_Variables["d_b"], field);

        if (field->getType()->isRealType()) {
          auto diff1 = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Mul, dAMem,
                                           b);

          auto diff2 = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Mul, a,
                                           dBMem);

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
          assert(DTE.isValid() &&
                 "Required derived type essentials not found!!");
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
                                             GetCurrentScope())
                            .get();
      AddToCurrentBlock(returnExpr);
    };
    auto FD = BuildDerivedArithmeticFn(
        m_ASTHelper.BuildDeclNameInfo("dDivide"),
        ComputeDerivedMultiplyDivideFnType(),
        [this](FunctionDecl* FD) {
          return BuildDerivedMultiplyDivideFnParams(FD);
        },
        buildDiffBody);
    return FD;
  }

  llvm::ArrayRef<clang::ParmVarDecl*>
  DerivedTypeInitialiser::BuildDerivedAddSubFnParams(DeclContext* DC) {
    auto d_a = m_ASTHelper.BuildParmVarDecl(DC, &m_Context.Idents.get("d_a"),
                                            GetDerivedParamType());
    auto d_b = m_ASTHelper.BuildParmVarDecl(DC, &m_Context.Idents.get("d_b"),
                                            GetDerivedParamType());
    return {d_a, d_b};
  }

  llvm::ArrayRef<clang::ParmVarDecl*>
  DerivedTypeInitialiser::BuildDerivedMultiplyDivideFnParams(DeclContext* DC) {
    auto a = m_ASTHelper.BuildParmVarDecl(DC, &m_Context.Idents.get("a"),
                                          GetNonDerivedParamType());
    auto d_a = m_ASTHelper.BuildParmVarDecl(DC, &m_Context.Idents.get("d_a"),
                                            GetDerivedParamType());
    auto b = m_ASTHelper.BuildParmVarDecl(DC, &m_Context.Idents.get("b"),
                                          GetNonDerivedParamType());
    auto d_b = m_ASTHelper.BuildParmVarDecl(DC, &m_Context.Idents.get("d_b"),
                                            GetDerivedParamType());
    return {a, d_a, b, d_b};
  }

  void
  DerivedTypeInitialiser::BuildAndStoreDRE(llvm::ArrayRef<ValueDecl*> decls) {
    for (auto decl : decls) {
      m_Variables[decl->getNameAsString()] = m_ASTHelper.BuildDeclRefExpr(decl);
    }
  }

  DerivedTypeEssentials DerivedTypeInitialiser::CreateDerivedTypeEssentials() {
    return DerivedTypeEssentials(m_YQType, m_XQType, m_TangentQType, m_Tangent,
                                 m_DerivedAddFn, m_DerivedSubFn,
                                 m_DerivedMultiplyFn, m_DerivedDivideFn,
                                 m_InitialiseSeedsFn);
  }

  void DerivedTypeInitialiser::InitialiseIndependentFields(
      Expr* base, llvm::SmallVector<Expr*, 4> path) {
    // base->dumpColor();
    // llvm::errs()<<"\n\n";
    QualType baseQType = base->getType();
    if (base->getType()->isPointerType()) {
      baseQType = base->getType()->getPointeeType();
    }
    // llvm::errs()<<"Reaching here\n";
    if (!(baseQType->isArrayType()) && !m_DTH.GetYType(baseQType).isNull() &&
        m_DTH.GetYType(baseQType)->isRealType()) {
      Expr* E = base;
      for (auto elem : path) {
        if (auto idx = dyn_cast<IntegerLiteral>(elem)) {
          E = m_ASTHelper.BuildBuiltinArraySubscriptExpr(E, idx);
        } else if (auto SL = dyn_cast<StringLiteral>(elem)) {
          auto RD = E->getType()->getAsCXXRecordDecl();
          auto field = m_ASTHelper.FindRecordDeclMember(RD, SL->getString());
          // llvm::errs()<<"field: "<<field->getNameAsString()<<"\n";
          E = m_ASTHelper.BuildMemberExpr(E, field);
        }
      }
      auto assignExpr = m_ASTHelper.BuildOp(BinaryOperatorKind::BO_Assign, E,
                                            ConstantFolder::
                                                synthesizeLiteral(E->getType(),
                                                                  m_Context,
                                                                  1));
      AddToCurrentBlock(assignExpr);
      return;
    }

    if (auto CAT = dyn_cast<ConstantArrayType>(baseQType.getTypePtr())) {
      auto size = CAT->getSize().getZExtValue();
      for (unsigned i = 0; i < size; ++i) {
        auto idx = ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context,
                                                     i);
        auto AS = m_ASTHelper.BuildBuiltinArraySubscriptExpr(base, idx);
        auto updated_path = path;
        updated_path.push_back(idx);
        InitialiseIndependentFields(AS, updated_path);
      }
    } else if (baseQType->isStructureOrClassType()) {
      auto RD = baseQType->getAsCXXRecordDecl();
      // llvm::errs()<<"RD: "<<RD<<"\n";
      for (auto field : RD->fields()) {
        auto ME = m_ASTHelper.BuildMemberExpr(base, field);
        auto updated_path = path;
        updated_path.push_back(
            m_ASTHelper.BuildStringLiteral(field->getNameAsString()));
        InitialiseIndependentFields(ME, updated_path);
      }
    }
  }

  CXXMethodDecl* DerivedTypeInitialiser::BuildInitialiseSeedsFn() {
    m_Variables.clear();
    auto memFnType = ComputeInitialiseSeedsFnType();
    DeclarationNameInfo nameInfo = m_ASTHelper.BuildDeclNameInfo(
        "InitialiseSeeds");

    auto buildBody = [&, this](FunctionDecl* FD) {
      auto memFn = cast<CXXMethodDecl>(FD);
      BeginBlock();
      auto thisExpr = clad_compat::Sema_BuildCXXThisExpr(m_Sema, memFn);
      InitialiseIndependentFields(thisExpr, {});
      return EndBlock();
    };

    auto memFn = cast<CXXMethodDecl>(m_ASTHelper.BuildFunction(
        m_Sema, m_Tangent, nameInfo, memFnType,
        ast_helper::ScopeHandler(m_Sema, m_CurScope),
        [](FunctionDecl* FD) -> llvm::ArrayRef<ParmVarDecl*> { return {}; },
        buildBody));
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
  }

  Scope* DerivedTypeInitialiser::GetCurrentScope() { return m_CurScope; }

  void DerivedTypeInitialiser::PerformPendingInstantiations() {
    m_Sema.RequireCompleteType(noLoc, m_YQType, diag::err_typecheck_decl_incomplete_type);
    m_Sema.RequireCompleteType(noLoc, m_XQType, diag::err_typecheck_decl_incomplete_type);
  }
} // namespace clad
