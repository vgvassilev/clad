#include "clad/Differentiator/ASTHelper.h"

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "clad/Differentiator/Compatibility.h"
using namespace clang;

namespace clad {
  static SourceLocation noLoc;

  ASTHelper::ASTHelper(Sema& sema)
      : m_Sema(sema), m_Context(sema.getASTContext()) {}

  clang::CXXRecordDecl*
  ASTHelper::FindCXXRecordDecl(clang::DeclarationName recordName) {
    return ASTHelper::FindCXXRecordDecl(m_Sema, recordName);
  }

  // TODO: Enable finding with nested name specifiers
  CXXRecordDecl* ASTHelper::FindCXXRecordDecl(clang::Sema& semaRef,
                                              DeclarationName recordName) {
    auto& C = semaRef.getASTContext();
    LookupResult R(semaRef, recordName, noLoc,
                   Sema::LookupNameKind::LookupTagName);
    CXXScopeSpec CSS;
    semaRef.LookupQualifiedName(R, C.getTranslationUnitDecl(), CSS);
    if (!R.isSingleResult())
      return nullptr;
    auto foundDecl = R.getFoundDecl();
    // TODO: Maybe make a more general function support other type of tag names
    // as well.
    if (auto RD = dyn_cast<CXXRecordDecl>(foundDecl)) {
      return RD;
    }
    return nullptr;
  }
  QualType ASTHelper::GetFundamentalType(llvm::StringRef typeName) const {
    return ASTHelper::GetFundamentalType(m_Sema, typeName);
  }

  QualType ASTHelper::GetFundamentalType(Sema& semaRef,
                                         llvm::StringRef typeName) {
    auto& C = semaRef.getASTContext();
    if (typeName == "int")
      return C.IntTy;
    if (typeName == "double")
      return C.DoubleTy;
    if (typeName == "float")
      return C.FloatTy;
    if (typeName == "long_double")
      return C.LongDoubleTy;
    if (typeName == "long_long")
      return C.LongTy;
    if (typeName == "void")
      return C.VoidTy;
    return QualType();
  }

  bool ASTHelper::IsFundamentalType(llvm::StringRef name) const {
    return ASTHelper::IsFundamentalType(m_Sema, name);
  }

  bool ASTHelper::IsFundamentalType(Sema& semaRef, llvm::StringRef name) {
    auto qType = GetFundamentalType(semaRef, name);
    return !qType.isNull();
  }

  QualType ASTHelper::ComputeQTypeFromTypeName(llvm::StringRef typeName) const {
    return ASTHelper::ComputeQTypeFromTypeName(m_Sema, typeName);
  }

  // TODO: Add support for non-fundamental types
  QualType ASTHelper::ComputeQTypeFromTypeName(Sema& semaRef,
                                               llvm::StringRef typeName) {
    auto& C = semaRef.getASTContext();
    auto qType = GetFundamentalType(semaRef, typeName);
    if (!qType.isNull())
      return qType;

    auto& typeNameId = C.Idents.get(typeName);
    DeclarationName typeDeclName(&typeNameId);
    auto RD = FindCXXRecordDecl(semaRef, typeDeclName);
    if (RD) {
      qType = RD->getTypeForDecl()->getCanonicalTypeInternal();
      return qType;
    }
    return QualType();
  }

  clang::ValueDecl*
  ASTHelper::FindRecordDeclMember(CXXRecordDecl* RD,
                                  llvm::StringRef memberName) {
    return ASTHelper::FindRecordDeclMember(m_Sema, RD, memberName);
  }

  clang::ValueDecl*
  ASTHelper::FindRecordDeclMember(Sema& semaRef, CXXRecordDecl* RD,
                                  llvm::StringRef memberName) {
    for (auto field : RD->fields()) {
      llvm::errs() << "field name: " << field->getName() << "\n";
      if (field->getName() == memberName) {
        return field;
      }
    }
    return nullptr;
  }

  MemberExpr* ASTHelper::BuildMemberExpr(Expr* base, ValueDecl* member) {
    return ASTHelper::BuildMemberExpr(m_Sema, base, member);
  }

  MemberExpr* ASTHelper::BuildMemberExpr(Sema& semaRef, Expr* base,
                                         ValueDecl* member) {
    auto& C = semaRef.getASTContext();
    auto DAP = DeclAccessPair::make(member, member->getAccess());
    CXXScopeSpec CSS;
    DeclarationName DN(&C.Idents.get(member->getName()));
    DeclarationNameInfo DNI(DN, noLoc);
    return semaRef.BuildMemberExpr(base, false, noLoc, &CSS, noLoc, member, DAP,
                                   false, DNI, member->getType(),
                                   ExprValueKind::VK_LValue,
                                   ExprObjectKind::OK_Ordinary);
  }
  clang::CXXNewExpr* ASTHelper::CreateNewExprFor(clang::QualType qType,
                                                 clang::Expr* initializer,
                                                 SourceLocation B) {
    return ASTHelper::CreateNewExprFor(m_Sema, qType, initializer, B);
  }
  clang::CXXNewExpr* ASTHelper::CreateNewExprFor(clang::Sema& semaRef,
                                                 clang::QualType qType,
                                                 Expr* initializer,
                                                 SourceLocation B) {
    auto& C = semaRef.getASTContext();
    initializer->dumpColor();
    auto newExpr = semaRef
                       .BuildCXXNew(SourceRange(), false, noLoc, MultiExprArg(),
                                    noLoc, SourceRange(), qType,
                                    C.getTrivialTypeSourceInfo(qType),
                                    Optional<Expr*>(), SourceRange(B, B),
                                    initializer)
                       .getAs<CXXNewExpr>();
    return newExpr;
  }

  clang::CXXConstructorDecl*
  ASTHelper::FindCopyConstructor(clang::CXXRecordDecl* RD) {
    return ASTHelper::FindCopyConstructor(m_Sema, RD);
  }

  clang::CXXConstructorDecl*
  ASTHelper::FindCopyConstructor(clang::Sema& semaRef,
                                 clang::CXXRecordDecl* RD) {
    auto& C = semaRef.getASTContext();
    auto qType = RD->getTypeForDecl()->getCanonicalTypeInternal();
    auto constructorDeclName = C.DeclarationNames.getCXXConstructorName(
        C.getCanonicalType(qType));
    LookupResult R(semaRef, constructorDeclName, noLoc,
                   Sema::LookupNameKind::LookupOrdinaryName);
    semaRef.LookupQualifiedName(R, RD);

    for (auto foundDecl : R) {
      if (auto constructor = dyn_cast<CXXConstructorDecl>(foundDecl)) {
        if (constructor->isCopyConstructor())
          return constructor;
      }
    }
    return nullptr;
  }

  clang::FunctionDecl*
  ASTHelper::FindUniqueFnDecl(clang::DeclContext* DC,
                              clang::DeclarationName fnName) {
    return ASTHelper::FindUniqueFnDecl(m_Sema, DC, fnName);
  }

  clang::FunctionDecl*
  ASTHelper::FindUniqueFnDecl(clang::Sema& semaRef, clang::DeclContext* DC,
                              clang::DeclarationName fnName) {
    LookupResult R(semaRef, fnName, noLoc,
                   Sema::LookupNameKind::LookupOrdinaryName);
    semaRef.LookupQualifiedName(R, DC);
    if (R.isSingleResult()) {
      if (auto method = dyn_cast<FunctionDecl>(R.getFoundDecl()))
        return method;
    }

    return nullptr;
  }

  Expr* ASTHelper::IgnoreParenImpCastsUnaryOp(Expr* E) {
    return ASTHelper::IgnoreParenImpCastsUnaryOp(m_Sema, E);
  }

  clang::Expr* ASTHelper::IgnoreParenImpCastsUnaryOp(clang::Sema& semaRef,
                                                     clang::Expr* E) {
    E = E->IgnoreParenImpCasts();
    while (auto UE = dyn_cast<UnaryOperator>(E)) {
      E = UE->getSubExpr()->IgnoreParenImpCasts();
    }
    return E;
  }

  DeclarationName ASTHelper::CreateDeclName(llvm::StringRef name) {
    return ASTHelper::CreateDeclName(m_Sema, name);
  }
  DeclarationName ASTHelper::CreateDeclName(Sema& semaRef,
                                            llvm::StringRef name) {
    auto& C = semaRef.getASTContext();
    auto& nameId = C.Idents.get(name);
    return DeclarationName(&nameId);
  }

  DeclarationNameInfo ASTHelper::CreateDeclNameInfo(llvm::StringRef name) {
    return ASTHelper::CreateDeclNameInfo(m_Sema, name);
  }

  DeclarationNameInfo ASTHelper::CreateDeclNameInfo(Sema& semaRef,
                                                    llvm::StringRef name) {
    return DeclarationNameInfo(CreateDeclName(semaRef, name), noLoc);
  }

  Expr* ASTHelper::BuildCXXCopyConstructExpr(QualType qType, Expr* E) {
    return ASTHelper::BuildCXXCopyConstructExpr(m_Sema, qType, E);
  }

  Expr* ASTHelper::BuildCXXCopyConstructExpr(Sema& semaRef, QualType qType,
                                             Expr* E) {
    llvm::errs() << "In BuildCXXCopyConstructorExpr\n";
    assert(qType->getAsCXXRecordDecl() &&
           "Currently default constructor expression is only supported for "
           "class types");
    auto copyConstructor = FindCopyConstructor(semaRef,
                                               qType->getAsCXXRecordDecl());
    copyConstructor->dumpColor();
    E = semaRef.ImpCastExprToType(E, qType.withConst(), CastKind::CK_NoOp)
            .get();
    E->dumpColor();
    auto initialize = semaRef
                          .BuildCXXConstructExpr(
                              noLoc, qType, copyConstructor, false,
                              MultiExprArg(E), true, false, false, false,
                              CXXConstructExpr::ConstructionKind::CK_Complete,
                              SourceRange())
                          .get();
    initialize->dumpColor();
    return initialize;
  }

  QualType ASTHelper::FindCorrespondingType(llvm::StringRef name) {
    return ASTHelper::FindCorrespondingType(m_Sema, name);
  }

  clang::QualType ASTHelper::FindCorrespondingType(Sema& semaRef,
                                                   llvm::StringRef name) {
    if (IsFundamentalType(semaRef, name)) {
      return GetFundamentalType(semaRef, name);
    }
    auto RD = FindCXXRecordDecl(semaRef, CreateDeclName(semaRef, name));
    if (RD) {
      // TODO: Do canonical type here will handle CVR qualifications?
      return RD->getTypeForDecl()->getCanonicalTypeInternal();
    }
    return QualType();
  }

  NamespaceDecl* ASTHelper::FindCladNamespace() {
    return ASTHelper::FindCladNamespace(m_Sema);
  }

  NamespaceDecl* ASTHelper::FindCladNamespace(Sema& semaRef) {
    static NamespaceDecl* ND = nullptr;
    if (ND)
      return ND;
    auto& C = semaRef.getASTContext();
    DeclarationName cladDName = CreateDeclName(semaRef, "clad");
    LookupResult R(semaRef, cladDName, noLoc, Sema::LookupNamespaceName,
                   clad_compat::Sema_ForVisibleRedeclaration);
    semaRef.LookupQualifiedName(R, C.getTranslationUnitDecl());
    assert(!R.empty() && "cannot find clad namespace");
    ND = cast<NamespaceDecl>(R.getFoundDecl());
    return ND;
  }

  clang::SourceLocation ASTHelper::GetValidSL() {
    return ASTHelper::GetValidSL(m_Sema);
  }
  clang::SourceLocation ASTHelper::GetValidSL(Sema& semaRef) {
    auto& SM = semaRef.getSourceManager();
    return SM.getLocForStartOfFile(SM.getMainFileID());
  }

  clang::SourceRange ASTHelper::GetValidSR() {
    return ASTHelper::GetValidSR(m_Sema);
  }
  clang::SourceRange ASTHelper::GetValidSR(Sema& semaRef) {
    auto SL = GetValidSL(semaRef);
    return SourceRange(SL, SL);
  }

  clang::VarDecl* ASTHelper::BuildVarDecl(clang::DeclContext* DC,
                                          clang::IdentifierInfo* II,
                                          clang::QualType qType) {
    return ASTHelper::BuildVarDecl(m_Sema, DC, II, qType);
  }

  clang::VarDecl* ASTHelper::BuildVarDecl(Sema& semaRef, clang::DeclContext* DC,
                                          clang::IdentifierInfo* II,
                                          clang::QualType qType) {
    auto& C = semaRef.getASTContext();
    auto VD = VarDecl::Create(C, DC, GetValidSL(semaRef), GetValidSL(semaRef),
                              II, qType, C.getTrivialTypeSourceInfo(qType),
                              SC_None);
    return VD;
  }

  clang::ParmVarDecl* ASTHelper::BuildParmVarDecl(clang::DeclContext* DC,
                                                  clang::IdentifierInfo* II,
                                                  clang::QualType qType) {
    return ASTHelper::BuildParmVarDecl(m_Sema, DC, II, qType);
  }
  clang::ParmVarDecl* ASTHelper::BuildParmVarDecl(clang::Sema& semaRef,
                                                  clang::DeclContext* DC,
                                                  clang::IdentifierInfo* II,
                                                  clang::QualType qType) {
    auto& C = semaRef.getASTContext();
    auto PVD = ParmVarDecl::Create(C, DC, noLoc, noLoc, II, qType,
                                   C.getTrivialTypeSourceInfo(qType),
                                   StorageClass::SC_None, nullptr);
    return PVD;
  }

  clang::DeclRefExpr* ASTHelper::BuildDeclRefExpr(clang::ValueDecl* VD,
                                                  clang::QualType qType) {
    return ASTHelper::BuildDeclRefExpr(m_Sema, VD, qType);
  }

  clang::DeclRefExpr* ASTHelper::BuildDeclRefExpr(clang::Sema& semaRef,
                                                  clang::ValueDecl* VD,
                                                  clang::QualType qType) {
    if (qType.isNull()) {
      qType = VD->getType();
    }                                                    
    // TODO: Confirm that we should always take non-reference type here.
    qType = qType.getNonReferenceType();
    return semaRef.BuildDeclRefExpr(VD, qType, ExprValueKind::VK_LValue, noLoc);
  }

  clang::DeclStmt* ASTHelper::BuildDeclStmt(clang::Decl* D) {
    return ASTHelper::BuildDeclStmt(m_Sema, D);
  }

  clang::DeclStmt* ASTHelper::BuildDeclStmt(clang::Sema& semaRef, clang::Decl* D) {
    auto DS = semaRef
                  .ActOnDeclStmt(semaRef.ConvertDeclToDeclGroup(D), noLoc,
                                 noLoc)
                  .getAs<DeclStmt>();
    return DS;                
  }
} // namespace clad
