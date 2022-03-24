#include "clad/Differentiator/CladUtils.h"

#include "ConstantFolder.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Lookup.h"
#include "clad/Differentiator/Compatibility.h"
#include "llvm/ADT/SmallVector.h"

using namespace clang;
namespace clad {
  namespace utils {
    static SourceLocation noLoc{};
    
    std::string ComputeEffectiveFnName(const FunctionDecl* FD) {
      // TODO: Add cases for more operators
      switch (FD->getOverloadedOperator()) {
        case OverloadedOperatorKind::OO_Call: return "operator_call";
        default: return FD->getNameAsString();
      }
    }

    CompoundStmt* PrependAndCreateCompoundStmt(ASTContext& C, Stmt* initial,
                                               Stmt* S) {
      llvm::SmallVector<Stmt*, 16> block;
      block.push_back(S);
      if (CompoundStmt* CS = dyn_cast<CompoundStmt>(initial))
        block.append(CS->body_begin(), CS->body_end());
      else 
        block.push_back(initial);
      auto stmtsRef = llvm::makeArrayRef(block.begin(), block.end());
      return clad_compat::CompoundStmt_Create(C, stmtsRef, noLoc, noLoc);
    }

    CompoundStmt* AppendAndCreateCompoundStmt(ASTContext& C, Stmt* initial,
                                              Stmt* S) {
      llvm::SmallVector<Stmt*, 16> block;
      assert(isa<CompoundStmt>(initial) &&
             "initial should be of type `clang::CompoundStmt`");
      if (CompoundStmt* CS = dyn_cast<CompoundStmt>(initial))
        block.append(CS->body_begin(), CS->body_end());
      block.push_back(S);
      auto stmtsRef = llvm::makeArrayRef(block.begin(), block.end());
      return clad_compat::CompoundStmt_Create(C, stmtsRef, noLoc, noLoc);
    }

    void BuildNNS(clang::Sema& semaRef, DeclContext* DC, CXXScopeSpec& CSS,
                  bool addGlobalNS) {
      assert(DC && "Must provide a non null DeclContext");

      // parent name specifier should be added first
      if (DC->getParent())
        BuildNNS(semaRef, DC->getParent(), CSS);

      ASTContext& C = semaRef.getASTContext();

      if (auto ND = dyn_cast<NamespaceDecl>(DC)) {
        CSS.Extend(C, ND,
                   /*NamespaceLoc=*/noLoc,
                   /*ColonColonLoc=*/noLoc);
      } else if (auto RD = dyn_cast<CXXRecordDecl>(DC)) {
        auto RDQType = RD->getTypeForDecl()->getCanonicalTypeInternal();
        auto RDTypeSourceInfo = C.getTrivialTypeSourceInfo(RDQType);
        CSS.Extend(C,
                   /*TemplateKWLoc=*/noLoc, RDTypeSourceInfo->getTypeLoc(),
                   /*ColonColonLoc=*/noLoc);
      } else if (addGlobalNS && isa<TranslationUnitDecl>(DC)) {
        CSS.MakeGlobal(C, /*ColonColonLoc=*/noLoc);
      }
    }

    DeclContext* FindDeclContext(clang::Sema& semaRef, clang::DeclContext* DC1,
                                 clang::DeclContext* DC2) {
      // llvm::errs()<<"DC1 name: "<<DC1->getDeclKindName()<<"\n";
      // llvm::errs()<<"DC2 name: "<<DC2->getDeclKindName()<<"\n";
      // cast<Decl>(DC1)->dumpColor();
      llvm::SmallVector<clang::DeclContext*, 4> contexts;
      assert((isa<NamespaceDecl>(DC1) || isa<TranslationUnitDecl>(DC1)) &&
             "DC1 can only be extended if it is a "
             "namespace or translation unit decl.");
      while (DC2) {
        // llvm::errs()<<"DC2 name: "<<DC2->getDeclKindName()<<"\n";
        if (isa<TranslationUnitDecl>(DC2))
          break;
        if (isa<LinkageSpecDecl>(DC2)) {
          DC2 = DC2->getParent();  
          continue;
        }
        // We don't want to 'extend' the DC1 context with class declarations.
        // There are 2 main reasons for this:
        // - Class declaration context cannot be extended the way namespace
        // declaration contexts can.
        //
        // - Primary usage of `FindDeclarationContext` is to create the correct
        // declaration context for searching some particular custom derivative.
        // But for class functions, we would 'need' to create custom derivative
        // in the original declaration context only (We don't support custom
        // derivatives for class functions yet). We cannot use the default
        // context that starts from `clad::custom_derivatives::`. This is
        // because custom derivatives of class functions need to have same
        // access permissions as the original member function.
        //
        // We may need to change this behaviour if in the future
        // `FindDeclarationContext` function is being used for some place other
        // than finding custom derivative declaration context as well.
        //
        // Silently return nullptr if DC2 contains any CXXRecord declaration
        // context.
        if (isa<CXXRecordDecl>(DC2))
          return nullptr;
        assert(isa<NamespaceDecl>(DC2) &&
               "DC2 should only consists of namespace, CXXRecord and "
               "translation unit declaration context.");
        contexts.push_back(DC2);
        DC2 = DC2->getParent();
      }
      DeclContext* DC = DC1;
      for (int i = contexts.size() - 1; i >= 0; --i) {
        NamespaceDecl* ND = cast<NamespaceDecl>(contexts[i]);
        DC = LookupNSD(semaRef, ND->getIdentifier()->getName(),
                       /*shouldExist=*/false, DC1);
        if (!DC)
          return nullptr;
        DC1 = DC;
      }
      return DC->getPrimaryContext();
    }

    NamespaceDecl* LookupNSD(Sema& S, llvm::StringRef namespc, bool shouldExist,
                             DeclContext* DC) {
      ASTContext& C = S.getASTContext();
      if (!DC)
        DC = C.getTranslationUnitDecl();
      // Find the builtin derivatives/numerical diff namespace
      DeclarationName Name = &C.Idents.get(namespc);
      LookupResult R(S, Name, SourceLocation(), Sema::LookupNamespaceName,
                     clad_compat::Sema_ForVisibleRedeclaration);
      S.LookupQualifiedName(R, DC,
                            /*allowBuiltinCreation*/ false);
      if (!shouldExist && R.empty())
        return nullptr;
      assert(!R.empty() && "Cannot find the specified namespace!");
      NamespaceDecl* ND = cast<NamespaceDecl>(R.getFoundDecl());
      return cast<NamespaceDecl>(ND->getPrimaryContext());
    }

    clang::DeclContext* GetOutermostDC(Sema& semaRef, clang::DeclContext* DC) {
      ASTContext& C = semaRef.getASTContext();
      assert(DC && "Invalid DC");
      while (DC) {
        if (DC->getParent() == C.getTranslationUnitDecl())
          break;
        DC = DC->getParent();
      }
      return DC;
    }
    
    StringLiteral* CreateStringLiteral(ASTContext& C, llvm::StringRef str) {
      // Copied and adapted from clang::Sema::ActOnStringLiteral.
      QualType CharTyConst = C.CharTy.withConst();
      QualType
          StrTy = clad_compat::getConstantArrayType(C, CharTyConst,
                                                    llvm::APInt(/*numBits=*/32,
                                                                str.size() + 1),
                                                    /*SizeExpr=*/nullptr,
                                                    /*ASM=*/ArrayType::Normal,
                                                    /*IndexTypeQuals*/ 0);
      StringLiteral* SL = StringLiteral::Create(C, str,
                                                /*Kind=*/StringLiteral::Ascii,
                                                /*Pascal=*/false, StrTy, noLoc);
      return SL;
    }

    bool isArrayOrPointerType(const clang::QualType QT) {
      return QT->isArrayType() || QT->isPointerType();
    }

    clang::DeclarationNameInfo BuildDeclarationNameInfo(clang::Sema& S,
                                                        llvm::StringRef name) {
      ASTContext& C = S.getASTContext();
      IdentifierInfo* II = &C.Idents.get(name);
      return DeclarationNameInfo(II, noLoc);
    }

    bool HasAnyReferenceOrPointerArgument(const clang::FunctionDecl* FD) {
      for (auto PVD : FD->parameters()) {
        if (PVD->getType()->isReferenceType() ||
            isArrayOrPointerType(PVD->getType()))
          return true;
      }
      return false;
    }

    bool IsReferenceOrPointerType(QualType T) {
      return T->isReferenceType() || isArrayOrPointerType(T);
    }

    bool SameCanonicalType(clang::QualType T1, clang::QualType T2) {
      return T1.getCanonicalType() == T2.getCanonicalType();
    }

    MemberExpr* BuildMemberExpr(Sema& semaRef, Scope* S, Expr* base,
                                llvm::StringRef memberName) {
      UnqualifiedId id;
      id.setIdentifier(GetIdentifierInfo(semaRef, memberName), noLoc);
      CXXScopeSpec SS;
      bool isArrow = base->getType()->isPointerType();
      auto ME =
          semaRef
              .ActOnMemberAccessExpr(S, base, noLoc,
                                     isArrow ? tok::TokenKind::arrow
                                             : tok::TokenKind::period,
                                     SS, noLoc, id, /*ObjCImpDecl=*/nullptr)
              .getAs<MemberExpr>();
      return ME;
    }

    bool isDifferentiableType(QualType T) {
      // FIXME: Handle arbitrary depth pointer types and arbitrary dimension
      // array type as well.
      if (isArrayOrPointerType(T))
        T = T->getPointeeOrArrayElementType()->getCanonicalTypeInternal();
      T = T.getNonReferenceType();
      if (T->isRealType() || T->isStructureOrClassType())
        return true;
      return false;
    }

    SourceLocation GetValidSLoc(Sema& semaRef) {
      auto& SM = semaRef.getSourceManager();
      return SM.getLocForStartOfFile(SM.getMainFileID());
    }

    clang::ParenExpr* BuildParenExpr(clang::Sema& semaRef, clang::Expr* E) {
      return semaRef.ActOnParenExpr(noLoc, noLoc, E).getAs<ParenExpr>();
    }

    clang::IdentifierInfo* GetIdentifierInfo(Sema& semaRef,
                                             llvm::StringRef identifier) {
      ASTContext& C = semaRef.getASTContext();
      return &C.Idents.get(identifier);
    }

    clang::ParmVarDecl*
    BuildParmVarDecl(clang::Sema& semaRef, clang::DeclContext* DC,
                     clang::IdentifierInfo* II, clang::QualType T,
                     clang::StorageClass SC, clang::Expr* defArg,
                     clang::TypeSourceInfo* TSI) {
      ASTContext& C = semaRef.getASTContext();
      if (!TSI)
        TSI = C.getTrivialTypeSourceInfo(T, noLoc);
      ParmVarDecl* PVD =
          ParmVarDecl::Create(C, DC, noLoc, noLoc, II, T, TSI, SC, defArg);
      return PVD;
    }

    clang::QualType GetValueType(clang::QualType T) {
      QualType valueType = T;
      if (T->isPointerType())
        valueType = T->getPointeeType();
      else if (T->isReferenceType()) 
        valueType = T.getNonReferenceType();
      // FIXME: `QualType::getPointeeOrArrayElementType` loses type qualifiers.
      else if (T->isArrayType()) 
        valueType =
            T->getPointeeOrArrayElementType()->getCanonicalTypeInternal();
      return valueType;
    }

    clang::Expr* BuildCladArrayInitByConstArray(clang::Sema& semaRef,
                                                clang::Expr* constArrE) {
      assert(isa<ConstantArrayType>(constArrE->getType()) &&
             "Expected a constant array expression!");
      ASTContext& C = semaRef.getASTContext();
      auto CAT = cast<ConstantArrayType>(constArrE->getType());
      Expr* sizeE = ConstantFolder::synthesizeLiteral(
          C.getSizeType(), C, CAT->getSize().getZExtValue());
      llvm::SmallVector<Expr*, 2> args = {constArrE, sizeE};
      return semaRef.ActOnInitList(noLoc, args, noLoc).get();
    }

    bool IsStaticMethod(const clang::FunctionDecl* FD) {
      if (auto MD = dyn_cast<CXXMethodDecl>(FD))
        return MD->isStatic();
      return false;
    }
  } // namespace utils
} // namespace clad