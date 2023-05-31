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
      switch (FD->getOverloadedOperator()) {
      case OverloadedOperatorKind::OO_Plus:
        return "operator_plus";
      case OverloadedOperatorKind::OO_Minus:
        return "operator_minus";
      case OverloadedOperatorKind::OO_Star:
        return "operator_star";
      case OverloadedOperatorKind::OO_Slash:
        return "operator_slash";
      case OverloadedOperatorKind::OO_Percent:
        return "operator_percent";
      case OverloadedOperatorKind::OO_Caret:
        return "operator_caret";
      case OverloadedOperatorKind::OO_Amp:
        return "operator_amp";
      case OverloadedOperatorKind::OO_Pipe:
        return "operator_pipe";
      case OverloadedOperatorKind::OO_Tilde:
        return "operator_tilde";
      case OverloadedOperatorKind::OO_Exclaim:
        return "operator_exclaim";
      case OverloadedOperatorKind::OO_Equal:
        return "operator_equal";
      case OverloadedOperatorKind::OO_Less:
        return "operator_less";
      case OverloadedOperatorKind::OO_Greater:
        return "operator_greater";
      case OverloadedOperatorKind::OO_PlusEqual:
        return "operator_plus_equal";
      case OverloadedOperatorKind::OO_MinusEqual:
        return "operator_minus_equal";
      case OverloadedOperatorKind::OO_StarEqual:
        return "operator_star_equal";
      case OverloadedOperatorKind::OO_SlashEqual:
        return "operator_slash_equal";
      case OverloadedOperatorKind::OO_PercentEqual:
        return "operator_percent_equal";
      case OverloadedOperatorKind::OO_CaretEqual:
        return "operator_caret_equal";
      case OverloadedOperatorKind::OO_AmpEqual:
        return "operator_amp_equal";
      case OverloadedOperatorKind::OO_PipeEqual:
        return "operator_pipe_equal";
      case OverloadedOperatorKind::OO_LessLess:
        return "operator_less_less";
      case OverloadedOperatorKind::OO_GreaterGreater:
        return "operator_greater_greater";
      case OverloadedOperatorKind::OO_GreaterGreaterEqual:
        return "operator_greater_greater_equal";
      case OverloadedOperatorKind::OO_LessLessEqual:
        return "operator_less_less_equal";
      case OverloadedOperatorKind::OO_EqualEqual:
        return "operator_equal_equal";
      case OverloadedOperatorKind::OO_ExclaimEqual:
        return "operator_exclaim_equal";
      case OverloadedOperatorKind::OO_LessEqual:
        return "operator_less_equal";
      case OverloadedOperatorKind::OO_GreaterEqual:
        return "operator_greater_equal";
#if CLANG_VERSION_MAJOR > 5
      case OverloadedOperatorKind::OO_Spaceship:
        return "operator_spaceship";
#endif
      case OverloadedOperatorKind::OO_AmpAmp:
        return "operator_AmpAmp";
      case OverloadedOperatorKind::OO_PipePipe:
        return "operator_pipe_pipe";
      case OverloadedOperatorKind::OO_PlusPlus:
        return "operator_plus_plus";
      case OverloadedOperatorKind::OO_MinusMinus:
        return "operator_minus_minus";
      case OverloadedOperatorKind::OO_Comma:
        return "operator_comma";
      case OverloadedOperatorKind::OO_ArrowStar:
        return "operator_arrow_star";
      case OverloadedOperatorKind::OO_Arrow:
        return "operator_arrow";
      case OverloadedOperatorKind::OO_Call:
        return "operator_call";
      case OverloadedOperatorKind::OO_Subscript:
        return "operator_subscript";
      default:
        return FD->getNameAsString();
      }
    }

    CompoundStmt* PrependAndCreateCompoundStmt(ASTContext& C, Stmt* initial,
                                               Stmt* S) {
      llvm::SmallVector<Stmt*, 16> block;
      block.push_back(S);
      CompoundStmt* CS = dyn_cast<CompoundStmt>(initial);
      if (CS)
        block.append(CS->body_begin(), CS->body_end());
      else 
        block.push_back(initial);
      auto stmtsRef = clad_compat::makeArrayRef(block.begin(), block.end());
      return clad_compat::CompoundStmt_Create(C, stmtsRef /**/CLAD_COMPAT_CLANG15_CompoundStmt_Create_ExtraParam1(CS), noLoc, noLoc);
    }

    CompoundStmt* AppendAndCreateCompoundStmt(ASTContext& C, Stmt* initial,
                                              Stmt* S) {
      llvm::SmallVector<Stmt*, 16> block;
      assert(isa<CompoundStmt>(initial) &&
             "initial should be of type `clang::CompoundStmt`");
      CompoundStmt* CS = dyn_cast<CompoundStmt>(initial);
      if (CS)
        block.append(CS->body_begin(), CS->body_end());
      block.push_back(S);
      auto stmtsRef = clad_compat::makeArrayRef(block.begin(), block.end());
      return clad_compat::CompoundStmt_Create(C, stmtsRef /**/ CLAD_COMPAT_CLANG15_CompoundStmt_Create_ExtraParam1(CS), noLoc, noLoc);
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
                   /*NamespaceLoc=*/utils::GetValidSLoc(semaRef),
                   /*ColonColonLoc=*/utils::GetValidSLoc(semaRef));
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

    // Adds a namespace specifier to the given type if the type is not already an elaborated type.
    clang::QualType AddNamespaceSpecifier(clang::Sema& semaRef, clang::ASTContext &C, clang::QualType QT) {
      auto typePtr = QT.getTypePtr();
      if (typePtr->isRecordType() && !typePtr->getAs<clang::ElaboratedType>()) {
        CXXScopeSpec CSS;
        clang::CXXRecordDecl const* recordDecl = typePtr->getAsCXXRecordDecl();
        clang::DeclContext const* declContext = static_cast<clang::DeclContext const*>(recordDecl);
        utils::BuildNNS(semaRef, const_cast<clang::DeclContext*>(declContext), CSS);
        NestedNameSpecifier* NS = CSS.getScopeRep();
        if (auto* Prefix = NS->getPrefix())
          return C.getElaboratedType(ETK_None, Prefix, QT);
      }
      return QT;
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
                                                /*Kind=*/clad_compat::StringKind_Ordinary,
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

    bool IsCladValueAndPushforwardType(clang::QualType T) {
      return T.getAsString().find("ValueAndPushforward") !=
             std::string::npos;
    }

    clang::SourceRange GetValidSRange(clang::Sema& semaRef) {
      SourceLocation validSL = GetValidSLoc(semaRef);
      return SourceRange(validSL, validSL);
    }

    CXXNewExpr* BuildCXXNewExpr(Sema& semaRef, QualType qType,
                                clang::Expr* arraySize, Expr* initializer,
                                clang::TypeSourceInfo* TSI) {
      auto& C = semaRef.getASTContext();
      if (!TSI)
        TSI = C.getTrivialTypeSourceInfo(qType);
      auto newExpr =
          semaRef
              .BuildCXXNew(
                  SourceRange(), false, noLoc, MultiExprArg(), noLoc,
                  SourceRange(), qType, TSI,
                  (arraySize ? arraySize : clad_compat::ArraySize_None()),
                  initializer ? GetValidSRange(semaRef) : SourceRange(),
                  initializer)
              .getAs<CXXNewExpr>();
      return newExpr;
    }

    clang::Expr* BuildStaticCastToRValue(clang::Sema& semaRef, clang::Expr* E) {
      ASTContext& C = semaRef.getASTContext();
      QualType T = E->getType();
      T = T.getNonReferenceType();
      T = C.getRValueReferenceType(T);
      TypeSourceInfo* TSI = C.getTrivialTypeSourceInfo(T);
      Expr* rvalueCastE =
          semaRef
              .BuildCXXNamedCast(noLoc, tok::TokenKind::kw_static_cast, TSI, E,
                                 noLoc, noLoc)
              .get();
      return rvalueCastE;
    }

    bool IsRValue(const clang::Expr* E) {
      return clad_compat::IsPRValue(E) || E->isXValue();
    }

    void AppendIndividualStmts(llvm::SmallVectorImpl<clang::Stmt*>& block,
                               clang::Stmt* S) {
      if (auto CS = dyn_cast_or_null<CompoundStmt>(S))
        for (auto stmt : CS->body())
          block.push_back(stmt);
      else if (S)
        block.push_back(S);
    }
    
    MemberExpr*
    BuildMemberExpr(clang::Sema& semaRef, clang::Scope* S, clang::Expr* base,
                    llvm::ArrayRef<clang::StringRef> fields) {
      MemberExpr* ME = nullptr;
      for (auto field : fields) {
        ME = BuildMemberExpr(semaRef, S, base, field);
        base = ME;
      }
      return ME;
    }

    clang::FieldDecl* LookupDataMember(clang::Sema& semaRef, clang::RecordDecl* RD,
                          llvm::StringRef name) {
      LookupResult R(semaRef, BuildDeclarationNameInfo(semaRef, name),
                     Sema::LookupNameKind::LookupMemberName);
      CXXScopeSpec CSS;
      semaRef.LookupQualifiedName(R, RD, CSS);
      if (R.empty())
        return nullptr;
      assert(R.isSingleResult() && "Lookup in valid classes should always "
                                   "return a single data member result.");
      auto D = R.getFoundDecl();
      // We are looking data members only!
      if (auto FD = dyn_cast<FieldDecl>(D))
        return FD;
      return nullptr;
    }

    bool IsValidMemExprPath(clang::Sema& semaRef, clang::RecordDecl* RD,
                     llvm::ArrayRef<llvm::StringRef> fields) {
      for (std::size_t i = 0; i < fields.size(); ++i) {
        FieldDecl* FD = LookupDataMember(semaRef, RD, fields[i]);
        if (!FD)
          return false;
        if (FD->getType()->isRecordType())
          RD = FD->getType()->getAsCXXRecordDecl();
        // Current Field declaration is not of record type, therefore
        // it cannot have any field within it. And any member access
        // ('.') expression would be an invalid path.
        else if (i != fields.size() - 1)
          return false;
      }
      return true;
    }

    clang::QualType
    ComputeMemExprPathType(clang::Sema& semaRef, clang::RecordDecl* RD,
                           llvm::ArrayRef<llvm::StringRef> fields) {
      assert(IsValidMemExprPath(semaRef, RD, fields) &&
             "Invalid field path specified!");
      QualType T = RD->getTypeForDecl()->getCanonicalTypeInternal();
      for (auto field : fields) {
        auto FD = LookupDataMember(semaRef, RD, field);
        if (FD->getType()->isRecordType())
          RD = FD->getType()->getAsCXXRecordDecl();
        T = FD->getType();
      }
      return T;
    }

    bool hasNonDifferentiableAttribute(const clang::Decl* D) {
      for (auto* Attr : D->specific_attrs<clang::AnnotateAttr>())
        if (Attr->getAnnotation() == "non_differentiable")
          return true;
      return false;
    }

    bool hasNonDifferentiableAttribute(const clang::Expr* E) {
      // Check MemberExpr
      if (const clang::MemberExpr* ME = clang::dyn_cast<clang::MemberExpr>(E)) {
        // Check attribute of the member declaration
        if (auto* memberDecl = ME->getMemberDecl()) {
          if (hasNonDifferentiableAttribute(memberDecl))
            return true;
        }

        // Check attribute of the base class
        if (auto* classType = ME->getBase()->getType()->getAsCXXRecordDecl()) {
          if (hasNonDifferentiableAttribute(classType))
            return true;
        }
      }
      // Check CallExpr
      else if (const clang::CallExpr* CE =
                   clang::dyn_cast<clang::CallExpr>(E)) {
        // Check if the function is non-differentiable
        if (auto FD = CE->getDirectCallee()) {
          if (hasNonDifferentiableAttribute(FD))
            return true;
        }

        // Check if the base class is non-differentiable
        if (const auto* ME = dyn_cast<CXXMemberCallExpr>(CE)) {
          if (auto* classType = ME->getImplicitObjectArgument()
                                    ->getType()
                                    ->getAsCXXRecordDecl()) {
            if (hasNonDifferentiableAttribute(classType))
              return true;
          }
        }
      }
      // If E is not a MemberExpr or CallExpr or doesn't have a
      // non-differentiable attribute
      return false;
    }
  } // namespace utils
} // namespace clad
