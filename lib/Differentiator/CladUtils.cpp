#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"

#include "ConstantFolder.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/AST/QualTypeNames.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/TemplateDeduction.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include <memory>
#include <vector>

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
      case OverloadedOperatorKind::OO_Spaceship:
        return "operator_spaceship";
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
        if (isa<CXXConstructorDecl>(FD))
          return "constructor";
        if (isa<CXXConversionDecl>(FD))
          return "conversion_operator";
        return FD->getNameAsString();
      }
    }

    Stmt* unwrapIfSingleStmt(Stmt* S) {
      if (!S)
        return nullptr;
      if (!isa<CompoundStmt>(S))
        return S;
      auto* CS = cast<CompoundStmt>(S);
      if (CS->size() == 0)
        return nullptr;
      if (CS->size() == 1)
        return CS->body_front();
      return CS;
    }

    bool hasEmptyBody(const clang::FunctionDecl* FD) {
      FD = FD->getCanonicalDecl();
      if (const FunctionDecl* TIP = FD->getTemplateInstantiationPattern())
        FD = TIP;
      // Derivatives with real locations are user-provided ones. If a
      // user-provided derivative doesn't have a body at this point, we consider
      // it to be empty.
      if (!FD->hasBody())
        return FD->getLocation().isValid();
      return utils::unwrapIfSingleStmt(FD->getBody()) == nullptr;
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
        if (!ND->isInline())
          CSS.Extend(C, ND, /*NamespaceLoc=*/utils::GetValidSLoc(semaRef),
                     /*ColonColonLoc=*/utils::GetValidSLoc(semaRef));
      } else if (auto RD = dyn_cast<CXXRecordDecl>(DC)) {
        auto RDQType = RD->getTypeForDecl()->getCanonicalTypeInternal();
        auto RDTypeSourceInfo = C.getTrivialTypeSourceInfo(RDQType);
        CSS.Extend(C,
                   CLAD_COMPAT_CLANG21_CSSExtendKWLocExtraParam(noLoc)
                       RDTypeSourceInfo->getTypeLoc(),
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
        const clang::CXXRecordDecl* recordDecl = typePtr->getAsCXXRecordDecl();
        const auto* declContext =
            static_cast<const clang::DeclContext*>(recordDecl);
        utils::BuildNNS(semaRef, const_cast<clang::DeclContext*>(declContext), CSS);
        NestedNameSpecifier* NS = CSS.getScopeRep();
        if (auto* Prefix = NS->getPrefix())
          return C.getElaboratedType(clad_compat::ElaboratedTypeKeyword_None,
                                     Prefix, QT);
      }
      return QT;
    }

    DeclContext* FindDeclContext(clang::Sema& semaRef, clang::DeclContext* DC1,
                                 const clang::DeclContext* DC2) {
      llvm::SmallVector<const clang::DeclContext*, 4> contexts;
      assert((isa<NamespaceDecl>(DC1) || isa<TranslationUnitDecl>(DC1)) &&
             "DC1 can only be extended if it is a "
             "namespace or translation unit decl.");
      while (DC2) {
        // If somewhere along the way we reach DC1, then we can break the loop.
        if (DC2->Equals(DC1))
          break;
        if (isa<TranslationUnitDecl>(DC2))
          break;
        if (isa<LinkageSpecDecl>(DC2)) {
          DC2 = DC2->getParent();
          continue;
        }
        if (DC2->isInlineNamespace()) {
          // Inline namespace can be skipped from context, because its members
          // are automatically searched from the parent namespace.
          // This will also help us to deal with intermediate inline namespaces
          // like std::__1::, as present in std functions for libc++.
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
        const auto* ND = cast<NamespaceDecl>(contexts[i]);
        if (ND->getIdentifier())
          DC = LookupNSD(semaRef, ND->getIdentifier()->getName(),
                         /*shouldExist=*/false, DC1);
        if (!DC)
          return nullptr;
        DC1 = DC;
      }
      return DC->getPrimaryContext();
    }

    LookupResult LookupQualifiedName(llvm::StringRef name, clang::Sema& S,
                                     clang::DeclContext* DC) {
      ASTContext& C = S.getASTContext();
      DeclarationName declName = &C.Idents.get(name);
      LookupResult Result(S, declName, SourceLocation(),
                          Sema::LookupOrdinaryName);
      if (!DC)
        DC = C.getTranslationUnitDecl();
      S.LookupQualifiedName(Result, DC);

      if (auto* CXXRD = dyn_cast<CXXRecordDecl>(DC))
        Result.setNamingClass(CXXRD);

      return Result;
    }

    NamespaceDecl* LookupNSD(Sema& S, llvm::StringRef namespc, bool shouldExist,
                             DeclContext* DC) {
      ASTContext& C = S.getASTContext();
      if (!DC)
        DC = C.getTranslationUnitDecl();
      // Find the builtin derivatives/numerical diff namespace
      DeclarationName Name = &C.Idents.get(namespc);
      LookupResult R(S, Name, SourceLocation(), Sema::LookupNamespaceName,
                     CLAD_COMPAT_Sema_ForVisibleRedeclaration);
      S.LookupQualifiedName(R, DC,
                            /*allowBuiltinCreation*/ false);
      if (!shouldExist && R.empty())
        return nullptr;
      assert(!R.empty() && "Cannot find the specified namespace!");
      NamespaceDecl* ND = cast<NamespaceDecl>(R.getFoundDecl());
      return cast<NamespaceDecl>(ND->getPrimaryContext());
    }

    StringLiteral* CreateStringLiteral(ASTContext& C, llvm::StringRef str) {
      // Copied and adapted from clang::Sema::ActOnStringLiteral.
      QualType CharTyConst = C.CharTy.withConst();
      QualType StrTy = clad_compat::getConstantArrayType(
          C, CharTyConst, llvm::APInt(/*numBits=*/32, str.size() + 1),
          /*SizeExpr=*/nullptr,
          /*ASM=*/clad_compat::ArraySizeModifier_Normal,
          /*IndexTypeQuals*/ 0);
      StringLiteral* SL = StringLiteral::Create(
          C, str,
          /*Kind=*/clad_compat::StringLiteralKind_Ordinary,
          /*Pascal=*/false, StrTy, noLoc);
      return SL;
    }

    bool isArrayOrPointerType(clang::QualType QT) {
      return QT->isArrayType() || QT->isPointerType();
    }

    bool isLinearConstructor(const clang::CXXConstructorDecl* CD,
                             const clang::ASTContext& C) {
      // Trivial constructors are linear
      if (CD->isTrivial())
        return true;
      // If the body is not empty, the constructor is not considered linear
      if (!isa<CompoundStmt>(CD->getBody()))
        return false;
      auto* CS = cast<CompoundStmt>(CD->getBody());
      if (!CS->body_empty())
        return false;
      // If some of the inits is non-linear, the constructor is not
      for (CXXCtorInitializer* CI : CD->inits()) {
        Expr* init = CI->getInit()->IgnoreImplicit();
        Expr::EvalResult dummy;
        if (!(isa<DeclRefExpr>(init) || isa<CXXConstructExpr>(init) ||
              clad_compat::Expr_EvaluateAsConstantExpr(init, dummy, C)))
          return false;
      }
      // The constructor is linear
      return true;
    }

    bool isElidableConstructor(const clang::CXXConstructorDecl* CD,
                               const clang::ASTContext& C) {
      const CXXRecordDecl* RD = CD->getParent();
      if (CD->isCopyOrMoveConstructor() && CD->isTrivial())
        return true;
      if (RD->isAggregate())
        return true;
      if (utils::unwrapIfSingleStmt(CD->getBody()))
        return false;
      bool isZeroOrCopyCtor = true;
      for (CXXCtorInitializer* CI : CD->inits()) {
        if (CI->isMemberInitializer()) {
          QualType memberTy = CI->getMember()->getType();
          if (memberTy->isPointerType() || memberTy->isLValueReferenceType())
            continue;
        }
        Expr* init = CI->getInit()->IgnoreImplicit();
        Expr::EvalResult value;
        bool isZero = false;
        if (clad_compat::Expr_EvaluateAsConstantExpr(init, value, C)) {
          const auto* val = &value.Val;
          if (val->isArray() && val->hasArrayFiller())
            val = &val->getArrayFiller();
          if (val->isInt())
            isZero = val->getInt() == 0;
          else if (val->isFloat())
            isZero = val->getFloat().isZero();
        }

        if (!(isa<DeclRefExpr>(init) || isa<CXXConstructExpr>(init) ||
              isZero)) {
          isZeroOrCopyCtor = false;
          break;
        }
      }
      return isZeroOrCopyCtor;
    }

    void getRecordDeclFields(
        const clang::RecordDecl* RD,
        llvm::SmallVectorImpl<const clang::FieldDecl*>& fields) {
      for (const auto* field : RD->fields())
        fields.push_back(field);
      if (const auto* CRD = dyn_cast<CXXRecordDecl>(RD))
        for (const CXXBaseSpecifier& base : CRD->bases())
          if (const auto* baseRT = base.getType()->getAs<clang::RecordType>())
            getRecordDeclFields(baseRT->getDecl(), fields);
    }

    clang::DeclarationNameInfo BuildDeclarationNameInfo(clang::Sema& S,
                                                        llvm::StringRef name) {
      ASTContext& C = S.getASTContext();
      IdentifierInfo* II = &C.Idents.get(name);
      return DeclarationNameInfo(II, noLoc);
    }

    bool IsRealFunction(const clang::FunctionDecl* FD) {
      if (isa<CXXMethodDecl>(FD) || !FD->getReturnType()->isRealType())
        return false;
      for (auto PVD : FD->parameters()) {
        if (!PVD->getType()->isRealType())
          return false;
      }
      return true;
    }

    bool IsReferenceOrPointerArg(const Expr* arg) {
      // The argument is passed by reference if it's an L-value
      // and the parameter type is L-value too.
      const Expr* subExpr = arg->IgnoreImplicit();
      if (const auto* DAE = dyn_cast<CXXDefaultArgExpr>(subExpr))
        subExpr = DAE->getExpr()->IgnoreImplicit();
      bool isRefType = arg->isGLValue() && subExpr->isGLValue();
      return isRefType || isArrayOrPointerType(arg->getType());
    }

    bool isSameCanonicalType(clang::QualType T1, clang::QualType T2) {
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
                     clang::TypeSourceInfo* TSI, clang::SourceLocation Loc) {
      ASTContext& C = semaRef.getASTContext();
      if (!TSI)
        TSI = C.getTrivialTypeSourceInfo(T, noLoc);
      if (Loc.isInvalid())
        Loc = utils::GetValidSLoc(semaRef);
      ParmVarDecl* PVD =
          ParmVarDecl::Create(C, DC, Loc, Loc, II, T, TSI, SC, defArg);
      return PVD;
    }

    clang::QualType GetValueType(clang::QualType T) {
      QualType valueType = T;
      if (T->isPointerType())
        valueType = T->getPointeeType();
      else if (T->isReferenceType())
        valueType = T.getNonReferenceType();
      else if (const auto* AT = dyn_cast<clang::ArrayType>(T))
        valueType = AT->getElementType();
      return valueType;
    }

    clang::QualType GetNonConstValueType(clang::QualType T) {
      QualType valueType = GetValueType(T);
      valueType.removeLocalConst();
      // If the const-ness of the type is hidden with sugar, e.g.
      // `class_name<double>::const_value_type`, the approach above
      // does not work and we have to desugar the type explicitly.
      QualType canonicalType = valueType.getCanonicalType();
      if (canonicalType.isConstQualified()) {
        canonicalType.removeLocalConst();
        return canonicalType;
      }
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
                                clang::TypeSourceInfo* TSI,
                                clang::MultiExprArg ArgExprs) {
      auto& C = semaRef.getASTContext();
      if (!TSI)
        TSI = C.getTrivialTypeSourceInfo(qType);
      if (clad_compat::isa_and_nonnull<ImplicitValueInitExpr>(initializer))
        // If the initializer is an implicit value init expression, then
        // we don't need to pass it explicitly to the CXXNewExpr. As, clang
        // internally adds it when initializer is ParenListExpr and
        // DirectInitRange is valid.
        initializer = semaRef.ActOnParenListExpr(noLoc, noLoc, {}).get();
      auto newExpr =
          semaRef
              .BuildCXXNew(
                  SourceRange(), false, noLoc, ArgExprs, noLoc, SourceRange(),
                  qType, TSI,
                  arraySize ? arraySize : clad_compat::llvm_Optional<Expr*>(),
                  initializer ? GetValidSRange(semaRef) : SourceRange(),
                  initializer)
              .getAs<CXXNewExpr>();
      return newExpr;
    }

    /// Removes the local const qualifiers from a QualType and returns a new
    /// type.
    clang::QualType getNonConstType(clang::QualType T, clang::Sema& S) {
      bool isLValueRefType = T->isLValueReferenceType();
      if (const auto* CAT = llvm::dyn_cast<clang::ConstantArrayType>(T)) {
        QualType elemType = GetNonConstValueType(T);
        T = S.getASTContext().getConstantArrayType(
            elemType, CAT->getSize(), CAT->getSizeExpr(),
            CAT->getSizeModifier(), CAT->getIndexTypeCVRQualifiers());
      }
      T = T.getNonReferenceType();
      bool isConstPtrType =
          T->isPointerType() && T->getPointeeType().isConstQualified();
      if (isConstPtrType)
        T = T->getPointeeType();
      clang::Qualifiers quals(T.getQualifiers());
      quals.removeConst();
      clang::QualType nonConstType =
          S.BuildQualifiedType(T.getUnqualifiedType(), noLoc, quals);
      if (isConstPtrType)
        nonConstType = S.getASTContext().getPointerType(nonConstType);
      if (isLValueRefType)
        return S.getASTContext().getLValueReferenceType(nonConstType);
      return nonConstType;
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
      if (const auto* VD = dyn_cast<VarDecl>(D)) {
        QualType VDElemTy = utils::GetValueType(VD->getType());
        const CXXRecordDecl* RD = VDElemTy->getAsCXXRecordDecl();
        if (RD && clad::utils::hasNonDifferentiableAttribute(RD))
          return true;
      }
      return false;
    }

    bool hasElidableReverseForwAttribute(const clang::Decl* D) {
      for (auto* Attr : D->specific_attrs<clang::AnnotateAttr>())
        if (Attr->getAnnotation() == "elidable_reverse_forw")
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
      } else if (const auto* CXXCE =
                     clang::dyn_cast<clang::CXXConstructExpr>(E)) {
        if (auto* typeDecl = CXXCE->getType()->getAsCXXRecordDecl())
          if (hasNonDifferentiableAttribute(typeDecl))
            return true;
      }
      // If E is not a MemberExpr or CallExpr or doesn't have a
      // non-differentiable attribute
      return false;
    }

    void GetInnermostReturnExpr(const clang::Expr* E,
                                llvm::SmallVectorImpl<clang::Expr*>& Exprs) {
      struct Finder : public StmtVisitor<Finder> {
        llvm::SmallVectorImpl<clang::Expr*>& m_Exprs;

      public:
        Finder(clang::Expr* E, llvm::SmallVectorImpl<clang::Expr*>& Exprs)
            : m_Exprs(Exprs) {
          Visit(E);
        }

        void VisitBinaryOperator(clang::BinaryOperator* BO) {
          if (BO->isAssignmentOp() || BO->isCompoundAssignmentOp())
            Visit(BO->getLHS());
        }

        void VisitConditionalOperator(clang::ConditionalOperator* CO) {
          // FIXME: in cases like (cond ? x : y) = 2; both x and y will be
          // stored.
          Visit(CO->getTrueExpr());
          Visit(CO->getFalseExpr());
        }

        void VisitUnaryOperator(clang::UnaryOperator* UnOp) {
          auto opCode = UnOp->getOpcode();
          if (opCode == clang::UO_PreInc || opCode == clang::UO_PreDec)
            Visit(UnOp->getSubExpr());
          else if (opCode == UnaryOperatorKind::UO_Real ||
                   opCode == UnaryOperatorKind::UO_Imag) {
            /// FIXME: Considering real/imaginary part atomic is
            /// not always correct since the subexpression can
            /// be more complex than just a DeclRefExpr.
            /// (e.g. `__real (n++ ? z1 : z2)`)
            m_Exprs.push_back(UnOp);
          } else if (opCode == UnaryOperatorKind::UO_Deref)
            m_Exprs.push_back(UnOp);
        }

        void VisitDeclRefExpr(clang::DeclRefExpr* DRE) {
          m_Exprs.push_back(DRE);
        }

        void VisitParenExpr(clang::ParenExpr* PE) { Visit(PE->getSubExpr()); }

        void VisitMemberExpr(clang::MemberExpr* ME) { m_Exprs.push_back(ME); }

        void VisitArraySubscriptExpr(clang::ArraySubscriptExpr* ASE) {
          m_Exprs.push_back(ASE);
        }

        void VisitImplicitCastExpr(clang::ImplicitCastExpr* ICE) {
          Visit(ICE->getSubExpr());
        }
      };
      // FIXME: Fix the constness on the callers of this function.
      Finder finder(const_cast<clang::Expr*>(E), Exprs);
    }

    bool IsAutoOrAutoPtrType(QualType T) {
      if (isa<clang::AutoType>(T))
        return true;

      if (const auto* const pointerType = dyn_cast<clang::PointerType>(T))
        return IsAutoOrAutoPtrType(pointerType->getPointeeType());

      return false;
    }

    bool UsefulToStore(const Expr* E) {
      assert(E && "Must be non-null.");
      E = E->IgnoreParenImpCasts();
      // FIXME: find a more general way to determine that or add more options.
      if (isa<DeclRefExpr>(E) || isa<FloatingLiteral>(E) ||
          isa<IntegerLiteral>(E))
        return false;
      if (const auto* UO = dyn_cast<UnaryOperator>(E)) {
        auto OpKind = UO->getOpcode();
        if (OpKind == UO_Plus || OpKind == UO_Minus)
          return UsefulToStore(UO->getSubExpr());
        return false;
      }
      if (const auto* ASE = dyn_cast<ArraySubscriptExpr>(E))
        return UsefulToStore(ASE->getBase()) || UsefulToStore(ASE->getIdx());
      return true;
    }

    bool ContainsFunctionCalls(const clang::Stmt* S) {
      class CallExprFinder : public RecursiveASTVisitor<CallExprFinder> {
      public:
        bool hasCallExpr = false;

        bool VisitCallExpr(CallExpr* CE) {
          hasCallExpr = true;
          return false;
        }
      };
      CallExprFinder finder;
      finder.TraverseStmt(const_cast<Stmt*>(S));
      return finder.hasCallExpr;
    }

    void SetSwitchCaseSubStmt(SwitchCase* SC, Stmt* subStmt) {
      if (auto* caseStmt = dyn_cast<CaseStmt>(SC))
        caseStmt->setSubStmt(subStmt);
      else
        cast<DefaultStmt>(SC)->setSubStmt(subStmt);
    }

    bool IsZeroOrNullValue(const clang::Expr* E) {
      if (!E)
        return true;
      if (const auto* ICE = dyn_cast<ImplicitCastExpr>(E))
        return IsZeroOrNullValue(ICE->getSubExpr());
      if (isa<CXXNullPtrLiteralExpr>(E))
        return true;
      if (const auto* FL = dyn_cast<FloatingLiteral>(E))
        return FL->getValue().isZero();
      if (const auto* IL = dyn_cast<IntegerLiteral>(E))
        return IL->getValue() == 0;
      if (const auto* SL = dyn_cast<StringLiteral>(E))
        return SL->getLength() == 0;
      return false;
    }

    bool IsMemoryDeallocationFunction(const clang::FunctionDecl* FD) {
      if (FD->getNameAsString() == "cudaFree")
        return true;
#if CLANG_VERSION_MAJOR > 12
      return FD->getBuiltinID() == Builtin::ID::BIfree;
#else
      return FD->getNameAsString() == "free";
#endif
    }

    bool isNonConstReferenceType(clang::QualType QT) {
      return QT->isReferenceType() &&
             !QT.getNonReferenceType().isConstQualified();
    }

    bool isCopyable(const clang::CXXRecordDecl* RD) {
      if (RD->defaultedCopyConstructorIsDeleted())
        return false;
      if (RD->hasUserDeclaredCopyConstructor()) {
        std::string qualifiedName = RD->getQualifiedNameAsString();
        // FIXME: I don't know why Clang things that unique_ptr has
        // user-declared copy constructor.
        if (qualifiedName == "std::unique_ptr")
          return false;
      }
      return true;
    }

    NamespaceDecl* GetCladNamespace(Sema& S) {
      static NamespaceDecl* Result = nullptr;
      if (Result)
        return Result;
      DeclarationName CladName = &S.getASTContext().Idents.get("clad");
      LookupResult CladR(S, CladName, noLoc, Sema::LookupNamespaceName,
                         CLAD_COMPAT_Sema_ForVisibleRedeclaration);
      S.LookupQualifiedName(CladR, S.getASTContext().getTranslationUnitDecl());
      assert(!CladR.empty() && "cannot find clad namespace");
      Result = cast<NamespaceDecl>(CladR.getFoundDecl());
      return Result;
    }

    Expr* getZeroInit(QualType T, Sema& S) {
      // FIXME: Consolidate other uses of synthesizeLiteral for creation 0 or 1.
      if (T->isVoidType() || isa<VariableArrayType>(T))
        return nullptr;
      if ((T->isScalarType() || T->isPointerType()) && !T->isReferenceType())
        return ConstantFolder::synthesizeLiteral(T, S.getASTContext(),
                                                 /*val=*/0);
      if (isa<ConstantArrayType>(T)) {
        Expr* zero =
            ConstantFolder::synthesizeLiteral(T, S.getASTContext(), /*val=*/0);
        return S.ActOnInitList(noLoc, {zero}, noLoc).get();
      }
      if (const auto* RD = T->getAsCXXRecordDecl())
        if (RD->hasDefinition() && !RD->isUnion() && RD->isAggregate()) {
          llvm::SmallVector<Expr*, 4> adjParams;
          for (const FieldDecl* FD : RD->fields())
            adjParams.push_back(getZeroInit(FD->getType(), S));
          return S.ActOnInitList(noLoc, adjParams, noLoc).get();
        }
      return S.ActOnInitList(noLoc, {}, noLoc).get();
    }

    QualType InstantiateTemplate(Sema& S, TemplateDecl* CladClassDecl,
                                 TemplateArgumentListInfo& TLI) {
      // This will instantiate tape<T> type and return it.
      QualType TT = S.CheckTemplateIdType(TemplateName(CladClassDecl),
                                          GetValidSLoc(S), TLI);
      // Get clad namespace and its identifier clad::.
      CXXScopeSpec CSS;
      CSS.Extend(S.getASTContext(), GetCladNamespace(S), GetValidSLoc(S),
                 GetValidSLoc(S));
      NestedNameSpecifier* NS = CSS.getScopeRep();

      // Create elaborated type with namespace specifier,
      // i.e. class<T> -> clad::class<T>
      return S.getASTContext().getElaboratedType(
          clad_compat::ElaboratedTypeKeyword_None, NS, TT);
    }

    clang::QualType GetRestoreTrackerType(clang::Sema& S) {
      static QualType T;
      if (!T.isNull())
        return T;
      NamespaceDecl* CladNS = GetCladNamespace(S);
      CXXScopeSpec CSS;
      CSS.Extend(S.getASTContext(), CladNS, noLoc, noLoc);
      DeclarationName TrackerName =
          &S.getASTContext().Idents.get("restore_tracker");
      LookupResult TrackerR(S, TrackerName, noLoc, Sema::LookupUsingDeclName,
                            CLAD_COMPAT_Sema_ForVisibleRedeclaration);
      S.LookupQualifiedName(TrackerR, CladNS, CSS);
      assert(!TrackerR.empty() && "cannot find clad::restore_tracker");

      // This will instantiate restore_tracker<T> type and return it.
      auto* RD = cast<RecordDecl>(TrackerR.getFoundDecl());
      ASTContext& C = S.getASTContext();
      T = C.getRecordType(RD);
      // Get clad namespace and its identifier clad::.
      NestedNameSpecifier* NS = CSS.getScopeRep();

      // Create elaborated type with namespace specifier,
      // i.e. class<T> -> clad::class<T>
      T = C.getElaboratedType(clad_compat::ElaboratedTypeKeyword_None, NS, T);
      return T;
    }

    TemplateDecl* LookupTemplateDeclInCladNamespace(Sema& S,
                                                    llvm::StringRef ClassName) {
      NamespaceDecl* CladNS = GetCladNamespace(S);
      CXXScopeSpec CSS;
      CSS.Extend(S.getASTContext(), CladNS, noLoc, noLoc);
      DeclarationName TapeName = &S.getASTContext().Idents.get(ClassName);
      LookupResult TapeR(S, TapeName, noLoc, Sema::LookupUsingDeclName,
                         CLAD_COMPAT_Sema_ForVisibleRedeclaration);
      S.LookupQualifiedName(TapeR, CladNS, CSS);
      assert(!TapeR.empty() && isa<TemplateDecl>(TapeR.getFoundDecl()) &&
             "cannot find clad::tape");
      return cast<TemplateDecl>(TapeR.getFoundDecl());
    }

    Expr* MatchOverloadType(Sema& S, QualType FnTy, LookupResult& Overloads,
                            TemplateSpecCandidateSet& FailedCandidates) {
      CXXScopeSpec SS;
      ASTContext& C = S.getASTContext();
      // Check if any of the custom derivative signature satisfy the
      // requirements.
      for (LookupResult::iterator I = Overloads.begin(), E = Overloads.end();
           I != E; ++I) {
        NamedDecl* candidate = I.getDecl();
        // Shadow decls don't provide enough information, go to the actual decl.
        if (auto* usingShadow = dyn_cast<UsingShadowDecl>(candidate))
          candidate = usingShadow->getTargetDecl();

        // Overload is a template, try to match the signature.
        if (auto* FTD = dyn_cast<FunctionTemplateDecl>(candidate)) {
          FunctionDecl* Specialization = nullptr;
          sema::TemplateDeductionInfo Info(FailedCandidates.getLocation());
          TemplateArgumentListInfo ExplicitTemplateArgs;
          auto TDK = S.DeduceTemplateArguments(FTD, &ExplicitTemplateArgs, FnTy,
                                               Specialization, Info);

          // Instantiation with the required signature succeeded.
          if (TDK == clad_compat::CLAD_COMPAT_TemplateSuccess)
            return S.BuildDeclarationNameExpr(SS, Overloads, /*ADL=*/false)
                .get();

          FailedCandidates.addCandidate().set(
              I.getPair(), FTD->getTemplatedDecl(),
              MakeDeductionFailureInfo(C, TDK, Info));

          // Instantiation of parameters suceeded but clang doesn't consider
          // deduction successful because of the auto return type.
          if (Specialization && !Specialization->isTemplated() &&
              Specialization->getReturnType()->isUndeducedAutoType())
            return S.BuildDeclarationNameExpr(SS, Overloads, /*ADL=*/false)
                .get();
        }
        auto* FD = dyn_cast<FunctionDecl>(candidate);
        if (!FD)
          continue;
        // Overload is just a FunctionDecl, check if the signature matches.
        if (C.hasSameFunctionTypeIgnoringExceptionSpec(FD->getType(), FnTy))
          return S.BuildDeclarationNameExpr(SS, Overloads, /*ADL=*/false).get();
      }
      return nullptr;
    }

    void DiagnoseSignatureMismatch(Sema& S, QualType FnTy,
                                   const LookupResult& Overloads) {
      ASTContext& C = S.getASTContext();
      std::string Name = Overloads.getLookupName().getAsString();
      unsigned noteId = S.Diags.getCustomDiagID(
          DiagnosticsEngine::Note,
          "candidate '%0'"
          "%select{| has different class%diff{ (expected $ but has $)|}1,2"
          "| has different number of parameters (expected %2 but has %3)"
          "| has type mismatch at %ordinal2 parameter"
          "%diff{ (expected $ but has $)|}3,4"
          "| has different return type%diff{ ($ expected but has $)|}2,3"
          "| has different qualifiers (expected %2 but found %3)"
          "| has different exception specification}1");

      for (const NamedDecl* ND : Overloads) {
        if (const auto* usingShadow = dyn_cast<UsingShadowDecl>(ND))
          ND = usingShadow->getTargetDecl();
        if (!isa<FunctionDecl>(ND))
          continue;
        const auto* FD = cast<FunctionDecl>(ND);
        auto PD = PartialDiagnostic(noteId, C.getDiagAllocator()) << Name;
        S.HandleFunctionTypeMismatch(PD, FD->getType(), FnTy);
        S.Diag(FD->getLocation(), PD);
      }
    }

    bool isMemoryType(QualType T) {
      T = T.getCanonicalType();
      if (T->isReferenceType())
        return !T.getNonReferenceType().isConstQualified();
      if (T->isPointerType())
        return !T->getPointeeType().isConstQualified();
      if (const auto* RT = T->getAs<RecordType>()) {
        const RecordDecl* RD = RT->getDecl();
        if (const auto* CRD = dyn_cast<CXXRecordDecl>(RD))
          if (!isCopyable(CRD))
            return true;
        if (const auto* CRD = dyn_cast<CXXRecordDecl>(RD))
          for (const CXXBaseSpecifier& base : CRD->bases())
            if (isMemoryType(base.getType()))
              return true;
        for (const auto* field : RD->fields())
          if (isMemoryType(field->getType()))
            return true;
        return false;
      }
      return false;
    }

    bool shouldUseRestoreTracker(const FunctionDecl* FD) {
      const auto* MD = dyn_cast<CXXMethodDecl>(FD);
      if (MD && MD->isInstance() && !MD->isConst())
        return true;
      // FIXME: clad::restore_tracker is not thread-safe.
      // We shoudn't disable reverse_forw for CUDA
      if (FD->hasAttr<clang::CUDAGlobalAttr>() ||
          FD->hasAttr<clang::CUDADeviceAttr>() ||
          FD->hasAttr<clang::CUDAHostAttr>())
        return false;
      for (const ParmVarDecl* PVD : FD->parameters()) {
        // Some functions (like in Kokkos) have dummy parameters for
        // metaprogramming. They don't have names and are not unused. They are
        // not essential for the analysis but sometimes, due to pointer types
        // like `enable_if<...>::type*`, can unnecessarily trigger reverse_forw.
        if (PVD->getDeclName().isEmpty())
          continue;
        QualType paramTy = PVD->getType();
        if (paramTy->isReferenceType() &&
            paramTy.getNonReferenceType()->isRealType())
          continue;
        if (isMemoryType(paramTy))
          return true;
      }
      return false;
    }

    bool hasMemoryTypeParams(const FunctionDecl* FD) {
      if (const auto* MD = dyn_cast<CXXMethodDecl>(FD))
        if (MD->isInstance() && !MD->isConst())
          return true;
      for (const ParmVarDecl* PVD : FD->parameters())
        if (isMemoryType(PVD->getType()))
          return true;
      return false;
    }

    QualType InstantiateTemplate(Sema& S, TemplateDecl* CladClassDecl,
                                 ArrayRef<QualType> TemplateArgs) {
      // Create a list of template arguments.
      TemplateArgumentListInfo TLI{};
      for (auto T : TemplateArgs) {
        TemplateArgument TA = T;
        TLI.addArgument(TemplateArgumentLoc(
            TA, S.getASTContext().getTrivialTypeSourceInfo(T)));
      }

      return InstantiateTemplate(S, CladClassDecl, TLI);
    }

    QualType GetCladMatrixOfType(Sema& S, clang::QualType T) {
      static TemplateDecl* matrixDecl = nullptr;
      if (!matrixDecl)
        matrixDecl =
            utils::LookupTemplateDeclInCladNamespace(S,
                                                     /*ClassName=*/"matrix");
      return InstantiateTemplate(S, matrixDecl, {T});
    }

    QualType GetCladArrayOfType(Sema& S, clang::QualType T) {
      static TemplateDecl* arrayDecl = nullptr;
      if (!arrayDecl)
        arrayDecl = LookupTemplateDeclInCladNamespace(S, /*ClassName=*/"array");
      return utils::InstantiateTemplate(S, arrayDecl, {T});
    }

    bool IsDifferentiableType(QualType T) {
      QualType origType = T;
      T = T.getCanonicalType();
      // FIXME: arbitrary dimension array type as well.
      while (utils::isArrayOrPointerType(T) || T->isReferenceType())
        T = utils::GetValueType(T);
      T = T.getNonReferenceType();
      if (T->isEnumeralType())
        return false;
      if (T->isRealType() || T->isStructureOrClassType() || T->isUnionType())
        return true;
      if (origType->isPointerType() && T->isVoidType())
        return true;
      return false;
    }

    bool exprDependsOnVarDecl(const clang::Expr* E, const VarDecl* VD) {
      class DREFinder : public RecursiveASTVisitor<DREFinder> {
      public:
        DREFinder(const VarDecl* VD) : declToFind(VD) {}
        const VarDecl* declToFind;
        bool dependsOnDecl = false;

        bool VisitDeclRefExpr(DeclRefExpr* DRE) {
          if (const auto* VD = dyn_cast<VarDecl>(DRE->getDecl())) {
            if (!VD->getType()->isBuiltinType() || VD == declToFind) {
              dependsOnDecl = true;
              return false;
            }
          }
          return true;
        }
      };
      DREFinder finder(VD);
      finder.TraverseStmt(const_cast<Expr*>(E));
      return finder.dependsOnDecl;
    }

    QualType GetCladArrayRefOfType(Sema& S, QualType T) {
      static TemplateDecl* arrayRefDecl = nullptr;
      if (!arrayRefDecl)
        arrayRefDecl = utils::LookupTemplateDeclInCladNamespace(
            S, /*ClassName=*/"array_ref");
      return utils::InstantiateTemplate(S, arrayRefDecl, {T});
    }

    QualType GetParameterDerivativeType(Sema& S, DiffMode Mode, QualType Type) {
      ASTContext& C = S.getASTContext();
      if (Mode == DiffMode::vector_pushforward || Mode == DiffMode::jacobian) {
        QualType valueType = GetNonConstValueType(Type);
        QualType resType;
        if (isArrayOrPointerType(Type)) {
          // If the parameter is a pointer or an array, then the derivative will
          // be a reference to the matrix.
          resType = GetCladMatrixOfType(S, valueType);
          resType = C.getLValueReferenceType(resType);
        } else {
          // If the parameter is not a pointer or an array, then the derivative
          // will be a clad array.
          resType = GetCladArrayOfType(S, valueType);

          // Add const qualifier if the parameter is const.
          if (Type.getNonReferenceType().isConstQualified())
            resType.addConst();

          // Add reference qualifier if the parameter is a reference.
          if (Type->isReferenceType())
            resType = C.getLValueReferenceType(resType);
        }
        if (Mode == DiffMode::jacobian)
          resType = C.getPointerType(resType.getNonReferenceType());
        return resType;
      }

      if (Mode == DiffMode::reverse || Mode == DiffMode::pullback) {
        QualType ValueType = GetNonConstValueType(Type);
        QualType nonRefValueType = ValueType.getNonReferenceType();
        return C.getPointerType(nonRefValueType);
      }

      if (Mode == DiffMode::vector_forward_mode) {
        QualType valueType = GetNonConstValueType(Type);
        if (isArrayOrPointerType(Type))
          // Generate array reference type for the derivative.
          return GetCladArrayRefOfType(S, valueType);
        // Generate pointer type for the derivative.
        return C.getPointerType(valueType);
      }

      return Type;
    }

    static bool isNAT(QualType T) {
      T = GetValueType(T);
      if (const auto* RT = T->getAs<RecordType>()) {
        const RecordDecl* RD = RT->getDecl();
        if (RD->getNameAsString() == "__nat")
          return true;
      }
      return false;
    }

    QualType
    GetDerivativeType(Sema& S, const clang::FunctionDecl* FD, DiffMode mode,
                      llvm::ArrayRef<const clang::ValueDecl*> diffParams,
                      bool forCustomDerv, bool shouldUseRestoreTracker,
                      bool isForErrorEstimation) {
      ASTContext& C = S.getASTContext();
      if (mode == DiffMode::forward)
        return FD->getType();

      QualType FnTy = FD->getType();

      if (const auto* AnnotatedFnTy = dyn_cast<AttributedType>(FnTy))
        FnTy = AnnotatedFnTy->getEquivalentType();

      const auto* FnProtoTy = llvm::cast<FunctionProtoType>(FnTy);
      FunctionProtoType::ExtProtoInfo EPI = FnProtoTy->getExtProtoInfo();
      llvm::SmallVector<QualType, 16> FnTypes;
      FnTypes.reserve(2 * FnProtoTy->getNumParams() + 1);
      for (QualType T : FnProtoTy->getParamTypes()) {
        // FIXME: We handle parameters with default values by setting them
        // explicitly. However, some of them have private types and cannot be
        // set. For this reason, we ignore std::__nat. We need to come up with a
        // general solution.
        if (isNAT(T))
          break;
        FnTypes.push_back(T);
      }
      if (mode == DiffMode::reverse || mode == DiffMode::pullback)
        for (QualType& T : FnTypes)
          T = utils::replaceStdInitListWithCladArray(S, T);

      QualType oRetTy = FD->getReturnType();
      if (const auto* CD = dyn_cast<CXXConstructorDecl>(FD))
        oRetTy = CD->getThisType()->getPointeeType();
      QualType dRetTy = C.VoidTy;
      bool returnVoid = mode == DiffMode::reverse ||
                        mode == DiffMode::pullback ||
                        mode == DiffMode::vector_forward_mode;
      if (mode == DiffMode::reverse_mode_forward_pass) {
        if (isMemoryType(oRetTy) || isa<CXXConstructorDecl>(FD)) {
          TemplateDecl* valAndAdjointTempDecl =
              utils::LookupTemplateDeclInCladNamespace(S, "ValueAndAdjoint");
          dRetTy = utils::InstantiateTemplate(S, valAndAdjointTempDecl,
                                              {oRetTy, oRetTy});
        } else {
          dRetTy = oRetTy;
        }
      } else if (mode == DiffMode::hessian ||
                 mode == DiffMode::hessian_diagonal) {
        QualType argTy = C.getPointerType(oRetTy);
        FnTypes.push_back(argTy);
        return C.getFunctionType(dRetTy, FnTypes, EPI);
      } else if (!returnVoid && !oRetTy->isVoidType()) {
        // Handle pushforwards
        TemplateDecl* valueAndPushforward =
            utils::LookupTemplateDeclInCladNamespace(S, "ValueAndPushforward");
        QualType PushFwdTy = utils::GetParameterDerivativeType(S, mode, oRetTy);
        dRetTy = utils::InstantiateTemplate(S, valueAndPushforward,
                                            {oRetTy, PushFwdTy});
      } else if (mode == DiffMode::pullback) {
        // Handle pullbacks
        QualType argTy = oRetTy.getNonReferenceType();
        argTy = utils::getNonConstType(argTy, S);
        if (!argTy->isVoidType() && !argTy->isPointerType() &&
            !utils::isNonConstReferenceType(oRetTy)) {
          if (isa<CXXConstructorDecl>(FD))
            argTy = C.getPointerType(argTy);
          FnTypes.push_back(argTy);
        }
      }

      QualType thisTy;
      if (const auto* MD = dyn_cast<CXXMethodDecl>(FD)) {
        const CXXRecordDecl* RD = MD->getParent();
        if (MD->isInstance() && !RD->isLambda() && mode != DiffMode::jacobian &&
            !isa<CXXConstructorDecl>(MD)) {
          thisTy = MD->getThisType();
          QualType dthisTy = utils::GetParameterDerivativeType(S, mode, thisTy);
          FnTypes.push_back(dthisTy);
          if (MD->isConst()) {
            QualType constObjTy = C.getConstType(thisTy->getPointeeType());
            thisTy = C.getPointerType(constObjTy);
          }
        }
      }

      // Iterate over all but the "this" type and extend the signature to add
      // the extra parameters.
      for (size_t i = 0, e = FnProtoTy->getNumParams(); i < e; ++i) {
        // FIXME: We handle parameters with default values by setting them
        // explicitly. However, some of them have private types and cannot be
        // set. For this reason, we ignore std::__nat. We need to come up with a
        // general solution.
        if (isNAT(FnProtoTy->getParamType(i)))
          break;
        QualType PVDTy = FnTypes[i];
        if (mode == DiffMode::jacobian &&
            !(utils::isArrayOrPointerType(PVDTy) || PVDTy->isReferenceType()))
          continue;
        // FIXME: Make this system consistent across modes.
        if (returnVoid) {
          // Check if (IsDifferentiableType(PVDTy))
          // FIXME: We can't use std::find(DVI.begin(), DVI.end()) because the
          // operator== considers params and intervals as different entities and
          // breaks the hessian tests. We should implement more robust checks in
          // DiffInputVarInfo to check if this is a variable we differentiate
          // wrt.
          for (const ValueDecl* param : diffParams)
            if (param == FD->getParamDecl(i))
              FnTypes.push_back(
                  utils::GetParameterDerivativeType(S, mode, PVDTy));
        } else if (mode == DiffMode::reverse_mode_forward_pass ||
                   utils::IsDifferentiableType(PVDTy))
          FnTypes.push_back(utils::GetParameterDerivativeType(S, mode, PVDTy));
      }

      if (forCustomDerv && !thisTy.isNull()) {
        FnTypes.insert(FnTypes.begin(), thisTy);
        EPI.TypeQuals.removeConst();
      }

      if (mode == DiffMode::reverse_mode_forward_pass) {
        if (isa<CXXConversionDecl>(FD) || isa<CXXConstructorDecl>(FD)) {
          QualType typeTag = utils::GetCladTagOfType(S, oRetTy);
          FnTypes.insert(FnTypes.begin(), typeTag);
        }

        if (shouldUseRestoreTracker) {
          QualType trackerTy = GetRestoreTrackerType(S);
          trackerTy = C.getLValueReferenceType(trackerTy);
          FnTypes.push_back(trackerTy);
        }
      }

      if (isForErrorEstimation)
        FnTypes.push_back(C.getLValueReferenceType(C.DoubleTy));

      return C.getFunctionType(dRetTy, FnTypes, EPI);
    }

    QualType GetCladTagOfType(Sema& S, QualType T) {
      static clang::TemplateDecl* CladTag = nullptr;
      if (!CladTag)
        CladTag = utils::LookupTemplateDeclInCladNamespace(S, "Tag");
      return utils::InstantiateTemplate(S, CladTag, {T});
    }

    Expr* GetCladTagExpr(Sema& S, QualType T) {
      QualType CladTagTy = utils::GetCladTagOfType(S, T);
      return S
          .BuildCXXTypeConstructExpr(S.getASTContext().getTrivialTypeSourceInfo(
                                         CladTagTy, utils::GetValidSLoc(S)),
                                     utils::GetValidSLoc(S), MultiExprArg{},
                                     utils::GetValidSLoc(S),
                                     /*ListInitialization=*/false)
          .get();
    }

    bool canUsePushforwardInRevMode(const FunctionDecl* FD) {
      return FD->getNumParams() == 1 && utils::IsRealFunction(FD);
    }

    QualType makeTypeReadable(Sema& S, QualType Ty) {
      ASTContext& C = S.getASTContext();
      QualType retTy =
          TypeName::getFullyQualifiedType(Ty, C, /*WithGlobalNsPrefix=*/false);

      // FIXME: Add a type visitor which can fold stl-specific idioms such as
      // X::value_type, X::size_type.
      // FIXME: Drop the default arguments such as std::allocator.
      return retTy;
    }

    QualType replaceStdInitListWithCladArray(Sema& S, QualType origTy) {
      QualType T = origTy.getNonReferenceType();
      QualType elemType;
      if (!S.isStdInitializerList(utils::GetValueType(T), &elemType))
        return origTy;
      T = utils::GetCladArrayOfType(S, elemType);
      if (origTy->isLValueReferenceType())
        return S.getASTContext().getLValueReferenceType(T);
      return T;
    }

    static bool isInjectiveE(const clang::Expr* E) {
      class InjectiveCheckerExpr
          : public clang::RecursiveASTVisitor<InjectiveCheckerExpr> {
        struct IdxNode {
          struct ExprOrBinOp {
            const clang::DeclRefExpr* m_E = nullptr;
            clang::BinaryOperatorKind m_Opcode;
            enum class Kind { Expr, Opcode } m_Kind;

            ExprOrBinOp(const clang::DeclRefExpr* E)
                : m_E(E), m_Kind(Kind::Expr) {}
            ExprOrBinOp(clang::BinaryOperatorKind op)
                : m_Opcode(op), m_Kind(Kind::Opcode) {}

            [[nodiscard]] bool isExpr() const { return m_Kind == Kind::Expr; }
            [[nodiscard]] bool isOpcode() const {
              return m_Kind == Kind::Opcode;
            }
          };

          ExprOrBinOp Node;
          std::unique_ptr<IdxNode> left;
          std::unique_ptr<IdxNode> right;

          IdxNode(ExprOrBinOp N) : Node(N) {}
        };
        std::unique_ptr<IdxNode> m_Root;
        IdxNode* m_ParentNode = nullptr;

        enum class side { left, right } m_Side;

      public:
        InjectiveCheckerExpr() = default;
        /// This function recursively checks whether a given pattern matches the
        /// previously computed graph. It uses a fairly standard graph
        /// comparison algorithm.
        bool comparePatternToTree(const IdxNode* current,
                                  const std::vector<std::string>& patternIdx,
                                  size_t i = 1) {
          // If current is not initialized and child is empty or does not exist,
          // we have a match.
          if (!current && (i >= patternIdx.size() || patternIdx[i].empty()))
            return true;

          if (!current || i >= patternIdx.size() || patternIdx[i].empty())
            return false;

          const std::string& expected = patternIdx[i];

          if (current->Node.isOpcode()) {
            std::string actualOp =
                clang::BinaryOperator::getOpcodeStr(current->Node.m_Opcode)
                    .str();
            if (actualOp != expected)
              return false;
          } else if (current->Node.isExpr()) {
            std::string actualName =
                current->Node.m_E->getNameInfo().getAsString();
            if (actualName != expected)
              return false;
          }

          bool sameOrder =
              comparePatternToTree(current->left.get(), patternIdx, 2 * i) &&
              comparePatternToTree(current->right.get(), patternIdx, 2 * i + 1);

          if (sameOrder)
            return true;
          // Here we account for a sub-tree rotation wrt to the current node. If
          // there is no match at this point, we compare a pattern to a graph
          // with the rotation.
          return comparePatternToTree(current->left.get(), patternIdx,
                                      2 * i + 1) &&
                 comparePatternToTree(current->right.get(), patternIdx, 2 * i);
        }

        bool isInjectiveIdx(const clang::Expr* E) {
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          TraverseStmt(const_cast<clang::Expr*>(E));
          std::vector<std::string> pattern = {"", "+", "threadIdx", "*",
                                              "", "",  "blockIdx",  "blockDim"};
          return comparePatternToTree(m_Root.get(), pattern);
        }

        bool TraverseBinaryOperator(clang::BinaryOperator* BinOp) {
          const auto opCode = BinOp->getOpcode();
          if (opCode == BO_Add || opCode == BO_Mul) {
            std::unique_ptr<IdxNode> curr = std::make_unique<IdxNode>(opCode);
            IdxNode* currPtr = nullptr;

            if (!m_Root) {
              m_Root = std::move(curr);
              currPtr = m_Root.get();
            } else {
              currPtr = curr.get();

              if (m_Side == side::left)
                m_ParentNode->left = std::move(curr);

              if (m_Side == side::right)
                m_ParentNode->right = std::move(curr);
            }

            Expr* L = BinOp->getLHS();
            Expr* R = BinOp->getRHS();

            m_ParentNode = currPtr;
            m_Side = side::left;
            TraverseStmt(L);
            m_ParentNode = currPtr;

            m_Side = side::right;
            TraverseStmt(R);
          }
          return true;
        }

        bool TraverseDeclRefExpr(clang::DeclRefExpr* DRE) {
          std::unique_ptr<IdxNode> curr = std::make_unique<IdxNode>(DRE);

          if (m_ParentNode) {
            if (m_Side == side::left)
              m_ParentNode->left = std::move(curr);

            if (m_Side == side::right)
              m_ParentNode->right = std::move(curr);
          }
          return true;
        }

      } checker;
      return checker.isInjectiveIdx(E);
    }
    /// Checks if a variable is changed before a certain usage.
    ///
    /// \param[in] ADC
    /// \param[in] idxVD VarDecl of a variable we check for the liveness.
    /// \param[in] stopE endpoint of the traversal(usage of a variable), we are
    /// only interested in what is before this Expr. \returns whether a given
    /// variable is modified after initialization before certain use.
    static bool isLiveIdx(clang::AnalysisDeclContext* ADC,
                          clang::VarDecl* idxVD, const clang::Expr* stopE) {
      class LiveIdxChecker : public RecursiveASTVisitor<LiveIdxChecker> {
        clang::AnalysisDeclContext* m_ADC;
        clang::VarDecl* m_idxVD;
        const clang::Expr* m_stopE;
        bool m_isLive = true;
        bool m_Modifying = false;

      public:
        LiveIdxChecker(clang::AnalysisDeclContext* ADC, clang::VarDecl* idxVD,
                       const clang::Expr* stopE)
            : m_ADC(ADC), m_idxVD(idxVD), m_stopE(stopE) {}

        bool isLiveE() {
          for (auto* block : *m_ADC->getCFG()) {
            for (const clang::CFGElement& Element : *block) {
              if (Element.getKind() == clang::CFGElement::Statement) {
                const clang::Stmt* S =
                    Element.castAs<clang::CFGStmt>().getStmt();
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                if (!TraverseStmt(const_cast<clang::Stmt*>(S)))
                  return m_isLive;
              }
            }
          }
          return m_isLive;
        }

        bool TraverseBinaryOperator(clang::BinaryOperator* BinOp) {
          Expr* L = BinOp->getLHS();
          Expr* R = BinOp->getRHS();

          if (BinOp->isAssignmentOp()) {
            m_Modifying = true;
            TraverseStmt(L);
            m_Modifying = false;
            TraverseStmt(R);
          }
          return true;
        }

        bool TraverseUnaryOperator(clang::UnaryOperator* UnOp) {
          const auto opCode = UnOp->getOpcode();
          Expr* E = UnOp->getSubExpr();
          if (opCode == UO_PostInc || opCode == UO_PostDec ||
              opCode == UO_PreInc || opCode == UO_PreDec) {
            m_Modifying = true;
            TraverseStmt(E);
            m_Modifying = false;
          }
          return true;
        }

        bool TraverseArraySubscriptExpr(clang::ArraySubscriptExpr* ASE) {
          m_Modifying = false;
          TraverseStmt(ASE->getIdx());
          return true;
        }

        bool TraverseDeclRefExpr(clang::DeclRefExpr* DRE) {
          if (DRE == m_stopE)
            return false;
          if (auto* VD = dyn_cast<VarDecl>(DRE->getDecl())) {
            if (m_Modifying && VD->getName() == m_idxVD->getName()) {
              m_isLive = false;
              return false;
            }
          }
          return true;
        }
      } checker(ADC, idxVD, stopE->IgnoreImplicit());

      return checker.isLiveE();
    }

    bool isInjective(const clang::Expr* E, clang::AnalysisDeclContext* ADC) {
      class InjectiveChecker
          : public clang::RecursiveASTVisitor<InjectiveChecker> {
        clang::AnalysisDeclContext* m_ADC;
        const clang::Expr* m_E;

      public:
        InjectiveChecker(clang::AnalysisDeclContext* ADC, const clang::Expr* E)
            : m_ADC(ADC), m_E(E) {}

        bool isInjectiveIdx(const clang::Expr* E) {
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          return TraverseStmt(const_cast<clang::Expr*>(E));
        }

        bool TraverseBinaryOperator(clang::BinaryOperator* BinOp) {
          const auto opCode = BinOp->getOpcode();
          Expr* L = BinOp->getLHS();
          Expr* R = BinOp->getRHS();

          if (opCode == BO_Add || opCode == BO_Mul) {
            Expr::EvalResult dummy;

            bool isConstL = clad_compat::Expr_EvaluateAsConstantExpr(
                L, dummy, m_ADC->getASTContext());
            bool isConstR = clad_compat::Expr_EvaluateAsConstantExpr(
                R, dummy, m_ADC->getASTContext());

            if (isConstL && isConstR)
              return false;

            if (!isConstL && isConstR)
              return TraverseStmt(L);

            if (!isConstL && !isConstR)
              return isInjectiveE(BinOp);

            if (!isConstR)
              return TraverseStmt(R);
          }
          return false;
        }

        bool TraverseDeclRefExpr(clang::DeclRefExpr* DRE) {
          if (auto* VD = dyn_cast<VarDecl>(DRE->getDecl()))
            if (auto* init = VD->getInit())
              return isInjectiveE(init->IgnoreImpCasts()) &&
                     isLiveIdx(m_ADC, VD, m_E);
          return false;
        }

        bool TraverseIntegerLiteral(clang::IntegerLiteral* IL) { return false; }

      } checker(ADC, E);

      return checker.isInjectiveIdx(E);
    }

    bool hasUnusedReturnValue(ASTContext& C, const clang::CallExpr* CE) {
      const Expr* E = CE;
      do {
        const auto& parents = C.getParents(*E);
        assert(parents.size() == 1 && "One parent stmt is expected.");
        const Stmt* S = parents[0].get<Stmt>();
        // If the parent is a Stmt but not an Expr, then the result is not used.
        if (!S)
          return false;
        E = dyn_cast<Expr>(S);
        if (!E)
          return true;
        // If we're dealing with an implicit expr (e.g. ExprWithCleanups),
        // go up to get more information.
      } while (E->IgnoreImplicit() != E);
      return false;
    }

    /// Called in ShouldRecompute. In CUDA, to access a current thread/block id
    /// we use functions that do not change the state of any variable, since no
    /// point to store the value.
    static bool isCUDABuiltInIndex(const Expr* E) {
      const clang::Expr* B = E->IgnoreImplicit();
      if (const auto* pseudoE = llvm::dyn_cast<PseudoObjectExpr>(B)) {
        if (const auto* opaqueE =
                llvm::dyn_cast<OpaqueValueExpr>(pseudoE->getSemanticExpr(0))) {
          const Expr* innerE = opaqueE->getSourceExpr()->IgnoreImplicit();
          QualType innerT = innerE->getType();
          if (innerT.isConstQualified())
            return true;
        }
      }
      return false;
    }

    bool ShouldRecompute(const Expr* E, const ASTContext& C) {
      return !(utils::ContainsFunctionCalls(E) || E->HasSideEffects(C)) ||
             isCUDABuiltInIndex(E);
    }
  } // namespace utils
} // namespace clad
