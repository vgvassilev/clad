//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clad/Differentiator/VisitorBase.h"

#include "ConstantFolder.h"

#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/ErrorEstimator.h"
#include "clad/Differentiator/StmtClone.h"
#include "clad/Differentiator/CladUtils.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/TemplateBase.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Template.h"

#include <algorithm>
#include <numeric>

#include "clad/Differentiator/Compatibility.h"

using namespace clang;

namespace clad {
  clang::CompoundStmt* VisitorBase::MakeCompoundStmt(const Stmts& Stmts) {
    auto Stmts_ref = clad_compat::makeArrayRef(Stmts.data(), Stmts.size());
    return clad_compat::CompoundStmt_Create(m_Context, Stmts_ref /**/ CLAD_COMPAT_CLANG15_CompoundStmt_Create_ExtraParam2(FPOptionsOverride()), noLoc, noLoc);
  }

  bool VisitorBase::isUnusedResult(const Expr* E) {
    const Expr* ignoreExpr;
    SourceLocation ignoreLoc;
    SourceRange ignoreRange;
    return E->isUnusedResultAWarning(
        ignoreExpr, ignoreLoc, ignoreRange, ignoreRange, m_Context);
  }

  bool VisitorBase::addToBlock(Stmt* S, Stmts& block) {
    if (!S)
      return false;
    if (Expr* E = dyn_cast<Expr>(S)) {
      if (isUnusedResult(E))
        return false;
    }
    block.push_back(S);
    return true;
  }

  bool VisitorBase::addToCurrentBlock(Stmt* S) {
    return addToBlock(S, getCurrentBlock());
  }

  VarDecl* VisitorBase::BuildVarDecl(QualType Type, IdentifierInfo* Identifier,
                                     Expr* Init, bool DirectInit,
                                     TypeSourceInfo* TSI,
                                     VarDecl::InitializationStyle IS) {

    // add namespace specifier in variable declaration if needed.
    Type = utils::AddNamespaceSpecifier(m_Sema, m_Context, Type);
    auto VD =
        VarDecl::Create(m_Context, m_Sema.CurContext, m_Function->getLocation(),
                        m_Function->getLocation(), Identifier, Type, TSI,
                        SC_None);

    if (Init) {
      m_Sema.AddInitializerToDecl(VD, Init, DirectInit);
      VD->setInitStyle(IS);
    } else {
      m_Sema.ActOnUninitializedDecl(VD);
    }
    m_Sema.FinalizeDeclaration(VD);
    // Add the identifier to the scope and IdResolver
    m_Sema.PushOnScopeChains(VD, getCurrentScope(), /*AddToContext*/ false);
    return VD;
  }

  void VisitorBase::updateReferencesOf(Stmt* InSubtree) {
    utils::ReferencesUpdater up(m_Sema, getCurrentScope(), m_Function);
    up.TraverseStmt(InSubtree);
  }

  VarDecl* VisitorBase::BuildVarDecl(QualType Type, llvm::StringRef prefix,
                                     Expr* Init, bool DirectInit,
                                     TypeSourceInfo* TSI,
                                     VarDecl::InitializationStyle IS) {
    return BuildVarDecl(Type, CreateUniqueIdentifier(prefix), Init, DirectInit,
                        TSI, IS);
  }

  NamespaceDecl* VisitorBase::BuildNamespaceDecl(IdentifierInfo* II,
                                                 bool isInline) {
    // Check if the namespace is being redeclared.
    NamespaceDecl* PrevNS = nullptr;
    // From Sema::ActOnStartNamespaceDef:
    if (II) {
      LookupResult R(m_Sema,
                     II,
                     noLoc,
                     Sema::LookupOrdinaryName,
                     clad_compat::Sema_ForVisibleRedeclaration);
      m_Sema.LookupQualifiedName(R, m_Sema.CurContext->getRedeclContext());
      NamedDecl* FoundDecl =
          R.isSingleResult() ? R.getRepresentativeDecl() : nullptr;
      PrevNS = dyn_cast_or_null<NamespaceDecl>(FoundDecl);
    } else {
      // Is anonymous namespace.
      DeclContext* Parent = m_Sema.CurContext->getRedeclContext();
      if (TranslationUnitDecl* TU = dyn_cast<TranslationUnitDecl>(Parent)) {
        PrevNS = TU->getAnonymousNamespace();
      } else {
        NamespaceDecl* ND = cast<NamespaceDecl>(Parent);
        PrevNS = ND->getAnonymousNamespace();
      }
    }
    NamespaceDecl* NDecl = clad_compat::NamespaceDecl_Create(
        m_Context, m_Sema.CurContext, isInline, noLoc, noLoc, II, PrevNS);
    if (II)
      m_Sema.PushOnScopeChains(NDecl, m_CurScope);
    else {
      // Link the anonymous namespace into its parent.
      // From Sema::ActOnStartNamespaceDef:
      DeclContext* Parent = m_Sema.CurContext->getRedeclContext();
      if (TranslationUnitDecl* TU = dyn_cast<TranslationUnitDecl>(Parent)) {
        TU->setAnonymousNamespace(NDecl);
      } else {
        cast<NamespaceDecl>(Parent)->setAnonymousNamespace(NDecl);
      }
      m_Sema.CurContext->addDecl(NDecl);
      if (!PrevNS) {
        UsingDirectiveDecl* UD =
            UsingDirectiveDecl::Create(m_Context,
                                       Parent,
                                       noLoc,
                                       noLoc,
                                       NestedNameSpecifierLoc(),
                                       noLoc,
                                       NDecl,
                                       Parent);
        UD->setImplicit();
        Parent->addDecl(UD);
      }
    }
    // Namespace scope and declcontext. Must be exited by the user.
    beginScope(Scope::DeclScope);
    m_Sema.PushDeclContext(m_CurScope, NDecl);
    return NDecl;
  }

  NamespaceDecl* VisitorBase::RebuildEnclosingNamespaces(DeclContext* DC) {
    if (NamespaceDecl* ND = dyn_cast_or_null<NamespaceDecl>(DC)) {
      NamespaceDecl* Head = RebuildEnclosingNamespaces(ND->getDeclContext());
      NamespaceDecl* NewD =
          BuildNamespaceDecl(ND->getIdentifier(), ND->isInline());
      return Head ? Head : NewD;
    } else {
      m_Sema.CurContext = DC;
      return nullptr;
    }
  }

  DeclStmt* VisitorBase::BuildDeclStmt(Decl* D) {
    Stmt* DS =
        m_Sema.ActOnDeclStmt(m_Sema.ConvertDeclToDeclGroup(D), noLoc, noLoc)
            .get();
    return cast<DeclStmt>(DS);
  }

  DeclStmt* VisitorBase::BuildDeclStmt(llvm::MutableArrayRef<Decl*> Decls) {
    auto DGR = DeclGroupRef::Create(m_Context, Decls.data(), Decls.size());
    return new (m_Context) DeclStmt(DGR, noLoc, noLoc);
  }

  DeclRefExpr* VisitorBase::BuildDeclRef(DeclaratorDecl* D,
                                         const CXXScopeSpec* SS /*=nullptr*/) {
    QualType T = D->getType();
    T = T.getNonReferenceType();
    return cast<DeclRefExpr>(clad_compat::GetResult<Expr*>(
        m_Sema.BuildDeclRefExpr(D, T, VK_LValue, noLoc, SS)));
  }

  IdentifierInfo*
  VisitorBase::CreateUniqueIdentifier(llvm::StringRef nameBase) {
    // For intermediate variables, use numbered names (_t0), for everything
    // else first try a name without number (e.g. first try to use _d_x and
    // use _d_x0 only if _d_x is taken).
    bool countedName = nameBase.startswith("_") &&
                       !nameBase.startswith("_d_") &&
                       !nameBase.startswith("_delta_");
    std::size_t idx = 0;
    std::size_t& id = countedName ? m_idCtr[nameBase.str()] : idx;
    std::string idStr = countedName ? std::to_string(id) : "";
    if (countedName)
      id += 1;
    for (;;) {
      IdentifierInfo* name = &m_Context.Idents.get(nameBase.str() + idStr);
      LookupResult R(
          m_Sema, DeclarationName(name), noLoc, Sema::LookupOrdinaryName);
      m_Sema.LookupName(R, m_CurScope, /*AllowBuiltinCreation*/ false);
      if (R.empty()) {
        return name;
      } else {
        idStr = std::to_string(id);
        id += 1;
      }
    }
  }

  Expr* VisitorBase::BuildParens(Expr* E) {
    if (!E)
      return nullptr;
    Expr* ENoCasts = E->IgnoreCasts();
    // In our case, there is no reason to build parentheses around something
    // that is not a binary or ternary operator.
    if (isa<BinaryOperator>(ENoCasts) ||
        (isa<CXXOperatorCallExpr>(ENoCasts) &&
         cast<CXXOperatorCallExpr>(ENoCasts)->getNumArgs() == 2) ||
        isa<ConditionalOperator>(ENoCasts) ||
        isa<CXXBindTemporaryExpr>(ENoCasts))
      return m_Sema.ActOnParenExpr(noLoc, noLoc, E).get();
    else
      return E;
  }

  Expr* VisitorBase::StoreAndRef(Expr* E, llvm::StringRef prefix,
                                 bool forceDeclCreation,
                                 VarDecl::InitializationStyle IS) {
    return StoreAndRef(E, getCurrentBlock(), prefix, forceDeclCreation, IS);
  }
  Expr* VisitorBase::StoreAndRef(Expr* E, Stmts& block, llvm::StringRef prefix,
                                 bool forceDeclCreation,
                                 VarDecl::InitializationStyle IS) {
    assert(E && "cannot infer type from null expression");
    QualType Type = E->getType();
    if (E->isModifiableLvalue(m_Context) == Expr::MLV_Valid)
      Type = m_Context.getLValueReferenceType(Type);
    return StoreAndRef(E, Type, block, prefix, forceDeclCreation, IS);
  }

  /// For an expr E, decides if it is useful to store it in a temporary variable
  /// and replace E's further usage by a reference to that variable to avoid
  /// recomputiation.
  static bool UsefulToStore(Expr* E) {
    if (!E)
      return false;
    Expr* B = E->IgnoreParenImpCasts();
    // FIXME: find a more general way to determine that or add more options.
    if (isa<DeclRefExpr>(B) || isa<FloatingLiteral>(B) ||
        isa<IntegerLiteral>(B))
      return false;
    if (isa<UnaryOperator>(B)) {
      auto UO = cast<UnaryOperator>(B);
      auto OpKind = UO->getOpcode();
      if (OpKind == UO_Plus || OpKind == UO_Minus)
        return UsefulToStore(UO->getSubExpr());
      return false;
    }
    if (isa<ArraySubscriptExpr>(B)) {
      auto ASE = cast<ArraySubscriptExpr>(B);
      return UsefulToStore(ASE->getBase()) || UsefulToStore(ASE->getIdx());
    }
    return true;
  }

  Expr* VisitorBase::StoreAndRef(Expr* E, QualType Type, Stmts& block,
                                 llvm::StringRef prefix, bool forceDeclCreation,
                                 VarDecl::InitializationStyle IS) {
    if (!forceDeclCreation) {
      // If Expr is simple (i.e. a reference or a literal), there is no point
      // in storing it as there is no evaluation going on.
      if (!UsefulToStore(E))
        return E;
    }
    // Create variable declaration.
    VarDecl* Var = BuildVarDecl(Type, CreateUniqueIdentifier(prefix), E,
                                /*DirectInit=*/false,
                                /*TSI=*/nullptr, IS);

    // Add the declaration to the body of the gradient function.
    addToBlock(BuildDeclStmt(Var), block);

    // Return reference to the declaration instead of original expression.
    return BuildDeclRef(Var);
  }

  Stmt* VisitorBase::Clone(const Stmt* S) {
    Stmt* clonedStmt = m_Builder.m_NodeCloner->Clone(S);
    updateReferencesOf(clonedStmt);
    return clonedStmt;
  }
  Expr* VisitorBase::Clone(const Expr* E) {
    const Stmt* S = E;
    return llvm::cast<Expr>(Clone(S));
  }

  Expr* VisitorBase::BuildOp(UnaryOperatorKind OpCode, Expr* E,
                             SourceLocation OpLoc) {
    if (!E)
      return nullptr;
    return m_Sema.BuildUnaryOp(nullptr, OpLoc, OpCode, E).get();
  }

  Expr* VisitorBase::BuildOp(clang::BinaryOperatorKind OpCode, Expr* L, Expr* R,
                             SourceLocation OpLoc) {
    if (!L || !R)
      return nullptr;
    return m_Sema.BuildBinOp(nullptr, OpLoc, OpCode, L, R).get();
  }

  Expr* VisitorBase::getZeroInit(QualType T) {
    if (T->isScalarType())
      return ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
    else
      return m_Sema.ActOnInitList(noLoc, {}, noLoc).get();
  }

  std::pair<const clang::Expr*, llvm::SmallVector<const clang::Expr*, 4>>
  VisitorBase::SplitArraySubscript(const Expr* ASE) {
    llvm::SmallVector<const clang::Expr*, 4> Indices{};
    const Expr* E = ASE->IgnoreParenImpCasts();
    while (auto S = dyn_cast<ArraySubscriptExpr>(E)) {
      Indices.push_back(S->getIdx());
      E = S->getBase()->IgnoreParenImpCasts();
    }
    std::reverse(std::begin(Indices), std::end(Indices));
    return std::make_pair(E, std::move(Indices));
  }

  Expr* VisitorBase::BuildArraySubscript(
      Expr* Base, const llvm::SmallVectorImpl<clang::Expr*>& Indices) {
    Expr* result = Base;
    for (Expr* I : Indices)
      result =
          m_Sema.CreateBuiltinArraySubscriptExpr(result, noLoc, I, noLoc).get();
    return result;
  }

  NamespaceDecl* VisitorBase::GetCladNamespace() {
    static NamespaceDecl* Result = nullptr;
    if (Result)
      return Result;
    DeclarationName CladName = &m_Context.Idents.get("clad");
    LookupResult CladR(m_Sema,
                       CladName,
                       noLoc,
                       Sema::LookupNamespaceName,
                       clad_compat::Sema_ForVisibleRedeclaration);
    m_Sema.LookupQualifiedName(CladR, m_Context.getTranslationUnitDecl());
    assert(!CladR.empty() && "cannot find clad namespace");
    Result = cast<NamespaceDecl>(CladR.getFoundDecl());
    return Result;
  }

  TemplateDecl*
  VisitorBase::LookupTemplateDeclInCladNamespace(llvm::StringRef ClassName) {
    NamespaceDecl* CladNS = GetCladNamespace();
    CXXScopeSpec CSS;
    CSS.Extend(m_Context, CladNS, noLoc, noLoc);
    DeclarationName TapeName = &m_Context.Idents.get(ClassName);
    LookupResult TapeR(m_Sema,
                       TapeName,
                       noLoc,
                       Sema::LookupUsingDeclName,
                       clad_compat::Sema_ForVisibleRedeclaration);
    m_Sema.LookupQualifiedName(TapeR, CladNS, CSS);
    assert(!TapeR.empty() && isa<TemplateDecl>(TapeR.getFoundDecl()) &&
           "cannot find clad::tape");
    return cast<TemplateDecl>(TapeR.getFoundDecl());
  }

  QualType VisitorBase::InstantiateTemplate(TemplateDecl* CladClassDecl,
                                            TemplateArgumentListInfo& TLI) {
    // This will instantiate tape<T> type and return it.
    QualType TT =
        m_Sema.CheckTemplateIdType(TemplateName(CladClassDecl), noLoc, TLI);
    // Get clad namespace and its identifier clad::.
    CXXScopeSpec CSS;
    CSS.Extend(m_Context, GetCladNamespace(), noLoc, noLoc);
    NestedNameSpecifier* NS = CSS.getScopeRep();

    // Create elaborated type with namespace specifier,
    // i.e. class<T> -> clad::class<T>
    return m_Context.getElaboratedType(ETK_None, NS, TT);
  }

  QualType VisitorBase::InstantiateTemplate(TemplateDecl* CladClassDecl,
                                            ArrayRef<QualType> TemplateArgs) {
    // Create a list of template arguments.
    TemplateArgumentListInfo TLI{};
    for (auto T : TemplateArgs) {
      TemplateArgument TA = T;
      TLI.addArgument(
          TemplateArgumentLoc(TA, m_Context.getTrivialTypeSourceInfo(T)));
    }

    return VisitorBase::InstantiateTemplate(CladClassDecl, TLI);
  }

  TemplateDecl* VisitorBase::GetCladTapeDecl() {
    static TemplateDecl* Result = nullptr;
    if (!Result)
      Result = LookupTemplateDeclInCladNamespace(/*ClassName=*/"tape");
    return Result;
  }

  LookupResult VisitorBase::LookupCladTapeMethod(llvm::StringRef name) {
    NamespaceDecl* CladNS = GetCladNamespace();
    CXXScopeSpec CSS;
    CSS.Extend(m_Context, CladNS, noLoc, noLoc);
    DeclarationName Name = &m_Context.Idents.get(name);
    LookupResult R(m_Sema, Name, noLoc, Sema::LookupOrdinaryName);
    m_Sema.LookupQualifiedName(R, CladNS, CSS);
    assert(!R.empty() && isa<FunctionTemplateDecl>(R.getRepresentativeDecl()) &&
           "cannot find requested name");
    return R;
  }

  LookupResult& VisitorBase::GetCladTapePush() {
    static clad_compat::llvm_Optional<LookupResult> Result{};
    if (!Result)
      Result = LookupCladTapeMethod("push");
    return clad_compat::llvm_Optional_GetValue(Result);
  }

  DeclRefExpr* VisitorBase::GetCladTapePushDRE() {
    LookupResult& pushLR = GetCladTapePush();
    CXXScopeSpec CSS;
    CSS.Extend(m_Context, GetCladNamespace(), noLoc, noLoc);
    DeclRefExpr* pushDRE = m_Sema.BuildDeclarationNameExpr(CSS, pushLR, false)
                               .getAs<DeclRefExpr>();
    return pushDRE;
  }

  LookupResult& VisitorBase::GetCladTapePop() {
    static clad_compat::llvm_Optional<LookupResult> Result{};
    if (!Result)
      Result = LookupCladTapeMethod("pop");
    return clad_compat::llvm_Optional_GetValue(Result);
  }

  LookupResult& VisitorBase::GetCladTapeBack() {
    static clad_compat::llvm_Optional<LookupResult> Result{};
    if (!Result)
      Result = LookupCladTapeMethod("back");
    return clad_compat::llvm_Optional_GetValue(Result);
  }

  QualType VisitorBase::GetCladTapeOfType(QualType T) {
    return InstantiateTemplate(GetCladTapeDecl(), {T});
  }

  Expr* VisitorBase::BuildCallExprToMemFn(Expr* Base,
                                          StringRef MemberFunctionName,
                                          MutableArrayRef<Expr*> ArgExprs, ValueDecl* memberDecl) {
    UnqualifiedId Member;
    Member.setIdentifier(&m_Context.Idents.get(MemberFunctionName), noLoc);
    CXXScopeSpec SS;
    bool isArrow = Base->getType()->isPointerType();
    auto ME = m_Sema
                  .ActOnMemberAccessExpr(getCurrentScope(), Base, noLoc,
                                         isArrow ? tok::TokenKind::arrow
                                                 : tok::TokenKind::period,
                                         SS, noLoc, Member,
                                         /*ObjCImpDecl=*/nullptr)
                  .getAs<MemberExpr>();
    // FIXME: This is a workaround and it's dependency should be removed soon.
    // Currently, member function derivatives are not being registered properly.
    // If there are member function derivatives overloads then only one of the
    // overload is being found by Sema lookup.
    // Ideally, if `MemberFunctionName` corresponds to an overloaded member
    // function name, then `Sema::ActOnMemberAccessExpr` should return an
    // unresolved expression and `Sema::ActOnCallExpr` should automatically
    // resolve this unresolved expression on the basis of the call arguments
    // provided. But currently, since only one of the member function overload
    // is found by the lookup, unresolved expression is not created and thus, we
    // explicitly should assign the correct member function whenever we can.
    if (memberDecl)
      ME->setMemberDecl(memberDecl);
    return m_Sema.ActOnCallExpr(getCurrentScope(), ME, noLoc, ArgExprs, noLoc)
        .get();
  }

  static QualType getRefQualifiedThisType(Sema& semaRef, CXXMethodDecl* MD) {
    ASTContext& C = semaRef.getASTContext();
    CXXRecordDecl* RD = MD->getParent();
    auto RDType = RD->getTypeForDecl();
    auto thisObjectQType = C.getQualifiedType(
        RDType, clad_compat::CXXMethodDecl_getMethodQualifiers(MD));
    if (MD->getRefQualifier() == RefQualifierKind::RQ_RValue)
      thisObjectQType = C.getRValueReferenceType(thisObjectQType);
    else if (MD->getRefQualifier() == RefQualifierKind::RQ_LValue)
      thisObjectQType = C.getLValueReferenceType(thisObjectQType);
    return thisObjectQType;
  }

  Expr* VisitorBase::BuildCallExprToMemFn(
      clang::CXXMethodDecl* FD, llvm::MutableArrayRef<clang::Expr*> argExprs,
      bool useRefQualifiedThisObj) {
    Expr* thisExpr = clad_compat::Sema_BuildCXXThisExpr(m_Sema, FD);
    bool isArrow = true;

    // C++ does not support perfect forwarding of `*this` object inside
    // a member function.
    // Cast `*this` to an rvalue if the current function has rvalue qualifier so
    // that correct method overload is resolved. We do not need to cast to
    // an lvalue because without any cast `*this` will always be considered an
    // lvalue.
    if (useRefQualifiedThisObj &&
        FD->getRefQualifier() != RefQualifierKind::RQ_None) {
      auto thisQType = getRefQualifiedThisType(m_Sema, FD);
      // Build `static_cast<ReferenceQualifiedThisObjectType>(*this)`
      // expression.
      thisExpr = m_Sema
                     .BuildCXXNamedCast(noLoc, tok::TokenKind::kw_static_cast,
                                        m_Context.getTrivialTypeSourceInfo(
                                            thisQType),
                                        BuildOp(UnaryOperatorKind::UO_Deref,
                                                thisExpr),
                                        noLoc, noLoc)
                     .get();
      isArrow = false;
    }
    NestedNameSpecifierLoc NNS(FD->getQualifier(),
                               /*Data=*/nullptr);
    auto DAP = DeclAccessPair::make(FD, FD->getAccess());
    auto memberExpr = MemberExpr::
        Create(m_Context, thisExpr, isArrow, noLoc, NNS, noLoc, FD, DAP,
               FD->getNameInfo(),
               /*TemplateArgs=*/nullptr, m_Context.BoundMemberTy,
               CLAD_COMPAT_ExprValueKind_R_or_PR_Value,
               ExprObjectKind::OK_Ordinary
                   CLAD_COMPAT_CLANG9_MemberExpr_ExtraParams(NOUR_None));
    return m_Sema
        .BuildCallToMemberFunction(getCurrentScope(), memberExpr, noLoc,
                                   argExprs, noLoc)
        .get();
  }

  Expr*
  VisitorBase::BuildCallExprToFunction(FunctionDecl* FD,
                                       llvm::MutableArrayRef<Expr*> argExprs,
                                       bool useRefQualifiedThisObj /*=false*/,
                                       const CXXScopeSpec* SS /*=nullptr*/) {
    Expr* call = nullptr;
    if (auto derMethod = dyn_cast<CXXMethodDecl>(FD)) {
      call = BuildCallExprToMemFn(derMethod, argExprs, useRefQualifiedThisObj);
    } else {
      Expr* exprFunc = BuildDeclRef(FD, SS);
      call = m_Sema
                 .ActOnCallExpr(
                     getCurrentScope(),
                     /*Fn=*/exprFunc,
                     /*LParenLoc=*/noLoc,
                     /*ArgExprs=*/llvm::MutableArrayRef<Expr*>(argExprs),
                     /*RParenLoc=*/m_Function->getLocation())
                 .get();
    }
    return call;
  }

  Expr* VisitorBase::BuildCallExprToCladFunction(
      llvm::StringRef name, llvm::MutableArrayRef<clang::Expr*> argExprs,
      llvm::ArrayRef<clang::TemplateArgument> templateArgs,
      SourceLocation loc) {
    DeclarationName declName = &m_Context.Idents.get(name);
    clang::LookupResult R(m_Sema, declName, noLoc, Sema::LookupOrdinaryName);

    // Find function declaration
    NamespaceDecl* CladNS = GetCladNamespace();
    CXXScopeSpec CSS;
    CSS.Extend(m_Context, CladNS, loc, loc);
    m_Sema.LookupQualifiedName(R, CladNS, CSS);

    // Build the template specialization expression.
    // FIXME: currently this doesn't print func<templates>(args...) while
    // dumping and only prints func(args...), we need to fix this.
    auto* FTD = dyn_cast<FunctionTemplateDecl>(R.getRepresentativeDecl());
    clang::TemplateArgumentList TL(TemplateArgumentList::OnStack, templateArgs);
    FunctionDecl* FD = m_Sema.InstantiateFunctionDeclaration(FTD, &TL, loc);

    return BuildCallExprToFunction(FD, argExprs, false, &CSS);
  }

  TemplateDecl* VisitorBase::GetCladArrayRefDecl() {
    static TemplateDecl* Result = nullptr;
    if (!Result)
      Result = LookupTemplateDeclInCladNamespace(/*ClassName=*/"array_ref");
    return Result;
  }

  QualType VisitorBase::GetCladArrayRefOfType(clang::QualType T) {
    return InstantiateTemplate(GetCladArrayRefDecl(), {T});
  }

  TemplateDecl* VisitorBase::GetCladArrayDecl() {
    static TemplateDecl* Result = nullptr;
    if (!Result)
      Result = LookupTemplateDeclInCladNamespace(/*ClassName=*/"array");
    return Result;
  }

  QualType VisitorBase::GetCladArrayOfType(clang::QualType T) {
    return InstantiateTemplate(GetCladArrayDecl(), {T});
  }

  TemplateDecl* VisitorBase::GetCladMatrixDecl() {
    static TemplateDecl* Result = nullptr;
    if (!Result)
      Result = LookupTemplateDeclInCladNamespace(/*ClassName=*/"matrix");
    return Result;
  }

  QualType VisitorBase::GetCladMatrixOfType(clang::QualType T) {
    return InstantiateTemplate(GetCladMatrixDecl(), {T});
  }

  Expr* VisitorBase::BuildIdentityMatrixExpr(clang::QualType T,
                                             MutableArrayRef<Expr*> Args,
                                             clang::SourceLocation Loc) {
    return BuildCallExprToCladFunction(/*name=*/"identity_matrix", Args, {T},
                                       Loc);
  }

  Expr* VisitorBase::BuildArrayRefSizeExpr(Expr* Base) {
    return BuildCallExprToMemFn(Base, /*MemberFunctionName=*/"size", {});
  }

  Expr* VisitorBase::BuildArrayRefSliceExpr(Expr* Base,
                                            MutableArrayRef<Expr*> Args) {
    return BuildCallExprToMemFn(Base, /*MemberFunctionName=*/"slice", Args);
  }

  bool VisitorBase::isCladArrayType(QualType QT) {
    // FIXME: Replace this check with a clang decl check
    return QT.getAsString().find("clad::array") != std::string::npos ||
           QT.getAsString().find("clad::array_ref") != std::string::npos;
  }

  bool VisitorBase::isCladValueAndPushforwardType(clang::QualType QT) {
    // FIXME: Replace this check with a clang decl check
    return QT.getAsString().find("ValueAndPushforward") != std::string::npos;
  }

  Expr* VisitorBase::GetSingleArgCentralDiffCall(
      Expr* targetFuncCall, Expr* targetArg, unsigned targetPos,
      unsigned numArgs, llvm::SmallVectorImpl<Expr*>& args) {
    QualType argType = targetArg->getType();
    int printErrorInf = m_Builder.shouldPrintNumDiffErrs();
    bool isSupported = argType->isArithmeticType();
    if (!isSupported)
      return nullptr;
    // Build function args.
    llvm::SmallVector<Expr*, 16U> NumDiffArgs;
    NumDiffArgs.push_back(targetFuncCall);
    NumDiffArgs.push_back(targetArg);
    NumDiffArgs.push_back(ConstantFolder::synthesizeLiteral(m_Context.IntTy,
                                                            m_Context,
                                                            targetPos));
    NumDiffArgs.push_back(ConstantFolder::synthesizeLiteral(m_Context.IntTy,
                                                            m_Context,
                                                            printErrorInf));
    NumDiffArgs.insert(NumDiffArgs.end(), args.begin(), args.begin() + numArgs);
    // Return the found overload.
    std::string Name = "forward_central_difference";
    return m_Builder.BuildCallToCustomDerivativeOrNumericalDiff(
        Name, NumDiffArgs, getCurrentScope(), /*OriginalFnDC=*/nullptr,
        /*forCustomDerv=*/false,
        /*namespaceShouldExist=*/false);
  }

  Expr* VisitorBase::GetMultiArgCentralDiffCall(
      Expr* targetFuncCall, QualType retType, unsigned numArgs,
      llvm::SmallVectorImpl<Stmt*>& NumericalDiffMultiArg,
      llvm::SmallVectorImpl<Expr*>& args,
      llvm::SmallVectorImpl<Expr*>& outputArgs) {
    int printErrorInf = m_Builder.shouldPrintNumDiffErrs();
    llvm::SmallVector<Expr*, 16U> NumDiffArgs = {};
    NumDiffArgs.push_back(targetFuncCall);
    // build the clad::tape<clad::array_ref>> = {};
    QualType RefType = GetCladArrayRefOfType(retType);
    QualType TapeType = GetCladTapeOfType(RefType);
    auto VD = BuildVarDecl(
        TapeType, "_t", getZeroInit(TapeType), /*DirectInit=*/false,
        /*TSI=*/nullptr, VarDecl::InitializationStyle::CInit);
    NumericalDiffMultiArg.push_back(BuildDeclStmt(VD));
    Expr* TapeRef = BuildDeclRef(VD);
    NumDiffArgs.push_back(TapeRef);
    NumDiffArgs.push_back(ConstantFolder::synthesizeLiteral(m_Context.IntTy,
                                                            m_Context,
                                                            printErrorInf));

    // Build the tape push expressions.
    VD->setLocation(m_Function->getLocation());
    m_Sema.AddInitializerToDecl(VD, getZeroInit(TapeType), false);
    CXXScopeSpec CSS;
    CSS.Extend(m_Context, GetCladNamespace(), noLoc, noLoc);
    LookupResult& Push = GetCladTapePush();
    auto PushDRE = m_Sema.BuildDeclarationNameExpr(CSS, Push, /*ADL*/ false)
                       .get();
    Expr* PushExpr;
    for (unsigned i = 0, e = numArgs; i < e; i++) {
      Expr* callArgs[] = {TapeRef, outputArgs[i]};
      PushExpr = m_Sema
                     .ActOnCallExpr(getCurrentScope(), PushDRE, noLoc, callArgs,
                                    noLoc)
                     .get();
      NumericalDiffMultiArg.push_back(PushExpr);
      NumDiffArgs.push_back(args[i]);
    }
    std::string Name = "central_difference";
    return m_Builder.BuildCallToCustomDerivativeOrNumericalDiff(
        Name, NumDiffArgs, getCurrentScope(), /*OriginalFnDC=*/nullptr,
        /*forCustomDerv=*/false,
        /*namespaceShouldExist=*/false);
  }

  void VisitorBase::CallExprDiffDiagnostics(llvm::StringRef funcName,
                                 SourceLocation srcLoc, bool isDerived){
    if (!isDerived) {
      // Function was not derived => issue a warning.
      diag(DiagnosticsEngine::Warning,
           srcLoc,
           "function '%0' was not differentiated because clad failed to "
           "differentiate it and no suitable overload was found in "
           "namespace 'custom_derivatives', and function may not be "
            "eligible for numerical differentiation.",
           {funcName});
    } else {
      diag(DiagnosticsEngine::Warning, noLoc,
           "Falling back to numerical differentiation for '%0' since no "
           "suitable overload was found and clad could not derive it. "
           "To disable this feature, compile your programs with "
           "-DCLAD_NO_NUM_DIFF.",
           {funcName});
    }
  }

  ParmVarDecl* VisitorBase::CloneParmVarDecl(const ParmVarDecl* PVD,
                                             IdentifierInfo* II,
                                             bool pushOnScopeChains,
                                             bool cloneDefaultArg) {
    Expr* newPVDDefaultArg = nullptr;
    if (PVD->hasDefaultArg() && cloneDefaultArg) {
      newPVDDefaultArg = Clone(PVD->getDefaultArg());
    }
    auto newPVD = ParmVarDecl::Create(
        m_Context, m_Sema.CurContext, noLoc, noLoc, II, PVD->getType(),
        PVD->getTypeSourceInfo(), PVD->getStorageClass(), newPVDDefaultArg);
    if (pushOnScopeChains && newPVD->getIdentifier()) {
      m_Sema.PushOnScopeChains(newPVD, getCurrentScope(),
                               /*AddToContext=*/false);
    }
    return newPVD;
  }

  QualType VisitorBase::DetermineCladArrayValueType(clang::QualType T) {
    assert(isCladArrayType(T) && "Not a clad::array or clad::array_ref type");
    auto specialization =
        cast<ClassTemplateSpecializationDecl>(T->getAsCXXRecordDecl());
    auto& TAL = specialization->getTemplateArgs();
    return TAL.get(0).getAsType();
  }
} // end namespace clad
