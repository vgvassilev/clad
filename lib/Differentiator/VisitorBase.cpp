//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clad/Differentiator/VisitorBase.h"

#include "ConstantFolder.h"

#include "llvm/Support/Casting.h"
#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/ErrorEstimator.h"
#include "clad/Differentiator/MultiplexExternalRMVSource.h"
#include "clad/Differentiator/Sins.h"
#include "clad/Differentiator/StmtClone.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/TemplateBase.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Template.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include <algorithm>
#include <numeric>

#include "clad/Differentiator/Compatibility.h"

using namespace clang;

namespace clad {
  clang::CompoundStmt* VisitorBase::MakeCompoundStmt(const Stmts& Stmts) {
    auto Stmts_ref = clad_compat::makeArrayRef(Stmts.data(), Stmts.size());
    return clad_compat::CompoundStmt_Create(
        m_Context,
        Stmts_ref /**/ CLAD_COMPAT_CLANG15_CompoundStmt_Create_ExtraParam2(
            FPOptionsOverride()),
        utils::GetValidSLoc(m_Sema), utils::GetValidSLoc(m_Sema));
  }

  bool VisitorBase::isUnusedResult(const Expr* E) {
    const Expr* ignoreExpr;
    SourceLocation ignoreLoc;
    SourceRange ignoreRange;
    return E->isUnusedResultAWarning(
        ignoreExpr, ignoreLoc, ignoreRange, ignoreRange, m_Context);
  }

  bool VisitorBase::addToCurrentBlock(Stmt* S) {
    return addToBlock(S, getCurrentBlock());
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

  ALLOW_ACCESS(Sema, CurScope, Scope*);

  clang::Scope*& VisitorBase::getCurrentScope() {
    return ACCESS(m_Sema, CurScope);
  }

  void VisitorBase::setCurrentScope(clang::Scope* S) {
    getCurrentScope() = S;
    assert(getEnclosingNamespaceOrTUScope() && "Lost path to base.");
  }

  clang::Scope* VisitorBase::getEnclosingNamespaceOrTUScope() {
    auto isNamespaceOrTUScope = [](const clang::Scope* S) {
      if (clang::DeclContext* DC = S->getEntity())
        return DC->isFileContext();
      return false;
    };
    clang::Scope* InnermostFileScope = getCurrentScope();
    while (InnermostFileScope && !isNamespaceOrTUScope(InnermostFileScope))
      InnermostFileScope = InnermostFileScope->getParent();

    return InnermostFileScope;
  }

  void VisitorBase::beginScope(unsigned ScopeFlags) {
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    auto* S = new clang::Scope(getCurrentScope(), ScopeFlags, m_Sema.Diags);
    setCurrentScope(S);
  }

  void VisitorBase::endScope() {
    // This will remove all the decls in the scope from the IdResolver.
    m_Sema.ActOnPopScope(noLoc, getCurrentScope());
    clang::Scope* oldScope = getCurrentScope();
    setCurrentScope(oldScope->getParent());
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    delete oldScope;
  }

  void VisitorBase::SetDeclInit(VarDecl* VD, Expr* Init, bool DirectInit) {
    if (!Init) {
      // Clang sets inits only once. Therefore, ActOnUninitializedDecl does
      // not reset the init and we have to do it manually.
      VD->setInit(nullptr);
      m_Sema.ActOnUninitializedDecl(VD);
      return;
    }

    // Clang sets inits only once. Therefore, AddInitializerToDecl does
    // not reset the declaration style to default and we have to do it manually.
    VarDecl::InitializationStyle defaultStyle{};
    VD->setInitStyle(defaultStyle);

    // Clang expects direct inits to be wrapped either in InitListExpr or
    // ParenListExpr.
    if (DirectInit && !isa<InitListExpr>(Init) && !isa<ParenListExpr>(Init))
      Init = m_Sema.ActOnParenListExpr(noLoc, noLoc, Init).get();
    m_Sema.AddInitializerToDecl(VD, Init, DirectInit);
  }

  VarDecl* VisitorBase::BuildVarDecl(QualType Type, IdentifierInfo* Identifier,
                                     Expr* Init, bool DirectInit,
                                     TypeSourceInfo* TSI) {
    return BuildVarDecl(Type, Identifier, getCurrentScope(), Init, DirectInit,
                        TSI);
  }
  VarDecl* VisitorBase::BuildVarDecl(QualType Type, IdentifierInfo* Identifier,
                                     Scope* Scope, Expr* Init, bool DirectInit,
                                     TypeSourceInfo* TSI) {
    // add namespace specifier in variable declaration if needed.
    Type = utils::AddNamespaceSpecifier(m_Sema, m_Context, Type);
    auto* VD = VarDecl::Create(
        m_Context, m_Sema.CurContext, m_DiffReq->getLocation(),
        m_DiffReq->getLocation(), Identifier, Type, TSI, SC_None);

    SetDeclInit(VD, Init, DirectInit);
    m_Sema.FinalizeDeclaration(VD);
    // Add the identifier to the scope and IdResolver
    m_Sema.PushOnScopeChains(VD, Scope, /*AddToContext*/ false);
    return VD;
  }

  void VisitorBase::updateReferencesOf(Stmt* InSubtree) {
    utils::ReferencesUpdater up(m_Sema, getCurrentScope(), m_DiffReq.Function,
                                m_DeclReplacements);
    up.TraverseStmt(InSubtree);
  }

  VarDecl* VisitorBase::BuildVarDecl(QualType Type, llvm::StringRef prefix,
                                     Expr* Init, bool DirectInit,
                                     TypeSourceInfo* TSI) {
    return BuildVarDecl(Type, CreateUniqueIdentifier(prefix), Init, DirectInit,
                        TSI);
  }

  VarDecl* VisitorBase::BuildGlobalVarDecl(QualType Type,
                                           llvm::StringRef prefix, Expr* Init,
                                           bool DirectInit,
                                           TypeSourceInfo* TSI) {
    return BuildVarDecl(Type, CreateUniqueIdentifier(prefix),
                        m_DerivativeFnScope, Init, DirectInit, TSI);
  }

  NamespaceDecl* VisitorBase::BuildNamespaceDecl(IdentifierInfo* II,
                                                 bool isInline) {
    // Check if the namespace is being redeclared.
    NamespaceDecl* PrevNS = nullptr;
    // From Sema::ActOnStartNamespaceDef:
    if (II) {
      LookupResult R(m_Sema, II, noLoc, Sema::LookupOrdinaryName,
                     CLAD_COMPAT_Sema_ForVisibleRedeclaration);
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
      m_Sema.PushOnScopeChains(NDecl, getCurrentScope());
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
    m_Sema.PushDeclContext(getCurrentScope(), NDecl);
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
    Stmt* DS = m_Sema
                   .ActOnDeclStmt(m_Sema.ConvertDeclToDeclGroup(D),
                                  D->getBeginLoc(), D->getEndLoc())
                   .get();
    return cast<DeclStmt>(DS);
  }

  DeclStmt* VisitorBase::BuildDeclStmt(llvm::MutableArrayRef<Decl*> Decls) {
    auto DGR = DeclGroupRef::Create(m_Context, Decls.data(), Decls.size());
    return new (m_Context) DeclStmt(DGR, noLoc, noLoc);
  }

  DeclRefExpr* VisitorBase::BuildDeclRef(DeclaratorDecl* D,
                                         NestedNameSpecifier* NNS /*=nullptr*/,
                                         ExprValueKind VK /*=VK_LValue*/) {
    CXXScopeSpec CSS;
    SourceLocation fakeLoc = utils::GetValidSLoc(m_Sema);
    // FIXME: Remove once BuildDeclRef can automatically deduce class
    // namespace specifiers.
    auto* MD = dyn_cast<CXXMethodDecl>(D);
    if (!NNS && MD && MD->isStatic()) {
      const CXXRecordDecl* RD = MD->getParent();
      IdentifierInfo* II = &m_Context.Idents.get(RD->getNameAsString());
      NNS = NestedNameSpecifier::Create(m_Context, II);
    }

    if (NNS) {
      CSS.MakeTrivial(m_Context, NNS, fakeLoc);
    } else {
      // If no CXXScopeSpec is provided we should try to find the common path
      // between the current scope (in which presumably we will make the call)
      // and where `D` is.
      llvm::SmallVector<DeclContext*, 4> DCs;
      DeclContext* DeclDC = D->getDeclContext();
      // FIXME: We should respect using clauses and shorten the qualified names.
      while (!DeclDC->isTranslationUnit()) {
        // Stop when we find the common ancestor.
        if (DeclDC->Equals(m_Sema.CurContext))
          break;

        // FIXME: We should extend that for classes and class templates. See
        // clang's getFullyQualifiedNestedNameSpecifier.
        if (DeclDC->isNamespace() && !DeclDC->isInlineNamespace())
          DCs.push_back(DeclDC);

        DeclDC = DeclDC->getParent();
      }

      for (unsigned i = DCs.size(); i > 0; --i)
        CSS.Extend(m_Context, cast<NamespaceDecl>(DCs[i - 1]), fakeLoc,
                   fakeLoc);
    }
    QualType T = D->getType();
    T = T.getNonReferenceType();
    return cast<DeclRefExpr>(clad_compat::GetResult<Expr*>(
        m_Sema.BuildDeclRefExpr(D, T, VK, D->getBeginLoc(), &CSS)));
  }

  IdentifierInfo*
  VisitorBase::CreateUniqueIdentifier(llvm::StringRef nameBase) {
    // For intermediate variables, use numbered names (_t0), for everything
    // else first try a name without number (e.g. first try to use _d_x and
    // use _d_x0 only if _d_x is taken).
    bool countedName = nameBase.starts_with("_") &&
                       !nameBase.starts_with("_d_") &&
                       !nameBase.starts_with("_delta_");
    std::size_t idx = 0;
    std::size_t& id = countedName ? m_idCtr[nameBase.str()] : idx;
    std::string idStr = countedName ? std::to_string(id) : "";
    if (countedName)
      id += 1;
    for (;;) {
      IdentifierInfo* name = &m_Context.Idents.get(nameBase.str() + idStr);
      LookupResult R(
          m_Sema, DeclarationName(name), noLoc, Sema::LookupOrdinaryName);
      m_Sema.LookupName(R, getCurrentScope(), /*AllowBuiltinCreation*/ false);
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
        isa<CXXBindTemporaryExpr>(ENoCasts)) {
      return m_Sema.ActOnParenExpr(E->getBeginLoc(), E->getEndLoc(), E).get();
    }
    return E;
  }

  Expr* VisitorBase::StoreAndRef(Expr* E, llvm::StringRef prefix,
                                 bool forceDeclCreation) {
    return StoreAndRef(E, getCurrentBlock(), prefix, forceDeclCreation);
  }
  Expr* VisitorBase::StoreAndRef(Expr* E, Stmts& block, llvm::StringRef prefix,
                                 bool forceDeclCreation) {
    assert(E && "cannot infer type from null expression");
    QualType Type = E->getType();
    if (E->isModifiableLvalue(m_Context) == Expr::MLV_Valid)
      Type = m_Context.getLValueReferenceType(Type);
    return StoreAndRef(E, Type, block, prefix, forceDeclCreation);
  }

  bool VisitorBase::UsefulToStore(Expr* E) {
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
                                 llvm::StringRef prefix,
                                 bool forceDeclCreation) {
    if (!forceDeclCreation) {
      // If Expr is simple (i.e. a reference or a literal), there is no point
      // in storing it as there is no evaluation going on.
      if (!UsefulToStore(E))
        return E;
    }
    // Create variable declaration.
    VarDecl* Var = BuildVarDecl(Type, CreateUniqueIdentifier(prefix), E,
                                /*DirectInit=*/false);

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

  QualType VisitorBase::CloneType(const QualType QT) {
    auto clonedType = m_Builder.m_NodeCloner->CloneType(QT);
    utils::ReferencesUpdater up(m_Sema, getCurrentScope(), m_DiffReq.Function,
                                m_DeclReplacements);
    up.updateType(clonedType);
    return clonedType;
  }
  Expr* VisitorBase::BuildOp(UnaryOperatorKind OpCode, Expr* E,
                             SourceLocation OpLoc) {
    if (!E)
      return nullptr;
    // Debug clang requires the location to be valid
    if (!OpLoc.isValid())
      OpLoc = utils::GetValidSLoc(m_Sema);
    // Call function for UnaryMinus
    if (OpCode == UO_Minus)
      return ResolveUnaryMinus(E->IgnoreCasts(), OpLoc);
    return m_Sema.BuildUnaryOp(nullptr, OpLoc, OpCode, E).get();
  }
  Expr* VisitorBase::ResolveUnaryMinus(Expr* E, SourceLocation OpLoc) {
    if (auto* UO = llvm::dyn_cast<clang::UnaryOperator>(E)) {
      if (UO->getOpcode() == clang::UO_Minus)
        return (UO->getSubExpr())->IgnoreParens();
    }
    Expr* E_LHS = E;
    while (auto* BO = llvm::dyn_cast<BinaryOperator>(E_LHS))
      E_LHS = BO->getLHS();
    if (auto* UO = llvm::dyn_cast<clang::UnaryOperator>(E_LHS->IgnoreCasts())) {
      if (UO->getOpcode() == clang::UO_Minus)
        E = m_Sema.ActOnParenExpr(E->getBeginLoc(), E->getEndLoc(), E).get();
    }
    return m_Sema.BuildUnaryOp(nullptr, OpLoc, clang::UO_Minus, E).get();
  }

  Expr* VisitorBase::BuildOp(clang::BinaryOperatorKind OpCode, Expr* L, Expr* R,
                             SourceLocation OpLoc) {
    if (!L || !R)
      return nullptr;
    // Debug clang requires the location to be valid
    if (!OpLoc.isValid())
      OpLoc = utils::GetValidSLoc(m_Sema);
    return m_Sema.BuildBinOp(nullptr, OpLoc, OpCode, L, R).get();
  }

  Expr* VisitorBase::getZeroInit(QualType T) {
    return utils::getZeroInit(T, m_Sema);
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
    SourceLocation fakeLoc = utils::GetValidSLoc(m_Sema);
    if (utils::isArrayOrPointerType(Base->getType())) {
      for (Expr* I : Indices)
        result =
            m_Sema.CreateBuiltinArraySubscriptExpr(result, fakeLoc, I, fakeLoc)
                .get();
    } else {
      Expr* idx = Indices.back();
      result = m_Sema
                   .ActOnArraySubscriptExpr(getCurrentScope(), Base, fakeLoc,
                                            idx, fakeLoc)
                   .get();
    }
    return result;
  }

  TemplateDecl* VisitorBase::GetCladTapeDecl() {
    static TemplateDecl* Result = nullptr;
    if (!Result)
      Result = utils::LookupTemplateDeclInCladNamespace(m_Sema,
                                                        /*ClassName=*/"tape");
    return Result;
  }

  LookupResult VisitorBase::LookupCladTapeMethod(llvm::StringRef name) {
    NamespaceDecl* CladNS = utils::GetCladNamespace(m_Sema);
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

  Expr* VisitorBase::GetFunctionCall(const std::string& funcName,
                                     const std::string& nmspace,
                                     llvm::SmallVectorImpl<Expr*>& callArgs) {
    CXXScopeSpec SS;
    DeclContext* DC = m_Context.getTranslationUnitDecl();
    if (!nmspace.empty()) {
      NamespaceDecl* NSD =
          utils::LookupNSD(m_Sema, nmspace, /*shouldExist=*/true);
      SS.Extend(m_Context, NSD, noLoc, noLoc);
      DC = NSD;
    }

    IdentifierInfo* II = &m_Context.Idents.get(funcName);
    DeclarationName name(II);
    DeclarationNameInfo DNI(name, noLoc);
    LookupResult R(m_Sema, DNI, Sema::LookupOrdinaryName);

    m_Sema.LookupQualifiedName(R, DC);
    Expr* UnresolvedLookup = nullptr;
    if (!R.empty())
      UnresolvedLookup =
          m_Sema.BuildDeclarationNameExpr(SS, R, /*ADL=*/false).get();
    auto MARargs = llvm::MutableArrayRef<Expr*>(callArgs);
    SourceLocation Loc;
    return m_Sema
        .ActOnCallExpr(getCurrentScope(), UnresolvedLookup, Loc, MARargs, Loc)
        .get();
  }

  DeclRefExpr* VisitorBase::GetCladTapePushDRE() {
    LookupResult& pushLR = GetCladTapePush();
    CXXScopeSpec CSS;
    CSS.Extend(m_Context, utils::GetCladNamespace(m_Sema), noLoc, noLoc);
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
    return utils::InstantiateTemplate(m_Sema, GetCladTapeDecl(), {T});
  }

  Expr* VisitorBase::BuildCallExprToMemFn(Expr* Base,
                                          StringRef MemberFunctionName,
                                          MutableArrayRef<Expr*> ArgExprs,
                                          SourceLocation Loc /*=noLoc*/) {
    if (Loc.isInvalid())
      Loc = m_DiffReq->getLocation();
    UnqualifiedId Member;
    Member.setIdentifier(&m_Context.Idents.get(MemberFunctionName), Loc);
    CXXScopeSpec SS;
    bool isArrow = Base->getType()->isPointerType();
    auto ME = m_Sema
                  .ActOnMemberAccessExpr(getCurrentScope(), Base, Loc,
                                         isArrow ? tok::TokenKind::arrow
                                                 : tok::TokenKind::period,
                                         SS, noLoc, Member,
                                         /*ObjCImpDecl=*/nullptr)
                  .getAs<MemberExpr>();
    return m_Sema.ActOnCallExpr(getCurrentScope(), ME, Loc, ArgExprs, Loc)
        .get();
  }

  static QualType getRefQualifiedThisType(Sema& semaRef, CXXMethodDecl* MD) {
    ASTContext& C = semaRef.getASTContext();
    CXXRecordDecl* RD = MD->getParent();
    auto RDType = RD->getTypeForDecl();
    auto thisObjectQType =
        C.getQualifiedType(RDType, MD->getMethodQualifiers());
    if (MD->getRefQualifier() == RefQualifierKind::RQ_RValue)
      thisObjectQType = C.getRValueReferenceType(thisObjectQType);
    else if (MD->getRefQualifier() == RefQualifierKind::RQ_LValue)
      thisObjectQType = C.getLValueReferenceType(thisObjectQType);
    return thisObjectQType;
  }

  Expr* VisitorBase::BuildCallExprToMemFn(
      clang::CXXMethodDecl* FD, llvm::MutableArrayRef<clang::Expr*> argExprs,
      bool useRefQualifiedThisObj, SourceLocation Loc /*=noLoc*/) {
    QualType ThisTy = FD->getThisType();
    Expr* thisExpr = m_Sema.BuildCXXThisExpr(Loc, ThisTy, /*IsImplicit=*/true);
    bool isArrow = true;
    if (Loc.isInvalid())
      Loc = m_DiffReq->getLocation();

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
    // Leads to printing this->Class::Function(x, y).
    // FIXME: Enable for static functions.
    NestedNameSpecifierLoc NNS /* = FD->getQualifierLoc()*/;
    auto DAP = DeclAccessPair::make(FD, FD->getAccess());
    auto* memberExpr =
        MemberExpr::Create(m_Context, thisExpr, isArrow, Loc, NNS, noLoc, FD,
                           DAP, FD->getNameInfo(),
                           /*TemplateArgs=*/nullptr, m_Context.BoundMemberTy,
                           CLAD_COMPAT_ExprValueKind_R_or_PR_Value,
                           ExprObjectKind::OK_Ordinary, NOUR_None);
    return m_Sema
        .BuildCallToMemberFunction(getCurrentScope(), memberExpr, Loc, argExprs,
                                   Loc)
        .get();
  }

  Expr*
  VisitorBase::BuildCallExprToFunction(FunctionDecl* FD,
                                       llvm::MutableArrayRef<Expr*> argExprs,
                                       bool useRefQualifiedThisObj /*=false*/) {
    Expr* call = nullptr;
    if (auto* MD = dyn_cast<CXXMethodDecl>(FD)) {
      if (MD->isInstance())
        call = BuildCallExprToMemFn(MD, argExprs, useRefQualifiedThisObj);
    } else {
      Expr* exprFunc = BuildDeclRef(FD);
      call = m_Sema
                 .ActOnCallExpr(
                     getCurrentScope(),
                     /*Fn=*/exprFunc,
                     /*LParenLoc=*/noLoc,
                     /*ArgExprs=*/llvm::MutableArrayRef<Expr*>(argExprs),
                     /*RParenLoc=*/m_DiffReq->getLocation())
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
    NamespaceDecl* CladNS = utils::GetCladNamespace(m_Sema);
    CXXScopeSpec CSS;
    CSS.Extend(m_Context, CladNS, loc, loc);
    m_Sema.LookupQualifiedName(R, CladNS, CSS);

    // Build the template specialization expression.
    // FIXME: currently this doesn't print func<templates>(args...) while
    // dumping and only prints func(args...), we need to fix this.
    auto* FTD = dyn_cast<FunctionTemplateDecl>(R.getRepresentativeDecl());
#if CLANG_VERSION_MAJOR < 19
    clang::TemplateArgumentList TL(TemplateArgumentList::OnStack, templateArgs);
#else
    auto& TL = *TemplateArgumentList::CreateCopy(m_Context, templateArgs);
#endif
    FunctionDecl* FD = m_Sema.InstantiateFunctionDeclaration(FTD, &TL, loc);

    return BuildCallExprToFunction(FD, argExprs,
                                   /*useRefQualifiedThisObj=*/false);
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

  Expr* VisitorBase::GetSingleArgCentralDiffCall(
      Expr* targetFuncCall, Expr* targetArg, unsigned targetPos,
      unsigned numArgs, llvm::SmallVectorImpl<Expr*>& args,
      Expr* CUDAExecConfig /*=nullptr*/) {
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
        Name, NumDiffArgs, getCurrentScope(),
        /*OriginalFnDC=*/nullptr,
        /*forCustomDerv=*/false,
        /*namespaceShouldExist=*/false, CUDAExecConfig);
  }

  void VisitorBase::CallExprDiffDiagnostics(const clang::FunctionDecl* FD,
                                            SourceLocation srcLoc) {
    bool NumDiffEnabled =
        !m_Sema.getPreprocessor().isMacroDefined("CLAD_NO_NUM_DIFF");
    // FIXME: Switch to the real diagnostics engine and pass FD directly.
    std::string funcName = FD->getNameAsString();
    diag(DiagnosticsEngine::Warning, srcLoc,
         "function '%0' was not differentiated because clad failed to "
         "differentiate it and no suitable overload was found in "
         "namespace 'custom_derivatives'",
         {funcName});
    if (NumDiffEnabled) {
      diag(DiagnosticsEngine::Note, srcLoc,
           "falling back to numerical differentiation for '%0' since no "
           "suitable overload was found and clad could not derive it; "
           "to disable this feature, compile your programs with "
           "-DCLAD_NO_NUM_DIFF",
           {funcName});
    } else {
      diag(DiagnosticsEngine::Note, srcLoc,
           "fallback to numerical differentiation is disabled by the "
           "'CLAD_NO_NUM_DIFF' macro; considering '%0' as 0",
           {funcName});
    }
  }

  ParmVarDecl* VisitorBase::CloneParmVarDecl(const ParmVarDecl* PVD,
                                             IdentifierInfo* II,
                                             bool pushOnScopeChains,
                                             bool cloneDefaultArg,
                                             SourceLocation Loc) {
    Expr* newPVDDefaultArg = nullptr;
    if (PVD->hasDefaultArg() && cloneDefaultArg) {
      newPVDDefaultArg = Clone(PVD->getDefaultArg());
    }
    if (Loc.isInvalid())
      Loc = PVD->getLocation();
    auto newPVD = ParmVarDecl::Create(
        m_Context, m_Sema.CurContext, Loc, Loc, II, PVD->getType(),
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

  void VisitorBase::ComputeEffectiveDOperands(StmtDiff& LDiff, StmtDiff& RDiff,
                                              clang::Expr*& derivedL,
                                              clang::Expr*& derivedR) {
    derivedL = LDiff.getExpr_dx();
    derivedR = RDiff.getExpr_dx();
    if (utils::isArrayOrPointerType(LDiff.getExpr()->getType()) &&
        !utils::isArrayOrPointerType(RDiff.getExpr()->getType()))
      derivedR = RDiff.getExpr();
    else if (utils::isArrayOrPointerType(RDiff.getExpr()->getType()) &&
             !utils::isArrayOrPointerType(LDiff.getExpr()->getType()))
      derivedL = LDiff.getExpr();
  }

  Stmt* VisitorBase::GetCladZeroInit(llvm::MutableArrayRef<Expr*> args) {
    static clad_compat::llvm_Optional<LookupResult> Result{};
    if (!Result)
      Result = LookupCladTapeMethod("zero_init");
    LookupResult& init = clad_compat::llvm_Optional_GetValue(Result);
    CXXScopeSpec CSS;
    CSS.Extend(m_Context, utils::GetCladNamespace(m_Sema), noLoc, noLoc);
    auto* pushDRE =
        m_Sema.BuildDeclarationNameExpr(CSS, init, false).getAs<DeclRefExpr>();
    return m_Sema.ActOnCallExpr(getCurrentScope(), pushDRE, noLoc, args, noLoc)
        .get();
  }

  clang::TemplateDecl* VisitorBase::GetCladConstructorPushforwardTag() {
    if (!m_CladConstructorPushforwardTag)
      m_CladConstructorPushforwardTag =
          utils::LookupTemplateDeclInCladNamespace(m_Sema,
                                                   "ConstructorPushforwardTag");
    return m_CladConstructorPushforwardTag;
  }

  clang::QualType
  VisitorBase::GetCladConstructorPushforwardTagOfType(clang::QualType T) {
    return utils::InstantiateTemplate(m_Sema,
                                      GetCladConstructorPushforwardTag(), {T});
  }

  clang::TemplateDecl* VisitorBase::GetCladConstructorReverseForwTag() {
    if (!m_CladConstructorPushforwardTag)
      m_CladConstructorReverseForwTag =
          utils::LookupTemplateDeclInCladNamespace(m_Sema,
                                                   "ConstructorReverseForwTag");
    return m_CladConstructorReverseForwTag;
  }

  clang::QualType
  VisitorBase::GetCladConstructorReverseForwTagOfType(clang::QualType T) {
    return utils::InstantiateTemplate(m_Sema,
                                      GetCladConstructorReverseForwTag(), {T});
  }

  FunctionDecl* VisitorBase::CreateDerivativeOverload() {
    auto diffParams = m_Derivative->parameters();
    auto diffNameInfo = m_Derivative->getNameInfo();
    // Calculate the total number of parameters that would be required for
    // automatic differentiation in the derived function if all args are
    // requested.
    // FIXME: Here we are assuming all function parameters are of differentiable
    // type. Ideally, we should not make any such assumption.
    std::size_t totalDerivedParamsSize = m_DiffReq->getNumParams() * 2;
    std::size_t numOfDerivativeParams = m_DiffReq->getNumParams();

    // Account for the this pointer.
    if (isa<CXXMethodDecl>(m_DiffReq.Function) &&
        !utils::IsStaticMethod(m_DiffReq.Function) &&
        (!m_DiffReq.Functor || m_DiffReq.Mode != DiffMode::jacobian))
      ++numOfDerivativeParams;
    // All output parameters will be of type `void*`. These
    // parameters will be casted to correct type before the call to the actual
    // derived function.
    // We require each output parameter to be of same type in the overloaded
    // derived function due to limitations of generating the exact derived
    // function type at the compile-time (without clad plugin help).
    QualType outputParamType = m_Context.getPointerType(m_Context.VoidTy);

    llvm::SmallVector<QualType, 16> paramTypes;

    // Add types for representing original function parameters.
    for (auto* PVD : m_DiffReq->parameters())
      paramTypes.push_back(PVD->getType());
    // Add types for representing parameter derivatives.
    // FIXME: We are assuming all function parameters are differentiable. We
    // should not make any such assumptions.
    for (std::size_t i = 0; i < numOfDerivativeParams; ++i)
      paramTypes.push_back(outputParamType);

    auto diffFuncOverloadEPI =
        dyn_cast<FunctionProtoType>(m_DiffReq->getType())->getExtProtoInfo();
    QualType diffFunctionOverloadType =
        m_Context.getFunctionType(m_Context.VoidTy, paramTypes,
                                  // Cast to function pointer.
                                  diffFuncOverloadEPI);

    // FIXME: We should not use const_cast to get the decl context here.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto* DC = const_cast<DeclContext*>(m_DiffReq->getDeclContext());
    m_Sema.CurContext = DC;
    DeclWithContext diffOverloadFDWC =
        m_Builder.cloneFunction(m_DiffReq.Function, *this, DC, noLoc,
                                diffNameInfo, diffFunctionOverloadType);
    FunctionDecl* diffOverloadFD = diffOverloadFDWC.first;

    beginScope(Scope::FunctionPrototypeScope | Scope::FunctionDeclarationScope |
               Scope::DeclScope);
    m_Sema.PushFunctionScope();
    m_Sema.PushDeclContext(getCurrentScope(), diffOverloadFD);

    llvm::SmallVector<ParmVarDecl*, 4> overloadParams;
    llvm::SmallVector<Expr*, 4> callArgs;

    overloadParams.reserve(totalDerivedParamsSize);
    callArgs.reserve(diffParams.size());

    for (auto* PVD : m_DiffReq->parameters()) {
      auto* VD = utils::BuildParmVarDecl(
          m_Sema, diffOverloadFD, PVD->getIdentifier(), PVD->getType(),
          PVD->getStorageClass(), /*defArg=*/nullptr, PVD->getTypeSourceInfo());
      overloadParams.push_back(VD);
      callArgs.push_back(BuildDeclRef(VD));
    }

    for (std::size_t i = 0; i < numOfDerivativeParams; ++i) {
      IdentifierInfo* II = nullptr;
      StorageClass SC = StorageClass::SC_None;
      std::size_t effectiveDiffIndex = m_DiffReq->getNumParams() + i;
      // `effectiveDiffIndex < diffParams.size()` implies that this
      // parameter represents an actual derivative of one of the function
      // original parameters.
      if (effectiveDiffIndex < diffParams.size()) {
        auto* GVD = diffParams[effectiveDiffIndex];
        II = CreateUniqueIdentifier("_temp_" + GVD->getNameAsString());
        SC = GVD->getStorageClass();
      } else {
        II = CreateUniqueIdentifier("_d_" + std::to_string(i));
      }
      auto* PVD = utils::BuildParmVarDecl(m_Sema, diffOverloadFD, II,
                                          outputParamType, SC);
      overloadParams.push_back(PVD);
    }

    for (auto* PVD : overloadParams)
      if (PVD->getIdentifier())
        m_Sema.PushOnScopeChains(PVD, getCurrentScope(),
                                 /*AddToContext=*/false);

    diffOverloadFD->setParams(overloadParams);
    diffOverloadFD->setBody(/*B=*/nullptr);

    beginScope(Scope::FnScope | Scope::DeclScope);
    m_DerivativeFnScope = getCurrentScope();
    beginBlock();

    // Build derivatives to be used in the call to the actual derived function.
    // These are initialised by effectively casting the derivative parameters of
    // overloaded derived function to the correct type.
    for (std::size_t i = m_DiffReq->getNumParams(); i < diffParams.size();
         ++i) {
      auto* overloadParam = overloadParams[i];
      auto* diffParam = diffParams[i];
      TypeSourceInfo* typeInfo =
          m_Context.getTrivialTypeSourceInfo(diffParam->getType());
      SourceLocation fakeLoc = utils::GetValidSLoc(m_Sema);
      auto* init = m_Sema
                       .BuildCStyleCastExpr(fakeLoc, typeInfo, fakeLoc,
                                            BuildDeclRef(overloadParam))
                       .get();

      auto* diffVD =
          BuildGlobalVarDecl(diffParam->getType(), diffParam->getName(), init);
      callArgs.push_back(BuildDeclRef(diffVD));
      addToCurrentBlock(BuildDeclStmt(diffVD));
    }

    // If the function is a global kernel, we need to transform it
    // into a device function when calling it inside the overload function
    // which is the final global kernel returned.
    if (m_Derivative->hasAttr<clang::CUDAGlobalAttr>()) {
      m_Derivative->dropAttr<clang::CUDAGlobalAttr>();
      m_Derivative->addAttr(clang::CUDADeviceAttr::CreateImplicit(m_Context));
    }

    Expr* callExpr = BuildCallExprToFunction(m_Derivative, callArgs,
                                             /*useRefQualifiedThisObj=*/true);
    addToCurrentBlock(callExpr);
    Stmt* diffOverloadBody = endBlock();

    diffOverloadFD->setBody(diffOverloadBody);

    endScope(); // Function body scope
    m_Sema.PopFunctionScopeInfo();
    m_Sema.PopDeclContext();
    endScope(); // Function decl scope

    return diffOverloadFD;
  }

  VisitorBase::~VisitorBase() = default;

  QualType
  VisitorBase::GetDerivativeType(llvm::ArrayRef<QualType> customParams) {
    llvm::SmallVector<const ValueDecl*, 4> diffParams{};
    for (const DiffInputVarInfo& VarInfo : m_DiffReq.DVI)
      diffParams.push_back(VarInfo.param);
    return utils::GetDerivativeType(m_Sema, m_DiffReq.Function, m_DiffReq.Mode,
                                    diffParams, /*moveBaseToParams=*/false,
                                    customParams);
  }

  FunctionDecl* VisitorBase::FindDerivedFunction(DiffRequest& request) {
    // Check if the call is recursive
    if (request == m_DiffReq)
      return m_Derivative;
    // Only definitions are differentiated
    if (request.Function->getDefinition())
      request.Function = request.Function->getDefinition();
    // Look for the derivative
    return m_Builder.FindDerivedFunction(request);
  }

  Expr* VisitorBase::BuildOperatorCall(OverloadedOperatorKind OOK,
                                       MutableArrayRef<Expr*> ArgExprs,
                                       SourceLocation OpLoc) {
    // First check operator kinds that are not considered binary/unary.

    // FIXME: Currently, Clad never uses arrow operators, all of them
    // are replaced with reverse_forw functions. This bit might become
    // useful in the future when Clad can remove some reverse_forw
    // functions in favor of original functions.
    // if (OOK == OO_Arrow)
    //   return m_Sema
    //       .BuildOverloadedArrowExpr(getCurrentScope(), ArgExprs[0], OpLoc)
    //       .get();

    if (OOK == OO_Call)
      return m_Sema
          .BuildCallToObjectOfClassType(getCurrentScope(), ArgExprs[0], OpLoc,
                                        ArgExprs.drop_front(), OpLoc)
          .get();

    if (OOK == OO_Subscript)
      return m_Sema
          .CreateOverloadedArraySubscriptExpr(OpLoc, OpLoc, ArgExprs[0],
                                              ArgExprs[1])
          .get();

    // Now deduce the kind based on the number of args.
    // Note for debugging: if the number of args is wrong,
    // the kind will be deduced incorrectly, and
    // getOverloadedOpcode will crash Clang.
    if (ArgExprs.size() == 2) {
      BinaryOperatorKind kind = BinaryOperator::getOverloadedOpcode(OOK);
      return BuildOp(kind, ArgExprs[0], ArgExprs[1], OpLoc);
    }

    if (ArgExprs.size() == 1) {
      UnaryOperatorKind kind = UnaryOperator::getOverloadedOpcode(OOK, true);
      return BuildOp(kind, ArgExprs[0], OpLoc);
    }
    return nullptr;
  }
} // end namespace clad
