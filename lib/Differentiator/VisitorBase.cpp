//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clad/Differentiator/VisitorBase.h"

#include "ConstantFolder.h"

#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/StmtClone.h"

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
    auto Stmts_ref = llvm::makeArrayRef(Stmts.data(), Stmts.size());
    return clad_compat::CompoundStmt_Create(m_Context, Stmts_ref, noLoc, noLoc);
  }

  DiffParamsWithIndices VisitorBase::parseDiffArgs(const Expr* diffArgs,
                                                   const FunctionDecl* FD) {
    DiffParams params{};
    auto E = diffArgs->IgnoreParenImpCasts();
    // Case 1)
    if (auto SL = dyn_cast<StringLiteral>(E)) {
      IndexIntervalTable indexes{};
      llvm::StringRef string = SL->getString().trim();
      if (string.empty()) {
        diag(DiagnosticsEngine::Error,
             diffArgs->getEndLoc(),
             "No parameters were provided");
        return {};
      }
      // Split the string by ',' characters, trim whitespaces.
      llvm::SmallVector<llvm::StringRef, 16> names{};
      llvm::StringRef name{};
      do {
        std::tie(name, string) = string.split(',');
        names.push_back(name.trim());
      } while (!string.empty());
      // Find function's parameters corresponding to the specified names.
      llvm::SmallVector<std::pair<llvm::StringRef, VarDecl*>, 16>
          param_names_map{};
      for (auto PVD : FD->parameters())
        param_names_map.emplace_back(PVD->getName(), PVD);
      for (const auto& name : names) {
        size_t loc = name.find('[');
        loc = (loc == llvm::StringRef::npos) ? name.size() : loc;
        llvm::StringRef base = name.substr(0, loc);

        auto it = std::find_if(
            std::begin(param_names_map),
            std::end(param_names_map),
            [&base](const std::pair<llvm::StringRef, VarDecl*>& p) {
              return p.first == base;
            });

        if (it == std::end(param_names_map)) {
          // Fail if the function has no parameter with specified name.
          diag(DiagnosticsEngine::Error,
               diffArgs->getEndLoc(),
               "Requested parameter name '%0' was not found among function "
               "parameters",
               {base});
          return {};
        }

        auto f_it = std::find(std::begin(params), std::end(params), it->second);

        if (f_it != params.end()) {
          diag(DiagnosticsEngine::Error,
               diffArgs->getEndLoc(),
               "Requested parameter '%0' was specified multiple times",
               {it->second->getName()});
          return {};
        }

        params.push_back(it->second);

        if (loc != name.size()) {
          llvm::StringRef interval(
              name.slice(loc + 1, name.find(']')));
          llvm::StringRef firstStr, lastStr;
          std::tie(firstStr, lastStr) = interval.split(':');

          if (lastStr.empty()) {
            // The string is not a range just a single index
            size_t index;
            firstStr.getAsInteger(10, index);
            indexes.push_back(IndexInterval(index));
          } else {
            size_t first, last;
            firstStr.getAsInteger(10, first);
            lastStr.getAsInteger(10, last);
            if (first >= last) {
              diag(DiagnosticsEngine::Error,
                   diffArgs->getEndLoc(),
                   "Range specified in '%0' is in incorrect format",
                   {name});
              return {};
            }
            indexes.push_back(IndexInterval(first, last));
          }
        } else {
          indexes.push_back(IndexInterval());
        }
      }
      // Return a sequence of function's parameters.
      return {params, indexes};
    }
    // Case 2)
    // Check if the provided literal can be evaluated as an integral value.
    llvm::APSInt intValue;
    if (clad_compat::Expr_EvaluateAsInt(E, intValue, m_Context)) {
      auto idx = intValue.getExtValue();
      // Fail if the specified index is invalid.
      if ((idx < 0) || (idx >= FD->getNumParams())) {
        diag(DiagnosticsEngine::Error,
             diffArgs->getEndLoc(),
             "Invalid argument index %0 among %1 argument(s)",
             {std::to_string(idx), std::to_string(FD->getNumParams())});
        return {};
      }
      params.push_back(FD->getParamDecl(idx));
      // Returns a single parameter.
      return {params, {}};
    }
    // Case 3)
    // Treat the default (unspecified) argument as a special case, as if all
    // function's arguments were requested.
    if (isa<CXXDefaultArgExpr>(E)) {
      std::copy(FD->param_begin(), FD->param_end(), std::back_inserter(params));
      // If the function has no parameters, then we cannot differentiate it."
      if (params.empty())
        diag(DiagnosticsEngine::Error,
             diffArgs->getEndLoc(),
             "Attempted to differentiate a function without parameters");
      // Returns the sequence with all the function's parameters.
      return {params, {}};
    }
    // Fail if the argument is not a string or numeric literal.
    diag(DiagnosticsEngine::Error,
         diffArgs->getEndLoc(),
         "Failed to parse the parameters, must be a string or numeric literal");
    return {{}, {}};
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

    auto VD =
        VarDecl::Create(m_Context, m_Sema.CurContext, m_Function->getLocation(),
                        m_Function->getLocation(), Identifier, Type, TSI,
                        SC_None);

    if (Init) {
      m_Sema.AddInitializerToDecl(VD, Init, DirectInit);
      VD->setInitStyle(IS);
    }
    // Add the identifier to the scope and IdResolver
    m_Sema.PushOnScopeChains(VD, getCurrentScope(), /*AddToContext*/ false);
    return VD;
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
    NamespaceDecl* NDecl = NamespaceDecl::Create(
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

  DeclRefExpr* VisitorBase::BuildDeclRef(DeclaratorDecl* D) {
    QualType T = D->getType();
    T = T.getNonReferenceType();
    return cast<DeclRefExpr>(clad_compat::GetResult<Expr*>(
        m_Sema.BuildDeclRefExpr(D, T, VK_LValue, noLoc)));
  }

  IdentifierInfo*
  VisitorBase::CreateUniqueIdentifier(llvm::StringRef nameBase) {
    // For intermediate variables, use numbered names (_t0), for everything
    // else first try a name without number (e.g. first try to use _d_x and
    // use _d_x0 only if _d_x is taken).
    bool countedName = nameBase.startswith("_") && !nameBase.startswith("_d_");
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
        isa<ConditionalOperator>(ENoCasts))
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
    return m_Sema.BuildUnaryOp(nullptr, OpLoc, OpCode, E).get();
  }

  Expr* VisitorBase::BuildOp(clang::BinaryOperatorKind OpCode, Expr* L, Expr* R,
                             SourceLocation OpLoc) {
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

  TemplateDecl* VisitorBase::GetCladClassDecl(llvm::StringRef ClassName) {
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

  QualType
  VisitorBase::GetCladClassOfType(TemplateDecl* CladClassDecl,
                                  MutableArrayRef<QualType> TemplateArgs) {
    // Create a list of template arguments.
    TemplateArgumentListInfo TLI{};
    for (auto T : TemplateArgs) {
      TemplateArgument TA = T;
      TLI.addArgument(
          TemplateArgumentLoc(TA, m_Context.getTrivialTypeSourceInfo(T)));
    }
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

  TemplateDecl* VisitorBase::GetCladTapeDecl() {
    static TemplateDecl* Result = nullptr;
    if (!Result)
      Result = GetCladClassDecl(/*ClassName=*/"tape");
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
    static llvm::Optional<LookupResult> Result{};
    if (Result)
      return Result.getValue();
    Result = LookupCladTapeMethod("push");
    return Result.getValue();
  }

  LookupResult& VisitorBase::GetCladTapePop() {
    static llvm::Optional<LookupResult> Result{};
    if (Result)
      return Result.getValue();
    Result = LookupCladTapeMethod("pop");
    return Result.getValue();
  }

  LookupResult& VisitorBase::GetCladTapeBack() {
    static llvm::Optional<LookupResult> Result{};
    if (Result)
      return Result.getValue();
    Result = LookupCladTapeMethod("back");
    return Result.getValue();
  }

  QualType VisitorBase::GetCladTapeOfType(QualType T) {
    return GetCladClassOfType(GetCladTapeDecl(), {T});
  }

  Expr* VisitorBase::BuildCallExprToMemFn(Expr* Base, bool isArrow,
                                          StringRef MemberFunctionName,
                                          MutableArrayRef<Expr*> ArgExprs) {
    UnqualifiedId Member;
    Member.setIdentifier(&m_Context.Idents.get(MemberFunctionName), noLoc);
    CXXScopeSpec SS;
    auto ME = m_Sema
                  .ActOnMemberAccessExpr(getCurrentScope(), Base, noLoc,
                                         isArrow ? tok::TokenKind::arrow
                                                 : tok::TokenKind::period,
                                         SS, noLoc, Member,
                                         /*ObjCImpDecl=*/nullptr)
                  .get();
    return m_Sema.ActOnCallExpr(getCurrentScope(), ME, noLoc, ArgExprs, noLoc)
        .get();
  }

  clang::Expr* 
  VisitorBase::BuildCallExprToMemFn(clang::CXXMethodDecl* FD,
                  llvm::MutableArrayRef<clang::Expr*> argExprs) {
    Expr* thisExpr = clad_compat::Sema_BuildCXXThisExpr(m_Sema, FD);
    NestedNameSpecifierLoc NNS(FD->getQualifier(),
                               /*Data=*/nullptr);
    auto DAP = DeclAccessPair::make(FD, FD->getAccess());
    auto memberExpr = MemberExpr::
        Create(m_Context, thisExpr, /*isArrow=*/true, noLoc, NNS, noLoc, FD,
               DAP, FD->getNameInfo(),
               /*TemplateArgs=*/nullptr, m_Context.BoundMemberTy,
               ExprValueKind::VK_RValue,
               ExprObjectKind::OK_Ordinary
               CLAD_COMPAT_CLANG9_MemberExpr_ExtraParams(NOUR_None));
    return m_Sema
        .BuildCallToMemberFunction(getCurrentScope(), memberExpr, noLoc,
                                   argExprs, noLoc)
        .get();
  }

  Expr* 
  VisitorBase::BuildCallExprToFunction(FunctionDecl* FD, 
                  llvm::MutableArrayRef<Expr*> argExprs) {
    Expr* call = nullptr;
    if (auto derMethod = dyn_cast<CXXMethodDecl>(FD)) {
      call = BuildCallExprToMemFn(derMethod, argExprs);
    } else {
      Expr* exprFunc = BuildDeclRef(FD);
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

  TemplateDecl* VisitorBase::GetCladArrayRefDecl() {
    static TemplateDecl* Result = nullptr;
    if (!Result)
      Result = GetCladClassDecl(/*ClassName=*/"array_ref");
    return Result;
  }

  QualType VisitorBase::GetCladArrayRefOfType(clang::QualType T) {
    return GetCladClassOfType(GetCladArrayRefDecl(), {T});
  }

  TemplateDecl* VisitorBase::GetCladArrayDecl() {
    static TemplateDecl* Result = nullptr;
    if (!Result)
      Result = GetCladClassDecl(/*ClassName=*/"array");
    return Result;
  }

  QualType VisitorBase::GetCladArrayOfType(clang::QualType T) {
    return GetCladClassOfType(GetCladArrayDecl(), {T});
  }

  Expr* VisitorBase::BuildArrayRefSizeExpr(Expr* Base) {
    return BuildCallExprToMemFn(Base, /*isArrow=*/false,
                                /*MemberFunctionName=*/"size", {});
  }

  Expr* VisitorBase::BuildArrayRefSliceExpr(Expr* Base,
                                            MutableArrayRef<Expr*> Args) {
    return BuildCallExprToMemFn(Base, /*isArrow=*/false,
                                /*MemberFunctionName=*/"slice", Args);
  }

  bool VisitorBase::isArrayRefType(QualType QT) {
    return QT.getAsString().find("clad::array_ref") != std::string::npos;
  }
} // end namespace clad