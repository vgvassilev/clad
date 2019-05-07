//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clad/Differentiator/DerivativeBuilder.h"

#include "ConstantFolder.h"

#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/StmtClone.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/SemaInternal.h"

#include "llvm/Support/SaveAndRestore.h"

#include <algorithm>

using namespace clang;

namespace clad {
  DerivativeBuilder::DerivativeBuilder(clang::Sema& S, plugin::CladPlugin& P)
    : m_Sema(S), m_CladPlugin(P), m_Context(S.getASTContext()),
      m_NodeCloner(new utils::StmtClone(m_Sema, m_Context)),
      m_BuiltinDerivativesNSD(nullptr) {}

  DerivativeBuilder::~DerivativeBuilder() {}

  static void registerDerivative(FunctionDecl* derivedFD, Sema& semaRef) {
    LookupResult R(semaRef, derivedFD->getNameInfo(), Sema::LookupOrdinaryName);
    semaRef.LookupQualifiedName(R, derivedFD->getDeclContext(),
                                /*allowBuiltinCreation*/ false);
    // Inform the decl's decl context for its existance after the lookup,
    // otherwise it would end up in the LookupResult.
    derivedFD->getDeclContext()->addDecl(derivedFD);

    if (R.empty())
      return;
    // Register the function on the redecl chain.
    derivedFD->setPreviousDecl(cast<FunctionDecl>(R.getFoundDecl()));

  }

  DeclWithContext DerivativeBuilder::Derive(const FunctionDecl* FD,
                                            const DiffRequest& request) {
    //m_Sema.CurContext = m_Context.getTranslationUnitDecl();
    assert(FD && "Must not be null.");
    // If FD is only a declaration, try to find its definition.
    if (!FD->getDefinition()) {
      if (request.VerboseDiags)
        diag(DiagnosticsEngine::Error, 
             request.CallContext ? request.CallContext->getLocStart() : noLoc,
             "attempted differentiation of function '%0', which does not have a "
             "definition", { FD->getNameAsString() });
      return {};
    }
    FD = FD->getDefinition();
    DeclWithContext result{};
    if (request.Mode == DiffMode::forward) {
      ForwardModeVisitor V(*this);
      result = V.Derive(FD, request);
    }
    else if (request.Mode == DiffMode::reverse) {
      ReverseModeVisitor V(*this);
      result = V.Derive(FD, request);
    }

    if (result.first)
      registerDerivative(result.first, m_Sema);
    return result;
  }

  DiffParams VisitorBase::parseDiffArgs(const Expr* diffArgs,
                                        const FunctionDecl* FD) {
    DiffParams params{};
    auto E = diffArgs->IgnoreParenImpCasts();
    // Case 1)
    if (auto SL = dyn_cast<StringLiteral>(E)) {
      llvm::StringRef string = SL->getString().trim();
      if (string.empty()) {
        diag(DiagnosticsEngine::Error, diffArgs->getLocEnd(),
             "No parameters were provided");
        return {};
      }
      // Split the string by ',' charachters, trim whitespaces.
      llvm::SmallVector<llvm::StringRef, 16> names{};
      llvm::StringRef name{};
      do {
        std::tie(name, string) = string.split(',');
        names.push_back(name.trim());
      } while (!string.empty());
      // Find function's parameters corresponding to the specified names.
      llvm::SmallVector<std::pair<llvm::StringRef, VarDecl*>, 16> param_names_map{};
      for (auto PVD : FD->parameters())
        param_names_map.emplace_back(PVD->getName(), PVD);
      for (const auto & name: names) {
        auto it = std::find_if(std::begin(param_names_map),
                    std::end(param_names_map),
                    [&name] (const std::pair<llvm::StringRef, VarDecl*> & p) {
                      return p.first == name; });
        if (it == std::end(param_names_map)) {
          // Fail if the function has no parameter with specified name.
          diag(DiagnosticsEngine::Error, diffArgs->getLocEnd(),
            "Requested parameter name '%0' was not found among function parameters",
            { name });
          return {};
        }
        params.push_back(it->second);
      }
      // Check if the same parameter was specified multiple times, fail if so.
      DiffParams unique_params{};
      for (const auto param : params) {
        auto it = std::find(std::begin(unique_params), std::end(unique_params), param);
        if (it != std::end(unique_params)) {
          diag(DiagnosticsEngine::Error, diffArgs->getLocEnd(),
            "Requested parameter '%0' was specified multiple times",
            { param->getName() });
          return {};
        }
        unique_params.push_back(param);
      }
      // Return a sequence of function's parameters.
      return unique_params;
    }
    // Case 2)
    // Check if the provided literal can be evaluated as an integral value.
    llvm::APSInt intValue;
    if (E->EvaluateAsInt(intValue, m_Context)) {
      auto idx = intValue.getExtValue();
      // Fail if the specified index is invalid.
      if ((idx < 0) || (idx >= FD->getNumParams())) {
        diag(DiagnosticsEngine::Error, diffArgs->getLocEnd(),
          "Invalid argument index %0 among %1 argument(s)",
          { std::to_string(idx), std::to_string(FD->getNumParams()) });
        return {};
      }
      params.push_back(FD->getParamDecl(idx));
      // Returns a single parameter.
      return params;
    }
    // Case 3)
    // Threat the default (unspecified) argument as a special case, as if all
    // function's arguments were requested.
    if (isa<CXXDefaultArgExpr>(E)) {
      std::copy(FD->param_begin(), FD->param_end(), std::back_inserter(params));
      // If the function has no parameters, then we cannot differentiate it."
      if (params.empty())
        diag(DiagnosticsEngine::Error, diffArgs->getLocEnd(),
             "Attempted to differentiate a function without parameters");
      // Returns the sequence with all the function's parameters.
      return params;
    }
    // Fail if the argument is not a string or numeric literal.
    diag(DiagnosticsEngine::Error, diffArgs->getLocEnd(),
         "Failed to parse the parameters, must be a string or numeric literal");
    return {};
  }

  bool VisitorBase::isUnusedResult(const Expr* E) {
    const Expr* ignoreExpr;
    SourceLocation ignoreLoc;
    SourceRange ignoreRange;
    return E->isUnusedResultAWarning(ignoreExpr, ignoreLoc, ignoreRange,
                                     ignoreRange, m_Context);
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

  VarDecl* VisitorBase::BuildVarDecl(QualType Type,
                                     IdentifierInfo* Identifier,
                                     Expr* Init,
                                     bool DirectInit) {

    auto VD = VarDecl::Create(m_Context,
                              m_Sema.CurContext,
                              noLoc,
                              noLoc,
                              Identifier,
                              Type,
                              nullptr, // FIXME: Should there be any TypeInfo?
                              SC_None);

    if (Init)
      m_Sema.AddInitializerToDecl(VD, Init, DirectInit);
    // Add the identifier to the scope and IdResolver
    m_Sema.PushOnScopeChains(VD, getCurrentScope(), /*AddToContext*/ false);
    return VD;
  }

  VarDecl* VisitorBase::BuildVarDecl(QualType Type,
                                     llvm::StringRef prefix,
                                     Expr* Init,
                                     bool DirectInit) {
    return BuildVarDecl(Type, CreateUniqueIdentifier(prefix), Init, DirectInit);
  }

  NamespaceDecl* VisitorBase::BuildNamespaceDecl(IdentifierInfo* II,
                                                 bool isInline) {
    // Check if the namespace is being redeclared.
    NamespaceDecl* PrevNS = nullptr;
    // From Sema::ActOnStartNamespaceDef:
    if (II) {
      LookupResult R(m_Sema, II, noLoc, Sema::LookupOrdinaryName,
                     Sema::ForRedeclaration);
      m_Sema.LookupQualifiedName(R, m_Sema.CurContext->getRedeclContext());
      NamedDecl* FoundDecl =
        R.isSingleResult() ? R.getRepresentativeDecl() : nullptr;
      PrevNS = dyn_cast_or_null<NamespaceDecl>(FoundDecl);
    }
    else {
      // Is anonymous namespace.
      DeclContext *Parent = m_Sema.CurContext->getRedeclContext();
      if (TranslationUnitDecl *TU = dyn_cast<TranslationUnitDecl>(Parent)) {
        PrevNS = TU->getAnonymousNamespace();
      } else {
        NamespaceDecl *ND = cast<NamespaceDecl>(Parent);
        PrevNS = ND->getAnonymousNamespace();
      }
    }
    NamespaceDecl* NDecl = NamespaceDecl::Create(m_Context, m_Sema.CurContext,
                                                 isInline, noLoc, noLoc, II,
                                                 PrevNS);
    if (II)
      m_Sema.PushOnScopeChains(NDecl, m_CurScope);
    else {
      // Link the anonymous namespace into its parent.
      // From Sema::ActOnStartNamespaceDef:
      DeclContext *Parent = m_Sema.CurContext->getRedeclContext();
      if (TranslationUnitDecl *TU = dyn_cast<TranslationUnitDecl>(Parent)) {
        TU->setAnonymousNamespace(NDecl);
      } else {
        cast<NamespaceDecl>(Parent)->setAnonymousNamespace(NDecl);
      }
      m_Sema.CurContext->addDecl(NDecl);
      if (!PrevNS) {
        UsingDirectiveDecl* UD =
          UsingDirectiveDecl::Create(m_Context, Parent, noLoc, noLoc,
                                     NestedNameSpecifierLoc(), noLoc,
                                     NDecl, Parent);
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
      NamespaceDecl* NewD = BuildNamespaceDecl(ND->getIdentifier(),
                                               ND->isInline());
      return Head ? Head : NewD;
    }
    else {
      m_Sema.CurContext = DC;
      return nullptr;
    }
  }

  DeclStmt* VisitorBase::BuildDeclStmt(Decl* D) {
    Stmt* DS = m_Sema.ActOnDeclStmt(m_Sema.ConvertDeclToDeclGroup(D), noLoc,
                                    noLoc).get();
    return cast<DeclStmt>(DS);
  }

  DeclStmt* VisitorBase::BuildDeclStmt(llvm::MutableArrayRef<Decl*> Decls) {
    auto DGR = DeclGroupRef::Create(m_Context, Decls.data(), Decls.size());
    return new (m_Context) DeclStmt(DGR, noLoc, noLoc);
  }

  DeclRefExpr* VisitorBase::BuildDeclRef(DeclaratorDecl* D) {
    QualType T = D->getType();
    T = T.getNonReferenceType();
    Expr* DRE = m_Sema.BuildDeclRefExpr(D, T, VK_LValue, noLoc).get();
    return cast<DeclRefExpr>(DRE);
  }

  IdentifierInfo*
  VisitorBase::CreateUniqueIdentifier(llvm::StringRef nameBase) {
    // For intermediate variables, use numbered names (_t0), for everything
    // else first try a name without number (e.g. first try to use _d_x and
    // use _d_x0 only if _d_x is taken).
    bool countedName = nameBase.startswith("_") && !nameBase.startswith("_d_");
    std::size_t idx = 0;
    std::size_t& id = countedName ? m_idCtr[nameBase] : idx;
    std::string idStr = countedName ? std::to_string(id) : "";
    if (countedName)
      id += 1;
    for (;;) {
      IdentifierInfo* name = &m_Context.Idents.get(nameBase.str() + idStr);
      LookupResult R(m_Sema, DeclarationName(name), noLoc, 
                     Sema::LookupOrdinaryName);
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
                                 bool forceDeclCreation) {
    return StoreAndRef(E, getCurrentBlock(), prefix, forceDeclCreation);
  }
  Expr* VisitorBase::StoreAndRef(Expr* E, Stmts& block, llvm::StringRef prefix,
                                 bool forceDeclCreation) {
    assert(E && "cannot infer type from null expression");
    return StoreAndRef(E, E->getType(), block, prefix, forceDeclCreation);
  }
 
  bool UsefulToStore(Expr* E) {
    if (!E)
      return false;
    Expr* B = E->IgnoreParenImpCasts();
    // FIXME: find a more general way to determine that or add more options.
    if (isa<DeclRefExpr>(B) || isa<FloatingLiteral>(B) || isa<IntegerLiteral>(B))
      return false;
    return true;
  }

  Expr* VisitorBase::StoreAndRef(Expr* E, QualType Type, Stmts& block,
                                 llvm::StringRef prefix, bool forceDeclCreation) {
    if (!forceDeclCreation) {
      // If Expr is simple (i.e. a reference or a literal), there is no point
      // in storing it as there is no evaluation going on.
      if (!UsefulToStore(E))
        return E;
    }
    // Create variable declaration.
    VarDecl* Var = BuildVarDecl(Type, CreateUniqueIdentifier(prefix), E);

    // Add the declaration to the body of the gradient function.
    addToBlock(BuildDeclStmt(Var), block);

    // Return reference to the declaration instead of original expression.
    return BuildDeclRef(Var);
  }

  ForwardModeVisitor::ForwardModeVisitor(DerivativeBuilder& builder):
    VisitorBase(builder) {}

  ForwardModeVisitor::~ForwardModeVisitor() {}

  DeclWithContext ForwardModeVisitor::Derive(const FunctionDecl* FD,
                                             const DiffRequest& request) {
    silenceDiags = !request.VerboseDiags;
    m_Function = FD;
    assert(!m_DerivativeInFlight
           && "Doesn't support recursive diff. Use DiffPlan.");
    m_DerivativeInFlight = true;

    DiffParams args{};
    if (request.Args)
      args = parseDiffArgs(request.Args, FD);
    else {
      //FIXME: implement gradient-vector products to fix the issue.
      assert((FD->getNumParams() <= 1) &&
             "nested forward mode differentiation for several args is broken");
      std::copy(FD->param_begin(), FD->param_end(), std::back_inserter(args));
    }
    if (args.empty())
      return {};
    if (args.size() > 1) {
      diag(DiagnosticsEngine::Error, request.Args->getLocEnd(),
        "Forward mode differentiation w.r.t. several parameters at once is not "
        "supported, call 'clad::differentiate' for each parameter separately");
      return {};
    }

    m_IndependentVar = args.back();
    // If param is not real (i.e. floating point or integral), we cannot
    // differentiate it.
    // FIXME: we should support custom numeric types in the future.
    if (!m_IndependentVar->getType()->isRealType()) {
      diag(DiagnosticsEngine::Error, m_IndependentVar->getLocEnd(),
           "attempted differentiation w.r.t. a parameter ('%0') which is not "
            "of a real type", { m_IndependentVar->getNameAsString() });
      return {};
    }
    m_DerivativeOrder = request.CurrentDerivativeOrder;
    std::string s = std::to_string(m_DerivativeOrder);
    std::string derivativeBaseName;
    if (m_DerivativeOrder == 1)
      s = "";
    switch (FD->getOverloadedOperator()) {
    default:
      derivativeBaseName = request.BaseFunctionName;
      break;
    case OO_Call:
      derivativeBaseName = "operator_call";
      break;
    }

    m_ArgIndex = std::distance(FD->param_begin(),
      std::find(FD->param_begin(), FD->param_end(), m_IndependentVar));
    IdentifierInfo* II = &m_Context.Idents.get(
      derivativeBaseName + "_d" + s + "arg" + std::to_string(m_ArgIndex));
    DeclarationNameInfo name(II, noLoc);
    FunctionDecl* derivedFD = nullptr;
    NamespaceDecl* enclosingNS = nullptr;
    llvm::SaveAndRestore<DeclContext*> SaveContext(m_Sema.CurContext);
    llvm::SaveAndRestore<Scope*> SaveScope(m_CurScope);
    DeclContext* DC = const_cast<DeclContext*>(m_Function->getDeclContext());
    m_Sema.CurContext = DC;
    if (isa<CXXMethodDecl>(FD)) {
      CXXRecordDecl* CXXRD = cast<CXXRecordDecl>(DC);
      derivedFD = CXXMethodDecl::Create(m_Context, CXXRD, noLoc, name,
                                        FD->getType(), FD->getTypeSourceInfo(),
                                        FD->getStorageClass(),
                                        FD->isInlineSpecified(),
                                        FD->isConstexpr(), noLoc);
      derivedFD->setAccess(FD->getAccess());
    } else {
      assert(isa<FunctionDecl>(FD) && "Must derive from FunctionDecl.");
      enclosingNS = RebuildEnclosingNamespaces(DC);
      derivedFD = FunctionDecl::Create(m_Context,
                                       m_Sema.CurContext, noLoc,
                                       name, FD->getType(),
                                       FD->getTypeSourceInfo(),
                                       FD->getStorageClass(),
                                       /*default*/
                                       FD->isInlineSpecified(),
                                       FD->hasWrittenPrototype(),
                                       FD->isConstexpr());
    }
    m_Derivative = derivedFD;

    llvm::SmallVector<ParmVarDecl*, 4> params;
    ParmVarDecl* newPVD = nullptr;
    const ParmVarDecl* PVD = nullptr;

    // Function declaration scope
    beginScope(Scope::FunctionPrototypeScope |
               Scope::FunctionDeclarationScope |
               Scope::DeclScope);
    m_Sema.PushFunctionScope();
    m_Sema.PushDeclContext(getCurrentScope(), m_Derivative);

    // FIXME: We should implement FunctionDecl and ParamVarDecl cloning.
    for(size_t i = 0, e = FD->getNumParams(); i < e; ++i) {
      PVD = FD->getParamDecl(i);
      Expr* clonedPVDDefaultArg = 0;
      if (PVD->hasDefaultArg())
        clonedPVDDefaultArg = Clone(PVD->getDefaultArg());

      newPVD = ParmVarDecl::Create(m_Context, m_Sema.CurContext, noLoc, noLoc,
                                   PVD->getIdentifier(), PVD->getType(),
                                   PVD->getTypeSourceInfo(),
                                   PVD->getStorageClass(),
                                   clonedPVDDefaultArg);

      // Make m_IndependentVar to point to the argument of the newly created
      // derivedFD.
      if (PVD == m_IndependentVar)
        m_IndependentVar = newPVD;

      params.push_back(newPVD);
      // Add the args in the scope and id chain so that they could be found.
      if (newPVD->getIdentifier())
        m_Sema.PushOnScopeChains(newPVD,
                                 getCurrentScope(),
                                 /*AddToContext*/ false);
    }

    llvm::ArrayRef<ParmVarDecl*> paramsRef
      = llvm::makeArrayRef(params.data(), params.size());
    derivedFD->setParams(paramsRef);
    derivedFD->setBody(nullptr);

    // Function body scope
    beginScope(Scope::FnScope | Scope::DeclScope);
    m_DerivativeFnScope = getCurrentScope();
    beginBlock();
    // For each function parameter variable, store its derivative value.
    for (auto param : params) {
      if (!param->getType()->isRealType())
        continue;
      // If param is independent variable, its derivative is 1, otherwise 0.
      int dValue = (param == m_IndependentVar);
      auto dParam = ConstantFolder::synthesizeLiteral(m_Context.IntTy,
                                                      m_Context, dValue);
      // For each function arg, create a variable _d_arg to store derivatives
      // of potential reassignments, e.g.:
      // double f_darg0(double x, double y) {
      //   double _d_x = 1;
      //   double _d_y = 0;
      //   ...
      auto dParamDecl = BuildVarDecl(param->getType(),
                                     "_d_" + param->getNameAsString(),
                                     dParam);
      addToCurrentBlock(BuildDeclStmt(dParamDecl));
      dParam = BuildDeclRef(dParamDecl);
      // Memorize the derivative of param, i.e. whenever the param is visited
      // in the future, it's derivative dParam is found (unless reassigned with
      // something new).
      m_Variables[param] = dParam;
    }

    Stmt* BodyDiff = Visit(FD->getBody()).getStmt();
    if (auto CS = dyn_cast<CompoundStmt>(BodyDiff))
      for (Stmt* S : CS->body())
        addToCurrentBlock(S);
    else
      addToCurrentBlock(BodyDiff);
    Stmt* derivativeBody = endBlock();
    derivedFD->setBody(derivativeBody);

    endScope(); // Function body scope
    m_Sema.PopFunctionScopeInfo();
    m_Sema.PopDeclContext();
    endScope(); // Function decl scope

    m_DerivativeInFlight = false;
    return { derivedFD, enclosingNS };
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

  Expr* VisitorBase::BuildOp(UnaryOperatorKind OpCode, Expr* E) {
    return m_Sema.BuildUnaryOp(nullptr, noLoc, OpCode, E).get(); 
  }

  Expr* VisitorBase::BuildOp(clang::BinaryOperatorKind OpCode,
                                   Expr* L, Expr* R) {
    return m_Sema.BuildBinOp(nullptr, noLoc, OpCode, L, R).get();
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

  Expr* VisitorBase::BuildArraySubscript(Expr* Base,
    const llvm::SmallVectorImpl<clang::Expr*>& Indices) {
    Expr* result = Base;
    for (Expr* I : Indices)
      result = m_Sema.CreateBuiltinArraySubscriptExpr(result, noLoc, I, noLoc).get();
    return result;
  }

  StmtDiff ForwardModeVisitor::VisitStmt(const Stmt* S) {
    diag(DiagnosticsEngine::Warning, S->getLocStart(),
         "attempted to differentiate unsupported statement, no changes applied");
    // Unknown stmt, just clone it.
    return StmtDiff(Clone(S));
  }

  StmtDiff ForwardModeVisitor::VisitCompoundStmt(const CompoundStmt* CS) {
    beginScope(Scope::DeclScope);
    beginBlock();
    for (Stmt* S : CS->body()) {
      StmtDiff SDiff = Visit(S);
      addToCurrentBlock(SDiff.getStmt_dx());
      addToCurrentBlock(SDiff.getStmt());
    }
    CompoundStmt* Result = endBlock();
    endScope();
    // Differentation of CompundStmt produces another CompoundStmt with both
    // original and derived statements, i.e. Stmt() is Result and Stmt_dx() is
    // null.
    return StmtDiff(Result);
  }

  StmtDiff ForwardModeVisitor::VisitIfStmt(const IfStmt* If) {
    // Control scope of the IfStmt. E.g., in if (double x = ...) {...}, x goes
    // to this scope.
    beginScope(Scope::DeclScope | Scope::ControlScope);
    // Create a block "around" if statement, e.g:
    // {
    //   ...
    //  if (...) {...}
    // }
    beginBlock();
    const Stmt* init = If->getInit();
    StmtDiff initResult = init ? Visit(init) : StmtDiff{};
    // If there is Init, it's derivative will be output in the block before if:
    // E.g., for:
    // if (int x = 1; ...) {...}
    // result will be:
    // {
    //   int _d_x = 0;
    //   if (int x = 1; ...) {...}
    // }
    // This is done to avoid variable names clashes.
    addToCurrentBlock(initResult.getStmt_dx());

    VarDecl* condVarClone = nullptr;
    if (VarDecl* condVarDecl = If->getConditionVariable()) {
      VarDeclDiff condVarDeclDiff = DifferentiateVarDecl(condVarDecl);
      condVarClone = condVarDeclDiff.getDecl();
      if (condVarDeclDiff.getDecl_dx())
        addToCurrentBlock(BuildDeclStmt(condVarDeclDiff.getDecl_dx()));
    }

    // Condition is just cloned as it is, not derived.
    // FIXME: if condition changes one of the variables, it may be reasonable
    // to derive it, e.g.
    // if (x += x) {...}
    // should result in:
    // {
    //   _d_y += _d_x
    //   if (y += x) {...}
    // }
    Expr* cond = Clone(If->getCond());

    auto VisitBranch =
      [this] (const Stmt* Branch) -> Stmt* {
        if (!Branch)
          return nullptr;

        if (isa<CompoundStmt>(Branch)) {
          StmtDiff BranchDiff = Visit(Branch);
          return BranchDiff.getStmt();
        } else {
          beginBlock();
          beginScope(Scope::DeclScope);
          StmtDiff BranchDiff = Visit(Branch);
          for (Stmt* S : BranchDiff.getBothStmts())
            addToCurrentBlock(S);
          CompoundStmt* Block = endBlock();
          endScope();
          if (Block->size() == 1)
            return Block->body_front();
          else
            return Block;
        }
      };

    Stmt* thenDiff = VisitBranch(If->getThen());
    Stmt* elseDiff = VisitBranch(If->getElse());

    Stmt* ifDiff = new (m_Context) IfStmt(m_Context, noLoc, If->isConstexpr(),
                                          initResult.getStmt(), condVarClone,
                                          cond, thenDiff, noLoc, elseDiff);
    addToCurrentBlock(ifDiff);
    CompoundStmt* Block = endBlock();
    // If IfStmt is the only statement in the block, remove the block:
    endScope();
    // {
    //   if (...) {...}
    // }
    // ->
    // if (...) {...}
    StmtDiff Result = (Block->size() == 1) ? StmtDiff(ifDiff) : StmtDiff(Block);
    return Result;
  }

  StmtDiff
  ForwardModeVisitor::VisitConditionalOperator(const ConditionalOperator* CO) {
    Expr* cond = Clone(CO->getCond());
    StmtDiff ifTrueDiff = Visit(CO->getTrueExpr());
    StmtDiff ifFalseDiff = Visit(CO->getFalseExpr());

    cond = StoreAndRef(cond);
    cond = m_Sema.ActOnCondition(m_CurScope, noLoc, cond,
                                 Sema::ConditionKind::Boolean).get().second;

    Expr* condExpr = m_Sema.ActOnConditionalOp(noLoc, noLoc, cond,
                                               ifTrueDiff.getExpr(),
                                               ifFalseDiff.getExpr()).get();

    Expr* condExprDiff = m_Sema.ActOnConditionalOp(noLoc, noLoc, cond,
                                                   ifTrueDiff.getExpr_dx(),
                                                   ifFalseDiff.getExpr_dx()).
                                                   get();

    return StmtDiff(condExpr, condExprDiff);
  }

  StmtDiff ForwardModeVisitor::VisitForStmt(const ForStmt* FS) {
    beginScope(Scope::DeclScope |
               Scope::ControlScope |
               Scope::BreakScope |
               Scope::ContinueScope);
    beginBlock();
    const Stmt* init = FS->getInit();
    StmtDiff initDiff = init ? Visit(init) : StmtDiff{};
    addToCurrentBlock(initDiff.getStmt_dx());
    VarDecl* condVarDecl = FS->getConditionVariable();
    VarDecl* condVarClone = nullptr;
    if (condVarDecl) {
       VarDeclDiff condVarResult = DifferentiateVarDecl(condVarDecl);
       condVarClone = condVarResult.getDecl();
       if (condVarResult.getDecl_dx())
         addToCurrentBlock(BuildDeclStmt(condVarResult.getDecl_dx()));
    }
    Expr* cond = FS->getCond() ? Clone(FS->getCond()) : nullptr;
    const Expr* inc = FS->getInc();

    // Differentiate the increment expression of the for loop
    beginBlock();
    StmtDiff incDiff = inc ? Visit(inc) : StmtDiff{};
    CompoundStmt* decls = endBlock();
    Expr* incResult = nullptr;
    if (decls->size()) {
      // If differentiation of the increment produces a statement for
      // temporary variable declaration, enclose the increment in lambda
      // since only expressions are allowed in the increment part of the for
      // loop. E.g.:
      // for (...; ...; x = x * std::sin(x))
      // ->
      // for (int i = 0; i < 10; [&] {
      //  double _t1 = std::sin(x);
      //  _d_x = _d_x * _t1 + x * custom_derivatives::sin_darg0(x) * (_d_x);
      //  x = x * _t1;
      // }())

      // FIXME: Here we make use some of the things that are used from Parser, it
      // seems to be the easiest way to create lambda
      LambdaIntroducer Intro;
      Intro.Default = LCD_ByRef;
      // FIXME: Using noLoc here results in assert failure. Any other valid
      // SourceLocation seems to work fine.
      Intro.Range.setBegin(inc->getLocStart());
      Intro.Range.setEnd(inc->getLocEnd());
      AttributeFactory AttrFactory;
      DeclSpec DS(AttrFactory);
      Declarator D(DS, Declarator::LambdaExprContext);
      m_Sema.PushLambdaScope();
      beginScope(Scope::BlockScope |
                 Scope::FnScope |
                 Scope::DeclScope);
      m_Sema.ActOnStartOfLambdaDefinition(Intro, D, getCurrentScope());
      beginBlock();
      StmtDiff incDiff = inc ? Visit(inc) : StmtDiff{};
      addToCurrentBlock(incDiff.getStmt_dx());
      addToCurrentBlock(incDiff.getStmt());
      CompoundStmt* incBody = endBlock();
      Expr* lambda = 
        m_Sema.ActOnLambdaExpr(noLoc, incBody, getCurrentScope()).get();
      endScope();
      incResult =
        m_Sema.ActOnCallExpr(getCurrentScope(),
                             lambda,
                             noLoc,
                             {},
                             noLoc).get(); 
    }
    else if (incDiff.getExpr_dx() && incDiff.getExpr()) {
      // If no declarations are required and only two Expressions are produced,
      // join them with comma expression.
      if (!isUnusedResult(incDiff.getExpr_dx()))
        incResult = BuildOp(BO_Comma, BuildParens(incDiff.getExpr_dx()),
                            BuildParens(incDiff.getExpr()));
      else
        incResult = incDiff.getExpr();
    }
    else if (incDiff.getExpr()) {
      incResult = incDiff.getExpr();
    }
    
    const Stmt* body = FS->getBody();
    beginScope(Scope::DeclScope);
    Stmt* bodyResult = nullptr;
    if (isa<CompoundStmt>(body)) {
      bodyResult = Visit(body).getStmt();
    }
    else {
      beginBlock();
      StmtDiff Result = Visit(body);
      for (Stmt* S : Result.getBothStmts())
        addToCurrentBlock(S);
      CompoundStmt* Block = endBlock();
      if (Block->size() == 1)
        bodyResult = Block->body_front();
      else
        bodyResult = Block; 
    }
    endScope();
 
    Stmt* forStmtDiff =
      new (m_Context) ForStmt(m_Context, initDiff.getStmt(), cond, condVarClone,
                              incResult, bodyResult, noLoc, noLoc, noLoc);
  
    addToCurrentBlock(forStmtDiff);
    CompoundStmt* Block = endBlock();
    endScope();

    StmtDiff Result = (Block->size() == 1) ?
      StmtDiff(forStmtDiff) : StmtDiff(Block);
    return Result;
  }

  StmtDiff ForwardModeVisitor::VisitReturnStmt(const ReturnStmt* RS) {
    StmtDiff retValDiff = Visit(RS->getRetValue());
    Stmt* returnStmt =
      m_Sema.ActOnReturnStmt(noLoc,
                             retValDiff.getExpr_dx(), // return the derivative
                             m_CurScope).get();
    return StmtDiff(returnStmt);
  }

  StmtDiff ForwardModeVisitor::VisitParenExpr(const ParenExpr* PE) {
    StmtDiff subStmtDiff = Visit(PE->getSubExpr());
    return StmtDiff(BuildParens(subStmtDiff.getExpr()),
                    BuildParens(subStmtDiff.getExpr_dx()));
  }

  StmtDiff ForwardModeVisitor::VisitMemberExpr(const MemberExpr* ME) {
    auto clonedME = dyn_cast<MemberExpr>(Clone(ME));
    // Copy paste from VisitDeclRefExpr.
    QualType Ty = ME->getType();
    if (clonedME->getMemberDecl() == m_IndependentVar)
      return StmtDiff(clonedME,
                      ConstantFolder::synthesizeLiteral(Ty, m_Context, 1));
    return StmtDiff(clonedME,
                    ConstantFolder::synthesizeLiteral(Ty, m_Context, 0));
  }

  StmtDiff ForwardModeVisitor::VisitInitListExpr(const InitListExpr* ILE) {
    llvm::SmallVector<Expr*, 16> clonedExprs(ILE->getNumInits());
    llvm::SmallVector<Expr*, 16> derivedExprs(ILE->getNumInits());
    for (unsigned i = 0, e = ILE->getNumInits(); i < e; i++) {
      StmtDiff ResultI = Visit(ILE->getInit(i));
      clonedExprs[i] = ResultI.getExpr();
      derivedExprs[i] = ResultI.getExpr_dx();
    }

    Expr* clonedILE = m_Sema.ActOnInitList(noLoc, clonedExprs, noLoc).get();
    Expr* derivedILE = m_Sema.ActOnInitList(noLoc, derivedExprs, noLoc).get();
    return StmtDiff(clonedILE, derivedILE);
  } 

  StmtDiff ForwardModeVisitor::VisitArraySubscriptExpr(const ArraySubscriptExpr* ASE) {
    auto ASI = SplitArraySubscript(ASE);
    const Expr* Base = ASI.first;
    const auto& Indices = ASI.second;
    Expr* clonedBase = Clone(Base);
    llvm::SmallVector<Expr*, 4> clonedIndices(Indices.size());
    std::transform(std::begin(Indices), std::end(Indices), std::begin(clonedIndices),
      [this](const Expr* E) { return Clone(E); });
    auto cloned = BuildArraySubscript(clonedBase, clonedIndices);

    auto zero = ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
    if (!isa<DeclRefExpr>(clonedBase->IgnoreParenImpCasts()))
      return StmtDiff(cloned, zero);
    auto DRE = cast<DeclRefExpr>(clonedBase->IgnoreParenImpCasts());
    if (!isa<VarDecl>(DRE->getDecl()))
      return StmtDiff(cloned, zero);
    auto VD = cast<VarDecl>(DRE->getDecl());
    // Check DeclRefExpr is a reference to an independent variable.
    auto it = m_Variables.find(VD);
    if (it == std::end(m_Variables))
      // Is not an independent variable, ignored.
      return StmtDiff(cloned, zero);

    Expr* target = it->second;
    // FIXME: fix when adding array inputs
    if (!target->getType()->isArrayType() && !target->getType()->isPointerType())
      return StmtDiff(cloned, zero);
    //llvm::APSInt IVal;
    //if (!I->EvaluateAsInt(IVal, m_Context))
    //  return;
    // Create the _result[idx] expression.
    auto result_at_is = BuildArraySubscript(target, clonedIndices);
    return StmtDiff(cloned, result_at_is);
  }
   
  StmtDiff ForwardModeVisitor::VisitDeclRefExpr(const DeclRefExpr* DRE) {
    DeclRefExpr* clonedDRE = nullptr;
    // Check if referenced Decl was "replaced" with another identifier inside
    // the derivative
    if (auto VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      auto it = m_DeclReplacements.find(VD);
      if (it != std::end(m_DeclReplacements))
        clonedDRE = BuildDeclRef(it->second);
      else
        clonedDRE = cast<DeclRefExpr>(Clone(DRE));
      // If current context is different than the context of the original
      // declaration (e.g. we are inside lambda), rebuild the DeclRefExpr
      // with Sema::BuildDeclRefExpr. This is required in some cases, e.g.
      // Sema::BuildDeclRefExpr is responsible for adding captured fields
      // to the underlying struct of a lambda.
      if (clonedDRE->getDecl()->getDeclContext() != m_Sema.CurContext) {
        auto referencedDecl = cast<VarDecl>(clonedDRE->getDecl());
        clonedDRE = cast<DeclRefExpr>(BuildDeclRef(referencedDecl));
      }
    } else
      clonedDRE = cast<DeclRefExpr>(Clone(DRE));
    
    if (auto VD = dyn_cast<VarDecl>(clonedDRE->getDecl())) {
      // If DRE references a variable, try to find if we know something about
      // how it is related to the independent variable.
      auto it = m_Variables.find(VD);
      if (it != std::end(m_Variables)) {
        // If a record was found, use the recorded derivative.
        auto dExpr = it->second;
        if (auto dVarDRE = dyn_cast<DeclRefExpr>(dExpr)) {
          auto dVar = cast<VarDecl>(dVarDRE->getDecl());
          if (dVar->getDeclContext() != m_Sema.CurContext)
            dExpr = BuildDeclRef(dVar);
        }
        return StmtDiff(clonedDRE, dExpr);
      }
    }
    // Is not a variable or is a reference to something unrelated to independent
    // variable. Derivative is 0.
    auto zero = ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
    return StmtDiff(clonedDRE, zero);
  }

  StmtDiff ForwardModeVisitor::VisitIntegerLiteral(const IntegerLiteral* IL) {
    llvm::APInt zero(m_Context.getIntWidth(m_Context.IntTy), /*value*/0);
    auto constant0 = IntegerLiteral::Create(m_Context, zero, m_Context.IntTy,
                                            noLoc);
    return StmtDiff(Clone(IL), constant0);
  }

  StmtDiff ForwardModeVisitor::VisitFloatingLiteral(const FloatingLiteral* FL) {
    llvm::APFloat zero = llvm::APFloat::getZero(FL->getSemantics());
    auto constant0 = FloatingLiteral::Create(m_Context, zero, true,
                                             FL->getType(), noLoc);
    return StmtDiff(Clone(FL), constant0);
  }

  // This method is derived from the source code of both
  // buildOverloadedCallSet() in SemaOverload.cpp
  // and ActOnCallExpr() in SemaExpr.cpp.
  bool DerivativeBuilder::overloadExists(Expr* UnresolvedLookup,
                                         llvm::MutableArrayRef<Expr*> ARargs) {
    if (UnresolvedLookup->getType() == m_Context.OverloadTy) {
      OverloadExpr::FindResult find = OverloadExpr::find(UnresolvedLookup);

      if (!find.HasFormOfMemberPointer) {
        OverloadExpr *ovl = find.Expression;

        if (isa<UnresolvedLookupExpr>(ovl)) {
          ExprResult result;
          SourceLocation Loc;
          OverloadCandidateSet CandidateSet(Loc,
                                            OverloadCandidateSet::CSK_Normal);
          Scope* S = m_Sema.getScopeForContext(m_Sema.CurContext);
          UnresolvedLookupExpr *ULE = cast<UnresolvedLookupExpr>(ovl);
          // Populate CandidateSet.
          m_Sema.buildOverloadedCallSet(S, UnresolvedLookup, ULE, ARargs, Loc,
                                        &CandidateSet, &result);

          OverloadCandidateSet::iterator Best;
          OverloadingResult OverloadResult =
            CandidateSet.BestViableFunction(m_Sema,
                                            UnresolvedLookup->getLocStart(),
                                            Best);
          if (OverloadResult) // No overloads were found.
            return true;
        }
      }
    }
    return false;
  }

  static NamespaceDecl* LookupBuiltinDerivativesNSD(ASTContext &C, Sema& S) {
    // Find the builtin derivatives namespace
    DeclarationName Name = &C.Idents.get("custom_derivatives");
    LookupResult R(S, Name, SourceLocation(), Sema::LookupNamespaceName,
                   Sema::ForRedeclaration);
    S.LookupQualifiedName(R, C.getTranslationUnitDecl(),
                          /*allowBuiltinCreation*/ false);
    assert(!R.empty() && "Cannot find builtin derivatives!");
    return cast<NamespaceDecl>(R.getFoundDecl());
  }

  Expr* DerivativeBuilder::findOverloadedDefinition(DeclarationNameInfo DNI,
                                       llvm::SmallVectorImpl<Expr*>& CallArgs) {
    if (!m_BuiltinDerivativesNSD)
      m_BuiltinDerivativesNSD = LookupBuiltinDerivativesNSD(m_Context, m_Sema);

    LookupResult R(m_Sema, DNI, Sema::LookupOrdinaryName);
    m_Sema.LookupQualifiedName(R, m_BuiltinDerivativesNSD,
                               /*allowBuiltinCreation*/ false);
    Expr* OverloadedFn = 0;
    if (!R.empty()) {
      CXXScopeSpec CSS;
      CSS.Extend(m_Context, m_BuiltinDerivativesNSD, noLoc, noLoc);
      Expr* UnresolvedLookup
        = m_Sema.BuildDeclarationNameExpr(CSS, R, /*ADL*/ false).get();

      llvm::MutableArrayRef<Expr*> MARargs
        = llvm::MutableArrayRef<Expr*>(CallArgs);

      SourceLocation Loc;
      Scope* S = m_Sema.getScopeForContext(m_Sema.CurContext);

      if (overloadExists(UnresolvedLookup, MARargs)) {
        return 0;
      }

      OverloadedFn = m_Sema.ActOnCallExpr(S, UnresolvedLookup, Loc,
                                          MARargs, Loc).get();
    }
    return OverloadedFn;
  }

  StmtDiff ForwardModeVisitor::VisitCallExpr(const CallExpr* CE) {
    const FunctionDecl* FD = CE->getDirectCallee();
    if (!FD) {
      diag(DiagnosticsEngine::Warning, CE->getLocStart(),
           "Differentiation of only direct calls is supported. Ignored");
      return StmtDiff(Clone(CE));
    }
    // Find the built-in derivatives namespace.
    std::string s = std::to_string(m_DerivativeOrder);
    if (m_DerivativeOrder == 1)
      s = "";
    // FIXME: add gradient-vector products to fix that.
    assert((CE->getNumArgs() <= 1) &&
           "forward differentiation of multi-arg calls is currently broken");
    IdentifierInfo* II = &m_Context.Idents.get(FD->getNameAsString() + "_d" +
                                               s + "arg0");
    DeclarationName name(II);
    SourceLocation DeclLoc;
    DeclarationNameInfo DNInfo(name, DeclLoc);

    SourceLocation noLoc;
    llvm::SmallVector<Expr*, 4> CallArgs{};
    // For f(g(x)) = f'(x) * g'(x)
    Expr* Multiplier = nullptr;
    for (size_t i = 0, e = CE->getNumArgs(); i < e; ++i) {
      StmtDiff argDiff = Visit(CE->getArg(i));
      if (!Multiplier)
        Multiplier = argDiff.getExpr_dx();
      else {
        Multiplier =
          BuildOp(BO_Add, Multiplier, argDiff.getExpr_dx());
      }
      CallArgs.push_back(argDiff.getExpr());
    }

    Expr* call = m_Sema.ActOnCallExpr(getCurrentScope(), Clone(CE->getCallee()),
                                      noLoc, llvm::MutableArrayRef<Expr*>(CallArgs),
                                      noLoc).get();

    // Try to find an overloaded derivative in 'custom_derivatives'
    Expr* callDiff = m_Builder.findOverloadedDefinition(DNInfo, CallArgs);

    // Check if it is a recursive call.
    if (!callDiff && (FD == m_Function)) {
      // The differentiated function is called recursively.
      Expr* derivativeRef =
        m_Sema.BuildDeclarationNameExpr(CXXScopeSpec(),
                                        m_Derivative->getNameInfo(),
                                        m_Derivative).get();
      callDiff =
        m_Sema.ActOnCallExpr(m_Sema.getScopeForContext(m_Sema.CurContext),
                             derivativeRef,
                             noLoc,
                             llvm::MutableArrayRef<Expr*>(CallArgs),
                             noLoc).get();
    }

    if (!callDiff) {
      // Overloaded derivative was not found, request the CladPlugin to 
      // derive the called function.
      DiffRequest request{};
      request.Function = FD;
      request.BaseFunctionName = FD->getNameAsString();
      request.Mode = DiffMode::forward;
      // Silence diag outputs in nested derivation process.
      request.VerboseDiags = false;

      FunctionDecl* derivedFD = plugin::ProcessDiffRequest(m_CladPlugin, request);
      // Clad failed to derive it.
      if (!derivedFD) {
        // Function was not derived => issue a warning.
        diag(DiagnosticsEngine::Warning, CE->getLocStart(),
             "function '%0' was not differentiated because clad failed to "
             "differentiate it and no suitable overload was found in "
             "namespace 'custom_derivatives'",
             { FD->getNameAsString() });

        auto zero = ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
        return StmtDiff(call, zero);
      }

      callDiff = m_Sema.ActOnCallExpr(getCurrentScope(), BuildDeclRef(derivedFD),
                                      noLoc, llvm::MutableArrayRef<Expr*>(CallArgs),
                                      noLoc).get();
    }
 
    if (Multiplier)
      callDiff = BuildOp(BO_Mul, callDiff, BuildParens(Multiplier));
    return StmtDiff(call, callDiff);
  }

  void VisitorBase::updateReferencesOf(Stmt* InSubtree) {
    utils::ReferencesUpdater up(m_Sema,
                                m_Builder.m_NodeCloner.get(),
                                getCurrentScope());
    up.TraverseStmt(InSubtree);
  }

  StmtDiff ForwardModeVisitor::VisitUnaryOperator(const UnaryOperator* UnOp) {
    StmtDiff diff = Visit(UnOp->getSubExpr());
    auto opKind = UnOp->getOpcode();
    Expr* op = BuildOp(opKind, diff.getExpr());
    // If opKind is unary plus or minus, apply that op to derivative.
    // Otherwise, the derivative is 0.
    // FIXME: add support for other unary operators
    if (opKind == UO_Plus || opKind == UO_Minus)
      return StmtDiff(op, BuildOp(opKind, diff.getExpr_dx()));
    else if (opKind == UO_PostInc || opKind == UO_PostDec ||
             opKind == UO_PreInc || opKind == UO_PreDec) {
      return StmtDiff(op, diff.getExpr_dx());
    }
    else {
      diag(DiagnosticsEngine::Warning, UnOp->getLocEnd(),
           "attempt to differentiate unsupported unary operator, derivative \
            set to 0");
      auto zero = ConstantFolder::synthesizeLiteral(m_Context.IntTy,
                                                    m_Context, 0);
      return StmtDiff(op, zero);
    }
  }

  StmtDiff ForwardModeVisitor::VisitBinaryOperator(const BinaryOperator* BinOp) {
    StmtDiff Ldiff = Visit(BinOp->getLHS());
    StmtDiff Rdiff = Visit(BinOp->getRHS());

    ConstantFolder folder(m_Context);
    auto opCode = BinOp->getOpcode();
    Expr* opDiff = nullptr;

    auto deriveMul = [this] (StmtDiff& Ldiff, StmtDiff& Rdiff) {
      Expr* LHS = BuildOp(BO_Mul, BuildParens(Ldiff.getExpr_dx()),
                          BuildParens(Rdiff.getExpr()));

      Expr* RHS = BuildOp(BO_Mul, BuildParens(Ldiff.getExpr()),
                          BuildParens(Rdiff.getExpr_dx()));

      return BuildOp(BO_Add, LHS, RHS);
    };

    auto deriveDiv = [this] (StmtDiff& Ldiff, StmtDiff& Rdiff) {
      Expr* LHS = BuildOp(BO_Mul, BuildParens(Ldiff.getExpr_dx()),
                          BuildParens(Rdiff.getExpr()));

      Expr* RHS = BuildOp(BO_Mul, BuildParens(Ldiff.getExpr()),
                          BuildParens(Rdiff.getExpr_dx()));

      Expr* nominator = BuildOp(BO_Sub, LHS, RHS);

      Expr* RParens = BuildParens(Rdiff.getExpr());
      Expr* denominator = BuildOp(BO_Mul, RParens, RParens);

      return BuildOp(BO_Div, BuildParens(nominator), BuildParens(denominator));
    };
        
    if (opCode == BO_Mul) {
      // If Ldiff.getExpr() and Rdiff.getExpr() require evaluation, store the
      // expressions in variables to avoid reevaluation.
      Ldiff = { StoreAndRef(Ldiff.getExpr()), Ldiff.getExpr_dx() };
      Rdiff = { StoreAndRef(Rdiff.getExpr()), Rdiff.getExpr_dx() };

      opDiff = deriveMul(Ldiff, Rdiff);
    }
    else if (opCode == BO_Div) {
      Ldiff = { StoreAndRef(Ldiff.getExpr()), Ldiff.getExpr_dx() };
      Rdiff = { StoreAndRef(Rdiff.getExpr()), Rdiff.getExpr_dx() };

      opDiff = deriveDiv(Ldiff, Rdiff);
    }
    else if (opCode == BO_Add)
      opDiff = BuildOp(BO_Add, Ldiff.getExpr_dx(), Rdiff.getExpr_dx());
    else if (opCode == BO_Sub)
      opDiff = BuildOp(BO_Sub, Ldiff.getExpr_dx(),
                       BuildParens(Rdiff.getExpr_dx()));
    else if (BinOp->isAssignmentOp()) {
      if (!Ldiff.getExpr_dx()->isGLValue()) {
        diag(DiagnosticsEngine::Warning, BinOp->getLocEnd(),
             "derivative of an assignment attempts to assign to unassignable "
             "expr, assignment ignored");
        opDiff = ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
      }
      else if (opCode == BO_Assign || opCode == BO_AddAssign ||
               opCode == BO_SubAssign)
        opDiff = BuildOp(opCode, Ldiff.getExpr_dx(), Rdiff.getExpr_dx());
      else if (opCode == BO_MulAssign) {
        Ldiff = { StoreAndRef(Ldiff.getExpr()), Ldiff.getExpr_dx() };
        Rdiff = { StoreAndRef(Rdiff.getExpr()), Rdiff.getExpr_dx() };
        opDiff = BuildOp(BO_Assign, Ldiff.getExpr_dx(), deriveMul(Ldiff, Rdiff));
      }
      else if (opCode == BO_DivAssign) {
        Ldiff = { StoreAndRef(Ldiff.getExpr()), Ldiff.getExpr_dx() };
        Rdiff = { StoreAndRef(Rdiff.getExpr()), Rdiff.getExpr_dx() };
        opDiff = BuildOp(BO_Assign, Ldiff.getExpr_dx(), deriveDiv(Ldiff, Rdiff));
      }
    }
    else if (opCode == BO_Comma) {
      if (!isUnusedResult(Ldiff.getExpr_dx()))
        opDiff = BuildOp(BO_Comma, BuildParens(Ldiff.getExpr_dx()),
                         BuildParens(Rdiff.getExpr_dx()));
      else
        opDiff = Rdiff.getExpr_dx();
    }
    if (!opDiff) {
      //FIXME: add support for other binary operators
      diag(DiagnosticsEngine::Warning, BinOp->getLocEnd(),
           "attempt to differentiate unsupported binary operator, derivative \
            set to 0");
      opDiff = ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
    }
    opDiff = folder.fold(opDiff);
    // Recover the original operation from the Ldiff and Rdiff instead of
    // cloning the tree.
    Expr* op = BuildOp(opCode, Ldiff.getExpr(), Rdiff.getExpr());

    return StmtDiff(op, opDiff);
  }

  VarDeclDiff ForwardModeVisitor::DifferentiateVarDecl(const VarDecl* VD) {
    StmtDiff initDiff = VD->getInit() ? Visit(VD->getInit()) : StmtDiff{};
    VarDecl* VDClone = BuildVarDecl(VD->getType(),
                                    VD->getNameAsString(),
                                    initDiff.getExpr(),
                                    VD->isDirectInit());
    VarDecl* VDDerived = BuildVarDecl(VD->getType(),
                                      "_d_" + VD->getNameAsString(),
                                      initDiff.getExpr_dx());
    m_Variables.emplace(VDClone, BuildDeclRef(VDDerived));
    return VarDeclDiff(VDClone, VDDerived);
  }

  StmtDiff ForwardModeVisitor::VisitDeclStmt(const DeclStmt* DS) {
    llvm::SmallVector<Decl*, 4> decls;
    llvm::SmallVector<Decl*, 4> declsDiff;
    // For each variable declaration v, create another declaration _d_v to
    // store derivatives for potential reassignments. E.g.
    // double y = x;
    // ->
    // double _d_y = _d_x; double y = x;
    for (auto D : DS->decls()) {
      if (auto VD = dyn_cast<VarDecl>(D)) {
        VarDeclDiff VDDiff = DifferentiateVarDecl(VD);
        // Check if decl's name is the same as before. The name may be changed
        // if decl name collides with something in the derivative body.
        // This can happen in rare cases, e.g. when the original function
        // has both y and _d_y (here _d_y collides with the name produced by
        // the derivation process), e.g.
        // double f(double x) {
        //   double y = x;
        //   double _d_y = x;
        // } 
        // ->
        // double f_darg0(double x) {
        //   double _d_x = 1;
        //   double _d_y = _d_x; // produced as a derivative for y
        //   double y = x;
        //   double _d__d_y = _d_x;
        //   double _d_y = x; // copied from original funcion, collides with _d_y
        // }
        if (VDDiff.getDecl()->getDeclName() != VD->getDeclName())
          m_DeclReplacements[VD] = VDDiff.getDecl();
        decls.push_back(VDDiff.getDecl());
        declsDiff.push_back(VDDiff.getDecl_dx());
      } else {
        diag(DiagnosticsEngine::Warning, D->getLocEnd(),
             "Unsupported declaration");
      }
    }

    Stmt* DSClone = BuildDeclStmt(decls);
    Stmt* DSDiff = BuildDeclStmt(declsDiff);
    return StmtDiff(DSClone, DSDiff);
  }

  StmtDiff
  ForwardModeVisitor::VisitImplicitCastExpr(const ImplicitCastExpr* ICE) {
    StmtDiff subExprDiff = Visit(ICE->getSubExpr());
    // Casts should be handled automatically when the result is used by
    // Sema::ActOn.../Build...
    return StmtDiff(subExprDiff.getExpr(), subExprDiff.getExpr_dx());
  }

  StmtDiff
  ForwardModeVisitor::
  VisitCXXOperatorCallExpr(const CXXOperatorCallExpr* OpCall) {
    // This operator gets emitted when there is a binary operation containing
    // overloaded operators. Eg. x+y, where operator+ is overloaded.
    diag(DiagnosticsEngine::Error, OpCall->getLocEnd(),
         "We don't support overloaded operators yet!");
    return {};
  }

  StmtDiff ForwardModeVisitor::VisitCXXDefaultArgExpr(const CXXDefaultArgExpr* DE) {
    return Visit(DE->getExpr());
  }
  
  StmtDiff ForwardModeVisitor::VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr* BL) {
    llvm::APInt zero(m_Context.getIntWidth(m_Context.IntTy), /*value*/0);
    auto constant0 = IntegerLiteral::Create(m_Context, zero, m_Context.IntTy,
                                            noLoc);
    return StmtDiff(Clone(BL), constant0);
  }

  ReverseModeVisitor::ReverseModeVisitor(DerivativeBuilder& builder):
    VisitorBase(builder) {}

  ReverseModeVisitor::~ReverseModeVisitor() {}

  DeclWithContext ReverseModeVisitor::Derive(const FunctionDecl* FD,
                                             const DiffRequest& request) {
    silenceDiags = !request.VerboseDiags;
    m_Function = FD;
    assert(m_Function && "Must not be null.");

    DiffParams args {};
    if (request.Args)
      args = parseDiffArgs(request.Args, FD);
    else
      std::copy(FD->param_begin(), FD->param_end(), std::back_inserter(args));      
    if (args.empty())
      return {};
    auto derivativeBaseName = m_Function->getNameAsString();
    std::string gradientName = derivativeBaseName + "_grad";
    // To be consistent with older tests, nothing is appended to 'f_grad' if 
    // we differentiate w.r.t. all the parameters at once.
    if (!std::equal(FD->param_begin(), FD->param_end(), std::begin(args)))
      for (auto arg : args) {
        auto it = std::find(FD->param_begin(), FD->param_end(), arg);
        auto idx = std::distance(FD->param_begin(), it);
        gradientName += ('_' + std::to_string(idx));
      }
    IdentifierInfo* II = &m_Context.Idents.get(gradientName);
    DeclarationNameInfo name(II, noLoc);

    // A vector of types of the gradient function parameters.
    llvm::SmallVector<QualType, 16> paramTypes(m_Function->getNumParams() + 1);
    std::transform(m_Function->param_begin(),
                   m_Function->param_end(),
                   std::begin(paramTypes),
                   [] (const ParmVarDecl* PVD) {
                     return PVD->getType();
                   });
    // The last parameter is the output parameter of the R* type.
    paramTypes.back() = m_Context.getPointerType(m_Function->getReturnType());
    // For a function f of type R(A1, A2, ..., An),
    // the type of the gradient function is void(A1, A2, ..., An, R*).
    QualType gradientFunctionType =
      m_Context.getFunctionType(m_Context.VoidTy,
                                llvm::ArrayRef<QualType>(paramTypes.data(),
                                                         paramTypes.size()),
                                // Cast to function pointer.
                                FunctionProtoType::ExtProtoInfo());

    // Create the gradient function declaration.
    FunctionDecl* gradientFD = nullptr;
    NamespaceDecl* enclosingNS = nullptr;
    llvm::SaveAndRestore<DeclContext*> SaveContext(m_Sema.CurContext);
    llvm::SaveAndRestore<Scope*> SaveScope(m_CurScope);
    DeclContext* DC = const_cast<DeclContext*>(m_Function->getDeclContext());
    m_Sema.CurContext = DC;
    if (isa<CXXMethodDecl>(m_Function)) {
      CXXRecordDecl* CXXRD = cast<CXXRecordDecl>(DC);
      gradientFD = CXXMethodDecl::Create(m_Context,
                                         CXXRD,
                                         noLoc,
                                         name,
                                         gradientFunctionType,
                                         m_Function->getTypeSourceInfo(),
                                         m_Function->getStorageClass(),
                                         m_Function->isInlineSpecified(),
                                         m_Function->isConstexpr(),
                                         noLoc);
      gradientFD->setAccess(m_Function->getAccess());
    }
    else if (isa<FunctionDecl>(m_Function)) {
      enclosingNS = RebuildEnclosingNamespaces(DC);
      gradientFD = FunctionDecl::Create(m_Context, m_Sema.CurContext, noLoc,
                                        name, gradientFunctionType,
                                        m_Function->getTypeSourceInfo(),
                                        m_Function->getStorageClass(),
                                        m_Function->isInlineSpecified(),
                                        m_Function->hasWrittenPrototype(),
                                        m_Function->isConstexpr());
    } else {
      diag(DiagnosticsEngine::Error, m_Function->getLocEnd(),
           "attempted differentiation of '%0' which is of unsupported type",
           { m_Function->getNameAsString() });
      return {};
    }
    m_Derivative = gradientFD;

    // Function declaration scope
    beginScope(Scope::FunctionPrototypeScope |
               Scope::FunctionDeclarationScope |
               Scope::DeclScope);
    m_Sema.PushFunctionScope();
    m_Sema.PushDeclContext(getCurrentScope(), m_Derivative);

    // Create parameter declarations.
    llvm::SmallVector<ParmVarDecl*, 4> params(paramTypes.size());
    std::transform(m_Function->param_begin(), m_Function->param_end(), 
      std::begin(params),
      [&] (const ParmVarDecl* PVD) {
        auto VD = ParmVarDecl::Create(m_Context, gradientFD, noLoc, noLoc,
                                      PVD->getIdentifier(), PVD->getType(),
                                      PVD->getTypeSourceInfo(),
                                      PVD->getStorageClass(),
                                      // Clone default arg if present.
                                      (PVD->hasDefaultArg() ?
                                        Clone(PVD->getDefaultArg()) : nullptr));
        if (VD->getIdentifier())
          m_Sema.PushOnScopeChains(VD, getCurrentScope(), /*AddToContext*/ false);
        auto it = std::find(std::begin(args), std::end(args), PVD);
        if (it != std::end(args))
          *it = VD;
        return VD;
    });
    // The output paremeter "_result".
    params.back() = ParmVarDecl::Create(m_Context, gradientFD, noLoc,
                                        noLoc, &m_Context.Idents.get("_result"),
                                        paramTypes.back(),
                                        m_Context.getTrivialTypeSourceInfo(
                                          paramTypes.back(), noLoc),
                                        params.front()->getStorageClass(),
                                        /* No default value */ nullptr);
    if (params.back()->getIdentifier())
      m_Sema.PushOnScopeChains(params.back(), getCurrentScope(),
                               /*AddToContext*/ false);

    llvm::ArrayRef<ParmVarDecl*> paramsRef = llvm::makeArrayRef(params.data(),
                                                                params.size());
    gradientFD->setParams(paramsRef);
    gradientFD->setBody(nullptr);

    // Reference to the output parameter.
    m_Result = BuildDeclRef(params.back());

    auto idx = 0;
    for (auto arg : args) {
      // FIXME: fix when adding array inputs, now we are just skipping all
      // array/pointer inputs (not treating them as independent variables).
      if (arg->getType()->isArrayType() || arg->getType()->isPointerType()) {
        idx += 1;
        continue;
      }
      auto size_type = m_Context.getSizeType();
      auto size_type_bits = m_Context.getIntWidth(size_type);
      // Create the idx literal.
      auto i = IntegerLiteral::Create(m_Context, llvm::APInt(size_type_bits, idx),
                                      size_type, noLoc);
      // Create the _result[idx] expression.
      auto result_at_i = m_Sema.CreateBuiltinArraySubscriptExpr(m_Result, noLoc,
                                                                i, noLoc).get();
      m_Variables[arg] = result_at_i;
      idx += 1;
    }

    // Function body scope.
    beginScope(Scope::FnScope | Scope::DeclScope);
    m_DerivativeFnScope = getCurrentScope();
    beginBlock();
    // Start the visitation process which outputs the statements in the current
    // block.
    StmtDiff BodyDiff = Visit(FD->getBody());
    Stmt* Forward = BodyDiff.getStmt();
    Stmt* Reverse = BodyDiff.getStmt_dx();
    // Create the body of the function.
    // Firstly, all "global" Stmts are put into fn's body.
    for (Stmt* S : m_Globals)
      addToCurrentBlock(S, forward);
    // Forward pass.
    if (auto CS = dyn_cast<CompoundStmt>(Forward))
      for (Stmt* S : CS->body())
        addToCurrentBlock(S, forward);
    else
      addToCurrentBlock(Forward, forward);
    // Reverse pass.
    if (auto RCS = dyn_cast<CompoundStmt>(Reverse))
      for (Stmt* S : RCS->body())
        addToCurrentBlock(S, forward);
    else
      addToCurrentBlock(Reverse, forward);
    Stmt* gradientBody = endBlock();
    m_Derivative->setBody(gradientBody);

    endScope(); // Function body scope
    m_Sema.PopFunctionScopeInfo();
    m_Sema.PopDeclContext();
    endScope(); // Function decl scope

    return { gradientFD, enclosingNS };
  }
  
  StmtDiff ReverseModeVisitor::VisitStmt(const Stmt* S) {
    diag(DiagnosticsEngine::Warning, S->getLocStart(),
         "attempted to differentiate unsupported statement, no changes applied");
    // Unknown stmt, just clone it.
    return StmtDiff(Clone(S));
  }

  StmtDiff ReverseModeVisitor::VisitCompoundStmt(const CompoundStmt* CS) {
    beginScope(Scope::DeclScope);
    beginBlock(forward);
    beginBlock(reverse);
    for (Stmt* S : CS->body()) {
      StmtDiff SDiff = DifferentiateSingleStmt(S);
      addToCurrentBlock(SDiff.getStmt(), forward);
      addToCurrentBlock(SDiff.getStmt_dx(), reverse);
    }
    CompoundStmt* Forward = endBlock(forward);
    CompoundStmt* Reverse = endBlock(reverse);
    endScope();
    return StmtDiff(Forward, Reverse);
  }

  StmtDiff ReverseModeVisitor::VisitIfStmt(const clang::IfStmt* If) {
    // Control scope of the IfStmt. E.g., in if (double x = ...) {...}, x goes
    // to this scope.
    beginScope(Scope::DeclScope | Scope::ControlScope);
    Expr* cond = Clone(If->getCond());
    // Condition has to be stored as a "global" variable, to take the correct
    // branch in the reverse pass.
    cond = GlobalStoreAndRef(cond, "_cond");
    // Create a block "around" if statement, e.g:
    // {
    //   ...
    //  if (...) {...}
    // }
    beginBlock(forward);
    beginBlock(reverse);
    const Stmt* init = If->getInit();
    StmtDiff initResult = init ? Visit(init) : StmtDiff{};
    // If there is Init, it's derivative will be output in the block before if:
    // E.g., for:
    // if (int x = 1; ...) {...}
    // result will be:
    // {
    //   int _d_x = 0;
    //   if (int x = 1; ...) {...}
    // }
    // This is done to avoid variable names clashes.
    addToCurrentBlock(initResult.getStmt_dx());

    VarDecl* condVarClone = nullptr;
    if (VarDecl* condVarDecl = If->getConditionVariable()) {
      VarDeclDiff condVarDeclDiff = DifferentiateVarDecl(condVarDecl);
      condVarClone = condVarDeclDiff.getDecl();
      if (condVarDeclDiff.getDecl_dx())
        addToBlock(BuildDeclStmt(condVarDeclDiff.getDecl_dx()), m_Globals);
    }

    // Condition is just cloned as it is, not derived.
    // FIXME: if condition changes one of the variables, it may be reasonable
    // to derive it, e.g.
    // if (x += x) {...}
    // should result in:
    // {
    //   _d_y += _d_x
    //   if (y += x) {...}
    // }

    auto unwrapIfSingleStmt = [] (Stmt* S) -> Stmt* {
      if (!isa<CompoundStmt>(S))
        return S;
      auto CS = cast<CompoundStmt>(S);
      if (CS->size() == 1)
        return CS->body_front();
      else
        return CS;
    };

    auto VisitBranch =
      [&] (const Stmt* Branch) -> StmtDiff {
        if (!Branch)
          return {};
        if (isa<CompoundStmt>(Branch)) {
          StmtDiff BranchDiff = Visit(Branch);
          return BranchDiff;
        } else {
          beginBlock(forward);
          StmtDiff BranchDiff = DifferentiateSingleStmt(Branch);
          addToCurrentBlock(BranchDiff.getStmt(), forward);
          Stmt* Forward = unwrapIfSingleStmt(endBlock(forward));
          Stmt* Reverse = unwrapIfSingleStmt(BranchDiff.getStmt_dx());
          return StmtDiff(Forward, Reverse);
        }
      };

    StmtDiff thenDiff = VisitBranch(If->getThen());
    StmtDiff elseDiff = VisitBranch(If->getElse());

    // It is problematic to specify both condVarDecl and cond thorugh 
    // Sema::ActOnIfStmt, therefore we directly use the IfStmt constructor.
    cond = m_Sema.ActOnCondition(m_CurScope, noLoc, cond,
                                 Sema::ConditionKind::Boolean).get().second;
    Stmt* Forward = new (m_Context) IfStmt(m_Context, noLoc, If->isConstexpr(),
                                           initResult.getStmt(), condVarClone,
                                           cond, thenDiff.getStmt(), noLoc,
                                           elseDiff.getStmt());
    Stmt* Reverse = new (m_Context) IfStmt(m_Context, noLoc, If->isConstexpr(),
                                           initResult.getStmt_dx(), condVarClone,
                                           cond, thenDiff.getStmt_dx(), noLoc,
                                           elseDiff.getStmt_dx());
    addToCurrentBlock(Forward, forward);
    CompoundStmt* ForwardBlock = endBlock(forward);
    addToCurrentBlock(Reverse, reverse);
    CompoundStmt* ReverseBlock = endBlock(reverse);
    endScope();
    return StmtDiff(unwrapIfSingleStmt(ForwardBlock),
                    unwrapIfSingleStmt(ReverseBlock));
  }

  StmtDiff ReverseModeVisitor::VisitConditionalOperator(
    const clang::ConditionalOperator* CO) {
    Expr* cond = Clone(CO->getCond());
    // Condition has to be stored as a "global" variable, to take the correct
    // branch in the reverse pass.
    cond = GlobalStoreAndRef(cond, "_cond");
    cond = m_Sema.ActOnCondition(m_CurScope, noLoc, cond,
                                 Sema::ConditionKind::Boolean).get().second;

    auto ifTrue = CO->getTrueExpr();
    auto ifFalse = CO->getFalseExpr();

    auto VisitBranch =
      [&] (Stmt* branch, Expr* ifTrue, Expr* ifFalse) {
        if (!branch)
          return StmtDiff{};
        auto condExpr = m_Sema.ActOnConditionalOp(noLoc, noLoc, cond, ifTrue,
                                                  ifFalse).get();

        auto dStmt = BuildParens(condExpr);
        return Visit(branch, dStmt);
    };

    auto zero = ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);

    StmtDiff ifTrueDiff = VisitBranch(ifTrue, dfdx(), zero);
    StmtDiff ifFalseDiff = VisitBranch(ifFalse, zero, dfdx());

    Expr* condExpr = m_Sema.ActOnConditionalOp(noLoc, noLoc, cond,
                                               ifTrueDiff.getExpr(),
                                               ifFalseDiff.getExpr()).get();
    // If result is a glvalue, we should keep it as it can potentially be assigned
    // as in (c ? a : b) = x;
    if (ifTrueDiff.getExpr_dx() && ifFalseDiff.getExpr_dx() &&
        ifTrueDiff.getExpr_dx()->isGLValue() &&
        ifFalseDiff.getExpr_dx()->isGLValue() && CO->isGLValue()) {
      Expr* ResultRef = m_Sema.ActOnConditionalOp(noLoc, noLoc, cond,
                                                  ifTrueDiff.getExpr_dx(),
                                                  ifFalseDiff.getExpr_dx()).get();
      return StmtDiff(condExpr, ResultRef);
    }
    return StmtDiff(condExpr);
  }

  StmtDiff ReverseModeVisitor::VisitCXXDefaultArgExpr(const CXXDefaultArgExpr* DE) {
    return Visit(DE->getExpr());
  }

  StmtDiff ReverseModeVisitor::VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr* BL) {
    return Clone(BL);
  }

  StmtDiff ReverseModeVisitor::VisitReturnStmt(const ReturnStmt* RS) {
    // Initially, df/df = 1.
    const Expr* value = RS->getRetValue();
    QualType type = value->getType();
    auto dfdf = ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 1);
    ExprResult tmp = dfdf;
    dfdf = m_Sema.ImpCastExprToType(tmp.get(), type,
                                    m_Sema.PrepareScalarCast(tmp, type)).get();
    StmtDiff ReturnDiff = DifferentiateSingleStmt(value, dfdf);
    Stmt* Reverse = ReturnDiff.getStmt_dx();
    // If the original function returns at this point, some part of the reverse
    // pass (corresponding to other branches that do not return here) must be 
    // skipped. We create a label in the reverse pass and jump to it via goto.
    LabelDecl* LD = LabelDecl::Create(m_Context, m_Sema.CurContext, noLoc,
                                      CreateUniqueIdentifier("_label"));
    m_Sema.PushOnScopeChains(LD, m_DerivativeFnScope, true);
    // Attach label to the last Stmt in the corresponding Reverse Stmt.
    if (!Reverse)
      Reverse = m_Sema.ActOnNullStmt(noLoc).get();
    Stmt* LS = m_Sema.ActOnLabelStmt(noLoc, LD, noLoc, Reverse).get();
    addToCurrentBlock(LS, reverse);
    // addToCurrentBlock(ReturnDiff.getStmt(), forward);
    // Create goto to the label.
    return m_Sema.ActOnGotoStmt(noLoc, noLoc, LD).get();
  }

  StmtDiff ReverseModeVisitor::VisitParenExpr(const ParenExpr* PE) {
    StmtDiff subStmtDiff = Visit(PE->getSubExpr(), dfdx());
    return StmtDiff(BuildParens(subStmtDiff.getExpr()),
                    BuildParens(subStmtDiff.getExpr_dx()));
  }

  StmtDiff ReverseModeVisitor::VisitInitListExpr(const InitListExpr* ILE) {
    llvm::SmallVector<Expr*, 16> clonedExprs(ILE->getNumInits());
    for (unsigned i = 0, e = ILE->getNumInits(); i < e; i++) {
      Expr* I = ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, i);
      Expr* array_at_i = m_Sema.CreateBuiltinArraySubscriptExpr(dfdx(), noLoc,
                                                               I, noLoc).get();
      Expr* clonedEI = Visit(ILE->getInit(i), array_at_i).getExpr();
      clonedExprs[i] = clonedEI;
    }

    Expr* clonedILE = m_Sema.ActOnInitList(noLoc, clonedExprs, noLoc).get();
    return StmtDiff(clonedILE);
  } 

  StmtDiff ReverseModeVisitor::VisitArraySubscriptExpr(const ArraySubscriptExpr* ASE) {
    auto ASI = SplitArraySubscript(ASE);
    const Expr* Base = ASI.first;
    const auto& Indices = ASI.second;
    Expr* clonedBase = Clone(Base);
    llvm::SmallVector<Expr*, 4> clonedIndices(Indices.size());
    std::transform(std::begin(Indices), std::end(Indices), std::begin(clonedIndices),
      [this](const Expr* E) { return Clone(E); });
    auto cloned = BuildArraySubscript(clonedBase, clonedIndices);
    
    if (!isa<DeclRefExpr>(clonedBase->IgnoreParenImpCasts()))
      return StmtDiff(cloned);
    auto DRE = cast<DeclRefExpr>(clonedBase->IgnoreParenImpCasts());
    if (!isa<VarDecl>(DRE->getDecl()))
      return StmtDiff(cloned);
    auto VD = cast<VarDecl>(DRE->getDecl());
    // Check DeclRefExpr is a reference to an independent variable.
    auto it = m_Variables.find(VD);
    Expr* target = nullptr;
    if (it == std::end(m_Variables)) {
      // FIXME: implement proper detection
      if (VD->getName() == "p")
        target = m_Result;
      else
      // Is not an independent variable, ignored.
        return StmtDiff(cloned);
    } else 
      target = it->second;

    Expr* result = nullptr;
    if (!target->getType()->isArrayType() && !target->getType()->isPointerType())
      result = target;
    else
      // Create the _result[idx] expression.
      result = BuildArraySubscript(target, clonedIndices);
    // Create the (target += dfdx) statement.
    if (dfdx()) {
      auto add_assign = BuildOp(BO_AddAssign, result, dfdx());
      // Add it to the body statements.
      addToCurrentBlock(add_assign, reverse);
    }
    return StmtDiff(cloned, result);
  }    

  StmtDiff ReverseModeVisitor::VisitDeclRefExpr(const DeclRefExpr* DRE) {
    DeclRefExpr* clonedDRE = nullptr;
    // Check if referenced Decl was "replaced" with another identifier inside
    // the derivative
    if (auto VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      auto it = m_DeclReplacements.find(VD);
      if (it != std::end(m_DeclReplacements))
        clonedDRE = BuildDeclRef(it->second);
      else
        clonedDRE = cast<DeclRefExpr>(Clone(DRE));
      // If current context is different than the context of the original
      // declaration (e.g. we are inside lambda), rebuild the DeclRefExpr
      // with Sema::BuildDeclRefExpr. This is required in some cases, e.g.
      // Sema::BuildDeclRefExpr is responsible for adding captured fields
      // to the underlying struct of a lambda.
      if (clonedDRE->getDecl()->getDeclContext() != m_Sema.CurContext) {
        auto referencedDecl = cast<VarDecl>(clonedDRE->getDecl());
        clonedDRE = cast<DeclRefExpr>(BuildDeclRef(referencedDecl));
      }
    } else
      clonedDRE = cast<DeclRefExpr>(Clone(DRE));

    if (auto decl = dyn_cast<VarDecl>(clonedDRE->getDecl())) {
      // Check DeclRefExpr is a reference to an independent variable.
      auto it = m_Variables.find(decl);
      if (it == std::end(m_Variables)) {
        // Is not an independent variable, ignored.
        return StmtDiff(clonedDRE);
      }
      // Create the (_result[idx] += dfdx) statement.
      if (dfdx()) {
        auto add_assign = BuildOp(BO_AddAssign, it->second, dfdx());
        // Add it to the body statements.
        addToCurrentBlock(add_assign, reverse);
      }
      return StmtDiff(clonedDRE, it->second);
    }
    
    return StmtDiff(clonedDRE);
  }

  StmtDiff ReverseModeVisitor::VisitIntegerLiteral(const IntegerLiteral* IL) {
    return StmtDiff(Clone(IL));
  }

  StmtDiff ReverseModeVisitor::VisitFloatingLiteral(const FloatingLiteral* FL) {
    return StmtDiff(Clone(FL));
  }

  StmtDiff ReverseModeVisitor::VisitCallExpr(const CallExpr* CE) {
    const FunctionDecl* FD = CE->getDirectCallee();
    if (!FD) {
      diag(DiagnosticsEngine::Warning, CE->getLocEnd(),
           "Differentiation of only direct calls is supported. Ignored");
      return StmtDiff(Clone(CE));
    }

    auto NArgs = FD->getNumParams();
    // If the function has no args then we assume that it is not related
    // to independent variables and does not contribute to gradient.
    if (!NArgs)
      return StmtDiff(Clone(CE));

    llvm::SmallVector<Expr*, 16> CallArgs{};
    // If the result does not depend on the result of the call, just clone
    // the call and visit arguments (since they may contain side-effects like
    // f(x = y))
    if (!dfdx()) {
      for (const Expr* Arg : CE->arguments()) {
        StmtDiff ArgDiff = Visit(Arg);
        CallArgs.push_back(ArgDiff.getExpr());
      }
      Expr* call = m_Sema.ActOnCallExpr(getCurrentScope(), Clone(CE->getCallee()),
                                      noLoc, llvm::MutableArrayRef<Expr*>(CallArgs),
                                      noLoc).get();
      return call;
    }

    llvm::SmallVector<VarDecl*, 16> ArgResultDecls{};
    // Save current index in the current block, to potentially put some statements
    // there later.
    std::size_t insertionPoint = getCurrentBlock(reverse).size();
    for (const Expr* Arg : CE->arguments()) {
      // Create temporary variables corresponding to derivative of each argument,
      // so that they can be reffered to when arguments is visited. Variables
      // will be initialized later after arguments is visited. This is done to
      // reduce cloning complexity and only clone once.
      Expr* dArg = StoreAndRef(nullptr, Arg->getType(), reverse, "_r", /*force*/true);
      ArgResultDecls.push_back(cast<VarDecl>(cast<DeclRefExpr>(dArg)->getDecl()));
      // Visit using unitialized reference.
      StmtDiff ArgDiff = Visit(Arg, dArg);
      // Save cloned arg in a "global" variable, so that it is accesible from the
      // reverse pass.
      CallArgs.push_back(GlobalStoreAndRef(ArgDiff.getExpr()));
    }

    VarDecl* ResultDecl = nullptr;
    Expr* Result = nullptr;
    Expr* OverloadedDerivedFn = nullptr;
    // If the function has a single arg, we look for a derivative w.r.t. to
    // this arg (it is unlikely that we need gradient of a one-dimensional'
    // function).
    bool asGrad = true;
    if (NArgs == 1) {
      IdentifierInfo* II = &m_Context.Idents.get(FD->getNameAsString() + "_darg0");
      // Try to find it in builtin derivatives
      DeclarationName name(II);
      DeclarationNameInfo DNInfo(name, noLoc);
      OverloadedDerivedFn = m_Builder.findOverloadedDefinition(DNInfo, CallArgs);
      if (OverloadedDerivedFn)
        asGrad = false;
    }
    // If it has more args or f_darg0 was not found, we look for its gradient.
    if (!OverloadedDerivedFn) {
      IdentifierInfo* II = &m_Context.Idents.get(FD->getNameAsString() + "_grad");
      // We also need to create an array to store the result of gradient call.
      auto size_type_bits = m_Context.getIntWidth(m_Context.getSizeType());
      auto ArrayType =
        m_Context.getConstantArrayType(CE->getType(),
                                       llvm::APInt(size_type_bits, NArgs),
                                       ArrayType::ArraySizeModifier::Normal,
                                       0); // No IndexTypeQualifiers

      // Create {} array initializer to fill it with zeroes.
      auto ZeroInitBraces = m_Sema.ActOnInitList(noLoc, {}, noLoc).get();
      // Declare: Type _gradX[Nargs] = {};
      ResultDecl = BuildVarDecl(ArrayType, CreateUniqueIdentifier("_grad"),
                                ZeroInitBraces);
      Result = BuildDeclRef(ResultDecl);
      // Pass the array as the last parameter for gradient.
      CallArgs.push_back(Result);

      // Try to find it in builtin derivatives
      DeclarationName name(II);
      DeclarationNameInfo DNInfo(name, noLoc);
      OverloadedDerivedFn = m_Builder.findOverloadedDefinition(DNInfo, CallArgs);
    }
    // Derivative was not found, check if it is a recursive call
    if (!OverloadedDerivedFn) {
      if (FD == m_Function) {
        // Recursive call.
        auto selfRef = m_Sema.BuildDeclarationNameExpr(CXXScopeSpec(),
                                                       m_Derivative->getNameInfo(),
                                                       m_Derivative).get();

        OverloadedDerivedFn = m_Sema.ActOnCallExpr(getCurrentScope(), selfRef,
                                                   noLoc,
                                                   llvm::MutableArrayRef<Expr*>(
                                                     CallArgs),
                                                   noLoc).get();
      } else {
        // Overloaded derivative was not found, request the CladPlugin to 
        // derive the called function.
        DiffRequest request{};
        request.Function = FD;
        request.BaseFunctionName = FD->getNameAsString();
        request.Mode = DiffMode::reverse;
        // Silence diag outputs in nested derivation process.
        request.VerboseDiags = false;

        FunctionDecl* derivedFD = plugin::ProcessDiffRequest(m_CladPlugin, request);
        // Clad failed to derive it.
        if (!derivedFD) {
          // Function was not derived => issue a warning.
         diag(DiagnosticsEngine::Warning, CE->getLocStart(),
              "function '%0' was not differentiated because clad failed to "
              "differentiate it and no suitable overload was found in "
              "namespace 'custom_derivatives'",
              { FD->getNameAsString() });
         return StmtDiff(Clone(CE));
        }
        OverloadedDerivedFn = m_Sema.ActOnCallExpr(getCurrentScope(),
                                                   BuildDeclRef(derivedFD),
                                                   noLoc,
                                                   llvm::MutableArrayRef<Expr*>(
                                                     CallArgs),
                                                   noLoc).get();
      }
    }

    if (OverloadedDerivedFn) {
      // Derivative was found.
      if (!asGrad) {
        // If the derivative is called through _darg0 instead of _grad.
        Expr* d = BuildOp(BO_Mul, dfdx(), OverloadedDerivedFn);
        ArgResultDecls[0]->setInit(d);
      } else {
        // Put Result array declaration in the function body.
        // Call the gradient, passing Result as the last Arg.
        auto& block = getCurrentBlock(reverse);
        auto it = std::next(std::begin(block), insertionPoint);
        // Insert Result array declaration and gradient call to the block at
        // the saved point.
        block.insert(it, { BuildDeclStmt(ResultDecl), OverloadedDerivedFn });
        // Visit each arg with df/dargi = df/dxi * Result[i].
        for (unsigned i = 0; i < CE->getNumArgs(); i++) {
          auto size_type = m_Context.getSizeType();
          auto size_type_bits = m_Context.getIntWidth(size_type);
          // Create the idx literal.
          auto I = IntegerLiteral::Create(m_Context,
                                          llvm::APInt(size_type_bits, i),
                                          size_type, noLoc);
          // Create the Result[I] expression.
          auto ithResult = m_Sema.CreateBuiltinArraySubscriptExpr(Result, noLoc,
                                                                  I, noLoc).get();
          auto di = BuildOp(BO_Mul, dfdx(), ithResult);
          ArgResultDecls[i]->setInit(di);
        }
      }
    }

    // If additional _result parameter was added, pop it.
    if (asGrad)
      CallArgs.pop_back();
    // Re-clone function arguments again, since they are required at 2 places:
    // call to gradient and call to original function.
    // At this point, each arg is either a simple expression or a reference
    // to a temporary variable. Therefore cloning it has constant complexity.
    std::transform(std::begin(CallArgs), std::end(CallArgs), std::begin(CallArgs),
     [this] (Expr* E) { return Clone(E); });
    // Recreate the original call expression.
    Expr* call = m_Sema.ActOnCallExpr(getCurrentScope(), Clone(CE->getCallee()),
                                      noLoc, llvm::MutableArrayRef<Expr*>(CallArgs),
                                      noLoc).get();
    return StmtDiff(call);
  }

  StmtDiff ReverseModeVisitor::VisitUnaryOperator(const UnaryOperator* UnOp) {
    auto opCode  = UnOp->getOpcode();
    StmtDiff diff{};
    // If it is a post-increment/decrement operator, its result is a reference and 
    // we should return it.
    Expr* ResultRef = nullptr;
    if (opCode == UO_Plus)
      //xi = +xj
      //dxi/dxj = +1.0
      //df/dxj += df/dxi * dxi/dxj = df/dxi
      diff = Visit(UnOp->getSubExpr(), dfdx());
    else if (opCode == UO_Minus) {
      //xi = -xj
      //dxi/dxj = -1.0
      //df/dxj += df/dxi * dxi/dxj = -df/dxi
      auto d = BuildOp(UO_Minus, dfdx());
      diff = Visit(UnOp->getSubExpr(), d);
    }
    else if (opCode == UO_PostInc || opCode == UO_PostDec) {
      diff = Visit(UnOp->getSubExpr(), dfdx());
      ResultRef = diff.getExpr_dx();
    }
    else if (opCode == UO_PreInc || opCode == UO_PreDec) {
      diff = Visit(UnOp->getSubExpr(), dfdx());
    }
    else {
      diag(DiagnosticsEngine::Warning, UnOp->getLocEnd(),
           "attempt to differentiate unsupported unary operator, ignored");
      return Clone(UnOp);
    }
    Expr* op = BuildOp(opCode, diff.getExpr());
    return StmtDiff(op, ResultRef);
  }

  StmtDiff ReverseModeVisitor::VisitBinaryOperator(const BinaryOperator* BinOp) {
    auto opCode = BinOp->getOpcode();
    StmtDiff Ldiff{};
    StmtDiff Rdiff{};
    auto L = BinOp->getLHS();
    auto R = BinOp->getRHS();
    // If it is an assignment operator, its result is a reference to LHS and 
    // we should return it.
    Expr* ResultRef = nullptr;

    if (opCode == BO_Add) {
      //xi = xl + xr
      //dxi/xl = 1.0
      //df/dxl += df/dxi * dxi/xl = df/dxi
      Ldiff = Visit(L, dfdx());
      //dxi/xr = 1.0
      //df/dxr += df/dxi * dxi/xr = df/dxi
      Rdiff = Visit(R, dfdx());
    }
    else if (opCode == BO_Sub) {
      //xi = xl - xr
      //dxi/xl = 1.0
      //df/dxl += df/dxi * dxi/xl = df/dxi
      Ldiff = Visit(L, dfdx());
      //dxi/xr = -1.0
      //df/dxl += df/dxi * dxi/xr = -df/dxi
      auto dr = BuildOp(UO_Minus, dfdx());
      Rdiff = Visit(R, dr);
    }
    else if (opCode == BO_Mul) {
      //xi = xl * xr
      //dxi/xl = xr
      //df/dxl += df/dxi * dxi/xl = df/dxi * xr
      // Create uninitialized "global" variable for the right multiplier.
      // It will be assigned later after R is visited and cloned. This allows
      // to reduce cloning complexity and only clones once. Storing it in a 
      // global variable allows to save current result and make it accessible
      // in the reverse pass.
      Expr* RStored = GlobalStoreAndRef(nullptr, R->getType());
      Expr* dl = nullptr;
      if (dfdx()) {
        dl = BuildOp(BO_Mul, dfdx(), RStored);
        dl = StoreAndRef(dl, reverse);
      }
      Ldiff = Visit(L, dl);
      //dxi/xr = xl
      //df/dxr += df/dxi * dxi/xr = df/dxi * xl
      // Store left multiplier and assign it with L.
      Expr* LStored = GlobalStoreAndRef(Ldiff.getExpr());
      Expr* dr = nullptr;
      if (dfdx()) {
        dr = BuildOp(BO_Mul, LStored, dfdx());
        dr = StoreAndRef(dr, reverse);
      }
      Rdiff = Visit(R, dr);
      // Assign right multiplier's variable with R.
      addToCurrentBlock(BuildOp(BO_Assign, RStored, Rdiff.getExpr()), forward);
      std::tie(Ldiff, Rdiff) = std::make_pair(LStored, RStored);
    }
    else if (opCode == BO_Div) {
      //xi = xl / xr
      //dxi/xl = 1 / xr
      //df/dxl += df/dxi * dxi/xl = df/dxi * (1/xr)
      Expr* RStored = GlobalStoreAndRef(nullptr, R->getType());
      Expr* dl = nullptr;
      if (dfdx()) {
        dl = BuildOp(BO_Div, dfdx(), RStored);
        dl = StoreAndRef(dl, reverse);
      }
      Ldiff = Visit(L, dl);
      //dxi/xr = -xl / (xr * xr)
      //df/dxl += df/dxi * dxi/xr = df/dxi * (-xl /(xr * xr))
      // Wrap R * R in parentheses: (R * R). otherwise code like 1 / R * R is
      // produced instead of 1 / (R * R).
      Expr* LStored = GlobalStoreAndRef(Ldiff.getExpr());
      Expr* dr = nullptr;
      if (dfdx()) {
        Expr* RxR = m_Sema.ActOnParenExpr(noLoc, noLoc,
                                          BuildOp(BO_Mul, RStored, RStored)).get();
        dr = BuildOp(BO_Mul, dfdx(),
                           BuildOp(UO_Minus, BuildOp(BO_Div, LStored, RxR)));
        dr = StoreAndRef(dr, reverse);
      }
      Rdiff = Visit(R, dr);
      addToCurrentBlock(BuildOp(BO_Assign, RStored, Rdiff.getExpr()), forward);
      std::tie(Ldiff, Rdiff) = std::make_pair(LStored, RStored);
    }
    else if (BinOp->isAssignmentOp()) {
      if (!L->isGLValue()) {
        diag(DiagnosticsEngine::Warning, BinOp->getLocEnd(),
             "derivative of an assignment attempts to assign to unassignable "
             "expr, assignment ignored");
        return Clone(BinOp);
      }
      // Visit LHS, but delay emission of its derivative statements, save them
      // in Lblock
      beginBlock(reverse);
      Ldiff = Visit(L, dfdx());
      auto Lblock = endBlock(reverse);
      Expr* LCloned = Ldiff.getExpr();
      // For x, AssignedDiff is _d_x, for x[i] its _d_x[i], for reference exprs
      // like (x = y) it propagates recursively, so _d_x is also returned.
      Expr* AssignedDiff = Ldiff.getExpr_dx();
      if (!AssignedDiff)
        return Clone(BinOp);
      ResultRef = AssignedDiff;
      // If assigned expr is dependent, first update its derivative;
      if (Lblock->body_back())
        addToCurrentBlock(Lblock->body_back(), reverse);
      // Save old value for the derivative of LHS, to avoid problems with cases
      // like x = x.
      auto oldValue = StoreAndRef(AssignedDiff, reverse, "_r_d", /*force*/ true);
      if (opCode == BO_Assign) {
        Rdiff = Visit(R, oldValue);
      }
      else if (opCode == BO_AddAssign) {
        addToCurrentBlock(BuildOp(BO_AddAssign, AssignedDiff, oldValue), reverse);
        Rdiff = Visit(R, oldValue);
      }
      else if (opCode == BO_SubAssign) {
        addToCurrentBlock(BuildOp(BO_AddAssign, AssignedDiff, oldValue), reverse);
        Rdiff = Visit(R, BuildOp(UO_Minus, oldValue));
      }
      else if (opCode == BO_MulAssign) {
        Expr* RStored = GlobalStoreAndRef(nullptr, R->getType());
        addToCurrentBlock(BuildOp(BO_AddAssign, AssignedDiff,
                                  BuildOp(BO_Mul, oldValue, RStored)), reverse);
        // Create a reference variable to keep the result of LHS, since it must
        // be used on 2 places: when storing to a global variable accessible from
        // the reverse pass, and when rebuilding the original expression for the
        // forward pass. This allows to avoid executing same expression with
        // side effects twice. E.g., on
        //   double r = (x *= y) *= z;
        // instead of:
        //   _t0 = (x *= y);
        //   double r = (x *= y) *= z;
        // which modifies x twice, we get:
        //   double & _ref0 = (x *= y);
        //   _t0 = _ref0;
        //   double r = _ref0 *= z;
        QualType RefType = m_Context.getLValueReferenceType(L->getType());
        Expr* LRef = StoreAndRef(LCloned, RefType, forward, "_ref", /*force*/ true);
        Expr* LStored = GlobalStoreAndRef(LRef);
        Expr* dr = BuildOp(BO_Mul, LStored, oldValue);
        dr = StoreAndRef(dr, reverse);
        Rdiff = Visit(R, dr);
        addToCurrentBlock(BuildOp(BO_Assign, RStored, Rdiff.getExpr()), forward);
        std::tie(Ldiff, Rdiff) = std::make_pair(LRef, RStored);
      }
      else if (opCode == BO_DivAssign) {
        Expr* RStored = GlobalStoreAndRef(nullptr, R->getType());
        addToCurrentBlock(BuildOp(BO_AddAssign, AssignedDiff,
                                  BuildOp(BO_Div, oldValue, RStored)), reverse);
        QualType RefType = m_Context.getLValueReferenceType(L->getType());
        Expr* LRef = StoreAndRef(LCloned, RefType, forward, "_ref", /*force*/ true);
        Expr* LStored = GlobalStoreAndRef(LRef);
        Expr* RxR = m_Sema.ActOnParenExpr(noLoc, noLoc, BuildOp(BO_Mul, RStored,
                                                                RStored)).get();
        Expr* dr = BuildOp(BO_Mul, oldValue, BuildOp(UO_Minus,
                                                     BuildOp(BO_Div, LStored, RxR)));
        dr = StoreAndRef(dr, reverse);
        Rdiff = Visit(R, dr);
        addToCurrentBlock(BuildOp(BO_Assign, RStored, Rdiff.getExpr()), forward);
        std::tie(Ldiff, Rdiff) = std::make_pair(LRef, RStored);
      }
      else
        llvm_unreachable("unknown assignment opCode");
      // Update the derivative.
      addToCurrentBlock(BuildOp(BO_SubAssign, AssignedDiff, oldValue), reverse);
      // Output statements from Visit(L).
      auto begin = Lblock->body_rbegin();
      auto end = Lblock->body_rend();
      if (begin != end)
        for (auto it = std::next(begin); it != end; ++it)
          addToCurrentBlock(*it, reverse);
    }
    else if (opCode == BO_Comma) {
      auto zero = ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
      Ldiff = Visit(L, zero);
      Rdiff = Visit(R, dfdx());
      ResultRef = Ldiff.getExpr();
    }
    else {
      diag(DiagnosticsEngine::Warning, BinOp->getLocEnd(),
           "attempt to differentiate unsupported binary operator, ignored");
      return Clone(BinOp);
    }
    Expr* op = BuildOp(opCode, Ldiff.getExpr(), Rdiff.getExpr());
    return StmtDiff(op, ResultRef);
  }

  VarDeclDiff ReverseModeVisitor::DifferentiateVarDecl(const VarDecl* VD) {
    auto zero = getZeroInit(VD->getType());
    VarDecl* VDDerived = BuildVarDecl(VD->getType(),
                                      "_d_" + VD->getNameAsString(), zero);
    StmtDiff initDiff = VD->getInit() ?
                          Visit(VD->getInit(), BuildDeclRef(VDDerived)) :
                          StmtDiff{};
    VarDecl* VDClone = BuildVarDecl(VD->getType(),VD->getNameAsString(),
                                    initDiff.getExpr(), VD->isDirectInit());
    m_Variables.emplace(VDClone, BuildDeclRef(VDDerived));
    return VarDeclDiff(VDClone, VDDerived);
  }

  StmtDiff ReverseModeVisitor::DifferentiateSingleStmt(const Stmt* S,
                                                       Expr* expr) {
    beginBlock(reverse);
    StmtDiff SDiff = Visit(S, expr);
    addToCurrentBlock(SDiff.getStmt_dx(), reverse);
    CompoundStmt* RCS = endBlock(reverse);
    Stmt* ReverseResult = nullptr;
    if (RCS->body_empty())
      ReverseResult = nullptr;
    else if (RCS->size() == 1)
      ReverseResult = RCS->body_front();
    else {
      std::reverse(RCS->body_begin(), RCS->body_end());
      ReverseResult = RCS;
    }
    return StmtDiff(SDiff.getStmt(), ReverseResult);
  }

  StmtDiff ReverseModeVisitor::VisitDeclStmt(const DeclStmt* DS) {
    llvm::SmallVector<Decl*, 4> decls;
    llvm::SmallVector<Decl*, 4> declsDiff;
    // For each variable declaration v, create another declaration _d_v to
    // store derivatives for potential reassignments. E.g.
    // double y = x;
    // ->
    // double _d_y = _d_x; double y = x;
    for (auto D : DS->decls()) {
      if (auto VD = dyn_cast<VarDecl>(D)) {
        VarDeclDiff VDDiff = DifferentiateVarDecl(VD);
        // Check if decl's name is the same as before. The name may be changed
        // if decl name collides with something in the derivative body.
        // This can happen in rare cases, e.g. when the original function
        // has both y and _d_y (here _d_y collides with the name produced by
        // the derivation process), e.g.
        // double f(double x) {
        //   double y = x;
        //   double _d_y = x;
        // } 
        // ->
        // double f_darg0(double x) {
        //   double _d_x = 1;
        //   double _d_y = _d_x; // produced as a derivative for y
        //   double y = x;
        //   double _d__d_y = _d_x;
        //   double _d_y = x; // copied from original funcion, collides with _d_y
        // }
        if (VDDiff.getDecl()->getDeclName() != VD->getDeclName())
          m_DeclReplacements[VD] = VDDiff.getDecl();
        decls.push_back(VDDiff.getDecl());
        declsDiff.push_back(VDDiff.getDecl_dx());
      } else {
        diag(DiagnosticsEngine::Warning, D->getLocEnd(),
             "Unsupported declaration");
      }
    }

    Stmt* DSClone = BuildDeclStmt(decls);
    Stmt* DSDiff = BuildDeclStmt(declsDiff);
    addToBlock(DSDiff, m_Globals);
    return StmtDiff(DSClone);
  }

  StmtDiff ReverseModeVisitor::VisitImplicitCastExpr(const ImplicitCastExpr* ICE) {
    StmtDiff subExprDiff = Visit(ICE->getSubExpr(), dfdx());
    // Casts should be handled automatically when the result is used by
    // Sema::ActOn.../Build...
    return StmtDiff(subExprDiff.getExpr(), subExprDiff.getExpr_dx());
  }

  StmtDiff ReverseModeVisitor::VisitMemberExpr(const MemberExpr* ME) {
    // We do not treat struct members as independent variables, so they are not
    // differentiated.
    return StmtDiff(Clone(ME));
  }

  Expr* ReverseModeVisitor::GlobalStoreAndRef(Expr* E, QualType Type,
                                              llvm::StringRef prefix) {
    // Save current scope and temporarily go to topmost function scope.
    llvm::SaveAndRestore<Scope*> SaveScope(m_CurScope);
    assert(m_DerivativeFnScope && "must be set");
    m_CurScope = m_DerivativeFnScope;

    VarDecl* Var = BuildVarDecl(Type, CreateUniqueIdentifier(prefix));

    // Add the declaration to the body of the gradient function.
    addToBlock(BuildDeclStmt(Var), m_Globals);
    Expr* Ref = BuildDeclRef(Var);
    if (E) {
      Expr* Set = BuildOp(BO_Assign, Ref, E);
      addToCurrentBlock(Set, forward);
    }

    // Return reference to the declaration instead of original expression.
    return Ref;
  }
    
  Expr* ReverseModeVisitor::GlobalStoreAndRef(Expr* E, llvm::StringRef prefix) {
    assert(E && "cannot infer type");
    return GlobalStoreAndRef(E, E->getType(), prefix);
  }

} // end namespace clad
