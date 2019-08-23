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
#include "clang/AST/TemplateBase.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Template.h"

#include "llvm/Support/SaveAndRestore.h"

#include <algorithm>
#include <numeric>

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
    } else if (request.Mode == DiffMode::hessian) {
      HessianModeVisitor H(*this);
      result = H.Derive(FD, request);
    } else {
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
                                     bool DirectInit,
                                     TypeSourceInfo* TSI) {

    auto VD = VarDecl::Create(m_Context,
                              m_Sema.CurContext,
                              noLoc,
                              noLoc,
                              Identifier,
                              Type,
                              TSI,
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
                                     bool DirectInit,
                                     TypeSourceInfo* TSI) {
    return BuildVarDecl(Type, CreateUniqueIdentifier(prefix), Init, DirectInit, TSI);
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
    QualType Type = E->getType();
    if (E->isModifiableLvalue(m_Context) == Expr::MLV_Valid)
      Type = m_Context.getLValueReferenceType(Type);
    return StoreAndRef(E, Type, block, prefix, forceDeclCreation);
  }

  /// For an expr E, decides if it is useful to store it in a temporary variable
  /// and replace E's further usage by a reference to that variable to avoid
  /// recomputiation.
  static bool UsefulToStore(Expr* E) {
    if (!E)
      return false;
    Expr* B = E->IgnoreParenImpCasts();
    // FIXME: find a more general way to determine that or add more options.
    if (isa<DeclRefExpr>(B) || isa<FloatingLiteral>(B) || isa<IntegerLiteral>(B))
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

  HessianModeVisitor::HessianModeVisitor(DerivativeBuilder& builder):
  VisitorBase(builder) {}

  HessianModeVisitor::~HessianModeVisitor() {}

  DeclWithContext HessianModeVisitor::Derive(const clang::FunctionDecl* FD,
                                             const DiffRequest& request) {
    DiffParams args {};
    if (request.Args)
      args = parseDiffArgs(request.Args, FD);
    else
      std::copy(FD->param_begin(), FD->param_end(), std::back_inserter(args));

    std::vector<FunctionDecl*> secondDerivativeColumns;

    // Ascertains the independent arguments and differentiates the function
    // in forward and reverse mode by calling ProcessDiffRequest twice each
    // iteration, storing each generated second derivative function
    // (corresponds to columns of Hessian matrix) in a vector for private method
    // merge.
    for (auto independentArg : args) {
      DiffRequest independentArgRequest = request;
      // Converts an independent argument from VarDecl to a StringLiteral Expr
      QualType CharTyConst = m_Context.CharTy.withConst();
      QualType StrTy =
        m_Context.getConstantArrayType(CharTyConst,
                             llvm::APInt(32,
                               independentArg->getNameAsString().size() + 1),
                             ArrayType::Normal,
                             /*IndexTypeQuals*/0);
      StringLiteral* independentArgString =
          StringLiteral::Create(m_Context,
                                independentArg->getName(),
                                StringLiteral::Ascii, false, StrTy, noLoc);

      // Derives function once in forward mode w.r.t to independentArg
      independentArgRequest.Args = independentArgString;
      independentArgRequest.Mode = DiffMode::forward;
      independentArgRequest.CallUpdateRequired = false;
      FunctionDecl* firstDerivative =
                          plugin::ProcessDiffRequest(m_CladPlugin,
                                                     independentArgRequest);

      // Further derives function w.r.t to all args in reverse mode
      independentArgRequest.Mode = DiffMode::reverse;
      independentArgRequest.Function = firstDerivative;
      independentArgRequest.Args = nullptr;
      FunctionDecl* secondDerivative =
                plugin::ProcessDiffRequest(m_CladPlugin,
                                           independentArgRequest);

      secondDerivativeColumns.push_back(secondDerivative);
    }
    return Merge(secondDerivativeColumns, request);
  }

  // Combines all generated second derivative functions into a
  // single hessian function by creating CallExprs to each individual
  // secon derivative function in FunctionBody.
  DeclWithContext HessianModeVisitor::Merge(std::vector<FunctionDecl*>
                                            secDerivFuncs,
                                            const DiffRequest& request) {
    DiffParams args;
    // request.Function is original function passed in from clad::hessian
    m_Function = request.Function;
    std::copy(m_Function->param_begin(), m_Function->param_end(),
            std::back_inserter(args));

    std::string hessianFuncName = request.BaseFunctionName + "_hessian";
    IdentifierInfo* II = &m_Context.Idents.get(hessianFuncName);
    DeclarationNameInfo name(II, noLoc);

    llvm::SmallVector<QualType, 16> paramTypes(m_Function->getNumParams()
                                              + 1);

    std::transform(m_Function->param_begin(),
                   m_Function->param_end(),
                   std::begin(paramTypes),
                   [] (const ParmVarDecl* PVD) {
                     return PVD->getType();
                   });

    paramTypes.back() =
                m_Context.getPointerType(m_Function->getReturnType());

    QualType hessianFunctionType =
                      m_Context.getFunctionType(m_Context.VoidTy,
                        llvm::ArrayRef<QualType>(paramTypes.data(),
                        paramTypes.size()),
                        // Cast to function pointer.
                        FunctionProtoType::ExtProtoInfo());

    // Create the gradient function declaration.
    DeclContext* DC = const_cast<DeclContext*>(m_Function->getDeclContext());

    FunctionDecl* hessianFD = nullptr;
    NamespaceDecl* enclosingNS = nullptr;
    llvm::SaveAndRestore<DeclContext*> SaveContext(m_Sema.CurContext);
    llvm::SaveAndRestore<Scope*> SaveScope(m_CurScope);
    m_Sema.CurContext = DC;

    if (isa<CXXMethodDecl>(m_Function)) {
      CXXRecordDecl* CXXRD = cast<CXXRecordDecl>(DC);
      hessianFD = CXXMethodDecl::Create(m_Context,
                                        CXXRD,
                                        noLoc,
                                        name,
                                        hessianFunctionType,
                                        m_Function->getTypeSourceInfo(),
                                        m_Function->getStorageClass(),
                                        m_Function->isInlineSpecified(),
                                        m_Function->isConstexpr(),
                                        noLoc);
    }
    else if (isa<FunctionDecl>(m_Function)) {
      enclosingNS = RebuildEnclosingNamespaces(DC);
      hessianFD = FunctionDecl::Create(m_Context, m_Sema.CurContext, noLoc,
        name, hessianFunctionType,
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

    beginScope(Scope::FunctionPrototypeScope |
               Scope::FunctionDeclarationScope |
               Scope::DeclScope);
    m_Sema.PushFunctionScope();
    m_Sema.PushDeclContext(getCurrentScope(), hessianFD);

    llvm::SmallVector<ParmVarDecl*, 4> params(paramTypes.size());
    std::transform(m_Function->param_begin(), m_Function->param_end(),
      std::begin(params),
      [&] (const ParmVarDecl* PVD) {
        auto VD = ParmVarDecl::Create(m_Context, hessianFD, noLoc, noLoc,
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
    params.back() = ParmVarDecl::Create(m_Context, hessianFD, noLoc,
                                  noLoc, &m_Context.Idents.get("hessianMatrix"),
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
    hessianFD->setParams(paramsRef);
    Expr* m_Result = BuildDeclRef(params.back());
    std::vector<Stmt*> CompStmtSave;

    beginScope(Scope::FnScope | Scope::DeclScope);
    m_DerivativeFnScope = getCurrentScope();

    // Creates callExprs to the second derivative functions genereated
    // and creates maps array elements to input array.
    for (size_t i = 0, e = secDerivFuncs.size(); i < e; ++i)
    {
      Expr* exprFunc = BuildDeclRef(secDerivFuncs[i]);
      const int numIndependentArgs = secDerivFuncs[i]->getNumParams();

      auto size_type = m_Context.getSizeType();
      auto size_type_bits = m_Context.getIntWidth(size_type);
      // Create the idx literal.
      auto idx = IntegerLiteral::Create(m_Context, llvm::APInt(size_type_bits,
                                        (i * (numIndependentArgs-1))),
                                        size_type, noLoc);
      // Create the hessianMatrix[idx] expression.
      auto arrayExpr = m_Sema.CreateBuiltinArraySubscriptExpr(m_Result, noLoc,
                idx, noLoc).get();
      // Creates the &hessianMatrix[idx] expression.
      auto addressArrayExpr = m_Sema.BuildUnaryOp(nullptr, noLoc, UO_AddrOf,
                                                  arrayExpr).get();

      // Transforms ParmVarDecls into Expr paramters for insertion into function
      std::vector<Expr*> DeclRefToParams;
      DeclRefToParams.resize(params.size());
      std::transform(params.begin(), std::prev(params.end()),
      std::begin(DeclRefToParams),
        [&] (ParmVarDecl* PVD) {
          auto VD = BuildDeclRef(PVD);
          return VD;
        });
      DeclRefToParams.pop_back();
      DeclRefToParams.push_back(addressArrayExpr);

      Expr* call =
                  m_Sema.ActOnCallExpr(getCurrentScope(), exprFunc, noLoc,
                           llvm::MutableArrayRef<Expr*>(DeclRefToParams),
                           noLoc).get();
      CompStmtSave.push_back(call);
    }

    auto StmtsRef = llvm::makeArrayRef(CompStmtSave.data(),
                                     CompStmtSave.size());
    CompoundStmt* CS = new (m_Context) clang::CompoundStmt(m_Context,
                                                           StmtsRef,
                                                           noLoc,
                                                           noLoc);
    hessianFD->setBody(CS);
    endScope(); // Function body scope
    m_Sema.PopFunctionScopeInfo();
    m_Sema.PopDeclContext();
    endScope(); // Function decl scope

    return {hessianFD, enclosingNS};
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

  NamespaceDecl* VisitorBase::GetCladNamespace() {
    static NamespaceDecl* Result = nullptr;
    if (Result)
      return Result;
    DeclarationName CladName = &m_Context.Idents.get("clad");
    LookupResult CladR(m_Sema, CladName, noLoc, Sema::LookupNamespaceName,
                       Sema::ForRedeclaration);
    m_Sema.LookupQualifiedName(CladR, m_Context.getTranslationUnitDecl());
    assert(!CladR.empty() && "cannot find clad namespace");
    Result = cast<NamespaceDecl>(CladR.getFoundDecl());
    return Result;
  }

  TemplateDecl* VisitorBase::GetCladTapeDecl() {
    static TemplateDecl* Result = nullptr;
    if (Result)
      return Result;
    NamespaceDecl* CladNS = GetCladNamespace();
    CXXScopeSpec CSS;
    CSS.Extend(m_Context, CladNS, noLoc, noLoc);
    DeclarationName TapeName = &m_Context.Idents.get("tape");
    LookupResult TapeR(m_Sema, TapeName, noLoc, Sema::LookupUsingDeclName,
                       Sema::ForRedeclaration);
    m_Sema.LookupQualifiedName(TapeR, CladNS, CSS);
    assert(!TapeR.empty() && isa<TemplateDecl>(TapeR.getFoundDecl()) &&
           "cannot find clad::tape");
    Result = cast<TemplateDecl>(TapeR.getFoundDecl());
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
    // Get declaration of clad::tape template.
    TemplateDecl* CladTapeDecl = GetCladTapeDecl();
    // Create a list of template arguments: single argument <T> in that case.
    TemplateArgument TA = T;
    TemplateArgumentListInfo TLI{};
    TLI.addArgument(TemplateArgumentLoc(TA, m_Context.CreateTypeSourceInfo(T)));
    // This will instantiate tape<T> type and return it.
    QualType TT = m_Sema.CheckTemplateIdType(TemplateName(CladTapeDecl), noLoc,
                                             TLI);
    // Get clad napespace and its identifier clad::.
    IdentifierInfo* CladId = GetCladNamespace()->getIdentifier();
    NestedNameSpecifier* NS = NestedNameSpecifier::Create(m_Context, CladId);
    // Create elaborated type with namespace specifier, i.e. tape<T> -> clad::tape<T>
    return m_Context.getElaboratedType(ETK_None, NS, TT);
  }

  Expr* ReverseModeVisitor::CladTapeResult::Last() {
    LookupResult& Back = V.GetCladTapeBack();
    CXXScopeSpec CSS;
    CSS.Extend(V.m_Context, V.GetCladNamespace(), noLoc, noLoc);
    Expr* BackDRE = V.m_Sema.BuildDeclarationNameExpr(CSS, Back,
                                                      /*ADL*/ false).get();
    Expr* Call = V.m_Sema.ActOnCallExpr(V.getCurrentScope(), BackDRE, noLoc, Ref,
                                        noLoc).get();
    return Call;
  }

  ReverseModeVisitor::CladTapeResult ReverseModeVisitor::MakeCladTapeFor(Expr* E) {
    assert(E && "must be provided");
    QualType TapeType = GetCladTapeOfType(E->getType());
    LookupResult& Push = GetCladTapePush();
    LookupResult& Pop = GetCladTapePop();
    Expr* TapeRef = BuildDeclRef(GlobalStoreImpl(TapeType, "_t"));
    auto VD = cast<VarDecl>(cast<DeclRefExpr>(TapeRef)->getDecl());
    // Add fake location, since Clang AST does assert(Loc.isValid()) somewhere.
    VD->setLocation(m_Function->getLocation());
    m_Sema.AddInitializerToDecl(VD, getZeroInit(TapeType), false);
    CXXScopeSpec CSS;
    CSS.Extend(m_Context, GetCladNamespace(), noLoc, noLoc);
    auto PopDRE = m_Sema.BuildDeclarationNameExpr(CSS, Pop, /*ADL*/ false).get();
    auto PushDRE = m_Sema.BuildDeclarationNameExpr(CSS, Push, /*ADL*/ false).get();
    Expr* PopExpr = m_Sema.ActOnCallExpr(getCurrentScope(), PopDRE, noLoc, TapeRef,
                                         noLoc).get();
    Expr* CallArgs[] = { TapeRef, E };
    Expr* PushExpr = m_Sema.ActOnCallExpr(getCurrentScope(), PushDRE, noLoc,
                                          CallArgs, noLoc).get();
    return CladTapeResult{*this, PushExpr, PopExpr, TapeRef};
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
    // FIXME: fix potential side-effects from evaluating both sides of conditional.
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

  // A function used to wrap result of visiting E in a lambda. Returns a call
  // to the built lambda. Func is a functor that will be invoked inside lambda
  // scope and block. Statements inside lambda are expected to be added by
  // addToCurrentBlock from func invocation.
  template <typename F>
  static Expr* wrapInLambda(VisitorBase& V, Sema& S, const Expr* E, F&& func) {
    // FIXME: Here we use some of the things that are used from Parser, it
    // seems to be the easiest way to create lambda.
    LambdaIntroducer Intro;
    Intro.Default = LCD_ByRef;
    // FIXME: Using noLoc here results in assert failure. Any other valid
    // SourceLocation seems to work fine.
    Intro.Range.setBegin(E->getLocStart());
    Intro.Range.setEnd(E->getLocEnd());
    AttributeFactory AttrFactory;
    DeclSpec DS(AttrFactory);
    Declarator D(DS, Declarator::LambdaExprContext);
    S.PushLambdaScope();
    V.beginScope(Scope::BlockScope | Scope::FnScope | Scope::DeclScope);
    S.ActOnStartOfLambdaDefinition(Intro, D, V.getCurrentScope());
    V.beginBlock();
    func();
    CompoundStmt* body = V.endBlock();
    Expr* lambda = S.ActOnLambdaExpr(noLoc, body, V.getCurrentScope()).get();
    V.endScope();
    return S.ActOnCallExpr(V.getCurrentScope(), lambda, noLoc, {}, noLoc).get();
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

      incResult = wrapInLambda(*this, m_Sema, inc,
        [&] {
          StmtDiff incDiff = inc ? Visit(inc) : StmtDiff{};
          addToCurrentBlock(incDiff.getStmt_dx());
          addToCurrentBlock(incDiff.getStmt());
        });
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
      if (Ldiff.getExpr_dx()->isModifiableLvalue(m_Context) != Expr::MLV_Valid) {
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
    VisitorBase(builder), m_Result(nullptr) {}

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

    if (request.Mode == DiffMode::jacobian) {
      isJacobianCalc = true;
      args.pop_back();
    }

    auto derivativeBaseName = m_Function->getNameAsString();
    std::string gradientName = derivativeBaseName + "_grad";
    // To be consistent with older tests, nothing is appended to 'f_grad' if
    // we differentiate w.r.t. all the parameters at once.
    if (request.Mode == DiffMode::jacobian && !std::equal(FD->param_begin(), FD->param_end(), std::begin(args))) {
      for (auto arg : args) {
        auto it = std::find(FD->param_begin(), FD->param_end(), arg);
        auto idx = std::distance(FD->param_begin(), it);
        gradientName += ('_' + std::to_string(idx));
      }
    } else if (!std::equal(FD->param_begin(), FD->param_end(), std::begin(args)))
      for (auto arg : args) {
        auto it = std::find(FD->param_begin(), FD->param_end(), arg);
        auto idx = std::distance(FD->param_begin(), it);
        gradientName += ('_' + std::to_string(idx));
      }
    IdentifierInfo* II = &m_Context.Idents.get(gradientName);
    DeclarationNameInfo name(II, noLoc);

    // A vector of types of the gradient function parameters.
    llvm::SmallVector<QualType, 16> paramTypes(m_Function->getNumParams() + 1);
    if (request.Mode == DiffMode::jacobian) {
      // paramTypes.resize(m_Function->getNumParams());
      outputArrayStr = m_Function->getParamDecl((m_Function->getNumParams() - 1))->getNameAsString();
      // printf("%s\n", outputArrayStr.c_str());
    }
    std::transform(m_Function->param_begin(),
                   m_Function->param_end(),
                   std::begin(paramTypes),
                   [] (const ParmVarDecl* PVD) {
                     return PVD->getType();
                   });

    if (request.Mode == DiffMode::jacobian) {
      paramTypes.back() = m_Function->getParamDecl((m_Function->getNumParams() - 1))->getOriginalType();
    } else {
      paramTypes.back() = m_Context.getPointerType(m_Function->getReturnType());
    }
    // if (request.Mode != DiffMode::jacobian) {
      // The last parameter is the output parameter of the R* type.
      
    // }
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
    
    // Turns output array dimension input into APSInt
    auto PVDTotalArgs = m_Function->getParamDecl((m_Function->getNumParams() - 1));
    auto VD = ParmVarDecl::Create(m_Context, gradientFD, noLoc, noLoc,
                                  PVDTotalArgs->getIdentifier(), PVDTotalArgs->getType(),
                                  PVDTotalArgs->getTypeSourceInfo(),
                                  PVDTotalArgs->getStorageClass(),
                                  // Clone default arg if present.
                                  (PVDTotalArgs->hasDefaultArg() ?
                                    Clone(PVDTotalArgs->getDefaultArg()) : nullptr));
    auto DRETotalArgs = (Expr*) BuildDeclRef(VD);
    llvm::APSInt Result;
    DRETotalArgs->EvaluateAsInt(Result, m_Context);
    numParams = args.size();
    
    // Creates the ArraySubscriptExprs for the independent variables
    auto idx = 0;
    for (auto arg : args) {
      arg->dump();
      // FIXME: fix when adding array inputs, now we are just skipping all
      // array/pointer inputs (not treating them as independent variables).
      if (arg->getType()->isArrayType() || arg->getType()->isPointerType()) {
        if (arg->getName() == "p")
          m_Variables[arg] = m_Result;
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
      // VarDecl* argTemp = arg;
      m_IndependentVars.push_back(arg);
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

  static Stmt* unwrapIfSingleStmt(Stmt* S) {
    if (!S)
      return nullptr;
    if (!isa<CompoundStmt>(S))
      return S;
    auto CS = cast<CompoundStmt>(S);
    if (CS->size() == 0)
      return nullptr;
    else if (CS->size() == 1)
      return CS->body_front();
    else
      return CS;
  }

  StmtDiff ReverseModeVisitor::VisitIfStmt(const clang::IfStmt* If) {
    // Control scope of the IfStmt. E.g., in if (double x = ...) {...}, x goes
    // to this scope.
    beginScope(Scope::DeclScope | Scope::ControlScope);

    StmtDiff cond = Clone(If->getCond());
    // Condition has to be stored as a "global" variable, to take the correct
    // branch in the reverse pass.
    // If we are inside loop, the condition has to be stored in a stack after
    // the if statement.
    Expr* PushCond = nullptr;
    Expr* PopCond = nullptr;
    if (isInsideLoop) {
      // If we are inside for loop, cond will be stored in the following way:
      // forward:
      // _t = cond;
      // if (_t) { ... }
      // clad::push(..., _t);
      // reverse:
      // if (clad::pop(...)) { ... }
      // Simply doing
      // if (clad::push(..., _t) { ... }
      // is incorrect when if contains return statement inside: return will
      // skip corresponding push.
      cond = StoreAndRef(cond.getExpr(), forward, "_t", /*force*/ true);
      StmtDiff condPushPop = GlobalStoreAndRef(cond.getExpr(), "_cond");
      PushCond = condPushPop.getExpr();
      PopCond = condPushPop.getExpr_dx();
    }
    else
      cond = GlobalStoreAndRef(cond.getExpr(), "_cond");
    // Convert cond to boolean condition. We are modifying each Stmt in StmtDiff.
    for (Stmt*& S : cond.getBothStmts())
      if (S)
        S = m_Sema.ActOnCondition(m_CurScope, noLoc, cast<Expr>(S),
                                  Sema::ConditionKind::Boolean).get().second;

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
    Stmt* Forward = new (m_Context) IfStmt(m_Context, noLoc, If->isConstexpr(),
                                           initResult.getStmt(), condVarClone,
                                           cond.getExpr(), thenDiff.getStmt(),
                                           noLoc, elseDiff.getStmt());
    addToCurrentBlock(Forward, forward);

    Expr* reverseCond = cond.getExpr_dx();
    if (isInsideLoop) {
      addToCurrentBlock(PushCond, forward);
      reverseCond = PopCond;
    }
    Stmt* Reverse = new (m_Context) IfStmt(m_Context, noLoc, If->isConstexpr(),
                                           initResult.getStmt_dx(), condVarClone,
                                           reverseCond, thenDiff.getStmt_dx(),
                                           noLoc, elseDiff.getStmt_dx());
    addToCurrentBlock(Reverse, reverse);
    CompoundStmt* ForwardBlock = endBlock(forward);
    CompoundStmt* ReverseBlock = endBlock(reverse);
    endScope();
    return StmtDiff(unwrapIfSingleStmt(ForwardBlock),
                    unwrapIfSingleStmt(ReverseBlock));
  }

  StmtDiff ReverseModeVisitor::VisitConditionalOperator(
    const clang::ConditionalOperator* CO) {
    StmtDiff cond = Clone(CO->getCond());
    // Condition has to be stored as a "global" variable, to take the correct
    // branch in the reverse pass.
    cond = GlobalStoreAndRef(cond.getExpr(), "_cond");
    // Convert cond to boolean condition. We are modifying each Stmt in StmtDiff.
    for (Stmt*& S : cond.getBothStmts())
      S = m_Sema.ActOnCondition(m_CurScope, noLoc, cast<Expr>(S),
                                Sema::ConditionKind::Boolean).get().second;

    auto ifTrue = CO->getTrueExpr();
    auto ifFalse = CO->getFalseExpr();

    auto VisitBranch =
      [&] (const Expr* Branch, Expr* dfdx) -> std::pair<StmtDiff, StmtDiff> {
        auto Result = DifferentiateSingleExpr(Branch, dfdx);
        StmtDiff BranchDiff = Result.first;
        StmtDiff ExprDiff = Result.second;
        Stmt* Forward = unwrapIfSingleStmt(BranchDiff.getStmt());
        Stmt* Reverse = unwrapIfSingleStmt(BranchDiff.getStmt_dx());
        return { StmtDiff(Forward, Reverse), ExprDiff };
    };

    StmtDiff ifTrueDiff;
    StmtDiff ifTrueExprDiff;
    StmtDiff ifFalseDiff;
    StmtDiff ifFalseExprDiff;

    std::tie(ifTrueDiff, ifTrueExprDiff) = VisitBranch(ifTrue, dfdx());
    std::tie(ifFalseDiff, ifFalseExprDiff) = VisitBranch(ifFalse, dfdx());

    auto BuildIf =
      [&] (Expr* Cond, Stmt* Then, Stmt* Else) -> Stmt* {
        if (!Then && !Else)
          return nullptr;
        if (!Then)
          Then = m_Sema.ActOnNullStmt(noLoc).get();
        return new (m_Context) IfStmt(m_Context, noLoc, false, nullptr,  nullptr,
                                      Cond, Then, noLoc, Else);
      };

    Stmt* Forward = BuildIf(cond.getExpr(), ifTrueDiff.getStmt(),
                            ifFalseDiff.getStmt());
    Stmt* Reverse = BuildIf(cond.getExpr_dx(), ifTrueDiff.getStmt_dx(),
                            ifFalseDiff.getStmt_dx());
    if (Forward)
      addToCurrentBlock(Forward, forward);
    if (Reverse)
      addToCurrentBlock(Reverse, reverse);

    Expr* condExpr = m_Sema.ActOnConditionalOp(noLoc, noLoc, cond.getExpr(),
                                               ifTrueExprDiff.getExpr(),
                                               ifFalseExprDiff.getExpr()).get();
    // If result is a glvalue, we should keep it as it can potentially be assigned
    // as in (c ? a : b) = x;
    if ((CO->isModifiableLvalue(m_Context) == Expr::MLV_Valid) &&
        ifTrueExprDiff.getExpr_dx() && ifFalseExprDiff.getExpr_dx()) {
      Expr* ResultRef = m_Sema.ActOnConditionalOp(noLoc, noLoc, cond.getExpr_dx(),
                                                  ifTrueExprDiff.getExpr_dx(),
                                                  ifFalseExprDiff.getExpr_dx()).get();
      if (ResultRef->isModifiableLvalue(m_Context) != Expr::MLV_Valid)
        ResultRef = nullptr;
      return StmtDiff(condExpr, ResultRef);
    }
    return StmtDiff(condExpr);
  }

  StmtDiff ReverseModeVisitor::VisitForStmt(const ForStmt* FS) {
    beginScope(Scope::DeclScope | Scope::ControlScope | Scope::BreakScope |
               Scope::ContinueScope);
    // Counter that is used to count number of executed iterations of the loop,
    // to be able to use the same number of iterations in reverse pass.
    Expr* Counter = nullptr;
    Expr* Pop = nullptr;
    // If current loop is inside another loop, counter also has to be stored
    // in a tape.
    if (isInsideLoop) {
      auto zero = ConstantFolder::synthesizeLiteral(m_Context.getSizeType(),
                                                    m_Context, 0);
      auto CounterTape = MakeCladTapeFor(zero);
      addToCurrentBlock(CounterTape.Push, forward);
      Counter = CounterTape.Last();
      Pop = CounterTape.Pop;
    }
    else
      Counter = GlobalStoreAndRef(getZeroInit(m_Context.IntTy),
                                  m_Context.getSizeType(), "_t",
                                  /*force*/ true).getExpr();
    beginBlock(forward);
    beginBlock(reverse);
    const Stmt* init = FS->getInit();
    StmtDiff initResult = init ? Visit(init) : StmtDiff{};

    VarDecl* condVarDecl = FS->getConditionVariable();
    VarDecl* condVarClone = nullptr;
    if (condVarDecl) {
       VarDeclDiff condVarResult = DifferentiateVarDecl(condVarDecl);
       condVarClone = condVarResult.getDecl();
       // If there is a variable declared, store its derivative as a global,
       // as we usually do with declarations in reverse mode.
       if (condVarResult.getDecl_dx())
         addToBlock(BuildDeclStmt(condVarResult.getDecl_dx()), m_Globals);
    }

    // FIXME: for now we assume that cond has no differentiable effects,
    // but it is not generally true, e.g. for (...; (x = y); ...)...
    StmtDiff cond;
    if (FS->getCond())
      cond = Clone(FS->getCond());
    const Expr* inc = FS->getInc();

    // Save the isInsideLoop value (we may be inside another loop).
    llvm::SaveAndRestore<bool> SaveIsInsideLoop(isInsideLoop);
    isInsideLoop = true;

    Expr* CounterIncrement = BuildOp(UO_PostInc, Counter);
    // Differentiate the increment expression of the for loop
    // incExprDiff.getExpr() is the reconstructed expression, incDiff.getStmt()
    // a block with all the intermediate statements used to reconstruct it on
    // the forward pass, incDiff.getStmt_dx() is the reverse pass block.
    StmtDiff incDiff;
    StmtDiff incExprDiff;
    if (inc)
      std::tie(incDiff, incExprDiff) = DifferentiateSingleExpr(inc);
    Expr* incResult = nullptr;
    // If any additional statements were created, enclose them into lambda.
    CompoundStmt* Additional = cast<CompoundStmt>(incDiff.getStmt());
    bool anyNonExpr = std::any_of(Additional->body_begin(), Additional->body_end(),
      [] (Stmt* S) { return !isa<Expr>(S); });
    if (anyNonExpr) {
      incResult = wrapInLambda(*this, m_Sema, inc,
        [&] {
          std::tie(incDiff, incExprDiff) = DifferentiateSingleExpr(inc);
          for (Stmt* S : cast<CompoundStmt>(incDiff.getStmt())->body())
            addToCurrentBlock(S);
          addToCurrentBlock(incDiff.getExpr());
        });
    }
    // Otherwise, join all exprs by comma operator.
    else if (incExprDiff.getExpr()) {
      auto CommaJoin = [this] (Expr* Acc, Stmt* S) {
        Expr* E = cast<Expr>(S);
        return BuildOp(BO_Comma, E, BuildParens(Acc));
      };
      incResult = std::accumulate(Additional->body_rbegin(),
                                  Additional->body_rend(), incExprDiff.getExpr(),
                                  CommaJoin);
    }

    const Stmt* body = FS->getBody();
    StmtDiff BodyDiff;
    if (isa<CompoundStmt>(body)) {
      BodyDiff = Visit(body);
      beginBlock(forward);
      // Add loop increment in in the first place in the body.
      addToCurrentBlock(CounterIncrement);
      for (Stmt* S : cast<CompoundStmt>(BodyDiff.getStmt())->body())
        addToCurrentBlock(S);
      BodyDiff = { endBlock(forward), BodyDiff.getStmt_dx() };
    }
    else {
      beginScope(Scope::DeclScope);
      beginBlock(forward);
      // Add loop increment in in the first place in the body.
      addToCurrentBlock(CounterIncrement);
      BodyDiff = DifferentiateSingleStmt(body);
      addToCurrentBlock(BodyDiff.getStmt(), forward);
      Stmt* Forward = endBlock(forward);
      Stmt* Reverse = unwrapIfSingleStmt(BodyDiff.getStmt_dx());
      BodyDiff = { Forward, Reverse };
      endScope();
    }

    Stmt* Forward = new (m_Context) ForStmt(m_Context, initResult.getStmt(),
                                            cond.getExpr(), condVarClone,
                                            incResult, BodyDiff.getStmt(),
                                            noLoc, noLoc, noLoc);

    // Create a condition testing counter for being zero, and its decrement.
    // To match the number of iterations in the forward pass, the reverse loop
    // will look like: for(; Counter; Counter--) ...
    Expr* CounterCondition = m_Sema.ActOnCondition(m_CurScope, noLoc, Counter,
                                                   Sema::ConditionKind::Boolean).
                                                   get().second;
    Expr* CounterDecrement = BuildOp(UO_PostDec, Counter);

    beginBlock(reverse);
    // First, reverse the original loop increment expression, then loop's body.
    addToCurrentBlock(incDiff.getStmt_dx(), reverse);
    addToCurrentBlock(BodyDiff.getStmt_dx(), reverse);
    CompoundStmt* ReverseBody = endBlock(reverse);
    std::reverse(ReverseBody->body_begin(), ReverseBody->body_end());
    Stmt* ReverseResult = unwrapIfSingleStmt(ReverseBody);
    Stmt* Reverse = new (m_Context) ForStmt(m_Context, nullptr,
                                            CounterCondition, condVarClone,
                                            CounterDecrement, ReverseResult,
                                            noLoc, noLoc, noLoc);
    addToCurrentBlock(Forward, forward);
    Forward = endBlock(forward);
    addToCurrentBlock(Pop, reverse);
    addToCurrentBlock(Reverse, reverse);
    Reverse = endBlock(reverse);
    endScope();

    return { unwrapIfSingleStmt(Forward), unwrapIfSingleStmt(Reverse) };
  }

  StmtDiff ReverseModeVisitor::VisitCXXDefaultArgExpr(const CXXDefaultArgExpr* DE) {
    return Visit(DE->getExpr(), dfdx());
  }

  StmtDiff ReverseModeVisitor::VisitCXXBoolLiteralExpr(const CXXBoolLiteralExpr* BL) {
    return Clone(BL);
  }

  StmtDiff ReverseModeVisitor::VisitReturnStmt(const ReturnStmt* RS) {
    // Initially, df/df = 1.
    const Expr* value = RS->getRetValue();
    // value->dumpPretty(m_Context);
    // printf("\n");

    QualType type = value->getType();
    auto dfdf = ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 1);
    ExprResult tmp = dfdf;
    dfdf = m_Sema.ImpCastExprToType(tmp.get(), type,
                                    m_Sema.PrepareScalarCast(tmp, type)).get();
    auto ReturnResult = DifferentiateSingleExpr(value, dfdf);
    StmtDiff ReturnDiff = ReturnResult.first;
    StmtDiff ExprDiff = ReturnResult.second;
    Stmt* Reverse = ReturnDiff.getStmt_dx();
    // printf("Haha:\n");
    // Reverse->dump();
    // Reverse->dump();
    // If the original function returns at this point, some part of the reverse
    // pass (corresponding to other branches that do not return here) must be
    // skipped. We create a label in the reverse pass and jump to it via goto.
    LabelDecl* LD = LabelDecl::Create(m_Context, m_Sema.CurContext, noLoc,
                                      CreateUniqueIdentifier("_label"));
    // LD->dump();
    m_Sema.PushOnScopeChains(LD, m_DerivativeFnScope, true);
    // Attach label to the last Stmt in the corresponding Reverse Stmt.
    if (!Reverse)
      Reverse = m_Sema.ActOnNullStmt(noLoc).get();
    Stmt* LS = m_Sema.ActOnLabelStmt(noLoc, LD, noLoc, Reverse).get();
    // LS->dump();
    // LS->dumpPretty(m_Context);
    addToCurrentBlock(LS, reverse);
    for (Stmt* S : cast<CompoundStmt>(ReturnDiff.getStmt())->body())
      addToCurrentBlock(S, forward);
    // Since returned expression may have some side effects affecting reverse
    // computation (e.g. assignments), we also have to emit it to execute it.
    StoreAndRef(ExprDiff.getExpr(), forward,
                m_Function->getNameAsString() + "_return", /*force*/ true);
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
    StmtDiff BaseDiff = Visit(Base);
    llvm::SmallVector<Expr*, 4> clonedIndices(Indices.size());
    llvm::SmallVector<Expr*, 4> reverseIndices(Indices.size());
    for (std::size_t i = 0; i < Indices.size(); i++) {
      StmtDiff IdxDiff = Visit(Indices[i]);
      StmtDiff IdxStored = GlobalStoreAndRef(IdxDiff.getExpr());
      clonedIndices[i] = IdxStored.getExpr();
      reverseIndices[i] = IdxStored.getExpr_dx();
    }
    auto cloned = BuildArraySubscript(BaseDiff.getExpr(), clonedIndices);

    Expr* target = BaseDiff.getExpr_dx();
    if (!target)
      return cloned;
    Expr* result = nullptr;
    if (!target->getType()->isArrayType() && !target->getType()->isPointerType())
      result = target;
    else
      // Create the _result[idx] expression.
      result = BuildArraySubscript(target, reverseIndices);
    // Create the (target += dfdx) statement.
    if (dfdx()) {
      printf("kanjdska\n");
      auto add_assign = BuildOp(BO_AddAssign, result, dfdx());
      // Add it to the body statements.
      addToCurrentBlock(add_assign, reverse);
    }
    // result->dump();
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
      
      if (isJacobianCalc) {
        auto it = m_JacVariables[outputArrayCursor].find(decl);
        if (it == std::end(m_JacVariables[outputArrayCursor])) {
          // Is not an independent variable, ignored.
          return StmtDiff(clonedDRE);
        }
        // Create the (_result[idx] += dfdx) statement.
        if (dfdx()) {
          auto add_assign = BuildOp(BO_AddAssign, it->second, dfdx());
          // Add it to the body statements.
          addToCurrentBlock(add_assign, reverse);
          // add_assign->dumpPretty(m_Context);
          // printf("\n");
        }
      } else {      
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
          // add_assign->dumpPretty(m_Context);
          // printf("\n");
        }
        return StmtDiff(clonedDRE, it->second);
      }

      return StmtDiff(clonedDRE);
      }
  }

  StmtDiff ReverseModeVisitor::VisitIntegerLiteral(const IntegerLiteral* IL) {
    return StmtDiff(Clone(IL));
  }

  StmtDiff ReverseModeVisitor::VisitFloatingLiteral(const FloatingLiteral* FL) {
    return StmtDiff(Clone(FL));
  }

  StmtDiff ReverseModeVisitor::VisitCallExpr(const CallExpr* CE) {
    printf("hdkgkl\n");
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
    llvm::SmallVector<Expr*, 16> ReverseCallArgs{};
    // If the result does not depend on the result of the call, just clone
    // the call and visit arguments (since they may contain side-effects like
    // f(x = y))
    if (!dfdx()) {
      for (const Expr* Arg : CE->arguments()) {
        StmtDiff ArgDiff = Visit(Arg, dfdx());
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
      // Arg->dumpPretty(m_Context);
      ArgResultDecls.push_back(cast<VarDecl>(cast<DeclRefExpr>(dArg)->getDecl()));
      // Visit using unitialized reference.
      StmtDiff ArgDiff = Visit(Arg, dArg);
      // Save cloned arg in a "global" variable, so that it is accesible from the
      // reverse pass.
      ArgDiff = GlobalStoreAndRef(ArgDiff.getExpr());
      CallArgs.push_back(ArgDiff.getExpr());
      ReverseCallArgs.push_back(ArgDiff.getExpr_dx());
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
      OverloadedDerivedFn = m_Builder.findOverloadedDefinition(DNInfo, ReverseCallArgs);
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
      ReverseCallArgs.push_back(Result);

      // Try to find it in builtin derivatives
      DeclarationName name(II);
      DeclarationNameInfo DNInfo(name, noLoc);
      OverloadedDerivedFn = m_Builder.findOverloadedDefinition(DNInfo, ReverseCallArgs);
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
                                                     ReverseCallArgs),
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
                                                     ReverseCallArgs),
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
      auto RDelayed = DelayedGlobalStoreAndRef(R);
      StmtDiff RResult = RDelayed.Result;
      Expr* dl = nullptr;
      if (dfdx()) {
        dl = BuildOp(BO_Mul, dfdx(), RResult.getExpr_dx());
        dl = StoreAndRef(dl, reverse);
      }
      Ldiff = Visit(L, dl);
      //dxi/xr = xl
      //df/dxr += df/dxi * dxi/xr = df/dxi * xl
      // Store left multiplier and assign it with L.
      Expr* LStored = Ldiff.getExpr();
      // RDelayed.isConstant == true implies that R is a constant expression,
      // therefore we can skip visiting it.
      if (!RDelayed.isConstant) {
        Expr* dr = nullptr;
        if (dfdx()) {
          StmtDiff LResult = GlobalStoreAndRef(LStored);
          LStored = LResult.getExpr();
          dr = BuildOp(BO_Mul, LResult.getExpr_dx(), dfdx());
          dr = StoreAndRef(dr, reverse);
        }
        Rdiff = Visit(R, dr);
        // Assign right multiplier's variable with R.
        RDelayed.Finalize(Rdiff.getExpr());
      }
      std::tie(Ldiff, Rdiff) = std::make_pair(LStored, RResult.getExpr());
    }
    else if (opCode == BO_Div) {
      //xi = xl / xr
      //dxi/xl = 1 / xr
      //df/dxl += df/dxi * dxi/xl = df/dxi * (1/xr)
      auto RDelayed = DelayedGlobalStoreAndRef(R);
      StmtDiff RResult = RDelayed.Result;
      Expr* RStored = StoreAndRef(RResult.getExpr_dx(), reverse);
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
      Expr* LStored = Ldiff.getExpr();
      if (!RDelayed.isConstant) {
        Expr* dr = nullptr;
        if (dfdx()) {
          StmtDiff LResult = GlobalStoreAndRef(LStored);
          LStored = LResult.getExpr();
          Expr* RxR = BuildParens(BuildOp(BO_Mul, RStored, RStored));
          dr = BuildOp(BO_Mul, dfdx(),
                       BuildOp(UO_Minus, BuildOp(BO_Div, LResult.getExpr_dx(), RxR)));
          dr = StoreAndRef(dr, reverse);
        }
        Rdiff = Visit(R, dr);
        RDelayed.Finalize(Rdiff.getExpr());
      }
      std::tie(Ldiff, Rdiff) = std::make_pair(LStored, RResult.getExpr());
    }
    else if (BinOp->isAssignmentOp()) {
      if (L->isModifiableLvalue(m_Context) != Expr::MLV_Valid) {
        diag(DiagnosticsEngine::Warning, BinOp->getLocEnd(),
             "derivative of an assignment attempts to assign to unassignable "
             "expr, assignment ignored");
        return Clone(BinOp);
      }

      if (auto ASE = dyn_cast<ArraySubscriptExpr>(L)) {
        // whichArray = dyn_cast<IntegerLiteral>(ASE->getIdx());
        printf("asd: %s\n", outputArrayStr.c_str());
        auto DRE = dyn_cast<DeclRefExpr>(ASE->getBase()->IgnoreImplicit());
        // QualType type = m_Context.getDecayedType(DRE->getType());
        QualType type = DRE->getType()->getPointeeType();
        std::string DRE_str = DRE->getDecl()->getNameAsString();
        llvm::APSInt Result;
        ASE->getIdx()->EvaluateAsInt(Result, m_Context);
        outputArrayCursor = Result.getExtValue();
        if (DRE_str == outputArrayStr) {
          std::unordered_map<const clang::VarDecl*, clang::Expr*> temp_m_Variables;
          for (int i = 0; i < numParams; i++) {
            auto size_type = m_Context.getSizeType();
            auto size_type_bits = m_Context.getIntWidth(size_type);
            auto idx = IntegerLiteral::Create(m_Context, llvm::APInt(size_type_bits, i + (outputArrayCursor*numParams)),
                                                size_type, noLoc);
            // Create the _result[idx] expression.
            auto result_at_i = m_Sema.CreateBuiltinArraySubscriptExpr(m_Result, noLoc,
                                                                        idx, noLoc).get();
            temp_m_Variables[m_IndependentVars[i]] = result_at_i;
          }
          m_JacVariables.push_back(temp_m_Variables);
                    
          printf("Success\n");
          auto dfdf = ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 1);
          ExprResult tmp = dfdf;
          dfdf = m_Sema.ImpCastExprToType(tmp.get(), type,
                                          m_Sema.PrepareScalarCast(tmp, type)).get();
          // dfdf->dump();
          auto ReturnResult = DifferentiateSingleExpr(R, dfdf);
          StmtDiff ReturnDiff = ReturnResult.first;
          StmtDiff ExprDiff = ReturnResult.second;
          Stmt* Reverse = ReturnDiff.getStmt_dx();
          addToCurrentBlock(Reverse, reverse);
          for (Stmt* S : cast<CompoundStmt>(ReturnDiff.getStmt())->body())
            addToCurrentBlock(S, forward);
          
          Reverse->dumpPretty(m_Context);
        }
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
      auto Lblock_begin = Lblock->body_rbegin();
      auto Lblock_end = Lblock->body_rend();
      if (dfdx() && Lblock->size()) {
        addToCurrentBlock(*Lblock_begin, reverse);
        Lblock_begin = std::next(Lblock_begin);
      }
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
        auto RDelayed = DelayedGlobalStoreAndRef(R);
        StmtDiff RResult = RDelayed.Result;
        addToCurrentBlock(BuildOp(BO_AddAssign, AssignedDiff,
                                  BuildOp(BO_Mul, oldValue, RResult.getExpr_dx())),
                          reverse);
        Expr* LRef = LCloned;
        if (!RDelayed.isConstant) {
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
          if (LCloned->HasSideEffects(m_Context)) {
            QualType RefType = m_Context.getLValueReferenceType(L->getType());
            LRef = StoreAndRef(LCloned, RefType, forward, "_ref", /*force*/ true);
          }
          StmtDiff LResult = GlobalStoreAndRef(LRef);
          if (isInsideLoop)
            addToCurrentBlock(LResult.getExpr(), forward);
          Expr* dr = BuildOp(BO_Mul, LResult.getExpr_dx(), oldValue);
          dr = StoreAndRef(dr, reverse);
          Rdiff = Visit(R, dr);
          RDelayed.Finalize(Rdiff.getExpr());
        }
        std::tie(Ldiff, Rdiff) = std::make_pair(LRef, RResult.getExpr());
      }
      else if (opCode == BO_DivAssign) {
        auto RDelayed = DelayedGlobalStoreAndRef(R);
        StmtDiff RResult = RDelayed.Result;
        Expr* RStored = StoreAndRef(RResult.getExpr_dx(), reverse);
        addToCurrentBlock(BuildOp(BO_AddAssign, AssignedDiff,
                                  BuildOp(BO_Div, oldValue, RStored)), reverse);
        Expr* LRef = LCloned;
        if (!RDelayed.isConstant) {
          if (LCloned->HasSideEffects(m_Context)) {
            QualType RefType = m_Context.getLValueReferenceType(L->getType());
            LRef = StoreAndRef(LCloned, RefType, forward, "_ref", /*force*/ true);
          }
          StmtDiff LResult = GlobalStoreAndRef(LRef);
          if (isInsideLoop)
            addToCurrentBlock(LResult.getExpr(), forward);
          Expr* RxR = BuildParens(BuildOp(BO_Mul, RStored, RStored));
          Expr* dr = BuildOp(BO_Mul, oldValue,
                             BuildOp(UO_Minus, BuildOp(BO_Div, LResult.getExpr_dx(),
                                                       RxR)));
          dr = StoreAndRef(dr, reverse);
          Rdiff = Visit(R, dr);
          RDelayed.Finalize(Rdiff.getExpr());
        }
        std::tie(Ldiff, Rdiff) = std::make_pair(LRef, RResult.getExpr());
      }
      else
        llvm_unreachable("unknown assignment opCode");
      // Update the derivative.
      addToCurrentBlock(BuildOp(BO_SubAssign, AssignedDiff, oldValue), reverse);
      // Output statements from Visit(L).
      for (auto it = Lblock_begin; it != Lblock_end; ++it)
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
                                                       Expr* dfdS) {
    beginBlock(reverse);
    StmtDiff SDiff = Visit(S, dfdS);
    addToCurrentBlock(SDiff.getStmt_dx(), reverse);
    CompoundStmt* RCS = endBlock(reverse);
    std::reverse(RCS->body_begin(), RCS->body_end());
    Stmt* ReverseResult = unwrapIfSingleStmt(RCS);
    return StmtDiff(SDiff.getStmt(), ReverseResult);
  }

  std::pair<StmtDiff, StmtDiff> ReverseModeVisitor::DifferentiateSingleExpr(
    const Expr* E, Expr* dfdE) {
    beginBlock(forward);
    beginBlock(reverse);
    StmtDiff EDiff = Visit(E, dfdE);
    // EDiff.getExpr_dx()->dump();
    CompoundStmt* RCS = endBlock(reverse);
    // printf("RCS:\n");
    // RCS->dumpPretty(m_Context);
    Stmt* ForwardResult = endBlock(forward);
    std::reverse(RCS->body_begin(), RCS->body_end());
    Stmt* ReverseResult = unwrapIfSingleStmt(RCS);
    return { StmtDiff(ForwardResult, ReverseResult), EDiff };
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

  bool ReverseModeVisitor::UsefulToStoreGlobal(Expr* E) {
    if (isInsideLoop)
      return !E->isEvaluatable(m_Context, Expr::SE_NoSideEffects);
    if (!E)
      return false;
    // Use stricter policy when inside loops: IsEvaluatable is also true for
    // arithmetical expressions consisting of constants, e.g. (1 + 2)*3. This
    // chech is more expensive, but it doesn't make sense to push such constants
    // into stack.
    if (isInsideLoop)
      return !E->isEvaluatable(m_Context, Expr::SE_NoSideEffects);
    Expr* B = E->IgnoreParenImpCasts();
    // FIXME: find a more general way to determine that or add more options.
    if (isa<FloatingLiteral>(B) || isa<IntegerLiteral>(B))
      return false;
    if (isa<UnaryOperator>(B)) {
      auto UO = cast<UnaryOperator>(B);
      auto OpKind = UO->getOpcode();
      if (OpKind == UO_Plus || OpKind == UO_Minus)
        return UsefulToStoreGlobal(UO->getSubExpr());
      return true;
    }
    return true;
  }

  VarDecl* ReverseModeVisitor::GlobalStoreImpl(QualType Type,
                                            llvm::StringRef prefix) {
    // Save current scope and temporarily go to topmost function scope.
    llvm::SaveAndRestore<Scope*> SaveScope(m_CurScope);
    assert(m_DerivativeFnScope && "must be set");
    m_CurScope = m_DerivativeFnScope;

    VarDecl* Var = BuildVarDecl(Type, CreateUniqueIdentifier(prefix));

    // Add the declaration to the body of the gradient function.
    addToBlock(BuildDeclStmt(Var), m_Globals);
    return Var;
  }

  StmtDiff ReverseModeVisitor::GlobalStoreAndRef(Expr* E, QualType Type,
                                                 llvm::StringRef prefix,
                                                 bool force) {
    assert(E && "must be provided, otherwise use DelayedGlobalStoreAndRef");
    if (!force && !UsefulToStoreGlobal(E))
      return { E, E };

    if (isInsideLoop) {
      auto CladTape = MakeCladTapeFor(E);
      Expr* Push = CladTape.Push;
      Expr* Pop = CladTape.Pop;
      return { Push, Pop };
    } else {
      Expr* Ref = BuildDeclRef(GlobalStoreImpl(Type, prefix));
      if (E) {
        Expr* Set = BuildOp(BO_Assign, Ref, E);
        addToCurrentBlock(Set, forward);
      }
      return { Ref, Ref };
    }
  }

  StmtDiff ReverseModeVisitor::GlobalStoreAndRef(Expr* E, llvm::StringRef prefix,
                                                 bool force) {
    assert(E && "cannot infer type");
    return GlobalStoreAndRef(E, E->getType(), prefix, force);
  }

  void ReverseModeVisitor::DelayedStoreResult::Finalize(Expr* New) {
    if (isConstant)
      return;
    if (isInsideLoop) {
      auto Push = cast<CallExpr>(Result.getExpr());
     *std::prev(Push->arg_end()) = V.m_Sema.DefaultLvalueConversion(New).get();
    } else {
      V.addToCurrentBlock(V.BuildOp(BO_Assign, Result.getExpr(), New),
                          ReverseModeVisitor::forward);
    }
  }

  ReverseModeVisitor::DelayedStoreResult
  ReverseModeVisitor::DelayedGlobalStoreAndRef(Expr* E, llvm::StringRef prefix) {
    assert(E && "must be provided");
    if (!UsefulToStoreGlobal(E)) {
      Expr* Cloned = Clone(E);
      return DelayedStoreResult{*this, StmtDiff{ Cloned, Cloned },
                                /*isConstant*/ true, /*isInsideLoop*/ false};
    }
    if (isInsideLoop) {
      Expr* dummy = E;
      auto CladTape = MakeCladTapeFor(dummy);
      Expr* Push = CladTape.Push;
      Expr* Pop = CladTape.Pop;
      return DelayedStoreResult{*this, StmtDiff{ Push, Pop },
                                /*isConstant*/ false, /*isInsideLoop*/ true};
    } else {
      Expr* Ref = BuildDeclRef(GlobalStoreImpl(E->getType(), prefix));
      // Return reference to the declaration instead of original expression.
      return DelayedStoreResult{*this, StmtDiff{ Ref, Ref },
                                /*isConstant*/ false, /*isInsideLoop*/ false};
    }
  }

} // end namespace clad
