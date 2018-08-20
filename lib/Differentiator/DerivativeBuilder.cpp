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

using namespace clang;


namespace clad {
  DerivativeBuilder::DerivativeBuilder(clang::Sema& S)
    : m_Sema(S), m_Context(S.getASTContext()),
      m_NodeCloner(new utils::StmtClone(m_Context)),
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

  FunctionDecl* DerivativeBuilder::Derive(FunctionDeclInfo& FDI,
                                          const DiffPlan& plan) {
    FunctionDecl* result = nullptr;
    if (plan.getMode() == DiffMode::forward) {
      ForwardModeVisitor V(*this);
      result = V.Derive(FDI, plan);
    }
    else if (plan.getMode() == DiffMode::reverse) {
      ReverseModeVisitor V(*this);
      result = V.Derive(FDI, plan);
    }

    if (result)
      registerDerivative(result, m_Sema);
    return result;
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
    m_Sema.PushOnScopeChains(VD, m_CurScope, /*AddToContext*/ false);
    return VD;
  }

  VarDecl* VisitorBase::BuildVarDecl(QualType Type,
                                     llvm::StringRef prefix,
                                     Expr* Init,
                                     bool DirectInit) {
    return BuildVarDecl(Type,
                        CreateUniqueIdentifier(prefix, m_tmpId),
                        Init,
                        DirectInit);
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

  DeclRefExpr* VisitorBase::BuildDeclRef(VarDecl* D) {
    Expr* DRE = m_Sema.BuildDeclRefExpr(D, D->getType(), VK_LValue, noLoc).get();
    return cast<DeclRefExpr>(DRE);
  }

  IdentifierInfo*
  VisitorBase::CreateUniqueIdentifier(llvm::StringRef nameBase,
                                      std::size_t id) {

    // For intermediate variables, use numbered names (_t0), for everything
    // else first try a name without number (e.g. first try to use _d_x and
    // use _d_x0 only if _d_x is taken).
    std::string idStr = "";
    if (nameBase == "_t") {
      idStr = std::to_string(id);
      id += 1;
    }
    for (;;) {
      IdentifierInfo* name = &m_Context.Idents.get(nameBase.str() + idStr);
      LookupResult R(m_Sema,
                     DeclarationName(name),
                     noLoc,
                     Sema::LookupOrdinaryName);
      m_Sema.LookupName(R, m_CurScope, /*AllowBuiltinCreation*/ false);
      if (R.empty()) {
        m_tmpId = id;
        return name;
      } else {
        idStr = std::to_string(id);
        id += 1;
      }
    }
  }

   Expr* VisitorBase::BuildParens(Expr* E) {
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
    return StoreAndRef(E, E->getType(), prefix, forceDeclCreation);
  }

  Expr* VisitorBase::StoreAndRef(Expr* E,
                                 QualType Type,
                                 llvm::StringRef prefix,
                                 bool forceDeclCreation) {
    if (!forceDeclCreation) {
      // If Expr is simple (i.e. a reference or a literal), there is no point
      // in storing it as there is no evaluation going on.
      Expr* B = E->IgnoreParenImpCasts();
      // FIXME: find a more general way to determine that or add more options.
      if (isa<DeclRefExpr>(B) || isa<FloatingLiteral>(B) || 
          isa<IntegerLiteral>(B))
        return E;
    }
    // Create variable declaration.
    VarDecl* Var = BuildVarDecl(Type, CreateUniqueIdentifier(prefix, m_tmpId), E);

    // Add the declaration to the body of the gradient function.
    addToCurrentBlock(BuildDeclStmt(Var));

    // Return reference to the declaration instead of original expression.
    return BuildDeclRef(Var);
  }

  ForwardModeVisitor::ForwardModeVisitor(DerivativeBuilder& builder):
    VisitorBase(builder) {}

  ForwardModeVisitor::~ForwardModeVisitor() {}

  FunctionDecl* ForwardModeVisitor::Derive(
    FunctionDeclInfo& FDI,
    const DiffPlan& plan) {
    FunctionDecl* FD = FDI.getFD();
    m_Function = FD;
    assert(FD && "Must not be null.");
    assert(!m_DerivativeInFlight
           && "Doesn't support recursive diff. Use DiffPlan.");
    m_DerivativeInFlight = true;
#ifndef NDEBUG
    bool notInArgs = true;
    for (unsigned i = 0; i < FD->getNumParams(); ++i)
      if (FDI.getPVD() == FD->getParamDecl(i)) {
        notInArgs = false;
        break;
      }
    assert(!notInArgs && "Must pass in a param of the FD.");
#endif


    m_IndependentVar = FDI.getPVD(); // FIXME: Use only one var.
    m_DerivativeOrder = plan.getCurrentDerivativeOrder();
    std::string s = std::to_string(m_DerivativeOrder);
    std::string derivativeBaseName;
    if (m_DerivativeOrder == 1)
      s = "";
    switch (FD->getOverloadedOperator()) {
    default:
      derivativeBaseName = plan.begin()->getFD()->getNameAsString();
      break;
    case OO_Call:
      derivativeBaseName = "operator_call";
      break;
    }

    m_ArgIndex = plan.getArgIndex();
    IdentifierInfo* II = &m_Context.Idents.get(
      derivativeBaseName + "_d" + s + "arg" + std::to_string(m_ArgIndex));
    DeclarationNameInfo name(II, noLoc);
    FunctionDecl* derivedFD = 0;
    if (isa<CXXMethodDecl>(FD)) {
      CXXRecordDecl* CXXRD = cast<CXXRecordDecl>(FD->getDeclContext());
      derivedFD = CXXMethodDecl::Create(m_Context, CXXRD, noLoc, name,
                                        FD->getType(), FD->getTypeSourceInfo(),
                                        FD->getStorageClass(),
                                        FD->isInlineSpecified(),
                                        FD->isConstexpr(), noLoc);
      derivedFD->setAccess(FD->getAccess());
    } else {
      assert(isa<FunctionDecl>(FD) && "Must derive from FunctionDecl.");
      derivedFD = FunctionDecl::Create(m_Context,
                                       FD->getDeclContext(), noLoc,
                                       name, FD->getType(),
                                       FD->getTypeSourceInfo(),
                                       FD->getStorageClass(),
                                       /*default*/
                                       FD->isInlineSpecified(),
                                       FD->hasWrittenPrototype(),
                                       FD->isConstexpr()
                                       );
    }
    m_Derivative = derivedFD;

    llvm::SmallVector<ParmVarDecl*, 4> params;
    ParmVarDecl* newPVD = 0;
    ParmVarDecl* PVD = 0;

    std::unique_ptr<Scope> FnScope { new Scope(m_Sema.TUScope, Scope::FnScope,
                                               m_Sema.getDiagnostics()) };
    m_CurScope = FnScope.get();
    m_Sema.CurContext = m_Derivative;

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
                                 m_CurScope,
                                 /*AddToContext*/ false);
    }

    llvm::ArrayRef<ParmVarDecl*> paramsRef
      = llvm::makeArrayRef(params.data(), params.size());
    derivedFD->setParams(paramsRef);
    derivedFD->setBody(nullptr);

    Sema::SynthesizedFunctionScope Scope(m_Sema, derivedFD);
    // Begin function body.
    beginBlock();
    // For each function parameter variable, store its derivative value.
    for (auto param : params) {
      // If param is not real (i.e. floating point or integral), we cannot
      // differentiate it.
      // FIXME: we should support custom numeric types in the future.
      if (!param->getType()->isRealType()) {
        if (param != m_IndependentVar)
          continue;
        diag(DiagnosticsEngine::Error, PVD->getLocEnd(),
             "attempted differentiation w.r.t. a parameter ('%0') which is not "
             "of a real type", { m_IndependentVar->getNameAsString() });
        return nullptr;
      }
      // If param is independent variable, its derivative is 1, otherwise 0.
      int dValue = (param == m_IndependentVar);
      auto dParam = ConstantFolder::synthesizeLiteral(param->getType(),
                                                      m_Context,
                                                      dValue);
      // Memorize the derivative of param, i.e. whenever the param is visited
      // in the future, it's derivative dParam is found (unless reassigned with
      // something new).
      m_Variables[param] = dParam;
    }

    if (!FD->getDefinition()) {
      diag(DiagnosticsEngine::Error, FD->getLocEnd(),
           "attempted differentiation of function '%0', which does not have a "
           "definition", { FD->getNameAsString() });
      return nullptr;
    }
    Stmt* BodyDiff = Visit(FD->getDefinition()->getBody()).getStmt();
    if (isa<CompoundStmt>(BodyDiff))
      for (Stmt* S : cast<CompoundStmt>(BodyDiff)->body())
        addToCurrentBlock(S);
    else
      addToCurrentBlock(BodyDiff);
    Stmt* derivativeBody = endBlock();
    derivedFD->setBody(derivativeBody);

    m_DerivativeInFlight = false;
    return derivedFD;
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
    return m_Sema.BuildUnaryOp(nullptr,
                               noLoc,
                               OpCode,
                               E).get();
  }

  Expr* VisitorBase::BuildOp(clang::BinaryOperatorKind OpCode,
                                   Expr* L, Expr* R) {
    return m_Sema.BuildBinOp(nullptr, noLoc, OpCode, L, R).get();
  }

  StmtDiff ForwardModeVisitor::VisitStmt(const Stmt* S) {
    diag(DiagnosticsEngine::Warning, S->getLocEnd(),
         "attempted to differentiate unsupported statement, no changes applied");
    // Unknown stmt, just clone it.
    return StmtDiff(Clone(S));
  }

  StmtDiff ForwardModeVisitor::VisitCompoundStmt(const CompoundStmt* CS) {
    beginBlock();
    for (Stmt* S : CS->body()) {
      StmtDiff SDiff = Visit(S);
      if (SDiff.getStmt_dx())
        addToCurrentBlock(SDiff.getStmt_dx());
      addToCurrentBlock(SDiff.getStmt());
    }
    CompoundStmt* Result = endBlock();
    // Differentation of CompundStmt produces another CompoundStmt with both
    // original and derived statements, i.e. Stmt() is Result and Stmt_dx() is
    // null.
    return StmtDiff(Result);
  }

  StmtDiff ForwardModeVisitor::VisitIfStmt(const IfStmt* If) {
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
    if (initResult.getStmt_dx())
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
          StmtDiff BranchDiff = Visit(Branch);
          for (Stmt* S : BranchDiff.getBothStmts())
            if (S)
              addToCurrentBlock(S);
          CompoundStmt* Block = endBlock();
          if (Block->size() == 1)
            return Block->body_front();
          else
            return Block;
        }
      };

    Stmt* thenDiff = VisitBranch(If->getThen());
    Stmt* elseDiff = VisitBranch(If->getElse());

    Stmt* ifDiff = new (m_Context) IfStmt(m_Context,
                                         noLoc,
                                         If->isConstexpr(),
                                         initResult.getStmt(),
                                         condVarClone,
                                         cond,
                                         thenDiff,
                                         noLoc,
                                         elseDiff);
    addToCurrentBlock(ifDiff);
    CompoundStmt* Block = endBlock();
    // If IfStmt is the only statement in the block, remove the block:
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

  StmtDiff ForwardModeVisitor::VisitDeclRefExpr(const DeclRefExpr* DRE) {
    auto clonedDRE = cast<DeclRefExpr>(Clone(DRE));
    if (auto VD = dyn_cast<VarDecl>(clonedDRE->getDecl())) {
      // If DRE references a variable, try to find if we know something about
      // how it is related to the independent variable.
      auto it = m_Variables.find(VD);
      if (it != std::end(m_Variables)) {
        // If a record was found, use the recorded derivative.
        Expr* dVarDRE = it->second;
        return StmtDiff(clonedDRE, dVarDRE);
      }
    }
    // Is not a variable or is a reference to something unrelated to independent
    // variable. Derivative is 0.
    auto zero = ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
    return StmtDiff(clonedDRE, zero);
  }

  StmtDiff ForwardModeVisitor::VisitIntegerLiteral(
    const IntegerLiteral* IL) {
    llvm::APInt zero(m_Context.getIntWidth(m_Context.IntTy), /*value*/0);
    auto constant0 = IntegerLiteral::Create(m_Context, zero, m_Context.IntTy,
                                            noLoc);
    return StmtDiff(Clone(IL), constant0);
  }

  StmtDiff ForwardModeVisitor::VisitFloatingLiteral(
    const FloatingLiteral* FL) {
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
    // Find the built-in derivatives namespace.
    std::string s = std::to_string(m_DerivativeOrder);
    if (m_DerivativeOrder == 1)
      s = "";
    IdentifierInfo* II = 0;
    if (m_ArgIndex == 1)
      II = &m_Context.Idents.get(CE->getDirectCallee()->getNameAsString() +
                                 "_d" + s + "arg0");
    else
      II = &m_Context.Idents.get(CE->getDirectCallee()->getNameAsString() +
                                 "_d" + s + "arg" + std::to_string(m_ArgIndex));
    DeclarationName name(II);
    SourceLocation DeclLoc;
    DeclarationNameInfo DNInfo(name, DeclLoc);

    SourceLocation noLoc;
    llvm::SmallVector<Expr*, 4> CallArgs;
    // For f(g(x)) = f'(x) * g'(x)
    Expr* Multiplier = 0;
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

    Expr* call =
      m_Sema.ActOnCallExpr(m_Sema.getScopeForContext(m_Sema.CurContext),
                           Clone(CE->getCallee()),
                           noLoc,
                           llvm::MutableArrayRef<Expr*>(CallArgs),
                           noLoc).get();

    Expr* callDiff =
      m_Builder.findOverloadedDefinition(DNInfo, CallArgs);


    // Check if it is a recursive call.
    if (!callDiff && (CE->getDirectCallee() == m_Function)) {
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

    if (callDiff) {
      // f_darg0 function was found.
      if (Multiplier)
        callDiff = BuildOp(BO_Mul,
                           callDiff,
                           BuildParens(Multiplier));
      return StmtDiff(call, callDiff);
    }

    Expr* OverloadedFnInFile
      = m_Builder.findOverloadedDefinition(CE->getDirectCallee()->getNameInfo(),
                                           CallArgs);

    if (OverloadedFnInFile) {
      // Take the function to derive from the source.
      const FunctionDecl* FD = CE->getDirectCallee();
      // Get the definition, if any.
      const FunctionDecl* mostRecentFD = FD->getMostRecentDecl();
      while (mostRecentFD && !mostRecentFD->isThisDeclarationADefinition()) {
        mostRecentFD = mostRecentFD->getPreviousDecl();
      }
      if (!mostRecentFD || !mostRecentFD->isThisDeclarationADefinition()) {
        diag(DiagnosticsEngine::Error, FD->getLocEnd(),
             "attempted differentiation of function '%0', which does not have a \
              definition", { FD->getNameAsString() });
        auto zero = ConstantFolder::synthesizeLiteral(call->getType(),
                                                      m_Context, 0);
        return StmtDiff(call, zero);
      }

      // Look for a declaration of a function to differentiate
      // in the derivatives namespace.
      LookupResult R(m_Sema, CE->getDirectCallee()->getNameInfo(),
                     Sema::LookupOrdinaryName);
      m_Sema.LookupQualifiedName(R, m_Builder.m_BuiltinDerivativesNSD,
                                 /*allowBuiltinCreation*/ false);
      {
        DeclContext::lookup_result res
          = m_Context.getTranslationUnitDecl()->lookup(name);
        bool shouldAdd = true;
        for (DeclContext::lookup_iterator I = res.begin(), E = res.end();
             I != E; ++I) {
          for (LookupResult::iterator J = R.begin(), E = R.end(); J != E; ++J)
            if (cast<ValueDecl>(*I)->getType().getTypePtr()
                == cast<ValueDecl>(J.getDecl())->getType().getTypePtr()) {
              shouldAdd = false;
              break;
            }
          if (shouldAdd)
            R.addDecl(*I);
          shouldAdd = true;
        }
        assert(!R.empty() && "Must be reachable");
      }      // Update function name in the source.
      CXXScopeSpec CSS;
      Expr* ResolvedLookup
        = m_Sema.BuildDeclarationNameExpr(CSS, R, /*ADL*/ false).get();
      CallExpr* clonedCE = dyn_cast<CallExpr>(Clone(CE));
      clonedCE->setCallee(ResolvedLookup);
      // FIXME: What is this part doing? Is it reachable at all?
      // Shouldn't it be multiplied by arg derivatives?
      return StmtDiff(call, clonedCE);
    }

    // Function was not derived => issue a warning.
    diag(DiagnosticsEngine::Warning, CE->getDirectCallee()->getLocEnd(),
         "function '%0' was not differentiated because it is not declared in "
         "namespace 'custom_derivatives'",
         { CE->getDirectCallee()->getNameAsString() });

    auto zero = ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
    return StmtDiff(call, zero);
  }

  void VisitorBase::updateReferencesOf(Stmt* InSubtree) {
    utils::ReferencesUpdater up(m_Sema,
                                m_Builder.m_NodeCloner.get(),
                                m_CurScope);
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
    else {
      diag(DiagnosticsEngine::Warning, UnOp->getLocEnd(),
           "attempt to differentiate unsupported unary operator, derivative \
            set to 0");
      auto zero =
        ConstantFolder::synthesizeLiteral(op->getType(),
                                          m_Context,
                                          0);
      return StmtDiff(op, zero);
    }
  }

  StmtDiff ForwardModeVisitor::VisitBinaryOperator(
    const BinaryOperator* BinOp) {

    StmtDiff Ldiff = Visit(BinOp->getLHS());
    StmtDiff Rdiff = Visit(BinOp->getRHS());

    ConstantFolder folder(m_Context);
    auto opCode = BinOp->getOpcode();
    Expr* opDiff = nullptr;

    if (opCode == BO_Mul) {
      // If Ldiff.getExpr() and Rdiff.getExpr() require evaluation, store the
      // expressions in variables to avoid reevaluation.
      Ldiff = { StoreAndRef(Ldiff.getExpr()), Ldiff.getExpr_dx() };
      Rdiff = { StoreAndRef(Rdiff.getExpr()), Rdiff.getExpr_dx() };

      Expr* LHS = BuildOp(BO_Mul, BuildParens(Ldiff.getExpr_dx()),
                          BuildParens(Rdiff.getExpr()));

      Expr* RHS = BuildOp(BO_Mul, BuildParens(Ldiff.getExpr()),
                          BuildParens(Rdiff.getExpr_dx()));

      opDiff = BuildOp(BO_Add, LHS, RHS);
    }
    else if (opCode == BO_Div) {
      Ldiff = { StoreAndRef(Ldiff.getExpr()), Ldiff.getExpr_dx() };
      Rdiff = { StoreAndRef(Rdiff.getExpr()), Rdiff.getExpr_dx() };

      Expr* LHS = BuildOp(BO_Mul, BuildParens(Ldiff.getExpr_dx()),
                          BuildParens(Rdiff.getExpr()));

      Expr* RHS = BuildOp(BO_Mul, BuildParens(Ldiff.getExpr()),
                          BuildParens(Rdiff.getExpr_dx()));

      Expr* nominator = BuildOp(BO_Sub, LHS, RHS);

      Expr* RParens = BuildParens(Rdiff.getExpr());
      Expr* denominator = BuildOp(BO_Mul, RParens, RParens);

      opDiff = BuildOp(BO_Div, BuildParens(nominator), BuildParens(denominator));
    }
    else if (opCode == BO_Add)
      opDiff = BuildOp(BO_Add, Ldiff.getExpr_dx(), Rdiff.getExpr_dx());
    else if (opCode == BO_Sub)
      opDiff = BuildOp(BO_Sub, Ldiff.getExpr_dx(), 
                       BuildParens(Rdiff.getExpr_dx()));
    else {
      //FIXME: add support for other binary operators
      diag(DiagnosticsEngine::Warning, BinOp->getLocEnd(),
           "attempt to differentiate unsupported binary operator, derivative \
            set to 0");
      opDiff =
        ConstantFolder::synthesizeLiteral(BinOp->getType(),
                                          m_Context,
                                          0);
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
                                    VD->getIdentifier(),
                                    initDiff.getExpr(),
                                    VD->isDirectInit());
    VarDecl* VDDerived = BuildVarDecl(VD->getType(),
                                      "_d_" + VD->getNameAsString(),
                                      initDiff.getExpr_dx());
    if (initDiff.getExpr_dx())
      m_Variables.emplace(VDClone, BuildDeclRef(VDDerived));
    return VarDeclDiff(VDClone, VDDerived);
  }

  StmtDiff ForwardModeVisitor::VisitDeclStmt(const DeclStmt* DS) {
    llvm::SmallVector<Decl*, 4> decls;
    llvm::SmallVector<Decl*, 4> declsDiff;
    for (auto D : DS->decls()) {
      if (auto VD = dyn_cast<VarDecl>(D)) {
        VarDeclDiff VDDiff = DifferentiateVarDecl(VD);
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

  ReverseModeVisitor::ReverseModeVisitor(DerivativeBuilder& builder):
    VisitorBase(builder) {}

  ReverseModeVisitor::~ReverseModeVisitor() {}

  FunctionDecl* ReverseModeVisitor::Derive(
    FunctionDeclInfo& FDI, const DiffPlan& plan) {
    m_Function = FDI.getFD();
    assert(m_Function && "Must not be null.");

    // We name the gradient of f as f_grad.
    auto derivativeBaseName = m_Function->getNameAsString();
    IdentifierInfo* II = &m_Context.Idents.get(derivativeBaseName + "_grad");
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
    if (isa<CXXMethodDecl>(m_Function)) {
      auto CXXRD = cast<CXXRecordDecl>(m_Function->getDeclContext());
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
      gradientFD = FunctionDecl::Create(m_Context,
                                        m_Function->getDeclContext(),
                                        noLoc,
                                        name,
                                        gradientFunctionType,
                                        m_Function->getTypeSourceInfo(),
                                        m_Function->getStorageClass(),
                                        m_Function->isInlineSpecified(),
                                        m_Function->hasWrittenPrototype(),
                                        m_Function->isConstexpr());
    } else {
      diag(DiagnosticsEngine::Error, m_Function->getLocEnd(),
           "attempted differentiation of '%0' which is of unsupported type",
           { m_Function->getNameAsString() });
      return nullptr;
    }
    m_Derivative = gradientFD;

    std::unique_ptr<Scope> FnScope { new Scope(m_Sema.TUScope, Scope::FnScope,
                                               m_Sema.getDiagnostics()) };
    m_CurScope = FnScope.get();
    m_Sema.CurContext = m_Derivative;

    // Create parameter declarations.
    llvm::SmallVector<ParmVarDecl*, 4> params(paramTypes.size());
    std::transform(m_Function->param_begin(),
                   m_Function->param_end(),
                   std::begin(params),
                   [&] (const ParmVarDecl* PVD) {
                     auto VD =
                       ParmVarDecl::Create(m_Context,
                                           gradientFD,
                                           noLoc,
                                           noLoc,
                                           PVD->getIdentifier(),
                                           PVD->getType(),
                                           PVD->getTypeSourceInfo(),
                                           PVD->getStorageClass(),
                                           // Clone default arg if present.
                                           (PVD->hasDefaultArg() ?
                                             Clone(PVD->getDefaultArg()) :
                                             nullptr));
                     if (VD->getIdentifier())
                       m_Sema.PushOnScopeChains(VD,
                                                m_CurScope,
                                                /*AddToContext*/ false);
                     return VD;
                   });
    // The output paremeter "_result".
    params.back() =
      ParmVarDecl::Create(m_Context,
                          gradientFD,
                          noLoc,
                          noLoc,
                          &m_Context.Idents.get("_result"),
                          paramTypes.back(),
                          m_Context.getTrivialTypeSourceInfo(paramTypes.back(),
                                                             noLoc),
                          params.front()->getStorageClass(),
                          // No default value.
                          nullptr);
    if (params.back()->getIdentifier())
      m_Sema.PushOnScopeChains(params.back(),
                               m_CurScope,
                               /*AddToContext*/ false);

    llvm::ArrayRef<ParmVarDecl*> paramsRef =
      llvm::makeArrayRef(params.data(), params.size());
    gradientFD->setParams(paramsRef);
    gradientFD->setBody(nullptr);

    Sema::SynthesizedFunctionScope Scope(m_Sema, gradientFD);
    // Reference to the output parameter.
    m_Result = BuildDeclRef(params.back());
    // Initially, df/df = 1.
    auto dfdf = ConstantFolder::synthesizeLiteral(m_Function->getReturnType(),
                                                  m_Context,
                                                  1.0);

    beginBlock();
    // Start the visitation process which outputs the statements in the current
    // block.
    Stmt* functionBody = m_Function->getMostRecentDecl()->getBody();
    Visit(functionBody, dfdf);
    // Create the body of the function.
    Stmt* gradientBody = endBlock();

    gradientFD->setBody(gradientBody);
    // Cleanup the IdResolver chain.
    for(FunctionDecl::param_iterator I = gradientFD->param_begin(),
        E = gradientFD->param_end(); I != E; ++I) {
      if ((*I)->getIdentifier()) {
        m_CurScope->RemoveDecl(*I);
        //m_Sema.IdResolver.RemoveDecl(*I); // FIXME: Understand why that's bad
      }
    }
    return gradientFD;
  }

  void ReverseModeVisitor::VisitCompoundStmt(const CompoundStmt* CS) {
    for (CompoundStmt::const_body_iterator I = CS->body_begin(),
           E = CS->body_end(); I != E; ++I)
        Visit(*I, dfdx());
  }

  void ReverseModeVisitor::VisitIfStmt(const clang::IfStmt* If) {
    if (If->getConditionVariable())
        // FIXME:Visit(If->getConditionVariableDeclStmt(), dfdx());
        llvm_unreachable("variable declarations are not currently supported");
    Expr* cond = Clone(If->getCond());
    const Stmt* thenStmt = If->getThen();
    const Stmt* elseStmt = If->getElse();

    Stmt* thenBody = nullptr;
    Stmt* elseBody = nullptr;
    if (thenStmt) {
      beginBlock();
      Visit(thenStmt, dfdx());
      thenBody = endBlock();
    }
    if (elseStmt) {
      beginBlock();
      Visit(elseStmt, dfdx());
      elseBody = endBlock();
    }

    IfStmt* ifStmt =
      new (m_Context) IfStmt(m_Context,
                             noLoc,
                             If->isConstexpr(),
                             // FIXME: add init for condition variable
                             nullptr,
                             // FIXME: add condition variable decl
                             nullptr,
                             cond,
                             thenBody,
                             noLoc,
                             elseBody);
    getCurrentBlock().push_back(ifStmt);
  }

  void ReverseModeVisitor::VisitConditionalOperator(
    const clang::ConditionalOperator* CO) {
    Expr* condVar = StoreAndRef(Clone(CO->getCond()), "_t", /*force*/ true);
    auto cond =
      m_Sema.ActOnCondition(m_CurScope,
                            noLoc,
                            condVar,
                            Sema::ConditionKind::Boolean).get().second;
    auto ifTrue = CO->getTrueExpr();
    auto ifFalse = CO->getFalseExpr();

    auto VisitBranch =
      [&] (Stmt* branch, Expr* ifTrue, Expr* ifFalse) {
        if (!branch)
          return;
        auto condExpr =
          new (m_Context) ConditionalOperator(cond,
                                              noLoc,
                                              ifTrue,
                                              noLoc,
                                              ifFalse,
                                              ifTrue->getType(),
                                              VK_RValue,
                                              OK_Ordinary);

        auto dStmt = new (m_Context) ParenExpr(noLoc,
                                               noLoc,
                                               condExpr);
        Visit(branch, dStmt);
    };

    auto zero = ConstantFolder::synthesizeLiteral(dfdx()->getType(),
                                                  m_Context,
                                                  0);
    //xi = (cond ? ifTrue : ifFalse)
    //dxi/d ifTrue = (cond ? 1 : 0)
    //df/d ifTrue += df/dxi * dxi/d ifTrue = (cond ? df/dxi : 0)
    VisitBranch(ifTrue, dfdx(), zero);
    //dxi/d ifFalse = (cond ? 0 : 1)
    //df/d ifFalse += df/dxi * dxi/d ifFalse = (cond ? 0 : df/dxi)
    VisitBranch(ifFalse, zero, dfdx());
  }

  void ReverseModeVisitor::VisitReturnStmt(const ReturnStmt* RS) {
    Visit(RS->getRetValue(), dfdx());
  }

  void ReverseModeVisitor::VisitParenExpr(const ParenExpr* PE) {
    Visit(PE->getSubExpr(), dfdx());
  }

  void ReverseModeVisitor::VisitDeclRefExpr(const DeclRefExpr* DRE) {
    auto decl = DRE->getDecl();
    // Check DeclRefExpr is a reference to an independent variable.
    auto it = std::find(m_Function->param_begin(),
                        m_Function->param_end(),
                        decl);
    if (it == m_Function->param_end()) {
      // Is not an independent variable, ignored.
      return;
    }
    auto idx = std::distance(m_Function->param_begin(), it);
    auto size_type = m_Context.getSizeType();
    auto size_type_bits = m_Context.getIntWidth(size_type);
    // Create the idx literal.
    auto i = IntegerLiteral::Create(m_Context,
                                    llvm::APInt(size_type_bits, idx),
                                    size_type,
                                    noLoc);
    // Create the _result[idx] expression.
    auto result_at_i =
      m_Sema.CreateBuiltinArraySubscriptExpr(m_Result,
                                             noLoc,
                                             i,
                                             noLoc).get();
    // Create the (_result[idx] += dfdx) statement.
    auto add_assign = BuildOp(BO_AddAssign, result_at_i, dfdx());
    // Add it to the body statements.
    getCurrentBlock().push_back(add_assign);
  }

  void ReverseModeVisitor::VisitIntegerLiteral(const IntegerLiteral* IL) {
    // Nothing to do with it.
  }

  void ReverseModeVisitor::VisitFloatingLiteral(const FloatingLiteral* FL) {
    // Nothing to do with it.
  }

  void ReverseModeVisitor::VisitCallExpr(const CallExpr* CE) {
    auto FD = CE->getDirectCallee();
    if (!FD) {
      diag(DiagnosticsEngine::Warning, CE->getLocEnd(),
           "Differentiation of only direct calls is supported. Ignored");
      return;
    }
    IdentifierInfo* II = nullptr;
    auto NArgs = FD->getNumParams();
    // If the function has no args then we assume that it is not related
    // to independent variables and does not contribute to gradient.
    if (!NArgs)
      return;

    llvm::SmallVector<Expr*, 16> CallArgs(CE->getNumArgs());
    std::transform(CE->arg_begin(), CE->arg_end(), std::begin(CallArgs),
      [this](const Expr* Arg) { return Clone(Arg); });

    VarDecl* ResultDecl = nullptr;
    Expr* Result = nullptr;
    // If the function has a single arg, we look for a derivative w.r.t. to
    // this arg (it is unlikely that we need gradient of a one-dimensional'
    // function).
    if (NArgs == 1)
      II = &m_Context.Idents.get(FD->getNameAsString() + "_darg0");
    // If it has more args, we look for its gradient.
    else {
      II = &m_Context.Idents.get(FD->getNameAsString() + "_grad");
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
      ResultDecl = BuildVarDecl(ArrayType,
                                CreateUniqueIdentifier("_grad", m_tmpId),
                                ZeroInitBraces);
      Result = BuildDeclRef(ResultDecl);
      // Pass the array as the last parameter for gradient.
      CallArgs.push_back(Result);
    }

    // Try to find it in builtin derivatives
    DeclarationName name(II);
    DeclarationNameInfo DNInfo(name, noLoc);
    auto OverloadedDerivedFn =
      m_Builder.findOverloadedDefinition(DNInfo, CallArgs);

    // Derivative was not found, check if it is a recursive call
    if (!OverloadedDerivedFn) {
      if (FD != m_Function) {
        // Not a recursive call, derivative was not found, ignore.
        // Issue a warning.
        diag(DiagnosticsEngine::Warning, CE->getDirectCallee()->getLocEnd(),
             "function '%0' was not differentiated because it is not declared \
              in namespace 'custom_derivatives'",
             { FD->getNameAsString() });
        return;
      }
      // Recursive call.
      auto selfRef = m_Sema.BuildDeclarationNameExpr(CXXScopeSpec(),
                                                     m_Derivative->getNameInfo(),
                                                     m_Derivative).get();

      OverloadedDerivedFn =
        m_Sema.ActOnCallExpr(m_Sema.getScopeForContext(m_Sema.CurContext),
                             selfRef,
                             noLoc,
                             llvm::MutableArrayRef<Expr*>(CallArgs),
                             noLoc).get();
    }

    if (OverloadedDerivedFn) {
      // Derivative was found.
      if (NArgs == 1) {
        // If function has a single arg, call it and store a result.
        Result = StoreAndRef(OverloadedDerivedFn);
        auto d = BuildOp(BO_Mul, dfdx(), Result);
        auto dTmp = StoreAndRef(d);
        Visit(CE->getArg(0), dTmp);
      } else {
        // Put Result array declaration in the function body.
        getCurrentBlock().push_back(BuildDeclStmt(ResultDecl));
        // Call the gradient, passing Result as the last Arg.
        getCurrentBlock().push_back(OverloadedDerivedFn);
        // Visit each arg with df/dargi = df/dxi * Result[i].
        for (unsigned i = 0; i < CE->getNumArgs(); i++) {
          auto size_type = m_Context.getSizeType();
          auto size_type_bits = m_Context.getIntWidth(size_type);
          // Create the idx literal.
          auto I =
            IntegerLiteral::Create(m_Context,
                                   llvm::APInt(size_type_bits, i),
                                   size_type,
                                   noLoc);
          // Create the Result[I] expression.
          auto ithResult =
            m_Sema.CreateBuiltinArraySubscriptExpr(Result,
                                                   noLoc,
                                                   I,
                                                   noLoc).get();
          auto di = BuildOp(BO_Mul, dfdx(), ithResult);
          auto diTmp = StoreAndRef(di);
          Visit(CE->getArg(i), diTmp);
        }
      }
    }
  }

  void ReverseModeVisitor::VisitUnaryOperator(const UnaryOperator* UnOp) {
    auto opCode = UnOp->getOpcode();
    if (opCode == UO_Plus)
      //xi = +xj
      //dxi/dxj = +1.0
      //df/dxj += df/dxi * dxi/dxj = df/dxi
      Visit(UnOp->getSubExpr(), dfdx());
    else if (opCode == UO_Minus) {
      //xi = -xj
      //dxi/dxj = -1.0
      //df/dxj += df/dxi * dxi/dxj = -df/dxi
      auto d = BuildOp(UO_Minus, dfdx());
      Visit(UnOp->getSubExpr(), d);
    }
    else
      llvm_unreachable("unsupported unary operator");
  }

  void ReverseModeVisitor::VisitBinaryOperator(const BinaryOperator* BinOp) {
    auto opCode = BinOp->getOpcode();

    auto L = BinOp->getLHS();
    auto R = BinOp->getRHS();

    if (opCode == BO_Add) {
      //xi = xl + xr
      //dxi/xl = 1.0
      //df/dxl += df/dxi * dxi/xl = df/dxi
      Visit(L, dfdx());
      //dxi/xr = 1.0
      //df/dxr += df/dxi * dxi/xr = df/dxi
      Visit(R, dfdx());
    }
    else if (opCode == BO_Sub) {
      //xi = xl - xr
      //dxi/xl = 1.0
      //df/dxl += df/dxi * dxi/xl = df/dxi
      Visit(L, dfdx());
      //dxi/xr = -1.0
      //df/dxl += df/dxi * dxi/xr = -df/dxi
      auto dr = BuildOp(UO_Minus, dfdx());
      Visit(R, dr);
    }
    else if (opCode == BO_Mul) {
      //xi = xl * xr
      //dxi/xl = xr
      //df/dxl += df/dxi * dxi/xl = df/dxi * xr
      auto dl = BuildOp(BO_Mul, dfdx(), Clone(R));
      auto dlTmp = StoreAndRef(dl);
      Visit(L, dlTmp);
      //dxi/xr = xl
      //df/dxr += df/dxi * dxi/xr = df/dxi * xl
      auto dr = BuildOp(BO_Mul, Clone(L), dfdx());
      auto drTmp = StoreAndRef(dr);
      Visit(R, drTmp);
    }
    else if (opCode == BO_Div) {
      //xi = xl / xr
      //dxi/xl = 1 / xr
      //df/dxl += df/dxi * dxi/xl = df/dxi * (1/xr)
      auto clonedR = Clone(R);
      auto dl = BuildOp(BO_Div, dfdx(), clonedR);
      auto dlTmp = StoreAndRef(dl);
      Visit(L, dlTmp);
      //dxi/xr = -xl / (xr * xr)
      //df/dxl += df/dxi * dxi/xr = df/dxi * (-xl /(xr * xr))
      // Wrap R * R in parentheses: (R * R). otherwise code like 1 / R * R is
      // produced instead of 1 / (R * R).
      auto RxR =
        m_Sema.ActOnParenExpr(noLoc,
                              noLoc,
                              BuildOp(BO_Mul, clonedR, clonedR)).get();
      auto dr =
        BuildOp(BO_Mul,
                dfdx(),
                BuildOp(UO_Minus,
                        BuildOp(BO_Div, Clone(L), RxR)));
      auto drTmp = StoreAndRef(dr);
      Visit(R, drTmp);
    }
    else
      llvm_unreachable("unsupported binary operator");
  }

  void ReverseModeVisitor::VisitDeclStmt(const DeclStmt* DS) {
    llvm_unreachable("declarations are not supported yet");
  }

  void ReverseModeVisitor::VisitImplicitCastExpr(const ImplicitCastExpr* ICE) {
    Visit(ICE->getSubExpr(), dfdx());
  }

  void ReverseModeVisitor::VisitMemberExpr(const MemberExpr* ME) {
    // We do not treat struct members as independent variables, so they are not
    // differentiated.
  }


} // end namespace clad
