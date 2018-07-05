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
  static SourceLocation noLoc{};

  CompoundStmt* NodeContext::wrapInCompoundStmt(clang::ASTContext& C) const {
    assert(!isSingleStmt() && "Must be more than 1");
    llvm::ArrayRef<Stmt*> stmts
    = llvm::makeArrayRef(m_Stmts.data(), m_Stmts.size());
    clang::SourceLocation noLoc;
    return new (C) clang::CompoundStmt(C, stmts, noLoc, noLoc);
  }
  
  DerivativeBuilder::DerivativeBuilder(clang::Sema& S)
    : m_Sema(S), m_Context(S.getASTContext()),
      m_NodeCloner(new utils::StmtClone(m_Context)) {
    // Find the builtin derivatives namespace
    DeclarationName Name = &m_Context.Idents.get("custom_derivatives");
    LookupResult R(m_Sema, Name, SourceLocation(), Sema::LookupNamespaceName,
                   Sema::ForRedeclaration);
    m_Sema.LookupQualifiedName(R, m_Context.getTranslationUnitDecl(),
                               /*allowBuiltinCreation*/ false);
    assert(!R.empty() && "Cannot find builtin derivatives!");
    m_BuiltinDerivativesNSD = cast<NamespaceDecl>(R.getFoundDecl());
  }

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

  FunctionDecl* DerivativeBuilder::Derive(
    FunctionDeclInfo& FDI,
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

  CompoundStmt* VisitorBase::MakeCompoundStmt(
    const llvm::SmallVector<clang::Stmt*, 16> & Stmts) {
    auto Stmts_ref = llvm::makeArrayRef(Stmts.data(), Stmts.size());
    return new (m_Context) clang::CompoundStmt(
      m_Context,
      Stmts_ref,
      noLoc,
      noLoc);
  }

  ForwardModeVisitor::ForwardModeVisitor(DerivativeBuilder& builder):
    VisitorBase(builder) {}

  ForwardModeVisitor::~ForwardModeVisitor() {}

  FunctionDecl* ForwardModeVisitor::Derive(
    FunctionDeclInfo& FDI,
    const DiffPlan& plan) {
    FunctionDecl* FD = FDI.getFD();
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
    }
    else {
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

    llvm::SmallVector<ParmVarDecl*, 4> params;
    ParmVarDecl* newPVD = 0;
    ParmVarDecl* PVD = 0;

    // We will use the m_CurScope to do the needed lookups.
    m_CurScope.reset(new Scope(m_Sema.TUScope, Scope::FnScope,
                               m_Sema.getDiagnostics()));

    // FIXME: We should implement FunctionDecl and ParamVarDecl cloning.
    for(size_t i = 0, e = FD->getNumParams(); i < e; ++i) {
      PVD = FD->getParamDecl(i);
      Expr* clonedPVDDefaultArg = 0;
      if (PVD->hasDefaultArg())
        clonedPVDDefaultArg = Clone(PVD->getDefaultArg()).getExpr();

      newPVD = ParmVarDecl::Create(m_Context, derivedFD, noLoc, noLoc,
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
      if (newPVD->getIdentifier()) {
        m_CurScope->AddDecl(newPVD);
        m_Sema.IdResolver.AddDecl(newPVD);
      }
    }
    llvm::ArrayRef<ParmVarDecl*> paramsRef
      = llvm::makeArrayRef(params.data(), params.size());
    derivedFD->setParams(paramsRef);
    derivedFD->setBody(0);

    // This is creating a 'fake' function scope. See SemaDeclCXX.cpp
    Sema::SynthesizedFunctionScope Scope(m_Sema, derivedFD);
    Stmt* derivativeBody = Visit(FD->getMostRecentDecl()->getBody()).getStmt();

    derivedFD->setBody(derivativeBody);
    // Cleanup the IdResolver chain.
    for(FunctionDecl::param_iterator I = derivedFD->param_begin(),
        E = derivedFD->param_end(); I != E; ++I) {
      if ((*I)->getIdentifier()) {
        m_CurScope->RemoveDecl(*I);
        //m_Sema.IdResolver.RemoveDecl(*I); // FIXME: Understand why that's bad
      }
    }

    m_DerivativeInFlight = false;
    return derivedFD;
  }

  Stmt* DerivativeBuilder::Clone(const Stmt* S) {
    Stmt* clonedStmt = m_NodeCloner->Clone(S);
    updateReferencesOf(clonedStmt);
    return clonedStmt;
  }
  Expr* DerivativeBuilder::Clone(const Expr* E) {
    const Stmt* S = E;
    return llvm::cast<Expr>(Clone(S));
  }

  Expr* DerivativeBuilder::BuildOp(UnaryOperatorKind OpCode, Expr* E) {
    return m_Sema.BuildUnaryOp(nullptr, noLoc, OpCode, E).get();
  }
  Expr* DerivativeBuilder::BuildOp(
    clang::BinaryOperatorKind OpCode,
    Expr* L,
    Expr* R) {
    return m_Sema.BuildBinOp(nullptr, noLoc, OpCode, L, R).get();
  }

  NodeContext ForwardModeVisitor::Clone(const Stmt* S) {
    return NodeContext(m_Builder.Clone(S));
  }

  NodeContext ForwardModeVisitor::VisitStmt(const Stmt* S) {
    return Clone(S);
  }

  NodeContext ForwardModeVisitor::VisitCompoundStmt(const CompoundStmt* CS) {
    llvm::SmallVector<Stmt*, 16> stmts;
    for (CompoundStmt::const_body_iterator I = CS->body_begin(),
           E = CS->body_end(); I != E; ++I)
      stmts.push_back(Visit(*I).getStmt());

    llvm::ArrayRef<Stmt*> stmtsRef(stmts.data(), stmts.size());
    return new (m_Context) CompoundStmt(m_Context, stmtsRef, noLoc, noLoc);
  }

  NodeContext ForwardModeVisitor::VisitIfStmt(const IfStmt* If) {
    IfStmt* clonedIf = Clone(If).getAs<IfStmt>();
    clonedIf->setThen(Visit(clonedIf->getThen()).getStmt());
    if (clonedIf->getElse())
      clonedIf->setElse(Visit(clonedIf->getElse()).getStmt());
    return NodeContext(clonedIf);
  }

  NodeContext ForwardModeVisitor::VisitConditionalOperator(
    const ConditionalOperator* CO) {
    auto cond = Clone(CO->getCond()).getExpr();
    auto ifTrue = Visit(CO->getTrueExpr()).getExpr();
    auto ifFalse = Visit(CO->getFalseExpr()).getExpr();

    auto condExpr =
      new (m_Context) ConditionalOperator(
        cond,
        noLoc,
        ifTrue,
        noLoc,
        ifFalse,
        ifTrue->getType(),
        VK_RValue, // FIXME: check if we do not need lvalue sometimes
        OK_Ordinary);
    // For some reason clang would not geterate parentheses to keep the correct
    // order.
    auto parens =
      new (m_Context) ParenExpr(
        noLoc,
        noLoc,
        condExpr);

    return NodeContext(parens);
  }

  NodeContext ForwardModeVisitor::VisitReturnStmt(const ReturnStmt* RS) {
     //ReturnStmt* clonedStmt = m_NodeCloner->Clone(RS);
    Expr* retVal = Visit(Clone(RS->getRetValue()).getExpr()).getExpr();

    // Note here getCurScope is the TU unit, since we've done parsing and there
    // is no active scope.
    Stmt* clonedStmt = m_Sema.ActOnReturnStmt(
      noLoc,
      retVal,
      m_Sema.getCurScope()).get();
    return NodeContext(clonedStmt);
  }
  
  NodeContext ForwardModeVisitor::VisitParenExpr(const ParenExpr* PE) {
    ParenExpr* clonedPE = Clone(PE).getAs<ParenExpr>();
    Expr* retVal = Visit(clonedPE->getSubExpr()).getExpr();
    clonedPE->setSubExpr(retVal);
    clonedPE->setType(retVal->getType());
    return NodeContext(clonedPE);
  }

  NodeContext ForwardModeVisitor::VisitMemberExpr(const MemberExpr* ME) {
    MemberExpr* clonedME = Clone(ME).getAs<MemberExpr>();
    // Copy paste from VisitDeclRefExpr.
    QualType Ty = ME->getType();
    if (clonedME->getMemberDecl() == m_IndependentVar)
      return ConstantFolder::synthesizeLiteral(Ty, m_Context, 1);
    return ConstantFolder::synthesizeLiteral(Ty, m_Context, 0);
  }

  NodeContext ForwardModeVisitor::VisitDeclRefExpr(const DeclRefExpr* DRE) {
    DeclRefExpr* clonedDRE = Clone(DRE).getAs<DeclRefExpr>();
    QualType Ty = DRE->getType();
    if (clonedDRE->getDecl() == m_IndependentVar)
      // Return 1 literal if this is the independent variable.
      return ConstantFolder::synthesizeLiteral(Ty, m_Context, 1);
    return ConstantFolder::synthesizeLiteral(Ty, m_Context, 0);
  }

  NodeContext ForwardModeVisitor::VisitIntegerLiteral(
    const IntegerLiteral* IL) {
    llvm::APInt zero(m_Context.getIntWidth(m_Context.IntTy), /*value*/0);
    IntegerLiteral* constant0 = IntegerLiteral::Create(m_Context, zero,
                                                       m_Context.IntTy,
                                                       noLoc);
    return NodeContext(constant0);
  }

  NodeContext ForwardModeVisitor::VisitFloatingLiteral(
    const FloatingLiteral* FL) {
    FloatingLiteral* clonedStmt = Clone(FL).getAs<FloatingLiteral>();
    llvm::APFloat zero = llvm::APFloat::getZero(clonedStmt->getSemantics());
    clonedStmt->setValue(m_Context, zero);
    return NodeContext(clonedStmt);
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
          CandidateSet.BestViableFunction(
            m_Sema,
            UnresolvedLookup->getLocStart(), Best);
          if (OverloadResult) // No overloads were found.
            return true;
        }
      }
    }
    return false;
  }
  
  Expr* DerivativeBuilder::findOverloadedDefinition
  (DeclarationNameInfo DNI, llvm::SmallVector<Expr*, 4> CallArgs) {
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
  
  NodeContext ForwardModeVisitor::VisitCallExpr(const CallExpr* CE) {
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
      if (!Multiplier)
        Multiplier = Visit(CE->getArg(i)).getExpr();
      else {
        Multiplier =
          BuildOp(BO_Add, Multiplier, Visit(CE->getArg(i)).getExpr());
      }
      CallArgs.push_back(Clone(CE->getArg(i)).getExpr());
    }

    if (Multiplier)
      Multiplier = m_Sema.ActOnParenExpr(noLoc, noLoc, Multiplier).get();

    Expr* OverloadedDerivedFn =
      m_Builder.findOverloadedDefinition(DNInfo, CallArgs);
    if (OverloadedDerivedFn) {
      if (Multiplier)
        return BuildOp(BO_Mul, OverloadedDerivedFn, Multiplier);
      return NodeContext(OverloadedDerivedFn);
    }

    Expr* OverloadedFnInFile
       = m_Builder.findOverloadedDefinition(
           CE->getDirectCallee()->getNameInfo(), CallArgs);

    if (OverloadedFnInFile) {
      // Take the function to derive from the source.
      const FunctionDecl* FD = CE->getDirectCallee();
      // Get the definition, if any.
      const FunctionDecl* mostRecentFD = FD->getMostRecentDecl();
      while (mostRecentFD && !mostRecentFD->isThisDeclarationADefinition()) {
        mostRecentFD = mostRecentFD->getPreviousDecl();
      }
      if (!mostRecentFD || !mostRecentFD->isThisDeclarationADefinition()) {
        SourceLocation IdentifierLoc = FD->getLocEnd();
        unsigned err_differentiating_undefined_function
          = m_Sema.Diags.getCustomDiagID(DiagnosticsEngine::Error,
                                         "attempted differention of function "
                                      "'%0', which does not have a definition");
        m_Sema.Diag(IdentifierLoc, err_differentiating_undefined_function)
          << FD->getNameAsString();
        return NodeContext(0);
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
      CallExpr* clonedCE = Clone(CE).getAs<CallExpr>();
      clonedCE->setCallee(ResolvedLookup);
      return NodeContext(clonedCE);
    }

    // Function was not derived => issue a warning.
    SourceLocation IdentifierLoc = CE->getDirectCallee()->getLocEnd();
    unsigned warn_function_not_declared_in_custom_derivatives
      = m_Sema.Diags.getCustomDiagID(DiagnosticsEngine::Warning,
                                     "function '%0' was not differentiated "
                                     "because it is not declared in namespace "
                                     "'custom_derivatives' attempted "
                                     "differention of function '%0', which "
                                     "does not have a definition");
    m_Sema.Diag(IdentifierLoc, warn_function_not_declared_in_custom_derivatives)
      << CE->getDirectCallee()->getNameAsString();
    return Clone(CE);
  }
  
  void DerivativeBuilder::updateReferencesOf(Stmt* InSubtree) {
    utils::ReferencesUpdater up(m_Sema, m_NodeCloner.get(), m_CurScope.get());
    up.TraverseStmt(InSubtree);
  }

  NodeContext ForwardModeVisitor::VisitUnaryOperator(
    const UnaryOperator* UnOp) {
    UnaryOperator* clonedUnOp = Clone(UnOp).getAs<UnaryOperator>();
    clonedUnOp->setSubExpr(Visit(clonedUnOp->getSubExpr()).getExpr());
    return NodeContext(clonedUnOp);
  }

  NodeContext ForwardModeVisitor::VisitBinaryOperator(
    const BinaryOperator* BinOp) {
    BinaryOperator* clonedBO = Clone(BinOp).getAs<BinaryOperator>();
    m_Builder.updateReferencesOf(clonedBO->getRHS());
    m_Builder.updateReferencesOf(clonedBO->getLHS());

    Expr* lhs_derived = Visit(clonedBO->getLHS()).getExpr();
    Expr* rhs_derived = Visit(clonedBO->getRHS()).getExpr();

    ConstantFolder folder(m_Context);
    BinaryOperatorKind opCode = clonedBO->getOpcode();
    if (opCode == BO_Mul || opCode == BO_Div) {
      Expr* newBOLHS = BuildOp(BO_Mul, lhs_derived, clonedBO->getRHS());
      //newBOLHS = folder.fold(cast<BinaryOperator>(newBOLHS));
      Expr* newBORHS = BuildOp(BO_Mul, clonedBO->getLHS(), rhs_derived);
      //newBORHS = folder.fold(cast<BinaryOperator>(newBORHS));
      if (opCode == BO_Mul) {
        Expr* newBO_Add = BuildOp(BO_Add, newBOLHS, newBORHS);


        Expr* PE = m_Sema.ActOnParenExpr(noLoc, noLoc, newBO_Add).get();
        return NodeContext(folder.fold(PE));
      }
      else if (opCode == BO_Div) {
        Expr* newBO_Sub = BuildOp(BO_Sub, newBOLHS, newBORHS);

        Expr* newBO_Mul_denom =
          BuildOp(BO_Mul, clonedBO->getRHS(), clonedBO->getRHS());

        Expr* PE_lhs =
          m_Sema.ActOnParenExpr(noLoc, noLoc, newBO_Sub).get();
        Expr* PE_rhs =
          m_Sema.ActOnParenExpr(noLoc, noLoc, newBO_Mul_denom).get();

        Expr* newBO_Div = BuildOp(BO_Div, PE_lhs, PE_rhs);

        return NodeContext(folder.fold(newBO_Div));
      }
    }
    else if (opCode == BO_Add || opCode == BO_Sub) {
      // enforce precedence for substraction
      rhs_derived = m_Sema.ActOnParenExpr(noLoc, noLoc, rhs_derived).get();
      BinaryOperator* newBO =
        dyn_cast<BinaryOperator>(BuildOp(opCode, lhs_derived, rhs_derived));
      assert(m_Context.hasSameUnqualifiedType(newBO->getLHS()->getType(),
                                              newBO->getRHS()->getType())
             && "Must be the same types.");


      return NodeContext(folder.fold(newBO));
    }

    if (!clonedBO->isAssignmentOp()) // Skip LHS in assignments.
      clonedBO->setLHS(lhs_derived);
    clonedBO->setRHS(rhs_derived);

    return NodeContext(folder.fold(clonedBO));
  }

  NodeContext ForwardModeVisitor::VisitDeclStmt(const DeclStmt* DS) {
    DeclStmt* clonedDS = Clone(DS).getAs<DeclStmt>();
    // Iterate through the declaration(s) contained in DS.
    for (DeclStmt::decl_iterator I = clonedDS->decl_begin(),
         E = clonedDS->decl_end(); I != E; ++I) {
      if (VarDecl* VD = dyn_cast<VarDecl>(*I)) {
        m_CurScope->AddDecl(VD);
        //TODO: clean the idResolver chain!!!!!
        if (VD->getIdentifier())
          m_Sema.IdResolver.AddDecl(VD);
      }
    }
    return NodeContext(clonedDS);
  }

  NodeContext ForwardModeVisitor::VisitImplicitCastExpr(
    const ImplicitCastExpr* ICE) {
    NodeContext result = Visit(ICE->getSubExpr());
    if (result.getExpr() == ICE->getSubExpr())
      return NodeContext(Clone(ICE).getExpr());
    return NodeContext(result.getExpr());
  }

  NodeContext
  ForwardModeVisitor::VisitCXXOperatorCallExpr(
    const CXXOperatorCallExpr* OpCall) {
    // This operator gets emitted when there is a binary operation containing
    // overloaded operators. Eg. x+y, where operator+ is overloaded.
    assert(0 && "We don't support overloaded operators yet!");
    return Clone(OpCall);
  }

  Stmt* ReverseModeVisitor::Clone(const Stmt* s) {
    return m_Builder.Clone(s);
  }
  Expr* ReverseModeVisitor::Clone(const Expr* e) {
    return m_Builder.Clone(e);
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
    std::transform(m_Function->param_begin(), m_Function->param_end(), std::begin(paramTypes),
      [] (const ParmVarDecl* PVD) {
        return PVD->getType();
      });
    // The last parameter is the output parameter of the R* type.
    paramTypes.back() = m_Context.getPointerType(m_Function->getReturnType());
    // For a function f of type R(A1, A2, ..., An),
    // the type of the gradient function is void(A1, A2, ..., An, R*).
    auto gradientFunctionType = m_Context.getFunctionType(
      m_Context.VoidTy,
      llvm::ArrayRef<QualType>(paramTypes.data(), paramTypes.size()),
      FunctionProtoType::ExtProtoInfo()); // Should we do something with ExtProtoInfo?

    // Create the gradient function declaration.
    auto gradientFD = FunctionDecl::Create(
      m_Context,
      m_Function->getDeclContext(),
      noLoc,
      name,
      gradientFunctionType,
      m_Function->getTypeSourceInfo(), // What is TypeSourceInfo for?
      m_Function->getStorageClass(),
      m_Function->isInlineSpecified(),
      m_Function->hasWrittenPrototype(),
      m_Function->isConstexpr());

    m_CurScope.reset(new Scope(m_Sema.TUScope, Scope::FnScope,
                               m_Sema.getDiagnostics()));

    // Create parameter declarations.
    llvm::SmallVector<ParmVarDecl*, 4> params(paramTypes.size());
    std::transform(m_Function->param_begin(), m_Function->param_end(), std::begin(params),
      [&] (const ParmVarDecl* PVD) {
        auto VD = ParmVarDecl::Create(
          m_Context,
          gradientFD,
          noLoc,
          noLoc,
          PVD->getIdentifier(),
          PVD->getType(),
          PVD->getTypeSourceInfo(), // Is it OK?
          PVD->getStorageClass(),
          nullptr); // No default values.
        if (VD->getIdentifier()) {
          m_CurScope->AddDecl(VD);
          m_Sema.IdResolver.AddDecl(VD);
        }
        return VD;
      });
    // The output paremeter.
    params.back() =
      ParmVarDecl::Create(
        m_Context,
        gradientFD,
        noLoc,
        noLoc,
        &m_Context.Idents.get("_result"), // We name it "_result".
        paramTypes.back(),
        m_Context.getTrivialTypeSourceInfo(paramTypes.back(), noLoc),
        params.front()->getStorageClass(),
        nullptr); // No default value.
    if (params.back()->getIdentifier()) {
      m_CurScope->AddDecl(params.back());
      m_Sema.IdResolver.AddDecl(params.back());
    }

    llvm::ArrayRef<ParmVarDecl*> paramsRef =
      llvm::makeArrayRef(params.data(), params.size());
    gradientFD->setParams(paramsRef);
    gradientFD->setBody(nullptr);

    Sema::SynthesizedFunctionScope Scope(m_Sema, gradientFD);
    // Reference to the output parameter.
    m_Result = m_Sema.BuildDeclRefExpr(
      params.back(),
      paramTypes.back(),
      VK_LValue,
      noLoc).get();
    // Initially, df/df = 1.
    auto dfdf =
      ConstantFolder::synthesizeLiteral(
        m_Function->getReturnType(),
        m_Context,
        1.0);

    auto bodyStmts = startBlock();
    // Start the visitation process which outputs the statements in the current
    // block.
    auto functionBody = m_Function->getMostRecentDecl()->getBody();
    Visit(functionBody, dfdf);
    // Create the body of the function.
    auto gradientBody = finishBlock();

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
        assert(false && "variable declarations are not currently supported");
    auto cond = Clone(If->getCond());
    auto thenStmt = If->getThen();
    auto elseStmt = If->getElse();
   
    Stmt* thenBody = nullptr;
    Stmt* elseBody = nullptr;
    if (thenStmt) {
      auto thenBlock = startBlock();
      Visit(thenStmt, dfdx());
      thenBody = finishBlock();
    }
    if (elseStmt) {
      auto elseBlock = startBlock();
      Visit(elseStmt, dfdx());
      elseBody = finishBlock();
    }

    auto ifStmt = new (m_Context) IfStmt(
      m_Context,
      noLoc,
      If->isConstexpr(),
      nullptr, // FIXME: add init for condition variable
      nullptr, // FIXME: add condition variable decl
      cond,
      thenBody, noLoc,
      elseBody);
    currentBlock().push_back(ifStmt);  
  }

  void ReverseModeVisitor::VisitConditionalOperator(
    const clang::ConditionalOperator* CO) {
    auto cond = Clone(CO->getCond());
    auto ifTrue = CO->getTrueExpr();
    auto ifFalse = CO->getFalseExpr();

    auto VisitBranch =
      [&] (Stmt* branch, Expr* ifTrue, Expr* ifFalse) {
        if (!branch)
          return;
        auto condExpr =
          new (m_Context) ConditionalOperator(
            cond,
            noLoc,
            ifTrue,
            noLoc,
            ifFalse,
            ifTrue->getType(),
            VK_RValue,
            OK_Ordinary);
        // For some reason clang would not geterate parentheses to keep the correct
        // order.
        auto dStmt =
          new (m_Context) ParenExpr(
            noLoc,
            noLoc,
            condExpr);

        Visit(branch, dStmt);
    };
   
    // FIXME: not optimal, creates two (condExpr ? ... : ...) expressions,
    // so cond is unnesarily checked twice. 
    // Can be improved by storing the result of condExpr in a temporary.

    auto zero = ConstantFolder::synthesizeLiteral(dfdx()->getType(), m_Context, 0);
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
    auto it = std::find(
        m_Function->param_begin(),
        m_Function->param_end(),
        decl);

    if (it == m_Function->param_end())
        assert(false && "Only references to function args are currently supported");
    auto idx = std::distance(
        m_Function->param_begin(),
        it);
    auto size_type = m_Context.getSizeType();
    auto size_type_bits = m_Context.getIntWidth(size_type);
    // Create the idx literal.
    auto i =
      IntegerLiteral::Create(
        m_Context,
        llvm::APInt(size_type_bits, idx),
        size_type,
        noLoc);
    // Create the _result[idx] expression.
    auto result_at_i =
      m_Sema.CreateBuiltinArraySubscriptExpr(
        m_Result,
        noLoc,
        i,
        noLoc).get();
    // Create the (_result[idx] += dfdx) statement.
    auto add_assign = BuildOp(BO_AddAssign, result_at_i, dfdx());
    // Add it to the body statements.
    currentBlock().push_back(add_assign);
  }

  void ReverseModeVisitor::VisitIntegerLiteral(const IntegerLiteral* IL) {
    // Nothing to do with it.
  }

  void ReverseModeVisitor::VisitFloatingLiteral(const FloatingLiteral* FL) {
    // Nothing to do with it.
  }
  
  void ReverseModeVisitor::VisitCallExpr(const CallExpr* CE) {
    assert(false && "calling functions is not supported yet");
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
      assert(false && "unsupported unary operator");
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
      Visit(L, dl);
      //dxi/xr = xl
      //df/dxr += df/dxi * dxi/xr = df/dxi * xl
      auto dr = BuildOp(BO_Mul, Clone(L), dfdx());
      Visit(R, dr);
    }
    else if (opCode == BO_Div) {
      //xi = xl / xr
      //dxi/xl = 1 / xr
      //df/dxl += df/dxi * dxi/xl = df/dxi * (1/xr)
      auto one = ConstantFolder::synthesizeLiteral(R->getType(), m_Context, 1.0);
      auto clonedR = Clone(R);
      auto dl = BuildOp(BO_Div, one, clonedR);
      Visit(L, dl);
      //dxi/xr = -xl / (xr * xr)
      //df/dxl += df/dxi * dxi/xr = df/dxi * (-xl /(xr * xr))
      auto RxR = BuildOp(BO_Mul, clonedR, clonedR);
      auto dr = BuildOp(
        UO_Minus,
        BuildOp(BO_Div, Clone(L), RxR));
      Visit(R, dr);
    }
    else
      assert(false && "unsupported binary operator");
  }

  void ReverseModeVisitor::VisitDeclStmt(const DeclStmt* DS) {
    assert(false && "declarations are not supported yet");
  }

  void ReverseModeVisitor::VisitImplicitCastExpr(const ImplicitCastExpr* ICE) {
    Visit(ICE->getSubExpr(), dfdx());
  }

  void ReverseModeVisitor::VisitMemberExpr(const MemberExpr* ME) {
    assert(false && "not supported yet");
  }

  
} // end namespace clad
