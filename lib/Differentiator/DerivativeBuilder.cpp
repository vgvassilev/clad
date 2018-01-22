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
  CompoundStmt* NodeContext::wrapInCompoundStmt(clang::ASTContext& C) const {
    assert(!isSingleStmt() && "Must be more than 1");
    llvm::ArrayRef<Stmt*> stmts
    = llvm::makeArrayRef(m_Stmts.data(), m_Stmts.size());
    clang::SourceLocation noLoc;
    return new (C) clang::CompoundStmt(C, stmts, noLoc, noLoc);
  }
  
  DerivativeBuilder::DerivativeBuilder(clang::Sema& S)
    : m_Sema(S), m_Context(S.getASTContext()), m_IndependentVar(0),
      m_DerivativeInFlight(false), m_DerivativeOrder(0) {
    m_NodeCloner.reset(new utils::StmtClone(m_Context));
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

  FunctionDecl* DerivativeBuilder::Derive(FunctionDeclInfo& FDI, DiffPlan* plan) {
    clang::FunctionDecl* FD = FDI.getFD();
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

    if (!m_NodeCloner) {
      m_NodeCloner.reset(new utils::StmtClone(m_Context));
    }

    SourceLocation noLoc;
    m_DerivativeOrder = plan->getCurrentDerivativeOrder();
    std::string s = std::to_string(m_DerivativeOrder);
    std::string derivativeBaseName;
    if (m_DerivativeOrder == 1)
      s = "";
    switch (FD->getOverloadedOperator()) {
    default:
      derivativeBaseName = plan->begin()->getFD()->getNameAsString();
      break;
    case OO_Call:
      derivativeBaseName = "operator_call";
      break;
    }

    m_ArgIndex = plan->getArgIndex();
    IdentifierInfo* II = &m_Context.Idents.get(derivativeBaseName + "_d" + s +
                                               "arg" + std::to_string(m_ArgIndex));
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
        clonedPVDDefaultArg = VisitStmt(PVD->getDefaultArg()).getExpr();

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

    registerDerivative(derivedFD, m_Sema);

    m_DerivativeInFlight = false;
    return derivedFD;
    // DiffPlans plans;
    // plans.push_back(plan);
    // DiffCollector collector(DeclGroupRef(derivedFD), plans, m_Sema);
    // return derivedFD;
  }

  NodeContext DerivativeBuilder::VisitStmt(const Stmt* S) {
    Stmt* clonedStmt = m_NodeCloner->Clone(S);
    updateReferencesOf(clonedStmt);
    return NodeContext(clonedStmt);
  }

  NodeContext DerivativeBuilder::VisitCompoundStmt(const CompoundStmt* CS) {
    llvm::SmallVector<Stmt*, 16> stmts;
    for (CompoundStmt::const_body_iterator I = CS->body_begin(),
           E = CS->body_end(); I != E; ++I)
      stmts.push_back(Visit(*I).getStmt());

    llvm::ArrayRef<Stmt*> stmtsRef(stmts.data(), stmts.size());
    SourceLocation noLoc;
    return new (m_Context) CompoundStmt(m_Context, stmtsRef, noLoc, noLoc);
  }

  NodeContext DerivativeBuilder::VisitIfStmt(const IfStmt* If) {
    IfStmt* clonedIf = VisitStmt(If).getAs<IfStmt>();
    clonedIf->setThen(Visit(clonedIf->getThen()).getStmt());
    if (clonedIf->getElse())
      clonedIf->setElse(Visit(clonedIf->getElse()).getStmt());
    return NodeContext(clonedIf);
  }

  NodeContext DerivativeBuilder::VisitReturnStmt(const ReturnStmt* RS) {
     //ReturnStmt* clonedStmt = m_NodeCloner->Clone(RS);
    Expr* retVal = Visit(VisitStmt(RS->getRetValue()).getExpr()).getExpr();
    SourceLocation noLoc;

    // Note here getCurScope is the TU unit, since we've done parsing and there
    // is no active scope.
    Stmt* clonedStmt = m_Sema.ActOnReturnStmt(noLoc, retVal, m_Sema.getCurScope()).get();
    return NodeContext(clonedStmt);
  }
  
  NodeContext DerivativeBuilder::VisitParenExpr(const ParenExpr* PE) {
    ParenExpr* clonedPE = VisitStmt(PE).getAs<ParenExpr>();
    Expr* retVal = Visit(clonedPE->getSubExpr()).getExpr();
    clonedPE->setSubExpr(retVal);
    clonedPE->setType(retVal->getType());
    return NodeContext(clonedPE);
  }

  NodeContext DerivativeBuilder::VisitMemberExpr(const MemberExpr* ME) {
    MemberExpr* clonedME = VisitStmt(ME).getAs<MemberExpr>();
    // Copy paste from VisitDeclRefExpr.
    QualType Ty = ME->getType();
    if (clonedME->getMemberDecl() == m_IndependentVar)
      return ConstantFolder::synthesizeLiteral(Ty, m_Context, 1);
    return ConstantFolder::synthesizeLiteral(Ty, m_Context, 0);
  }

  NodeContext DerivativeBuilder::VisitDeclRefExpr(const DeclRefExpr* DRE) {
    DeclRefExpr* clonedDRE = VisitStmt(DRE).getAs<DeclRefExpr>();
    SourceLocation noLoc;
    QualType Ty = DRE->getType();
    if (clonedDRE->getDecl()->getNameAsString() ==
        m_IndependentVar->getNameAsString())
      // Return 1 literal if this is the independent variable.
      return ConstantFolder::synthesizeLiteral(Ty, m_Context, 1);
    return ConstantFolder::synthesizeLiteral(Ty, m_Context, 0);
  }

  NodeContext DerivativeBuilder::VisitIntegerLiteral(const IntegerLiteral* IL) {
    SourceLocation noLoc;
    llvm::APInt zero(m_Context.getIntWidth(m_Context.IntTy), /*value*/0);
    IntegerLiteral* constant0 = IntegerLiteral::Create(m_Context, zero,
                                                       m_Context.IntTy,
                                                       noLoc);
    return NodeContext(constant0);
  }

  NodeContext DerivativeBuilder::VisitFloatingLiteral(const FloatingLiteral* FL) {
    FloatingLiteral* clonedStmt = m_NodeCloner->Clone(FL);
    llvm::APFloat zero = llvm::APFloat::getZero(clonedStmt->getSemantics());
    clonedStmt->setValue(m_Context, zero);
    return NodeContext(clonedStmt);
  }

  // This method is derived from the source code of both buildOverloadedCallSet()
  // in SemaOverload.cpp and ActOnCallExpr() in SemaExpr.cpp.
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
  
  NodeContext DerivativeBuilder::VisitCallExpr(const CallExpr* CE) {
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
        Multiplier = m_Sema.BuildBinOp(/*Scope*/0, noLoc, BO_Add, Multiplier,
                                       Visit(CE->getArg(i)).getExpr()).get();
      }
      CallArgs.push_back(VisitStmt(CE->getArg(i)).getExpr());
    }

    if (Multiplier)
      Multiplier = m_Sema.ActOnParenExpr(noLoc, noLoc, Multiplier).get();

    Expr* OverloadedDerivedFn = findOverloadedDefinition(DNInfo, CallArgs);
    if (OverloadedDerivedFn) {
      if (Multiplier)
        return m_Sema.BuildBinOp(/*Scope*/0, SourceLocation(),
                                 BO_Mul, OverloadedDerivedFn, Multiplier).get();
      return NodeContext(OverloadedDerivedFn);
    }

    Expr* OverloadedFnInFile
       = findOverloadedDefinition(CE->getDirectCallee()->getNameInfo(), CallArgs);

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
      m_Sema.LookupQualifiedName(R, m_BuiltinDerivativesNSD,
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
      CallExpr* clonedCE = VisitStmt(CE).getAs<CallExpr>();
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
    return VisitStmt(CE);
  }

  namespace {
    class Updater : public RecursiveASTVisitor<Updater> {
    private:
      Sema& m_Sema; // We don't own.
      utils::StmtClone* m_NodeCloner; // We don't own.
      Scope* m_CurScope; // We don't own.
    public:
      Updater(Sema& SemaRef, utils::StmtClone* C, Scope* S) 
        : m_Sema(SemaRef), m_NodeCloner(C), m_CurScope(S) {}
      bool VisitDeclRefExpr(DeclRefExpr* DRE) {
        // If the declaration's decl context encloses the derivative's decl
        // context we must not update anything.
        if (DRE->getDecl()->getDeclContext()->Encloses(m_Sema.CurContext)) {
          return true;
        }
        DeclarationNameInfo DNI = DRE->getNameInfo();

        LookupResult R(m_Sema, DNI, Sema::LookupOrdinaryName);
        m_Sema.LookupName(R, m_CurScope, /*allowBuiltinCreation*/ false);

        if (R.empty())
          return true;  // Nothing to update.

       // FIXME: Handle the case when there are overloads found. Update
       // it with the best match.
       //
       // FIXME: This is the right way to go in principe, however there is no
       // properly built decl context.
       // m_Sema.MarkDeclRefReferenced(clonedDRE);
       if (!R.isSingleResult())
         return true;

        if (ValueDecl* VD = dyn_cast<ValueDecl>(R.getFoundDecl())) {
          DRE->setDecl(VD);
          VD->setReferenced();
          VD->setIsUsed();
        }
        return true;
      }
    };
  } // end anon namespace
  
  void DerivativeBuilder::updateReferencesOf(Stmt* InSubtree) {
    Updater up(m_Sema, m_NodeCloner.get(), m_CurScope.get());
    up.TraverseStmt(InSubtree);
  }

  NodeContext DerivativeBuilder::VisitUnaryOperator(const UnaryOperator* UnOp) {
    UnaryOperator* clonedUnOp = VisitStmt(UnOp).getAs<UnaryOperator>();
    clonedUnOp->setSubExpr(Visit(clonedUnOp->getSubExpr()).getExpr());
    return NodeContext(clonedUnOp);
  }

  NodeContext DerivativeBuilder::VisitBinaryOperator(const BinaryOperator* BinOp) {
    BinaryOperator* clonedBO = VisitStmt(BinOp).getAs<BinaryOperator>();
    updateReferencesOf(clonedBO->getRHS());
    updateReferencesOf(clonedBO->getLHS());

    Expr* lhs_derived = Visit(clonedBO->getLHS()).getExpr();
    Expr* rhs_derived = Visit(clonedBO->getRHS()).getExpr();

    SourceLocation noLoc;
    ConstantFolder folder(m_Context);
    BinaryOperatorKind opCode = clonedBO->getOpcode();
    if (opCode == BO_Mul || opCode == BO_Div) {
      Expr* newBOLHS = m_Sema.BuildBinOp(/*Scope*/0, noLoc, BO_Mul, lhs_derived, clonedBO->getRHS()).get();
      //newBOLHS = folder.fold(cast<BinaryOperator>(newBOLHS));
      Expr* newBORHS = m_Sema.BuildBinOp(/*Scope*/0, noLoc, BO_Mul, clonedBO->getLHS(), rhs_derived).get();
      //newBORHS = folder.fold(cast<BinaryOperator>(newBORHS));
      if (opCode == BO_Mul) {
        Expr* newBO_Add = m_Sema.BuildBinOp(/*Scope*/0, noLoc, BO_Add, newBOLHS,
                                            newBORHS).get();


        Expr* PE = m_Sema.ActOnParenExpr(noLoc, noLoc, newBO_Add).get();
        return NodeContext(folder.fold(PE));
      }
      else if (opCode == BO_Div) {
        Expr* newBO_Sub = m_Sema.BuildBinOp(/*Scope*/0, noLoc, BO_Sub, newBOLHS,
                                            newBORHS).get();

        Expr* newBO_Mul_denom = m_Sema.BuildBinOp(/*Scope*/0, noLoc, BO_Mul,
                                                  clonedBO->getRHS(), clonedBO->getRHS()).get();

        Expr* PE_lhs = m_Sema.ActOnParenExpr(noLoc, noLoc, newBO_Sub).get();
        Expr* PE_rhs = m_Sema.ActOnParenExpr(noLoc, noLoc, newBO_Mul_denom).get();

        Expr* newBO_Div = m_Sema.BuildBinOp(/*Scope*/0, noLoc, BO_Div,
                                            PE_lhs, PE_rhs).get();

        return NodeContext(folder.fold(newBO_Div));
      }
    }
    else if (opCode == BO_Add || opCode == BO_Sub) {
      // enforce precedence for substraction
      rhs_derived = m_Sema.ActOnParenExpr(noLoc, noLoc, rhs_derived).get();
      BinaryOperator* newBO = m_Sema.BuildBinOp(/*Scope*/0, noLoc, opCode,
                                                lhs_derived, rhs_derived).getAs<BinaryOperator>();
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

  NodeContext DerivativeBuilder::VisitDeclStmt(const DeclStmt* DS) {
    DeclStmt* clonedDS = VisitStmt(DS).getAs<DeclStmt>();
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

  NodeContext DerivativeBuilder::VisitImplicitCastExpr(const ImplicitCastExpr* ICE) {
    NodeContext result = Visit(ICE->getSubExpr());
    if (result.getExpr() == ICE->getSubExpr())
      return NodeContext(VisitStmt(ICE).getExpr());
    return NodeContext(result.getExpr());
  }

  NodeContext
  DerivativeBuilder::VisitCXXOperatorCallExpr(const CXXOperatorCallExpr* OpCall) {
    // This operator gets emitted when there is a binary operation containing
    // overloaded operators. Eg. x+y, where operator+ is overloaded.
    assert(0 && "We don't support overloaded operators yet!");
    return VisitStmt(OpCall);
  }
} // end namespace clad
