//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clad/Differentiator/DerivativeBuilder.h"
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
     : m_Sema(S), m_Context(S.getASTContext()), m_DerivedFD(0),
       m_IndependentVar(0) {
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

  FunctionDecl* DerivativeBuilder::Derive(FunctionDecl* FD, ValueDecl* argVar) {
    assert(FD && "Must not be null.");
#ifndef NDEBUG
    bool notInArgs = true;
    for (unsigned i = 0; i < FD->getNumParams(); ++i)
      if (argVar == FD->getParamDecl(i)) {
        notInArgs = false;
        break;
      }
    assert(!notInArgs && "Must pass in a param of the FD.");
#endif


    m_IndependentVar = argVar; // FIXME: Use only one var.
    
    if (!m_NodeCloner) {
      m_NodeCloner.reset(new utils::StmtClone(m_Context));
    }
    
    SourceLocation noLoc;
    IdentifierInfo* II
      = &m_Context.Idents.get(FD->getNameAsString() + "_derived_" +
                             m_IndependentVar->getNameAsString());
    DeclarationName name(II);
    FunctionDecl* derivedFD = FunctionDecl::Create(m_Context,
                                                   FD->getDeclContext(), noLoc,
                                                   noLoc, name, FD->getType(),
                                                   FD->getTypeSourceInfo(),
                                                   FD->getStorageClass(),
                                                   /*default*/
                                                   FD->isInlineSpecified(),
                                                   FD->hasWrittenPrototype(),
                                                   FD->isConstexpr()
                                                   );
    llvm::SmallVector<ParmVarDecl*, 4> params;
    ParmVarDecl* newPVD = 0;
    ParmVarDecl* PVD = 0;

    // We will use the m_CurScope to do the needed lookups.
    m_CurScope.reset(new Scope(m_Sema.TUScope, Scope::FnScope,
                               m_Sema.getDiagnostics()));

    // FIXME: We should implement FunctionDecl and ParamVarDecl cloning.
    for(size_t i = 0, e = FD->getNumParams(); i < e; ++i) {
      PVD = FD->getParamDecl(i);
      newPVD = ParmVarDecl::Create(m_Context, derivedFD, noLoc, noLoc,
                                   PVD->getIdentifier(), PVD->getType(),
                                   PVD->getTypeSourceInfo(),
                                   PVD->getStorageClass(),
                                   m_NodeCloner->Clone(PVD->getDefaultArg()));
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
    FunctionDecl* oldDerivedFD = m_DerivedFD;
    m_DerivedFD = derivedFD;
    Stmt* derivativeBody = Visit(FD->getBody()).getStmt();
    m_DerivedFD = oldDerivedFD;

    derivedFD->setBody(derivativeBody);
    // Cleanup the IdResolver chain.
    for(FunctionDecl::param_iterator I = derivedFD->param_begin(),
        E = derivedFD->param_end(); I != E; ++I) {
      if ((*I)->getIdentifier()) {
        m_CurScope->RemoveDecl(*I);
        //m_Sema.IdResolver.RemoveDecl(*I); // FIXME: Understand why that's bad
      }
    }

    return derivedFD;
  }

  NodeContext DerivativeBuilder::VisitStmt(Stmt* S) {
    return NodeContext(m_NodeCloner->Clone(S));
  }

  NodeContext DerivativeBuilder::VisitCompoundStmt(CompoundStmt* CS) {
    llvm::SmallVector<Stmt*, 16> stmts;
    for (CompoundStmt::body_iterator I = CS->body_begin(), E = CS->body_end();
         I != E; ++I)
      stmts.push_back(Visit(*I).getStmt());
    
    llvm::ArrayRef<Stmt*> stmtsRef(stmts.data(), stmts.size());
    SourceLocation noLoc;
    return new (m_Context) CompoundStmt(m_Context, stmtsRef, noLoc, noLoc);
  }

  NodeContext DerivativeBuilder::VisitIfStmt(IfStmt* If) {
    IfStmt* clonedIf = m_NodeCloner->Clone(If);
    updateReferencesOf(clonedIf->getCond());
    clonedIf->setThen(Visit(clonedIf->getThen()).getStmt());
    if (clonedIf->getElse())
      clonedIf->setElse(Visit(clonedIf->getElse()).getStmt());
    return NodeContext(clonedIf);
  }

  NodeContext DerivativeBuilder::VisitReturnStmt(ReturnStmt* RS) {
     //ReturnStmt* clonedStmt = m_NodeCloner->Clone(RS);
    Expr* retVal = Visit(RS->getRetValue()->IgnoreImpCasts()).getExpr();
    SourceLocation noLoc;

    // Note here getCurScope is the TU unit, since we've done parsing and there
    // is no active scope.
    Stmt* clonedStmt = m_Sema.ActOnReturnStmt(noLoc, retVal, m_Sema.getCurScope()).get();
    return NodeContext(clonedStmt);
  }
  
  NodeContext DerivativeBuilder::VisitParenExpr(ParenExpr* PE) {
    ParenExpr* clonedPE = m_NodeCloner->Clone(PE);
    Expr* retVal = cast<Expr>(Visit(clonedPE->getSubExpr()).getStmt());
    clonedPE->setSubExpr(retVal);
    return NodeContext(clonedPE);
  }
  
  NodeContext DerivativeBuilder::VisitDeclRefExpr(DeclRefExpr* DRE) {
    DeclRefExpr* clonedDRE = m_NodeCloner->Clone(DRE);
    SourceLocation noLoc;
    if (clonedDRE->getDecl()->getNameAsString() ==
        m_IndependentVar->getNameAsString()) {
      llvm::APInt one(m_Context.getIntWidth(m_Context.IntTy), /*value*/1);
      IntegerLiteral* constant1 = IntegerLiteral::Create(m_Context, one,
                                                         m_Context.IntTy,
                                                         noLoc);
      return NodeContext(constant1);
    }
    else {
      llvm::APInt zero(m_Context.getIntWidth(m_Context.IntTy), /*value*/0);
      IntegerLiteral* constant0 = IntegerLiteral::Create(m_Context, zero,
                                                         m_Context.IntTy,
                                                         noLoc);
      return NodeContext(constant0);
    }
  }
  
  NodeContext DerivativeBuilder::VisitIntegerLiteral(IntegerLiteral* IL) {
    SourceLocation noLoc;
    llvm::APInt zero(m_Context.getIntWidth(m_Context.IntTy), /*value*/0);
    IntegerLiteral* constant0 = IntegerLiteral::Create(m_Context, zero,
                                                       m_Context.IntTy,
                                                       noLoc);
    return NodeContext(constant0);
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
  
  NodeContext DerivativeBuilder::VisitCallExpr(CallExpr* CE) {
    // Find the built-in derivatives namespace.
    IdentifierInfo* II
      = &m_Context.Idents.get(CE->getDirectCallee()->getNameAsString() +
                              "_derived_" + m_IndependentVar->getNameAsString());
    DeclarationName name(II);
    SourceLocation DeclLoc;
    DeclarationNameInfo DNInfo(name, DeclLoc);
    
    llvm::SmallVector<Expr*, 4> CallArgs;
    for (size_t i = 0, e = CE->getNumArgs(); i < e; ++i) {
      CallArgs.push_back(m_NodeCloner->Clone(CE->getArg(i)->IgnoreImpCasts()));
    }
    
    Expr* OverloadedDerivedFn = findOverloadedDefinition(DNInfo, CallArgs);
    if (OverloadedDerivedFn) {
      return NodeContext(OverloadedDerivedFn);
    }
    
    // Look for a declaration of a function to differentiate
    // in the derivatives namespace.
    LookupResult R(m_Sema, CE->getDirectCallee()->getNameInfo(),
                   Sema::LookupOrdinaryName);
    m_Sema.LookupQualifiedName(R, m_BuiltinDerivativesNSD,
                               /*allowBuiltinCreation*/ false);    
    Expr* OverloadedFnInFile
    = findOverloadedDefinition(CE->getDirectCallee()->getNameInfo(), CallArgs);
    
    if (OverloadedFnInFile) {
      // Take the function to derive from the source.
      FunctionDecl* FD = CE->getDirectCallee();
      // Get the definition, if any.
      FunctionDecl* mostRecentFD = FD->getMostRecentDecl();
      while (mostRecentFD && !mostRecentFD->isThisDeclarationADefinition()) {
        mostRecentFD = mostRecentFD->getPreviousDecl();
      }
      if (!mostRecentFD || !mostRecentFD->isThisDeclarationADefinition()) {
        SourceLocation IdentifierLoc = FD->getLocEnd();
        m_Sema.Diag(IdentifierLoc, diag::err_differentiating_undefined_function)
          << FD->getNameAsString();
        return NodeContext(CE);
      }

      if (m_DerivedFD) { // If recursively deriving. FIXME: 
        ValueDecl* oldIndependentVar = m_IndependentVar;
        unsigned index;
        for (index = 0; index < m_DerivedFD->getNumParams(); ++index) {
          if (m_DerivedFD->getParamDecl(index) == oldIndependentVar)
            break;
        }
        assert(index <= m_DerivedFD->getNumParams() && "Not found");
        m_IndependentVar = mostRecentFD->getParamDecl(index - 1);
        FunctionDecl* derivedFD = 0;
        if (mostRecentFD->getNumParams() > 0)
          derivedFD = Derive(mostRecentFD, m_IndependentVar);
        m_IndependentVar = oldIndependentVar;
        R.clear();
        R.addDecl(derivedFD);
        //      derivedFD->dumpColor();
      }
      // Update function name in the source.
      CXXScopeSpec CSS;
      Expr* ResolvedLookup
        = m_Sema.BuildDeclarationNameExpr(CSS, R, /*ADL*/ false).get();
      CallExpr* clonedCE = m_NodeCloner->Clone(CE);
      clonedCE->setCallee(ResolvedLookup);
      return NodeContext(clonedCE);
    }
    
    // Function was not derived => issue a warning.
    SourceLocation IdentifierLoc = CE->getDirectCallee()->getLocEnd();
    m_Sema.Diag(IdentifierLoc, diag::warn_function_not_declared_in_custom_derivatives)
      << CE->getDirectCallee()->getNameAsString();
    return NodeContext(CE);
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
        
        if (ValueDecl* VD = dyn_cast<ValueDecl>(R.getFoundDecl())) {
          // FIXME: Handle the case when there are overloads found. Update
          // it with the best match.
          // 
          // FIXME: This is the right way to go in principe, however there is no
          // properly built decl context.
          // m_Sema.MarkDeclRefReferenced(clonedDRE);

          DRE->setDecl(VD);
          VD->setReferenced();
          VD->setIsUsed();
        }
        return true;
      }
    };
  } // end anon namespace
  
  Expr* DerivativeBuilder::updateReferencesOf(Expr* InSubtree) {
    Updater up(m_Sema, m_NodeCloner.get(), m_CurScope.get());
    up.TraverseStmt(InSubtree);
    return InSubtree;
  }

  NodeContext DerivativeBuilder::VisitUnaryOperator(UnaryOperator* UnOp) {
    UnaryOperator* clonedUnOp = m_NodeCloner->Clone(UnOp);
    clonedUnOp->setSubExpr(Visit(clonedUnOp->getSubExpr()->IgnoreImpCasts()).getExpr());
    return NodeContext(clonedUnOp);
  }

  NodeContext DerivativeBuilder::VisitBinaryOperator(BinaryOperator* BinOp) {
    Expr* rhs = updateReferencesOf(BinOp->getRHS());
    Expr* lhs = updateReferencesOf(BinOp->getLHS());

    Expr* lhs_derived = cast<Expr>((Visit(lhs->IgnoreImpCasts())).getStmt());
    Expr* rhs_derived = cast<Expr>((Visit(rhs->IgnoreImpCasts())).getStmt());

    BinaryOperatorKind opCode = BinOp->getOpcode();
    if (opCode == BO_Mul || opCode == BO_Div) {
      SourceLocation noLoc;

      Expr* newBO_Mul_left
        = m_Sema.BuildBinOp(/*Scope*/0, noLoc, BO_Mul, lhs_derived, rhs).get();

      Expr* newBO_Mul_Right
        = m_Sema.BuildBinOp(/*Scope*/0, noLoc, BO_Mul, lhs, rhs_derived).get();


      SourceLocation L, R;
      if (opCode == BO_Mul) {
        Expr* newBO_Add = m_Sema.BuildBinOp(/*Scope*/0, noLoc, BO_Add,
                                            newBO_Mul_left,
                                            newBO_Mul_Right).get();


        Expr* PE = m_Sema.ActOnParenExpr(L, R, newBO_Add).get();
        return NodeContext(PE);
      }
      else {
        Expr* newBO_Sub = m_Sema.BuildBinOp(/*Scope*/0, noLoc, BO_Sub,
                                            newBO_Mul_left,
                                            newBO_Mul_Right).get();

        Expr* newBO_Mul_denom = m_Sema.BuildBinOp(/*Scope*/0, noLoc, BO_Mul,
                                                  rhs, rhs).get();

        SourceLocation L, R;
        Expr* PE_lhs = m_Sema.ActOnParenExpr(L, R, newBO_Sub).get();
        Expr* PE_rhs = m_Sema.ActOnParenExpr(L, R, newBO_Mul_denom).get();

        Expr* newBO_Div = m_Sema.BuildBinOp(/*Scope*/0, noLoc, BO_Div,
                                            PE_lhs, PE_rhs).get();

        return NodeContext(newBO_Div);
      }
    }

    if (opCode == BO_Add || opCode == BO_Sub) {
      SourceLocation L, R;

      BinOp->setLHS(lhs_derived);
      // enforce precedence for substraction
      BinOp->setRHS(m_Sema.ActOnParenExpr(L, R, rhs_derived).get());

      return NodeContext(BinOp);
    }

    return NodeContext(BinOp);
  }

  NodeContext DerivativeBuilder::VisitDeclStmt(DeclStmt* DS) {
    DeclStmt* clonedDS = m_NodeCloner->Clone(DS);
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
  
  NodeContext
  DerivativeBuilder::VisitCXXOperatorCallExpr(CXXOperatorCallExpr* OpCall) {
    // This operator gets emitted when there is a binary operation containing
    // overloaded operators. Eg. x+y, where operator+ is overloaded.
    assert(0 && "We don't support overloaded operators yet!");
    return NodeContext(OpCall);
  }  
} // end namespace clad
