//--------------------------------------------------------------------*- C++ -*-
// AutoDiff - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "autodiff/Differentiator/DerivativeBuilder.h"
#include "autodiff/Differentiator/StmtClone.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/SemaInternal.h"

using namespace clang;

namespace autodiff {
  CompoundStmt* NodeContext::wrapInCompoundStmt(clang::ASTContext& C) const {
    assert(!isSingleStmt() && "Must be more than 1");
    llvm::ArrayRef<Stmt*> stmts
    = llvm::makeArrayRef(m_Stmts.data(), m_Stmts.size());
    clang::SourceLocation noLoc;
    return new (C) clang::CompoundStmt(C, stmts, noLoc, noLoc);
  }
  
  DerivativeBuilder::DerivativeBuilder(clang::Sema& S)
    : m_Sema(S), m_Context(S.getASTContext()) {
    m_NodeCloner.reset(new utils::StmtClone(m_Context));
    // Find the builtin derivatives namespace
    DeclarationName Name = &m_Context.Idents.get("custom_derivatives");
    LookupResult R(m_Sema, Name, SourceLocation(), Sema::LookupNamespaceName,
                   Sema::ForRedeclaration);
    m_Sema.LookupQualifiedName(R, m_Context.getTranslationUnitDecl(),
                               /*allowBuiltinCreation*/ false);
    assert(!R.empty() && "Cannot find builtin derivatives!");
    m_BuiltinDerivativesNSD = cast<NamespaceDecl>(R.getFoundDecl());
    // Initialize pointers.
    derivedFD = 0;
    independentVar = 0;
  }

  DerivativeBuilder::~DerivativeBuilder() {}
  
  FunctionDecl* DerivativeBuilder::Derive(FunctionDecl* FD, ValueDecl* argVar) {
    assert(FD && "Must not be null.");
    independentVar = argVar;
    
    if (!m_NodeCloner) {
      m_NodeCloner.reset(new utils::StmtClone(m_Context));
    }
    
    SourceLocation noLoc;
    IdentifierInfo* II
      = &m_Context.Idents.get(FD->getNameAsString() + "_derived_" +
                             independentVar->getNameAsString());
    DeclarationName name(II);
    derivedFD = FunctionDecl::Create(m_Context, FD->getDeclContext(), noLoc, 
                                     noLoc, name, FD->getType(), 
                                     FD->getTypeSourceInfo(),
                                     FD->getStorageClass(),
                                     /*default*/
                                     FD->isInlineSpecified(),
                                     FD->hasWrittenPrototype(),
                                     FD->isConstexpr()
                                     );
    // We will use the m_CurScope to do the needed lookups.
    m_CurScope.reset(new Scope(m_Sema.TUScope, Scope::FnScope,
                               m_Sema.getDiagnostics()));
    m_CurScope->setEntity(derivedFD);
    
    llvm::SmallVector<ParmVarDecl*, 4> params;
    ParmVarDecl* newPVD = 0;
    ParmVarDecl* PVD = 0;
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
      m_CurScope->AddDecl(newPVD);
      if (newPVD->getIdentifier())
        m_Sema.IdResolver.AddDecl(newPVD);
    }
    llvm::ArrayRef<ParmVarDecl*> paramsRef
      = llvm::makeArrayRef(params.data(), params.size());
    derivedFD->setParams(paramsRef);
    Stmt* derivativeBody = Visit(FD->getBody()).getStmt();
    derivedFD->setBody(derivativeBody);
    
    // Cleanup the IdResolver chain.
    for(FunctionDecl::param_iterator I = derivedFD->param_begin(),
        E = derivedFD->param_end(); I != E; ++I) {
      if ((*I)->getIdentifier())
        m_Sema.IdResolver.RemoveDecl(*I);
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
  
  NodeContext DerivativeBuilder::VisitReturnStmt(ReturnStmt* RS) {
    ReturnStmt* clonedStmt = m_NodeCloner->Clone(RS);
    Expr* retVal
      = cast<Expr>(Visit(clonedStmt->getRetValue()->IgnoreImpCasts()).getStmt());
    clonedStmt->setRetValue(retVal);
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
        independentVar->getNameAsString()) {
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
          OverloadCandidateSet CandidateSet(Loc);
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
      = m_Sema.BuildDeclarationNameExpr(CSS, R, /*ADL*/ false).take();

      llvm::MutableArrayRef<Expr*> MARargs
      = llvm::MutableArrayRef<Expr*>(CallArgs);
            
      SourceLocation Loc;
      Scope* S = m_Sema.getScopeForContext(m_Sema.CurContext);
      
      if (overloadExists(UnresolvedLookup, MARargs)) {
        return 0;
      }
      
      OverloadedFn = m_Sema.ActOnCallExpr(S, UnresolvedLookup, Loc,
                                          MARargs, Loc).take();
    }
    return OverloadedFn;
  }
  
  NodeContext DerivativeBuilder::VisitCallExpr(CallExpr* CE) {
    // Find the built-in derivatives namespace.
    IdentifierInfo* II
    = &m_Context.Idents.get(CE->getDirectCallee()->getNameAsString() +
                            "_derived_" + independentVar->getNameAsString());
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
      
      FunctionDecl* derivedFD = Derive(mostRecentFD, independentVar);
//      derivedFD->dumpColor();
      // Update function name in the source.
      R.clear();
      R.addDecl(derivedFD);
      CXXScopeSpec CSS;
      Expr* ResolvedLookup
      = m_Sema.BuildDeclarationNameExpr(CSS, R, /*ADL*/ false).take();
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
  
  Expr* DerivativeBuilder::updateReferencesOf(Expr* InSubtree) {
    class Updater : public RecursiveASTVisitor<Updater> {
    private:
      Sema& m_Sema; // We don't own.
      utils::StmtClone* m_NodeCloner; // We don't own.
      Scope* m_CurScope; // We don't own.
    public:
      Updater(Sema& SemaRef, utils::StmtClone* C, Scope* S) 
        : m_Sema(SemaRef), m_NodeCloner(C), m_CurScope(S) {}
      bool VisitDeclRefExpr(DeclRefExpr* DRE) {
        DeclarationNameInfo DNI = DRE->getNameInfo();
        
        LookupResult R(m_Sema, DNI, Sema::LookupOrdinaryName);
        m_Sema.LookupName(R, m_CurScope, /*allowBuiltinCreation*/ false);
        
        if (ValueDecl* VD = dyn_cast<ValueDecl>(R.getFoundDecl())) {
          // FIXME: This is the right way to go in principe, however there is no
          // properly built decl context.
          //DeclContext* DC = static_cast<DeclContext*>(m_CurScope->getEntity())
          //Sema::ContextRAII PushCtxRAII(m_Sema, DC);
          // m_Sema.MarkDeclRefReferenced(clonedDRE);

          DRE->setDecl(VD);
          VD->setReferenced();
          VD->setUsed();
        }
        return true;
      }
    };
    Updater up(m_Sema, m_NodeCloner.get(), m_CurScope.get());
    up.TraverseStmt(InSubtree);
    return InSubtree;
  }
  
  NodeContext DerivativeBuilder::VisitBinaryOperator(BinaryOperator* BinOp) {
    Expr* rhs = updateReferencesOf(BinOp->getRHS());
    Expr* lhs = updateReferencesOf(BinOp->getLHS());
    
    Expr* lhs_derived = cast<Expr>((Visit(lhs->IgnoreImpCasts())).getStmt());
    Expr* rhs_derived = cast<Expr>((Visit(rhs->IgnoreImpCasts())).getStmt());
    
    BinaryOperatorKind opCode = BinOp->getOpcode();
    if (opCode == BO_Mul || opCode == BO_Div) {
      SourceLocation noLoc;
      QualType qType = BinOp->getType();
      ExprValueKind VK = BinOp->getValueKind();
      ExprObjectKind OK = BinOp->getObjectKind();
      bool fpContractable = BinOp->isFPContractable();
      
      BinaryOperator* newBO_Mul_left
      = new (m_Context) BinaryOperator(lhs_derived, rhs, BO_Mul,
                                       qType, VK, OK, noLoc, fpContractable);
      BinaryOperator* newBO_Mul_Right
      = new (m_Context) BinaryOperator(lhs, rhs_derived, BO_Mul,
                                       qType, VK, OK, noLoc, fpContractable);
      
      SourceLocation L, R;
      if (opCode == BO_Mul) {
        BinaryOperator* newBO_Add
          = new (m_Context) BinaryOperator(newBO_Mul_left, newBO_Mul_Right,
                                           BO_Add, qType, VK, OK, noLoc, 
                                           fpContractable);
        // enforce precedence for addition
        ParenExpr* PE = new (m_Context) ParenExpr(L, R, newBO_Add);
        return NodeContext(PE);
      }
      else {
        BinaryOperator* newBO_Sub
          = new (m_Context) BinaryOperator(newBO_Mul_left,newBO_Mul_Right,BO_Sub,
                                         qType, VK, OK, noLoc, fpContractable);
        BinaryOperator* newBO_Mul_denom
          = new (m_Context) BinaryOperator(rhs, rhs, BO_Mul,
                                           qType, VK, OK, noLoc, fpContractable);
        SourceLocation L, R;
        ParenExpr* PE_lhs = new (m_Context) ParenExpr(L, R, newBO_Sub);
        ParenExpr* PE_rhs = new (m_Context) ParenExpr(L, R, newBO_Mul_denom);
        
        BinaryOperator* newBO_Div
        = new (m_Context) BinaryOperator(PE_lhs, PE_rhs, BO_Div,
                                         qType, VK, OK, noLoc, fpContractable);
        return NodeContext(newBO_Div);
      }
    }
    
    if (opCode == BO_Add || opCode == BO_Sub) {
      SourceLocation L, R;
      
      BinOp->setLHS(lhs_derived);
      // enforce precedence for substraction
      BinOp->setRHS(new (m_Context) ParenExpr(L, R, rhs_derived));
      
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
} // end namespace autodiff
