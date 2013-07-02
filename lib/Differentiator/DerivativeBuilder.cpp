//--------------------------------------------------------------------*- C++ -*-
// AutoDiff - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "autodiff/Differentiator/DerivativeBuilder.h"
#include "autodiff/Differentiator/StmtClone.h"

#include "clang/AST/ASTContext.h"

using namespace clang;

namespace autodiff {
  DerivativeBuilder::DerivativeBuilder() {}
  DerivativeBuilder::~DerivativeBuilder() {}

  const FunctionDecl* DerivativeBuilder::Derive(FunctionDecl* FD) {
    assert(FD && "Must not be null.");
    m_Context = &FD->getASTContext();
    if (!m_NodeCloner) {
      m_NodeCloner.reset(new utils::StmtClone(*m_Context));
    }
    Stmt* derivativeBody = Visit(FD->getBody()).getStmt();
    
    SourceLocation noLoc;
    IdentifierInfo* II 
      = &m_Context->Idents.get(FD->getNameAsString() + "_derived");
    DeclarationName name(II);
    FunctionDecl* derivedFD 
      = FunctionDecl::Create(*m_Context, FD->getDeclContext(), noLoc, noLoc,
                             name, FD->getType(), FD->getTypeSourceInfo(),
                             FD->getStorageClass(),
                             /*default*/
                             FD->isInlineSpecified(),
                             FD->hasWrittenPrototype(),
                             FD->isConstexpr()
                             );
    llvm::SmallVector<ParmVarDecl*, 4> params;
    ParmVarDecl* newPVD = 0;
    ParmVarDecl* PVD = 0;
    for(size_t i = 0, e = FD->getNumParams(); i < e; ++i) {
      PVD = FD->getParamDecl(i);
      newPVD = ParmVarDecl::Create(*m_Context, derivedFD, noLoc, noLoc, 
                                   PVD->getIdentifier(), PVD->getType(), 
                                   PVD->getTypeSourceInfo(), 
                                   PVD->getStorageClass(), 
                                   m_NodeCloner->Clone(PVD->getDefaultArg()));
      params.push_back(newPVD);
    }
    llvm::ArrayRef<ParmVarDecl*> paramsRef 
      = llvm::makeArrayRef(params.data(), params.size());
    derivedFD->setParams(paramsRef);
    derivedFD->setBody(derivativeBody);
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
    return new (*m_Context) CompoundStmt(*m_Context, stmtsRef, noLoc, noLoc);
  }

  NodeContext DerivativeBuilder::VisitReturnStmt(ReturnStmt* RS) {
    Expr* retVal = cast<Expr>(Visit(RS->getRetValue()).getStmt());
    RS->setRetValue(retVal);
    return NodeContext(RS);
  }

  NodeContext DerivativeBuilder::VisitBinaryOperator(BinaryOperator* BinOp) {
    // Visit RHS and LHS
    BinOp->setRHS(Visit(BinOp->getRHS()).getExpr());
    BinOp->setLHS(Visit(BinOp->getLHS()).getExpr());
    
    if (BinOp->isMultiplicativeOp()) {
      Expr* rhs = BinOp->getRHS();
      Expr* lhs = BinOp->getLHS();
      DeclRefExpr* rhsDRE = 0;
      DeclRefExpr* lhsDRE = 0;
      if (DeclRefExpr* DRE = dyn_cast<DeclRefExpr>(rhs->IgnoreImpCasts()))
        rhsDRE = DRE;

      if (DeclRefExpr* DRE = dyn_cast<DeclRefExpr>(lhs->IgnoreImpCasts()))
        lhsDRE = DRE;
      
      if (lhsDRE && rhsDRE && lhsDRE->getDecl() == rhsDRE->getDecl()) {
        llvm::APInt two(m_Context->getIntWidth(m_Context->IntTy), /*value*/2);
        SourceLocation noLoc;
        IntegerLiteral* constant2 = IntegerLiteral::Create(*m_Context, two,
                                                           m_Context->IntTy,
                                                           noLoc);
        BinaryOperator* newBinOp 
          = new (*m_Context) BinaryOperator(constant2, rhs, BinOp->getOpcode(),
                                            BinOp->getType(),
                                            BinOp->getValueKind(), 
                                            BinOp->getObjectKind(),
                                            noLoc,
                                            BinOp->isFPContractable());
        NodeContext nc = NodeContext(newBinOp);
        return nc;
      }
    }

    return NodeContext(BinOp);
  }

  NodeContext 
  DerivativeBuilder::VisitCXXOperatorCallExpr(CXXOperatorCallExpr* OpCall) {
    // This operator gets emitted when there is a binary operation containing
    // overloaded operators. Eg. x+y, where operator+ is overloaded.
    assert(0 && "We don't support overloaded operators yet!");
    return NodeContext(OpCall);
  }

} // end namespace autodiff
