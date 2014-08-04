#include "clad/Differentiator/DiffPlanner.h"

#include "clang/AST/ASTContext.h"
#include "clang/Sema/Sema.h"

using namespace clang;

namespace clad {
  FunctionDeclInfo::FunctionDeclInfo(FunctionDecl* FD, ParmVarDecl* PVD)
    : m_FD(FD), m_PVD(PVD) {
#ifndef NDEBUG
    if (!PVD) // Can happen if there was error deducing the argument
      return;
    bool inArgs = false;
    for (unsigned i = 0; i < FD->getNumParams(); ++i)
      if (PVD == FD->getParamDecl(i)) {
        inArgs = true;
        break;
      }
    assert(inArgs && "Must pass in a param of the FD.");
#endif
  }

  LLVM_DUMP_METHOD void FunctionDeclInfo::dump() const {
    if (!isValid())
      llvm::errs() << "<invalid> FD :"<< m_FD << " , PVD:" << m_PVD << "\n";
    else {
      m_FD->dump();
      m_PVD->dump();
    }
  }

  void DiffPlan::updateCall(FunctionDecl* FD, Sema& SemaRef) {
    assert(m_CallToUpdate && "Must be set");
    DeclRefExpr* DRE = 0;
    if (ImplicitCastExpr* ICE
        = dyn_cast<ImplicitCastExpr>(m_CallToUpdate->getArg(0))){
      DRE = dyn_cast<DeclRefExpr>(ICE->getSubExpr());
    }
    // Handle the case of member function.
    else if (UnaryOperator* UnOp
             = dyn_cast<UnaryOperator>(m_CallToUpdate->getArg(0))){
      DRE = dyn_cast<DeclRefExpr>(UnOp->getSubExpr());
    }
    if (DRE)
      DRE->setDecl(FD);
    if (CXXDefaultArgExpr* Arg
        = dyn_cast<CXXDefaultArgExpr>(m_CallToUpdate->getArg(2))) {
      clang::LangOptions LangOpts;
      LangOpts.CPlusPlus = true;
      clang::PrintingPolicy Policy(LangOpts);

      std::string s;
      llvm::raw_string_ostream Out(s);
      FD->print(Out, Policy);
      Out.flush();

      // Copied and adapted from clang::Sema::ActOnStringLiteral.
      ASTContext& C = SemaRef.getASTContext();
      QualType CharTyConst = C.CharTy;
      CharTyConst.addConst();
      // Get an array type for the string, according to C99 6.4.5.  This includes
      // the nul terminator character as well as the string length for pascal
      // strings.
      QualType StrTy = C.getConstantArrayType(CharTyConst,
                                              llvm::APInt(32, Out.str().size() + 1),
                                              ArrayType::Normal,
                                              /*IndexTypeQuals*/0);


      StringLiteral* SL = StringLiteral::Create(C, Out.str(),
                                                StringLiteral::Ascii,
                                                /*Pascal*/false, StrTy,
                                                SourceLocation());
      Expr* newArg = SemaRef.ImpCastExprToType(SL, Arg->getType(),
                                               CK_ArrayToPointerDecay).get();
      m_CallToUpdate->setArg(2, newArg);
    }
  }

  LLVM_DUMP_METHOD void DiffPlan::dump() {
    for (const_iterator I = begin(), E = end(); I != E; ++I) {
      I->dump();
      llvm::errs() << "\n";
    }
  }

  ParmVarDecl* DiffCollector::getIndependentArg(Expr* argExpr, FunctionDecl* FD) {
    assert(!m_TopMostFDI && "Must be not in implicit fn collecting mode.");
    bool isIndexOutOfRange = false;
    llvm::APSInt result;
    ASTContext& C = m_Sema.getASTContext();
    DiagnosticsEngine& Diags = m_Sema.Diags;
    if (argExpr->EvaluateAsInt(result, C)) {
      const int64_t argIndex = result.getSExtValue();
      const int64_t argNum = FD->getNumParams();
      //TODO: Implement the argument checks in the DerivativeBuilder
      if (argNum == 0) {
        unsigned DiagID
          = Diags.getCustomDiagID(DiagnosticsEngine::Error,
                                  "Trying to differentiate function '%0' taking no arguments");
        Diags.Report(argExpr->getLocStart(), DiagID)
          << FD->getNameAsString();
        return 0;
      }
      //if arg is int but do not exists print error
      else if (argIndex >= argNum || argIndex < 0) {
        isIndexOutOfRange = true;
        unsigned DiagID
          = Diags.getCustomDiagID(DiagnosticsEngine::Error,
                                  "Invalid argument index %0 among %1 argument(s)");
        Diags.Report(argExpr->getLocStart(), DiagID)
          << (int)argIndex
          << (int)argNum;
        return 0;
      }
      else
        return FD->getParamDecl(argIndex);
    }
    else if (!isIndexOutOfRange){
      unsigned DiagID
        = Diags.getCustomDiagID(DiagnosticsEngine::Error,
                                "Must be an integral value");
      Diags.Report(argExpr->getLocStart(), DiagID);
      return 0;
    }
    return 0;
  }
  DiffCollector::DiffCollector(DeclGroupRef DGR, DiffPlans& plans, Sema& S)
     : m_DiffPlans(plans), m_TopMostFDI(0), m_Sema(S) {
    if (DGR.isSingleDecl())
      TraverseDecl(DGR.getSingleDecl());
  }

  void DiffCollector::UpdatePlan(clang::FunctionDecl* FD, DiffPlan* plan) {
    if (plan->getDerivativeOrder() -1 == 0)
      return;
    assert(plan->getDerivativeOrder() > 0
           && "Must be called on high order derivatives");
    plan->setDerivativeOrder(plan->getDerivativeOrder() -1);
    plan->push_back(FunctionDeclInfo(FD, FD->getParamDecl(0)));
    m_DiffPlans.push_back(*plan);
    TraverseDecl(FD);
    m_DiffPlans.pop_back();
  }

  bool DiffCollector::VisitCallExpr(CallExpr* E) {
    if (FunctionDecl *FD = E->getDirectCallee()) {
      if (m_TopMostFDI) {
        int index = -1;
        for (unsigned i = 0; i < m_TopMostFDI->getFD()->getNumParams(); ++i)
          if (index != -1)
            if (FD->getParamDecl(index) == m_TopMostFDI->getPVD()) {
              index = i - 1; // Decrement by 1, adapting to FD's 0 param list.
              break;
            }
        if (index > -1) {
          FunctionDeclInfo FDI(FD, FD->getParamDecl(index));
          getCurrentPlan().push_back(FDI);
        }
      }
      // We need to find our 'special' diff annotated such:
      // clad::differentiate(...) __attribute__((annotate("D")))
      else if (const AnnotateAttr* A = FD->getAttr<AnnotateAttr>())
        if (A->getAnnotation().equals("D")) {
          DeclRefExpr* DRE = 0;

          // Handle the case of function.
          if (ImplicitCastExpr* ICE = dyn_cast<ImplicitCastExpr>(E->getArg(0))){
              DRE = dyn_cast<DeclRefExpr>(ICE->getSubExpr());
          }
          // Handle the case of member function.
          else if (UnaryOperator* UnOp = dyn_cast<UnaryOperator>(E->getArg(0))){
            DRE = dyn_cast<DeclRefExpr>(UnOp->getSubExpr());
          }
          if (DRE) {
            assert(isa<FunctionDecl>(DRE->getDecl()) && "Must not happen.");
            // We know that is *our* diff function.

            // Add new plan for describing the differentiation for the function.
            m_DiffPlans.push_back(DiffPlan());

            llvm::APSInt derivativeOrderAPSInt
              = FD->getTemplateSpecializationArgs()->get(0).getAsIntegral();
            // We know the first template spec argument is of unsigned type
            assert(derivativeOrderAPSInt.isUnsigned() && "Must be unsigned");
            unsigned derivativeOrder = derivativeOrderAPSInt.getZExtValue();
            getCurrentPlan().setDerivativeOrder(derivativeOrder);

            getCurrentPlan().setCallToUpdate(E);
            FunctionDecl* cand = cast<FunctionDecl>(DRE->getDecl());
            ParmVarDecl* candPVD = getIndependentArg(E->getArg(1), cand);
            FunctionDeclInfo FDI(cand, candPVD);
            m_TopMostFDI = &FDI;
            TraverseDecl(cand);
            m_TopMostFDI = 0;
            getCurrentPlan().push_back(FDI);
            // while (--derivativeOrder) {
            //   ParmVarDecl* candPVD = getIndependentArg(E->getArg(1), FDI.getFD());
            //   FunctionDeclInfo FDI1(FDI.getFD(), candPVD);
            //   m_TopMostFDI = &FDI1;
            //   TraverseDecl(FDI1.getFD());
            //   m_TopMostFDI = 0;
            //   getCurrentPlan().push_back(FDI1);
            // }
          }
        }
      }
    return true;     // return false to abort visiting.
  }
} // end namespace
