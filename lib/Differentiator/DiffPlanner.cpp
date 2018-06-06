#include "clad/Differentiator/DiffPlanner.h"

#include "clang/AST/ASTContext.h"
#include "clang/Sema/Sema.h"

using namespace clang;

namespace clad {
  // FIXME:
  // Currently, the global variable is used to store the pointer to the last
  // call which requires update.
  // This is needed to share this information between different invocations of
  // CladPlugin::HandleTopLevelDecl in which clad::gradient and make_gradient
  // are processed (see Differentiator.h).
  clang::CallExpr* global_GradientToUpdate = nullptr;

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
    if (m_FD)
      m_FD->dump();
    else
      llvm::errs() << "<invalid> FD: " << m_FD << '\n';
    
    if (m_PVD)
      m_PVD->dump();
    else
      llvm::errs() << "<invalid> PVD: " << m_PVD << '\n';
  }

  void DiffPlan::updateCall(FunctionDecl* FD, Sema& SemaRef) {
    auto call =
      (getMode() == DiffMode::forward) ?
       m_CallToUpdate :
       global_GradientToUpdate;
    // Index of "code" parameter:
    // 2 for clad::differentiate, 1 for clad::gradient.
    auto codeArgIdx = static_cast<int>(call->getNumArgs()) - 1;
    assert(call && "Must be set");
    DeclRefExpr* DRE = 0;
    if (ImplicitCastExpr* ICE
        = dyn_cast<ImplicitCastExpr>(call->getArg(0))){
      DRE = dyn_cast<DeclRefExpr>(ICE->getSubExpr());
    }
    // Handle the case of member function.
    else if (UnaryOperator* UnOp
             = dyn_cast<UnaryOperator>(call->getArg(0))){
      DRE = dyn_cast<DeclRefExpr>(UnOp->getSubExpr());
    }
    // Set the new declaration.
    if (DRE)
      DRE->setDecl(FD);
    // Update the code parameter.
    if (CXXDefaultArgExpr* Arg
        = dyn_cast<CXXDefaultArgExpr>(call->getArg(codeArgIdx))) {
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
      // Get an array type for the string, according to C99 6.4.5. This includes
      // the nul terminator character as well as the string length for pascal
      // strings.
      QualType StrTy =
        C.getConstantArrayType(
         CharTyConst,
         llvm::APInt(32, Out.str().size() + 1),
         ArrayType::Normal,
         /*IndexTypeQuals*/0);

      StringLiteral* SL =
        StringLiteral::Create(
         C,
         Out.str(),
         StringLiteral::Ascii,
         /*Pascal*/false,
         StrTy,
         SourceLocation());
      Expr* newArg =
        SemaRef.ImpCastExprToType(
          SL,
          Arg->getType(),
          CK_ArrayToPointerDecay).get();
      call->setArg(codeArgIdx, newArg);
    }
  }

  LLVM_DUMP_METHOD void DiffPlan::dump() {
    for (const_iterator I = begin(), E = end(); I != E; ++I) {
      I->dump();
      llvm::errs() << "\n";
    }
  }

  ParmVarDecl* DiffCollector::getIndependentArg(
    Expr* argExpr, FunctionDecl* FD) {
    assert(!m_TopMostFDI && "Must be not in implicit fn collecting mode.");
    bool isIndexOutOfRange = false;
    llvm::APSInt result;
    ASTContext& C = m_Sema.getASTContext();
    DiagnosticsEngine& Diags = m_Sema.Diags;
    if (argExpr->EvaluateAsInt(result, C)) {
      const int64_t argIndex = result.getSExtValue();
      getCurrentPlan().setArgIndex(argIndex);
      const int64_t argNum = FD->getNumParams();
      //TODO: Implement the argument checks in the DerivativeBuilder
      if (argNum == 0) {
        unsigned DiagID
          = Diags.getCustomDiagID(
              DiagnosticsEngine::Error,
              "Trying to differentiate function '%0' taking no arguments");
        Diags.Report(argExpr->getLocStart(), DiagID)
          << FD->getNameAsString();
        return 0;
      }
      //if arg is int but do not exists print error
      else if (argIndex >= argNum || argIndex < 0) {
        isIndexOutOfRange = true;
        unsigned DiagID
          = Diags.getCustomDiagID(
            DiagnosticsEngine::Error,
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
    if (plan->getCurrentDerivativeOrder() == 
        plan->getRequestedDerivativeOrder())
      return;
    assert(plan->getRequestedDerivativeOrder() > 1
           && "Must be called on high order derivatives");
    plan->setCurrentDerivativeOrder(plan->getCurrentDerivativeOrder() + 1);
    plan->push_back(
      FunctionDeclInfo(FD, FD->getParamDecl(plan->getArgIndex())));
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
      else if (const AnnotateAttr* A = FD->getAttr<AnnotateAttr>()) {
        DeclRefExpr* DRE = nullptr;

        // Handle the case of function.
        if (ImplicitCastExpr* ICE = dyn_cast<ImplicitCastExpr>(E->getArg(0))){
           DRE = dyn_cast<DeclRefExpr>(ICE->getSubExpr());
        }
        // Handle the case of member function.
        else if (UnaryOperator* UnOp = dyn_cast<UnaryOperator>(E->getArg(0))){
          DRE = dyn_cast<DeclRefExpr>(UnOp->getSubExpr());
        }
        if (DRE) {          
          auto && label = A->getAnnotation();
          if (label.equals("D")) {
            // A call to clad::differentiate was found.

            m_DiffPlans.push_back(DiffPlan());
            getCurrentPlan().setMode(DiffMode::forward);

            llvm::APSInt derivativeOrderAPSInt
              = FD->getTemplateSpecializationArgs()->get(0).getAsIntegral();
            // We know the first template spec argument is of unsigned type
            assert(derivativeOrderAPSInt.isUnsigned() && "Must be unsigned");
            unsigned derivativeOrder = derivativeOrderAPSInt.getZExtValue();
            getCurrentPlan().m_RequestedDerivativeOrder = derivativeOrder;

            getCurrentPlan().setCallToUpdate(E);
            FunctionDecl* cand = cast<FunctionDecl>(DRE->getDecl());
            ParmVarDecl* candPVD = getIndependentArg(E->getArg(1), cand);
            FunctionDeclInfo FDI(cand, candPVD);
            m_TopMostFDI = &FDI;
            TraverseDecl(cand);
            m_TopMostFDI = 0;
            getCurrentPlan().push_back(FDI);
          }
          else if (label.equals("G")) {
            // A call to clad::gradient was found.

            m_DiffPlans.push_back(DiffPlan());
            getCurrentPlan().setMode(DiffMode::reverse);

            FunctionDeclInfo FDI(cast<FunctionDecl>(DRE->getDecl()), nullptr);
            getCurrentPlan().push_back(FDI);
          }
          else if (label.equals("GR")) {
            // A call to make_gradient was found.

            global_GradientToUpdate = E;
          }
        }
      }  
    }
    return true;     // return false to abort visiting.
  }
} // end namespace
