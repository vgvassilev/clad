#include "clad/Differentiator/DiffPlanner.h"

#include "clang/AST/ASTContext.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/TemplateDeduction.h"

using namespace clang;

namespace clad {
  static SourceLocation noLoc;

  void DiffPlan::updateCall(FunctionDecl* FD, Sema& SemaRef) {
    auto call = m_CallToUpdate;
    // Index of "code" parameter:
    auto codeArgIdx = static_cast<int>(call->getNumArgs()) - 1;
    assert(call && "Must be set");
    assert(FD && "Trying to update with null FunctionDecl");

    DeclRefExpr* oldDRE = nullptr;
    // Handle the case of function pointer.
    if (auto ICE = dyn_cast<ImplicitCastExpr>(call->getArg(0))){
      oldDRE = dyn_cast<DeclRefExpr>(ICE->getSubExpr());
    }
    // Handle the case of member function.
    else if (auto UnOp = dyn_cast<UnaryOperator>(call->getArg(0))){
      oldDRE = dyn_cast<DeclRefExpr>(UnOp->getSubExpr());
    }
    else
      llvm_unreachable("Trying to differentiate something unsupported");

    ASTContext& C = SemaRef.getASTContext();
    // Create ref to generated FD.
    Expr* DRE = DeclRefExpr::Create(C, oldDRE->getQualifierLoc(), noLoc,
                                    FD, false, FD->getNameInfo(), FD->getType(),
                                    oldDRE->getValueKind());
    // FIXME: I am not sure if the following part is necessary:
    // using call->setArg(0, DRE) seems to be sufficient,
    // though the real AST allways contains the ImplicitCastExpr (function ->
    // function ptr cast) or UnaryOp (method ptr call).
    auto oldArg = call->getArg(0);
    if (auto oldCast = dyn_cast<ImplicitCastExpr>(oldArg)) {
      // Cast function to function pointer.
      auto newCast = ImplicitCastExpr::Create(C,
                                              C.getPointerType(FD->getType()),
                                              oldCast->getCastKind(),
                                              DRE,
                                              nullptr,
                                              oldCast->getValueKind());
      call->setArg(0, newCast);
    }
    else if (auto oldUnOp = dyn_cast<UnaryOperator>(oldArg)) {
      // Add the "&" operator
      auto newUnOp = SemaRef.BuildUnaryOp(nullptr,
                                          noLoc,
                                          oldUnOp->getOpcode(),
                                          DRE).get();
      call->setArg(0, newUnOp);
    }
    else
      llvm_unreachable("Trying to differentiate something unsupported");

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
      QualType CharTyConst = C.CharTy;
      CharTyConst.addConst();
      // Get an array type for the string, according to C99 6.4.5. This includes
      // the nul terminator character as well as the string length for pascal
      // strings.
      QualType StrTy =
        C.getConstantArrayType(CharTyConst,
                               llvm::APInt(32, Out.str().size() + 1),
                               ArrayType::Normal,
                               /*IndexTypeQuals*/0);

      StringLiteral* SL =
        StringLiteral::Create(C,
                              Out.str(),
                              StringLiteral::Ascii,
                              /*Pascal*/false,
                              StrTy,
                              noLoc);
      Expr* newArg =
        SemaRef.ImpCastExprToType(SL,
                                  Arg->getType(),
                                  CK_ArrayToPointerDecay).get();
      call->setArg(codeArgIdx, newArg);
    }

    // Replace old specialization of clad::gradient with a new one that matches
    // the type of new argument.

    auto CladGradientFDeclOld = call->getDirectCallee();
    auto CladGradientExprOld = call->getCallee();
    auto CladGradientFTemplate = CladGradientFDeclOld->getPrimaryTemplate();

    FunctionDecl* CladGradientFDeclNew = nullptr;
    sema::TemplateDeductionInfo Info(noLoc);
    // Create/get template specialization of clad::gradient that matches
    // argument types. Result is stored to CladGradientFDeclNew.
    SemaRef.DeduceTemplateArguments(CladGradientFTemplate,
                                    /* ExplicitTemplateArgs */ nullptr,
                                    /* Args */
                                    llvm::ArrayRef<Expr*>(call->getArgs(),
                                                          call->getNumArgs()),
                                    /* Specialization */ CladGradientFDeclNew,
                                    Info,
                                    /* PartialOverloading */ false,
                                    /* CheckNonDependent */
                                    [] (llvm::ArrayRef<QualType>) {
                                      return false;
                                    });
    // DeclRefExpr for new specialization.
    auto CladGradientExprNew =
      SemaRef.BuildDeclRefExpr(CladGradientFDeclNew,
                               CladGradientFDeclNew->getType(),
                               CladGradientExprOld->getValueKind(),
                               CladGradientExprOld->getLocEnd()).get();
    // Add function to pointer cast.
    CladGradientExprNew =
      SemaRef.CallExprUnaryConversions(CladGradientExprNew).get();
    // Replace the old clad::gradient by the new one.
    call->setCallee(CladGradientExprNew);
  }

  LLVM_DUMP_METHOD void DiffPlan::dump() {
    for (const_iterator I = begin(), E = end(); I != E; ++I) {
      (*I)->dump();
      llvm::errs() << "\n";
    }
  }

  DiffCollector::DiffCollector(DeclGroupRef DGR, DiffPlans& plans, Sema& S)
     : m_DiffPlans(plans), m_TopMostFD(nullptr), m_Sema(S) {
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
    plan->push_back(FD);
    m_DiffPlans.push_back(*plan);
    TraverseDecl(FD);
    m_DiffPlans.pop_back();
  }

  bool DiffCollector::VisitCallExpr(CallExpr* E) {
    if (FunctionDecl *FD = E->getDirectCallee()) {
      // We need to find our 'special' diff annotated such:
      // clad::differentiate(...) __attribute__((annotate("D")))
      if (const AnnotateAttr* A = FD->getAttr<AnnotateAttr>()) {
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
            auto FD = cast<FunctionDecl>(DRE->getDecl());
            m_TopMostFD = FD;
            TraverseDecl(FD);
            m_TopMostFD = nullptr;
            getCurrentPlan().push_back(FD);
            getCurrentPlan().m_DiffArgs = E->getArg(1);
          }
          else if (label.equals("G")) {
            // A call to clad::gradient was found.

            m_DiffPlans.push_back(DiffPlan());
            getCurrentPlan().setMode(DiffMode::reverse);
            getCurrentPlan().setCallToUpdate(E);
            auto FD = cast<FunctionDecl>(DRE->getDecl());
            getCurrentPlan().push_back(FD);
            getCurrentPlan().m_DiffArgs = E->getArg(1);
          }
        }
      }
    }
    return true;     // return false to abort visiting.
  }
} // end namespace
