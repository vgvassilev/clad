#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/CladUtils.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Sema/TemplateDeduction.h"

#include "llvm/Support/SaveAndRestore.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"

using namespace clang;

namespace clad {
  static SourceLocation noLoc;

  /// Returns `DeclRefExpr` node corresponding to the function, method or
  /// functor argument which is to be differentiated.
  ///
  /// \param[in] call A clad differentiation function call expression
  /// \param SemaRef Reference to Sema
  DeclRefExpr* getArgFunction(CallExpr* call, Sema& SemaRef) {
    struct Finder :
      RecursiveASTVisitor<Finder> {
        Sema& m_SemaRef;
        SourceLocation m_BeginLoc;
        DeclRefExpr* m_FnDRE = nullptr;
        Finder(Sema& SemaRef, SourceLocation beginLoc)
            : m_SemaRef(SemaRef), m_BeginLoc(beginLoc) {}

        // Required for visiting lambda declarations.
        bool shouldVisitImplicitCode() const { return true; }

        bool VisitDeclRefExpr(DeclRefExpr* DRE) {
          if (auto VD = dyn_cast<VarDecl>(DRE->getDecl())) {
            auto varType = VD->getType().getTypePtr();
            // If variable is of class type, set `m_FnDRE` to
            // `DeclRefExpr` of overloaded call operator method of
            // the class type.
            if (varType->isStructureOrClassType()) {
              auto RD = varType->getAsCXXRecordDecl();
              TraverseDecl(RD);
            } else {
              TraverseStmt(VD->getInit());
            }
          } else if (isa<FunctionDecl>(DRE->getDecl()))
            m_FnDRE = DRE;
          return false;
        }

        bool VisitCXXRecordDecl(CXXRecordDecl* RD) {
          auto callOperatorDeclName =
              m_SemaRef.getASTContext().DeclarationNames.getCXXOperatorName(
                  OverloadedOperatorKind::OO_Call);
          LookupResult R(m_SemaRef,
                         callOperatorDeclName,
                         noLoc,
                         Sema::LookupNameKind::LookupMemberName);
          // We do not want diagnostics that would fire because of this lookup.
          R.suppressDiagnostics();
          m_SemaRef.LookupQualifiedName(R, RD);

          // Emit error diagnostics
          if (R.empty()) {
            const char diagFmt[] = "'%0' has no defined operator()";
            auto diagId =
                m_SemaRef.Diags.getCustomDiagID(DiagnosticsEngine::Level::Error,
                                                diagFmt);
            m_SemaRef.Diag(m_BeginLoc, diagId) << RD->getName();
            return false;
          } else if (!R.isSingleResult()) {
            const char diagFmt[] =
                "'%0' has multiple definitions of operator(). "
                "Multiple definitions of call operators are not supported.";
            auto diagId =
                m_SemaRef.Diags.getCustomDiagID(DiagnosticsEngine::Level::Error,
                                                diagFmt);
            m_SemaRef.Diag(m_BeginLoc, diagId) << RD->getName();

            // Emit diagnostics for candidate functions
            for (auto oper = R.begin(), operEnd = R.end(); oper != operEnd;
                 ++oper) {
              auto candidateFn = cast<CXXMethodDecl>(oper.getDecl());
              m_SemaRef.NoteOverloadCandidate(candidateFn,
                                              cast<FunctionDecl>(candidateFn));
            }
            return false;
          } else if (R.isSingleResult() == 1 &&
                     cast<CXXMethodDecl>(R.getFoundDecl())->getAccess() !=
                         AccessSpecifier::AS_public) {
            const char diagFmt[] =
                "'%0' contains %1 call operator. Differentiation of "
                "private/protected call operator is not supported.";

            auto diagId =
                m_SemaRef.Diags.getCustomDiagID(DiagnosticsEngine::Level::Error,
                                                diagFmt);
            // Compute access specifier name so that it can be used in
            // diagnostic message.
            const char* callOperatorAS =
                (cast<CXXMethodDecl>(R.getFoundDecl())->getAccess() ==
                         AccessSpecifier::AS_private
                     ? "private"
                     : "protected");
            m_SemaRef.Diag(m_BeginLoc, diagId)
                << RD->getName() << callOperatorAS;
            auto callOperator = cast<CXXMethodDecl>(R.getFoundDecl());

            bool isImplicit = true;

            // compute if the corresponding access specifier of the found
            // call operator is implicit or explicit.
            for (auto decl : RD->decls()) {
              if (decl == callOperator)
                break;
              if (isa<AccessSpecDecl>(decl)) {
                isImplicit = false;
                break;
              }
            }

            // Emit diagnostics for the found call operator
            m_SemaRef.Diag(callOperator->getBeginLoc(),
                           diag::note_access_natural)
                << (unsigned)(callOperator->getAccess() ==
                              AccessSpecifier::AS_protected)
                << isImplicit;

            return false;
          }

          assert(R.isSingleResult() &&
                 "Multiple definitions of call operators are not supported");
          assert(R.isSingleResult() == 1 &&
                 cast<CXXMethodDecl>(R.getFoundDecl())->getAccess() ==
                     AccessSpecifier::AS_public &&
                 "Differentiation of private/protected call operators are "
                 "not supported");
          auto callOperator = cast<CXXMethodDecl>(R.getFoundDecl());
          // Creating `DeclRefExpr` of the found overloaded call operator
          // method, to maintain consistency with member function
          // differentiation.
          CXXScopeSpec CSS;
          BuildNNS(callOperator->getDeclContext(), CSS);

          // `ExprValueKind::VK_RValue` is used because functions are
          // decomposed to function pointers and thus a temporary is
          // created for the function pointer.
          auto newFnDRE = clad_compat::GetResult<Expr*>(
              m_SemaRef.BuildDeclRefExpr(callOperator,
                                         callOperator->getType(),
                                         CLAD_COMPAT_ExprValueKind_R_or_PR_Value,
                                         noLoc,
                                         &CSS));
          m_FnDRE = cast<DeclRefExpr>(newFnDRE);          
          return false;
        }
      private:
        /// Creates nested name specifier associated with declaration context
        /// argument `DC`. 
        ///
        /// For example, given a structure defined as,
        /// namespace A {
        /// namespace B {
        ///   struct SomeStruct {};
        /// }
        /// }
        ///
        /// Passing `SomeStruct` as declaration context will create
        /// nested name specifier of the form `::A::B::struct SomeClass::`
        /// in `CXXScopeSpec` argument `CSS`.
        /// \note Currently only namespace and class/struct nested name specifiers
        /// are supported.
        ///
        /// \param[in] DC
        /// \param[out] CSS
        void BuildNNS(DeclContext* DC, CXXScopeSpec& CSS) {
          assert(DC && "Must provide a non null DeclContext");

          // parent name specifier should be added first
          if (DC->getParent())
            BuildNNS(DC->getParent(), CSS);

          ASTContext& Context = m_SemaRef.getASTContext();

          if (auto ND = dyn_cast<NamespaceDecl>(DC)) {
            CSS.Extend(Context, ND,
                       /*NamespaceLoc=*/noLoc,
                       /*ColonColonLoc=*/noLoc);
          } else if (auto RD = dyn_cast<CXXRecordDecl>(DC)) {
            auto RDQType = RD->getTypeForDecl()->getCanonicalTypeInternal();
            auto RDTypeSourceInfo = Context.getTrivialTypeSourceInfo(RDQType);
            CSS.Extend(Context,
                       /*TemplateKWLoc=*/noLoc, RDTypeSourceInfo->getTypeLoc(),
                       /*ColonColonLoc=*/noLoc);
          } else if (isa<TranslationUnitDecl>(DC)) {
            CSS.MakeGlobal(Context, /*ColonColonLoc=*/noLoc);
          }
        }
    } finder(SemaRef, call->getArg(0)->getBeginLoc());
    finder.TraverseStmt(call->getArg(0));

    assert(cast<NamespaceDecl>(call->getDirectCallee()->getDeclContext())
                   ->getName() == "clad" &&
           "Should be called for clad:: special functions!");
    return finder.m_FnDRE;
  }

  void DiffRequest::updateCall(FunctionDecl* FD, FunctionDecl* OverloadedFD,
                               Sema& SemaRef) {
    CallExpr* call = this->CallContext;
    // Index of "code" parameter:
    auto codeArgIdx = static_cast<int>(call->getNumArgs()) - 1;
    auto derivedFnArgIdx = codeArgIdx - 1;

    assert(call && "Must be set");
    assert(FD && "Trying to update with null FunctionDecl");

    DeclRefExpr* oldDRE = getArgFunction(call, SemaRef);

    assert(oldDRE && "Trying to differentiate something unsupported");

    ASTContext& C = SemaRef.getASTContext();

    FunctionDecl* replacementFD = OverloadedFD ? OverloadedFD : FD;
    // Create ref to generated FD.
    Expr* DRE =
        DeclRefExpr::Create(C, oldDRE->getQualifierLoc(), noLoc, replacementFD,
                            /*RefersToEnclosingVariableOrCapture=*/false,
                            replacementFD->getNameInfo(),
                            replacementFD->getType(), oldDRE->getValueKind());

    // Add the "&" operator
    auto newUnOp =
        SemaRef.BuildUnaryOp(nullptr, noLoc, UnaryOperatorKind::UO_AddrOf, DRE)
            .get();
    call->setArg(derivedFnArgIdx, newUnOp);

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

      StringLiteral* SL = utils::CreateStringLiteral(C,
                                                     Out.str());
      Expr* newArg =
        SemaRef.ImpCastExprToType(SL,
                                  Arg->getType(),
                                  CK_ArrayToPointerDecay).get();
      call->setArg(codeArgIdx, newArg);
    }
  }

  DiffCollector::DiffCollector(DeclGroupRef DGR, DiffInterval& Interval,
                               const DerivativesSet& Derivatives,
                               DiffSchedule& plans, clang::Sema& S)
    : m_Interval(Interval), m_GeneratedDerivatives(Derivatives),
      m_DiffPlans(plans), m_TopMostFD(nullptr), m_Sema(S) {

    if (Interval.empty())
      return;


    for (Decl* D : DGR) {
      // Skip over the derivatives that we produce.
      if (m_GeneratedDerivatives.count(D))
        continue;
      TraverseDecl(D);
    }
  }

  /// Returns true if `FD` is a call operator; otherwise returns false.
  static bool isCallOperator(ASTContext& Context, const FunctionDecl* FD) {
    if (auto method = dyn_cast<CXXMethodDecl>(FD)) {
      DeclarationName
          callOperatorDeclName = Context.DeclarationNames.getCXXOperatorName(
              OverloadedOperatorKind::OO_Call);
      return method->getNameInfo().getName() == callOperatorDeclName;              
    }
    return false;
  }

  bool DiffCollector::isInInterval(SourceLocation Loc) const {
    const SourceManager &SM = m_Sema.getSourceManager();
    for (size_t i = 0, e = m_Interval.size(); i < e; ++i) {
      SourceLocation B = m_Interval[i].getBegin();
      SourceLocation E = m_Interval[i].getEnd();
      assert((i == e-1 || E.isValid()) && "Unexpected open interval");
      assert(E.isInvalid() || SM.isBeforeInTranslationUnit(B, E));
      if (E.isValid() &&
          clad_compat::SourceManager_isPointWithin(SM, Loc, B, E))
        return true;
      else if (SM.isBeforeInTranslationUnit(B, Loc))
        return true;
    }
    return false;
  }

  bool DiffCollector::VisitCallExpr(CallExpr* E) {
    // Check if we should look into this.
    // FIXME: Generated code does not usually have valid source locations.
    // In that case we should ask the enclosing ast nodes for a source
    // location and check if it is within range.
    SourceLocation endLoc = E->getEndLoc();
    if (endLoc.isInvalid() || !isInInterval(endLoc))
        return true;

    FunctionDecl* FD = E->getDirectCallee();
    if (!FD)
      return true;
    // We need to find our 'special' diff annotated such:
    // clad::differentiate(...) __attribute__((annotate("D")))
    // TODO: why not check for its name? clad::differentiate/gradient?
    const AnnotateAttr* A = FD->getAttr<AnnotateAttr>();
    if (A &&
        (A->getAnnotation().equals("D") || A->getAnnotation().equals("G") ||
         A->getAnnotation().equals("H") || A->getAnnotation().equals("J") ||
         A->getAnnotation().equals("E"))) {
      // A call to clad::differentiate or clad::gradient was found.
      DeclRefExpr* DRE = getArgFunction(E, m_Sema);
      if (!DRE)
        return true;
      DiffRequest request{};

      if (A->getAnnotation().equals("D")) {
        request.Mode = DiffMode::forward;
        llvm::APSInt derivativeOrderAPSInt
          = FD->getTemplateSpecializationArgs()->get(0).getAsIntegral();
        // We know the first template spec argument is of unsigned type
        assert(derivativeOrderAPSInt.isUnsigned() && "Must be unsigned");
        unsigned derivativeOrder = derivativeOrderAPSInt.getZExtValue();
        request.RequestedDerivativeOrder = derivativeOrder;
      } else if (A->getAnnotation().equals("H")) {
        request.Mode = DiffMode::hessian;
      } else if (A->getAnnotation().equals("J")) {
        request.Mode = DiffMode::jacobian;
      } else if (A->getAnnotation().equals("G")) {
        request.Mode = DiffMode::reverse;
      } else {
        request.Mode = DiffMode::error_estimation;
      }
      request.CallContext = E;
      request.CallUpdateRequired = true;
      request.VerboseDiags = true;
      request.Args = E->getArg(1);
      auto derivedFD = cast<FunctionDecl>(DRE->getDecl());
      request.Function = derivedFD;
      request.BaseFunctionName = utils::ComputeEffectiveFnName(request.Function);

      if (isCallOperator(m_Sema.getASTContext(), request.Function)) {
        request.Functor = cast<CXXMethodDecl>(request.Function)->getParent();
      }

      // FIXME: add support for nested calls to clad::differentiate/gradient
      // inside differentiated functions
      assert(!m_TopMostFD &&
             "nested clad::differentiate/gradient are not yet supported");
      llvm::SaveAndRestore<const FunctionDecl*> saveTopMost = m_TopMostFD;
      m_TopMostFD = FD;
      TraverseDecl(derivedFD);
      m_DiffPlans.push_back(std::move(request));
    }
    /*else if (m_TopMostFD) {
      // If another function is called inside differentiated function,
      // this will be handled by Forward/ReverseModeVisitor::Derive.
    }*/
    return true;     // return false to abort visiting.
  }
} // end namespace
