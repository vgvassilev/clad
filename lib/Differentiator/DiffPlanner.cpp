#include "clad/Differentiator/DiffPlanner.h"

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

#include <algorithm>

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

      // Copied and adapted from clang::Sema::ActOnStringLiteral.
      QualType CharTyConst = C.CharTy;
      CharTyConst.addConst();
      // Get an array type for the string, according to C99 6.4.5. This includes
      // the nul terminator character as well as the string length for pascal
      // strings.
      QualType StrTy =
          clad_compat::getConstantArrayType(C, CharTyConst,
                                            llvm::APInt(32,
                                                        Out.str().size() + 1),
                                            /*SizeExpr=*/nullptr,
                                            ArrayType::Normal,
                                            /*IndexTypeQuals*/ 0);

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
  }

  DiffCollector::DiffCollector(DeclGroupRef DGR, DiffInterval& Interval,
                               DiffSchedule& plans, clang::Sema& S)
      : m_Interval(Interval), m_DiffPlans(plans), m_TopMostFD(nullptr),
        m_Sema(S) {

    if (Interval.empty())
      return;

    for (Decl* D : DGR) {
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

  void DiffRequest::UpdateDiffParamsInfo(Sema& semaRef) {
    auto& C = semaRef.getASTContext();
    const Expr* diffArgs = Args;
    const FunctionDecl* FD = Function;
    FD = FD->getDefinition();
    if (!diffArgs || !FD) {
      DiffParamsInfo = {{}, {}};
      return;
    }
    DiffParams params{};
    auto E = diffArgs->IgnoreParenImpCasts();
    // Case 1)
    if (auto SL = dyn_cast<StringLiteral>(E)) {
      IndexIntervalTable indexes{};
      llvm::StringRef string = SL->getString().trim();
      if (string.empty()) {
        utils::EmitDiag(semaRef, DiagnosticsEngine::Error,
                        diffArgs->getEndLoc(), "No parameters were provided");
        return;
      }
      // Split the string by ',' characters, trim whitespaces.
      llvm::SmallVector<llvm::StringRef, 16> names{};
      llvm::StringRef name{};
      do {
        std::tie(name, string) = string.split(',');
        names.push_back(name.trim());
      } while (!string.empty());
      // Stores parameters and field declarations to be used as candidates for
      // independent arguments.
      // If we are differentiating a call operator that have no parameters,
      // then candidates for independent argument are member variables of the
      // class that defines the call operator.
      // Otherwise, candidates are parameters of the function that we are
      // differentiating.
      llvm::SmallVector<std::pair<llvm::StringRef, ValueDecl*>, 16>
          candidates{};

      // find and store candidate parameters.
      if (FD->param_empty() && Functor) {
        for (FieldDecl* fieldDecl : Functor->fields())
          candidates.emplace_back(fieldDecl->getName(), fieldDecl);

      } else {
        for (auto PVD : FD->parameters())
          candidates.emplace_back(PVD->getName(), PVD);
      }

      // Ensure that diff params are always considered in the same order.
      std::sort(
          names.begin(), names.end(),
          [&candidates](llvm::StringRef a, llvm::StringRef b) {
            auto a_it = std::find_if(
                candidates.begin(), candidates.end(),
                [a](const std::pair<llvm::StringRef, ValueDecl*>& candidate) {
                  return candidate.first == a;
                });
            auto b_it = std::find_if(
                candidates.begin(), candidates.end(),
                [b](const std::pair<llvm::StringRef, ValueDecl*>& candidate) {
                  return candidate.first == b;
                });
            return a_it < b_it;
          });

      for (const auto& name : names) {
        size_t loc = name.find('[');
        loc = (loc == llvm::StringRef::npos) ? name.size() : loc;
        llvm::StringRef base = name.substr(0, loc);

        auto it = std::find_if(std::begin(candidates), std::end(candidates),
                               [&base](
                                   const std::pair<llvm::StringRef, ValueDecl*>&
                                       p) { return p.first == base; });

        if (it == std::end(candidates)) {
          // Fail if the function has no parameter with specified name.
          utils::EmitDiag(semaRef, DiagnosticsEngine::Error,
                          diffArgs->getEndLoc(),
                          "Requested parameter name '%0' was not found among "
                          "function "
                          "parameters",
                          {base});
          return;
        }

        auto f_it = std::find(std::begin(params), std::end(params), it->second);

        if (f_it != params.end()) {
          utils::
              EmitDiag(semaRef, DiagnosticsEngine::Error, diffArgs->getEndLoc(),
                       "Requested parameter '%0' was specified multiple times",
                       {it->second->getName()});
          return;
        }

        params.push_back(it->second);

        if (loc != name.size()) {
          llvm::StringRef interval(name.slice(loc + 1, name.find(']')));
          llvm::StringRef firstStr, lastStr;
          std::tie(firstStr, lastStr) = interval.split(':');

          if (lastStr.empty()) {
            // The string is not a range just a single index
            size_t index;
            firstStr.getAsInteger(10, index);
            indexes.push_back(IndexInterval(index));
          } else {
            size_t first, last;
            firstStr.getAsInteger(10, first);
            lastStr.getAsInteger(10, last);
            if (first >= last) {
              utils::EmitDiag(semaRef, DiagnosticsEngine::Error,
                              diffArgs->getEndLoc(),
                              "Range specified in '%0' is in incorrect format",
                              {name});
              return;
            }
            indexes.push_back(IndexInterval(first, last));
          }
        } else {
          indexes.push_back(IndexInterval());
        }
      }
      // Return a sequence of function's parameters.
      DiffParamsInfo = {params, indexes};
      return;
    }
    // Case 2)
    // Check if the provided literal can be evaluated as an integral value.
    llvm::APSInt intValue;
    if (clad_compat::Expr_EvaluateAsInt(E, intValue, C)) {
      auto idx = intValue.getExtValue();
      // If we are differentiating a call operator that have no parameters, then
      // we need to search for independent parameters in fields of the
      // class that defines the call operator instead.
      if (FD->param_empty() && Functor) {
        size_t totalFields = std::distance(Functor->field_begin(),
                                           Functor->field_end());
        // Fail if the specified index is invalid.
        if ((idx < 0) || idx >= totalFields) {
          utils::EmitDiag(semaRef, DiagnosticsEngine::Error,
                          diffArgs->getEndLoc(),
                          "Invalid member variable index '%0' of '%1' member "
                          "variable(s)",
                          {std::to_string(idx), std::to_string(totalFields)});
          return;
        }
        params.push_back(*std::next(Functor->field_begin(), idx));
      } else {
        // Fail if the specified index is invalid.
        if ((idx < 0) || (idx >= FD->getNumParams())) {
          utils::EmitDiag(semaRef, DiagnosticsEngine::Error,
                          diffArgs->getEndLoc(),
                          "Invalid argument index '%0' of '%1' argument(s)",
                          {std::to_string(idx),
                           std::to_string(FD->getNumParams())});
          return;
        }
        params.push_back(FD->getParamDecl(idx));
      }
      // Returns a single parameter.
      DiffParamsInfo = {params, {}};
      return;
    }
    // Case 3)
    // Treat the default (unspecified) argument as a special case, as if all
    // function's arguments were requested.
    if (isa<CXXDefaultArgExpr>(E)) {
      std::copy(FD->param_begin(), FD->param_end(), std::back_inserter(params));
      // If the function has no parameters, then we cannot differentiate it."
      if (params.empty()) {
        utils::EmitDiag(semaRef, DiagnosticsEngine::Error,
                        CallContext->getEndLoc(),
                        "Attempted to differentiate a function without "
                        "parameters");
        return;
      }
      IndexIntervalTable indexes{};
      // insert an empty index for each parameter.
      for (unsigned i=0; i<params.size(); ++i)
        indexes.push_back(IndexInterval());
      // Returns the sequence with all the function's parameters.
      DiffParamsInfo = {params, indexes};
      return;
    }
    // Fail if the argument is not a string or numeric literal.
    utils::EmitDiag(semaRef, DiagnosticsEngine::Error, diffArgs->getEndLoc(),
                    "Failed to parse the parameters, must be a string or "
                    "numeric literal");
    DiffParamsInfo = {{}, {}};
    return;
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
