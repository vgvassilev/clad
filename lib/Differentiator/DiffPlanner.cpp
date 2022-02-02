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
          utils::BuildNNS(m_SemaRef, callOperator->getDeclContext(), CSS,
                          /*addGlobalNS=*/true);

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
    DeclRefExpr* DRE =
        DeclRefExpr::Create(C, oldDRE->getQualifierLoc(), noLoc, replacementFD,
                            /*RefersToEnclosingVariableOrCapture=*/false,
                            replacementFD->getNameInfo(),
                            replacementFD->getType(), oldDRE->getValueKind());

    // We have a DeclRefExpr pointing to a member declaration, which is an
    // lvalue. However, due to an inconsistency of the expression classfication
    // in clang we need to change it to an r-value to avoid an assertion when
    // building a unary op. See llvm/llvm-project#53958.
    if (isa<CXXMethodDecl>(DRE->getDecl()))
      DRE->setValueKind(CLAD_COMPAT_ExprValueKind_R_or_PR_Value);

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

      StringLiteral* SL = utils::CreateStringLiteral(C, Out.str());
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
    DVI.clear();
    auto& C = semaRef.getASTContext();
    const Expr* diffArgs = Args;
    const FunctionDecl* FD = Function;
    FD = FD->getDefinition();
    if (!diffArgs || !FD) {
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
      llvm::SmallVector<llvm::StringRef, 16> diffParamsSpec{};
      do {
        llvm::StringRef pInfo{};
        std::tie(pInfo, string) = string.split(',');
        diffParamsSpec.push_back(pInfo.trim());
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

      auto computeParamName = [](llvm::StringRef diffSpec) {
        std::size_t idx = diffSpec.find_first_of(".[");
        return diffSpec.substr(0, idx);
      };

      // Ensure that diff params are always considered in the same order.
      // This is required to uniquely identify differentiation requests.
      std::sort(
          diffParamsSpec.begin(), diffParamsSpec.end(),
          [&candidates, &computeParamName](llvm::StringRef a,
                                           llvm::StringRef b) {
            auto a_it = std::find_if(
                candidates.begin(), candidates.end(),
                [a, &computeParamName](
                    const std::pair<llvm::StringRef, ValueDecl*>& candidate) {
                  return candidate.first == computeParamName(a);
                });
            auto b_it = std::find_if(
                candidates.begin(), candidates.end(),
                [b, &computeParamName](
                    const std::pair<llvm::StringRef, ValueDecl*>& candidate) {
                  return candidate.first == computeParamName(b);
                });
            return a_it < b_it;
          });

      for (const auto& diffSpec : diffParamsSpec) {
        DiffInputVarInfo dVarInfo;

        dVarInfo.source = diffSpec.str();
        llvm::StringRef pName = computeParamName(diffSpec);
        auto it = std::find_if(std::begin(candidates), std::end(candidates),
                               [&pName](
                                   const std::pair<llvm::StringRef, ValueDecl*>&
                                       p) { return p.first == pName; });

        if (it == std::end(candidates)) {
          // Fail if the function has no parameter with specified name.
          utils::EmitDiag(semaRef, DiagnosticsEngine::Error,
                          diffArgs->getEndLoc(),
                          "Requested parameter name '%0' was not found among "
                          "function "
                          "parameters",
                          {pName});
          return;
        }

        auto f_it = std::find_if(std::begin(DVI), std::end(DVI),
                                 [&it](const DiffInputVarInfo& dVarInfo) {
                                   return dVarInfo.param == it->second;
                                 });

        if (f_it != DVI.end()) {
          utils::
              EmitDiag(semaRef, DiagnosticsEngine::Error, diffArgs->getEndLoc(),
                       "Requested parameter '%0' was specified multiple times",
                       {it->second->getName()});
          return;
        }

        dVarInfo.param = it->second;
        
        std::size_t lSqBracketIdx = diffSpec.find("[");
        if (lSqBracketIdx != llvm::StringRef::npos) {
          llvm::StringRef interval(diffSpec.slice(lSqBracketIdx + 1, diffSpec.find(']')));
          llvm::StringRef firstStr, lastStr;
          std::tie(firstStr, lastStr) = interval.split(':');

          if (lastStr.empty()) {
            // The string is not a range just a single index
            size_t index;
            firstStr.getAsInteger(10, index);
            dVarInfo.paramIndexInterval = IndexInterval(index);
          } else {
            size_t first, last;
            firstStr.getAsInteger(10, first);
            lastStr.getAsInteger(10, last);
            if (first >= last) {
              utils::EmitDiag(semaRef, DiagnosticsEngine::Error,
                              diffArgs->getEndLoc(),
                              "Range specified in '%0' is in incorrect format",
                              {diffSpec});
              return;
            }
            dVarInfo.paramIndexInterval = IndexInterval(first, last);
          }
        } else {
          dVarInfo.paramIndexInterval = IndexInterval();
        }

        std::size_t dotIdx = diffSpec.find(".");
        dotIdx += (dotIdx != StringRef::npos);
        StringRef fieldsSpec = diffSpec.substr(dotIdx);
        while (!fieldsSpec.empty()) {
          StringRef fieldName;
          std::tie(fieldName, fieldsSpec) = fieldsSpec.split('.');
          dVarInfo.fields.push_back(fieldName.str());
        }

        if (!dVarInfo.param->getType()->isRecordType() &&
            !dVarInfo.fields.empty()) {
          utils::EmitDiag(
              semaRef, DiagnosticsEngine::Level::Error, diffArgs->getEndLoc(),
              "Fields can only be provided for class type parameters. "
              "Field information is incorrectly specified in '%0' "
              "for non-class type parameter '%1'",
              {diffSpec, pName});
          return;
        }

        if (!dVarInfo.fields.empty()) {
          RecordDecl* RD = dVarInfo.param->getType()->getAsCXXRecordDecl();
          llvm::SmallVector<llvm::StringRef, 4> ref(dVarInfo.fields.begin(),
                                                    dVarInfo.fields.end());
          bool isValid = utils::IsValidMemExprPath(semaRef, RD, ref);
          if (!isValid) {
            utils::EmitDiag(
                semaRef, DiagnosticsEngine::Level::Error, diffArgs->getEndLoc(),
                "Path specified by fields in '%0' is invalid.", {diffSpec});
            return;
          }
        }

        DVI.push_back(dVarInfo);
      }
      return;
    }
    // Case 2)
    // Check if the provided literal can be evaluated as an integral value.
    llvm::APSInt intValue;
    if (clad_compat::Expr_EvaluateAsInt(E, intValue, C)) {
      DiffInputVarInfo dVarInfo;
      auto idx = intValue.getExtValue();
      // If we are differentiating a call operator that have no parameters, then
      // we need to search for independent parameters in fields of the
      // class that defines the call operator instead.
      if (FD->param_empty() && Functor) {
        int totalFields = std::distance(Functor->field_begin(),
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
        dVarInfo.param = *std::next(Functor->field_begin(), idx);
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
        dVarInfo.param = FD->getParamDecl(idx);
      }
      // Returns a single parameter.
      DVI.push_back(dVarInfo);
      return;
    }
    // Case 3)
    // Treat the default (unspecified) argument as a special case, as if all
    // function's arguments were requested.
    if (isa<CXXDefaultArgExpr>(E)) {
      std::copy(FD->param_begin(), FD->param_end(), std::back_inserter(params));
      
      // If the function has no parameters, then we cannot differentiate it."
      // and if the DiffMode is Jacobian, we must have atleast 2 parameters.
      if (params.empty() || (params.size()==1 && this->Mode == DiffMode::jacobian)) {
        utils::EmitDiag(semaRef, DiagnosticsEngine::Error,
                        CallContext->getEndLoc(),
                        "Attempted to differentiate a function without "
                        "parameters");
        return;
      }

      // If it is a Vector valued function, the last parameter is to store the
      // output vector and hence is not a differentiable parameter, so we must
      // pop it out
      if (this->Mode == DiffMode::jacobian){
        params.pop_back();
      }

      // insert an empty index for each parameter.
      for (unsigned i=0; i<params.size(); ++i) {
        DiffInputVarInfo dVarInfo(params[i], IndexInterval());
        DVI.push_back(dVarInfo);
      }
      return;
    }
    // Fail if the argument is not a string or numeric literal.
    utils::EmitDiag(semaRef, DiagnosticsEngine::Error, diffArgs->getEndLoc(),
                    "Failed to parse the parameters, must be a string or "
                    "numeric literal");
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

      // Check if enzyme is requested in case of Forward Mode or reverse Mode
      if (A->getAnnotation().equals("D") || A->getAnnotation().equals("G")) {
        signed enzyme_request = FD->getTemplateSpecializationArgs()
                                    ->get(0)
                                    .getAsIntegral()
                                    .getZExtValue();

        if (enzyme_request == -1)
          request.use_enzyme = true;
      }

      if (A->getAnnotation().equals("D")) {
        request.Mode = DiffMode::forward;
        llvm::APSInt derivativeOrderAPSInt
          = FD->getTemplateSpecializationArgs()->get(0).getAsIntegral();

        unsigned derivativeOrder = 1;
        if (!request.use_enzyme)
          derivativeOrder = derivativeOrderAPSInt.getZExtValue();
        request.RequestedDerivativeOrder = derivativeOrder;
      } else if (A->getAnnotation().equals("H")) {
        request.Mode = DiffMode::hessian;
      } else if (A->getAnnotation().equals("J")) {
        request.Mode = DiffMode::jacobian;
      } else if (A->getAnnotation().equals("G")) {
        request.Mode = DiffMode::reverse;
      } else {
        request.Mode = DiffMode::error_estimation;
        llvm::APSInt val =
            FD->getTemplateSpecializationArgs()->get(0).getAsIntegral();
        request.PrintFPErrors = val.getZExtValue();
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
