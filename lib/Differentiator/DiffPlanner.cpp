#include "clad/Differentiator/DiffPlanner.h"

#include "clad/Differentiator/DiffMode.h"

#include "ActivityAnalyzer.h"
#include "TBRAnalyzer.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTLambda.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Analysis/CallGraph.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Sema/TemplateDeduction.h"

#include "clad/Differentiator/CladConfig.h"
#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"

#include <algorithm>
#include <string>

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

    assert(call && "Must be set");
    assert(FD && "Trying to update with null FunctionDecl");

    DeclRefExpr* oldDRE = getArgFunction(call, SemaRef);

    assert(oldDRE && "Trying to differentiate something unsupported");

    ASTContext& C = SemaRef.getASTContext();

    FunctionDecl* replacementFD = OverloadedFD ? OverloadedFD : FD;

    auto codeArgIdx = -1;
    auto derivedFnArgIdx = -1;
    auto idx = 0;
    for (auto* arg : call->arguments()) {
      if (auto* default_arg_expr = dyn_cast<CXXDefaultArgExpr>(arg)) {
        std::string argName = default_arg_expr->getParam()->getNameAsString();
        if (argName == "derivedFn")
          derivedFnArgIdx = idx;
        else if (argName == "code")
          codeArgIdx = idx;
      }
      ++idx;
    }

    // Index of "CUDAkernel" parameter:
    int numArgs = static_cast<int>(call->getNumArgs());
    if (numArgs > 4) {
      auto kernelArgIdx = numArgs - 1;
      auto* cudaKernelFlag =
          SemaRef
              .ActOnCXXBoolLiteral(noLoc,
                                   replacementFD->hasAttr<CUDAGlobalAttr>()
                                       ? tok::kw_true
                                       : tok::kw_false)
              .get();
      call->setArg(kernelArgIdx, cudaKernelFlag);
      numArgs--;
    }

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

    if (derivedFnArgIdx != -1) {
      // Add the "&" operator
      auto* newUnOp =
          SemaRef
              .BuildUnaryOp(nullptr, noLoc, UnaryOperatorKind::UO_AddrOf, DRE)
              .get();
      call->setArg(derivedFnArgIdx, newUnOp);
    }

    // Update the code parameter if it was found.
    if (codeArgIdx != -1) {
      if (auto* Arg = dyn_cast<CXXDefaultArgExpr>(call->getArg(codeArgIdx))) {
        clang::LangOptions LangOpts;
        LangOpts.CPlusPlus = true;
        clang::PrintingPolicy Policy(LangOpts);
        Policy.Bool = true;

        std::string s;
        llvm::raw_string_ostream Out(s);
        FD->print(Out, Policy);
        Out.flush();

        StringLiteral* SL = utils::CreateStringLiteral(C, Out.str());
        Expr* newArg =
            SemaRef
                .ImpCastExprToType(SL, Arg->getType(), CK_ArrayToPointerDecay)
                .get();
        call->setArg(codeArgIdx, newArg);
      }
    }
  }

  DiffCollector::DiffCollector(DeclGroupRef DGR, DiffInterval& Interval,
                               clad::DynamicGraph<DiffRequest>& requestGraph,
                               clang::Sema& S, RequestOptions& opts)
      : m_Interval(Interval), m_DiffRequestGraph(requestGraph), m_Sema(S),
        m_Options(opts) {

    if (Interval.empty())
      return;

    assert(!m_TopMostReq && "Traversal already in flight!");
    for (Decl* D : DGR) {
      TraverseDecl(D);
    }
    m_TopMostReq = nullptr;
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
    // Diff info for pullbacks is generated automatically,
    // its parameters are not provided by the user.
    if (Mode == DiffMode::experimental_pullback) {
      // Might need to update DVI args, as they may be pointing to the
      // declaration parameters, not the definition parameters.
      if (!Function->getPreviousDecl())
        // If the function was never declared before, we can safely assume
        // that the parameters are correctly referring to the definition ones.
        return;
      const FunctionDecl* FD = Function->getPreviousDecl();
      for (size_t i = 0, e = DVI.size(), paramIdx = 0;
           i < e && paramIdx < FD->getNumParams(); ++i) {
        const auto* param = DVI[i].param;
        while (paramIdx < FD->getNumParams() &&
               FD->getParamDecl(paramIdx) != param) {
            ++paramIdx;
        }
        if (paramIdx != FD->getNumParams())
            // Update the parameter to point to the definition parameter.
            DVI[i].param = Function->getParamDecl(paramIdx);
      }
      return;
    }
    DVI.clear();
    auto& C = semaRef.getASTContext();
    const Expr* diffArgs = Args;
    const FunctionDecl* FD = Function;
    if (!DeclarationOnly)
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
        // Check if diffSpec represents an index of an independent variable.
        if ('0' <= diffSpec[0] && diffSpec[0] <= '9') {
          unsigned idx = std::stoi(dVarInfo.source);
          // Fail if the specified index is invalid.
          if (idx >= FD->getNumParams()) {
            utils::EmitDiag(
                semaRef, DiagnosticsEngine::Error, diffArgs->getEndLoc(),
                "Invalid argument index '%0' of '%1' argument(s)",
                {std::to_string(idx), std::to_string(FD->getNumParams())});
            return;
          }
          dVarInfo.param = FD->getParamDecl(idx);
          DVI.push_back(dVarInfo);
          continue;
        }
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

          static constexpr unsigned Radix = 10;
          if (lastStr.empty()) {
            // The string is not a range just a single index
            size_t index;
            if (firstStr.getAsInteger(Radix, index)) {
                utils::EmitDiag(semaRef, DiagnosticsEngine::Error,
                                diffArgs->getEndLoc(),
                                "Could not parse index '%0'", {diffSpec});
                return;
            }
            dVarInfo.paramIndexInterval = IndexInterval(index);
          } else {
            size_t first, last;
            if (firstStr.getAsInteger(Radix, first) ||
                lastStr.getAsInteger(Radix, last)) {
                utils::EmitDiag(semaRef, DiagnosticsEngine::Error,
                                diffArgs->getEndLoc(),
                                "Could not parse range '%0'", {diffSpec});
                return;
            }
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
    Expr::EvalResult res;
    Expr::SideEffectsKind AllowSideEffects =
        Expr::SideEffectsKind::SE_NoSideEffects;
    if (E->EvaluateAsInt(res, C, AllowSideEffects)) {
      intValue = res.Val.getInt();
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
      if (params.empty()) {
        utils::EmitDiag(semaRef, DiagnosticsEngine::Error,
                        CallContext->getEndLoc(),
                        "Attempted to differentiate a function without "
                        "parameters");
        return;
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

  void DiffRequest::print(llvm::raw_ostream& Out) const {
    Out << '<';
    PrintingPolicy P(Function->getASTContext().getLangOpts());
    P.TerseOutput = true;
    P.FullyQualifiedName = true;
    Function->print(Out, P, /*Indentation=*/0, /*PrintInstantiation=*/true);
    Out << ">[name=" << BaseFunctionName << ", "
        << "order=" << CurrentDerivativeOrder << ", "
        << "mode=" << DiffModeToString(Mode) << ", "
        << "args='";
    if (Args)
      Args->printPretty(Out, /*Helper=*/nullptr, P);
    for (unsigned i = 0, e = DVI.size(); i < e; i++) {
      DVI[i].print(Out);
      if (i != e - 1)
        Out << ',';
    }
    Out << "'";
    if (EnableTBRAnalysis)
      Out << ", tbr";
    Out << ']';
    Out.flush();
  }

  bool DiffRequest::shouldBeRecorded(Expr* E) const {
    if (!EnableTBRAnalysis)
      return true;

    if (!isa<DeclRefExpr>(E) && !isa<ArraySubscriptExpr>(E) &&
        !isa<MemberExpr>(E))
      return true;

    // FIXME: currently, we allow all pointer operations to be stored.
    // This is not correct, but we need to implement a more advanced analysis
    // to determine which pointer operations are useful to store.
    if (E->getType()->isPointerType())
      return true;

    if (!m_TbrRunInfo.HasAnalysisRun) {
      TBRAnalyzer analyzer(Function->getASTContext(),
                           m_TbrRunInfo.ToBeRecorded);
      analyzer.Analyze(Function);
      m_TbrRunInfo.HasAnalysisRun = true;
    }
    auto found = m_TbrRunInfo.ToBeRecorded.find(E->getBeginLoc());
    return found != m_TbrRunInfo.ToBeRecorded.end();
  }

  bool DiffRequest::shouldHaveAdjoint(const VarDecl* VD) const {
    if (!EnableVariedAnalysis)
      return true;

    if (!m_ActivityRunInfo.HasAnalysisRun) {
      ArrayRef<ParmVarDecl*> FDparam = Function->parameters();
      std::vector<ParmVarDecl*> derivedParam;

      for (auto* parameter : FDparam) {
        QualType parType = parameter->getType();
        while (parType->isPointerType())
          parType = parType->getPointeeType();
        if (!parType.isConstQualified())
          derivedParam.push_back(parameter);
      }

      std::copy(derivedParam.begin(), derivedParam.end(),
                std::inserter(m_ActivityRunInfo.ToBeRecorded,
                              m_ActivityRunInfo.ToBeRecorded.end()));

      VariedAnalyzer analyzer(Function->getASTContext(),
                              m_ActivityRunInfo.ToBeRecorded);
      analyzer.Analyze(Function);
      m_ActivityRunInfo.HasAnalysisRun = true;
    }
    auto found = m_ActivityRunInfo.ToBeRecorded.find(VD);
    return found != m_ActivityRunInfo.ToBeRecorded.end();
  }

  std::string DiffRequest::ComputeDerivativeName() const {
    if (Mode != DiffMode::forward && Mode != DiffMode::reverse &&
        Mode != DiffMode::vector_forward_mode) {
      std::string name = BaseFunctionName + "_" + DiffModeToString(Mode);
      if (!Function->hasAttr<clang::CUDAGlobalAttr>())
        return name;

      // CUDA
      for (unsigned i = 0, e = Function->getNumParams(); i < e; ++i)
        if (HasIndependentParameter(Function->getParamDecl(i)))
          name += "_" + std::to_string(i);

      return name;
    }

    if (DVI.empty())
      return "<no independent variable specified>";

    // FIXME: Harmonize names accross modes. We have darg0 but dvec_0 and _grad.
    std::string argInfo;
    for (const DiffInputVarInfo& dParamInfo : DVI) {
      // If we differentiate w.r.t all arguments we do not need to specify them.
      if (DVI.size() == Function->getNumParams() && Mode != DiffMode::forward)
        break;

      const ValueDecl* IndP = dParamInfo.param;
      // If we are differentiating a call operator, that has no parameters,
      // then the specified independent argument is a member variable of the
      // class defining the call operator.
      // Thus, we need to find index of the member variable instead.
      unsigned idx = ~0U;
      if (Function->param_empty() && Functor) {
        auto it = std::find(Functor->field_begin(), Functor->field_end(), IndP);
        idx = std::distance(Functor->field_begin(), it);
      } else {
        const auto* it =
            std::find(Function->param_begin(), Function->param_end(), IndP);
        idx = std::distance(Function->param_begin(), it);
      }
      argInfo += ((Mode == DiffMode::forward) ? "" : "_") + std::to_string(idx);

      if (dParamInfo.paramIndexInterval.isValid()) {
        assert(utils::isArrayOrPointerType(IndP->getType()) && "Not array?");
        // FIXME: What about ranges [Start;Finish)?
        argInfo += "_" + std::to_string(dParamInfo.paramIndexInterval.Start);
      }

      for (const std::string& field : dParamInfo.fields)
        argInfo += "_" + field;
    }

    if (Mode == DiffMode::vector_forward_mode) {
      if (DVI.size() != Function->getNumParams())
        return BaseFunctionName + "_dvec" + argInfo;
      return BaseFunctionName + "_dvec";
    }

    if (Mode == DiffMode::reverse) {
      if (DVI.size() != Function->getNumParams())
        return BaseFunctionName + "_grad" + argInfo;
      if (use_enzyme)
        return BaseFunctionName + "_grad" + "_enzyme";
      return BaseFunctionName + "_grad";
    }

    std::string s;
    if (CurrentDerivativeOrder > 1)
      s = std::to_string(CurrentDerivativeOrder);

    return BaseFunctionName + "_d" + s + "arg" + argInfo;
  }

  bool DiffRequest::HasIndependentParameter(const ParmVarDecl* PVD) const {
    // FIXME: We store the original function's params in DVI and here we need to
    // compare with the cloned ones by name. We can compare the pointers instead
    // of strings if we built the function cloning in the DiffRequest.
    for (const DiffInputVarInfo& dParam : DVI) {
      if (PVD->getName() == dParam.param->getNameAsString())
        return true;

      // FIXME: Gross hack to handle shouldUseCudaAtomicOps...
      std::string pName = "_d_" + dParam.param->getNameAsString();
      if (pName == PVD->getName())
        return true;
    }
    return false;
  }

  ///\returns true on error.
  static bool ProcessInvocationArgs(Sema& S, SourceLocation endLoc,
                                    const RequestOptions& ReqOpts,
                                    const FunctionDecl* FD,
                                    DiffRequest& request) {
    const AnnotateAttr* A = FD->getAttr<AnnotateAttr>();
    std::string Annotation = A->getAnnotation().str();
    if (Annotation == "E") {
      // Error estimation has no options yet.
      request.Mode = DiffMode::error_estimation;
      return false;
    }

    if (Annotation == "D")
      request.Mode = DiffMode::forward;
    else if (Annotation == "H")
      request.Mode = DiffMode::hessian;
    else if (Annotation == "J")
      request.Mode = DiffMode::jacobian;
    else if (Annotation == "G")
      request.Mode = DiffMode::reverse;
    else {
      utils::EmitDiag(S, DiagnosticsEngine::Error, endLoc, "Unknown mode '%0'",
                      A->getAnnotation());
      return true;
    }

    request.EnableTBRAnalysis = ReqOpts.EnableTBRAnalysis;
    request.EnableVariedAnalysis = ReqOpts.EnableVariedAnalysis;

    const TemplateArgumentList* TAL = FD->getTemplateSpecializationArgs();
    assert(TAL && "Call must have specialization args!");

    // bitmask_opts is a template pack of unsigned integers, so we need to
    // do bitwise or of all the values to get the final value.
    unsigned bitmasked_opts_value = 0;
    const auto template_arg = TAL->get(0);
    if (template_arg.getKind() == TemplateArgument::Pack)
      for (const auto& arg : TAL->get(0).pack_elements())
        bitmasked_opts_value |= arg.getAsIntegral().getExtValue();

    bool enable_tbr_in_req =
        clad::HasOption(bitmasked_opts_value, clad::opts::enable_tbr);
    bool disable_tbr_in_req =
        clad::HasOption(bitmasked_opts_value, clad::opts::disable_tbr);
    bool enable_va_in_req =
        clad::HasOption(bitmasked_opts_value, clad::opts::enable_va);
    bool disable_va_in_req =
        clad::HasOption(bitmasked_opts_value, clad::opts::disable_va);

    // Sanity checks.
    if (enable_tbr_in_req && disable_tbr_in_req) {
      utils::EmitDiag(S, DiagnosticsEngine::Error, endLoc,
                      "Both enable and disable TBR options are specified.");
      return true;
    }
    if (enable_va_in_req && disable_va_in_req) {
      utils::EmitDiag(S, DiagnosticsEngine::Error, endLoc,
                      "Both enable and disable VA options are specified.");
      return true;
    }
    if (enable_tbr_in_req && request.Mode == DiffMode::forward) {
      utils::EmitDiag(S, DiagnosticsEngine::Error, endLoc,
                      "TBR analysis is not meant for forward mode AD.");
      return true;
    }

    // reverse vector mode is not yet supported.
    if (request.Mode == DiffMode::reverse &&
        clad::HasOption(bitmasked_opts_value, clad::opts::vector_mode)) {
      utils::EmitDiag(S, DiagnosticsEngine::Error, endLoc,
                      "Reverse vector mode is not yet supported.");
      return true;
    }

    // Override the default value of TBR analysis.
    if (enable_tbr_in_req || disable_tbr_in_req)
      request.EnableTBRAnalysis = enable_tbr_in_req && !disable_tbr_in_req;

    // Override the default value of TBR analysis.
    if (enable_va_in_req || disable_va_in_req)
      request.EnableVariedAnalysis = enable_va_in_req && !disable_va_in_req;

    // Check for clad::hessian<diagonal_only>.
    if (clad::HasOption(bitmasked_opts_value, clad::opts::diagonal_only)) {
      if (request.Mode == DiffMode::hessian) {
        request.Mode = DiffMode::hessian_diagonal;
        return false;
      }
      utils::EmitDiag(S, DiagnosticsEngine::Error, endLoc,
                      "Diagonal only option is only valid for Hessian mode.");
      return true;
    }

    if (clad::HasOption(bitmasked_opts_value, clad::opts::use_enzyme))
      request.use_enzyme = true;

    if (request.Mode == DiffMode::forward) {
      // Check for clad::differentiate<N>.
      if (unsigned order = clad::GetDerivativeOrder(bitmasked_opts_value))
        request.RequestedDerivativeOrder = order;

      // Check for clad::differentiate<immediate_mode>.
      if (clad::HasOption(bitmasked_opts_value, clad::opts::immediate_mode))
        request.ImmediateMode = true;

      // Check for clad::differentiate<vector_mode>.
      if (clad::HasOption(bitmasked_opts_value, clad::opts::vector_mode)) {
        request.Mode = DiffMode::vector_forward_mode;

        // Currently only first order derivative is supported.
        if (request.RequestedDerivativeOrder != 1) {
          utils::EmitDiag(S, DiagnosticsEngine::Error, endLoc,
                          "Only first order derivative is supported for now "
                          "in vector forward mode.");
          return true;
        }

        // We don't yet support enzyme with vector mode.
        if (request.use_enzyme) {
          utils::EmitDiag(S, DiagnosticsEngine::Error, endLoc,
                          "Enzyme's vector mode is not yet supported.");
          return true;
        }
      }
    }

    return false;
  }

  static bool HasCustomDerivativeForDecl(Sema& S, const DiffRequest& R) {
    NamespaceDecl* cladNS = utils::LookupNSD(S, "clad", /*shouldExist=*/true);
    NamespaceDecl* customDerNS = utils::LookupNSD(
        S, "custom_derivatives", /*shouldExist=*/false, cladNS);
    if (!customDerNS)
      return false;

    const Expr* callSite = R.CallContext;
    const DeclContext* originalFnDC = nullptr;
    // Check if the callSite is not associated with a shadow declaration.
    if (const auto* ME = dyn_cast<CXXMemberCallExpr>(callSite)) {
      originalFnDC = ME->getMethodDecl()->getParent();
    } else if (const auto* CE = dyn_cast<CallExpr>(callSite)) {
      const Expr* Callee = CE->getCallee()->IgnoreParenCasts();
      if (const auto* DRE = dyn_cast<DeclRefExpr>(Callee))
        originalFnDC = DRE->getFoundDecl()->getDeclContext();
      else if (const auto* MemberE = dyn_cast<MemberExpr>(Callee))
        originalFnDC = MemberE->getFoundDecl().getDecl()->getDeclContext();
    } else if (const auto* CtorExpr = dyn_cast<CXXConstructExpr>(callSite)) {
      originalFnDC = CtorExpr->getConstructor()->getDeclContext();
    }

    DeclContext* DC = customDerNS;

    if (isa<RecordDecl>(originalFnDC)) {
      return true;
      // FIXME: Re-enable
      // DC = utils::LookupNSD(S, "class_functions", /*shouldExist=*/false, DC);
    } else
      DC = utils::FindDeclContext(S, DC, originalFnDC);

    if (!DC)
      return false;

    std::string Name = R.BaseFunctionName;
    if (R.Mode == DiffMode::experimental_pullback && R->getNumParams() > 1)
      Name += "_pullback";
    else // if (Mode == DiffMode::experimental_pullback)
      Name += "_pushforward";

    IdentifierInfo* II = &S.getASTContext().Idents.get(Name);
    DeclarationNameInfo DNInfo(II, utils::GetValidSLoc(S));
    LookupResult Found(S, DNInfo, Sema::LookupOrdinaryName);
    S.LookupQualifiedName(Found, DC);

    return !Found.empty();
  }

  // FIXME: Add varied analysis.
  static bool allArgumentsAreLiterals(const CallExpr* CE) {
    for (const Expr* A : CE->arguments()) {
      A = A->IgnoreImpCasts();
      if (!isa<CXXNullPtrLiteralExpr>(A) && !isa<CharacterLiteral>(A) &&
          !isa<CXXBoolLiteralExpr>(A) && !isa<FixedPointLiteral>(A) &&
          !isa<ImaginaryLiteral>(A) && !isa<IntegerLiteral>(A) &&
          !isa<ObjCBoolLiteralExpr>(A) && !isa<ObjCStringLiteral>(A) &&
          !isa<FloatingLiteral>(A) && !isa<ObjCArrayLiteral>(A) &&
          !isa<StringLiteral>(A) && !isa<CompoundLiteralExpr>(A))
        return false; // non-constant found.
    }
    return true; // all constants.
  }

  bool DiffCollector::VisitCallExpr(CallExpr* E) {
    // Check if we should look into this.
    DiffRequest request;

    FunctionDecl* FD = E->getDirectCallee();
    if (!FD)
      return true;

    // FIXME: We might want to support nested calls to differentiate/gradient
    // inside differentiated functions.
    if (!m_TopMostReq) {
      // FIXME: Generated code does not usually have valid source locations.
      // In that case we should ask the enclosing ast nodes for a source
      // location and check if it is within range.
      SourceLocation endLoc = E->getEndLoc();
      if (endLoc.isInvalid() || !isInInterval(endLoc))
        return true;

      // We need to find our 'special' diff annotated such:
      // clad::differentiate(...) __attribute__((annotate("D")))
      // TODO: why not check for its name? clad::differentiate/gradient?
      const AnnotateAttr* A = FD->getAttr<AnnotateAttr>();

      if (!A)
        return true;

      std::string Annotation = A->getAnnotation().str();
      if (Annotation != "D" && Annotation != "G" && Annotation != "H" &&
          Annotation != "J" && Annotation != "E")
        return true;

      // A call to clad::differentiate or clad::gradient was found.
      if (DeclRefExpr* DRE = getArgFunction(E, m_Sema))
        request.Function = cast<FunctionDecl>(DRE->getDecl());
      else
        return true;

      if (ProcessInvocationArgs(m_Sema, endLoc, m_Options, FD, request))
        return true;

      request.VerboseDiags = true;
      // The root of the differentiation request graph should update the
      // CladFunction object with the generated call.
      request.CallUpdateRequired = true;

      request.Args = E->getArg(1);
      m_TopMostReq = &request;
    } else {
      // Don't build propagators for calls that do not contribute in
      // differentiable way to the result.
      if (!isa<CXXMemberCallExpr>(E) && !isa<CXXOperatorCallExpr>(E) &&
          allArgumentsAreLiterals(E))
        return true;

      request.Function = FD;
      if (m_TopMostReq->Mode == DiffMode::forward)
        request.Mode = DiffMode::experimental_pushforward;
      else if (m_TopMostReq->Mode == DiffMode::reverse)
        request.Mode = DiffMode::experimental_pullback;
      else {
        // propagatorReq.Mode = request.Mode;
      }
      request.VerboseDiags = false;
      request.EnableTBRAnalysis = m_TopMostReq->EnableTBRAnalysis;
      request.EnableVariedAnalysis = m_TopMostReq->EnableVariedAnalysis;
      // propagatorReq.CUDAGlobalArgsIndexes = globalCallArgs;

      // const auto* MD = dyn_cast<CXXMethodDecl>(FD);
      for (size_t i = 0, e = FD->getNumParams(); i < e; ++i) {
        // if (MD && isLambdaCallOperator(MD)) {
        if (const auto* paramDecl = FD->getParamDecl(i))
          request.DVI.push_back(paramDecl);
        //}
        // FIXME:
        // else if (DerivedCallOutputArgs[i + (bool)MD]) {
        //  propagatorReq.DVI.push_back(FD->getParamDecl(i));
        //}
      }
    }

    if (isCallOperator(m_Sema.getASTContext(), request.Function))
      request.Functor = cast<CXXMethodDecl>(request.Function)->getParent();
    request.CallContext = E;
    request.BaseFunctionName = utils::ComputeEffectiveFnName(request.Function);

    // Recurse into call graph.
    TraverseFunctionDeclOnce(request.Function);
    if (!HasCustomDerivativeForDecl(m_Sema, request))
      m_DiffRequestGraph.addNode(request, /*isSource=*/true);

    if (m_IsTraversingTopLevelDecl)
      m_TopMostReq = nullptr;

    return true;
  }
} // end namespace
