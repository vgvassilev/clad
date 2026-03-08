#include "clad/Differentiator/DiffPlanner.h"

#include "clad/Differentiator/DiffMode.h"

#include "ActivityAnalyzer.h"
#include "TBRAnalyzer.h"
#include "UsefulAnalyzer.h"

#include "clad/Differentiator/CladConfig.h"
#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"
#include "clad/Differentiator/DerivativeBuilder.h"
#include "clad/Differentiator/DerivedFnCollector.h"
#include "clad/Differentiator/Timers.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTLambda.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LLVM.h" // isa, dyn_cast
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/PartialDiagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/Version.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Sema/TemplateDeduction.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <utility>

using namespace clang;

namespace clad {
/// Updates \c DR with the target function to differentiate.
///
/// \param[in,out] DR - The DiffRequest being updated.
/// \param[in] call A clad differentiation function call expression
/// \param[in] SemaRef Reference to Sema
static bool findTargetFunction(DiffRequest& DR, CallExpr* call, Sema& SemaRef) {
  struct Finder : RecursiveASTVisitor<Finder> {
    Sema& m_SemaRef;
    DiffRequest& m_DR;
    SourceLocation m_BeginLoc;
    Finder(Sema& SemaRef, DiffRequest& DR, SourceLocation beginLoc)
        : m_SemaRef(SemaRef), m_DR(DR), m_BeginLoc(beginLoc) {}

    // Required for visiting lambda declarations.
    bool shouldVisitImplicitCode() const { return true; }
    bool VisitExplicitCastExpr(ExplicitCastExpr* CastE) {
      if (CastE->getCastKind() != CK_ReinterpretMemberPointer)
        return true;

      // Handle the cases where the user has forced overload resolution via an
      // explicit cast: clad::differentiate(static_cast<...>(f))
      assert(!m_DR.Function && "Function already set!");
      TraverseStmt(CastE->getSubExpr());

      // We will need to update the selected function.
      if (const auto* MPTy = CastE->getType()->getAs<MemberPointerType>()) {
        CXXRecordDecl* SelectedCXXRD = MPTy->getMostRecentCXXRecordDecl();
        ASTContext& C = m_SemaRef.getASTContext();
        QualType MPPointeeTy = MPTy->getPointeeType();
        const FunctionDecl* FD = m_DR.Function;

        if (C.hasSameType(FD->getType(), MPPointeeTy))
          return false; // Type match we are done.

        // We need to find the right function that we should differentiate.
        auto BestMatch = [&C, &FD, &MPPointeeTy](const CXXRecordDecl* RD) {
          for (NamedDecl* ND : RD->lookup(FD->getDeclName())) {
            if (FD == ND)
              continue;
            auto* NewFD = cast<FunctionDecl>(ND);
            // FIXME: Call Sema::IsOverload to ensure both decls are compatible.
            if (C.hasSameType(NewFD->getType(), MPPointeeTy)) {
              FD = NewFD;
              return false; // exit
            }
          }
          return true;
        };

        if (BestMatch(SelectedCXXRD)) {
          CXXBasePaths Paths(/*FindAmbiguities=*/false, /*RecordPaths=*/false,
                             /* DetectVirtual=*/false);
          auto BestMatchInBase = [&BestMatch](const CXXBaseSpecifier* Specifier,
                                              CXXBasePath& Path) {
            const auto* Base = Specifier->getType()->getAsCXXRecordDecl();
            return !BestMatch(Base);
          };
          SelectedCXXRD->lookupInBases(BestMatchInBase, Paths);
        }

        assert(m_DR.Function != FD && "Could not find the overload");
        m_DR.Function = FD;
      }

      return false; // we traversed the subexpression already, stop.
    }

    bool VisitDeclRefExpr(DeclRefExpr* DRE) {
      if (auto VD = dyn_cast<VarDecl>(DRE->getDecl())) {
        QualType VarTy = VD->getType();
        // If variable is of class type, set the result to the overloaded call
        // operator method of the class type.
        if (auto* CXXRD = VarTy->getAsCXXRecordDecl())
          TraverseDecl(CXXRD);
        else
          TraverseStmt(VD->getInit());
      } else if (auto* FD = dyn_cast<FunctionDecl>(DRE->getDecl()))
        m_DR.Function = FD;
      return false;
    }

    bool VisitCXXRecordDecl(CXXRecordDecl* RD) {
      auto callOperatorDeclName =
          m_SemaRef.getASTContext().DeclarationNames.getCXXOperatorName(
              OverloadedOperatorKind::OO_Call);
      LookupResult R(m_SemaRef, callOperatorDeclName, noLoc,
                     Sema::LookupNameKind::LookupMemberName);
      // We do not want diagnostics that would fire because of this lookup.
      R.suppressDiagnostics();
      m_SemaRef.LookupQualifiedName(R, RD);

      // Emit error diagnostics
      if (R.empty()) {
        utils::diag(m_SemaRef, DiagnosticsEngine::Error, m_BeginLoc,
                    "%0 has no defined operator()")
            << RD << m_BeginLoc;
        return false;
      }

      if (!R.isSingleResult()) {
        utils::diag(m_SemaRef, DiagnosticsEngine::Error, m_BeginLoc,
                    "%0 has multiple definitions of operator(); "
                    "multiple definitions of call operators are not supported")
            << RD << m_BeginLoc;

        // Emit diagnostics for candidate functions
        for (auto oper = R.begin(), operEnd = R.end(); oper != operEnd;
             ++oper) {
          auto candidateFn = cast<CXXMethodDecl>(oper.getDecl());
          m_SemaRef.NoteOverloadCandidate(candidateFn,
                                          cast<FunctionDecl>(candidateFn));
        }
        return false;
      }

      NamedDecl* FoundDecl = R.getFoundDecl();
      AccessSpecifier FoundDeclAccess = FoundDecl->getAccess();
      if (FoundDeclAccess != AccessSpecifier::AS_public) {
        // Compute access specifier name so that it can be used in
        // diagnostic message.
        const char* callOperatorAS =
            (FoundDeclAccess == AccessSpecifier::AS_private ? "private"
                                                            : "protected");
        utils::diag(m_SemaRef, DiagnosticsEngine::Error, m_BeginLoc,
                    "%0 contains %1 call operator; differentiation of "
                    "non-public call operators is not supported")
            << RD << callOperatorAS << m_BeginLoc;

        // Compute if the access specifier of the found operator is implicit.
        bool isImplicit = true;
        for (auto decl : RD->decls()) {
          if (decl == FoundDecl)
            break;
          if (isa<AccessSpecDecl>(decl)) {
            isImplicit = false;
            break;
          }
        }
        // Emit diagnostics for the found call operator
        m_SemaRef.Diag(FoundDecl->getBeginLoc(), diag::note_access_natural)
            << (unsigned)(FoundDeclAccess == AccessSpecifier::AS_protected)
            << isImplicit;

        return false;
      }

      auto* CallOperator = cast<CXXMethodDecl>(FoundDecl);
      m_DR.Function = CallOperator;
      m_DR.Functor = CallOperator->getParent();
      // Mark the declaration as used to instantiate templates, etc.
      bool OdrUse = !CallOperator->isVirtual() ||
                    CallOperator->getDevirtualizedMethod(
                        /*Base=*/nullptr, m_SemaRef.getLangOpts().AppleKext);
      m_SemaRef.MarkAnyDeclReferenced(m_BeginLoc, CallOperator, OdrUse);
      return false;
    }
  } finder(SemaRef, DR, call->getArg(0)->getBeginLoc());
  finder.TraverseStmt(call->getArg(0));

  assert(cast<NamespaceDecl>(call->getDirectCallee()->getDeclContext())
                 ->getName() == "clad" &&
         "Should be called for clad:: special functions!");
  return finder.m_DR.Function;
}
static QualType GetDerivedFunctionType(const CallExpr* CE) {
  const auto* CXXRD = CE->getType()->getAsCXXRecordDecl();
  const auto* Spec = cast<ClassTemplateSpecializationDecl>(CXXRD);
  assert(Spec && "Called with the wrong expression!");
  const TemplateArgument& TemplArg = Spec->getTemplateArgs().get(/*Idx=*/0);
  return TemplArg.getAsType();
}

  void DiffRequest::updateCall(FunctionDecl* FD, FunctionDecl* OverloadedFD,
                               Sema& SemaRef) {
    assert(isa<CallExpr>(this->CallContext) &&
           "Trying to update an unsupported expression");
    auto* call = cast<CallExpr>(this->CallContext);

    assert(FD && "Trying to update with null FunctionDecl");

    ASTContext& C = SemaRef.getASTContext();

    FunctionDecl* replacementFD = OverloadedFD ? OverloadedFD : FD;

    clad_compat::llvm_Optional<unsigned> codeArgIdx;
    clad_compat::llvm_Optional<unsigned> derivedFnArgIdx;
    for (unsigned i = 0, e = call->getNumArgs(); i < e; ++i) {
      if (const auto* ArgExpr = dyn_cast<CXXDefaultArgExpr>(call->getArg(i))) {
        std::string argName = ArgExpr->getParam()->getNameAsString();
        if (argName == "derivedFn")
          derivedFnArgIdx = i;
        else if (argName == "code")
          codeArgIdx = i;
      }
    }

    if (!derivedFnArgIdx)
      return;

    // Index of "CUDAkernel" parameter:
    if (call->getNumArgs() > 4) {
      auto kernelArgIdx = call->getNumArgs() - 1;
      auto* cudaKernelFlag =
          SemaRef
              .ActOnCXXBoolLiteral(noLoc,
                                   replacementFD->hasAttr<CUDAGlobalAttr>()
                                       ? tok::kw_true
                                       : tok::kw_false)
              .get();
      call->setArg(kernelArgIdx, cudaKernelFlag);
    }

    ExprValueKind VK = VK_LValue;
    // We have a DeclRefExpr pointing to a member declaration, which is an
    // lvalue. However, due to an inconsistency of the expression classfication
    // in clang we need to change it to an r-value to avoid an assertion when
    // building a unary op. See llvm/llvm-project#53958.
    if (const auto* MD = dyn_cast<CXXMethodDecl>(replacementFD))
      if (MD->isInstance())
        VK = CLAD_COMPAT_ExprValueKind_R_or_PR_Value;
    CXXScopeSpec CSS;
    utils::BuildNNS(SemaRef, replacementFD->getDeclContext(), CSS,
                    /*addGlobalNS=*/true);
    Expr* Arg = SemaRef.BuildDeclRefExpr(
        replacementFD, replacementFD->getType(), VK, noLoc, &CSS);
    // Add the "&" operator
    Arg = SemaRef
              .BuildUnaryOp(/*Scope=*/nullptr, noLoc,
                            UnaryOperatorKind::UO_AddrOf, Arg)
              .get();
    // Take into account if the user selected an overload by a cast expr.
    if (isa<ExplicitCastExpr>(call->getArg(0))) {
      QualType Ty = GetDerivedFunctionType(call);
      TypeSourceInfo* TSI = C.getTrivialTypeSourceInfo(Ty, noLoc);
      Arg = SemaRef.BuildCStyleCastExpr(noLoc, TSI, noLoc, Arg).get();
    }
    call->setArg(*derivedFnArgIdx, Arg);

    if (ImmediateMode) {
      assert(!codeArgIdx && "We found the index of the code argument!");
      return;
    }
    // Update the code parameter if it was found.
    clang::LangOptions LangOpts;
    LangOpts.CPlusPlus = true;
    clang::PrintingPolicy Policy(LangOpts);
    Policy.Bool = true;

    std::string s;
    llvm::raw_string_ostream Out(s);
    FD->print(Out, Policy);
    Out.flush();

    StringLiteral* SL = utils::CreateStringLiteral(C, Out.str());
    QualType CodeArgTy = call->getArg(*codeArgIdx)->getType();
    Expr* CodeArg =
        SemaRef.ImpCastExprToType(SL, CodeArgTy, CK_ArrayToPointerDecay).get();
    call->setArg(*codeArgIdx, CodeArg);
  }

  DiffCollector::DiffCollector(DeclGroupRef DGR, DiffInterval& Interval,
                               clad::DynamicGraph<DiffRequest>& requestGraph,
                               clang::Sema& S, RequestOptions& opts,
                               OwnedAnalysisContexts& AllAnalysisDC)
      : m_Interval(Interval), m_DiffRequestGraph(requestGraph),
        m_AllAnalysisDC(AllAnalysisDC), m_Sema(S), m_Options(opts) {

    if (Interval.empty())
      return;

    assert(!m_TopMostReq && "Traversal already in flight!");

    for (Decl* D : DGR)
      TraverseDecl(D);
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
    if (Mode == DiffMode::pullback)
      return;

    DVI.clear();
    auto& C = semaRef.getASTContext();
    const Expr* diffArgs = Args;
    const FunctionDecl* FD = Function;
    // On this stage, templated functions are not yet fully instantiated.
    // Uninstantiated functions lack much information like we need: they don't
    // have bodies and information about defition. Therefore, we have to force
    // the instantiation.
    semaRef.PerformPendingInstantiations();
    if (!DeclarationOnly)
      FD = FD->getDefinition();
    if (!diffArgs || !FD) {
      return;
    }
    DiffParams params{};
    auto E = diffArgs->IgnoreParenImpCasts();
    // Case 1)
    SourceLocation dArgsL = diffArgs->getBeginLoc();
    if (auto SL = dyn_cast<StringLiteral>(E)) {
      IndexIntervalTable indexes{};
      llvm::StringRef string = SL->getString().trim();
      if (string.empty()) {
        utils::diag(semaRef, DiagnosticsEngine::Error, dArgsL,
                    "no parameters were provided")
            << dArgsL;
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
            utils::diag(semaRef, DiagnosticsEngine::Error, dArgsL,
                        "invalid argument index %0 of %1 argument(s)")
                << idx << FD->getNumParams() << dArgsL;
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
          utils::diag(semaRef, DiagnosticsEngine::Error, dArgsL,
                      "requested parameter name '%0' was not found among "
                      "function parameters")
              << pName << dArgsL;
          return;
        }

        auto f_it = std::find_if(std::begin(DVI), std::end(DVI),
                                 [&it](const DiffInputVarInfo& dVarInfo) {
                                   return dVarInfo.param == it->second;
                                 });

        if (f_it != DVI.end()) {
          utils::diag(semaRef, DiagnosticsEngine::Error, dArgsL,
                      "requested parameter %0 was specified multiple times")
              << it->second << dArgsL;
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
              utils::diag(semaRef, DiagnosticsEngine::Error, dArgsL,
                          "could not parse index '%0'")
                  << diffSpec << dArgsL;
              return;
            }
            dVarInfo.paramIndexInterval = IndexInterval(index);
          } else {
            size_t first, last;
            if (firstStr.getAsInteger(Radix, first) ||
                lastStr.getAsInteger(Radix, last)) {
              utils::diag(semaRef, DiagnosticsEngine::Error, dArgsL,
                          "could not parse range '%0'")
                  << diffSpec << dArgsL;
              return;
            }
            if (first >= last) {
              utils::diag(semaRef, DiagnosticsEngine::Error, dArgsL,
                          "range specified in '%0' is in incorrect format")
                  << diffSpec << dArgsL;
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
          utils::diag(semaRef, DiagnosticsEngine::Error, dArgsL,
                      "fields can only be provided for class type parameters; "
                      "field information is incorrectly specified in '%0' "
                      "for non-class type parameter '%1'")
              << diffSpec << pName << dArgsL;
          return;
        }

        if (!dVarInfo.fields.empty()) {
          RecordDecl* RD = dVarInfo.param->getType()->getAsCXXRecordDecl();
          llvm::SmallVector<llvm::StringRef, 4> ref(dVarInfo.fields.begin(),
                                                    dVarInfo.fields.end());
          bool isValid = utils::IsValidMemExprPath(semaRef, RD, ref);
          if (!isValid) {
            utils::diag(semaRef, DiagnosticsEngine::Error, dArgsL,
                        "path specified by fields in '%0' is invalid")
                << diffSpec << dArgsL;
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
          utils::diag(semaRef, DiagnosticsEngine::Error, dArgsL,
                      "invalid member variable index %0 of %1 member "
                      "variable(s)")
              << std::to_string(idx) << totalFields << dArgsL;
          return;
        }
        dVarInfo.param = *std::next(Functor->field_begin(), idx);
      } else {
        // Fail if the specified index is invalid.
        if ((idx < 0) || (idx >= FD->getNumParams())) {
          utils::diag(semaRef, DiagnosticsEngine::Error, dArgsL,
                      "invalid argument index %0 of %1 argument(s)")
              << std::to_string(idx) << FD->getNumParams() << dArgsL;
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
        SourceLocation L = CallContext->getBeginLoc();
        utils::diag(semaRef, DiagnosticsEngine::Error, L,
                    "attempted to differentiate function with no parameters")
            << L;
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
    utils::diag(semaRef, DiagnosticsEngine::Error, dArgsL,
                "failed to parse the parameters, must be string or "
                "numeric literal");
    return;
  }

  void DiffRequest::print(llvm::raw_ostream& Out) const {
    const NamedDecl* ND = nullptr;
    if (Function)
      ND = Function;
    else
      ND = Global;
    if (!ND) {
      Out << "<- INVALID ->";
      return;
    }
    Out << '<';
    PrintingPolicy P(ND->getASTContext().getLangOpts());
    P.TerseOutput = true;
    P.FullyQualifiedName = true;
    ND->print(Out, P, /*Indentation=*/0, /*PrintInstantiation=*/true);
    Out << ">[name=" << BaseFunctionName << ", "
        << "order=" << CurrentDerivativeOrder << ", "
        << "mode=" << DiffModeToString(Mode) << ", "
        << "args='";
    if (Args)
      Args->printPretty(Out, /*Helper=*/nullptr, P);
    else
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

  bool DiffRequest::shouldBeRecorded(const Stmt* S) const {
    if (!EnableTBRAnalysis)
      return true;

    if (!m_TbrRunInfo.HasAnalysisRun && !isLambdaCallOperator(Function) &&
        Function->isDefined() && m_AnalysisDC) {
      TimedAnalysisRegion R("TBR " + BaseFunctionName);
      TBRAnalyzer analyzer(m_AnalysisDC, getToBeRecorded(),
                           &getModifiedParams(), &getUsedParams());
      analyzer.Analyze(*this);
    }
    auto found = m_TbrRunInfo.ToBeRecorded.find(S);
    return found != m_TbrRunInfo.ToBeRecorded.end();
  }

  bool DiffRequest::shouldHaveAdjointForw(const VarDecl* VD) const {
    if (!EnableUsefulAnalysis)
      return true;
    auto found = m_UsefulRunInfo.UsefulDecls.find(VD);
    return found != m_UsefulRunInfo.UsefulDecls.end();
  }

  bool DiffRequest::shouldHaveAdjoint(const Stmt* S) const {
    if (!EnableVariedAnalysis)
      return true;
    auto found = m_ActivityRunInfo.VariedS.find(S);
    return found != m_ActivityRunInfo.VariedS.end();
  }
  bool DiffRequest::shouldHaveAdjoint(const VarDecl* VD) const {
    if (!EnableVariedAnalysis)
      return true;
    return getVariedDecls().find(VD) != getVariedDecls().end();
  }
  bool DiffRequest::isVaried(const Expr* E) const {
    // FIXME: We should consider removing pullback requests from the
    // diff graph.
    class VariedChecker : public RecursiveASTVisitor<VariedChecker> {
      const DiffRequest& m_Request;

    public:
      VariedChecker(const DiffRequest& DR) : m_Request(DR) {}
      bool isVariedE(const clang::Expr* E) {
        auto j = m_Request.getVariedStmt().find(E);
        if (j != m_Request.getVariedStmt().end())
          return true;
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        return !TraverseStmt(const_cast<clang::Expr*>(E));
      }
      bool VisitDeclRefExpr(const clang::DeclRefExpr* DRE) {
        if (!isa<VarDecl>(DRE->getDecl()))
          return true;
        if (m_Request.shouldHaveAdjoint(dyn_cast<VarDecl>(DRE->getDecl())))
          return false;
        return true;
      }
      // The sub-stmt of OpaqueValueExpr is not visited automatically
      bool VisitOpaqueValueExpr(const clang::OpaqueValueExpr* OVE) {
        return TraverseStmt(OVE->getSourceExpr());
      }
      // FIXME: This is a temporary measure until we add support for
      // `this` in varied analysis.
      bool VisitCXXThisExpr(const clang::CXXThisExpr* TE) { return false; }
    } analyzer(*this);
    return analyzer.isVariedE(E);
  }

  std::string DiffRequest::ComputeDerivativeName() const {
    if (Mode != DiffMode::forward && Mode != DiffMode::reverse &&
        Mode != DiffMode::vector_forward_mode) {
      std::string name = BaseFunctionName + "_" + DiffModeToString(Mode);
      for (auto index : CUDAGlobalArgsIndexes)
        name += "_" + std::to_string(index);

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
  static bool ProcessInvocationArgs(Sema& S, SourceLocation BeginLoc,
                                    const RequestOptions& ReqOpts,
                                    const FunctionDecl* FD,
                                    DiffRequest& request) {
    const AnnotateAttr* A = FD->getAttr<AnnotateAttr>();
    std::string Annotation = A->getAnnotation().str();
    if (Annotation == "E") {
      // Error estimation has no options yet.
      request.Mode = DiffMode::reverse;
      request.EnableErrorEstimation = true;
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
    else
      llvm_unreachable("unknown mode");
    if (request.Mode == DiffMode::reverse || request.Mode == DiffMode::hessian)
      request.EnableTBRAnalysis = ReqOpts.EnableTBRAnalysis;
    request.EnableVariedAnalysis = ReqOpts.EnableVariedAnalysis;
    request.EnableUsefulAnalysis = ReqOpts.EnableUsefulAnalysis;

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
    bool enable_ua_in_req =
        clad::HasOption(bitmasked_opts_value, clad::opts::enable_ua);
    bool disable_ua_in_req =
        clad::HasOption(bitmasked_opts_value, clad::opts::disable_ua);
    // Sanity checks.
    if (enable_tbr_in_req && disable_tbr_in_req) {
      utils::diag(S, DiagnosticsEngine::Error, BeginLoc,
                  "both enable and disable TBR options are specified")
          << BeginLoc;
      return true;
    }
    if (enable_va_in_req && disable_va_in_req) {
      utils::diag(S, DiagnosticsEngine::Error, BeginLoc,
                  "both enable and disable VA options are specified")
          << BeginLoc;
      return true;
    }
    if (enable_ua_in_req && disable_ua_in_req) {
      utils::diag(S, DiagnosticsEngine::Error, BeginLoc,
                  "both enable and disable UA options are specified")
          << BeginLoc;
      return true;
    }
    if (enable_tbr_in_req && request.Mode == DiffMode::forward) {
      utils::diag(S, DiagnosticsEngine::Error, BeginLoc,
                  "tbr analysis is not meant for forward mode AD")
          << BeginLoc;
      return true;
    }

    // reverse vector mode is not yet supported.
    if (request.Mode == DiffMode::reverse &&
        clad::HasOption(bitmasked_opts_value, clad::opts::vector_mode)) {
      utils::diag(S, DiagnosticsEngine::Error, BeginLoc,
                  "reverse vector mode is not yet supported")
          << BeginLoc;
      return true;
    }

    // Override the default value of TBR analysis.
    if (enable_tbr_in_req || disable_tbr_in_req)
      request.EnableTBRAnalysis = enable_tbr_in_req && !disable_tbr_in_req;

    // Override the default value of VA analysis.
    if (enable_va_in_req || disable_va_in_req)
      request.EnableVariedAnalysis = enable_va_in_req && !disable_va_in_req;

    // Override the default value of UA analysis.
    if (enable_ua_in_req || disable_ua_in_req)
      request.EnableUsefulAnalysis = enable_ua_in_req && !disable_ua_in_req;

    // Check for clad::hessian<diagonal_only>.
    if (clad::HasOption(bitmasked_opts_value, clad::opts::diagonal_only)) {
      if (request.Mode == DiffMode::hessian) {
        request.Mode = DiffMode::hessian_diagonal;
        return false;
      }
      utils::diag(S, DiagnosticsEngine::Error, BeginLoc,
                  "diagonal only option is only valid for hessian mode");
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
          utils::diag(S, DiagnosticsEngine::Error, BeginLoc,
                      "only first order derivative is supported for now "
                      "in vector forward mode")
              << BeginLoc;
          return true;
        }

        // We don't yet support enzyme with vector mode.
        if (request.use_enzyme) {
          utils::diag(S, DiagnosticsEngine::Error, BeginLoc,
                      "enzyme's vector mode is not yet supported")
              << BeginLoc;
          return true;
        }
      }
    }

    return false;
  }

  static bool allArgumentsAreLiterals(const CallExpr::arg_range& args,
                                      const DiffRequest* request) {
    return std::none_of(args.begin(), args.end(), [&request](const Expr* A) {
      return request->isVaried(A);
    });
  }

  static Expr* getOverloadExpr(Sema& S, DeclContext* DC, DiffRequest& R) {
    // Error estimation only uses forward mode derivatives if they are
    // user-prodived to handle builtin derivatives. If found, we have to change
    // the mode of the request.
    if (R.EnableErrorEstimation && R.Mode == DiffMode::pullback &&
        utils::canUsePushforwardInRevMode(R.Function)) {
      R.Mode = DiffMode::pushforward;
      R.EnableErrorEstimation = false;
      if (Expr* overload = getOverloadExpr(S, DC, R)) {
        R.DVI.clear();
        return overload;
      }
      R.Mode = DiffMode::pullback;
      R.EnableErrorEstimation = true;
    }
    llvm::SmallVector<const ValueDecl*, 4> diffParams{};
    for (const DiffInputVarInfo& VarInfo : R.DVI)
      diffParams.push_back(VarInfo.param);
    QualType dTy = utils::GetDerivativeType(S, R.Function, R.Mode, diffParams,
                                            /*forCustomDerv=*/true,
                                            /*shouldUseRestoreTracker=*/false);
    // We disable diagnostics for methods and operators because they often have
    // ideantical names: `constructor_pullback`, `operator_star_pushforward`,
    // etc. If we turn it on, every such operator will trigger diagnostics
    // because of our STL and Kokkos custom derivatives.
    // FIXME: Add a way to silence the diagnostics.
    bool enableDiagnostics =
        !isa<CXXMethodDecl>(R.Function) && !R->isOverloadedOperator() &&
        R.BaseFunctionName != "forward" && !R.EnableErrorEstimation;

    ASTContext& C = S.getASTContext();
    auto LookupPropagator = [&C, &S, &DC](const std::string& Name) {
      IdentifierInfo* II = &C.Idents.get(Name);
      DeclarationNameInfo DNInfo(II, utils::GetValidSLoc(S));
      LookupResult Found(S, DNInfo, Sema::LookupOrdinaryName);
      S.LookupQualifiedName(Found, DC);
      return Found;
    };

    std::string Name = R.ComputeDerivativeName();
    LookupResult Found = LookupPropagator(Name);
    // This is a hack to reuse the builtin derivatives for vector mode.
    if (Found.empty() && R.Mode == DiffMode::vector_pushforward)
      Found = LookupPropagator(R.BaseFunctionName + "_pushforward");

    if (Found.empty())
      return nullptr; // Nothing found.

    TemplateSpecCandidateSet FailedCandidates(R.CallContext->getBeginLoc(),
                                              /*ForTakingAddress=*/false);
    if (Expr* overload =
            utils::MatchOverloadType(S, dTy, Found, FailedCandidates))
      return overload;

    if (!enableDiagnostics)
      return nullptr;

    // We did not match the found candidates. Warn and offer the user hints.
    auto errId = S.Diags.getCustomDiagID(
        DiagnosticsEngine::Error,
        "user-defined derivative for %0 was provided but not used; "
        "expected signature %1 does not match");
    S.Diag(R.CallContext->getBeginLoc(), errId) << R.Function << dTy;
    FailedCandidates.NoteCandidates(S, R.CallContext->getBeginLoc());
    utils::DiagnoseSignatureMismatch(S, dTy, Found);

    return nullptr;
  }

  bool DiffCollector::LookupCustomDerivativeDecl(DiffRequest& request) {
    NamespaceDecl* cladNS =
        utils::LookupNSD(m_Sema, "clad", /*shouldExist=*/true);
    NamespaceDecl* customDerNS = utils::LookupNSD(
        m_Sema, "custom_derivatives", /*shouldExist=*/false, cladNS);
    if (!customDerNS)
      return false;
    if (request.Mode == DiffMode::unknown)
      return true;

    const Expr* callSite = request.CallContext;
    assert(callSite && "Called lookup without CallContext");

    const Decl* fnDecl = nullptr;
    // Check if the callSite is not associated with a shadow declaration.
    if (request.Mode == DiffMode::pushforward ||
        request.Mode == DiffMode::pullback ||
        request.Mode == DiffMode::vector_pushforward) {
      if (const auto* ME = dyn_cast<CXXMemberCallExpr>(callSite)) {
        fnDecl = ME->getMethodDecl();
      } else if (const auto* CE = dyn_cast<CallExpr>(callSite)) {
        const Expr* Callee = CE->getCallee()->IgnoreParenCasts();
        if (const auto* DRE = dyn_cast<DeclRefExpr>(Callee))
          fnDecl = DRE->getFoundDecl();
        else if (const auto* MemberE = dyn_cast<MemberExpr>(Callee))
          fnDecl = MemberE->getFoundDecl().getDecl();
      } else if (const auto* CtorExpr = dyn_cast<CXXConstructExpr>(callSite)) {
        fnDecl = CtorExpr->getConstructor();
      }
    } else
      fnDecl = request.Function;
    DeclContext* DC = customDerNS;

    if (isa<CXXMethodDecl>(fnDecl))
      DC = utils::LookupNSD(m_Sema, "class_functions", /*shouldExist=*/false,
                            DC);
    else
      DC = utils::FindDeclContext(m_Sema, DC, fnDecl->getDeclContext());

    if (!DC)
      return false;

    assert(request.Mode != DiffMode::unknown &&
           "Called lookup without specified DiffMode");

    if (Expr* overload = getOverloadExpr(m_Sema, DC, request)) {
      // Overload found. Mark the request as custom derivative and save
      // the set of overloads to process later.
      request.CustomDerivative = overload;
      return true;
    }

    return false;
  }

  bool DiffCollector::VisitCallExpr(CallExpr* E) {
    // Check if we should look into this.
    DiffRequest request;

    FunctionDecl* FD = E->getDirectCallee();
    if (!FD)
      return true;

    bool nonDiff = false;
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

      // A call to clad::differentiate or clad::gradient was not found.
      if (!findTargetFunction(request, E, m_Sema))
        return true;

      request.VerboseDiags = true;
      // The root of the differentiation request graph should update the
      // CladFunction object with the generated call.
      request.CallUpdateRequired = true;
      request.CallContext = E;

      if (ProcessInvocationArgs(m_Sema, endLoc, m_Options, FD, request))
        return true;

      request.Args = E->getArg(1);
      request.UpdateDiffParamsInfo(m_Sema);
      if (request.Mode == DiffMode::reverse && request.EnableVariedAnalysis) {
        if (request.Args)
          for (const auto& dParam : request.DVI)
            request.addVariedDecl(cast<VarDecl>(dParam.param));
      }

      if (request.Function->hasAttr<CUDAGlobalAttr>())
        for (size_t i = 0, e = request.Function->getNumParams(); i < e; ++i)
          request.CUDAGlobalArgsIndexes.push_back(i);
      m_TopMostReq = &request;
    } else {
      // If the function contains annotation of non_differentiable, then Clad
      // should not produce any derivative expression for that function call,
      // and the function call in the primal should be used as it is.
      if (clad::utils::hasNonDifferentiableAttribute(E))
        nonDiff = true;

      request.VerboseDiags = false;
      request.EnableTBRAnalysis = m_TopMostReq->EnableTBRAnalysis;
      request.EnableVariedAnalysis = m_TopMostReq->EnableVariedAnalysis;
      request.EnableUsefulAnalysis = m_TopMostReq->EnableUsefulAnalysis;
      request.EnableErrorEstimation = m_TopMostReq->EnableErrorEstimation;
      request.CallContext = E;

      const auto* MD = dyn_cast<CXXMethodDecl>(FD);
      if (MD) {
        if (isLambdaCallOperator(MD) &&
            m_TopMostReq->Mode == DiffMode::reverse) {
          request.EnableVariedAnalysis = false;
          return true;
        }
        const CXXRecordDecl* CD = MD->getParent();
        if (clad::utils::hasNonDifferentiableAttribute(CD))
          nonDiff = true;
      }

      QualType returnType = FD->getReturnType();
      bool hasPointerOrRefReturn = utils::isNonConstReferenceType(returnType) ||
                                   returnType->isPointerType();
      // Don't build propagators for calls that do not contribute in
      // differentiable way to the result.
      if (!(MD && MD->isInstance()) && !hasPointerOrRefReturn &&
          allArgumentsAreLiterals(E->arguments(), m_ParentReq))
        nonDiff = true;
      // In the reverse mode, such functions don't have dfdx()
      if (!utils::hasMemoryTypeParams(FD) && hasPointerOrRefReturn &&
          m_TopMostReq->Mode == DiffMode::reverse)
        nonDiff = true;

      if (nonDiff && m_TopMostReq->Mode != DiffMode::reverse)
        return true;

      request.Function = FD;
      request.CallContext = E;
      bool canUsePushforwardInRevMode =
          m_TopMostReq->Mode == DiffMode::reverse &&
          !request.EnableErrorEstimation &&
          utils::canUsePushforwardInRevMode(FD);

      std::string FDName = FD->getNameAsString();
#if CLANG_VERSION_MAJOR < 16
      if (clang::AnalysisDeclContext::isInStdNamespace(FD) &&
          (FDName == "move" || FDName == "forward"))
        return true;
#endif
      // FIXME: hessians require second derivatives, i.e. apart from the
      // pushforward, we also need to schedule pushforward_pullback.
      if (m_ParentReq->CustomDerivative ||
          m_ParentReq->Mode == DiffMode::unknown)
        request.Mode = DiffMode::unknown;
      else if (m_TopMostReq->Mode == DiffMode::forward ||
               m_TopMostReq->Mode == DiffMode::hessian ||
               canUsePushforwardInRevMode)
        request.Mode = DiffMode::pushforward;
      else if (m_TopMostReq->Mode == DiffMode::reverse)
        request.Mode = DiffMode::pullback;
      else if (m_TopMostReq->Mode == DiffMode::vector_forward_mode ||
               m_TopMostReq->Mode == DiffMode::jacobian ||
               m_TopMostReq->Mode == DiffMode::vector_pushforward) {
        request.Mode = DiffMode::vector_pushforward;
      } else {
        assert(0 && "unexpected mode.");
        return true;
      }

      // FIXME: Use elidable_reverse_forw
      if (request.Mode == DiffMode::pullback &&
          (FDName == "cudaMemcpy" || FDName == "begin" || FDName == "end"))
        return true;

      if (request.Mode != DiffMode::pushforward &&
          request.Mode != DiffMode::vector_pushforward) {
        // CUDA device function call in global kernel gradient
        bool useCUDA = !m_TopMostReq->CUDAGlobalArgsIndexes.empty();
        for (size_t i = 0, e = FD->getNumParams(); i < e; ++i) {
          const auto* paramDecl = FD->getParamDecl(i);
          if (clad::utils::hasNonDifferentiableAttribute(paramDecl))
            continue;

          // FIXME: Attach Activity Analysis here.
          // If a parameter is non-differentiable, don't include
          // it in the DVI to avoid dynamic partial derivatives.
          const ParmVarDecl* PVD = nullptr;
          Expr* ArgE = E->getArg(i)->IgnoreParens()->IgnoreParenCasts();
          if (const auto* DRE = dyn_cast<DeclRefExpr>(ArgE))
            PVD = dyn_cast<ParmVarDecl>(DRE->getDecl());
          bool IsDifferentiableArg =
              !PVD || m_ParentReq->HasIndependentParameter(PVD);

          if (!utils::isArrayOrPointerType(paramDecl->getType()) ||
              IsDifferentiableArg)
            request.DVI.push_back(paramDecl);
          // FIXME: If we cannot deduce whether the argument is
          // differentiable, we should still add it to CUDAGlobalArgsIndexes.
          // i.e. remove `&& PVD`.
          // We know we should use atomic ops here
          if (useCUDA && PVD && IsDifferentiableArg)
            request.CUDAGlobalArgsIndexes.push_back(i);
        }
      }

      // Warn if we find pullbacks.
      if (canUsePushforwardInRevMode &&
          m_TopMostReq->Mode == DiffMode::reverse) {
        DiffRequest R = request;
        R.BaseFunctionName = utils::ComputeEffectiveFnName(R.Function);
        R.Mode = DiffMode::pullback;
        if (LookupCustomDerivativeDecl(R)) {
          utils::diag(
              m_Sema, DiagnosticsEngine::Warning, R.CallContext->getBeginLoc(),
              "unused function '%0_pullback'; '%0' is a real-domain, "
              "single-argument function and only pushforward is required")
              << R.Function->getName() << R.CallContext->getSourceRange();
          // Collect the unused decls.
          llvm::SmallVector<const NamedDecl*, 2> UnusedDecls;
          if (const auto* OvE = dyn_cast<OverloadExpr>(R.CustomDerivative)) {
            UnusedDecls.append(OvE->decls_begin(), OvE->decls_end());
          } else {
            const auto* DRE = cast<DeclRefExpr>(R.CustomDerivative);
            const auto* UnusedD = cast<FunctionDecl>(DRE->getDecl());
            UnusedDecls.push_back(UnusedD);
          }
          unsigned noteId = m_Sema.Diags.getCustomDiagID(
              DiagnosticsEngine::Note, "%0 is unused");
          for (const NamedDecl* UnusedD : UnusedDecls) {
            SourceLocation L = UnusedD->getLocation();
            m_Sema.Diag(L, noteId) << UnusedD << L;
          }
        }
      }
    }

    if (request.BaseFunctionName.empty())
      request.BaseFunctionName =
          utils::ComputeEffectiveFnName(request.Function);

    // FIXME: Here we copy all varied declarations down to the pullback, has to
    // be removed once AA and TBR are completely reworked, with better
    // branch-merging.
    if (m_ParentReq)
      for (const auto& decl : m_ParentReq->getVariedDecls())
        request.addVariedDecl(decl);

    llvm::SaveAndRestore<DiffRequest*> Saved(m_ParentReq, &request);
    if (request.Function->getDefinition())
      request.Function = request.Function->getDefinition();

    // Functions with side-effects require TBR.
    bool isNonConstMethod = false;
    if (const auto* MD = dyn_cast<CXXMethodDecl>(FD))
      isNonConstMethod = MD && MD->isInstance() && !MD->isConst();
    else if (const auto* OCE = dyn_cast<CXXOperatorCallExpr>(E))
      isNonConstMethod =
          utils::isNonConstReferenceType(OCE->getArg(0)->getType());
    bool requestTBR =
        request.EnableTBRAnalysis &&
        (request.Mode == DiffMode::pullback || isNonConstMethod) &&
        utils::hasMemoryTypeParams(request.Function) && request->isDefined() &&
        E->getDirectCallee();
    bool shouldUseRestoreTracker =
        utils::shouldUseRestoreTracker(request.Function);
    if (!(LookupCustomDerivativeDecl(request) || nonDiff) || requestTBR) {
      clang::CFG::BuildOptions Options;
      std::unique_ptr<AnalysisDeclContext> AnalysisDC =
          std::make_unique<AnalysisDeclContext>(
              /*AnalysisDeclContextManager=*/nullptr, request.Function,
              Options);

      if (request.EnableVariedAnalysis && request->isDefined()) {
        TimedAnalysisRegion R("VA " + request.BaseFunctionName);
        VariedAnalyzer analyzer(AnalysisDC.get(), request,
                                request.getVariedStmt());
        analyzer.Analyze();
      }

      if (m_TopMostReq->EnableUsefulAnalysis) {
        TimedAnalysisRegion R("UA " + request.BaseFunctionName);
        UsefulAnalyzer analyzer(AnalysisDC.get(), request.getUsefulDecls());
        analyzer.Analyze(request.Function);
      }

      m_AllAnalysisDC.push_back(std::move(AnalysisDC));
      request.m_AnalysisDC = m_AllAnalysisDC.back().get();

      //  Recurse into call graph.
      TraverseFunctionDeclOnce(request.Function);

      if (requestTBR) {
        TimedAnalysisRegion R("TBR " + request.BaseFunctionName);
        ParamInfo& modifiedParams = request.getModifiedParams();
        ParamInfo& usedParams = request.getUsedParams();
        TBRAnalyzer analyzer(request.m_AnalysisDC, request.getToBeRecorded(),
                             &modifiedParams, &usedParams);
        analyzer.Analyze(request);
        if (modifiedParams[FD].empty())
          shouldUseRestoreTracker = false;
        Saved.get()->addFunctionModifiedParams(FD, modifiedParams[FD]);
        Saved.get()->addFunctionUsedParams(FD, usedParams[FD]);
      }

      if (request.Mode == DiffMode::hessian ||
          request.Mode == DiffMode::hessian_diagonal) {
        DiffRequest forwRequest = request;
        forwRequest.Mode = DiffMode::forward;
        forwRequest.CallUpdateRequired = false;
        if (request.Mode == DiffMode::hessian_diagonal)
          forwRequest.RequestedDerivativeOrder = 2;
        for (const auto& dParam : request.DVI) {
          const auto* PVD = cast<ParmVarDecl>(dParam.param);
          auto indexInterval = dParam.paramIndexInterval;
          if (utils::isArrayOrPointerType(PVD->getType())) {
            // FIXME: We shouldn't synthesize Args strings.
            for (auto i = indexInterval.Start; i < indexInterval.Finish; ++i) {
              auto independentArgString =
                  PVD->getNameAsString() + "[" + std::to_string(i) + "]";
              forwRequest.Args = utils::CreateStringLiteral(
                  m_Sema.getASTContext(), independentArgString);
              forwRequest.UpdateDiffParamsInfo(m_Sema);
              LookupCustomDerivativeDecl(forwRequest);
              m_DiffRequestGraph.addNode(forwRequest, /*isSource=*/true);
            }
          } else {
            forwRequest.Args = utils::CreateStringLiteral(
                m_Sema.getASTContext(), PVD->getNameAsString());
            forwRequest.UpdateDiffParamsInfo(m_Sema);
            LookupCustomDerivativeDecl(forwRequest);
            m_DiffRequestGraph.addNode(forwRequest, /*isSource=*/true);
          }
        }
      }
    }

    if (request.Mode == DiffMode::pullback) {
      DiffRequest forwPassRequest;
      forwPassRequest.Function = request.Function;
      forwPassRequest.BaseFunctionName = request.BaseFunctionName;
      forwPassRequest.Mode = DiffMode::reverse_mode_forward_pass;
      forwPassRequest.CallContext = request.CallContext;
      forwPassRequest.UseRestoreTracker = shouldUseRestoreTracker;
      QualType returnType = request->getReturnType();
      if (LookupCustomDerivativeDecl(forwPassRequest) ||
          utils::isMemoryType(returnType) || shouldUseRestoreTracker)
        m_DiffRequestGraph.addNode(forwPassRequest, /*isSource=*/true);
    }

    if (!nonDiff && request.Mode != DiffMode::unknown)
      m_DiffRequestGraph.addNode(request, /*isSource=*/true);

    if (m_IsTraversingTopLevelDecl) {
      m_TopMostReq = nullptr;
      m_Traversed.clear();
    }

    return true;
  }

  bool DiffCollector::VisitDeclRefExpr(DeclRefExpr* DRE) {
    if (!m_ParentReq)
      return true;
    // FIXME: Add support for globals in other modes.
    if (m_ParentReq->Mode != DiffMode::reverse &&
        m_ParentReq->Mode != DiffMode::pullback)
      return true;

    // FIXME: In some cases, custom overloads are not found by DiffPlanner and
    // clad starts traversing builtin functions. This leads to some unnecessary
    // global adjoints being built and produces warnings in files where it's
    // impossible to expect them by tests. Because of this, global adjoints
    // are only created in the same file as the differentiated function for now.
    const clang::SourceManager& SM = m_Sema.getASTContext().getSourceManager();
    SourceLocation parentLoc = (*m_ParentReq)->getLocation();
    SourceLocation topMostLoc = (*m_TopMostReq)->getLocation();
    if (SM.getFileID(parentLoc) != SM.getFileID(topMostLoc))
      return true;

    if (const auto* VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      if (VD->isFileVarDecl() && !VD->getType().isConstQualified()) {
        DiffRequest request;
        request.DeclarationOnly = true;
        request.Global = VD;
        m_DiffRequestGraph.addNode(request, /*isSource=*/true);
      }
    }
    return true;
  }

  bool DiffCollector::VisitCXXConstructExpr(CXXConstructExpr* E) {
    // Don't visit CXXConstructExpr in outside differentiated functions.
    if (!m_TopMostReq)
      return true;

    // FIXME: add support for the forward mode
    if ((m_ParentReq->Mode != DiffMode::reverse) &&
        (m_ParentReq->Mode != DiffMode::pullback))
      return true;

    // FIXME: This only happens to perform nested TBR.
    // Constructors are not yet suported
    if (m_ParentReq->CustomDerivative)
      return true;

    CXXConstructorDecl* CD = E->getConstructor();
    DiffRequest forwPassRequest;
    forwPassRequest.Function = CD;
    forwPassRequest.BaseFunctionName = "constructor";
    forwPassRequest.Mode = DiffMode::reverse_mode_forward_pass;
    forwPassRequest.CallContext = E;
    QualType recordTy = CD->getThisType()->getPointeeType();
    bool elideRevForw =
        utils::isElidableConstructor(CD, m_Sema.getASTContext());
    if (LookupCustomDerivativeDecl(forwPassRequest) || !elideRevForw)
      m_DiffRequestGraph.addNode(forwPassRequest, /*isSource=*/true);

    // Don't build propagators for calls that do not contribute in
    // differentiable way to the result.
    if (allArgumentsAreLiterals(E->arguments(), m_ParentReq))
      return true;

    if (clad::utils::hasNonDifferentiableAttribute(CD->getParent()))
      return true;

    DiffRequest request;
    request.Function = CD;
    request.Mode = DiffMode::pullback;
    request.VerboseDiags = false;
    request.EnableTBRAnalysis = m_TopMostReq->EnableTBRAnalysis;
    request.EnableVariedAnalysis = m_TopMostReq->EnableVariedAnalysis;

    for (const auto* paramDecl : CD->parameters())
      request.DVI.push_back(paramDecl);

    request.BaseFunctionName = "constructor";
    request.CallContext = E;

    if (m_ParentReq)
      for (const auto& decl : m_ParentReq->getVariedDecls())
        request.addVariedDecl(decl);

    llvm::SaveAndRestore<DiffRequest*> Saved(m_ParentReq, &request);
    if (request.Function->getDefinition())
      request.Function = request.Function->getDefinition();

    if (m_Sema.isStdInitializerList(recordTy, /*elemType=*/nullptr))
      return true;

    if (!LookupCustomDerivativeDecl(request)) {
      clang::CFG::BuildOptions Options;
      std::unique_ptr<AnalysisDeclContext> AnalysisDC =
          std::make_unique<AnalysisDeclContext>(
              /*AnalysisDeclContextManager=*/nullptr, request.Function,
              Options);
      if (request.EnableVariedAnalysis) {
        TimedAnalysisRegion R("VA " + request.BaseFunctionName);
        VariedAnalyzer analyzer(AnalysisDC.get(), request,
                                request.getVariedStmt());
        analyzer.Analyze();
      }
      // FIXME: Add proper support for objects in VA and UA.
      m_AllAnalysisDC.push_back(std::move(AnalysisDC));
      request.m_AnalysisDC = m_AllAnalysisDC.back().get();

      // Recurse into call graph.
      TraverseFunctionDeclOnce(request.Function);
    }
    m_DiffRequestGraph.addNode(request, /*isSource=*/true);

    return true;
  }
  } // namespace clad
