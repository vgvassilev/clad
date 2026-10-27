#include "clad/Differentiator/DiffRequest.h"

#include "LivenessAnalyzer.h"
#include "LoopAnalyzer.h"
#include "TBRAnalyzer.h"

#include "clad/Differentiator/CladConfig.h"
#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"
#include "clad/Differentiator/DiffMode.h"
#include "clad/Differentiator/Options.h"
#include "clad/Differentiator/ParseDiffArgsTypes.h"
#include "clad/Differentiator/Timers.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTLambda.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

using namespace clang;

namespace clad {
static QualType GetDerivedFunctionType(const CallExpr* CE) {
  const auto* CXXRD = CE->getType()->getAsCXXRecordDecl();
  const auto* Spec = cast<ClassTemplateSpecializationDecl>(CXXRD);
  assert(Spec && "Called with the wrong expression!");
  const TemplateArgument& TemplArg = Spec->getTemplateArgs().get(/*Idx=*/0);
  return TemplArg.getAsType();
}

// Rewriting the call site mutates the request's associated AST.
// NOLINTNEXTLINE(readability-make-member-function-const)
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
            .ActOnCXXBoolLiteral(SourceLocation(),
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
  Expr* Arg = SemaRef.BuildDeclRefExpr(replacementFD, replacementFD->getType(),
                                       VK, SourceLocation(), &CSS);
  // Add the "&" operator
  Arg = SemaRef
            .BuildUnaryOp(/*Scope=*/nullptr, SourceLocation(),
                          UnaryOperatorKind::UO_AddrOf, Arg)
            .get();
  // Take into account if the user selected an overload by a cast expr.
  if (isa<ExplicitCastExpr>(call->getArg(0))) {
    QualType Ty = GetDerivedFunctionType(call);
    TypeSourceInfo* TSI = C.getTrivialTypeSourceInfo(Ty, SourceLocation());
    Arg = SemaRef
              .BuildCStyleCastExpr(SourceLocation(), TSI, SourceLocation(), Arg)
              .get();
  }
  call->setArg(*derivedFnArgIdx, Arg);

  // Update the code parameter if it was found. Use the context's
  // PrintingPolicy so DeclRefExpr / NestedNameSpecifier print the same
  // as the rest of the translation unit -- a fresh PrintingPolicy
  // built from a default-LangOptions object yields stricter defaults
  // on LLVM 22 (notably for unwritten-scope decisions), which then
  // makes locals print qualified as `::name` even though their DRE
  // carries no qualifier.
  clang::PrintingPolicy Policy = C.getPrintingPolicy();
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

const ReturnStmt* DiffRequest::getTailReturn() const {
  const FunctionDecl* Def = Function ? Function->getDefinition() : nullptr;
  if (!Def || !Def->hasBody())
    return nullptr;
  const Stmt* Body = Def->getBody();
  // The tail return is the body's last statement, when that is a return.
  if (const auto* CS = dyn_cast<CompoundStmt>(Body))
    return CS->body_empty() ? nullptr
                            : dyn_cast<ReturnStmt>(*CS->body_rbegin());
  return dyn_cast<ReturnStmt>(Body);
}

/// Whether \p S has a `break` or `continue` of the loop whose body it is:
/// not one of a nested loop, nor a `break` of a switch inside it.
static bool loopBodyHasBreakOrContinue(const Stmt* S,
                                       bool insideSwitch = false) {
  if (isa<ContinueStmt>(S))
    return true;
  if (isa<BreakStmt>(S))
    return !insideSwitch;
  if (isa<ForStmt>(S) || isa<WhileStmt>(S) || isa<DoStmt>(S) ||
      isa<CXXForRangeStmt>(S))
    return false;
  // A child may be absent, an omitted part of the statement.
  if (isa<SwitchStmt>(S)) {
    for (const Stmt* C : S->children())
      if (C && loopBodyHasBreakOrContinue(C, /*insideSwitch=*/true))
        return true;
    return false;
  }
  for (const Stmt* C : S->children())
    if (C && loopBodyHasBreakOrContinue(C, insideSwitch))
      return true;
  return false;
}

bool DiffRequest::hasBreakOrContinue(const Stmt* LoopBody) const {
  auto [It, Inserted] = m_BreakOrContinueInfo.try_emplace(LoopBody, false);
  if (Inserted)
    It->second = loopBodyHasBreakOrContinue(LoopBody);
  return It->second;
}

bool DiffRequest::hasEarlyReturns() const {
  if (m_EarlyReturnInfo.HasAnalysisRun)
    return m_EarlyReturnInfo.HasEarlyReturns;
  const FunctionDecl* Def = Function ? Function->getDefinition() : nullptr;
  if (!Def || !Def->hasBody())
    return false;
  // An early-return body has a return that is not the tail return. Returns
  // inside a nested lambda belong to that lambda's own function, so do not
  // descend into one.
  struct Finder : RecursiveASTVisitor<Finder> {
    const ReturnStmt* Tail = nullptr;
    bool Found = false;
    static bool TraverseLambdaExpr(LambdaExpr*) { return true; }
    bool VisitReturnStmt(ReturnStmt* RS) {
      if (RS != Tail)
        Found = true;
      return !Found; // stop at the first early return
    }
  } F;
  F.Tail = getTailReturn();
  F.TraverseStmt(Def->getBody());
  m_EarlyReturnInfo = {F.Found, /*HasAnalysisRun=*/true};
  return F.Found;
}

llvm::SmallVector<const ParmVarDecl*, 2> DiffRequest::getOutputParams() const {
  llvm::SmallVector<const ParmVarDecl*, 2> outputParams;
  if (Function && utils::CanReturnOutputTangent(Function))
    utils::CollectOutputParams(Function, outputParams);
  return outputParams;
}

const ParmVarDecl* DiffRequest::getOutputTangentParam() const {
  // Every other mode either keeps the primal's return type or has somewhere
  // of its own to put an output parameter's tangent, as a pushforward does.
  if (Mode != DiffMode::forward)
    return nullptr;
  llvm::SmallVector<const ParmVarDecl*, 2> outputParams = getOutputParams();
  if (outputParams.size() != 1)
    return nullptr;
  return outputParams.front();
}

const FunctionLoopFacts& DiffRequest::getLoopFacts() const {
  // Re-run for another function: a copied request keeps the facts of the one
  // it was made for, and ProcessDiffRequest re-points a request at the
  // definition of what it was planned for.
  if (!m_LoopFacts || m_LoopFacts->Fn != Function) {
    auto Facts = std::make_shared<FunctionLoopFacts>();
    Facts->Fn = Function;
    // Switched off, the facts stay empty, which every reader already has to
    // handle: it is what a function whose loops prove nothing looks like.
    if (Function && EnableLoopAnalysis)
      analyzeLoops(*this, *Facts);
    m_LoopFacts = std::move(Facts);
  }
  return *m_LoopFacts;
}

const LoopFacts& DiffRequest::getLoopFacts(const clang::Stmt* S) const {
  static const LoopFacts None;
  const auto& Loops = getLoopFacts().Loops;
  auto it = Loops.find(S);
  return it == Loops.end() ? None : it->second;
}

void DiffRequest::recordMiss(AnalysisMiss M, SourceLocation At) const {
  if (!m_Misses)
    m_Misses = std::make_shared<AnalysisMisses>();
  AnalysisMissRecord R{M, At};
  if (!llvm::is_contained(m_Misses->Records, R))
    m_Misses->Records.push_back(R);
}

llvm::ArrayRef<AnalysisMissRecord> DiffRequest::getAnalysisMisses() const {
  if (!m_Misses)
    return {};
  return m_Misses->Records;
}

llvm::ArrayRef<WrittenExtent> DiffRequest::getWrittenExtents() const {
  return getLoopFacts().Extents;
}

bool DiffRequest::writesVariable(const VarDecl* VD) const {
  // A reference names storage this walk does not follow, and static or
  // external storage is reachable from inside a callee. Neither can be
  // ruled out by looking at the body alone.
  if (VD->getType()->isReferenceType() || !VD->hasLocalStorage())
    return true;
  const FunctionDecl* Def = Function ? Function->getDefinition() : nullptr;
  if (!Def || !Def->hasBody())
    return true;
  if (!m_WrittenVarInfo.HasAnalysisRun) {
    utils::collectWrittenVars(Def->getBody(), m_WrittenVarInfo.Written);
    m_WrittenVarInfo.HasAnalysisRun = true;
  }
  return m_WrittenVarInfo.Written.count(VD) != 0;
}

bool DiffRequest::isUnreadCall(const Stmt* S) const {
  // Only a sweep that goes on into its own reverse can leave work out: a
  // reverse_forw hands its results to its caller, a jacobian its output.
  if (!EnableLivenessAnalysis ||
      (Mode != DiffMode::reverse && Mode != DiffMode::pullback))
    return false;
  if (!m_UnreadCallInfo.HasAnalysisRun) {
    findUnreadCalls(*this, m_UnreadCallInfo.Unread);
    m_UnreadCallInfo.HasAnalysisRun = true;
  }
  return m_UnreadCallInfo.Unread.count(S) != 0;
}

namespace {
/// Propagates "the tangent of this pointer may be null" along the pointer
/// initializations and assignments of a function body, to a fixpoint. Runs on
/// the primal, so that the classification does not depend on the order in
/// which the derivative code is generated: a read may precede the assignment
/// that makes the tangent null, as it does across loop iterations.
class MaybeNullPointerPropagator
    : public RecursiveASTVisitor<MaybeNullPointerPropagator> {
  std::set<const VarDecl*>* m_MaybeNull;
  bool m_Changed = false;

  /// Whether \p E reads a pointer whose tangent may be null. Pointer
  /// arithmetic and casts keep such a tangent null, so the whole expression
  /// is searched.
  bool readsMaybeNull(const Expr* E) const {
    if (!E)
      return false;
    if (const auto* DRE = dyn_cast<DeclRefExpr>(E->IgnoreParenImpCasts()))
      if (const auto* VD = dyn_cast<VarDecl>(DRE->getDecl()))
        if (m_MaybeNull->count(VD) != 0)
          return true;
    for (const Stmt* child : E->children())
      if (const auto* childExpr = dyn_cast_or_null<Expr>(child))
        if (readsMaybeNull(childExpr))
          return true;
    return false;
  }

  void taint(const VarDecl* VD) {
    if (VD && VD->getType()->isPointerType() && m_MaybeNull->insert(VD).second)
      m_Changed = true;
  }

public:
  explicit MaybeNullPointerPropagator(std::set<const VarDecl*>& maybeNull)
      : m_MaybeNull(&maybeNull) {}

  bool VisitVarDecl(VarDecl* VD) {
    // A pointer clad has no tangent for is given a null one.
    if (VD->getType()->isPointerType() &&
        (!VD->getInit() || readsMaybeNull(VD->getInit())))
      taint(VD);
    return true;
  }

  bool VisitBinaryOperator(BinaryOperator* BO) {
    if (BO->getOpcode() != BO_Assign || !BO->getType()->isPointerType() ||
        !readsMaybeNull(BO->getRHS()))
      return true;
    const auto* DRE =
        dyn_cast<DeclRefExpr>(BO->getLHS()->IgnoreParenImpCasts());
    if (DRE)
      taint(dyn_cast<VarDecl>(DRE->getDecl()));
    return true;
  }

  void runToFixpoint(Stmt* Body) {
    m_Changed = true;
    while (m_Changed) {
      m_Changed = false;
      TraverseStmt(Body);
    }
  }
};
} // namespace

bool DiffRequest::mayHaveNullTangent(const VarDecl* VD) const {
  if (!VD || !VD->getType()->isPointerType())
    return false;
  if (!m_NullTangentInfo.HasAnalysisRun) {
    m_NullTangentInfo.HasAnalysisRun = true;
    const FunctionDecl* Def = Function ? Function->getDefinition() : nullptr;
    if (Def && Def->hasBody()) {
      std::set<const VarDecl*>& MaybeNull = m_NullTangentInfo.MaybeNullPtrs;
      // Clad has no tangent buffer to pass for a const-qualified pointer
      // parameter; the pointers of the body derived from one inherit that.
      for (const ParmVarDecl* PVD : Def->parameters()) {
        QualType T = PVD->getType();
        if (T->isPointerType() && T->getPointeeType().isConstQualified())
          MaybeNull.insert(PVD);
      }
      MaybeNullPointerPropagator(MaybeNull).runToFixpoint(Def->getBody());
    }
  }
  return m_NullTangentInfo.MaybeNullPtrs.count(VD) != 0;
}

DiffRequest DiffRequest::pushforwardRequestForHessian(Sema& semaRef) const {
  DiffRequest pushforwardRequest = *this;
  pushforwardRequest.Mode = DiffMode::pushforward;
  pushforwardRequest.Args = nullptr;
  pushforwardRequest.CallUpdateRequired = false;
  pushforwardRequest.UseHessianVectorProducts = false;
  pushforwardRequest.UpdateDiffParamsInfo(semaRef);
  return pushforwardRequest;
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
  if (!diffArgs || !FD)
    return;
  DiffParams params{};
  const auto* E = diffArgs->IgnoreParenImpCasts();
  // Case 1)
  SourceLocation dArgsL = diffArgs->getBeginLoc();
  if (const auto* SL = dyn_cast<StringLiteral>(E)) {
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
    while (!string.empty()) {
      llvm::StringRef pInfo{};
      std::tie(pInfo, string) = string.split(',');
      diffParamsSpec.push_back(pInfo.trim());
    }
    // Stores parameters and field declarations to be used as candidates for
    // independent arguments.
    // If we are differentiating a call operator that have no parameters,
    // then candidates for independent argument are member variables of the
    // class that defines the call operator.
    // Otherwise, candidates are parameters of the function that we are
    // differentiating.
    llvm::SmallVector<std::pair<llvm::StringRef, ValueDecl*>, 16> candidates{};

    // find and store candidate parameters.
    if (FD->param_empty() && Functor)
      for (FieldDecl* fieldDecl : Functor->fields())
        candidates.emplace_back(fieldDecl->getName(), fieldDecl);
    else
      for (auto* PVD : FD->parameters())
        candidates.emplace_back(PVD->getName(), PVD);

    auto computeParamName = [](llvm::StringRef diffSpec) {
      std::size_t idx = diffSpec.find_first_of(".[");
      return diffSpec.substr(0, idx);
    };

    // Ensure that diff params are always considered in the same order.
    // This is required to uniquely identify differentiation requests.
    std::sort(
        diffParamsSpec.begin(), diffParamsSpec.end(),
        [&candidates, &computeParamName](llvm::StringRef a, llvm::StringRef b) {
          auto* a_it = std::find_if(
              candidates.begin(), candidates.end(),
              [a, &computeParamName](
                  const std::pair<llvm::StringRef, ValueDecl*>& candidate) {
                return candidate.first == computeParamName(a);
              });
          auto* b_it = std::find_if(
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
      auto* it = std::find_if(
          std::begin(candidates), std::end(candidates),
          [&pName](const std::pair<llvm::StringRef, ValueDecl*>& p) {
            return p.first == pName;
          });

      if (it == std::end(candidates)) {
        // Fail if the function has no parameter with specified name.
        utils::diag(semaRef, DiagnosticsEngine::Error, dArgsL,
                    "requested parameter name '%0' was not found among "
                    "function parameters")
            << pName << dArgsL;
        return;
      }

      auto* f_it = std::find_if(std::begin(DVI), std::end(DVI),
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
        llvm::StringRef interval(
            diffSpec.slice(lSqBracketIdx + 1, diffSpec.find(']')));
        llvm::StringRef firstStr;
        llvm::StringRef lastStr;
        std::tie(firstStr, lastStr) = interval.split(':');

        static constexpr unsigned Radix = 10;
        if (lastStr.empty()) {
          // The string is not a range just a single index
          size_t index = 0;
          if (firstStr.getAsInteger(Radix, index)) {
            utils::diag(semaRef, DiagnosticsEngine::Error, dArgsL,
                        "could not parse index '%0'")
                << diffSpec << dArgsL;
            return;
          }
          dVarInfo.paramIndexInterval = IndexInterval(index);
          dVarInfo.TotalCapacity = index + 1;
        } else {
          size_t first = 0;
          size_t last = 0;
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
          dVarInfo.TotalCapacity = last + 1;
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

      // A parameter passed by reference names its fields as one passed by
      // value does, so the fields are looked up in the class referred to.
      QualType paramTy = dVarInfo.param->getType().getNonReferenceType();
      if (!paramTy->isRecordType() && !dVarInfo.fields.empty()) {
        utils::diag(semaRef, DiagnosticsEngine::Error, dArgsL,
                    "fields can only be provided for class type parameters; "
                    "field information is incorrectly specified in '%0' "
                    "for non-class type parameter '%1'")
            << diffSpec << pName << dArgsL;
        return;
      }

      if (!dVarInfo.fields.empty()) {
        RecordDecl* RD = paramTy->getAsCXXRecordDecl();
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
      auto totalFields =
          std::distance(Functor->field_begin(), Functor->field_end());
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
    for (const auto* param : params) {
      DiffInputVarInfo dVarInfo(param, IndexInterval());
      DVI.push_back(dVarInfo);
    }
    return;
  }
  // Fail if the argument is not a string or numeric literal.
  utils::diag(semaRef, DiagnosticsEngine::Error, dArgsL,
              "failed to parse the parameters, must be string or "
              "numeric literal");
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
    TBRAnalyzer analyzer(m_AnalysisDC, getToBeRecorded(), &getModifiedParams(),
                         &getUsedParams());
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

bool DiffRequest::shouldHaveAdjoint(const clang::Stmt* S) const {
  if (!EnableVariedAnalysis)
    return true;
  auto found = m_ActivityRunInfo.VariedS.find(S);
  return found != m_ActivityRunInfo.VariedS.end();
}
bool DiffRequest::shouldHaveAdjoint(const clang::VarDecl* VD) const {
  if (!EnableVariedAnalysis)
    return true;
  return getVariedDecls().find(VD) != getVariedDecls().end();
}
bool DiffRequest::shouldHavePushforward(const CallExpr* CE) const {
  auto found = m_ActivityRunInfo.VariedCalls.find(CE);
  return found == m_ActivityRunInfo.VariedCalls.end() || found->second;
}

FunctionDecl* DiffRequest::getDefaultAdjoint(Sema& S, const CallExpr* CE,
                                             bool nonDiff) const {
  // A reverse_forw must supply an adjoint the caller can accumulate into,
  // even when a constant factory contributes nothing to the input gradient.
  if (nonDiff && Mode != DiffMode::reverse_mode_forward_pass)
    return nullptr;
  QualType returnType = CE->getDirectCallee()->getReturnType();
  if (!returnType->isRecordType() || !utils::isMemoryType(returnType))
    return nullptr;

  // Both StoreAndRef and GlobalStoreAndRef strip const from the stored value.
  QualType storedType = utils::getNonConstType(returnType, S);
  OpaqueValueExpr value(CE->getExprLoc(), storedType, VK_LValue);
  return utils::LookupCladZeroLike(S, &value);
}

bool DiffRequest::isVaried(const Expr* E) const {
  // FIXME: We should consider removing pullback requests from the
  // diff graph.
  class VariedChecker : public RecursiveASTVisitor<VariedChecker> {
    const DiffRequest& m_Request;

  public:
    VariedChecker(const DiffRequest& DR) : m_Request(DR) {}
    // The default expression is not exposed through children().
    bool VisitCXXDefaultArgExpr(CXXDefaultArgExpr* E) {
      return TraverseStmt(E->getExpr());
    }
    bool VisitCXXDefaultInitExpr(CXXDefaultInitExpr* E) {
      return TraverseStmt(E->getExpr());
    }
    bool TraverseInitListExpr(InitListExpr* E) {
      // Default member initializers only appear in the semantic form.
      InitListExpr* semantic = E->getSemanticForm();
      return TraverseSynOrSemInitListExpr(semantic ? semantic : E);
    }
    bool isVariedE(const clang::Expr* E) {
      // The call-activity pre-pass also fills the varied set, and its
      // conservative markings (an argument handed to a non-const pointer
      // parameter is varied no matter what it is) must only feed
      // shouldHavePushforward. Without the opt-in analysis, keep answering
      // from the expression alone, as before the pre-pass existed.
      if (m_Request.EnableVariedAnalysis &&
          m_Request.getVariedStmt().find(E) != m_Request.getVariedStmt().end())
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

bool DiffRequest::restoresCallWrites(const CallExpr* CE) const {
  const FunctionDecl* FD = CE->getDirectCallee();
  assert(FD && "Reverse mode asks only about a call to a known function");
  ParamInfo& modified = getModifiedParams();
  auto found = modified.find(FD);
  if (found == modified.end() || found->second.empty())
    return false;
  // A method operator takes its object as the first argument.
  std::size_t skip = isa<CXXOperatorCallExpr>(CE) && isa<CXXMethodDecl>(FD);
  const ASTContext& C = FD->getASTContext();
  return llvm::all_of(found->second, [&](const ParmVarDecl* PVD) {
    if (!PVD)
      return false;
    QualType T = PVD->getType();
    std::size_t argIdx = skip + PVD->getFunctionScopeIndex();
    return T->isLValueReferenceType() &&
           !T.getNonReferenceType().isConstQualified() &&
           !T.getNonReferenceType()->isArrayType() &&
           utils::reachesNoOtherMemory(C, T.getNonReferenceType()) &&
           argIdx < CE->getNumArgs() && shouldBeRecorded(CE->getArg(argIdx));
  });
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

bool DiffRequest::UpdateDiffOptions(Sema& S, SourceLocation BeginLoc,
                                    const Options& ReqOpts,
                                    const FunctionDecl* Invocation) {
  if (Invocation) {
    const auto* A = Invocation->getAttr<AnnotateAttr>();
    assert(A && "Clad invocation must have a mode annotation");
    llvm::StringRef Annotation = A->getAnnotation();
    if (Annotation == "D")
      Mode = DiffMode::forward;
    else if (Annotation == "H")
      Mode = DiffMode::hessian;
    else if (Annotation == "J")
      Mode = DiffMode::jacobian;
    else if (Annotation == "G")
      Mode = DiffMode::reverse;
    else if (Annotation == "E") {
      Mode = DiffMode::reverse;
      EnableErrorEstimation = true;
    } else if (Annotation != "A")
      llvm_unreachable("unknown mode");
  }

  EnableLoopAnalysis = ReqOpts.EnableLoopAnalysis;
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, FirstBit, Desc)               \
  Remark##Id##Analysis = ReqOpts.Remark##Id##Analysis;
#include "clad/Differentiator/Analyses.def"
  // Seed fallbacks before the error-estimation early return as well.
#define CLAD_FALLBACK(Id, Opt, Flag, Default, FirstBit, Desc)                  \
  Enable##Id##Fallback = ReqOpts.Enable##Id##Fallback;
#include "clad/Differentiator/Analyses.def"
  FallbacksSeeded = true;
  // Error estimation has no invocation options yet.
  if (EnableErrorEstimation)
    return false;

    // What the command line settled on for the translation unit, which the
    // clad::opts pairs below may still override for this one request.
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, FirstBit, Desc)               \
  Enable##Id##Analysis = ReqOpts.Enable##Id##Analysis;
#include "clad/Differentiator/Analyses.def"
  // TBR has nothing to do where there is no reverse sweep. A request that
  // asks for it there is an error below; the command line asking for it
  // everywhere is not, so it is dropped rather than diagnosed.
  if (Mode != DiffMode::reverse && Mode != DiffMode::hessian &&
      Mode != DiffMode::pullback)
    EnableTBRAnalysis = false;
  EmitPortingHints = ReqOpts.EmitPortingHints;

  if (!Invocation)
    return false;

  const TemplateArgumentList* TAL = Invocation->getTemplateSpecializationArgs();
  assert(TAL && "Call must have specialization args!");

  // bitmask_opts is a template pack of unsigned integers, so we need to
  // do bitwise or of all the values to get the final value.
  unsigned bitmasked_opts_value = 0;
  const auto template_arg = TAL->get(/*Idx=*/0);
  if (template_arg.getKind() == TemplateArgument::Pack)
    for (const auto& arg : TAL->get(/*Idx=*/0).pack_elements())
      bitmasked_opts_value |= arg.getAsIntegral().getExtValue();

      // What the request asks of each analysis, over the
      // whole-translation-unit answer it already carries. Generated from the
      // same table the command line is: an analysis that can be switched for
      // the translation unit can be switched for one request.
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, FirstBit, Desc)               \
  const bool enable_##Legacy##_in_req =                                        \
      clad::HasOption(bitmasked_opts_value, clad::opts::enable_##Legacy);      \
  const bool disable_##Legacy##_in_req =                                       \
      clad::HasOption(bitmasked_opts_value, clad::opts::disable_##Legacy);     \
  if (enable_##Legacy##_in_req && disable_##Legacy##_in_req) {                 \
    utils::diag(S, DiagnosticsEngine::Error, BeginLoc,                         \
                "both clad::opts::enable_" #Legacy                             \
                " and clad::opts::disable_" #Legacy " are specified")          \
        << BeginLoc;                                                           \
    return true;                                                               \
  }                                                                            \
  if (enable_##Legacy##_in_req || disable_##Legacy##_in_req)                   \
    Enable##Id##Analysis = enable_##Legacy##_in_req;
#include "clad/Differentiator/Analyses.def"

      // And of each fallback, the same way.
#define CLAD_FALLBACK(Id, Opt, Flag, Default, FirstBit, Desc)                  \
  {                                                                            \
    const bool Enable =                                                        \
        clad::HasOption(bitmasked_opts_value, clad::opts::enable_##Opt);       \
    const bool Disable =                                                       \
        clad::HasOption(bitmasked_opts_value, clad::opts::disable_##Opt);      \
    if (Enable && Disable) {                                                   \
      utils::diag(S, DiagnosticsEngine::Error, BeginLoc,                       \
                  "both clad::opts::enable_" #Opt                              \
                  " and clad::opts::disable_" #Opt " are specified")           \
          << BeginLoc;                                                         \
      return true;                                                             \
    }                                                                          \
    if (Enable || Disable)                                                     \
      Enable##Id##Fallback = Enable;                                           \
  }
#include "clad/Differentiator/Analyses.def"

  if (enable_tbr_in_req && Mode == DiffMode::forward) {
    utils::diag(S, DiagnosticsEngine::Error, BeginLoc,
                "tbr analysis is not meant for forward mode AD")
        << BeginLoc;
    return true;
  }

  // reverse vector mode is not yet supported.
  if (Mode == DiffMode::reverse &&
      clad::HasOption(bitmasked_opts_value, clad::opts::vector_mode)) {
    utils::diag(S, DiagnosticsEngine::Error, BeginLoc,
                "reverse vector mode is not yet supported")
        << BeginLoc;
    return true;
  }

  // Check for clad::hessian<diagonal_only>.
  if (clad::HasOption(bitmasked_opts_value, clad::opts::diagonal_only)) {
    if (Mode == DiffMode::hessian) {
      Mode = DiffMode::hessian_diagonal;
      return false;
    }
    utils::diag(S, DiagnosticsEngine::Error, BeginLoc,
                "diagonal only option is only valid for hessian mode");
    return true;
  }

  if (clad::HasOption(bitmasked_opts_value, clad::opts::use_enzyme))
    use_enzyme = true;

  if (Mode == DiffMode::forward) {
    // Check for clad::differentiate<N>.
    if (unsigned order = clad::GetDerivativeOrder(bitmasked_opts_value))
      RequestedDerivativeOrder = order;

    // Check for clad::differentiate<vector_mode>.
    if (clad::HasOption(bitmasked_opts_value, clad::opts::vector_mode)) {
      Mode = DiffMode::vector_forward_mode;

      // Currently only first order derivative is supported.
      if (RequestedDerivativeOrder != 1) {
        utils::diag(S, DiagnosticsEngine::Error, BeginLoc,
                    "only first order derivative is supported for now "
                    "in vector forward mode")
            << BeginLoc;
        return true;
      }

      // We don't yet support enzyme with vector mode.
      if (use_enzyme) {
        utils::diag(S, DiagnosticsEngine::Error, BeginLoc,
                    "enzyme's vector mode is not yet supported")
            << BeginLoc;
        return true;
      }
    }
  }

  return false;
}

} // namespace clad
