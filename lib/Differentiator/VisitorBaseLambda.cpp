//===- VisitorBaseLambda.cpp - Lambda support shared by the visitors ------===//
//
// Building a closure and resolving its captures, cloning a lambda with a
// fresh closure type, reading a capture and its adjoint, and refusing what
// is not supported yet.
//
//===----------------------------------------------------------------------===//

#include "clad/Differentiator/VisitorBase.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/DerivativeBuilder.h"
#include "clad/Differentiator/DiffMode.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/StmtClone.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/Type.h"
#include "clang/Basic/ExceptionSpecificationType.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/Lambda.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/Version.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SaveAndRestore.h"

#include <cassert>
#include <unordered_map>
#include <utility>

#include "clad/Differentiator/Compatibility.h"

using namespace clang;

namespace clad {
void VisitorBase::LambdaCaptures::collect(llvm::ArrayRef<Stmt*> Body) {
  class FreeVarCollector : public RecursiveASTVisitor<FreeVarCollector> {
  public:
    llvm::SmallSetVector<VarDecl*, 16> Referenced;
    llvm::SmallPtrSet<VarDecl*, 16> Declared;
    // Both forms of an initializer list, as resolve() walks them.
    bool shouldVisitImplicitCode() const { return true; }
    bool VisitDeclRefExpr(DeclRefExpr* DRE) {
      if (auto* VD = dyn_cast<VarDecl>(DRE->getDecl()))
        if (VD->isLocalVarDecl() || isa<ParmVarDecl>(VD))
          Referenced.insert(VD);
      return true;
    }
    bool VisitDeclStmt(DeclStmt* DS) {
      for (Decl* D : DS->decls())
        if (auto* VD = dyn_cast<VarDecl>(D))
          Declared.insert(VD);
      return true;
    }
  } C;
  for (Stmt* S : Body)
    C.TraverseStmt(S);
  for (VarDecl* VD : C.Referenced)
    if (!C.Declared.count(VD))
      m_Captures.insert(VD);
}

void VisitorBase::LambdaCaptures::orderCaptureDecls(
    llvm::SmallVectorImpl<Stmt*>& Prefix,
    llvm::SmallVectorImpl<Stmt*>& Suffix) {
  // A decl moves whole only when its initializer reads no local, no
  // parameter and no global the program can change: a zero, a literal, a
  // constant. Anything else is computed in place, because moving it would
  // run it on a path that returns before it -- past a guard the original
  // put in front of it -- and would read its operands before the forward
  // sweep has written them.
  auto selfContained = [&](const VarDecl* VD) {
    const Expr* Init = VD->getInit();
    if (!Init)
      return true;
    bool Ok = true;
    class RefChecker : public RecursiveASTVisitor<RefChecker> {
    public:
      bool* Ok = nullptr;
      bool VisitDeclRefExpr(DeclRefExpr* DRE) const {
        auto* RVD = dyn_cast<VarDecl>(DRE->getDecl());
        if (RVD && (RVD->isLocalVarDecl() || isa<ParmVarDecl>(RVD) ||
                    !RVD->getType().isConstQualified())) {
          *Ok = false;
          return false;
        }
        return true;
      }
    } RC;
    RC.Ok = &Ok;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    RC.TraverseStmt(const_cast<Expr*>(Init));
    return Ok;
  };
  llvm::SmallVector<Stmt*, 16> Remaining;
  for (Stmt* S : Suffix) {
    auto* DS = dyn_cast<DeclStmt>(S);
    bool AnyCaptured = DS && llvm::any_of(DS->decls(), [&](Decl* D) {
                         auto* VD = dyn_cast<VarDecl>(D);
                         return VD && contains(VD);
                       });
    if (!AnyCaptured) {
      Remaining.push_back(S);
      continue;
    }

    // At least one decl here is captured; rebuild the DeclStmt decl-by-decl.
    // A captured decl with a self-contained initializer moves whole; one
    // that isn't (e.g. `double y = n + x;`, reading a not-yet-hoisted `n`)
    // is split: a zero-initialized declaration precedes the lambda and a
    // plain assignment with the real value stays at the original spot.
    // Referencing a local before any DeclStmt for it has run makes CodeGen
    // treat it as a block-scope static instead of a real capture
    // (vgvassilev/clad#1940); splitting avoids that while still computing
    // the value -- and any side effects -- exactly where the original
    // control flow put it. Decls that are neither (an uncaptured neighbour
    // in `double a = n + x, c = 3.;`, or an array, which has no
    // whole-object assignment to split into) stay behind, regrouped into a
    // DeclStmt at the point their order requires.
    llvm::SmallVector<Decl*, 4> KeptHere;
    auto flushKept = [&]() {
      if (KeptHere.empty())
        return;
      Remaining.push_back(m_V.BuildDeclStmt(KeptHere));
      KeptHere.clear();
    };
    for (Decl* D : DS->decls()) {
      auto* VD = dyn_cast<VarDecl>(D);
      bool Hoistable = VD && contains(VD);
      if (Hoistable && selfContained(VD)) {
        flushKept();
        // Moved wholesale, the decl may still carry no real initial value:
        // default init of a scalar or of a trivially-default-constructible
        // record leaves indeterminate bits. In place that was fine -- the
        // original program wrote them before any read -- but hoisted above
        // the lambda they become readable by the reverse sweep on a path
        // that returns before those writes. The adjoints there are zero, so
        // only definedness is at stake; a zero initializer settles it.
        const Expr* Init = VD->getInit();
        bool Indeterminate = !Init;
        if (const auto* CE = dyn_cast_or_null<CXXConstructExpr>(Init))
          Indeterminate = CE->getConstructor()->isDefaultConstructor() &&
                          CE->getConstructor()->isTrivial();
        if (Indeterminate)
          if (Expr* Zero = m_V.getZeroInit(VD->getType()))
            m_V.SetDeclInit(VD, Zero);
        Prefix.push_back(m_V.BuildDeclStmt(VD));
        continue;
      }
      // Splitting needs a whole-object assignment, which an array type has
      // no spelling for; an element-wise copy would need
      // ReverseModeVisitor::BuildArrayAssignment, out of this shared
      // VisitorBase utility's reach. A record with a reference member has
      // none either: its assignment is deleted. Keep the decl --
      // findUseBeforeDecl still reports it if it really was captured.
      const auto* RD = VD ? VD->getType()->getAsCXXRecordDecl() : nullptr;
      bool HoldsReference =
          RD && llvm::any_of(RD->fields(), [](const FieldDecl* F) {
            return F->getType()->isReferenceType();
          });
      if (!Hoistable || isa<ArrayType>(VD->getType()) || HoldsReference) {
        KeptHere.push_back(D);
        continue;
      }
      // selfContained() is true for an initializer-less decl, so reaching
      // here means there is an initializer to move into the assignment.
      Expr* Init = VD->getInit();
      // A const clone cannot take that assignment. Placeholder plus one
      // assignment still leave the clone a single value for the rest of its
      // lifetime, so dropping the qualifier costs nothing. (A decl promoted
      // to the function global scope is split the same way and loses its
      // const up front, in ReverseModeVisitor::DifferentiateVarDecl.)
      QualType VDTy = VD->getType();
      if (VDTy.isLocalConstQualified()) {
        VDTy.removeLocalConst();
        VD->setType(VDTy);
        VD->setTypeSourceInfo(
            m_V.m_Context.getTrivialTypeSourceInfo(VDTy, noLoc));
      }
      Expr* DeclRef = m_V.BuildDeclRef(VD);
      // The lambda reads the placeholder on a path that returns before the
      // assignment below, so a pointer needs a dereferenceable one; its
      // dummy declaration goes into Prefix ahead of VD's.
      Expr* Placeholder =
          m_V.BuildDereferenceablePlaceholder(VD->getType(), Prefix);
      if (!Placeholder)
        Placeholder = m_V.getZeroInit(VD->getType());
      m_V.SetDeclInit(VD, Placeholder);
      flushKept();
      Prefix.push_back(m_V.BuildDeclStmt(VD));
      Remaining.push_back(m_V.BuildOp(BO_Assign, DeclRef, Init));
    }
    flushKept();
  }
  Suffix = std::move(Remaining);
}

void VisitorBase::LambdaCaptures::resolve(llvm::ArrayRef<Stmt*> Body) {
  class CapRefRebuilder : public RecursiveASTVisitor<CapRefRebuilder> {
  public:
    LambdaCaptures& Caps;
    explicit CapRefRebuilder(LambdaCaptures& Caps) : Caps(Caps) {}
    // An initializer list has a syntactic and a semantic form, and CodeGen
    // reads the semantic one, which the visitor skips unless told to visit
    // implicit code. A reference rebuilt in the syntactic form alone leaves
    // the emitted one uncaptured.
    bool shouldVisitImplicitCode() const { return true; }
    bool VisitStmt(Stmt* P) {
      for (Stmt*& Child : P->children()) {
        // `this` too is captured only where it is named; in a member
        // function's closure it was built outside it.
        if (auto* This = dyn_cast_or_null<CXXThisExpr>(Child)) {
          Child = Caps.m_V.m_Sema.BuildCXXThisExpr(
              This->getLocation(), This->getType(), This->isImplicit());
          continue;
        }
        auto* DRE = dyn_cast_or_null<DeclRefExpr>(Child);
        if (!DRE)
          continue;
        // A reference copied from another lambda keeps what it referred to
        // there; Sema says what it is here.
        auto* VD = dyn_cast<VarDecl>(DRE->getDecl());
        if (!VD || !Caps.contains(VD))
          continue;
        // BuildDeclRef produces a bare reference for a local (it never
        // name-qualifies one), which is what lets Sema capture it here.
        Child = Caps.m_V.BuildDeclRef(VD, clad_compat::nullNNS(),
                                      DRE->getValueKind());
      }
      return true;
    }
  };
  CapRefRebuilder CR(*this);
  for (Stmt* S : Body)
    CR.TraverseStmt(S);
}

#if CLANG_VERSION_MAJOR >= 17
/// Register every (original, clone) VarDecl pair, at any depth.
static void
registerClonedDecls(const Stmt* Orig, Stmt* Cloned,
                    std::unordered_map<const VarDecl*, VarDecl*>& Repls) {
  if (!Orig || !Cloned)
    return;
  if (const auto* DS = dyn_cast<DeclStmt>(Orig))
    if (auto* ClonedDS = dyn_cast<DeclStmt>(Cloned)) {
      auto O = DS->decl_begin();
      auto C = ClonedDS->decl_begin();
      for (; O != DS->decl_end() && C != ClonedDS->decl_end(); ++O, ++C)
        if (const auto* OVD = dyn_cast<VarDecl>(*O))
          if (auto* CVD = dyn_cast<VarDecl>(*C))
            if (OVD != CVD)
              Repls[OVD] = CVD;
    }
  auto O = Orig->child_begin();
  auto C = Cloned->child_begin();
  for (; O != Orig->child_end() && C != Cloned->child_end(); ++O, ++C)
    registerClonedDecls(*O, *C, Repls);
}
#endif // CLANG_VERSION_MAJOR >= 17

#if CLANG_VERSION_MAJOR > 16
VisitorBase::LambdaBuilder::LambdaBuilder(VisitorBase& V)
    : m_V(V), m_DS(m_AttrFactory),
      m_D(m_DS, clang::ParsedAttributesView::none(),
          clang::DeclaratorContext::LambdaExpr),
      m_SaveContext(V.m_Sema.CurContext), m_SaveDerivative(V.m_Derivative),
      m_SaveFnScope(V.m_DerivativeFnScope) {
  m_Intro.Default = LCD_None;
  m_Intro.Range.setBegin(noLoc);
  m_Intro.Range.setEnd(noLoc);
}

void VisitorBase::LambdaBuilder::cloneCaptures(const LambdaExpr* LE) {
  // Every capture is spelled out, an implicit one too: the body is rebuilt
  // from references made outside the new closure, from which Sema infers no
  // capture for a capture-default to cover.
  m_Intro.Default = clang::LCD_None;
  for (const clang::LambdaCapture& Cap : LE->captures())
    m_Intro.addCapture(Cap.getCaptureKind(), Cap.getLocation(),
                       Cap.getCapturedVar()->getIdentifier(), /*EllipsisLoc=*/
                       noLoc, clang::LambdaCaptureInitKind::NoInit,
                       clang::ExprResult(), clang::ParsedType(), SourceRange());
  m_Intro.Range.setBegin(LE->getBeginLoc());
  m_Intro.Range.setEnd(LE->getEndLoc());
}

void VisitorBase::LambdaBuilder::start(const LambdaExpr* LE,
                                       QualType CallOpType,
                                       ParamBuilder BuildParams,
                                       QualType TrailingReturnType) {
  DeclContext* DC =
      LE ? LE->getCallOperator()->getDeclContext() : m_V.m_Derivative;

  m_Started = true;
  m_V.beginScope(Scope::LambdaScope | Scope::DeclScope |
                 Scope::FunctionDeclarationScope |
                 Scope::FunctionPrototypeScope);
  m_V.m_Sema.PushLambdaScope();
  m_V.m_Sema.ActOnLambdaExpressionAfterIntroducer(m_Intro,
                                                  m_V.getCurrentScope());

  m_V.beginScope(Scope::FunctionPrototypeScope |
                 Scope::FunctionDeclarationScope | Scope::DeclScope);

  m_V.m_Sema.ActOnLambdaClosureQualifiers(m_Intro, noLoc);

  sema::LambdaScopeInfo* LSI = m_V.m_Sema.getCurLambda();
  LSI->CallOperator->setType(CallOpType);
  m_V.m_Sema.PushDeclContext(m_V.getCurrentScope(), LSI->CallOperator);
  LSI->Lambda->setDeclContext(DC);
  m_V.m_Derivative = LSI->CallOperator;

  llvm::SmallVector<ParmVarDecl*, 8> params;
  BuildParams(params);
  m_V.m_Derivative->setBody(m_V.MakeCompoundStmt({}));

  m_V.m_Sema.PopDeclContext();

  llvm::SmallVector<DeclaratorChunk::ParamInfo> ParamInfoLambda;
  for (auto* PVD : params)
    ParamInfoLambda.emplace_back(PVD->getIdentifier(), PVD->getLocation(), PVD,
                                 nullptr);
  // ActOnStartOfLambdaDefinition uses LParenLoc.isValid() to flip
  // LSI->ExplicitParams; an invalid LParen makes the lambda print without
  // its parameter list (`[]{...}` instead of `[](T x){...}`).
  SourceLocation paramListLoc = utils::GetValidSLoc(m_V.m_Sema);
  ParsedType TrailingReturn;
  SourceLocation TrailingReturnLoc;
  if (!TrailingReturnType.isNull()) {
    TrailingReturn = m_V.m_Sema.CreateParsedType(
        TrailingReturnType, m_V.m_Context.getTrivialTypeSourceInfo(
                                TrailingReturnType, paramListLoc));
    TrailingReturnLoc = paramListLoc;
  }
  m_D.AddTypeInfo(DeclaratorChunk::getFunction(
                      /*hasProto=*/true,
                      /*isAmbiguous=*/false,
                      /*LParenLoc=*/paramListLoc,
                      /*Params=*/ParamInfoLambda.data(),
                      /*NumParams=*/ParamInfoLambda.size(),
                      /*EllipsisLoc=*/SourceLocation(),
                      /*RParenLoc=*/paramListLoc,
                      /*RefQualifierIsLValueRef=*/true,
                      /*RefQualifierLoc=*/SourceLocation(),
                      /*MutableLoc=*/SourceLocation(),
                      /*ESpecType=*/EST_None,
                      /*ESpecRange=*/SourceRange(),
                      /*Exceptions=*/nullptr,
                      /*ExceptionRanges=*/nullptr,
                      /*NumExceptions=*/0,
                      /*NoexceptExpr=*/nullptr,
                      /*ExceptionSpecTokens=*/nullptr,
                      /*DeclsInPrototype=*/{},
                      /*LocalRangeBegin=*/noLoc,
                      /*LocalRangeEnd=*/noLoc,
                      /*Declarator=*/m_D,
                      /*TrailingReturnType=*/TrailingReturn,
                      /*TrailingReturnTypeLoc=*/TrailingReturnLoc),
                  /*EndLoc=*/SourceLocation());

  m_V.m_Sema.ActOnLambdaClosureParameters(m_V.getCurrentScope(),
                                          ParamInfoLambda);

  m_V.beginScope(Scope::BlockScope | Scope::FnScope | Scope::DeclScope |
                 Scope::CompoundStmtScope);

  m_V.m_Sema.ActOnStartOfLambdaDefinition(
      m_Intro, m_D,
      clad_compat::Sema_ActOnStartOfLambdaDefinition_ScopeOrDeclSpec(
          m_V.getCurrentScope(), m_DS));

  m_V.m_DerivativeFnScope = m_V.getCurrentScope();
}

Expr* VisitorBase::LambdaBuilder::finish(Stmt* Body) {
  // clad builds the body's statements without ActOnFinishFullExpr, so a
  // temporary in one of them, such as an argument bound to an rvalue
  // reference, leaves a cleanup pending. ActOnLambdaExpr finishes the
  // closure body and asserts there is none; clad's other derived functions
  // are not finished through Sema and never check.
  m_V.m_Sema.DiscardCleanupsInEvaluationContext();
  Expr* lambda =
      m_V.m_Sema
          .ActOnLambdaExpr(
              noLoc,
              Body /*,*/
                  CLAD_COMPAT_CLANG17_ActOnLambdaExpr_getCurrentScope_ExtraParam(
                      m_V))
          .get();
  m_V.endScope();
  m_V.endScope();
  m_V.endScope();
  m_Finished = true;
  return lambda;
}

VisitorBase::LambdaBuilder::~LambdaBuilder() {
  assert((!m_Started || m_Finished) &&
         "a started LambdaBuilder must be finished: bailing out in between "
         "leaves Sema inside a half-built closure");
}
#endif // CLANG_VERSION_MAJOR > 16

Expr* VisitorBase::buildClonedLambda(const LambdaExpr* LE) {
  // A primal copy needs a *fresh* closure type; a plain StmtClone reuses the
  // original closure, so two clones share the operator() body -- both a
  // one-parent-per-node violation and a node shared with the primal lambda.
  //
  // We reproduce by-copy/by-ref captures of a named variable, implicit ones
  // spelled out (each re-resolves by name against the derivative's in-scope
  // copy). Fall back to a plain clone for the kinds we do not model --
  // this-capture, init-capture, and packs.
#if CLANG_VERSION_MAJOR < 17
  // Lambda differentiation is unsupported below clang-17 (Lambdas.C is
  // UNSUPPORTED there) and the Sema lambda-introduction entry points used
  // below do not exist yet. Never reached; keep the source compilable.
  return cast<Expr>(Clone(LE));
#else
  for (const clang::LambdaCapture& Cap : LE->captures())
    if (!Cap.capturesVariable() || (Cap.getCaptureKind() != clang::LCK_ByCopy &&
                                    Cap.getCaptureKind() != clang::LCK_ByRef))
      return cast<Expr>(Clone(LE));

  const CXXMethodDecl* CallOp = LE->getCallOperator();

  LambdaBuilder LBuilder(*this);
  LBuilder.cloneCaptures(LE);
  LBuilder.start(
      LE, CallOp->getType(), [&](llvm::SmallVectorImpl<ParmVarDecl*>& params) {
        for (const ParmVarDecl* PVD : CallOp->parameters()) {
          IdentifierInfo* II = PVD->getIdentifier();
          if (!PVD->getDeclName())
            II = CreateUniqueIdentifier("arg");
          auto* newPVD = CloneParmVarDecl(PVD, II, /*pushOnScopeChains=*/true,
                                          /*cloneDefaultArg=*/false);
          // Remap references in the cloned body to the new
          // parameters.
          m_DeclReplacements[PVD] = newPVD;
          params.push_back(newPVD);
        }
      });

  beginBlock();
  const auto* Body = cast<CompoundStmt>(CallOp->getBody());
  for (Stmt* S : Body->body()) {
    // A nested lambda declaration must itself get a fresh closure (otherwise
    // the recursive clone would reuse it). Rebuild it here -- its own body is
    // remapped inside the recursive call -- and record the new variable so
    // later references in this body are updated to it.
    if (auto* InnerDS = dyn_cast<DeclStmt>(S))
      if (InnerDS->isSingleDecl())
        if (auto* InnerVD = dyn_cast<VarDecl>(InnerDS->getSingleDecl()))
          if (const Expr* InnerInit = InnerVD->getInit())
            if (const auto* InnerLE =
                    dyn_cast<LambdaExpr>(InnerInit->IgnoreImplicit())) {
              Expr* ClonedInner = buildClonedLambda(InnerLE);
              QualType AutoTy = m_Context.getAutoDeductType();
              TypeSourceInfo* TSI = m_Context.getTrivialTypeSourceInfo(AutoTy);
              VarDecl* NewInnerVD =
                  BuildVarDecl(AutoTy, InnerVD->getNameAsString(), ClonedInner,
                               InnerVD->isDirectInit(), TSI);
              m_DeclReplacements[InnerVD] = NewInnerVD;
              addToCurrentBlock(BuildDeclStmt(NewInnerVD));
              continue;
            }
    // Clone the statement and remap references to the lambda's own
    // parameters (and rebuilt inner lambdas). ReferencesUpdater only admits
    // declarations enclosed by the "function" it is given, so key it on the
    // lambda's call operator rather than m_DiffReq.Function (the outer
    // function being differentiated), which would reject the lambda locals.
    Stmt* clonedS = CloneNode(S);
    // Register before remapping, so references in this statement update too.
    registerClonedDecls(S, clonedS, m_DeclReplacements);
    utils::ReferencesUpdater up(m_Sema, getCurrentScope(), CallOp,
                                m_DeclReplacements);
    up.TraverseStmt(clonedS);
    addToCurrentBlock(clonedS);
  }
  return LBuilder.finish(endBlock());
#endif // CLANG_VERSION_MAJOR < 17
}

clang::VarDecl*
VisitorBase::getEnclosingCaptureClone(const clang::VarDecl* capVD) {
  // A lambda captures a local of the function being differentiated, and
  // every one of those is cloned before the lambda is reached.
  auto it = m_DeclReplacements.find(capVD);
  bool cloned = it != m_DeclReplacements.end() && it->second;
  assert(cloned && "a captured variable has no clone in the derivative");
  // Unreachable once the assert holds; a release build binds the variable
  // itself rather than nothing.
  return cloned ? it->second
                : const_cast<VarDecl*>(capVD); // NOLINT(*-const-cast)
}

const FieldDecl* VisitorBase::getCaptureField(const LambdaExpr* LE,
                                              const VarDecl* capVD) {
  // The closure has one field per capture, in capture order -- what
  // getCaptureFields walks too, with a map keyed by VarDecl before clang 16.
  auto Field = LE->getLambdaClass()->field_begin();
  for (const LambdaCapture& Cap : LE->captures()) {
    if (Cap.capturesVariable() && Cap.getCapturedVar() == capVD)
      return *Field;
    ++Field;
  }
  return nullptr;
}

Expr* VisitorBase::getLambdaDerivative(const VarDecl* VD) {
  auto clone = m_DeclReplacements.find(VD);
  if (clone == m_DeclReplacements.end())
    return nullptr;
  auto derived = m_Variables.find(clone->second);
  return derived == m_Variables.end() ? nullptr : buildAdjoint(derived->second);
}

std::pair<clang::Expr*, clang::Expr*> VisitorBase::getCaptureValueAndAdjoint(
    const clang::LambdaExpr* LE, const clang::VarDecl* capVD, bool isByCopy) {
  // A by-copy capture reads the snapshot taken where its lambda is created.
  auto snapIt = isByCopy
                    ? m_LambdaCaptureSnapshots.find(getCaptureField(LE, capVD))
                    : m_LambdaCaptureSnapshots.end();
  if (snapIt != m_LambdaCaptureSnapshots.end()) {
    VarDecl* snap = snapIt->second;
    auto dit = m_Variables.find(snap);
    assert(dit != m_Variables.end() && "snapshot adjoint must exist");
    return {BuildDeclRef(snap), buildAdjoint(dit->second)};
  }
  VarDecl* encVD = getEnclosingCaptureClone(capVD);
  auto it = m_Variables.find(encVD);
  assert(it != m_Variables.end() && "captured variable has no tracked adjoint");
  return {BuildDeclRef(encVD), buildAdjoint(it->second)};
}

/// Whether \p Target is inside a loop of \p S. A lambda's body starts afresh:
/// a loop around the lambda runs the lambda, not the statements inside it.
static bool isInsideLoop(const Stmt* S, const Stmt* Target, bool InLoop) {
  if (S == Target)
    return InLoop;
  bool Loop = !isa<LambdaExpr>(S) &&
              (InLoop || isa<ForStmt, WhileStmt, DoStmt, CXXForRangeStmt>(S));
  return llvm::any_of(S->children(), [&](const Stmt* Child) {
    return Child && isInsideLoop(Child, Target, Loop);
  });
}

bool VisitorBase::diagnoseUnsupportedLambda(const clang::LambdaExpr* LE) {
  const CXXMethodDecl* CallOp = LE->getCallOperator();
  if (!CallOp)
    return false;
  // The function, or the lambda, LE is written in.
  const auto* Enclosing =
      cast<FunctionDecl>(LE->getLambdaClass()->getDeclContext());
  for (const LambdaCapture& Cap : LE->captures())
    if (!Cap.capturesVariable()) {
      diag(DiagnosticsEngine::Error, Cap.getLocation(),
           "differentiation of a lambda that captures %select{'this'|a "
           "variable-length array}0 is not supported yet")
          << !Cap.capturesThis();
      return true;
    }
  if (LE->capture_size() && m_DiffReq.Mode == DiffMode::vector_forward_mode) {
    diag(DiagnosticsEngine::Error, CallOp->getLocation(),
         "vector-mode differentiation of a lambda with captures is not "
         "supported yet");
    return true;
  }
  if (LE->capture_size() &&
      isInsideLoop(Enclosing->getBody(), LE, /*InLoop=*/false)) {
    diag(DiagnosticsEngine::Error, CallOp->getLocation(),
         "differentiation of a lambda with captures created inside a loop is "
         "not supported yet");
    return true;
  }
  if (!CallOp->isConst()) {
    diag(DiagnosticsEngine::Error, CallOp->getLocation(),
         "differentiation of a mutable lambda is not supported: its captures "
         "change between calls");
    return true;
  }
  return false;
}

/// The arguments \p CE wrote for the function template it calls, or false
/// when one is of a kind that cannot be passed again as it is.
static bool getWrittenTemplateArgs(Sema& S, const CallExpr* CE,
                                   TemplateArgumentListInfo& TLI) {
  const auto* DRE =
      dyn_cast<DeclRefExpr>(CE->getCallee()->IgnoreParenImpCasts());
  const TemplateArgumentList* TAL =
      CE->getDirectCallee()->getTemplateSpecializationArgs();
  for (unsigned I = 0, N = DRE ? DRE->getNumTemplateArgs() : 0; I < N; ++I) {
    const TemplateArgument& Arg = TAL->get(I);
    if (Arg.getKind() == TemplateArgument::Type)
      TLI.addArgument(
          TemplateArgumentLoc(Arg, S.getASTContext().getTrivialTypeSourceInfo(
                                       Arg.getAsType(), TLI.getLAngleLoc())));
    else if (Arg.getKind() == TemplateArgument::Integral)
      TLI.addArgument(S.getTrivialTemplateArgumentLoc(
          Arg, Arg.getIntegralType(), TLI.getLAngleLoc()));
    else
      return false;
  }
  return true;
}

/// Whether \p FD can change a differentiable value only through the callables
/// it is passed: it names no variable that outlives it and holds a
/// floating-point number, and calls only those callables and functions that
/// pass them on, which are held to the same. \p Seen holds the functions
/// already checked, such as one that passes its callable on to itself.
static bool onlyCallsBack(const FunctionDecl* FD,
                          llvm::SmallPtrSetImpl<const FunctionDecl*>& Seen) {
  if (!Seen.insert(FD).second)
    return true;
  const Stmt* Body = FD->getBody();
  if (!Body)
    return false;
  struct Checker : RecursiveASTVisitor<Checker> {
    const FunctionDecl* m_FD;
    llvm::SmallPtrSetImpl<const FunctionDecl*>& m_Seen;
    bool OK = true;
    Checker(const FunctionDecl* FD,
            llvm::SmallPtrSetImpl<const FunctionDecl*>& Seen)
        : m_FD(FD), m_Seen(Seen) {}
    static bool isMoveOrForward(const CallExpr* CE) {
      const FunctionDecl* Callee = CE->getDirectCallee();
      return Callee && Callee->isInStdNamespace() && CE->getNumArgs() == 1 &&
             Callee->getIdentifier() &&
             (Callee->getName() == "move" || Callee->getName() == "forward");
    }
    // \p E is a callable the function was passed: the parameter, a copy of
    // it, or the parameter moved or forwarded.
    bool isPassedCallable(const Expr* E) const {
      E = E->IgnoreImplicit();
      if (const auto* CCE = dyn_cast<CXXConstructExpr>(E))
        if (CCE->getNumArgs() == 1)
          E = CCE->getArg(0)->IgnoreImplicit();
      if (const auto* CE = dyn_cast<CallExpr>(E))
        if (isMoveOrForward(CE))
          E = CE->getArg(0);
      const auto* DRE = dyn_cast<DeclRefExpr>(E->IgnoreParenImpCasts());
      const auto* PVD = DRE ? dyn_cast<ParmVarDecl>(DRE->getDecl()) : nullptr;
      return PVD && PVD->getDeclContext() == m_FD;
    }
    bool VisitDeclRefExpr(DeclRefExpr* DRE) {
      const auto* VD = dyn_cast<VarDecl>(DRE->getDecl());
      OK = !VD || !VD->hasGlobalStorage() ||
           utils::HoldsNoFloatingPoint(VD->getType());
      return OK;
    }
    bool VisitCallExpr(CallExpr* CE) {
      // f(i), with f a callable the function was passed.
      if (const auto* OCE = dyn_cast<CXXOperatorCallExpr>(CE))
        if (OCE->getOperator() == OO_Call && isPassedCallable(OCE->getArg(0)))
          return true;
      if (isPassedCallable(CE->getCallee()) ||
          (isMoveOrForward(CE) && isPassedCallable(CE->getArg(0))))
        return true;
      // g(f, i), a function the callable is passed on to, or one that takes
      // and gives no floating-point value, such as the conversion of
      // std::integral_constant to its value; it is held to the same.
      const FunctionDecl* Callee = CE->getDirectCallee();
      QualType Ret = Callee ? Callee->getReturnType() : QualType();
      const auto* MCE = dyn_cast<CXXMemberCallExpr>(CE);
      OK = Callee && (Ret->isVoidType() || utils::HoldsNoFloatingPoint(Ret)) &&
           (!MCE || passes(MCE->getImplicitObjectArgument())) &&
           llvm::all_of(CE->arguments(),
                        [&](const Expr* Arg) { return passes(Arg); }) &&
           onlyCallsBack(Callee, m_Seen);
      return OK;
    }
    // An object of a type that holds no floating-point number, built and
    // destroyed by functions held to the same, or a copy of the callable.
    bool VisitCXXConstructExpr(CXXConstructExpr* CCE) {
      if (CCE->getNumArgs() == 1 && isPassedCallable(CCE->getArg(0)))
        return true;
      const CXXConstructorDecl* Ctor = CCE->getConstructor();
      const CXXRecordDecl* RD = Ctor->getParent();
      const CXXDestructorDecl* Dtor = RD->getDestructor();
      OK = utils::HoldsNoFloatingPoint(CCE->getType()) &&
           llvm::all_of(CCE->arguments(),
                        [&](const Expr* Arg) { return passes(Arg); }) &&
           (Ctor->isTrivial() || onlyCallsBack(Ctor, m_Seen)) &&
           (!Dtor || Dtor->isTrivial() || onlyCallsBack(Dtor, m_Seen));
      return OK;
    }
    bool passes(const Expr* Arg) const {
      return isPassedCallable(Arg) ||
             utils::HoldsNoFloatingPoint(Arg->getType());
    }
  } Check(FD, Seen);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  Check.TraverseStmt(const_cast<Stmt*>(Body));
  // A constructor runs its member initializers before its body.
  if (const auto* Ctor = dyn_cast<CXXConstructorDecl>(FD))
    for (const CXXCtorInitializer* Init : Ctor->inits())
      if (Check.OK)
        Check.TraverseStmt(Init->getInit());
  return Check.OK;
}

bool VisitorBase::diagnoseUnsupportedCallback(const CallExpr* CE,
                                              const LambdaExpr* LE) {
  auto refuse = [&](llvm::StringRef Why) {
    diag(DiagnosticsEngine::Error, LE->getBeginLoc(),
         "differentiation of a call to %0 that passes a lambda %1 is not "
         "supported yet")
        << CE->getDirectCallee()->getDeclName() << Why;
    return true;
  };
#if CLANG_VERSION_MAJOR < 17
  return refuse("before clang 17");
#else
  const CXXMethodDecl* CallOp = LE->getCallOperator();
  if (LE->isGenericLambda())
    return refuse("with an 'auto' parameter");
  if (!CallOp->getReturnType()->isVoidType())
    return refuse("that returns a value");
  for (const LambdaCapture& Cap : LE->captures())
    if (!Cap.capturesVariable() || LE->isInitCapture(&Cap))
      return refuse("that captures 'this' or an initializer");
  // The callee passes nothing differentiable, so a parameter has no
  // derivative to give or take.
  for (const ParmVarDecl* PVD : CallOp->parameters()) {
    QualType T = PVD->getType();
    if (!utils::HoldsNoFloatingPoint(T) ||
        (T->isReferenceType() && !T.getNonReferenceType().isConstQualified()))
      return refuse("whose parameter is not an integer");
  }
  // A return ends one call, which a reverse loop cannot tell apart from the
  // end of the body.
  struct ReturnFinder : RecursiveASTVisitor<ReturnFinder> {
    bool Found = false;
    bool VisitReturnStmt(ReturnStmt*) {
      Found = true;
      return false;
    }
    bool TraverseLambdaExpr(LambdaExpr*) { return true; }
  } Returns;
  Returns.TraverseStmt(CallOp->getBody());
  if (Returns.Found)
    return refuse("that returns early");
  // The body of an inner one would be differentiated inside the closure built
  // for the outer one, whose parameters it cannot name.
  struct CallbackFinder : RecursiveASTVisitor<CallbackFinder> {
    bool Found = false;
    bool VisitCallExpr(CallExpr* CE) {
      Found = utils::GetCallbackLambda(CE) != nullptr;
      return !Found;
    }
  } Callbacks;
  Callbacks.TraverseStmt(CallOp->getBody());
  if (Callbacks.Found)
    return refuse("that itself passes a lambda to a function template");
  // Only the lambda's body is differentiated: the function itself must not
  // read or change a floating-point value of its own.
  llvm::SmallPtrSet<const FunctionDecl*, 8> Seen;
  if (!onlyCallsBack(CE->getDirectCallee(), Seen))
    return refuse("to a function that does more than call it");
  SourceLocation Loc = utils::GetValidSLoc(m_Sema);
  TemplateArgumentListInfo TLI(Loc, Loc);
  if (!getWrittenTemplateArgs(m_Sema, CE, TLI))
    return refuse("to a function with this kind of template argument");
  return false;
#endif
}

Expr* VisitorBase::buildCallbackClosure(
    const LambdaExpr* LE,
    llvm::function_ref<void(llvm::ArrayRef<ParmVarDecl*>)> BuildBody) {
#if CLANG_VERSION_MAJOR < 17
  return nullptr;
#else
  const CXXMethodDecl* CallOp = LE->getCallOperator();
  llvm::SmallVector<ParmVarDecl*, 2> Params;
  LambdaBuilder LBuilder(*this);
  LBuilder.captureByRef();
  LBuilder.start(/*LE=*/nullptr, CallOp->getType(),
                 [&](llvm::SmallVectorImpl<ParmVarDecl*>& NewParams) {
                   for (const ParmVarDecl* PVD : CallOp->parameters())
                     NewParams.push_back(CloneParmVarDecl(
                         PVD,
                         CreateUniqueIdentifier(
                             PVD->getDeclName() ? PVD->getName() : "arg"),
                         /*pushOnScopeChains=*/true,
                         /*cloneDefaultArg=*/false));
                   Params.assign(NewParams.begin(), NewParams.end());
                 });
  beginBlock();
  BuildBody(Params);
  LambdaCaptures Free(*this);
  Free.collect(getCurrentBlock());
  Free.resolve(getCurrentBlock());
  return LBuilder.finish(endBlock());
#endif
}

Expr* VisitorBase::buildCallWithCallback(const CallExpr* CE,
                                         const LambdaExpr* LE, Expr* Closure) {
  llvm::SmallVector<Expr*, 4> Args;
  for (const Expr* Arg : CE->arguments()) {
    if (isa<CXXDefaultArgExpr>(Arg))
      break;
    Args.push_back(utils::GetLambdaArg(Arg) == LE ? Closure : Clone(Arg));
  }
  SourceLocation Loc = utils::GetValidSLoc(m_Sema);
  TemplateArgumentListInfo TLI(Loc, Loc);
  getWrittenTemplateArgs(m_Sema, CE, TLI);
  FunctionTemplateDecl* FTD = CE->getDirectCallee()->getPrimaryTemplate();
  LookupResult R(m_Sema, FTD->getDeclName(), noLoc, Sema::LookupOrdinaryName);
  R.addDecl(FTD);
  R.resolveKind();
  CXXScopeSpec CSS;
  utils::BuildNNS(m_Sema, FTD->getDeclContext(), CSS);
  Expr* Callee = utils::BuildDeclarationNameExpr(m_Sema, CSS, R, /*ADL=*/false,
                                                 TLI.size() ? &TLI : nullptr);
  return m_Sema.ActOnCallExpr(getCurrentScope(), Callee, noLoc, Args, noLoc)
      .get();
}
} // end namespace clad
