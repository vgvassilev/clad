#include "EntryAdjointChecker.h"

#include "clad/Differentiator/CladUtils.h"

#include "clang/AST/APValue.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace clang;

namespace clad {
namespace {

/// A variable, or a part of one reached through `.` and constant `[N]`:
/// {gh, ""}, {gh, ".p"}, {gpa, "[1]"}. An element whose index is not a
/// constant is `[*]`, which stands for every element.
using Key = std::pair<const ValueDecl*, std::string>;

std::string nameOf(const Key& K) {
  return K.first->getNameAsString() + K.second;
}

/// Why the adjoint of a pointer has no value.
struct Origin {
  enum class Kind : std::uint8_t {
    /// It still holds what the global pointed to when the gradient started.
    Entry,
    /// A call may have reassigned it without updating its adjoint.
    CalleeAssigns,
    /// It was passed by reference, which does the same.
    PassedByReference,
    /// Its address was taken; what is written through that is not followed.
    AddressTaken,
  };
  Kind K = Kind::Entry;
  /// The global it came from and the part of it: `gh.p`.
  const VarDecl* Global = nullptr;
  std::string Name;
  /// For CalleeAssigns and PassedByReference, the callee; for those and for
  /// AddressTaken, where it happened.
  const FunctionDecl* Callee = nullptr;
  SourceLocation Where;
  /// The first variable it was copied into, and where.
  std::string CopyName;
  SourceLocation CopyLoc;
};

/// What is known about the adjoints of the pointers a function reads, at one
/// point of it. A part of a global is unknown unless an entry says otherwise;
/// the part of anything else is known.
struct State {
  /// No path reaches here. Such a state knows everything, and a join ignores
  /// it.
  bool Dead = false;
  /// nullopt: known. An entry stands for every part below it that has no
  /// entry of its own.
  std::map<Key, std::optional<Origin>> Parts;
};

/// One place where the derivative would use an adjoint that has no value.
struct Finding {
  /// Passing the pointer to a call whose derivative clad cannot see into,
  /// rather than accessing through it.
  bool IsPass = false;
  /// A use of a global reference, which has no adjoint at all.
  const VarDecl* Reference = nullptr;
  SourceLocation Loc;
  /// The pointer as the access spells it, and the callee it is passed to.
  std::string Pointer;
  const FunctionDecl* PassedTo = nullptr;
  Origin O;
  /// The calls that lead here from the differentiated function, innermost
  /// first.
  llvm::SmallVector<std::pair<const FunctionDecl*, SourceLocation>, 4> Chain;
};

/// Whether clad tracks the derivative of a value of type \p T: it is, or
/// holds, numbers. Characters and flags are not numbers to differentiate.
bool carries(QualType T) {
  T = T.getNonReferenceType().getCanonicalType();
  while (T->isArrayType() || T->isPointerType())
    T = utils::GetValueType(T).getCanonicalType();
  if (T->isVoidType() || T->isAnyCharacterType() || T->isBooleanType())
    return false;
  return utils::IsDifferentiableType(T);
}

bool isPointerToCarried(QualType T) {
  T = T.getNonReferenceType().getCanonicalType();
  return T->isPointerType() && carries(T->getPointeeType());
}

bool isToBoolean(CastKind CK) {
  return CK == CK_PointerToBoolean || CK == CK_IntegralToBoolean ||
         CK == CK_FloatingToBoolean || CK == CK_MemberPointerToBoolean ||
         CK == CK_FloatingComplexToBoolean || CK == CK_IntegralComplexToBoolean;
}

/// A cast after which an lvalue is still the same object, or a pointer still
/// points into the same one.
bool keepsObject(CastKind CK) {
  return CK == CK_NoOp || CK == CK_BitCast || CK == CK_DerivedToBase ||
         CK == CK_UncheckedDerivedToBase || CK == CK_BaseToDerived ||
         CK == CK_AddressSpaceConversion;
}

/// The globals a function may assign, as keys, and whether it assigns a
/// pointer member of the object it is called on -- directly or through what
/// it calls. Only the forward pass of the caller sees these assignments, and
/// it runs the primal function, which leaves their adjoints as they were.
struct Effects {
  std::set<Key> Globals;
  bool AssignsThisPointer = false;
  /// The parameters, by index, through which it may write what the caller
  /// passed by reference or by address.
  std::set<unsigned> WritesThroughParam;
  /// Whether its body was not seen, so that it may write through any of them.
  bool Opaque = false;
};

struct Context {
  Context(Sema& S, llvm::function_ref<CallDerivative(const FunctionDecl*)> How,
          FileID TopFile)
      : S(S), How(How), TopFile(TopFile) {}
  Sema& S;
  llvm::function_ref<CallDerivative(const FunctionDecl*)> How;
  FileID TopFile;
  /// How many function bodies the check may still walk.
  unsigned Budget = 256;
  llvm::SmallPtrSet<const FunctionDecl*, 8> Active;
  std::map<const FunctionDecl*, Effects> EffectsOf;
  std::vector<Finding> Findings;
};

/// The key \p E names, if it names a part of a variable.
std::optional<Key> keyOf(const Expr* E, const ASTContext& C,
                         const std::map<const VarDecl*, Key>* Aliases = nullptr,
                         const std::optional<Key>* This = nullptr) {
  E = E->IgnoreParens();
  if (const auto* ICE = dyn_cast<ImplicitCastExpr>(E))
    if (keepsObject(ICE->getCastKind()))
      return keyOf(ICE->getSubExpr(), C, Aliases, This);
  if (const auto* DRE = dyn_cast<DeclRefExpr>(E)) {
    const auto* VD = dyn_cast<VarDecl>(DRE->getDecl());
    if (!VD)
      return std::nullopt;
    if (Aliases) {
      auto It = Aliases->find(VD);
      if (It != Aliases->end())
        return It->second;
    }
    if (VD->getType()->isReferenceType())
      return std::nullopt;
    return Key{VD, ""};
  }
  if (const auto* ME = dyn_cast<MemberExpr>(E)) {
    if (const auto* VD = dyn_cast<VarDecl>(ME->getMemberDecl()))
      return Key{VD, ""}; // a static data member
    const auto* FD = dyn_cast<FieldDecl>(ME->getMemberDecl());
    if (!FD)
      return std::nullopt;
    std::optional<Key> Base;
    if (!ME->isArrow())
      Base = keyOf(ME->getBase(), C, Aliases, This);
    else if (This && *This &&
             isa<CXXThisExpr>(ME->getBase()->IgnoreParenImpCasts()))
      Base = **This;
    if (!Base)
      return std::nullopt;
    Base->second += "." + FD->getNameAsString();
    return Base;
  }
  if (const auto* ASE = dyn_cast<ArraySubscriptExpr>(E)) {
    const auto* Decay =
        dyn_cast<ImplicitCastExpr>(ASE->getBase()->IgnoreParens());
    if (!Decay || Decay->getCastKind() != CK_ArrayToPointerDecay)
      return std::nullopt;
    std::optional<Key> Base = keyOf(Decay->getSubExpr(), C, Aliases, This);
    if (!Base)
      return std::nullopt;
    Expr::EvalResult R;
    if (!ASE->getIdx()->isValueDependent() &&
        ASE->getIdx()->EvaluateAsInt(R, C))
      Base->second += "[" + std::to_string(R.Val.getInt().getExtValue()) + "]";
    else
      Base->second += "[*]";
    return Base;
  }
  return std::nullopt;
}

bool isGlobal(const ValueDecl* D) {
  const auto* VD = dyn_cast<VarDecl>(D);
  return VD && VD->isFileVarDecl();
}

/// Collects a function's Effects, following its calls.
class EffectsCollector : public RecursiveASTVisitor<EffectsCollector> {
  Context& m_C;
  Effects& m_E;
  const ASTContext& m_AST;

  /// \p E is written, or may be through what it is handed to: note the
  /// parameter of reference or pointer type it is reached through.
  void throughParam(const Expr* E) {
    while (true) {
      E = E->IgnoreParenImpCasts();
      if (const auto* UO = dyn_cast<UnaryOperator>(E)) {
        if (UO->getOpcode() != UO_Deref && UO->getOpcode() != UO_AddrOf)
          return;
        E = UO->getSubExpr();
      } else if (const auto* ME = dyn_cast<MemberExpr>(E)) {
        E = ME->getBase();
      } else if (const auto* ASE = dyn_cast<ArraySubscriptExpr>(E)) {
        E = ASE->getBase();
      } else {
        break;
      }
    }
    const auto* DRE = dyn_cast<DeclRefExpr>(E);
    const auto* PVD = DRE ? dyn_cast<ParmVarDecl>(DRE->getDecl()) : nullptr;
    if (PVD &&
        (PVD->getType()->isReferenceType() || PVD->getType()->isPointerType()))
      m_E.WritesThroughParam.insert(PVD->getFunctionScopeIndex());
  }

  void assigned(const Expr* LHS) {
    throughParam(LHS);
    QualType T = LHS->getType().getCanonicalType();
    if (!T->isPointerType() && !T->isRecordType() && !T->isArrayType())
      return;
    std::optional<Key> K = keyOf(LHS, m_AST);
    if (K && isGlobal(K->first)) {
      m_E.Globals.insert(*K);
      return;
    }
    const auto* ME = dyn_cast<MemberExpr>(LHS->IgnoreParenImpCasts());
    if (ME && ME->isArrow() &&
        isa<CXXThisExpr>(ME->getBase()->IgnoreParenImpCasts()) &&
        T->isPointerType())
      m_E.AssignsThisPointer = true;
  }

public:
  EffectsCollector(Context& C, Effects& E)
      : m_C(C), m_E(E), m_AST(C.S.getASTContext()) {}
  bool shouldVisitImplicitCode() const { return true; }
  bool TraverseLambdaExpr(LambdaExpr*) { return true; }
  bool VisitVarDecl(VarDecl* VD) {
    // A reference or pointer bound to a parameter may be written through.
    if (VD->getInit() &&
        (VD->getType()->isReferenceType() || VD->getType()->isPointerType()))
      throughParam(VD->getInit());
    return true;
  }
  bool VisitBinaryOperator(BinaryOperator* BO) {
    if (BO->isAssignmentOp())
      assigned(BO->getLHS());
    return true;
  }
  bool VisitUnaryOperator(UnaryOperator* UO) {
    if (UO->isIncrementDecrementOp() || UO->getOpcode() == UO_AddrOf)
      assigned(UO->getSubExpr());
    return true;
  }
  bool VisitCXXOperatorCallExpr(CXXOperatorCallExpr* CE) {
    if (CE->isAssignmentOp() && CE->getNumArgs())
      assigned(CE->getArg(0));
    return true;
  }
  bool VisitCallExpr(CallExpr* CE);
  bool VisitCXXConstructExpr(CXXConstructExpr* CE);
};

const Effects& effectsOf(Context& C, const FunctionDecl* FD) {
  FD = FD->getCanonicalDecl();
  auto It = C.EffectsOf.find(FD);
  if (It != C.EffectsOf.end())
    return It->second;
  // Inserted before the walk, so that recursion stops at the empty entry. A
  // library function does not assign the program's globals.
  C.EffectsOf[FD];
  const FunctionDecl* Def = FD->getDefinition();
  Effects Found;
  if (Def && !C.S.getSourceManager().isInSystemHeader(Def->getLocation()))
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    EffectsCollector(C, Found).TraverseDecl(const_cast<FunctionDecl*>(Def));
  else
    Found.Opaque = true;
  C.EffectsOf[FD] = std::move(Found);
  return C.EffectsOf[FD];
}

bool EffectsCollector::VisitCallExpr(CallExpr* CE) {
  // What a parameter is handed on to may be written.
  for (const Expr* Arg : CE->arguments())
    throughParam(Arg);
  if (const FunctionDecl* Callee = CE->getDirectCallee()) {
    const Effects& E = effectsOf(m_C, Callee);
    m_E.Globals.insert(E.Globals.begin(), E.Globals.end());
  }
  return true;
}

bool EffectsCollector::VisitCXXConstructExpr(CXXConstructExpr* CE) {
  for (const Expr* Arg : CE->arguments())
    throughParam(Arg);
  const Effects& E = effectsOf(m_C, CE->getConstructor());
  m_E.Globals.insert(E.Globals.begin(), E.Globals.end());
  return true;
}

/// Walks one function in execution order, keeping a State.
class Checker {
  Context& m_C;
  const ASTContext& m_AST;
  /// The calls that lead here, innermost first.
  llvm::SmallVector<std::pair<const FunctionDecl*, SourceLocation>, 4> m_Chain;
  /// Whether the globals this function names have adjoints: clad gives them
  /// one only in the file of the differentiated function.
  bool m_GlobalsHaveAdjoints = true;
  // These are set up by the caller of a callee's walk, another Checker.
  State m_S;
  /// Local references to pointers, and the pointers they name.
  std::map<const VarDecl*, Key> m_Aliases;
  /// The object a method is walked for, when it is a part of a variable.
  std::optional<Key> m_This;
  /// Whether a pointer this function returns has an adjoint with no value.
  std::optional<Origin> m_Returns;
  /// The results of the calls walked so far whose adjoint has no value.
  llvm::DenseMap<const Expr*, Origin> m_CallResults;
  /// The addresses passed straight to a call, which the call accounts for.
  llvm::SmallPtrSet<const UnaryOperator*, 4> m_PassedAddresses;
  /// The variables the constructor calls walked so far initialize.
  llvm::DenseMap<const CXXConstructExpr*, Key> m_Constructed;
  struct Jumps {
    bool IsLoop = false;
    std::vector<State> Breaks, Continues;
    /// For a switch, the state each case label is entered with.
    std::optional<State> SwitchEntry;
  };
  std::vector<Jumps> m_Jumps;

  //===--- What the state says ---------------------------------------===//

  std::optional<Origin> defaultOf(const Key& K) const {
    const auto* VD = dyn_cast<VarDecl>(K.first);
    if (!m_GlobalsHaveAdjoints || !VD || !VD->isFileVarDecl() ||
        VD->getType().isConstQualified())
      return std::nullopt;
    Origin O;
    O.Global = VD;
    O.Name = nameOf(K);
    return O;
  }

  static std::string parentOf(const std::string& Path) {
    std::size_t Pos = Path.find_last_of(".[");
    return Pos == std::string::npos ? std::string() : Path.substr(0, Pos);
  }

  std::optional<Origin> lookup(const State& S, const Key& K) const {
    if (S.Dead)
      return std::nullopt;
    for (std::string P = K.second;; P = parentOf(P)) {
      auto It = S.Parts.find({K.first, P});
      if (It != S.Parts.end()) {
        std::optional<Origin> O = It->second;
        // An entry for the whole of a global speaks for the part asked for.
        if (O && O->K == Origin::Kind::Entry && O->Global == K.first)
          O->Name = nameOf(K);
        return O;
      }
      if (P.empty())
        break;
    }
    return defaultOf(K);
  }
  std::optional<Origin> lookup(const Key& K) const { return lookup(m_S, K); }

  /// Whether any part of \p K has an adjoint with no value.
  std::optional<Origin> anyUnknownIn(const Key& K) const {
    if (std::optional<Origin> O = lookup(K))
      return O;
    for (const auto& Part : m_S.Parts)
      if (Part.first.first == K.first && Part.second &&
          Part.first.second.compare(0, K.second.size(), K.second) == 0)
        return Part.second;
    return std::nullopt;
  }

  void set(const Key& K, std::optional<Origin> O) {
    if (m_S.Dead)
      return;
    // An element whose index is not known may be any of them: knowing it
    // tells nothing about one, not knowing it unknows them all.
    std::size_t Any = K.second.find("[*]");
    if (Any != std::string::npos) {
      if (O)
        set({K.first, K.second.substr(0, Any)}, O);
      return;
    }
    for (auto It = m_S.Parts.begin(); It != m_S.Parts.end();) {
      const std::string& P = It->first.second;
      if (It->first.first == K.first && P.size() > K.second.size() &&
          P.compare(0, K.second.size(), K.second) == 0 &&
          (P[K.second.size()] == '.' || P[K.second.size()] == '['))
        It = m_S.Parts.erase(It);
      else
        ++It;
    }
    m_S.Parts[K] = std::move(O);
  }

  State merge(const State& A, const State& B) const {
    if (A.Dead)
      return B;
    if (B.Dead)
      return A;
    State R;
    std::set<Key> Keys;
    for (const auto& P : A.Parts)
      Keys.insert(P.first);
    for (const auto& P : B.Parts)
      Keys.insert(P.first);
    for (const Key& K : Keys) {
      std::optional<Origin> O = lookup(A, K);
      R.Parts[K] = O ? O : lookup(B, K);
    }
    return R;
  }

  bool sameKnowledge(const State& A, const State& B) const {
    if (A.Dead || B.Dead)
      return A.Dead == B.Dead;
    std::set<Key> Keys;
    for (const auto& P : A.Parts)
      Keys.insert(P.first);
    for (const auto& P : B.Parts)
      Keys.insert(P.first);
    for (const Key& K : Keys)
      if (lookup(A, K).has_value() != lookup(B, K).has_value())
        return false;
    return true;
  }

  static State dead() {
    State S;
    S.Dead = true;
    return S;
  }

  std::optional<Key> key(const Expr* E) const {
    return keyOf(E, m_AST, &m_Aliases, &m_This);
  }

  //===--- The adjoint of a pointer-typed value ----------------------===//

  /// Whether the adjoint of the pointer \p E evaluates to has no value.
  std::optional<Origin> value(const Expr* E) const {
    E = E->IgnoreParens();
    if (E->isGLValue() && E->getType()->isPointerType())
      return lvalue(E);
    if (const auto* CE = dyn_cast<CastExpr>(E)) {
      CastKind CK = CE->getCastKind();
      if (CK == CK_LValueToRValue)
        return lvalue(CE->getSubExpr());
      if (CK == CK_ArrayToPointerDecay)
        return address(CE->getSubExpr());
      if (keepsObject(CK) || (isa<ExplicitCastExpr>(CE) &&
                              CE->getSubExpr()->getType()->isPointerType()))
        return value(CE->getSubExpr());
      return std::nullopt;
    }
    if (const auto* UO = dyn_cast<UnaryOperator>(E)) {
      if (UO->getOpcode() == UO_AddrOf)
        return address(UO->getSubExpr());
      if (UO->isIncrementDecrementOp())
        return lvalue(UO->getSubExpr());
      // `+p`
      return value(UO->getSubExpr());
    }
    if (const auto* BO = dyn_cast<BinaryOperator>(E)) {
      switch (BO->getOpcode()) {
      case BO_Assign:
      case BO_Comma:
        return value(BO->getRHS());
      case BO_AddAssign:
      case BO_SubAssign:
        return lvalue(BO->getLHS());
      default:
        // `p + n`, `n + p`, `p - n`
        return value(BO->getLHS()->getType()->isPointerType() ? BO->getLHS()
                                                              : BO->getRHS());
      }
    }
    if (const auto* CO = dyn_cast<AbstractConditionalOperator>(E)) {
      if (std::optional<Origin> O = value(CO->getTrueExpr()))
        return O;
      return value(CO->getFalseExpr());
    }
    if (isa<CallExpr>(E)) {
      auto It = m_CallResults.find(E);
      if (It != m_CallResults.end())
        return It->second;
      return std::nullopt;
    }
    if (const auto* FE = dyn_cast<FullExpr>(E))
      return value(FE->getSubExpr());
    if (const auto* DAE = dyn_cast<CXXDefaultArgExpr>(E))
      return value(DAE->getExpr());
    if (const auto* DIE = dyn_cast<CXXDefaultInitExpr>(E))
      return value(DIE->getExpr());
    return std::nullopt;
  }

  /// The same for a pointer read out of the lvalue \p E.
  std::optional<Origin> lvalue(const Expr* E) const {
    E = E->IgnoreParens();
    if (std::optional<Key> K = key(E))
      return lookup(*K);
    if (const auto* UO = dyn_cast<UnaryOperator>(E))
      if (UO->getOpcode() == UO_Deref)
        if (const auto* Addr = dyn_cast<UnaryOperator>(
                UO->getSubExpr()->IgnoreParenImpCasts()))
          if (Addr->getOpcode() == UO_AddrOf)
            return lvalue(Addr->getSubExpr());
    if (const auto* BO = dyn_cast<BinaryOperator>(E)) {
      if (BO->isAssignmentOp())
        return lvalue(BO->getLHS());
      if (BO->getOpcode() == BO_Comma)
        return lvalue(BO->getRHS());
    }
    if (const auto* CO = dyn_cast<AbstractConditionalOperator>(E)) {
      if (std::optional<Origin> O = lvalue(CO->getTrueExpr()))
        return O;
      return lvalue(CO->getFalseExpr());
    }
    // Anything read through a pointer whose adjoint is known has an adjoint
    // read through that one, and a global reference is reported where it is
    // used.
    return std::nullopt;
  }

  /// The same for the pointer `&E`.
  std::optional<Origin> address(const Expr* E) const {
    E = E->IgnoreParens();
    if (const auto* ICE = dyn_cast<ImplicitCastExpr>(E))
      if (keepsObject(ICE->getCastKind()))
        return address(ICE->getSubExpr());
    if (const auto* UO = dyn_cast<UnaryOperator>(E))
      if (UO->getOpcode() == UO_Deref)
        return value(UO->getSubExpr());
    if (const auto* ME = dyn_cast<MemberExpr>(E))
      return ME->isArrow() ? value(ME->getBase()) : address(ME->getBase());
    if (const auto* ASE = dyn_cast<ArraySubscriptExpr>(E))
      return value(ASE->getBase());
    return std::nullopt;
  }

  /// The same for a whole object of class or array type: whether any
  /// pointer in it has an adjoint with no value.
  std::optional<Origin> object(const Expr* E) const {
    E = E->IgnoreParens();
    if (const auto* CE = dyn_cast<CastExpr>(E))
      if (keepsObject(CE->getCastKind()) ||
          CE->getCastKind() == CK_LValueToRValue)
        return object(CE->getSubExpr());
    if (const auto* FE = dyn_cast<FullExpr>(E))
      return object(FE->getSubExpr());
    if (const auto* MTE = dyn_cast<MaterializeTemporaryExpr>(E))
      return object(MTE->getSubExpr());
    if (const auto* BTE = dyn_cast<CXXBindTemporaryExpr>(E))
      return object(BTE->getSubExpr());
    if (const auto* CE = dyn_cast<CXXConstructExpr>(E)) {
      for (const Expr* Arg : CE->arguments())
        if (std::optional<Origin> O =
                Arg->getType()->isPointerType() ? value(Arg) : object(Arg))
          return O;
      return std::nullopt;
    }
    if (const auto* ILE = dyn_cast<InitListExpr>(E)) {
      for (const Expr* Init : ILE->inits())
        if (std::optional<Origin> O =
                Init->getType()->isPointerType() ? value(Init) : object(Init))
          return O;
      return std::nullopt;
    }
    if (std::optional<Key> K = key(E))
      return anyUnknownIn(*K);
    return std::nullopt;
  }

  /// \p K takes the value of \p E.
  void assign(const Key& K, QualType T, const Expr* E, SourceLocation Loc) {
    T = T.getNonReferenceType().getCanonicalType();
    std::optional<Origin> O;
    if (T->isPointerType())
      O = value(E);
    else if (T->isRecordType() || T->isArrayType())
      O = object(E);
    else
      return;
    if (O && !isGlobal(K.first) && O->CopyName.empty()) {
      O->CopyName = nameOf(K);
      O->CopyLoc = Loc;
    }
    set(K, O);
  }

  //===--- Reporting --------------------------------------------------===//

  void report(Finding F) {
    F.Chain = m_Chain;
    m_C.Findings.push_back(std::move(F));
  }

  /// An access through the pointer \p Ptr, whose derivative is needed when
  /// \p Carry.
  void access(const Expr* Ptr, bool Carry) {
    if (!Carry || !isPointerToCarried(Ptr->getType()))
      return;
    if (std::optional<Origin> O = value(Ptr)) {
      Finding F;
      F.Loc = Ptr->getBeginLoc();
      F.Pointer = O->Name;
      F.O = std::move(*O);
      report(std::move(F));
    }
  }

  //===--- Statements -------------------------------------------------===//

  void declare(const VarDecl* VD) {
    const Expr* Init = VD->getInit();
    if (!Init)
      return;
    if (const auto* CCE = dyn_cast<CXXConstructExpr>(Init->IgnoreImplicit()))
      m_Constructed[CCE] = {VD, ""};
    // An initializer is read, and a reference bound, for the derivative.
    expr(Init, /*Carry=*/true);
    QualType T = VD->getType();
    if (T->isReferenceType()) {
      if (T.getNonReferenceType()->isPointerType())
        if (std::optional<Key> K = key(Init))
          m_Aliases[VD] = *K;
      return;
    }
    if (const auto* ILE = dyn_cast<InitListExpr>(Init->IgnoreParens()))
      if (T->isRecordType() || T->isArrayType()) {
        set({VD, ""}, std::nullopt);
        initParts({VD, ""}, ILE);
        return;
      }
    assign({VD, ""}, T, Init, VD->getLocation());
  }

  /// The parts of \p K take the values of the elements of \p ILE.
  void initParts(const Key& K, const InitListExpr* ILE) {
    QualType T = ILE->getType().getCanonicalType();
    if (const auto* RD = T->getAsRecordDecl()) {
      unsigned I = 0;
      for (const FieldDecl* FD : RD->fields()) {
        if (I >= ILE->getNumInits())
          break;
        const Expr* Init = ILE->getInit(I++);
        Key Part{K.first, K.second + "." + FD->getNameAsString()};
        if (const auto* Sub = dyn_cast<InitListExpr>(Init->IgnoreParens()))
          initParts(Part, Sub);
        else
          assign(Part, FD->getType(), Init, Init->getBeginLoc());
      }
    } else if (T->isArrayType()) {
      for (unsigned I = 0, N = ILE->getNumInits(); I < N; ++I) {
        const Expr* Init = ILE->getInit(I);
        Key Part{K.first, K.second + "[" + std::to_string(I) + "]"};
        if (const auto* Sub = dyn_cast<InitListExpr>(Init->IgnoreParens()))
          initParts(Part, Sub);
        else
          assign(Part, Init->getType(), Init, Init->getBeginLoc());
      }
    }
  }

  void loop(const VarDecl* CondVar, const Expr* Cond,
            llvm::function_ref<void()> Body, const Expr* Inc, bool BodyFirst) {
    State Entry = m_S;
    State Exit;
    // The adjoints known on entry to an iteration are those known both on
    // entry to the loop and at the end of an iteration; knowledge only shrinks,
    // so this settles in as many rounds as there are keys.
    for (unsigned Round = 0;; ++Round) {
      m_S = Entry;
      State AfterCond = dead();
      auto condition = [&] {
        if (CondVar)
          declare(CondVar);
        if (Cond)
          expr(Cond, /*Carry=*/false);
        AfterCond = m_S;
      };
      if (!BodyFirst)
        condition();
      m_Jumps.push_back({/*IsLoop=*/true, {}, {}, std::nullopt});
      Body();
      for (const State& C : m_Jumps.back().Continues)
        m_S = merge(m_S, C);
      if (Inc)
        expr(Inc, /*Carry=*/false);
      if (BodyFirst)
        condition();
      State Back = m_S;
      std::vector<State> Breaks = std::move(m_Jumps.back().Breaks);
      m_Jumps.pop_back();
      // Without a condition, or with one that always holds, the loop is left
      // only by a break.
      bool AlwaysTrue = false;
      if (Cond && !CondVar && !Cond->isValueDependent())
        Cond->EvaluateAsBooleanCondition(AlwaysTrue, m_AST);
      Exit = ((Cond || CondVar) && !AlwaysTrue) ? AfterCond : dead();
      for (const State& B : Breaks)
        Exit = merge(Exit, B);
      State Next = merge(Entry, Back);
      if (sameKnowledge(Next, Entry) || Round == 64)
        break;
      Entry = std::move(Next);
    }
    m_S = std::move(Exit);
  }

  Jumps* innermost(bool LoopOnly) {
    for (auto It = m_Jumps.rbegin(); It != m_Jumps.rend(); ++It)
      if (It->IsLoop || !LoopOnly)
        return &*It;
    return nullptr;
  }

public:
  void stmt(const Stmt* S) {
    if (!S)
      return;
    if (const auto* E = dyn_cast<Expr>(S)) {
      // A value nobody reads has no derivative.
      expr(E, /*Carry=*/false);
      return;
    }
    if (const auto* DS = dyn_cast<DeclStmt>(S)) {
      for (const Decl* D : DS->decls())
        if (const auto* VD = dyn_cast<VarDecl>(D))
          declare(VD);
      return;
    }
    if (const auto* If = dyn_cast<IfStmt>(S)) {
      stmt(If->getInit());
      if (const VarDecl* CV = If->getConditionVariable())
        declare(CV);
      expr(If->getCond(), /*Carry=*/false);
      State Before = m_S;
      stmt(If->getThen());
      State Then = std::move(m_S);
      m_S = std::move(Before);
      stmt(If->getElse());
      m_S = merge(Then, m_S);
      return;
    }
    if (const auto* WS = dyn_cast<WhileStmt>(S)) {
      loop(
          WS->getConditionVariable(), WS->getCond(),
          [&] { stmt(WS->getBody()); }, nullptr, /*BodyFirst=*/false);
      return;
    }
    if (const auto* DS = dyn_cast<DoStmt>(S)) {
      loop(
          nullptr, DS->getCond(), [&] { stmt(DS->getBody()); }, nullptr,
          /*BodyFirst=*/true);
      return;
    }
    if (const auto* FS = dyn_cast<ForStmt>(S)) {
      stmt(FS->getInit());
      loop(
          FS->getConditionVariable(), FS->getCond(),
          [&] { stmt(FS->getBody()); }, FS->getInc(), /*BodyFirst=*/false);
      return;
    }
    if (const auto* RS = dyn_cast<CXXForRangeStmt>(S)) {
      stmt(RS->getInit());
      stmt(RS->getRangeStmt());
      stmt(RS->getBeginStmt());
      stmt(RS->getEndStmt());
      loop(
          nullptr, RS->getCond(),
          [&] {
            stmt(RS->getLoopVarStmt());
            stmt(RS->getBody());
          },
          RS->getInc(), /*BodyFirst=*/false);
      return;
    }
    if (const auto* SS = dyn_cast<SwitchStmt>(S)) {
      stmt(SS->getInit());
      if (const VarDecl* CV = SS->getConditionVariable())
        declare(CV);
      expr(SS->getCond(), /*Carry=*/false);
      State Entry = m_S;
      m_Jumps.push_back({/*IsLoop=*/false, {}, {}, Entry});
      // Control enters the body only at a label.
      m_S = dead();
      stmt(SS->getBody());
      State Exit = std::move(m_S);
      for (const State& B : m_Jumps.back().Breaks)
        Exit = merge(Exit, B);
      m_Jumps.pop_back();
      bool HasDefault = false;
      for (const SwitchCase* SC = SS->getSwitchCaseList(); SC;
           SC = SC->getNextSwitchCase())
        HasDefault |= isa<DefaultStmt>(SC);
      if (!HasDefault)
        Exit = merge(Exit, Entry);
      m_S = std::move(Exit);
      return;
    }
    if (const auto* SC = dyn_cast<SwitchCase>(S)) {
      for (auto It = m_Jumps.rbegin(); It != m_Jumps.rend(); ++It)
        if (It->SwitchEntry) {
          m_S = merge(m_S, *It->SwitchEntry);
          break;
        }
      stmt(SC->getSubStmt());
      return;
    }
    if (isa<BreakStmt>(S)) {
      if (Jumps* J = innermost(/*LoopOnly=*/false))
        J->Breaks.push_back(m_S);
      m_S = dead();
      return;
    }
    if (isa<ContinueStmt>(S)) {
      if (Jumps* J = innermost(/*LoopOnly=*/true))
        J->Continues.push_back(m_S);
      m_S = dead();
      return;
    }
    if (const auto* RS = dyn_cast<ReturnStmt>(S)) {
      if (const Expr* RV = RS->getRetValue()) {
        expr(RV, /*Carry=*/true);
        if (RV->getType()->isPointerType() && !m_Returns)
          m_Returns = value(RV);
      }
      m_S = dead();
      return;
    }
    if (const auto* TS = dyn_cast<CXXTryStmt>(S)) {
      State Entry = m_S;
      stmt(TS->getTryBlock());
      State Exit = m_S;
      for (unsigned I = 0, N = TS->getNumHandlers(); I < N; ++I) {
        m_S = merge(Entry, Exit);
        stmt(TS->getHandler(I)->getHandlerBlock());
        Exit = merge(Exit, m_S);
      }
      m_S = std::move(Exit);
      return;
    }
    for (const Stmt* Child : S->children())
      stmt(Child);
  }

  //===--- Expressions ------------------------------------------------===//

  /// Walks \p E in evaluation order. \p Carry is whether the derivative needs
  /// the adjoint of E's value: not for a condition, a comparison or a value
  /// nobody reads, which clad does not differentiate.
  void expr(const Expr* E, bool Carry) {
    if (!E)
      return;
    if (const auto* PE = dyn_cast<ParenExpr>(E))
      return expr(PE->getSubExpr(), Carry);
    if (const auto* CE = dyn_cast<CastExpr>(E))
      return expr(CE->getSubExpr(), Carry && !isToBoolean(CE->getCastKind()));
    if (const auto* UO = dyn_cast<UnaryOperator>(E)) {
      const Expr* Sub = UO->getSubExpr();
      switch (UO->getOpcode()) {
      case UO_Deref:
        expr(Sub, Carry);
        access(Sub, Carry);
        return;
      case UO_LNot:
        return expr(Sub, /*Carry=*/false);
      case UO_PreInc:
      case UO_PreDec:
      case UO_PostInc:
      case UO_PostDec:
        return expr(Sub, /*Carry=*/true);
      case UO_AddrOf:
        expr(Sub, Carry);
        // An address passed to a call is followed by what the call does.
        if (!m_PassedAddresses.count(UO))
          addressTaken(Sub, UO->getOperatorLoc(), nullptr);
        return;
      default:
        return expr(Sub, Carry);
      }
    }
    if (const auto* BO = dyn_cast<BinaryOperator>(E)) {
      const Expr* L = BO->getLHS();
      const Expr* R = BO->getRHS();
      if (BO->isComparisonOp()) {
        expr(L, /*Carry=*/false);
        expr(R, /*Carry=*/false);
        return;
      }
      if (BO->isLogicalOp()) {
        expr(L, /*Carry=*/false);
        State AfterL = m_S;
        expr(R, /*Carry=*/false);
        m_S = merge(AfterL, m_S);
        return;
      }
      if (BO->getOpcode() == BO_Comma) {
        expr(L, /*Carry=*/false);
        expr(R, Carry);
        return;
      }
      if (BO->isAssignmentOp()) {
        // The right operand is evaluated first, and both are differentiated.
        expr(R, /*Carry=*/true);
        expr(L, /*Carry=*/true);
        if (BO->getOpcode() == BO_Assign)
          if (std::optional<Key> K = key(L))
            assign(*K, L->getType(), R, BO->getOperatorLoc());
        return;
      }
      expr(L, Carry);
      expr(R, BO->isPtrMemOp() ? false : Carry);
      if (BO->getOpcode() == BO_PtrMemI)
        access(L, Carry);
      return;
    }
    if (const auto* CO = dyn_cast<ConditionalOperator>(E)) {
      expr(CO->getCond(), /*Carry=*/false);
      State Before = m_S;
      expr(CO->getTrueExpr(), Carry);
      State AfterTrue = std::move(m_S);
      m_S = std::move(Before);
      expr(CO->getFalseExpr(), Carry);
      m_S = merge(AfterTrue, m_S);
      return;
    }
    if (const auto* ME = dyn_cast<MemberExpr>(E)) {
      // A member clad does not differentiate reads as zero.
      bool Diff = Carry && !utils::hasNonDifferentiableAttribute(ME);
      expr(ME->getBase(), Diff);
      if (ME->isArrow())
        access(ME->getBase(), Diff);
      return;
    }
    if (const auto* ASE = dyn_cast<ArraySubscriptExpr>(E)) {
      expr(ASE->getBase(), Carry);
      expr(ASE->getIdx(), /*Carry=*/false);
      if (!ASE->getBase()->IgnoreParenImpCasts()->getType()->isArrayType())
        access(ASE->getBase(), Carry);
      return;
    }
    if (const auto* DRE = dyn_cast<DeclRefExpr>(E)) {
      const auto* VD = dyn_cast<VarDecl>(DRE->getDecl());
      if (Carry && VD && m_GlobalsHaveAdjoints && VD->isFileVarDecl() &&
          VD->getType()->isReferenceType() && carries(VD->getType())) {
        Finding F;
        F.Reference = VD;
        F.Loc = DRE->getLocation();
        report(std::move(F));
      }
      return;
    }
    if (const auto* CE = dyn_cast<CallExpr>(E))
      return call(CE, CE->getDirectCallee());
    if (const auto* CE = dyn_cast<CXXConstructExpr>(E))
      return call(CE, CE->getConstructor());
    // Not evaluated, or a function clad derives on its own.
    if (isa<UnaryExprOrTypeTraitExpr>(E) || isa<CXXTypeidExpr>(E) ||
        isa<CXXNoexceptExpr>(E) || isa<LambdaExpr>(E) ||
        isa<OpaqueValueExpr>(E))
      return;
    if (const auto* GSE = dyn_cast<GenericSelectionExpr>(E))
      return expr(GSE->getResultExpr(), Carry);
    if (const auto* SE = dyn_cast<StmtExpr>(E))
      return stmt(SE->getSubStmt());
    if (const auto* DAE = dyn_cast<CXXDefaultArgExpr>(E))
      return expr(DAE->getExpr(), Carry);
    if (const auto* DIE = dyn_cast<CXXDefaultInitExpr>(E))
      return expr(DIE->getExpr(), Carry);
    for (const Stmt* Child : E->children())
      if (const auto* CE = dyn_cast_or_null<Expr>(Child))
        expr(CE, Carry);
      else
        stmt(Child);
  }

  /// A pointer whose address is taken may be written through what that
  /// address is copied to, which clad does not follow.
  /// \p Callee, when given, is what the address is passed to.
  void addressTaken(const Expr* E, SourceLocation Loc,
                    const FunctionDecl* Callee) {
    std::optional<Key> K = key(E);
    if (!K || !isGlobal(K->first))
      return;
    QualType T = E->getType().getCanonicalType();
    if (!T->isPointerType() && !T->isRecordType() && !T->isArrayType())
      return;
    Origin O;
    O.K = Callee ? Origin::Kind::PassedByReference : Origin::Kind::AddressTaken;
    O.Global = cast<VarDecl>(K->first);
    O.Name = nameOf(*K);
    O.Callee = Callee;
    O.Where = Loc;
    set(*K, O);
  }

  /// The arguments of a call clad differentiates are read for its
  /// derivative whether or not its value is.
  void call(const Expr* E, const FunctionDecl* FD) {
    CallDerivative How = FD ? m_C.How(FD) : CallDerivative::None;
    bool Diff = How != CallDerivative::None;
    llvm::ArrayRef<const Expr*> Args;
    std::optional<Key> Object;
    if (const auto* CE = dyn_cast<CallExpr>(E)) {
      Args = {CE->getArgs(), CE->getNumArgs()};
      if (const auto* MCE = dyn_cast<CXXMemberCallExpr>(CE)) {
        // The object is an argument like any other.
        expr(MCE->getCallee(), Diff);
        if (const Expr* Obj = MCE->getImplicitObjectArgument())
          if (!Obj->getType()->isPointerType())
            Object = key(Obj);
      } else {
        expr(CE->getCallee(), /*Carry=*/false);
      }
    } else {
      const auto* CCE = cast<CXXConstructExpr>(E);
      Args = {CCE->getArgs(), CCE->getNumArgs()};
      // A constructor is walked for the variable it initializes, or for a
      // temporary named after it.
      auto It = m_Constructed.find(CCE);
      Object = It != m_Constructed.end() ? It->second
                                         : Key{CCE->getConstructor(), ""};
    }
    for (const Expr* Arg : Args)
      if (const auto* UO = dyn_cast<UnaryOperator>(Arg->IgnoreParenImpCasts()))
        if (UO->getOpcode() == UO_AddrOf && FD)
          m_PassedAddresses.insert(UO);
    for (const Expr* Arg : Args)
      expr(Arg, Diff);
    if (!FD)
      return;

    const FunctionDecl* Def = FD->getDefinition();
    if (How == CallDerivative::Pullback && Def && Def->hasBody() &&
        !m_C.Active.count(Def->getCanonicalDecl()) && m_C.Budget) {
      --m_C.Budget;
      Checker Callee(m_C, Def, m_Chain, E->getBeginLoc(), m_S);
      if (Object)
        Callee.m_This = Object;
      // What a constructor initializes starts out with nothing in it.
      if (isa<CXXConstructExpr>(E))
        Callee.set(*Object, std::nullopt);
      bool IsMember = isa<CXXOperatorCallExpr>(E) && isa<CXXMethodDecl>(Def) &&
                      cast<CXXMethodDecl>(Def)->isInstance();
      // An unprototyped call in C may pass fewer arguments than parameters.
      std::size_t N =
          std::min<std::size_t>(Def->getNumParams(), Args.size() - IsMember);
      for (unsigned I = 0; I < N; ++I) {
        unsigned A = I + IsMember;
        const ParmVarDecl* PVD = Def->getParamDecl(I);
        QualType PT = PVD->getType();
        if (PT->isReferenceType()) {
          if (PT.getNonReferenceType()->isPointerType())
            if (std::optional<Key> K = key(Args[A]))
              Callee.m_Aliases[PVD] = *K;
          continue;
        }
        Callee.assign({PVD, ""}, PT, Args[A], Args[A]->getBeginLoc());
      }
      if (IsMember && !Args.empty())
        Callee.m_This = key(Args[0]);
      Callee.run(Def);
      if (Callee.m_Returns && isa<CallExpr>(E))
        m_CallResults[E] = *Callee.m_Returns;
    } else if (How == CallDerivative::Opaque) {
      // The derivative takes the adjoint of each pointer argument, and clad
      // cannot see what it does with it.
      for (const Expr* Arg : Args)
        if (isPointerToCarried(Arg->getType()))
          if (std::optional<Origin> O = value(Arg)) {
            Finding F;
            F.IsPass = true;
            F.Loc = Arg->getBeginLoc();
            F.Pointer = O->Name;
            F.PassedTo = FD;
            F.O = std::move(*O);
            report(std::move(F));
          }
    }

    // What the call does to pointers, the forward pass does without their
    // adjoints: a pointer the call may reassign keeps the adjoint it had,
    // which then belongs to another object.
    const Effects& Eff = effectsOf(m_C, FD);
    bool IsMemberOp = isa<CXXOperatorCallExpr>(E) && isa<CXXMethodDecl>(FD) &&
                      cast<CXXMethodDecl>(FD)->isInstance();
    std::size_t N =
        std::min<std::size_t>(FD->getNumParams(), Args.size() - IsMemberOp);
    for (unsigned I = 0; I < N; ++I) {
      unsigned A = I + IsMemberOp;
      if (!Eff.Opaque && !Eff.WritesThroughParam.count(I))
        continue;
      const Expr* Arg = Args[A];
      if (const auto* UO = dyn_cast<UnaryOperator>(Arg->IgnoreParenImpCasts()))
        if (UO->getOpcode() == UO_AddrOf) {
          addressTaken(UO->getSubExpr(), Arg->getBeginLoc(), FD);
          continue;
        }
      if (!utils::isNonConstReferenceType(FD->getParamDecl(I)->getType()))
        continue;
      std::optional<Key> K = key(Arg);
      if (!K || !isGlobal(K->first))
        continue;
      Origin O;
      O.K = Origin::Kind::PassedByReference;
      O.Global = cast<VarDecl>(K->first);
      O.Name = nameOf(*K);
      O.Callee = FD;
      O.Where = Arg->getBeginLoc();
      set(*K, O);
    }
    for (const Key& K : Eff.Globals) {
      Origin O;
      O.K = Origin::Kind::CalleeAssigns;
      O.Global = cast<VarDecl>(K.first);
      O.Name = nameOf(K);
      O.Callee = FD;
      O.Where = E->getBeginLoc();
      set(K, O);
    }
    if (Eff.AssignsThisPointer && Object) {
      Origin O;
      O.K = Origin::Kind::CalleeAssigns;
      O.Name = nameOf(*Object);
      if (isGlobal(Object->first))
        O.Global = cast<VarDecl>(Object->first);
      O.Callee = FD;
      O.Where = E->getBeginLoc();
      set(*Object, O);
    }
  }

  Checker(Context& C, const FunctionDecl* FD,
          llvm::ArrayRef<std::pair<const FunctionDecl*, SourceLocation>> Chain,
          SourceLocation CallLoc, const State& S)
      : m_C(C), m_AST(C.S.getASTContext()), m_Chain(Chain.begin(), Chain.end()),
        m_S(S) {
    m_S.Dead = false;
    if (CallLoc.isValid())
      m_Chain.insert(m_Chain.begin(), {FD, CallLoc});
    const SourceManager& SM = C.S.getSourceManager();
    m_GlobalsHaveAdjoints =
        SM.getFileID(SM.getExpansionLoc(FD->getLocation())) == C.TopFile;
  }

  void run(const FunctionDecl* FD) {
    m_C.Active.insert(FD->getCanonicalDecl());
    if (const auto* CD = dyn_cast<CXXConstructorDecl>(FD))
      for (const CXXCtorInitializer* Init : CD->inits()) {
        expr(Init->getInit(), /*Carry=*/true);
        if (const FieldDecl* Field = Init->getMember())
          if (m_This)
            assign({m_This->first,
                    m_This->second + "." + Field->getNameAsString()},
                   Field->getType(), Init->getInit(),
                   Init->getSourceLocation());
      }
    stmt(FD->getBody());
    m_C.Active.erase(FD->getCanonicalDecl());
  }
};

void emit(Sema& S, const Finding& F) {
  using DE = DiagnosticsEngine;
  if (F.Reference) {
    utils::diag(S, DE::Error, F.Loc,
                "cannot differentiate this use of the global reference %0: "
                "clad cannot give a reference a derivative")
        << F.Reference;
    utils::diag(S, DE::Note, F.Reference->getLocation(),
                "pass what %0 refers to as a parameter instead")
        << F.Reference;
  } else {
    if (F.IsPass)
      utils::diag(S, DE::Error, F.Loc,
                  "cannot differentiate passing '%0' to %1: the derivative of "
                  "what it points to is unknown here")
          << F.Pointer << F.PassedTo;
    else
      utils::diag(S, DE::Error, F.Loc,
                  "cannot differentiate this access through '%0': the "
                  "derivative of what it points to is unknown here")
          << F.Pointer;
    const Origin& O = F.O;
    if (!O.CopyName.empty())
      utils::diag(S, DE::Note, O.CopyLoc, "'%0' is copied from '%1' here")
          << O.CopyName << O.Name;
    switch (O.K) {
    case Origin::Kind::Entry:
      utils::diag(S, DE::Note, O.Global->getLocation(),
                  "'%0' has no derivative to point to until the differentiated "
                  "code assigns it; assign it before the access, or pass what "
                  "it points to as a parameter")
          << O.Name;
      break;
    case Origin::Kind::CalleeAssigns:
      utils::diag(S, DE::Note, O.Where,
                  "'%0' may be reassigned by this call to %1, which does not "
                  "update its derivative")
          << O.Name << O.Callee;
      break;
    case Origin::Kind::PassedByReference:
      utils::diag(S, DE::Note, O.Where,
                  "'%0' is passed by reference to %1 here, which may reassign "
                  "it without updating its derivative")
          << O.Name << O.Callee;
      break;
    case Origin::Kind::AddressTaken:
      utils::diag(S, DE::Note, O.Where,
                  "the address of '%0' is taken here, and clad does not follow "
                  "its derivative through it")
          << O.Name;
      break;
    }
  }
  for (const auto& Call : F.Chain)
    utils::diag(S, DE::Note, Call.second,
                "in the derivative of %0, called here")
        << Call.first;
}

} // namespace

void diagnoseUnknownEntryAdjoints(
    Sema& S, const FunctionDecl* FD,
    llvm::function_ref<CallDerivative(const FunctionDecl*)> HowDifferentiated) {
  const FunctionDecl* Def = FD->getDefinition();
  if (!Def || !Def->hasBody())
    return;
  const SourceManager& SM = S.getSourceManager();
  Context C(S, HowDifferentiated,
            SM.getFileID(SM.getExpansionLoc(Def->getLocation())));
  Checker(C, Def, {}, SourceLocation(), State()).run(Def);

  // A loop is walked until what it knows settles, and a callee once per
  // call. Report the first access through each pointer, along each chain of
  // calls: fixing it fixes the rest.
  std::vector<const Finding*> Order;
  for (const Finding& F : C.Findings)
    Order.push_back(&F);
  auto Outer = [](const Finding* F) {
    return F->Chain.empty() ? F->Loc : F->Chain.back().second;
  };
  std::stable_sort(Order.begin(), Order.end(),
                   [&](const Finding* A, const Finding* B) {
                     if (Outer(A) != Outer(B))
                       return SM.isBeforeInTranslationUnit(Outer(A), Outer(B));
                     return SM.isBeforeInTranslationUnit(A->Loc, B->Loc);
                   });
  std::set<std::string> Seen;
  auto Last = std::remove_if(Order.begin(), Order.end(), [&](const Finding* F) {
    std::string Id = F->Reference ? F->Reference->getNameAsString()
                                  : F->Pointer + (F->IsPass ? "/pass" : "");
    for (const auto& Call : F->Chain)
      Id += "@" + std::to_string(Call.second.getRawEncoding());
    return !Seen.insert(Id).second;
  });
  Order.erase(Last, Order.end());
  for (const Finding* F : Order)
    emit(S, *F);
}

} // namespace clad
