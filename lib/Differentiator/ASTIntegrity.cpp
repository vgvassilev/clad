//--------------------------------------------------------------------*- C++ -//
// clad - the C++ Clang-based Automatic Differentiator
//
// See ASTIntegrity.h for the rationale.
//----------------------------------------------------------------------------//

#include "ASTIntegrity.h"

#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtOpenMP.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace clang;

namespace clad {

const Stmt* findSharedNode(const Stmt* Root) {
  if (!Root)
    return nullptr;
  // clad's hand-built body must be a tree in its Stmt CHILD-edge structure: no
  // node is a child (Stmt::children()) of two parents. Count child edges, NOT
  // RAV visits -- RecursiveASTVisitor also follows type and template-argument
  // edges, and Clang legitimately shares nodes across those (e.g. the
  // substituted `N` in a `Foo<5>` instantiation is one ConstantExpr reachable
  // from both the type and the body). Counting visits would flag those valid
  // shares; counting child edges flags only genuine multiple-parent reuse,
  // which is what clad must not produce.
  struct Counter : RecursiveASTVisitor<Counter> {
    llvm::DenseMap<const Stmt*, unsigned> ParentEdges;
    llvm::SmallPtrSet<const Stmt*, 32> SeenParents;
    // Visit implicit code: some genuine reuses have a second parent reachable
    // only through implicit nodes.
    bool shouldVisitImplicitCode() const { return true; }
    // But visit only the semantic form of an InitListExpr: with implicit code
    // on, RAV visits both the syntactic and semantic forms, which legitimately
    // share their element nodes (Clang's representation, not a reuse).
    bool TraverseInitListExpr(InitListExpr* ILE,
                              DataRecursionQueue* = nullptr) {
      InitListExpr* Sem = ILE->isSemanticForm() ? ILE : ILE->getSemanticForm();
      if (!Sem)
        Sem = ILE;
      WalkUpFromInitListExpr(Sem);
      for (Stmt* Ch : Sem->children())
        if (Ch)
          TraverseStmt(Ch);
      return true;
    }
    // Skip OpenMP clauses. Lowering `reduction(+:x)` makes Sema synthesize a
    // combiner (`.reduction.lhs = .reduction.lhs + .reduction.rhs`) that
    // references one helper DeclRefExpr from two parents -- Clang's own AST,
    // not clad reuse. TraverseOMPExecutableDirective walks only the clauses;
    // the associated loop body clad generates is still traversed by the
    // DEF_TRAVERSE_STMT children walk.
    bool TraverseOMPExecutableDirective(OMPExecutableDirective* /*D*/) {
      return true;
    }
    // An OpaqueValueExpr is Clang's mechanism for sharing a common
    // subexpression -- e.g. the `threadIdx` base of a `threadIdx.x`
    // __declspec(property) access, which Clang models as a PseudoObjectExpr
    // that references one OpaqueValueExpr from both its syntactic form and its
    // semantic expressions. It is shared by design, not clad reuse, so do not
    // descend into its source (which the OVE also shares); VisitStmt likewise
    // does not count edges to it.
    bool TraverseOpaqueValueExpr(OpaqueValueExpr* /*OVE*/,
                                 DataRecursionQueue* = nullptr) {
      return true;
    }
    // Skip default-argument expressions. Clang stores a call's default argument
    // once on the ParmVarDecl and every call site references that one
    // expression through a CXXDefaultArgExpr (e.g. the `0` in
    // thrust::reduce_by_key's default operators, shared across clad's cloned
    // calls). That is Clang's representation, not clad reuse.
    bool TraverseCXXDefaultArgExpr(CXXDefaultArgExpr* /*E*/,
                                   DataRecursionQueue* = nullptr) {
      return true;
    }
    bool TraverseCXXDefaultInitExpr(CXXDefaultInitExpr* /*E*/,
                                    DataRecursionQueue* = nullptr) {
      return true;
    }
    // Do not descend into type locations. A generated VarDecl's type can embed
    // expressions (a template argument, an array bound, a `std::enable_if<N <
    // _Dt>` SFINAE condition) that Clang shares across same-typed decls -- e.g.
    // the mersenne engine's `_Dt` word-size constant across the pushforward's
    // ValueAndPushforward<> temporaries. Those are Clang's shared AST, not a
    // clad reuse; only the statement tree is clad's hand-built output.
    // clang-22 added a trailing bool (TraverseQualifier) to these; a defaulted
    // parameter matches both the older one-argument call and the new one.
    bool TraverseTypeLoc(TypeLoc /*TL*/, bool = false) { return true; }
    bool TraverseType(QualType /*T*/, bool = false) { return true; }
    bool TraverseTemplateArgumentLoc(const TemplateArgumentLoc& /*ArgLoc*/) {
      return true;
    }
    bool TraverseTemplateArgument(const TemplateArgument& /*Arg*/) {
      return true;
    }
    bool VisitStmt(Stmt* S) {
      // RAV can visit one node twice when it is embedded in two type locs
      // (e.g. a template-argument ConstantExpr shared by several same-typed
      // VarDecls). Count each parent's children once, so a single parent seen
      // twice is not mistaken for two parents.
      if (!SeenParents.insert(S).second)
        return true;
      for (const Stmt* Ch : S->children())
        if (Ch && !isa<OpaqueValueExpr>(Ch))
          ++ParentEdges[Ch];
      return true;
    }
  } C;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  C.TraverseStmt(const_cast<Stmt*>(Root));
  for (const auto& Entry : C.ParentEdges)
    if (Entry.second >= 2)
      return Entry.first;
  return nullptr;
}

} // namespace clad
