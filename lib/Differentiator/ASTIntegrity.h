//--------------------------------------------------------------------*- C++ -//
// clad - the C++ Clang-based Automatic Differentiator
//
// AST-integrity checks for hand-synthesized derivative code.
//
// clad assembles a derivative's statements by hand and must keep them a proper
// tree: no node the child (Stmt::children()) of two parents. A node with two
// parents ("shared") lets a later in-place edit leak across contexts and
// silently corrupt codegen (and a shared aggregate initializer breaks CodeGen
// outright). This is a property of the hand-built body, not of Clang ASTs in
// general -- Clang shares nodes across type and template-argument edges and
// InitListExpr's syntactic/semantic forms -- so the check counts child edges,
// not visits (see findSharedNode).
//----------------------------------------------------------------------------//

#ifndef CLAD_AST_INTEGRITY_H
#define CLAD_AST_INTEGRITY_H

namespace clang {
class Stmt;
class ValueDecl;
class FunctionDecl;
} // namespace clang

namespace clad {

/// Return the first Stmt that occurs more than once under \p Root (i.e. is
/// shared with itself through two parent edges), or nullptr if \p Root is a
/// proper tree.
const clang::Stmt* findSharedNode(const clang::Stmt* Root);

/// Return the first node in \p Derivative that is also reachable from \p Primal
/// through Stmt::children() edges, or nullptr if the derivative splices no node
/// from its primal. The original function's AST outlives differentiation, so a
/// generated derivative sharing one of its nodes would corrupt the user's own
/// code if a later pass edits it. Child edges only: Clang legitimately shares
/// type/template-argument constants and default-argument expressions, which
/// Stmt::children() does not traverse.
const clang::Stmt* findPrimalSharedNode(const clang::Stmt* Derivative,
                                        const clang::Stmt* Primal);

/// Return the first local variable or parameter of \p Original that a
/// DeclRefExpr in \p Derivative still references, or nullptr if every reference
/// was remapped to a derivative-owned decl. A generated derivative must not
/// reference the original function's own params/locals -- they do not exist in
/// it; such a reference is a forgotten reference-remap (a primal clone that was
/// never registered in m_DeclReplacements) and would miscompile. Only VarDecls
/// are gated: function self-references (recursion) and decls declared outside
/// \p Original have a context \p Original does not enclose.
const clang::ValueDecl* findOriginalRef(const clang::Stmt* Derivative,
                                        const clang::FunctionDecl* Original);

/// The structural violations a generated derivative body may exhibit, each the
/// first offending node/decl or null. Purely a function of the AST -- the
/// caller supplies the differentiation context (whether the derivation was
/// clean) and decides how to react (assert in debug, diagnose in release).
struct IntegrityReport {
  /// A node that is the child of two parents within the derivative.
  const clang::Stmt* SharedNode = nullptr;
  /// A node spliced from the original function's still-live AST.
  const clang::Stmt* PrimalNode = nullptr;
  /// A reference left bound to one of the original function's own
  /// params/locals.
  const clang::ValueDecl* StrayRef = nullptr;
};

/// Run every structural integrity check on a generated \p Derivative body.
/// \p Original is the function being differentiated (its body is the primal),
/// or null when there is none (e.g. a synthesized overload). This only reads
/// the AST; StrayRef in particular is meaningful only for a clean derivation,
/// which the caller must establish before acting on it.
IntegrityReport verifyDerivative(const clang::Stmt* Derivative,
                                 const clang::FunctionDecl* Original);

} // namespace clad

#endif // CLAD_AST_INTEGRITY_H
