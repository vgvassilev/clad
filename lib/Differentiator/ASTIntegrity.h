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

} // namespace clad

#endif // CLAD_AST_INTEGRITY_H
