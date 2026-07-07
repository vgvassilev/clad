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

} // namespace clad

#endif // CLAD_AST_INTEGRITY_H
