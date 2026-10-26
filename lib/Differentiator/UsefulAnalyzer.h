#ifndef CLAD_DIFFERENTIATOR_USEFULANALYZER_H
#define CLAD_DIFFERENTIATOR_USEFULANALYZER_H

#include "AnalysisBase.h"

#include "clang/AST/RecursiveASTVisitor.h"

#include <set>

namespace clad {

/// Backward usefulness analysis. An entry marks a whole variable useful;
/// every entry is a scalar leaf, independent of the declaration's type.
/// \ingroup analyses
class UsefulAnalyzer : public clang::RecursiveASTVisitor<UsefulAnalyzer>,
                       public AnalysisBase {
  bool m_Useful = false;
  bool m_Marking = false;
  std::set<const clang::VarDecl*>& m_UsefulDecls;

  bool isUseful(const clang::VarDecl* VD);
  void markUseful(const clang::VarDecl* VD);
  void AnalyzeCFGBlock(const clang::CFGBlock& block);

public:
  UsefulAnalyzer(clang::AnalysisDeclContext* AnalysisDC,
                 std::set<const clang::VarDecl*>& Decls)
      : AnalysisBase(AnalysisDC), m_UsefulDecls(Decls) {}

  ~UsefulAnalyzer() = default;

  UsefulAnalyzer(const UsefulAnalyzer&) = delete;
  UsefulAnalyzer& operator=(const UsefulAnalyzer&) = delete;
  UsefulAnalyzer(const UsefulAnalyzer&&) = delete;
  UsefulAnalyzer& operator=(const UsefulAnalyzer&&) = delete;

  /// Runs useful analysis on FD.
  void Analyze(const clang::FunctionDecl* FD);
  bool VisitReturnStmt(clang::ReturnStmt* RS);
  bool VisitDeclRefExpr(clang::DeclRefExpr* DRE);
  bool VisitBinaryOperator(clang::BinaryOperator* BinOp);
  bool VisitDeclStmt(clang::DeclStmt* DS);
  bool VisitCallExpr(clang::CallExpr* CE);
  bool TraverseCallExpr(clang::CallExpr* CE);
  bool TraverseCXXMemberCallExpr(clang::CXXMemberCallExpr* CE);
  bool TraverseCXXOperatorCallExpr(clang::CXXOperatorCallExpr* CE);

private:
  /// Whether \p E names a useful variable anywhere in it, as the left side of
  /// an assignment is read.
  bool namesUseful(const clang::Expr* E);
  /// Whether \p CE writes a useful variable through an argument or its
  /// object. It reads all of its arguments to do so.
  bool writesUseful(const clang::CallExpr* CE);
  /// Traverses \p CE, marking its arguments when it writes a useful variable.
  template <class F> bool traverseCall(const clang::CallExpr* CE, F Base);
};
} // namespace clad
#endif // CLAD_DIFFERENTIATOR_USEFULANALYZER_H
