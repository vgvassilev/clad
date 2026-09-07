#ifndef CLAD_DIFFERENTIATOR_USEFULANALYZER_H
#define CLAD_DIFFERENTIATOR_USEFULANALYZER_H

#include "AnalysisBase.h"

#include "clang/AST/RecursiveASTVisitor.h"

#include <set>

namespace clad {

/// Backward usefulness analysis. An entry marks a whole variable useful;
/// every entry is a scalar leaf, independent of the declaration's type.
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
};
} // namespace clad
#endif // CLAD_DIFFERENTIATOR_USEFULANALYZER_H
