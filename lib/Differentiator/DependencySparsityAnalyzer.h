#ifndef CLAD_DIFFERENTIATOR_DEPENDENCYSPARSITYANALYZER_H
#define CLAD_DIFFERENTIATOR_DEPENDENCYSPARSITYANALYZER_H

#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Analysis/CFG.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"

#include <memory>
#include <set>
#include <unordered_map>
#include <utility>

namespace clad {

class DependencySparsityAnalyzer
    : public clang::RecursiveASTVisitor<DependencySparsityAnalyzer> {

  unsigned m_CurBlockID{};
  unsigned m_CurrOutputInd{};
  bool m_MarkingMode = true;
  std::unordered_map<clang::VarDecl*, int> m_ParameterNum;
  std::unique_ptr<clang::CFG> m_CFG;

  clang::ASTContext& m_Context;
  std::set<unsigned> m_CFGQueue;
  clang::CFGBlock* getCFGBlockByID(unsigned ID);

  std::set<std::pair<int, int>, utils::compare>& m_OutputDependencySet;

  // FIXME: Once VarData is reworked for AA and TBR, implement by-block
  // analysis.
  // std::map<unsigned, std::set<std::pair<int, int>>> m_BlockData;

  void AnalyzeCFGBlock(const clang::CFGBlock& block);

public:
  DependencySparsityAnalyzer(
      clang::ASTContext& Context,
      std::set<std::pair<int, int>, utils::compare>& OutputDependencySet)
      : m_Context(Context), m_OutputDependencySet(OutputDependencySet) {}
  ~DependencySparsityAnalyzer() = default;

  DependencySparsityAnalyzer(const DependencySparsityAnalyzer&) = delete;
  DependencySparsityAnalyzer&
  operator=(const DependencySparsityAnalyzer&) = delete;
  DependencySparsityAnalyzer(const DependencySparsityAnalyzer&&) = delete;
  DependencySparsityAnalyzer&
  operator=(const DependencySparsityAnalyzer&&) = delete;

  void Analyze(const clang::FunctionDecl* FD);
  bool VisitBinaryOperator(clang::BinaryOperator* BinOp);
  bool VisitDeclRefExpr(clang::DeclRefExpr* DRE);
  // bool VisitDeclStmt(clang::DeclStmt* DS);
  bool VisitArraySubscriptExpr(clang::ArraySubscriptExpr* ASE);
};

} // namespace clad

#endif // CLAD_DIFFERENTIATOR_DEPENDENCYSPARSITYANALYZER_H