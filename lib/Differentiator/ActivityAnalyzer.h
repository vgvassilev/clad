#ifndef CLAD_DIFFERENTIATOR_ACTIVITYANALYZER_H
#define CLAD_DIFFERENTIATOR_ACTIVITYANALYZER_H

#include "AnalysisBase.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"
#include "clad/Differentiator/DiffPlanner.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <set>
#include <utility>

namespace clad {

/// Class that implemets Varied part of the Activity analysis.
/// By performing static data-flow analysis, so called Varied variables
/// are determined, meaning variables that depend on input parameters
/// in a differentiable way. That result enables us to remove redundant
/// statements in the reverse mode, improving generated codes efficiency.
class VariedAnalyzer : public clang::RecursiveASTVisitor<VariedAnalyzer>,
                       public AnalysisBase {
  bool m_Varied = false;
  bool m_Marking = false;

  DiffRequest& m_DiffReq;
  std::set<const clang::Stmt*>& m_ResSet;

  struct LoopInfo {
    bool m_isModified = false;
    int m_LoopID = -1;
  };
  std::map<int, LoopInfo> m_LoopQueueMod;

  /// Checks if a variable is on the current branch.
  /// \param[in] VD - Variable declaration.
  /// @return Whether a variable is on the current branch.
  // bool isVaried(const clang::VarDecl* VD);
  /// Adds varied variable to current branch.
  /// \param[in] VD - Variable declaration.
  // void copyVarToCurBlock(const clang::VarDecl* VD);
  void markExpr(const clang::Stmt* S) { m_ResSet.insert(S); }

  bool areBlocksFixed(int order) {
    if (m_LoopQueueMod.empty())
      return false;
    for (auto& i : m_LoopQueueMod)
      if (i.second.m_LoopID == order && !i.second.m_isModified)
        return false;
    return true;
  }
  int getOrder() {
    int greatest = -1;
    if (m_LoopQueueMod.empty())
      return 0;
    for (auto& i : m_LoopQueueMod)
      if (i.second.m_LoopID > greatest)
        greatest = i.second.m_LoopID;
    return greatest;
  }

  void setVaried(const clang::Expr* E, bool isVaried = true);
  void AnalyzeCFGBlock(const clang::CFGBlock& block);

public:
  /// Constructor
  VariedAnalyzer(clang::AnalysisDeclContext* AnalysisDC, DiffRequest& request,
                 std::set<const clang::Stmt*>& resset)
      : AnalysisBase(AnalysisDC), m_DiffReq(request), m_ResSet(resset) {}

  /// Destructor
  ~VariedAnalyzer() = default;

  /// Delete copy/move operators and constructors.
  VariedAnalyzer(const VariedAnalyzer&) = delete;
  VariedAnalyzer& operator=(const VariedAnalyzer&) = delete;
  VariedAnalyzer(const VariedAnalyzer&&) = delete;
  VariedAnalyzer& operator=(const VariedAnalyzer&&) = delete;

  /// Runs Varied analysis.
  /// \param[in] FD Function to run the analysis on.
  //, std::set<const clang::ParmVarDecl*>& vPVD
  void Analyze();
  bool TraverseBinaryOperator(clang::BinaryOperator* BinOp);
  bool TraverseCallExpr(clang::CallExpr* CE);
  bool TraverseConditionalOperator(clang::ConditionalOperator* CO);
  bool TraverseDeclRefExpr(clang::DeclRefExpr* DRE);
  bool TraverseDeclStmt(clang::DeclStmt* DS);
  bool TraverseUnaryOperator(clang::UnaryOperator* UnOp);
};
} // namespace clad
#endif // CLAD_DIFFERENTIATOR_ACTIVITYANALYZER_H
