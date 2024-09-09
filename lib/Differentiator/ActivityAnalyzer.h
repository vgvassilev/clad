#ifndef CLAD_DIFFERENTIATOR_ACTIVITYANALYZER_H
#define CLAD_DIFFERENTIATOR_ACTIVITYANALYZER_H

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Analysis/CFG.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"

#include <algorithm>
#include <utility>
#include <iterator>
#include <set>
#include <unordered_map>

using namespace clang;

namespace clad {
class VariedAnalyzer : public clang::RecursiveASTVisitor<VariedAnalyzer> {

  bool m_Varied = false;
  bool m_Marking = false;

  std::set<const clang::VarDecl*>& m_VariedDecls;
  using VarsData = std::set<const clang::VarDecl*>;
  static std::unique_ptr<VarsData> createNewVarsData(VarsData toAssign) {
    return std::unique_ptr<VarsData>(new VarsData(std::move(toAssign)));
  }

  VarsData m_LoopMem;
  clang::CFGBlock* getCFGBlockByID(unsigned ID);

  static void merge(VarsData* targetData, VarsData* mergeData);
  ASTContext& m_Context;
  std::unique_ptr<clang::CFG> m_CFG;
  std::vector<std::unique_ptr<VarsData>> m_BlockData;
  std::vector<short> m_BlockPassCounter;
  unsigned m_CurBlockID{};
  std::set<unsigned> m_CFGQueue;

  void addToVaried(const clang::VarDecl* VD);
  bool isVaried(const clang::VarDecl* VD);

  void copyVarToCurBlock(const clang::VarDecl* VD);
  VarsData& getCurBlockVarsData() { return *m_BlockData[m_CurBlockID]; }

public:
  /// Constructor
  VariedAnalyzer(ASTContext& Context, std::set<const clang::VarDecl*>& Decls)
      : m_VariedDecls(Decls), m_Context(Context) {}

  /// Destructor
  ~VariedAnalyzer() = default;

  /// Delete copy/move operators and constructors.
  VariedAnalyzer(const VariedAnalyzer&) = delete;
  VariedAnalyzer& operator=(const VariedAnalyzer&) = delete;
  VariedAnalyzer(const VariedAnalyzer&&) = delete;
  VariedAnalyzer& operator=(const VariedAnalyzer&&) = delete;

  /// Visitors
  void Analyze(const clang::FunctionDecl* FD);
  void VisitCFGBlock(const clang::CFGBlock& block);
  bool VisitBinaryOperator(clang::BinaryOperator* BinOp);
  bool VisitCallExpr(clang::CallExpr* CE);
  bool VisitConditionalOperator(clang::ConditionalOperator* CO);
  bool VisitDeclRefExpr(clang::DeclRefExpr* DRE);
  bool VisitDeclStmt(clang::DeclStmt* DS);
  bool VisitUnaryOperator(clang::UnaryOperator* UnOp);
};
} // namespace clad
#endif // CLAD_DIFFERENTIATOR_ACTIVITYANALYZER_H