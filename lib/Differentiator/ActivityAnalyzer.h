#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Analysis/CFG.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
/// Class that implemets Varied part of the Activity analysis.
/// By performing static data-flow analysis, so called Varied variables
/// are determined, meaning variables that depend on input parameters
/// in a differentiable way. That result enables us to remove redundant
/// statements in the reverse mode, improving generated codes efficiency.
namespace clad {
using VarsData = std::set<const clang::VarDecl*>;
static inline void mergeVarsData(VarsData* targetData, VarsData* mergeData) {
  for (const clang::VarDecl* i : *mergeData)
    targetData->insert(i);
  for (const clang::VarDecl* i : *targetData)
    mergeData->insert(i);
}
class VariedAnalyzer : public clang::RecursiveASTVisitor<VariedAnalyzer> {

  bool m_Varied = false;
  bool m_Marking = false;
  bool m_shouldPushSucc = true;

  std::set<const clang::VarDecl*>& m_VariedDecls;
  // using VarsData = std::set<const clang::VarDecl*>;
  /// A helper method to allocate VarsData
  /// \param toAssign - Parameter to initialize new VarsData with.
  /// \return Unique pointer to a new object of type Varsdata.
  static std::unique_ptr<VarsData> createNewVarsData(VarsData toAssign) {
    return std::unique_ptr<VarsData>(new VarsData(std::move(toAssign)));
  }
  VarsData m_LoopMem;

  clang::CFGBlock* getCFGBlockByID(unsigned ID);

  clang::ASTContext& m_Context;
  std::unique_ptr<clang::CFG> m_CFG;
  std::vector<std::unique_ptr<VarsData>> m_BlockData;
  unsigned m_CurBlockID{};
  std::set<unsigned> m_CFGQueue;
  bool isVaried(const clang::VarDecl* VD) const;
  void copyVarToCurBlock(const clang::VarDecl* VD);
  VarsData& getCurBlockVarsData() { return *m_BlockData[m_CurBlockID]; }
  [[nodiscard]] const VarsData& getCurBlockVarsData() const {
    return const_cast<VariedAnalyzer*>(this)->getCurBlockVarsData();
  }
  void AnalyzeCFGBlock(const clang::CFGBlock& block);

public:
  /// Constructor
  VariedAnalyzer(clang::ASTContext& Context,
                 std::set<const clang::VarDecl*>& Decls)
      : m_VariedDecls(Decls), m_Context(Context) {}

  /// Destructor
  ~VariedAnalyzer() = default;

  /// Delete copy/move operators and constructors.
  VariedAnalyzer(const VariedAnalyzer&) = delete;
  VariedAnalyzer& operator=(const VariedAnalyzer&) = delete;
  VariedAnalyzer(const VariedAnalyzer&&) = delete;
  VariedAnalyzer& operator=(const VariedAnalyzer&&) = delete;

  /// Runs Varied analysis.
  /// \param FD Function to run the analysis on.
  void Analyze(const clang::FunctionDecl* FD);
  bool VisitBinaryOperator(clang::BinaryOperator* BinOp);
  bool VisitCallExpr(clang::CallExpr* CE);
  bool VisitConditionalOperator(clang::ConditionalOperator* CO);
  bool VisitDeclRefExpr(clang::DeclRefExpr* DRE);
  bool VisitDeclStmt(clang::DeclStmt* DS);
  bool VisitUnaryOperator(clang::UnaryOperator* UnOp);
};
} // namespace clad
#endif // CLAD_DIFFERENTIATOR_ACTIVITYANALYZER_H
