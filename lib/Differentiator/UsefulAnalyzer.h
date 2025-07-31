#ifndef CLAD_DIFFERENTIATOR_USEFULANALYZER_H
#define CLAD_DIFFERENTIATOR_USEFULANALYZER_H
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Analysis/CFG.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"

#include <algorithm>
#include <memory>
#include <stack>

namespace clad {

class UsefulAnalyzer : public clang::RecursiveASTVisitor<UsefulAnalyzer> {

  bool m_Useful = false;
  bool m_Marking = false;

  std::set<const clang::VarDecl*>& m_UsefulDecls;
  // std::set<const clang::VarDecl*>& m_VariedDecls;
  using VarsData = std::set<const clang::VarDecl*>;
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
  bool isUseful(const clang::VarDecl* VD) const;
  void copyVarToCurBlock(const clang::VarDecl* VD);
  VarsData& getCurBlockVarsData() { return *m_BlockData[m_CurBlockID]; }
  [[nodiscard]] const VarsData& getCurBlockVarsData() const {
    return const_cast<UsefulAnalyzer*>(this)->getCurBlockVarsData();
  }
  void AnalyzeCFGBlock(const clang::CFGBlock& block);

public:
  /// Constructor
  UsefulAnalyzer(clang::ASTContext& Context,
                 std::set<const clang::VarDecl*>& Decls)
      : m_UsefulDecls(Decls), m_Context(Context) {}

  /// Destructor
  ~UsefulAnalyzer() = default;

  /// Delete copy/move operators and constructors.
  UsefulAnalyzer(const UsefulAnalyzer&) = delete;
  UsefulAnalyzer& operator=(const UsefulAnalyzer&) = delete;
  UsefulAnalyzer(const UsefulAnalyzer&&) = delete;
  UsefulAnalyzer& operator=(const UsefulAnalyzer&&) = delete;

  /// Runs Varied analysis.
  /// \param FD Function to run the analysis on.
  void Analyze(const clang::FunctionDecl* FD);
  bool VisitReturnStmt(clang::ReturnStmt* RS);
  bool VisitDeclRefExpr(clang::DeclRefExpr* DRE);
  bool VisitBinaryOperator(clang::BinaryOperator* BinOp);
  bool VisitDeclStmt(clang::DeclStmt* DS);
  bool VisitCallExpr(clang::CallExpr* CE);
};
} // namespace clad
#endif // CLAD_DIFFERENTIATOR_USEFULANALYZER_H